import base64
import json
import os
import re
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from .parsing import heuristic_span_scores, spans
from .prompts import MODES, VERIFIER


def _env(name, default, cast=str):
    v = os.environ.get(name)
    return cast(v) if v not in (None, '') else default


def pick_frames(images, videos, num_frames):
    paths = []
    if videos:
        frames = videos[0] if isinstance(videos[0], (list, tuple)) else videos
        n = len(frames)
        k = min(num_frames, n)
        idx = [round(i * (n - 1) / max(1, k - 1)) for i in range(k)] if k > 1 else [0]
        paths = [frames[i] for i in idx]
    elif images:
        paths = list(images)[:num_frames]
    out = []
    for p in paths:
        if isinstance(p, dict):
            p = p.get('path')
        if isinstance(p, str) and os.path.exists(p):
            out.append(p)
    return out


def verifier_text(question, completion, gold, span_texts):
    return (f'Question:\n{question}\n\nCandidate completion:\n{completion}\n\nGround-truth answer:\n{gold}\n\n'
            f'need_span_credit=1\nspan_texts (return one score per span, same order):\n'
            f'{json.dumps(span_texts, ensure_ascii=False)}')


class VerifierClient:

    def __init__(self):
        self.url = _env('FLAIR_VERIFIER_URL', '')
        self.model = _env('FLAIR_VERIFIER_MODEL', 'verifier')
        self.model_file = _env('FLAIR_VERIFIER_MODEL_FILE', '')
        self.num_frames = _env('FLAIR_VERIFIER_FRAMES', 4, int)
        self.timeout = _env('FLAIR_VERIFIER_TIMEOUT', 120.0, float)
        self.workers = _env('FLAIR_VERIFIER_WORKERS', 32, int)
        self.max_tokens = _env('FLAIR_VERIFIER_MAX_TOKENS', 1024, int)
        self.retries = _env('FLAIR_VERIFIER_RETRIES', 2, int)
        self.pool = ThreadPoolExecutor(self.workers) if self.url else None
        self.lock = threading.Lock()
        self._model_mtime = None
        self.poll = _env('FLAIR_VERIFIER_POLL', 60.0, float)
        self._polled = 0.0

    @property
    def enabled(self):
        return bool(self.url)

    def _discover(self):
        now = time.time()
        if now - self._polled < self.poll:
            return self.model
        self._polled = now
        try:
            with urllib.request.urlopen(self.url.rstrip('/') + '/models', timeout=10) as r:
                names = [m['id'] for m in json.loads(r.read()).get('data', [])]
        except Exception:
            return self.model
        best = None
        for n in names:
            m = re.fullmatch(r'verifier_v(\d+)', n)
            if m and (best is None or int(m.group(1)) > best[0]):
                best = (int(m.group(1)), n)
        with self.lock:
            self.model = best[1] if best else ('verifier' if 'verifier' in names else self.model)
        return self.model

    def current_model(self):
        if not self.model_file:
            return self._discover() if self.url else self.model
        if os.path.exists(self.model_file):
            mtime = os.path.getmtime(self.model_file)
            if mtime != self._model_mtime:
                with self.lock:
                    try:
                        with open(self.model_file) as f:
                            self.model = json.load(f)['model']
                        self._model_mtime = mtime
                    except Exception:
                        pass
        return self.model

    def _media(self, images, videos):
        out = []
        for p in pick_frames(images, videos, self.num_frames):
            with open(p, 'rb') as f:
                b = base64.b64encode(f.read()).decode()
            out.append({'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{b}'}})
        return out

    def build_messages(self, question, completion, gold, span_texts, images=None, videos=None):
        text = verifier_text(question, completion, gold, span_texts)
        content = self._media(images, videos) + [{'type': 'text', 'text': text}]
        return [{'role': 'system', 'content': VERIFIER}, {'role': 'user', 'content': content}]

    def _post(self, messages):
        body = json.dumps({
            'model': self.current_model(),
            'messages': messages,
            'temperature': 0.0,
            'max_tokens': self.max_tokens,
        }).encode()
        req = urllib.request.Request(
            self.url.rstrip('/') + '/chat/completions', data=body, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())['choices'][0]['message']['content']

    def _call(self, item):
        messages, n_spans = item
        for attempt in range(self.retries + 1):
            try:
                return parse_verdict(self._post(messages), n_spans)
            except Exception:
                if attempt < self.retries:
                    time.sleep(1 + attempt)
        return None

    def score(self, items):
        if not self.enabled or not items:
            return [None] * len(items)
        return list(self.pool.map(self._call, items))


def _f(x, lo=0.0, hi=1.0):
    try:
        return max(lo, min(hi, float(x)))
    except Exception:
        return None


NUM = r'(-?[0-9]*\.?[0-9]+)'


def _salvage(text):
    obj = {}
    for k in ('temporal_grounding', 'spatial_grounding', 'human_alignment', 'reasoning_text_ok'):
        m = re.search(r'"%s"\s*:\s*%s' % (k, NUM), text)
        if m:
            obj[k] = m.group(1)
    m = re.search(r'"preferred_mode"\s*:\s*"([A-Za-z]+)"', text)
    if m:
        obj['preferred_mode'] = m.group(1)
    m = re.search(r'"span_scores"\s*:\s*\[([^\]]*)', text, re.S)
    if m:
        obj['span_scores'] = re.findall(NUM, m.group(1))
    return obj


def parse_verdict(text, n_spans):
    text = re.sub(r'<think>.*?</think>', '', text or '', flags=re.S)
    m = re.search(r'\{.*\}', text, re.S)
    obj = None
    if m:
        try:
            obj = json.loads(re.sub(r'//[^\n]*', '', m.group(0)))
        except Exception:
            obj = None
    if not isinstance(obj, dict):
        obj = _salvage(text)
    out = {
        'temporal_grounding': _f(obj.get('temporal_grounding')),
        'spatial_grounding': _f(obj.get('spatial_grounding')),
        'human_alignment': _f(obj.get('human_alignment')),
    }
    if any(v is None for v in out.values()):
        return None
    pm = str(obj.get('preferred_mode', '')).upper().strip()
    out['preferred_mode'] = pm if pm in MODES else None
    out['reasoning_text_ok'] = 1 if str(obj.get('reasoning_text_ok', 1)).strip() in ('1', '1.0', 'true', 'True') else 0
    ss = obj.get('span_scores')
    if isinstance(ss, list) and n_spans > 0 and len(ss) >= n_spans:
        vals = [_f(v, -1.0, 1.0) for v in ss[:n_spans]]
        out['span_scores'] = vals if all(v is not None for v in vals) else None
    else:
        out['span_scores'] = None
    return out


def heuristic_verdict(think):
    n_t = len(re.findall(r'<t>', think or ''))
    n_s = len(re.findall(r'<obj>[^<]*</obj>\s*<box>', think or ''))
    return {
        'temporal_grounding': min(1.0, n_t / 3.0),
        'spatial_grounding': min(1.0, n_s / 3.0),
        'human_alignment': 0.5 if (think or '').strip() else 0.0,
        'preferred_mode': None,
        'reasoning_text_ok': 1 if (think or '').strip() else 0,
        'span_scores': heuristic_span_scores(think),
    }


def span_texts(think):
    return [s for _, _, s in spans(think)]
