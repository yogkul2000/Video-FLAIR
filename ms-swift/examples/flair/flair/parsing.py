import re

from .prompts import MODES

SPECIAL = re.compile(r'<\|[^|>]*\|>|</s>|<eos>|<pad>')
SCHEMA = re.compile(r'^\s*<(DIRECT|CONCISE|DEEP)>\s*<think>(.*?)</think>\s*<answer>(.*?)</answer>\s*</\1>\s*$', re.S)
WRAPPER = re.compile(r'<(DIRECT|CONCISE|DEEP|ADAPTIVE)>')
THINK = re.compile(r'<think>(.*?)</think>', re.S)
ANSWER = re.compile(r'<answer>(.*?)</answer>', re.S)
GROUND_TAG = re.compile(r'</?(obj|box|t)>')
TAGGED = re.compile(r'<(obj|box|t)>.*?</\1>', re.S)
SPAN_SPLIT = re.compile(r'(?<=[.!?])\s+|\n+')
HEDGES = ('maybe', 'probably', 'guess', 'unsure', 'not sure')


def clean(text):
    return SPECIAL.sub('', text or '').strip()


def parse(text):
    text = clean(text)
    m = SCHEMA.match(text)
    if m:
        think, answer = m.group(2), m.group(3)
        return {
            'valid': True,
            'mode': m.group(1),
            'think': think.strip(),
            'answer': answer.strip(),
            'think_start': m.start(2),
            'think_end': m.end(2),
            'text': text,
        }
    w = WRAPPER.search(text)
    t = THINK.search(text)
    a = ANSWER.search(text)
    return {
        'valid': False,
        'mode': w.group(1) if w and w.group(1) in MODES else None,
        'think': t.group(1).strip() if t else '',
        'answer': a.group(1).strip() if a else '',
        'think_start': t.start(1) if t else -1,
        'think_end': t.end(1) if t else -1,
        'text': text,
    }


def plain_words(text):
    return len(GROUND_TAG.sub(' ', text or '').split())


def has_ground_tags(text):
    return bool(GROUND_TAG.search(text or ''))


def spans(think):
    out = []
    pos = 0
    for piece in SPAN_SPLIT.split(think or ''):
        if not piece:
            continue
        start = think.find(piece, pos)
        if start < 0:
            start = pos
        end = start + len(piece)
        pos = end
        if piece.strip():
            out.append((start, end, piece.strip()))
    return out


def heuristic_span_scores(think):
    out = []
    for _, _, s in spans(think):
        v = 0.0
        if '<obj>' in s and '<box>' in s:
            v += 0.20
        if '<t>' in s:
            v += 0.15
        if len(TAGGED.sub(' ', s).split()) >= 4:
            v += 0.10
        low = s.lower()
        if any(h in low for h in HEDGES):
            v -= 0.12
        out.append(max(-1.0, min(1.0, v)))
    return out


def span_score_mean(think, scores=None):
    if not (think or '').strip():
        return -0.2
    if scores is None:
        scores = heuristic_span_scores(think)
    if not scores:
        return -0.2
    return max(-1.0, min(1.0, sum(scores) / len(scores)))


def strip_ground_markup(think):
    think = re.sub(r'<t>\s*([^<]*?)\s*</t>', r'\1s', think)
    think = re.sub(r'<obj>\s*([^<]*?)\s*</obj>\s*<box>\s*([^<]*?)\s*</box>', r'\1 \2', think)
    think = re.sub(r'<box>\s*([^<]*?)\s*</box>', r'\1', think)
    think = re.sub(r'<obj>\s*([^<]*?)\s*</obj>', r'\1', think)
    return think
