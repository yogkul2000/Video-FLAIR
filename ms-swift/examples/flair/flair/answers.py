import json
import math
import re

LETTER = re.compile(r'^\(?([A-Z])\)?(?:[.:)\s]|$)')
NUM = re.compile(r'-?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][-+]?\d+)?')


def classify(gold, question=''):
    g = str(gold).strip()
    if re.fullmatch(r'\(?[A-Z]\)?', g):
        return 'mcq_letter'
    if g.lower().rstrip('.') in ('yes', 'no'):
        return 'yesno'
    if re.fullmatch(r'-?\d+(?:\.\d+)?', g):
        return 'number'
    if g.startswith('{'):
        try:
            obj = json.loads(g)
        except Exception:
            return 'open'
        if not isinstance(obj, dict):
            return 'open'
        if 'positive_points' in obj or 'negative_points' in obj:
            return 'segment'
        if 'boxes' in obj and isinstance(obj['boxes'], dict):
            return 'stvg' if 'time' in obj else 'track'
        if 'boxes' in obj:
            return 'bbox'
        if 'time' in obj:
            t = obj['time']
            return 'temporal_span' if isinstance(t, list) else 'temporal_single'
        return 'open'
    if '\\' in g or '$' in g:
        return 'math'
    if re.fullmatch(r'[-\d\s./,°%a-zA-Z²³π√]+', g) and NUM.search(g) and len(g) <= 24:
        return 'math'
    return 'open'


def _lcs(a, b):
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def rouge_l(pred, gold):
    p = re.findall(r'\w+', pred.lower())
    g = re.findall(r'\w+', gold.lower())
    lcs = _lcs(p, g)
    if lcs == 0:
        return 0.0
    prec, rec = lcs / len(p), lcs / len(g)
    return 2 * prec * rec / (prec + rec)


def _json(text):
    text = text.strip()
    text = re.sub(r'^```(?:json)?|```$', '', text).strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r'\{.*\}', text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None


def _box(v):
    if isinstance(v, (list, tuple)) and len(v) == 4:
        try:
            return [float(x) for x in v]
        except Exception:
            return None
    if isinstance(v, (list, tuple)) and len(v) == 1:
        return _box(v[0])
    return None


def iou(a, b):
    a, b = _box(a), _box(b)
    if a is None or b is None:
        return 0.0
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1]) + max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def tiou(a, b):
    try:
        a0, a1 = float(a[0]), float(a[1])
        b0, b1 = float(b[0]), float(b[1])
    except Exception:
        return 0.0
    if a1 < a0:
        a0, a1 = a1, a0
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else float(abs(a0 - b0) < 1e-6 and abs(a1 - b1) < 1e-6)


def temporal_single(p, g, tau=5.0):
    try:
        return max(0.0, 1.0 - abs(float(p) - float(g)) / tau)
    except Exception:
        return 0.0


def _span(v):
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return v
    return None


def _dict_boxes_score(pb, gb):
    if not isinstance(gb, dict) or not gb:
        return 0.0
    if not isinstance(pb, dict):
        return 0.0
    return sum(iou(pb.get(k), v) for k, v in gb.items()) / len(gb)


def _points_score(pp, gp, gold_box):
    if not isinstance(gp, list) or not gp:
        return None
    if not isinstance(pp, list) or not pp:
        return 0.0
    pts = []
    for p in pp:
        try:
            pts.append((float(p[0]), float(p[1])))
        except Exception:
            continue
    if not pts:
        return 0.0
    tot = 0.0
    for g in gp:
        try:
            gx, gy = float(g[0]), float(g[1])
        except Exception:
            continue
        d = min(math.hypot(px - gx, py - gy) for px, py in pts)
        tot += max(0.0, 1.0 - d / 150.0)
    return tot / len(gp)


def score_json(pred, gold, kind):
    p = _json(pred)
    g = _json(gold) if isinstance(gold, str) else gold
    if not isinstance(p, dict) or not isinstance(g, dict):
        if kind == 'bbox' and isinstance(g, dict):
            m = re.findall(r'-?\d+(?:\.\d+)?', pred)
            if len(m) == 4:
                return iou([float(x) for x in m], g.get('boxes'))
        return 0.0
    if kind == 'bbox':
        return iou(p.get('boxes'), g.get('boxes'))
    if kind == 'track':
        return _dict_boxes_score(p.get('boxes'), g.get('boxes'))
    if kind == 'stvg':
        return 0.5 * tiou(_span(p.get('time')) or (0, 0), g['time']) + 0.5 * _dict_boxes_score(p.get('boxes'), g.get('boxes'))
    if kind == 'segment':
        parts = [iou(p.get('boxes'), g.get('boxes'))]
        ps = _points_score(p.get('positive_points'), g.get('positive_points'), g.get('boxes'))
        if ps is not None:
            parts.append(ps)
        if 'time' in g:
            parts.append(temporal_single(p.get('time'), g['time'], tau=2.0))
        return sum(parts) / len(parts)
    if kind == 'temporal_span':
        s = _span(p.get('time'))
        return tiou(s, g['time']) if s else 0.0
    if kind == 'temporal_single':
        return temporal_single(p.get('time'), g['time'])
    return 0.0


def _latex_norm(s):
    s = s.strip().strip('$').strip()
    s = re.sub(r'\\(left|right|displaystyle|,|;|!|quad|qquad)', '', s)
    s = s.replace('\\dfrac', '\\frac').replace('\\tfrac', '\\frac')
    s = re.sub(r'\\text\{([^}]*)\}', r'\1', s)
    s = re.sub(r'\\mathrm\{([^}]*)\}', r'\1', s)
    s = s.replace('^\\circ', '°').replace('^{\\circ}', '°').replace('\\circ', '°')
    s = re.sub(r'\s+', '', s)
    s = s.rstrip('.')
    return s


def _latex_value(s):
    s = _latex_norm(s)
    s = re.sub(r'(cm|mm|km|m|kg|g|s|°|%|degrees?|units?)(²|³|\^2|\^3)?$', '', s)
    for _ in range(4):
        s = re.sub(r'\\frac\{([^{}]*)\}\{([^{}]*)\}', r'((\1)/(\2))', s)
        s = re.sub(r'\\sqrt\{([^{}]*)\}', r'sqrt(\1)', s)
    s = re.sub(r'\\sqrt(\d+)', r'sqrt(\1)', s)
    s = s.replace('\\pi', 'pi').replace('π', 'pi').replace('√', 'sqrt').replace('\\times', '*').replace('\\cdot', '*')
    s = s.replace('^', '**').replace('{', '(').replace('}', ')')
    s = re.sub(r'(\d|\))(pi|sqrt|\()', r'\1*\2', s)
    if not re.fullmatch(r'[0-9+\-*/().a-z]*', s) or not s:
        return None
    if re.search(r'[a-z]', re.sub(r'pi|sqrt', '', s)):
        return None
    try:
        return float(eval(s, {'__builtins__': {}}, {'pi': math.pi, 'sqrt': math.sqrt}))
    except Exception:
        return None


def score_math(pred, gold):
    a, b = _latex_norm(pred), _latex_norm(gold)
    if a == b:
        return 1.0
    va, vb = _latex_value(pred), _latex_value(gold)
    if va is not None and vb is not None:
        return 1.0 if abs(va - vb) <= 1e-4 * max(1.0, abs(vb)) else 0.0
    if ',' in b:
        sa = sorted(x for x in re.split(r'[,;]', a) if x)
        sb = sorted(x for x in re.split(r'[,;]', b) if x)
        return 1.0 if sa == sb else 0.0
    return 0.0


def extract_letter(pred):
    pred = pred.strip()
    m = LETTER.match(pred)
    if m:
        return m.group(1)
    m = re.search(r'\b([A-Z])\b', pred)
    return m.group(1) if m else None


def extract_number(pred):
    m = NUM.findall(pred.replace(' ', ''))
    if not m:
        return None
    try:
        return float(m[-1].replace(',', ''))
    except Exception:
        return None


def score(pred, gold, kind=None):
    pred = (pred or '').strip()
    gold = str(gold).strip()
    kind = kind or classify(gold)
    if not pred:
        return 0.0
    if kind == 'mcq_letter':
        return 1.0 if extract_letter(pred) == gold.strip('()') else 0.0
    if kind == 'yesno':
        p = re.sub(r'[^a-z]', '', pred.lower().split()[0]) if pred.split() else ''
        return 1.0 if p == gold.lower().rstrip('.') else 0.0
    if kind == 'number':
        v = extract_number(pred)
        if v is None:
            return 0.0
        return 1.0 / (1.0 + (v - float(gold)) ** 2)
    if kind in ('bbox', 'track', 'stvg', 'segment', 'temporal_span', 'temporal_single'):
        return score_json(pred, gold, kind)
    if kind == 'math':
        return score_math(pred, gold)
    return rouge_l(pred, gold)
