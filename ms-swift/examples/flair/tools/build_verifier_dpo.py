import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flair.prompts import VERIFIER
from flair.verifier import pick_frames, span_texts, verifier_text
from flair.parsing import parse

FIELDS = ('temporal_grounding', 'spatial_grounding', 'human_alignment')


def verdict_json(base, pick, other, m_star, spans_n):
    out = {f: round(pick(base.get(f, 0.0), other.get(f, 0.0)), 3) for f in FIELDS}
    out['preferred_mode'] = m_star or base.get('preferred_mode') or 'CONCISE'
    out['reasoning_text_ok'] = int(base.get('reasoning_text_ok', 1))
    ss = base.get('span_scores') or []
    out['span_scores'] = [round(float(x), 3) for x in ss] if len(ss) == spans_n else [0.0] * spans_n
    return json.dumps(out)


def sample(pair, side, frames):
    me, other = pair[side], pair['lo' if side == 'hi' else 'hi']
    st = span_texts(parse(me['completion'])['think'])
    imgs = pick_frames(pair.get('images'), pair.get('videos'), frames)
    text = verifier_text(pair['question'], me['completion'], pair['solution'], st)
    up = verdict_json(me['verdict'], max, other['verdict'], pair.get('m_star'), len(st))
    down = verdict_json(me['verdict'], min, other['verdict'], pair.get('m_star'), len(st))
    chosen, rejected = (up, down) if side == 'hi' else (down, up)
    if chosen == rejected:
        return None
    return {
        'messages': [
            {'role': 'system', 'content': VERIFIER},
            {'role': 'user', 'content': '<image>' * len(imgs) + text},
            {'role': 'assistant', 'content': chosen},
        ],
        'rejected_response': rejected,
        'images': imgs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairs', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--max_pairs', type=int, default=2000)
    ap.add_argument('--margin', type=float, default=0.15)
    ap.add_argument('--frames', type=int, default=4)
    ap.add_argument('--min_step', type=int, default=0)
    args = ap.parse_args()
    pairs = []
    with open(args.pairs) as f:
        for line in f:
            p = json.loads(line)
            if p.get('step', 0) >= args.min_step and p['hi']['composite'] - p['lo']['composite'] > args.margin:
                pairs.append(p)
    pairs = pairs[-args.max_pairs:]
    n = 0
    with open(args.output, 'w') as w:
        for p in pairs:
            for side in ('hi', 'lo'):
                r = sample(p, side, args.frames)
                if r:
                    w.write(json.dumps(r, ensure_ascii=False) + '\n')
                    n += 1
    print(json.dumps({'pairs': len(pairs), 'rows': n}))


if __name__ == '__main__':
    main()
