import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flair.answers import score
from flair.parsing import parse
from flair.prompts import MODE_PROMPTS


def expand(args):
    modes = args.probe_modes.split(',')
    n = 0
    with open(args.input) as f, open(args.output, 'w') as w:
        for line in f:
            row = json.loads(line)
            for k in range(args.samples):
                mode = modes[k % len(modes)]
                rec = dict(row)
                rec['messages'] = [{'role': 'system', 'content': MODE_PROMPTS[mode]}] + row['messages']
                rec['probe_mode'] = mode
                rec['probe_k'] = k
                w.write(json.dumps(rec, ensure_ascii=False) + '\n')
                n += 1
    print(json.dumps({'expanded': n}))


def _norm(x):
    if isinstance(x, dict):
        return _norm(x['path']) if 'path' in x else {k: _norm(v) for k, v in sorted(x.items())}
    if isinstance(x, (list, tuple)):
        return [_norm(v) for v in x]
    return x


def _key(row):
    users = [m.get('content') for m in row.get('messages') or [] if m.get('role') in ('system', 'user')]
    return json.dumps([users, _norm(row.get('images') or []), _norm(row.get('videos') or [])], sort_keys=True)


def select(args):
    meta = {}
    with open(args.probe) as f:
        for line in f:
            row = json.loads(line)
            meta[_key(row)] = (row['sid'], row['solution'], row.get('answer_type'))
    stats = {}
    missing = 0
    with open(args.results) as f:
        for line in f:
            row = json.loads(line)
            m = meta.get(_key(row))
            if m is None:
                missing += 1
                continue
            sid, solution, answer_type = m
            p = parse(row.get('response') or '')
            s = score(p['answer'], solution, answer_type) if p['answer'] else 0.0
            st = stats.setdefault(sid, {'scores': [], 'valid': 0})
            st['scores'].append(1.0 if s >= args.correct else 0.0)
            st['valid'] += int(p['valid'])
    keep = {}
    for sid, st in stats.items():
        mean = sum(st['scores']) / len(st['scores'])
        if args.min_acc < mean < args.max_acc:
            keep[sid] = mean
    rows = []
    with open(args.input) as f:
        for line in f:
            row = json.loads(line)
            if row['sid'] in keep:
                row['sft_acc'] = keep[row['sid']]
                rows.append(row)
    random.Random(args.seed).shuffle(rows)
    if args.limit:
        rows = rows[:args.limit]
    with open(args.output, 'w') as w:
        for r in rows:
            w.write(json.dumps(r, ensure_ascii=False) + '\n')
    hist = {}
    for st in stats.values():
        b = round(sum(st['scores']) / len(st['scores']), 1)
        hist[b] = hist.get(b, 0) + 1
    print(json.dumps({'unmatched': missing, 'scored': len(stats), 'kept': len(rows), 'acc_hist': dict(sorted(hist.items()))}))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    e = sub.add_parser('expand')
    e.add_argument('--input', required=True)
    e.add_argument('--output', required=True)
    e.add_argument('--samples', type=int, default=4)
    e.add_argument('--probe_modes', default='DIRECT,CONCISE,DEEP,ADAPTIVE')
    s = sub.add_parser('select')
    s.add_argument('--input', required=True)
    s.add_argument('--probe', required=True)
    s.add_argument('--results', required=True)
    s.add_argument('--output', required=True)
    s.add_argument('--min_acc', type=float, default=0.25)
    s.add_argument('--correct', type=float, default=0.5)
    s.add_argument('--max_acc', type=float, default=0.75)
    s.add_argument('--limit', type=int, default=0)
    s.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    expand(args) if args.cmd == 'expand' else select(args)


if __name__ == '__main__':
    main()
