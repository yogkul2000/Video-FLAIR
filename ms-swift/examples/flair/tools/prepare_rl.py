import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flair.answers import classify
from prepare_sft import media_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--media_root', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--modes', default='DIRECT,CONCISE,DEEP')
    ap.add_argument('--per_mode_limit', type=int, default=0)
    ap.add_argument('--exclude', default='')
    ap.add_argument('--timestamps', type=int, default=1)
    ap.add_argument('--check_media', type=int, default=0)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    modes = set(args.modes.split(','))
    media_root = os.path.abspath(args.media_root)
    skip = set()
    for path in filter(None, args.exclude.split(',')):
        with open(path) as f:
            for line in f:
                skip.add(json.loads(line).get('sid'))
    by_mode = {}
    seen = set()
    with open(args.input) as f:
        for line in f:
            row = json.loads(line)
            meta = row.get('meta', {})
            sid = meta.get('sid')
            if meta.get('mode') not in modes or sid in skip or sid in seen:
                continue
            if meta.get('gold_answer') in (None, ''):
                continue
            seen.add(sid)
            by_mode.setdefault(meta['mode'], []).append(row)
    out, missing, kinds = [], 0, {}
    for mode, rows in sorted(by_mode.items()):
        rng.shuffle(rows)
        if args.per_mode_limit:
            rows = rows[:args.per_mode_limit]
        for row in rows:
            meta = row['meta']
            user = media_text(row, meta) if args.timestamps else row['messages'][0]['content']
            gold = str(meta['gold_answer'])
            kind = classify(gold)
            rec = {'messages': [{'role': 'user', 'content': user}]}
            key = 'videos' if 'videos' in row else 'images'
            if key == 'videos':
                rec[key] = [[os.path.join(media_root, p) for p in v] for v in row[key]]
                flat = [p for v in rec[key] for p in v]
            else:
                rec[key] = [os.path.join(media_root, p) for p in row[key]]
                flat = rec[key]
            if args.check_media and not all(os.path.exists(p) for p in flat):
                missing += 1
                continue
            rec.update({'solution': gold, 'answer_type': kind, 'sid': meta['sid'], 'source_mode': mode})
            kinds[kind] = kinds.get(kind, 0) + 1
            out.append(rec)
    rng.shuffle(out)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(json.dumps({'rows': len(out), 'missing_media': missing, 'answer_types': kinds}))


if __name__ == '__main__':
    main()
