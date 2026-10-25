import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flair.parsing import THINK, strip_ground_markup
from flair.prompts import MODE_PROMPTS


def time_line(meta):
    ts = meta.get('frame_ts')
    if meta.get('kind') != 'video' or not ts:
        return ''
    return 'Frame timestamps (seconds): ' + ', '.join(f'{t:.2f}' for t in ts) + '\n'


def media_text(row, meta):
    q = row['messages'][0]['content']
    if q.startswith('<video>'):
        return '<video>' + time_line(meta) + q[len('<video>'):]
    return q


def rewrite_direct(completion):
    m = THINK.search(completion)
    if not m:
        return completion
    return completion[:m.start(1)] + strip_ground_markup(m.group(1)) + completion[m.end(1):]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--media_root', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--modes', default='DIRECT,CONCISE,DEEP')
    ap.add_argument('--stages', default='')
    ap.add_argument('--per_mode_limit', type=int, default=0)
    ap.add_argument('--adaptive_ratio', type=float, default=0.1)
    ap.add_argument('--direct_tags', choices=['strip', 'keep'], default='strip')
    ap.add_argument('--timestamps', type=int, default=1)
    ap.add_argument('--val_size', type=int, default=200)
    ap.add_argument('--check_media', type=int, default=0)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    modes = set(args.modes.split(','))
    stages = set(args.stages.split(',')) if args.stages else None
    media_root = os.path.abspath(args.media_root)
    by_mode = {}
    missing = 0
    with open(args.input) as f:
        for line in f:
            row = json.loads(line)
            meta = row.get('meta', {})
            if meta.get('mode') not in modes or (stages and meta.get('stage') not in stages):
                continue
            by_mode.setdefault(meta['mode'], []).append(row)
    out_rows = []
    for mode, rows in sorted(by_mode.items()):
        rng.shuffle(rows)
        if args.per_mode_limit:
            rows = rows[:args.per_mode_limit]
        for row in rows:
            meta = row['meta']
            completion = row['messages'][1]['content']
            if mode == 'DIRECT' and args.direct_tags == 'strip':
                completion = rewrite_direct(completion)
            prompt_mode = 'ADAPTIVE' if rng.random() < args.adaptive_ratio else mode
            user = media_text(row, meta) if args.timestamps else row['messages'][0]['content']
            rec = {
                'messages': [
                    {'role': 'system', 'content': MODE_PROMPTS[prompt_mode]},
                    {'role': 'user', 'content': user},
                    {'role': 'assistant', 'content': completion},
                ]
            }
            key = 'videos' if 'videos' in row else 'images'
            media = row[key]
            if key == 'videos':
                rec[key] = [[os.path.join(media_root, p) for p in v] for v in media]
                flat = [p for v in rec[key] for p in v]
            else:
                rec[key] = [os.path.join(media_root, p) for p in media]
                flat = rec[key]
            if args.check_media and not all(os.path.exists(p) for p in flat):
                missing += 1
                continue
            rec['sid'] = meta['sid']
            rec['flair_mode'] = mode
            rec['prompt_mode'] = prompt_mode
            out_rows.append(rec)
    rng.shuffle(out_rows)
    os.makedirs(args.output_dir, exist_ok=True)
    val = out_rows[:args.val_size]
    train = out_rows[args.val_size:]
    for name, rows in (('train', train), ('val', val)):
        with open(os.path.join(args.output_dir, f'sft_{name}.jsonl'), 'w') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
    counts = {}
    for r in out_rows:
        k = f"{r['flair_mode']}/{r['prompt_mode']}"
        counts[k] = counts.get(k, 0) + 1
    print(json.dumps({'train': len(train), 'val': len(val), 'missing_media': missing, 'counts': counts}))


if __name__ == '__main__':
    main()
