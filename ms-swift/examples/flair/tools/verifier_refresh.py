import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request


def post(url, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=600) as r:
        return r.read().decode()


def latest_adapter(out_dir):
    best, best_step = None, -1
    for root, dirs, files in os.walk(out_dir):
        if 'adapter_config.json' in files:
            name = os.path.basename(root)
            step = int(name.split('-')[-1]) if name.startswith('checkpoint-') and name.split('-')[-1].isdigit() else 0
            if step >= best_step:
                best, best_step = root, step
    return best


def write_model_file(path, name):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump({'model': name, 'time': time.time()}, f)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pair_dir', required=True)
    ap.add_argument('--work_dir', required=True)
    ap.add_argument('--server_url', required=True)
    ap.add_argument('--model_file', default='')
    ap.add_argument('--dpo_script', required=True)
    ap.add_argument('--every', type=int, default=100)
    ap.add_argument('--max_pairs', type=int, default=2000)
    ap.add_argument('--margin', type=float, default=0.15)
    ap.add_argument('--min_rows', type=int, default=64)
    ap.add_argument('--poll', type=float, default=30.0)
    ap.add_argument('--frames', type=int, default=4)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(args.work_dir, exist_ok=True)
    state_path = os.path.join(args.pair_dir, 'state.json')
    pairs_path = os.path.join(args.pair_dir, 'pairs.jsonl')
    log_path = os.path.join(args.work_dir, 'refresh_log.jsonl')
    done = -1
    version = 0
    while True:
        time.sleep(args.poll)
        if not os.path.exists(state_path) or not os.path.exists(pairs_path):
            continue
        try:
            with open(state_path) as f:
                step = int(json.load(f)['step'])
        except Exception:
            continue
        due = (step // args.every) * args.every
        if due <= 0 or due <= done:
            continue
        version += 1
        vdir = os.path.join(args.work_dir, f'v{version}')
        os.makedirs(vdir, exist_ok=True)
        data = os.path.join(vdir, 'dpo.jsonl')
        out = subprocess.run([
            sys.executable, os.path.join(here, 'build_verifier_dpo.py'), '--pairs', pairs_path, '--output', data,
            '--max_pairs', str(args.max_pairs), '--margin', str(args.margin), '--frames', str(args.frames)
        ], capture_output=True, text=True)
        info = json.loads(out.stdout.strip().splitlines()[-1]) if out.returncode == 0 else {'error': out.stderr[-2000:]}
        rec = {'version': version, 'step': step, 'build': info}
        if out.returncode != 0 or info.get('rows', 0) < args.min_rows:
            rec['status'] = 'skipped'
            version -= 1
            done = due
        else:
            env = dict(os.environ, DATASET=data, OUTPUT_DIR=os.path.join(vdir, 'train'))
            with open(os.path.join(vdir, 'train.log'), 'w') as lf:
                rc = subprocess.run(['bash', args.dpo_script], env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
            adapter = latest_adapter(os.path.join(vdir, 'train')) if rc == 0 else None
            if adapter is None:
                rec['status'] = f'dpo_failed_rc{rc}'
            else:
                name = f'verifier_v{version}'
                try:
                    post(args.server_url.rstrip('/') + '/load_lora_adapter', {'lora_name': name, 'lora_path': adapter})
                    if args.model_file:
                        write_model_file(args.model_file, name)
                    rec.update(status='swapped', adapter=adapter, model=name)
                except Exception as e:
                    rec['status'] = f'load_failed: {e}'
            done = due
        rec['time'] = time.time()
        with open(log_path, 'a') as f:
            f.write(json.dumps(rec) + '\n')
        print(json.dumps(rec), flush=True)


if __name__ == '__main__':
    main()
