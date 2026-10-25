#!/bin/bash
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"
if [ ! -d lmms-eval ]; then
  git clone https://github.com/evolvinglmms-lab/lmms-eval.git
  git -C lmms-eval checkout 08affba133b97b19351897f4d42081ead2b3412e
fi
cp -r overlay/lmms_eval/. lmms-eval/lmms_eval/
python - <<'PY'
p = "lmms-eval/lmms_eval/models/__init__.py"
s = open(p).read()
if '"flair": "Flair"' not in s:
    s = s.replace('    "dummy": "Dummy",\n', '    "dummy": "Dummy",\n    "flair": "Flair",\n    "flair_base": "FlairBase",\n', 1)
    open(p, "w").write(s)
PY
printf 'torch==2.11.0\ntransformers==5.12.1\nvllm==0.23.0\nav==19.0.1\n' > overrides.txt
uv pip install -e lmms-eval --override overrides.txt
