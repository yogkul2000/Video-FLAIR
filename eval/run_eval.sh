#!/bin/bash
set -e
MODEL=$(cd "$1" && pwd)
mkdir -p "$2"
OUT=$(cd "$2" && pwd)
shift 2
TASKS=${*:-"ai2d mmmu_val mathvista_testmini_format hallusion_bench_image mmvetv2 emma video_holmes videott_single_mc vsibench scivideobench video_mmmu_flair"}
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE/lmms-eval"
export API_TYPE=${API_TYPE:-openai}
export JUDGE_MIN_TOKENS=${JUDGE_MIN_TOKENS:-4096}
export MAX_PIXELS=${MAX_PIXELS:-1003520} VIDEO_MAX_PIXELS=${VIDEO_MAX_PIXELS:-401408}
for t in $TASKS; do
  python -m lmms_eval \
    --model flair \
    --model_args pretrained=$MODEL,mode=${MODE:-ADAPTIVE},frame_cache=$OUT/frames \
    --tasks $t \
    --batch_size 1 \
    --log_samples \
    --output_path $OUT/$t ${LIMIT:+--limit $LIMIT}
done
