#!/usr/bin/env bash
set -euo pipefail
: "${MODEL:?set MODEL to the merged SFT checkpoint}"
: "${RL_DATA:?set RL_DATA to the prepare_rl.py output}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TOOLS="$HERE/../tools"
NPROC_PER_NODE=${NPROC_PER_NODE:-$(nvidia-smi -L | wc -l)}
export NPROC_PER_NODE
export MAX_PIXELS=${MAX_PIXELS:-1003520}
export VIDEO_MAX_PIXELS=${VIDEO_MAX_PIXELS:-401408}
export FPS_MAX_FRAMES=${FPS_MAX_FRAMES:-32}
mkdir -p "$OUTPUT_DIR"
python "$TOOLS/difficulty_filter.py" expand --input "$RL_DATA" --output "$OUTPUT_DIR/probe.jsonl" --samples "${SAMPLES:-4}"
DEFAULT_MM='{"image": 32, "video": 2}'
LIMIT_MM=${LIMIT_MM:-$DEFAULT_MM}
swift infer \
    --model "$MODEL" \
    --model_type qwen3_vl \
    --infer_backend vllm \
    --external_plugins "$HERE/../flair/plugin.py" \
    --val_dataset "$OUTPUT_DIR/probe.jsonl" \
    --result_path "$OUTPUT_DIR/probe_results.jsonl" \
    --temperature "${TEMPERATURE:-1.0}" \
    --top_p 1.0 \
    --max_new_tokens "${MAX_COMPLETION:-1024}" \
    --vllm_max_model_len "${MAX_MODEL_LEN:-16384}" \
    --vllm_gpu_memory_utilization "${GPU_MEM:-0.85}" \
    --vllm_limit_mm_per_prompt "$LIMIT_MM" \
    --write_batch_size "${WRITE_BATCH:-2000}" \
    "$@"
python "$TOOLS/difficulty_filter.py" select --input "$RL_DATA" --probe "$OUTPUT_DIR/probe.jsonl" --results "$OUTPUT_DIR/probe_results.jsonl" \
    --output "$OUTPUT_DIR/rl_train.jsonl" --limit "${KEEP_LIMIT:-0}"
