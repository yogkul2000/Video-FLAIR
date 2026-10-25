#!/usr/bin/env bash
set -euo pipefail
: "${VERIFIER_MODEL:?set VERIFIER_MODEL to a Qwen3-VL-8B-Instruct path or hub id}"
DEFAULT_MM='{"image": 8}'
LIMIT_MM=${LIMIT_MM:-$DEFAULT_MM}
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
exec vllm serve "$VERIFIER_MODEL" \
    --served-model-name verifier \
    --host "${HOST:-0.0.0.0}" \
    --port "${PORT:-8100}" \
    --max-model-len "${MAX_MODEL_LEN:-16384}" \
    --gpu-memory-utilization "${GPU_MEM:-0.85}" \
    --limit-mm-per-prompt "$LIMIT_MM" \
    --enable-lora \
    --max-lora-rank "${LORA_RANK:-64}" \
    --max-loras "${MAX_LORAS:-2}" \
    "$@"
