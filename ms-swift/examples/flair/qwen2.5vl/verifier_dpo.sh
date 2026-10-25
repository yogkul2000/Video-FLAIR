#!/usr/bin/env bash
set -euo pipefail
: "${VERIFIER_MODEL:?set VERIFIER_MODEL}"
: "${DATASET:?set DATASET to build_verifier_dpo.py output}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"
NPROC_PER_NODE=${DPO_NPROC:-1}
export NPROC_PER_NODE
export MASTER_PORT=${DPO_MASTER_PORT:-29611}
export MAX_PIXELS=${MAX_PIXELS:-401408}
swift rlhf \
    --rlhf_type dpo \
    --model "$VERIFIER_MODEL" \
    --model_type qwen3_vl \
    --dataset "$DATASET" \
    --tuner_type lora \
    --lora_rank "${LORA_RANK:-64}" \
    --lora_alpha "${LORA_ALPHA:-128}" \
    --target_modules all-linear \
    --freeze_vit true \
    --torch_dtype bfloat16 \
    --attn_impl "${ATTN_IMPL:-sdpa}" \
    --beta "${DPO_BETA:-0.1}" \
    --num_train_epochs "${EPOCHS:-1}" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps "${GRAD_ACC:-8}" \
    --learning_rate "${LR:-5e-6}" \
    --warmup_ratio 0.05 \
    --max_length "${MAX_LENGTH:-8192}" \
    --truncation_strategy delete \
    --gradient_checkpointing true \
    --save_strategy epoch \
    --save_only_model true \
    --logging_steps 5 \
    --report_to "${REPORT_TO:-tensorboard}" \
    --output_dir "$OUTPUT_DIR" \
    "$@"
