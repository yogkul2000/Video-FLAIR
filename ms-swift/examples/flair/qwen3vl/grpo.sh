#!/usr/bin/env bash
set -euo pipefail
: "${MODEL:?set MODEL to the merged SFT checkpoint}"
: "${DATASET:?set DATASET to the difficulty-filtered RL jsonl}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
NPROC_PER_NODE=${NPROC_PER_NODE:-$(nvidia-smi -L | wc -l)}
export NPROC_PER_NODE
export MAX_PIXELS=${MAX_PIXELS:-1003520}
export VIDEO_MAX_PIXELS=${VIDEO_MAX_PIXELS:-401408}
export FPS_MAX_FRAMES=${FPS_MAX_FRAMES:-32}
export FLAIR_PAIR_DIR=${FLAIR_PAIR_DIR:-$OUTPUT_DIR/verifier_pairs}
export FLAIR_VERIFIER_MODEL_FILE=${FLAIR_VERIFIER_MODEL_FILE:-}
mkdir -p "$OUTPUT_DIR" "$FLAIR_PAIR_DIR"
DEFAULT_MM='{"image": 32, "video": 2}'
LIMIT_MM=${LIMIT_MM:-$DEFAULT_MM}
swift rlhf \
    --rlhf_type grpo \
    --model "$MODEL" \
    --model_type qwen3_vl \
    --dataset "$DATASET" \
    --external_plugins "$HERE/../flair/plugin.py" \
    --reward_funcs flair_ans flair_format flair_comply flair_cost flair_balance flair_select flair_verifier \
    --reward_weights ${REWARD_WEIGHTS:-1.0 0.25 0.6 0.2 0.2 0.4 0.35} \
    --scale_rewards gdpo \
    --loss_type dapo \
    --epsilon 0.2 \
    --epsilon_high 0.28 \
    --beta "${KL_BETA:-0.0}" \
    --num_generations "${K:-8}" \
    --temperature 1.0 \
    --top_p 1.0 \
    --max_completion_length "${MAX_COMPLETION:-1024}" \
    --max_length "${MAX_LENGTH:-16384}" \
    --use_vllm true \
    --vllm_mode "${VLLM_MODE:-colocate}" \
    --vllm_gpu_memory_utilization "${VLLM_GPU_MEM:-0.4}" \
    --vllm_max_model_len "${MAX_LENGTH:-16384}" \
    --vllm_limit_mm_per_prompt "$LIMIT_MM" \
    --sleep_level "${SLEEP_LEVEL:-1}" \
    --tuner_type "${TUNER_TYPE:-lora}" \
    --lora_rank "${LORA_RANK:-64}" \
    --lora_alpha "${LORA_ALPHA:-128}" \
    --target_modules all-linear \
    --freeze_vit true \
    --torch_dtype bfloat16 \
    --attn_impl "${ATTN_IMPL:-sdpa}" \
    --learning_rate "${LR:-1e-6}" \
    --warmup_ratio 0.05 \
    --num_train_epochs "${EPOCHS:-1}" \
    --max_steps "${MAX_STEPS:--1}" \
    --per_device_train_batch_size "${BATCH_SIZE:-8}" \
    --gradient_accumulation_steps "${GRAD_ACC:-2}" \
    --gradient_checkpointing true \
    --deepspeed "${DEEPSPEED:-zero2}" \
    --dataloader_num_workers "${NUM_WORKERS:-4}" \
    --save_steps "${SAVE_STEPS:-100}" \
    --save_total_limit "${SAVE_LIMIT:-5}" \
    --logging_steps 1 \
    --log_completions true \
    --report_to "${REPORT_TO:-tensorboard}" \
    --output_dir "$OUTPUT_DIR" \
    "$@"
