#!/usr/bin/env bash
set -euo pipefail
: "${MODEL:?set MODEL to a Qwen2.5-VL checkpoint path or hub id}"
: "${DATA_DIR:?set DATA_DIR to the prepare_sft.py output dir}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"
NPROC_PER_NODE=${NPROC_PER_NODE:-$(nvidia-smi -L | wc -l)}
export NPROC_PER_NODE
export MAX_PIXELS=${MAX_PIXELS:-1003520}
export VIDEO_MAX_PIXELS=${VIDEO_MAX_PIXELS:-401408}
export FPS_MAX_FRAMES=${FPS_MAX_FRAMES:-32}
swift sft \
    --model "$MODEL" \
    --model_type qwen2_5_vl \
    --dataset "$DATA_DIR/sft_train.jsonl" \
    --val_dataset "$DATA_DIR/sft_val.jsonl" \
    --load_from_cache_file true \
    --tuner_type "${TUNER_TYPE:-lora}" \
    --lora_rank "${LORA_RANK:-64}" \
    --lora_alpha "${LORA_ALPHA:-128}" \
    --target_modules all-linear \
    --freeze_vit true \
    --torch_dtype bfloat16 \
    --attn_impl "${ATTN_IMPL:-sdpa}" \
    --num_train_epochs "${EPOCHS:-1}" \
    --max_steps "${MAX_STEPS:--1}" \
    --per_device_train_batch_size "${BATCH_SIZE:-1}" \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps "${GRAD_ACC:-4}" \
    --learning_rate "${LR:-1e-4}" \
    --warmup_ratio 0.05 \
    --lr_scheduler_type cosine \
    --max_length "${MAX_LENGTH:-12288}" \
    --truncation_strategy delete \
    --gradient_checkpointing true \
    --deepspeed "${DEEPSPEED:-zero2}" \
    --dataloader_num_workers "${NUM_WORKERS:-4}" \
    --dataset_num_proc "${DATASET_PROC:-8}" \
    --eval_steps "${EVAL_STEPS:-200}" \
    --save_steps "${SAVE_STEPS:-200}" \
    --save_total_limit "${SAVE_LIMIT:-3}" \
    --logging_steps 5 \
    --report_to "${REPORT_TO:-tensorboard}" \
    --output_dir "$OUTPUT_DIR" \
    "$@"
