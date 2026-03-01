#!/bin/bash
# =============================================================
# Fine-tune Meta-Llama-3-8B-Instruct on DebugEval COAST data
# Single GPU: NVIDIA H100 80GB
# =============================================================

export PATH="$HOME/.local/bin:$PATH"

# Resolve project root (directory of this script)
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

DATA_PATH="$PROJECT_ROOT/Data/train/data.json"
OUTPUT_PATH="$PROJECT_ROOT/output/llama3-8b-finetuned"
MODEL_PATH="$PROJECT_ROOT/models/llama3-8b-instruct"
DS_CONFIG="$PROJECT_ROOT/neural_compiler/src/finetune/ds_config_llama3.json"
FINETUNE_SCRIPT="$PROJECT_ROOT/neural_compiler/src/finetune/fine-tune-llama3.py"

mkdir -p "$OUTPUT_PATH"
mkdir -p "$PROJECT_ROOT/logs"

echo "=================================================="
echo "  Training: Meta-Llama-3-8B-Instruct (LoRA)"
echo "  Data   : $DATA_PATH"
echo "  Output : $OUTPUT_PATH"
echo "  GPU    : CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}"
echo "=================================================="

deepspeed --include=localhost:${CUDA_VISIBLE_DEVICES:-0} \
    "$FINETUNE_SCRIPT" \
    --model_name_or_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_PATH" \
    --num_train_epochs 1 \
    --model_max_length 2048 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --eval_strategy "no" \
    --save_strategy "epoch" \
    --save_steps 100 \
    --learning_rate 2e-5 \
    --warmup_steps 30 \
    --logging_steps 1 \
    --lr_scheduler_type "cosine" \
    --gradient_checkpointing True \
    --report_to "tensorboard" \
    --deepspeed "$DS_CONFIG" \
    --bf16 True \
    --optim "adamw_torch" \
    --use_lora True
