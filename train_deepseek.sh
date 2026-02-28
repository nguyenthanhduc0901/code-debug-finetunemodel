#!/bin/bash
# =============================================================
# Fine-tune DeepSeek-Coder-6.7B-Instruct on DebugEval COAST data
# Single GPU: NVIDIA H100 80GB
# =============================================================

export PATH="$HOME/.local/bin:$PATH"

DATA_PATH="/home/ubuntu/COAST/Data/train/data.json"
OUTPUT_PATH="/home/ubuntu/COAST/output/deepseek-coder-6.7b-finetuned"
MODEL_PATH="/home/ubuntu/COAST/models/deepseek-coder-6.7b-instruct"
DS_CONFIG="/home/ubuntu/COAST/neural_compiler/src/finetune/ds_config_deepseek_coder.json"

deepspeed --include=localhost:0 \
    /home/ubuntu/COAST/neural_compiler/src/finetune/fine-tune-deepseek-coder.py \
    --model_name_or_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_PATH \
    --num_train_epochs 1 \
    --model_max_length 2048 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_steps 100 \
    --learning_rate 2e-5 \
    --warmup_steps 30 \
    --logging_steps 1 \
    --lr_scheduler_type "cosine" \
    --gradient_checkpointing True \
    --report_to "tensorboard" \
    --deepspeed $DS_CONFIG \
    --bf16 True \
    --use_lora True
