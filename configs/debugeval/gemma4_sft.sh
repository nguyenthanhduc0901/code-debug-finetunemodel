#!/bin/bash

source /venv/main/bin/activate

export PYTHONPATH=/workspace/finetune_gemma/tools/LLaMA-Factory/src:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUTPUT_DIR="/workspace/finetune_gemma/models/finetuned/debugeval/gemma4-sft"
mkdir -p $OUTPUT_DIR

echo "Starting SFT training for Gemma-4-E4B-it using LLaMA-Factory..."

llamafactory-cli train \
    --stage sft \
    --do_train \
    --model_name_or_path /workspace/finetune_gemma/models/base/gemma-4-E4B-it \
    --dataset debugeval_sft \
    --dataset_dir /workspace/finetune_gemma/data/debugeval/sft \
    --template gemma4 \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --lora_rank 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --output_dir $OUTPUT_DIR \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 2048 \
    --preprocessing_num_workers 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --warmup_steps 50 \
    --save_steps 200 \
    --learning_rate 5e-5 \
    --num_train_epochs 2.0 \
    --plot_loss \
    --bf16 True \
    --no_enable_thinking
