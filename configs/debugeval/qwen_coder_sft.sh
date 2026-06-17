#!/bin/bash
source /venv/main/bin/activate

# LLaMA-Factory needs to import modules from tools
export PYTHONPATH=/workspace/finetune_gemma/tools/LLaMA-Factory/src:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Directory to save the checkpoints
OUTPUT_DIR="/workspace/finetune_gemma/models/finetuned/debugeval/qwen-coder-3b-sft"
mkdir -p $OUTPUT_DIR

echo "Starting SFT training for Qwen2.5-Coder-3B-Instruct using LLaMA-Factory..."

# Run SFT Fine-tuning (all-linear modules LoRA)
llamafactory-cli train \
    --stage sft \
    --do_train \
    --model_name_or_path /workspace/finetune_gemma/models/base/Qwen2.5-Coder-3B-Instruct \
    --dataset debugeval_sft \
    --dataset_dir /workspace/finetune_gemma/data/debugeval/sft \
    --template qwen \
    --finetuning_type lora \
    --lora_target q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --lora_rank 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --output_dir $OUTPUT_DIR \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 2048 \
    --preprocessing_num_workers 4 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --warmup_steps 50 \
    --save_steps 200 \
    --learning_rate 5e-5 \
    --num_train_epochs 2.0 \
    --save_total_limit 2 \
    --plot_loss \
    --bf16 True
