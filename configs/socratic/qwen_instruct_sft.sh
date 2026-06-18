#!/bin/bash

set -e
source /venv/main/bin/activate

export PYTHONPATH=/workspace/finetune_gemma/tools/LLaMA-Factory/src:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BASE_DIR="/workspace/finetune_gemma"
MODEL_DIR="${BASE_DIR}/models/base/Qwen2.5-3B-Instruct"
OUTPUT_DIR="${BASE_DIR}/models/finetuned/socratic/qwen-3b-sft"
DATA_DIR="${BASE_DIR}/data/socratic/sft"

# Verify model exists
if [ ! -d "${MODEL_DIR}" ]; then
    echo "❌ Base model not found at ${MODEL_DIR}"
    echo "   Run ./download_model.sh first"
    exit 1
fi

# Verify data exists
if [ ! -f "${DATA_DIR}/socratic_train.json" ]; then
    echo "❌ Training data not found at ${DATA_DIR}/socratic_train.json"
    echo "   Run python convert_data.py first"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "=============================================="
echo "Starting SFT Training"
echo "  Base model: Qwen2.5-3B-Instruct (General)"
echo "  Dataset:    Socratic Debugging"
echo "  Method:     LoRA (rank 64, all-linear)"
echo "  Output:     ${OUTPUT_DIR}"
echo "=============================================="

llamafactory-cli train \
    --stage sft \
    --do_train \
    --model_name_or_path "${MODEL_DIR}" \
    --dataset socratic_debug \
    --dataset_dir "${DATA_DIR}" \
    --template qwen \
    --finetuning_type lora \
    --lora_target q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --lora_rank 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --output_dir "${OUTPUT_DIR}" \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 2048 \
    --preprocessing_num_workers 4 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 5 \
    --warmup_steps 30 \
    --save_steps 100 \
    --learning_rate 2e-5 \
    --num_train_epochs 5.0 \
    --save_total_limit 3 \
    --plot_loss \
    --bf16 True

echo ""
echo "✅ Training complete!"
echo "   Model saved to: ${OUTPUT_DIR}"
