#!/bin/bash
set -e
source /venv/main/bin/activate

export PYTHONPATH=/workspace/finetune_gemma/tools/LLaMA-Factory/src:$PYTHONPATH

BASE_DIR="/workspace/finetune_gemma"
MODEL_DIR="${BASE_DIR}/models/base/Qwen2.5-3B-Instruct"
ADAPTER_DIR="${BASE_DIR}/models/finetuned/socratic/qwen-3b-sft"
MERGED_DIR="${BASE_DIR}/models/finetuned/socratic/qwen-3b-merged"

# Verify paths
if [ ! -d "${MODEL_DIR}" ]; then
    echo "❌ Base model not found: ${MODEL_DIR}"
    exit 1
fi

if [ ! -f "${ADAPTER_DIR}/adapter_config.json" ]; then
    echo "❌ LoRA adapter not found: ${ADAPTER_DIR}"
    echo "   Run ./run_sft.sh first"
    exit 1
fi

mkdir -p "${MERGED_DIR}"

echo "=============================================="
echo "Merging LoRA Adapter with Base Model"
echo "  Base:    ${MODEL_DIR}"
echo "  Adapter: ${ADAPTER_DIR}"
echo "  Output:  ${MERGED_DIR}"
echo "=============================================="

llamafactory-cli export \
    --model_name_or_path "${MODEL_DIR}" \
    --adapter_name_or_path "${ADAPTER_DIR}" \
    --template qwen \
    --finetuning_type lora \
    --export_dir "${MERGED_DIR}" \
    --export_size 4 \
    --export_legacy_format False

echo ""
echo "✅ Merge complete!"
echo "   Merged model saved to: ${MERGED_DIR}"
ls -lh "${MERGED_DIR}"
