#!/bin/bash
# Serve the fine-tuned Socratic model using vLLM
# Supports both merged model and base+LoRA modes

set -e
source /venv/main/bin/activate

BASE_DIR="/workspace/finetune_gemma"
MERGED_DIR="${BASE_DIR}/models/finetuned/socratic/qwen-3b-merged"
BASE_MODEL="${BASE_DIR}/models/base/Qwen2.5-3B-Instruct"
ADAPTER_DIR="${BASE_DIR}/models/finetuned/socratic/qwen-3b-sft"

PORT=${PORT:-8001}
HOST=${HOST:-0.0.0.0}

# Prefer merged model, fall back to base model (for comparison)
if [ -d "${MERGED_DIR}" ] && [ -f "${MERGED_DIR}/config.json" ]; then
    MODEL_PATH="${MERGED_DIR}"
    echo "Using merged model: ${MODEL_PATH}"
    EXTRA_ARGS=""
elif [ -d "${BASE_MODEL}" ]; then
    MODEL_PATH="${BASE_MODEL}"
    echo "Using base model (no fine-tuning): ${MODEL_PATH}"
    EXTRA_ARGS=""
else
    echo "❌ No model found. Download or train first."
    exit 1
fi

echo "=============================================="
echo "Serving Socratic Debugging Model"
echo "  Model: ${MODEL_PATH}"
echo "  Host:  ${HOST}:${PORT}"
echo "=============================================="

python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.22 \
    --enforce-eager \
    --served-model-name "socratic-tutor" \
    ${EXTRA_ARGS}

