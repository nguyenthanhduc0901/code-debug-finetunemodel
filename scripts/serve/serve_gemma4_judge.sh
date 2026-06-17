#!/bin/bash
# Serve Gemma-4 base model as the Socratic Judge LLM using vLLM
#
# Configured for concurrent execution on 32GB RTX 5090:
# - Capped at --gpu-memory-utilization 0.45
# - Capped at --max-model-len 4096

set -e
source /venv/main/bin/activate

BASE_DIR="/workspace/finetune_gemma"
MODEL_PATH="${BASE_DIR}/models/base/gemma-4-E4B-it"

PORT=${PORT:-8002}
HOST=${HOST:-0.0.0.0}

if [ ! -d "${MODEL_PATH}" ]; then
    echo "❌ Gemma-4 model not found at ${MODEL_PATH}"
    exit 1
fi

echo "=============================================="
# Start serving
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.52 \
    --enforce-eager \
    --served-model-name "gemma-4-base" \
    --trust-remote-code \
    --language-model-only

