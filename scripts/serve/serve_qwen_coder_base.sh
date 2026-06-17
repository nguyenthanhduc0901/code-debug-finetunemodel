#!/bin/bash
source /venv/main/bin/activate

export CUDA_VISIBLE_DEVICES=0

echo "Starting vLLM serving for Qwen2.5-Coder-3B-Instruct from local directory..."

vllm serve /workspace/finetune_gemma/models/base/Qwen2.5-Coder-3B-Instruct \
    --host 0.0.0.0 --port 8888 \
    --served-model-name qwen2.5-coder-3b-instruct \
    --tensor-parallel-size 1 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --language-model-only \
    --enable-prefix-caching \
    --enable-request-id-headers \
    --disable-access-log-for-endpoints "/health,/metrics,/ping"
