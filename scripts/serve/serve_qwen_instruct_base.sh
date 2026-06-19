#!/bin/bash
source /venv/main/bin/activate

export CUDA_VISIBLE_DEVICES=0

echo "Starting vLLM serving for Qwen2.5-3B-Instruct (Base Model) on port 8001..."

vllm serve /workspace/finetune_gemma/models/base/Qwen2.5-3B-Instruct \
    --host 0.0.0.0 --port 8001 \
    --served-model-name qwen-instruct-base \
    --tensor-parallel-size 1 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.35 \
    --enforce-eager \
    --trust-remote-code \
    --language-model-only \
    --disable-access-log-for-endpoints "/health,/metrics,/ping"
