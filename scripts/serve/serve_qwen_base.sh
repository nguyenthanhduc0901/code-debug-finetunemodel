#!/bin/bash
source /venv/main/bin/activate

python3 -m vllm.entrypoints.openai.api_server \
    --model /workspace/finetune_gemma/models/base/Qwen2.5-3B-Instruct \
    --served-model-name qwen2.5-3b-base socratic-tutor \
    --port 8001 \
    --host 0.0.0.0 \
    --gpu-memory-utilization 0.85 \
    --enforce-eager
