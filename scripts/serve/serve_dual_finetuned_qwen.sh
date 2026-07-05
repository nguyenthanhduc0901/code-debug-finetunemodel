#!/bin/bash
source /venv/main/bin/activate

export CUDA_VISIBLE_DEVICES=0

echo "=================================================="
echo "🚀 Starting Dual Qwen Fine-Tuned Servers on vLLM"
echo "=================================================="

# 1. Qwen Socratic Tutor Server (Port 10100)
python3 -m vllm.entrypoints.openai.api_server \
    --model /workspace/finetune_gemma/models/finetuned/socratic/qwen-3b-2modules-merged \
    --served-model-name qwen-socratic socratic-tutor \
    --port 10100 \
    --host 0.0.0.0 \
    --gpu-memory-utilization 0.38 \
    --max-model-len 16384 \
    --enforce-eager &
SOCRATIC_PID=$!
echo "Started Qwen Socratic Tutor Server (PID: $SOCRATIC_PID) on Port 10100"

echo "⏳ Waiting for Qwen Socratic Tutor to be fully ready..."
while ! curl -s http://localhost:10100/v1/models >/dev/null; do
    sleep 3
done
echo "✅ Qwen Socratic Tutor is ready!"

# 2. Qwen Debug Code Server (Port 10200)
python3 -m vllm.entrypoints.openai.api_server \
    --model /workspace/finetune_gemma/models/finetuned/debugeval/qwen-coder-3b-merged \
    --served-model-name qwen-coder-sft debug-coder \
    --port 10200 \
    --host 0.0.0.0 \
    --gpu-memory-utilization 0.38 \
    --max-model-len 16384 \
    --enforce-eager &
CODER_PID=$!
echo "Started Qwen Debug Code Server (PID: $CODER_PID) on Port 10200"

echo "⏳ Waiting for Qwen Debug Code Server to be fully ready..."
while ! curl -s http://localhost:10200/v1/models >/dev/null; do
    sleep 3
done
echo "✅ Qwen Debug Code Server is ready!"

echo "=================================================="
echo "🎉 Both vLLM Servers are fully active & public!"
echo "Port 10100 (Public: 46492): Qwen Socratic Tutor"
echo "Port 10200 (Public: 46405): Qwen Debug Code"
echo "=================================================="

wait $SOCRATIC_PID $CODER_PID
