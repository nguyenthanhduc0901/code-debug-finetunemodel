#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
# source ./source.sh
# check_conda_env_and_activate llm-agent

set -m  # Enable job control

MODEL_DIR=$1
PORT=$2  # default to 8888
LORA_DIR=$3  # optional: path to LoRA adapter (e.g. output/deepseek-coder-6.7b-finetuned)
LORA_NAME=${4:-"deepseek_FT_cot"}  # alias used in inference scripts (--model arg)

if [ -z "$PORT" ]; then
    PORT=8888
fi

# parse CUDA_VISIBLE_DEVICES to get number of GPUs
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    N_GPUS=0
else
    IFS=',' read -ra GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
    N_GPUS=${#GPU_IDS[@]}
fi

echo "Serve model from $MODEL_DIR with $N_GPUS GPUs on port $PORT"

# Build optional LoRA args
if [ -n "$LORA_DIR" ]; then
    echo "Loading LoRA adapter from $LORA_DIR as model alias '$LORA_NAME'"
    LORA_ARGS="--enable-lora --lora-modules ${LORA_NAME}=${LORA_DIR}"
else
    LORA_ARGS=""
fi

python3 -m vllm.entrypoints.openai.api_server \
        --model $MODEL_DIR \
        --served-model-name vllm-agent \
        --tensor-parallel-size $N_GPUS \
        --host 0.0.0.0 --port $PORT \
        --swap-space 1 \
        $LORA_ARGS
