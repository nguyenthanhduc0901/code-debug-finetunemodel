source /venv/main/bin/activate

export CUDA_VISIBLE_DEVICES=0

vllm serve /workspace/finetune_gemma/models/base/gemma-4-E4B-it \
    --host 0.0.0.0 --port 8002 \
    --served-model-name gemma-4-base \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.70 \
    --enforce-eager \
    --trust-remote-code \
    --language-model-only
