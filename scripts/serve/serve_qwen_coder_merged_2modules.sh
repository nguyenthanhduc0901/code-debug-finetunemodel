source /venv/main/bin/activate

export CUDA_VISIBLE_DEVICES=0
export VLLM_USE_V1=0

vllm serve /workspace/finetune_gemma/models/finetuned/debugeval/qwen-coder-3b-2modules-merged \
    --host 0.0.0.0 --port 8888 \
    --served-model-name qwen-sft \
    --tensor-parallel-size 1 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --language-model-only \
    --enable-prefix-caching \
    --enable-request-id-headers \
    --disable-access-log-for-endpoints "/health,/metrics,/ping"
