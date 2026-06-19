source /venv/main/bin/activate

export CUDA_VISIBLE_DEVICES=0

vllm serve /workspace/finetune_gemma/models/finetuned/socratic/qwen-3b-merged \
    --host 0.0.0.0 --port 8001 \
    --served-model-name socratic-tutor \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.22 \
    --enforce-eager
