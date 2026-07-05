source /venv/main/bin/activate

python3 -m vllm.entrypoints.openai.api_server \
    --model /workspace/finetune_gemma/models/finetuned/socratic/qwen-3b-2modules-merged \
    --served-model-name qwen2.5-2modules-finetuned socratic-tutor \
    --port 8001 \
    --host 0.0.0.0 \
    --gpu-memory-utilization 0.85 \
    --enforce-eager
