#!/bin/bash
source /venv/main/bin/activate

# LLaMA-Factory needs to import modules from tools
export PYTHONPATH=/workspace/finetune_gemma/tools/LLaMA-Factory/src:$PYTHONPATH

echo "Merging LoRA adapter into base Qwen model..."

llamafactory-cli export \
    --model_name_or_path /workspace/finetune_gemma/models/base/Qwen2.5-Coder-3B-Instruct \
    --adapter_name_or_path /workspace/finetune_gemma/models/finetuned/debugeval/qwen-coder-3b-sft \
    --template qwen \
    --finetuning_type lora \
    --export_dir /workspace/finetune_gemma/models/finetuned/debugeval/qwen-coder-3b-merged \
    --export_size 2 \
    --export_device cpu \
    --export_legacy_format False

echo "Merge completed successfully. Saved to: /workspace/finetune_gemma/models/finetuned/debugeval/qwen-coder-3b-merged"
