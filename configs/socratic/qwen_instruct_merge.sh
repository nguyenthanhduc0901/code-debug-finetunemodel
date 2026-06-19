source /venv/main/bin/activate

export PYTHONPATH=/workspace/finetune_gemma/tools/LLaMA-Factory/src:$PYTHONPATH

llamafactory-cli export \
    --model_name_or_path /workspace/finetune_gemma/models/base/Qwen2.5-3B-Instruct \
    --adapter_name_or_path /workspace/finetune_gemma/models/finetuned/socratic/qwen-3b-sft \
    --template qwen \
    --finetuning_type lora \
    --export_dir /workspace/finetune_gemma/models/finetuned/socratic/qwen-3b-merged \
    --export_size 4 \
    --export_legacy_format False
