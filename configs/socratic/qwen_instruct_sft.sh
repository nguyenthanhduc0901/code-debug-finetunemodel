source /venv/main/bin/activate

export PYTHONPATH=/workspace/finetune_gemma/tools/LLaMA-Factory/src:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

llamafactory-cli train \
    --stage sft \
    --do_train \
    --model_name_or_path /workspace/finetune_gemma/models/base/Qwen2.5-3B-Instruct \
    --dataset socratic_debug \
    --dataset_dir /workspace/finetune_gemma/data/socratic/sft \
    --template qwen \
    --finetuning_type lora \
    --lora_target q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --lora_rank 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --output_dir /workspace/finetune_gemma/models/finetuned/socratic/qwen-3b-sft \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 2048 \
    --preprocessing_num_workers 4 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 5 \
    --warmup_steps 30 \
    --save_steps 100 \
    --learning_rate 2e-5 \
    --num_train_epochs 5.0 \
    --save_total_limit 3 \
    --plot_loss \
    --bf16 True
