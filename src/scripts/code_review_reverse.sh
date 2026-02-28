#export CUDA_VISIBLE_DEVICES=1,3
# MODEL options: 'deepseek_FT_cot' (our fine-tuned), 'deepseek_FT_no_cot',
#                'llama3_FT_cot', 'llama3_FT_no_cot', 'vllm-agent' (base via vLLM)
MODEL=${MODEL:-"deepseek_FT_cot"}
OUTPUT_DIR="output/eval_results/code_review_reverse"
mkdir -p "$OUTPUT_DIR"

python3 src/inference/main.py \
    --model "$MODEL" \
    --data_path "Data/eval/debugevalsuite_task124.jsonl" \
    --prompt_dir "src/prompts" \
    --output_dir "$OUTPUT_DIR" \
    --task "code_review_reverse" \
    --prompt_type "zero_shot" \
    --platform "all" \
    --n 1 \
    --temperature 0.2 \
    --top_p 0.95 \
    --max_tokens 1024