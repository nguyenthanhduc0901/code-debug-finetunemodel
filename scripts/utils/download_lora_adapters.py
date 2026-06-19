import os
import argparse
from huggingface_hub import snapshot_download
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
LORA_MODELS = {'gemma4_lora': {'repo_id': 'ntduc0901/gemma4-debugeval-lora', 'local_dir': os.path.join(PROJECT_ROOT, 'models/finetuned/debugeval/gemma4-sft')}, 'qwen_coder_lora': {'repo_id': 'ntduc0901/qwen-debugeval-lora', 'local_dir': os.path.join(PROJECT_ROOT, 'models/finetuned/debugeval/qwen-coder-3b-sft')}, 'qwen_socratic_lora': {'repo_id': 'ntduc0901/qwen-socratic-lora', 'local_dir': os.path.join(PROJECT_ROOT, 'models/finetuned/socratic/qwen-3b-sft')}}

def main():
    parser = argparse.ArgumentParser(description='Download Fine-tuned LoRA Adapters from Hugging Face Hub')
    parser.add_argument('--model', type=str, choices=['all', 'gemma4_lora', 'qwen_coder_lora', 'qwen_socratic_lora'], default='all', help='Specific LoRA model to download (default: all)')
    parser.add_argument('--token', type=str, default=os.environ.get('HF_TOKEN'), help='Hugging Face access token (required for private repositories)')
    args = parser.parse_args()
    selected_models = []
    if args.model == 'all':
        selected_models = list(LORA_MODELS.keys())
    else:
        selected_models = [args.model]
    for model_key in selected_models:
        model_info = LORA_MODELS[model_key]
        repo_id = model_info['repo_id']
        local_dir = model_info['local_dir']
        print(f'\n==========================================')
        print(f'Downloading LoRA Adapter: {repo_id}')
        print(f'Target Directory: {local_dir}')
        print(f'==========================================')
        os.makedirs(local_dir, exist_ok=True)
        try:
            snapshot_download(repo_id=repo_id, local_dir=local_dir, local_dir_use_symlinks=False, token=args.token)
            print(f'Successfully downloaded {repo_id} to {local_dir}!')
        except Exception as e:
            print(f'Error downloading {repo_id}: {e}')
if __name__ == '__main__':
    main()
