import os
import argparse
from huggingface_hub import HfApi, create_repo
USERNAME = 'ntduc0901'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
MODELS_TO_UPLOAD = [{'local_dir': os.path.join(PROJECT_ROOT, 'models/finetuned/debugeval/gemma4-sft'), 'repo_name': 'gemma4-debugeval-lora', 'base_model_id': 'google/gemma-4-E4B-it', 'local_base_path': os.path.join(PROJECT_ROOT, 'models/google/gemma-4-E4B-it')}, {'local_dir': os.path.join(PROJECT_ROOT, 'models/finetuned/debugeval/qwen-coder-3b-sft'), 'repo_name': 'qwen-debugeval-lora', 'base_model_id': 'Qwen/Qwen2.5-Coder-3B-Instruct', 'local_base_path': os.path.join(PROJECT_ROOT, 'models/base/Qwen2.5-Coder-3B-Instruct')}, {'local_dir': os.path.join(PROJECT_ROOT, 'models/finetuned/socratic/qwen-3b-sft'), 'repo_name': 'qwen-socratic-lora', 'base_model_id': 'Qwen/Qwen2.5-3B-Instruct', 'local_base_path': os.path.join(PROJECT_ROOT, 'finetune-socrates/models/base/Qwen2.5-3B-Instruct')}]

def patch_readme(readme_path, local_base_path, base_model_id):
    if not os.path.exists(readme_path):
        return
    print(f'Patching README: {readme_path}')
    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()
    patched_content = content.replace(local_base_path, base_model_id)
    alternative_local_path = local_base_path.replace('models/google/', 'models/base/')
    patched_content = patched_content.replace(alternative_local_path, base_model_id)
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(patched_content)
    print('README patched successfully.')

def main():
    parser = argparse.ArgumentParser(description='Upload Fine-tuned LoRA Adapters to Hugging Face Hub')
    parser.add_argument('--token', type=str, default=os.environ.get('HF_TOKEN'), help='Hugging Face access token with Write permission')
    args = parser.parse_args()
    token = args.token
    if not token:
        token = os.environ.get('HF_TOKEN')
    if not token:
        print('❌ Error: Hugging Face token is missing. Please set HF_TOKEN env var or use --token.')
        sys.exit(1)
    api = HfApi(token=token)
    for item in MODELS_TO_UPLOAD:
        local_dir = item['local_dir']
        repo_name = item['repo_name']
        repo_id = f'{USERNAME}/{repo_name}'
        base_model_id = item['base_model_id']
        local_base_path = item['local_base_path']
        if not os.path.exists(local_dir):
            print(f'❌ Local directory does not exist: {local_dir}. Skipping...')
            continue
        print(f'\n==========================================')
        print(f'Processing & Uploading: {repo_name}')
        print(f'==========================================')
        readme_path = os.path.join(local_dir, 'README.md')
        patch_readme(readme_path, local_base_path, base_model_id)
        try:
            print(f'Creating repo (private=True, exist_ok=True): {repo_id}...')
            create_repo(repo_id=repo_id, token=token, private=True, exist_ok=True)
            print(f'Uploading folder content...')
            api.upload_folder(folder_path=local_dir, repo_id=repo_id, repo_type='model', token=token)
            print(f'✅ Successfully uploaded {repo_name} to Hugging Face!')
        except Exception as e:
            print(f'❌ Error uploading {repo_name}: {e}')
if __name__ == '__main__':
    main()
