#!/usr/bin/env python3
import os
import argparse
from huggingface_hub import snapshot_download

# Base models to download
BASE_MODELS = {
    "gemma4": {
        "repo_id": "google/gemma-4-E4B-it",
        "local_dir": "/workspace/finetune_gemma/models/base/gemma-4-E4B-it"
    },
    "qwen_coder": {
        "repo_id": "Qwen/Qwen2.5-Coder-3B-Instruct",
        "local_dir": "/workspace/finetune_gemma/models/base/Qwen2.5-Coder-3B-Instruct"
    },
    "qwen_instruct": {
        "repo_id": "Qwen/Qwen2.5-3B-Instruct",
        "local_dir": "/workspace/finetune_gemma/models/base/Qwen2.5-3B-Instruct"
    }
}

def main():
    parser = argparse.ArgumentParser(description="Download Base Models from Hugging Face Hub")
    parser.add_argument(
        "--model", 
        type=str, 
        choices=["all", "gemma4", "qwen_coder", "qwen_instruct"], 
        default="all",
        help="Specific base model to download (default: all)"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face access token"
    )
    args = parser.parse_args()

    selected_models = []
    if args.model == "all":
        selected_models = list(BASE_MODELS.keys())
    else:
        selected_models = [args.model]

    for model_key in selected_models:
        model_info = BASE_MODELS[model_key]
        repo_id = model_info["repo_id"]
        local_dir = model_info["local_dir"]

        print(f"\n==========================================")
        print(f"Downloading Base Model: {repo_id}")
        print(f"Target Directory: {local_dir}")
        print(f"==========================================")

        os.makedirs(local_dir, exist_ok=True)

        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                token=args.token
            )
            print(f"✅ Successfully downloaded {repo_id} to {local_dir}!")
        except Exception as e:
            print(f"❌ Error downloading {repo_id}: {e}")

if __name__ == "__main__":
    main()
