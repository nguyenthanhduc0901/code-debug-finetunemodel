"""Upload a LoRA adapter directory to HuggingFace Hub."""
import argparse
from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser(description="Push a LoRA adapter to HuggingFace Hub")
    parser.add_argument("--adapter_dir", required=True, help="Local path to the adapter directory")
    parser.add_argument("--repo", required=True, help="HuggingFace repo id, e.g. username/my-lora-adapter")
    parser.add_argument("--token", required=True, help="HuggingFace API token")
    parser.add_argument("--private", action="store_true", default=False, help="Create as private repo")
    args = parser.parse_args()

    api = HfApi(token=args.token)

    # Create repo if it does not exist yet
    api.create_repo(repo_id=args.repo, private=args.private, exist_ok=True)

    print(f"Uploading {args.adapter_dir} → {args.repo} ...")
    api.upload_folder(
        folder_path=args.adapter_dir,
        repo_id=args.repo,
        repo_type="model",
        token=args.token,
    )
    print(f"Done! View at: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
