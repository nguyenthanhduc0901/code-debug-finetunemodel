"""
Internal helper called by download_assets.sh.
Usage: python3 _download_helper.py <what>
  what: deepseek_model | llama_model | deepseek_adapter | llama_adapter | data
"""
import os
import sys

try:
    from huggingface_hub import snapshot_download
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])
    from huggingface_hub import snapshot_download


def dl(repo_id, local_dir, repo_type="model", ignore_patterns=None):
    os.makedirs(local_dir, exist_ok=True)
    path = snapshot_download(
        repo_id,
        local_dir=local_dir,
        repo_type=repo_type,
        ignore_patterns=ignore_patterns or [],
    )
    print(f"  Saved to: {path}")


what = sys.argv[1] if len(sys.argv) > 1 else ""

if what == "deepseek_model":
    dl("deepseek-ai/deepseek-coder-6.7b-instruct",
       "models/deepseek-coder-6.7b-instruct",
       ignore_patterns=["*.bin"])

elif what == "llama_model":
    dl("NousResearch/Meta-Llama-3-8B-Instruct",
       "models/llama3-8b-instruct",
       ignore_patterns=["*.bin"])

elif what == "deepseek_adapter":
    dl("ntduc0901/deepseek-coder-6.7b-debugeval-lora",
       "output/deepseek-coder-6.7b-finetuned",
       repo_type="model")

elif what == "llama_adapter":
    dl("ntduc0901/llama3-8b-debugeval-lora",
       "output/llama3-8b-finetuned",
       repo_type="model")

elif what == "data":
    dl("yangweiqing/DebugEval",
       "Data",
       repo_type="dataset")

else:
    print(f"Unknown target: {what}", file=sys.stderr)
    sys.exit(1)
