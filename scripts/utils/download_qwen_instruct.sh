#!/bin/bash

set -e
source /venv/main/bin/activate

MODEL_ID="Qwen/Qwen2.5-3B-Instruct"
TARGET_DIR="/workspace/finetune_gemma/models/base/Qwen2.5-3B-Instruct"

echo "=============================================="
echo "Downloading ${MODEL_ID}"
echo "Target: ${TARGET_DIR}"
echo "=============================================="

mkdir -p "${TARGET_DIR}"

python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='${MODEL_ID}',
    local_dir='${TARGET_DIR}',
    local_dir_use_symlinks=False,
)
print('Download complete!')
"

echo ""
echo "Model saved to: ${TARGET_DIR}"
ls -lh "${TARGET_DIR}"
