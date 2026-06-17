#!/bin/bash
# Download and extract DebugEval dataset (raw and atcoder cases)
# Usage: ./download_debugeval_data.sh [DOWNLOAD_URL]

set -e

# Default download URL placeholder (replace with your hosted dataset URL)
DATA_URL="${1:-https://your-storage-bucket.com/debugeval_data.tar.gz}"
TARGET_DIR="/workspace/finetune_gemma/data/debugeval"

echo "=============================================="
echo "Preparing DebugEval Dataset"
echo "Target: ${TARGET_DIR}"
echo "=============================================="

mkdir -p "${TARGET_DIR}"

# Note: Since the dataset is large (atcoder cases & raw JSONL files),
# users setting this up locally can download the hosted tarball or copy it.
if [ -f "/workspace/finetune_gemma.tar.gz" ]; then
    echo "Found local backup at /workspace/finetune_gemma.tar.gz."
    echo "Extracting only the data/debugeval directory..."
    tar -xf /workspace/finetune_gemma.tar.gz -C /workspace/ --wildcards "finetune_gemma/data/debugeval/*"
    echo "✅ Extraction complete from backup!"
else
    echo "Downloading from: ${DATA_URL}"
    # In a real environment, you'd run:
    # wget -O "${TARGET_DIR}/debugeval_data.tar.gz" "${DATA_URL}"
    # tar -xzf "${TARGET_DIR}/debugeval_data.tar.gz" -C "${TARGET_DIR}"
    # rm "${TARGET_DIR}/debugeval_data.tar.gz"
    echo "⚠️ Please replace DATA_URL or provide a download URL as an argument to run the download."
    echo "Example: ./download_debugeval_data.sh https://my-bucket.com/debugeval_data.tar.gz"
fi

echo ""
echo "Current contents of ${TARGET_DIR}:"
ls -lh "${TARGET_DIR}"
