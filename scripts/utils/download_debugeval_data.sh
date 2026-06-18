#!/bin/bash
# Download and extract DebugEval dataset (raw and atcoder cases)
# Usage: ./download_debugeval_data.sh [DOWNLOAD_URL]

set -e

# Resolve paths dynamically relative to script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TARGET_DIR="${PROJECT_ROOT}/data/debugeval"
BACKUP_TAR="${PROJECT_ROOT}/../finetune_gemma.tar.gz"

# Default download URL placeholder (replace with your hosted dataset URL)
DATA_URL="${1:-https://your-storage-bucket.com/debugeval_data.tar.gz}"

echo "=============================================="
echo "Preparing DebugEval Dataset"
echo "Target: ${TARGET_DIR}"
echo "=============================================="

mkdir -p "${TARGET_DIR}"

# Note: Since the dataset is large (atcoder cases & raw JSONL files),
# users setting this up locally can download the hosted tarball or copy it.
if [ -f "${BACKUP_TAR}" ]; then
    echo "Found local backup at ${BACKUP_TAR}."
    echo "Extracting only the data/debugeval directory..."
    tar -xf "${BACKUP_TAR}" -C "${PROJECT_ROOT}/.." --wildcards "finetune_gemma/data/debugeval/*"
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
