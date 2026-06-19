#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TARGET_DIR="${PROJECT_ROOT}/data/debugeval"
BACKUP_TAR="${PROJECT_ROOT}/../finetune_gemma.tar.gz"

DATA_URL="${1:-https://your-storage-bucket.com/debugeval_data.tar.gz}"

echo "=============================================="
echo "Preparing DebugEval Dataset"
echo "Target: ${TARGET_DIR}"
echo "=============================================="

mkdir -p "${TARGET_DIR}"

if [ -f "${BACKUP_TAR}" ]; then
    echo "Found local backup at ${BACKUP_TAR}."
    echo "Extracting only the data/debugeval directory..."
    tar -xf "${BACKUP_TAR}" -C "${PROJECT_ROOT}/.." --wildcards "finetune_gemma/data/debugeval/*"
    echo "Extraction complete from backup!"
else
    echo "Downloading from: ${DATA_URL}"
    echo "Please replace DATA_URL or provide a download URL as an argument to run the download."
    echo "Example: ./download_debugeval_data.sh https://my-bucket.com/debugeval_data.tar.gz"
fi

echo ""
echo "Current contents of ${TARGET_DIR}:"
ls -lh "${TARGET_DIR}"
