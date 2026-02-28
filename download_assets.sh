#!/bin/bash
# =============================================================
# download_assets.sh
# Download model weights and DebugEval dataset from HuggingFace
#
# Usage:
#   bash download_assets.sh              # download both model & data
#   bash download_assets.sh --model-only
#   bash download_assets.sh --data-only
#
# Downloaded to:
#   models/deepseek-coder-6.7b-instruct/  (~13.5 GB)
#   Data/train/data.json                   (~30 MB)
#   Data/eval/debugevalsuite_task124.jsonl
#   Data/eval/debugevalsuite_task3.jsonl
#   Data/test_cases.zip
# =============================================================
set -e

export PATH="$HOME/.local/bin:$PATH"

MODE="both"
if [[ "$1" == "--model-only" ]]; then MODE="model"; fi
if [[ "$1" == "--data-only"  ]]; then MODE="data";  fi

echo "======================================================"
echo "  COAST - Download Assets  (mode: $MODE)"
echo "======================================================"

# ── check huggingface_hub ──────────────────────────────────
if ! python3 -c "import huggingface_hub" &>/dev/null; then
    echo "Installing huggingface_hub..."
    pip install huggingface_hub
fi

# ── Download model ─────────────────────────────────────────
download_model() {
    MODEL_DIR="models/deepseek-coder-6.7b-instruct"
    if [ -f "$MODEL_DIR/config.json" ]; then
        echo ""
        echo "[Model] Already exists at $MODEL_DIR — skipping."
        echo "  (Delete the folder to re-download)"
        return
    fi
    echo ""
    echo "[Model] Downloading deepseek-ai/deepseek-coder-6.7b-instruct (~13.5 GB)..."
    echo "  Destination: $MODEL_DIR"
    mkdir -p "$MODEL_DIR"
    python3 - <<'EOF'
import sys
from huggingface_hub import snapshot_download
path = snapshot_download(
    "deepseek-ai/deepseek-coder-6.7b-instruct",
    local_dir="models/deepseek-coder-6.7b-instruct",
    ignore_patterns=["*.bin"],   # prefer .safetensors
)
print(f"  Model saved to: {path}")
EOF
}

# ── Download dataset ───────────────────────────────────────
download_data() {
    DATA_DIR="Data"
    if [ -f "$DATA_DIR/train/data.json" ] && [ -f "$DATA_DIR/eval/debugevalsuite_task3.jsonl" ]; then
        echo ""
        echo "[Data] Already exists at $DATA_DIR — skipping."
        echo "  (Delete Data/train/ and Data/eval/ to re-download)"
        return
    fi
    echo ""
    echo "[Data] Downloading yangweiqing/DebugEval dataset..."
    echo "  Destination: $DATA_DIR"
    mkdir -p "$DATA_DIR"
    python3 - <<'EOF'
from huggingface_hub import snapshot_download
path = snapshot_download(
    "yangweiqing/DebugEval",
    repo_type="dataset",
    local_dir="Data",
)
print(f"  Dataset saved to: {path}")
EOF
}

# ── Run ────────────────────────────────────────────────────
cd "$(dirname "$0")"   # always run from project root

if [[ "$MODE" == "both"  || "$MODE" == "model" ]]; then download_model; fi
if [[ "$MODE" == "both"  || "$MODE" == "data"  ]]; then download_data;  fi

echo ""
echo "======================================================"
echo "  Download complete!"
echo ""
echo "  Quick verify:"
echo "    python3 -c \"import json; d=json.load(open('Data/train/data.json')); print('Train samples:', len(d))\""
echo "    ls models/deepseek-coder-6.7b-instruct/"
echo "======================================================"
