#!/bin/bash
# =============================================================
# download_assets.sh
# Download model weights, LoRA adapters and DebugEval dataset
#
# Usage:
#   bash download_assets.sh                  # download everything
#   bash download_assets.sh --deepseek-only  # DeepSeek base model only
#   bash download_assets.sh --llama-only     # Llama3 base model only
#   bash download_assets.sh --model-only     # both base models
#   bash download_assets.sh --adapter-only   # both fine-tuned LoRA adapters
#   bash download_assets.sh --data-only      # DebugEval dataset only
#
# Assets:
#   models/deepseek-coder-6.7b-instruct/       (~13.5 GB)
#   models/llama3-8b-instruct/                  (~16 GB)
#   output/deepseek-coder-6.7b-finetuned/       (~14 MB, LoRA adapter)
#   output/llama3-8b-finetuned/                 (~14 MB, LoRA adapter)
#   Data/train/data.json                         (~49 MB)
#   Data/eval/debugevalsuite_task124.jsonl
#   Data/eval/debugevalsuite_task3.jsonl
# =============================================================
set -e

export PATH="$HOME/.local/bin:$PATH"

# Always run from project root
cd "$(dirname "$0")"
HELPER="$(pwd)/_download_helper.py"

MODE="all"
case "$1" in
    --deepseek-only) MODE="deepseek" ;;
    --llama-only)    MODE="llama"    ;;
    --model-only)    MODE="models"   ;;
    --adapter-only)  MODE="adapters" ;;
    --data-only)     MODE="data"     ;;
esac

echo "======================================================"
echo "  COAST / NeuDebugger — Download Assets  (mode: $MODE)"
echo "======================================================"

# ───────────────────────────── helpers ──────────────────────
download_deepseek_model() {
    if [ -f "models/deepseek-coder-6.7b-instruct/config.json" ]; then
        echo "[Model] deepseek-coder-6.7b-instruct already exists — skipping."; return
    fi
    echo ""; echo "[Model] Downloading deepseek-ai/deepseek-coder-6.7b-instruct (~13.5 GB)..."
    python3 "$HELPER" deepseek_model
}

download_llama_model() {
    if [ -f "models/llama3-8b-instruct/config.json" ]; then
        echo "[Model] llama3-8b-instruct already exists — skipping."; return
    fi
    echo ""; echo "[Model] Downloading NousResearch/Meta-Llama-3-8B-Instruct (~16 GB)..."
    python3 "$HELPER" llama_model
}

download_deepseek_adapter() {
    if [ -f "output/deepseek-coder-6.7b-finetuned/adapter_model.safetensors" ]; then
        echo "[Adapter] deepseek-coder-6.7b-finetuned already exists — skipping."; return
    fi
    echo ""; echo "[Adapter] Downloading ntduc0901/deepseek-coder-6.7b-debugeval-lora (~14 MB)..."
    python3 "$HELPER" deepseek_adapter
}

download_llama_adapter() {
    if [ -f "output/llama3-8b-finetuned/adapter_model.safetensors" ]; then
        echo "[Adapter] llama3-8b-finetuned already exists — skipping."; return
    fi
    echo ""; echo "[Adapter] Downloading ntduc0901/llama3-8b-debugeval-lora (~14 MB)..."
    python3 "$HELPER" llama_adapter
}

download_data() {
    if [ -f "Data/train/data.json" ] && [ -f "Data/eval/debugevalsuite_task3.jsonl" ]; then
        echo "[Data] DebugEval dataset already exists — skipping."; return
    fi
    echo ""; echo "[Data] Downloading yangweiqing/DebugEval dataset..."
    python3 "$HELPER" data
}

# ───────────────────────────── run ──────────────────────────
case "$MODE" in
    all)
        download_deepseek_model
        download_llama_model
        download_deepseek_adapter
        download_llama_adapter
        download_data
        ;;
    models)
        download_deepseek_model
        download_llama_model
        ;;
    deepseek)  download_deepseek_model ;;
    llama)     download_llama_model ;;
    adapters)
        download_deepseek_adapter
        download_llama_adapter
        ;;
    data)      download_data ;;
esac

echo ""
echo "======================================================"
echo "  Download complete!"
echo ""
echo "  Quick verify:"
echo "    ls models/deepseek-coder-6.7b-instruct/"
echo "    ls models/llama3-8b-instruct/"
echo "    ls output/deepseek-coder-6.7b-finetuned/"
echo "    ls output/llama3-8b-finetuned/"
echo "    python3 -c \"import json; d=json.load(open('Data/train/data.json')); print('Train samples:', len(d))\""
echo "======================================================"
