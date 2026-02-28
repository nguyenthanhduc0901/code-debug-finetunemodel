#!/bin/bash
# =============================================================
# setup_env.sh
# Install all Python dependencies needed for COAST / NeuDebugger
# Usage: bash setup_env.sh
#
# Requirements: Python 3.10+, CUDA 11.8 or 12.x
# =============================================================
set -e

echo "======================================================"
echo "  COAST - Environment Setup"
echo "======================================================"

# ── 1. Ensure pip is available ─────────────────────────────
echo ""
echo "[1/4] Checking pip..."
if ! command -v pip &>/dev/null && ! python3 -m pip --version &>/dev/null; then
    echo "  pip not found. Installing via get-pip.py..."
    curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
    python3 /tmp/get-pip.py
fi
export PATH="$HOME/.local/bin:$PATH"
echo "  pip OK: $(pip --version)"

# ── 2. Detect CUDA version and install matching PyTorch ────
echo ""
echo "[2/4] Installing PyTorch (auto-detecting CUDA version)..."
CUDA_VER=$(nvcc --version 2>/dev/null | grep release | sed 's/.*release //' | sed 's/,.*//' | cut -d. -f1,2 | sed 's/\.//')
if [ -z "$CUDA_VER" ]; then
    # fallback: try nvidia-smi
    CUDA_VER=$(nvidia-smi 2>/dev/null | grep "CUDA Version" | sed 's/.*CUDA Version: //' | cut -d. -f1,2 | sed 's/\.//')
fi

if [ -z "$CUDA_VER" ]; then
    echo "  No GPU detected. Installing CPU-only PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
elif [ "$CUDA_VER" -ge 120 ]; then
    echo "  Detected CUDA 12.x → installing PyTorch cu121"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
elif [ "$CUDA_VER" -ge 118 ]; then
    echo "  Detected CUDA 11.8 → installing PyTorch cu118"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "  Detected CUDA $CUDA_VER → installing PyTorch cu117"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
fi

# ── 3. Install DeepSpeed (needs --no-build-isolation to find torch) ─
echo ""
echo "[3/4] Installing DeepSpeed..."
pip install "deepspeed>=0.15.0" --no-build-isolation

# ── 4. Install remaining dependencies ────────────────────────
echo ""
echo "[4/4] Installing remaining requirements..."
pip install -r requirements.txt

echo ""
echo "======================================================"
echo "  Setup complete! Verify with:"
echo "    python3 -c \"import torch, transformers, deepspeed, peft; print('All OK')\""
echo "======================================================"
