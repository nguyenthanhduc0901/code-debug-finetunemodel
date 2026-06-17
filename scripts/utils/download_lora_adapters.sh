#!/bin/bash
# Wrapper script to run download_lora_adapters.py

set -e
source /venv/main/bin/activate

python3 /workspace/finetune_gemma/scripts/utils/download_lora_adapters.py "$@"
