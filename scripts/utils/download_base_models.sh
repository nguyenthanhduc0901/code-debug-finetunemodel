#!/bin/bash
# Wrapper script to run download_base_models.py

set -e
source /venv/main/bin/activate

python3 /workspace/finetune_gemma/scripts/utils/download_base_models.py "$@"
