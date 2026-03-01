# NeuDebugger — Code Debugging with LLMs (DebugEval + COAST)

![title](https://github.com/NEUIR/COAST/blob/main/Figure/title.png)

<p align="center">
  <a href="https://arxiv.org/pdf/2408.05006">📜 Paper</a> •
  <a href="https://huggingface.co/datasets/yangweiqing/DebugEval">🤗 Dataset</a> •
  <a href="https://huggingface.co/ntduc0901">🤖 Fine-tuned Models</a>
</p>

## Overview

**DebugEval** is a benchmark for evaluating code-debugging ability of LLMs across four tasks:

| Task | Description |
|------|-------------|
| BUG Localization | Identify the buggy code snippet from multiple-choice options |
| BUG Identification | Classify the error type (Syntax / Reference / Logical / Multiple) |
| Code Repair | Generate the corrected version of buggy code |
| Code Review | Determine which of two code snippets contains a bug |

**COAST** is the multi-agent data synthesis framework used to generate training data that improves LLM debugging ability.

This repository fine-tunes two models:

| Model | Base | Fine-tuned Adapter (HuggingFace) |
|-------|------|----------------------------------|
| NeuDebugger-DeepSeek | [deepseek-ai/deepseek-coder-6.7b-instruct](https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-instruct) | [ntduc0901/deepseek-coder-6.7b-debugeval-lora](https://huggingface.co/ntduc0901/deepseek-coder-6.7b-debugeval-lora) |
| NeuDebugger-Llama3 | [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) | [ntduc0901/llama3-8b-debugeval-lora](https://huggingface.co/ntduc0901/llama3-8b-debugeval-lora) |

---

## Quick Start

### 1. Clone

```bash
git clone https://github.com/nguyenthanhduc0901/code-debug-finetunemodel
cd code-debug-finetunemodel
```

### 2. Install Dependencies

```bash
bash setup_env.sh
```

Auto-detects your CUDA version and installs matching PyTorch, then all other packages
(transformers, deepspeed, peft, trl, datasets, etc.).

> **Requirements:** Python 3.10+, CUDA 11.8+ and an NVIDIA GPU (H100 80GB recommended for fine-tuning)

### 3. Download All Assets

```bash
# Download everything: both base models + both LoRA adapters + DebugEval dataset
bash download_assets.sh
```

Individual flags:

```bash
bash download_assets.sh --model-only     # both base models only
bash download_assets.sh --deepseek-only  # DeepSeek-Coder-6.7B only
bash download_assets.sh --llama-only     # Llama3-8B only
bash download_assets.sh --adapter-only   # both fine-tuned LoRA adapters only
bash download_assets.sh --data-only      # DebugEval dataset only
```

**Asset summary:**

| Asset | Source | Local path | Size |
|-------|--------|-----------|------|
| DeepSeek-Coder-6.7B-Instruct | [HuggingFace](https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-instruct) | `models/deepseek-coder-6.7b-instruct/` | ~13.5 GB |
| Meta-Llama-3-8B-Instruct | [HuggingFace (mirror)](https://huggingface.co/NousResearch/Meta-Llama-3-8B-Instruct) | `models/llama3-8b-instruct/` | ~16 GB |
| NeuDebugger-DeepSeek LoRA | [ntduc0901/deepseek-coder-6.7b-debugeval-lora](https://huggingface.co/ntduc0901/deepseek-coder-6.7b-debugeval-lora) | `output/deepseek-coder-6.7b-finetuned/` | ~14 MB |
| NeuDebugger-Llama3 LoRA | [ntduc0901/llama3-8b-debugeval-lora](https://huggingface.co/ntduc0901/llama3-8b-debugeval-lora) | `output/llama3-8b-finetuned/` | ~14 MB |
| DebugEval dataset | [yangweiqing/DebugEval](https://huggingface.co/datasets/yangweiqing/DebugEval) | `Data/` | ~5.5 GB total |

---

## Quick Test: Base vs Fine-tuned Comparison

After downloading, run a side-by-side comparison of base model vs NeuDebugger:

```bash
# 4 test cases (fast, ~2 min)
python3 test_model.py

# 20 test cases across Python / C++ / Java, all 4 tasks (~5 min)
python3 eval_compare.py
```

Both scripts load DeepSeek-Coder-6.7B base, run inference, then reload with the LoRA adapter
and print a comparison table with scores.

---

## Full Inference & Evaluation Pipeline

The full pipeline uses vLLM for serving and supports `deepseek_FT_cot`, `deepseek_FT_no_cot`,
`llama3_FT_cot`, `llama3_FT_no_cot`, and base API models.

### Step 1 — Serve model with vLLM

```bash
# Serve base model (no LoRA)
bash src/serve/serve_ckpt.sh models/deepseek-coder-6.7b-instruct 8888

# Serve base model + LoRA adapter (model alias used in inference scripts)
bash src/serve/serve_ckpt.sh \
    models/deepseek-coder-6.7b-instruct 8888 \
    output/deepseek-coder-6.7b-finetuned deepseek_FT_cot

# Llama3 with LoRA
bash src/serve/serve_ckpt.sh \
    models/llama3-8b-instruct 8888 \
    output/llama3-8b-finetuned llama3_FT_cot
```

### Step 2 — Run inference scripts

```bash
# Set MODEL to one of:
#   deepseek_FT_cot | deepseek_FT_no_cot | llama3_FT_cot | llama3_FT_no_cot | vllm-agent

MODEL=deepseek_FT_cot bash src/scripts/error_type_identification.sh
MODEL=deepseek_FT_cot bash src/scripts/error_code_localization.sh
MODEL=deepseek_FT_cot bash src/scripts/code_repair.sh
MODEL=deepseek_FT_cot bash src/scripts/code_review.sh
MODEL=deepseek_FT_cot bash src/scripts/code_review_reverse.sh
```

Results are written to `output/eval_results/<task>/zero_shot/`.

### Step 3 — Calculate accuracy

```bash
python3 bug_loc_calculate_acc.py   # BUG Localization
python3 bug_iden_calculate_acc.py  # BUG Identification
python3 code_rep_calculate_acc.py  # Code Repair (pass@1)
python3 code_rev_calculate_acc.py  # Code Review
```

---

## Fine-Tuning (Reproduce from Scratch)

Both scripts use **LoRA** (rank=8, alpha=32) with **DeepSpeed ZeRO-2** on a single GPU.
Run after completing Steps 1–3 under Quick Start (models + dataset must be downloaded).

### Fine-tune DeepSeek-Coder-6.7B

```bash
bash train_deepseek.sh
# Adapter saved to: output/deepseek-coder-6.7b-finetuned/
```

### Fine-tune Meta-Llama-3-8B

```bash
bash train_llama3.sh
# Adapter saved to: output/llama3-8b-finetuned/
```

**Training configuration** (both models):

| Parameter | Value |
|-----------|-------|
| Epochs | 1 |
| Batch size | 4 × 8 (grad accum) = 32 |
| Learning rate | 2e-5 (cosine decay, 30 warmup steps) |
| Max seq length | 2048 |
| LoRA rank / alpha | 8 / 32 |
| Hardware | NVIDIA H100 80GB |
| DeepSpeed | ZeRO-2, no CPU offload |
| Training data | 24,892 COAST samples |

Training takes ~40–45 minutes per model on an H100.

---

## Upload Fine-tuned Adapter to HuggingFace

```bash
python3 push_to_hub.py \
    --adapter_dir output/llama3-8b-finetuned \
    --repo your-username/llama3-8b-debugeval-lora \
    --token hf_YOUR_TOKEN

# DeepSeek adapter:
python3 push_to_hub.py \
    --adapter_dir output/deepseek-coder-6.7b-finetuned \
    --repo your-username/deepseek-coder-6.7b-debugeval-lora \
    --token hf_YOUR_TOKEN
```

---

## Project Structure

```
.
├── README.md
├── setup_env.sh                  # install all Python deps
├── download_assets.sh            # download models, adapters, dataset
├── _download_helper.py           # internal Python helper for download_assets.sh
├── train_deepseek.sh             # fine-tune DeepSeek-Coder-6.7B
├── train_llama3.sh               # fine-tune Meta-Llama-3-8B
├── push_to_hub.py                # upload LoRA adapter to HuggingFace
├── test_model.py                 # quick 4-case base vs fine-tuned comparison
├── eval_compare.py               # extended 20-case comparison (Python/C++/Java)
├── bug_loc_calculate_acc.py      # BUG Localization accuracy calculator
├── bug_iden_calculate_acc.py     # BUG Identification accuracy calculator
├── code_rep_calculate_acc.py     # Code Repair pass@k calculator
├── code_rev_calculate_acc.py     # Code Review accuracy calculator
├── requirements.txt
├── Data/                         # DebugEval dataset (downloaded)
├── models/                       # base model weights (downloaded)
│   ├── deepseek-coder-6.7b-instruct/
│   └── llama3-8b-instruct/
├── output/                       # fine-tuned LoRA adapters (downloaded or trained)
│   ├── deepseek-coder-6.7b-finetuned/
│   └── llama3-8b-finetuned/
├── neural_compiler/
│   └── src/finetune/
│       ├── fine-tune-deepseek-coder.py
│       ├── fine-tune-llama3.py
│       ├── ds_config_deepseek_coder.json
│       └── ds_config_llama3.json
├── src/
│   ├── inference/                # inference runners + post-processing
│   ├── prompts/                  # prompt templates (zero-shot, per model/task)
│   ├── scripts/                  # one-liner eval launchers per task
│   ├── serve/                    # vLLM serving helpers
│   └── code_repair_eval_python/  # Python code execution sandbox for repair eval
└── OJ_Evaluation/                # online judge for C++/Java code repair eval
```

---

## Results

![performance](https://github.com/NEUIR/DebugEval/blob/main/Figure/performance_00.png)

---

## Citation

```bibtex
@misc{yang2025coastenhancingcodedebugging,
  title={COAST: Enhancing the Code Debugging Ability of LLMs through Communicative Agent Based Data Synthesis},
  author={Weiqing Yang and Hanbin Wang and Zhenghao Liu and Xinze Li and Yukun Yan and Shuo Wang and Yu Gu and Minghe Yu and Zhiyuan Liu and Ge Yu},
  year={2025},
  eprint={2408.05006},
  archivePrefix={arXiv},
  primaryClass={cs.SE},
  url={https://arxiv.org/abs/2408.05006},
}
```
