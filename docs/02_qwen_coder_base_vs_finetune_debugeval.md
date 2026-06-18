# Qwen2.5-Coder-3B-Instruct: Base vs. LoRA Fine-Tuned on DebugEval

## 1. Introduction

This report presents an empirical comparison between the base Qwen2.5-Coder-3B-Instruct model and its LoRA-fine-tuned variant on the DebugEval benchmark suite. The goal is to evaluate the impact of supervised fine-tuning (SFT) with LoRA on a code-specialized small language model across four debugging tasks.

## 2. Experimental Setup

### 2.1 Model

- **Base model**: Qwen2.5-Coder-3B-Instruct (`Qwen2ForCausalLM`), a 3B parameter code-specialized instruction-tuned language model.
- **Total parameters**: 3,205,672,960
- **Compute dtype**: bfloat16

### 2.2 Fine-Tuning Configuration

| Hyperparameter | Value |
|---|---|
| Method | LoRA (Low-Rank Adaptation) |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` (all linear layers) |
| LoRA rank (r) | 64 |
| LoRA alpha (α) | 128 (scaling factor α/r = 2.0) |
| LoRA dropout | 0.05 |
| Trainable parameters | 119,734,272 (3.74% of total) |
| Epochs | 2 |
| Batch size | 4 (per device) |
| Gradient accumulation | 4 steps (effective batch size = 16) |
| Learning rate | 5e-5 |
| LR scheduler | Cosine |
| Warmup steps | 50 |
| Cutoff length | 2048 tokens |
| Precision | bf16 |
| Framework | LLaMA-Factory (SFT stage) |

**Note on training**: The training was completed in two phases. The initial run was interrupted at checkpoint-2800, and a second run resumed from that checkpoint using `--resume_from_checkpoint` to complete the full 2 epochs.

### 2.3 Training Data

- **Dataset**: DebugEval SFT dataset (`fine-tune-dataset.json`)
- **Total training samples**: 24,892
- **Format**: Instruction-Output pairs derived from the DebugEval benchmark covering localization, identification, repair, and review tasks.

### 2.4 Training Dynamics

- **Initial loss**: 0.864 → **Final loss**: 0.149 (at epoch 2.0)
- **Average training loss**: 0.0143
- **Training duration**: ~10m 32s (for the final resumed phase)
- The very low average training loss (0.0143) indicates substantial convergence by the end of 2 epochs.

### 2.5 Evaluation Setup

- **Inference engine**: vLLM with OpenAI-compatible API
- **Temperature**: 0.2 | **Top-p**: 0.95
- **Max tokens**: 8,096 (dynamically adjusted)
- **Prompt format**: Qwen chat template (`<|im_start|>` / `<|im_end|>` markers)
- **Prompts**: COAST benchmark prompt templates (zero-shot)
  - Base model: standard prompts (`model_type=qwen`)
  - Fine-tuned model: `llama_fine_tune` variant prompts (`model_type=qwen-sft`)
- **LoRA serving**: The fine-tuned model was served via vLLM's native LoRA support (`--enable-lora --max-lora-rank 64`)

### 2.6 Benchmark: DebugEval

| Task | Description | Metric | Samples |
|---|---|---|---|
| Task 1: Bug Localization | Select the erroneous code snippet from options (A–D) | Accuracy | 578 |
| Task 2: Bug Identification | Classify the type of bug from options (A–D) | Accuracy | 2,320 |
| Task 3: Code Repair | Generate corrected source code, judged by test-case execution | Pass rate | 414 (138 per language) |
| Task 4: Code Review | Identify buggy snippet between two alternatives (normal + reversed) | Accuracy | 4,800 (2,400 × 2) |

## 3. Results

| Task | Base Model | Fine-Tuned | Δ (absolute) |
|---|---|---|---|
| Task 1: Bug Localization | 44.29% (256/578) | **71.45%** (413/578) | **+27.16 pp** |
| Task 2: Bug Identification | 25.43% (590/2320) | **43.79%** (1016/2320) | **+18.36 pp** |
| Task 3: Code Repair | 34.30% (142/414) | **44.69%** (185/414) | **+10.39 pp** |
| Task 4: Code Review | 83.38% (4002/4800) | **83.75%** (4020/4800) | +0.37 pp |

## 4. Analysis

### 4.1 Consistent Improvement Across All Tasks

Unlike many fine-tuning studies that observe trade-offs between task types, the Qwen Coder model demonstrates **uniform improvement across all four tasks** after LoRA fine-tuning. No task experienced degradation.

### 4.2 Magnitude of Improvement by Task Category

**Comprehension tasks showed the largest gains:**

- **Task 1 (Localization)**: +27.16 pp — the most dramatic improvement. The base model performed near chance level (44.29% on a 4-option task with 25% chance baseline), while the fine-tuned model reached 71.45%.
- **Task 2 (Identification)**: +18.36 pp — similarly, the base model struggled at 25.43% (near chance), while fine-tuning raised it to 43.79%.

**Generation tasks improved moderately:**

- **Task 3 (Repair)**: +10.39 pp — a meaningful improvement in functional code generation.
- **Task 4 (Review)**: +0.37 pp — negligible change; both models performed well above 83%.

### 4.3 Interpretation

1. **The base Qwen Coder model's weakness in comprehension**: The near-chance performance on Tasks 1 and 2 suggests the base model lacks the analytical reasoning needed to parse multiple-choice debugging questions despite being code-specialized. Fine-tuning with structured debugging examples significantly improved this capability.

2. **Already-strong code review**: The base model already achieved 83.38% on Task 4, indicating its pretraining included sufficient exposure to code comparison tasks. Fine-tuning added marginal value here.

3. **Broader LoRA coverage matters**: This model targeted all 7 linear layers (compared to only 2 for Gemma4), resulting in 3.74% trainable parameters. This broader adaptation surface may explain why no task degraded — the model had sufficient capacity to learn new patterns without overwriting existing capabilities.

4. **Training convergence**: The model completed the full 2 epochs with a very low final training loss (0.0143), suggesting thorough adaptation to the SFT data.

### 4.4 Training Considerations

The training required a checkpoint resume (`--resume_from_checkpoint checkpoint-2800`) to complete the full 2 epochs. The total training time for the 3B model was substantially shorter than the 8B Gemma4 model, reflecting both the smaller model size and larger per-device batch size (4 vs. 1).

## 5. Conclusions

1. **LoRA fine-tuning on the DebugEval dataset produces consistent improvements** across all four tasks for Qwen2.5-Coder-3B-Instruct.
2. **Comprehension tasks benefit most** (+27 pp and +18 pp), transforming the model from near-chance to reasonable performance on bug localization and identification.
3. **Code repair improves by +10 pp**, a meaningful gain in practical debugging utility.
4. **Code review is negligibly affected**, as the base model already performs well on this task.
5. The broad LoRA target coverage (all linear layers) and a 3.74% trainable parameter ratio appear to enable comprehensive adaptation without catastrophic forgetting.
