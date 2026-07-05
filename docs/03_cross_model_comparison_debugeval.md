# Cross-Model Comparison: Gemma-4 vs. Qwen Coder on DebugEval

## 1. Introduction

This report presents a comprehensive cross-model comparison of five model configurations evaluated on the DebugEval benchmark:

1. **Gemma-4-E4B-IT Base** (~8B parameters)
2. **Gemma-4-E4B-IT SFT (2-Module LoRA)** (fine-tuned on DebugEval)
3. **Qwen2.5-Coder-3B-Instruct Base** (~3B parameters)
4. **Qwen2.5-Coder-3B-Instruct SFT (7-Module LoRA)** (fine-tuned on DebugEval)
5. **Qwen2.5-Coder-3B-Instruct SFT (2-Module LoRA)** (fine-tuned on DebugEval)

The analysis examines how model architecture, model scale, target LoRA modules, and SFT hyperparameters interact across four debugging tasks.

## 2. Model Specifications

| Property | Gemma-4-E4B-IT SFT | Qwen2.5-Coder-3B 7-Mod SFT | Qwen2.5-Coder-3B 2-Mod SFT |
|---|---|---|---|
| Architecture | Gemma4ForConditionalGeneration | Qwen2ForCausalLM | Qwen2ForCausalLM |
| Total parameters | 7.96B | 3.21B | 3.21B |
| Model type | General multimodal | Code-specialized | Code-specialized |
| LoRA targets | `q_proj`, `v_proj` (2 modules) | All 7 linear layers | `q_proj`, `v_proj` (2 modules) |
| Trainable params | 18.2M (0.23% of total) | 119.7M (3.74% of total) | 3.68M (0.11% of total) |
| LoRA rank / alpha | 64 / 128 | 64 / 128 | 16 / 32 |
| Training epochs | 1 | 2 | 1 |
| Final training loss | 0.303 | 0.014 | 0.302 |
| Training data | DebugEval SFT (24,892 samples) | DebugEval SFT (24,892 samples) | DebugEval SFT (24,892 samples) |
| Effective batch size | 16 | 16 | 16 |
| Learning rate | 5e-5 | 5e-5 | 1.5e-5 |

---

## 3. Results

### 3.1 Full Results Table

| Task | Gemma4 Base | Gemma4 SFT (2-Mod) | Qwen Base | Qwen SFT (7-Mod) | Qwen SFT (2-Mod) |
|---|---|---|---|---|---|
| **Task 1: Bug Localization** | **79.24%** | 73.01% | 44.29% | 71.45% | 43.94% |
| **Task 2: Bug Identification** | **55.43%** | 44.61% | 25.43% | 43.79% | 39.40% |
| **Task 3: Code Repair** | 8.94% | 41.30% | 34.30% | **44.69%** | 43.48% |
| **Task 4: Code Review** | 81.75% | **86.65%** | 83.38% | 83.75% | 84.46% |

### 3.2 Fine-Tuning Effect (Δ from Base)

| Task | Gemma4 (2-Mod) Δ | Qwen (7-Mod) Δ | Qwen (2-Mod) Δ |
|---|---|---|---|
| **Task 1** | −6.23 pp | **+27.16 pp** | −0.35 pp |
| **Task 2** | −10.82 pp | **+18.36 pp** | +13.97 pp |
| **Task 3** | **+32.36 pp** | +10.39 pp | +9.18 pp |
| **Task 4** | +4.90 pp | +0.37 pp | +1.08 pp |


