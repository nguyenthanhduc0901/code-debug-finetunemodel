# Qwen2.5-3B-Instruct Fine-Tuned on Socratic Debugging Dataset

## 1. Introduction

This report evaluates the effectiveness of fine-tuning Qwen2.5-3B-Instruct on the Socratic Debugging dataset. The objective of this fine-tuning is to transform a general-purpose instruction-following model into a Socratic programming tutor — one that guides students to discover and fix bugs through probing questions rather than directly revealing answers.

Unlike the DebugEval experiments which focus on benchmark accuracy, this evaluation assesses **pedagogical quality**: whether the model asks questions, avoids revealing answers, stays on topic, and provides helpful guidance. 

This document compares the performance of the **Base Model (Qwen2.5-3B-Instruct)** against the **Fine-Tuned Model (Socratic SFT)** using both heuristic metrics and a local Gemma-4 LLM-as-Judge evaluation.

---

## 2. Experimental Setup

### 2.1 Model
- **Base model**: Qwen2.5-3B-Instruct (`Qwen2ForCausalLM`) — a general-purpose 3B parameter instruction-tuned model.
- **Fine-Tuned (SFT) model**: Qwen2.5-3B-Instruct with a LoRA adapter trained for 5 epochs on the Socratic dataset, merged back into the base weights.
- **Total parameters**: 3,205,672,960
- **Compute dtype**: bfloat16

### 2.2 Training Data: Socratic Debugging Dataset
- **Source**: Raw multi-turn debugging conversations in XML format.
- **Conversion**: Converted to ShareGPT format via `convert_socratic_data.py`.
- **Training samples**: 552
- **Test samples**: 77
- **Format**: Multi-turn conversations with system/human/gpt roles.

### 2.3 Fine-Tuning Configuration
| Hyperparameter | Value |
|---|---|
| Method | LoRA (Low-Rank Adaptation) |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| LoRA rank (r) | 64 |
| LoRA alpha (α) | 128 (scaling factor α/r = 2.0) |
| Trainable parameters | 119,734,272 (3.74% of total) |
| Epochs | 5 |
| Batch size | 2 (per device) |
| Gradient accumulation | 4 steps (effective batch size = 8) |
| Learning rate | 2e-5 |
| Precision | bf16 |

### 2.4 Deployment
Both models were served using vLLM under eager mode (`--enforce-eager`) on RTX 5090 to fit concurrently on the GPU:
- **Tutor Port (`8001`)**: Qwen2.5-3B-Instruct (Base or Merged)
- **Judge Port (`8002`)**: Gemma-4-E4B-it (acting as the Socratic Judge)

---

## 3. Evaluation Methodology

Two complementary evaluation methods were used:

### 3.1 Heuristic-Based Evaluation (`run_socratic_eval.py`)
This method applies rule-based keyword matching to the generated tutor responses:
- **Question Rate**: Contains question marks ("?").
- **Direct Fix Avoidance**: Absence of answer-revealing phrases ("the fix is", "change it to", etc.).
- **Socratic Phrase Rate**: Presence of guiding phrases ("what do you think", "let's trace", etc.).
- **Socratic Score**: Composite score (0–100): +40 for questions, +30 for Socratic phrases, +20 for avoiding direct fixes, +10 for debugging suggestions.

### 3.2 LLM-as-Judge Evaluation (`evaluate_with_gemma_judge.py`)
This method uses `Gemma-4-E4B-it` as an independent judge to score the tutor response across four dimensions:
- **Questions** (yes/no): Does the tutor ask guiding questions?
- **On-Topic** (1–5): Is the response relevant to the student's problem?
- **Helpful** (1–5): Does the response help the student discover the solution?
- **Reveal Answer** (yes/no): Does the tutor directly reveal the bug/fix?

*Composite Score formula:* `(questions_val + on_topic/5 + helpful/5 + (1 - reveal_answer_val)) / 4` (ranging from 0.0 to 1.0).

---

## 4. Results & Comparison

### 4.1 Heuristic Evaluation Results (77 samples)

| Metric | Qwen Base | Qwen Socratic 7-Mod SFT | Qwen Socratic 2-Mod SFT |
|---|:---:|:---:|:---:|
| **Average Socratic Score** | 73.2 / 100 | 68.6 / 100 | 70.4 / 100 |
| **Question Rate** | 93.5% (72/77) | 96.1% (74/77) | 93.5% (72/77) |
| **Direct Fix Avoidance** | 100.0% (77/77) | 100.0% (77/77) | 100.0% (77/77) |
| **Socratic Phrase Rate** | 44.2% (34/77) | 31.2% (24/77) | 40.3% (31/77) |
| **Suggests Debugging** | 26.0% (20/77) | 26.0% (20/77) | 9.1% (7/77) |
| **Average Response Length** | 55 words | 36 words | 14 words |

### 4.2 LLM Judge Evaluation Results

| Metric | Qwen Base | Qwen Socratic 7-Mod SFT | Qwen Socratic 2-Mod SFT |
|---|:---:|:---:|:---:|
| **Successfully parsed** | 100% (100/100) | 100% (100/100) | 100% (100/100) |
| **Avg Socratic Score (Judge)** | 81.6% (0.816) | 93.6% (0.936) | 96.7% (0.967) |
| **Questions Rate (Yes)** | 67.0% (67/100) | 89.0% (89/100) | 94.5% |
| **Reveal Answer (Yes - Leak)** | 22.0% (22/100) | 3.0% (3/100) | 0.0% |
| **Average On-Topic Score** | 4.82 / 5.00 | 4.93 / 5.00 | 4.84 / 5.00 |
| **Average Helpfulness Score** | 4.15 / 5.00 | 4.49 / 5.00 | 4.78 / 5.00 |


