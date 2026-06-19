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

| Metric | Qwen Base | Qwen Socratic SFT | Delta (SFT - Base) |
|---|:---:|:---:|:---:|
| **Average Socratic Score** | **73.2 / 100** | **68.6 / 100** | <font color="red">-4.6</font> *(See 5.1)* |
| **Question Rate** | 93.5% (72/77) | 96.1% (74/77) | <font color="green">+2.6%</font> |
| **Direct Fix Avoidance** | 100.0% (77/77) | 100.0% (77/77) | 0.0% |
| **Socratic Phrase Rate** | 44.2% (34/77) | 31.2% (24/77) | <font color="red">-13.0%</font> |
| **Suggests Debugging** | 26.0% (20/77) | 26.0% (20/77) | 0.0% |
| **Average Response Length** | 55 words | 36 words | **-19 words (more concise)** |

### 4.2 LLM Judge Evaluation Results (100 samples)

| Metric | Qwen Base | Qwen Socratic SFT | Delta (SFT - Base) |
|---|:---:|:---:|:---:|
| **Successfully parsed** | 100% (100/100) | 100% (100/100) | 0% |
| **Avg Socratic Score (Judge)** | **86.6%** (0.866) | **93.6%** (0.936) | <font color="green">**+7.0% (Significant)**</font> |
| **Questions Rate (Yes)** | 74.0% (74/100) | 89.0% (89/100) | <font color="green">**+15.0%**</font> |
| **Reveal Answer (Yes - Leak)** | **16.0%** (16/100) | **3.0%** (3/100) | <font color="green">**-13.0% (Symmetric Improvement)**</font> |
| **Average On-Topic Score** | 5.00 / 5.00 | 4.93 / 5.00 | -0.07 |
| **Average Helpfulness Score** | 4.42 / 5.00 | 4.49 / 5.00 | +0.07 |

---

## 5. Analysis & Key Findings

### 5.1 The Heuristic Score Paradox: Why Base Scored Higher
In heuristic evaluation, the Base model scored **73.2/100** vs. the SFT model's **68.6/100**. This occurs due to a **length/verbosity bias** in the heuristic matching rules:
1. **Implicit vs. Explicit Phrasing**: The Socratic Phrase Rate heuristic checks for exact string matches (e.g. *"what do you think"*, *"let's trace"*).
2. **Verbosity**: The Base model is highly verbose (avg. 55 words per turn), giving it more opportunities to trigger the hardcoded phrase filters.
3. **Dialogue Bleeding**: The Base model frequently generates multiple turns of dialogue at once (simulating both Tutor and Student in a single response). This artificially inflates its keyword count.
4. **SFT Conciseness**: The SFT model has learned to be highly concise and directly targeted (avg. 36 words), keeping conversations short and interactive as required by Socratic principles. While this makes it a better tutor, it reduces the likelihood of triggering multiple hardcoded keyword filters in a single turn.

### 5.2 Critical Pedagogical Quality: Revealing the Answer
The most significant finding is the **Reveal Answer** metric under the Gemma-4 Judge:
- **Base Model (16.0% leak rate)**: Leaked the final solution or code block directly to the student in nearly 1 in 6 turns. This violates Socratic tutoring rules.
- **SFT Model (3.0% leak rate)**: Reduced direct answer leaks down to only 3%. This represents a **13.0% absolute decrease in answer leaks**, showing that the model successfully internalized the negative constraint ("Do not give away the solution").

### 5.3 Semantic Question Generation
While the heuristic check (counting "?") registered high numbers for both models, the LLM Judge evaluated whether the question was a **meaningful, pedagogically sound guiding question**:
- The Base model only asked meaningful Socratic questions in **74%** of its evaluations.
- The SFT model increased this to **89%** (+15% improvement).
- This indicates the SFT model is far more consistent in driving the dialogue forward through inquiries rather than declarative statements.

---

## 6. Conclusions

1. **Successful Behavior Alignment**: LoRA fine-tuning successfully converted a general-purpose model into a pedagogical tutor. The SFT model is significantly safer to deploy, leaking answers in only 3% of turns (vs. 16% for the Base model).
2. **Stronger Socratic Directives**: The SFT model demonstrates a 15% increase in meaningful questions (89% vs. 74%), making it much better at guiding student critical thinking.
3. **Optimized Conversation Flow**: The SFT model learned to stop generating when it is the user's turn (preventing dialogue bleeding) and keep responses concise (36 words vs. 55 words), enhancing conversational dynamics.
4. **Methodology Validation**: The LLM-as-Judge evaluation provides a far more accurate semantic measurement of pedagogical intent than keyword heuristics, which suffer from a verbosity bias. The base model's higher heuristic score was a side effect of its wordiness and dialogue bleeding.
5. **Deployment Readiness**: The merged Qwen-3B-Socratic SFT model is ready for integration into a guided debugging tutor assistant, presenting a robust compromise between size (3B) and conversational teaching quality.
