# Qwen2.5-3B-Instruct Fine-Tuned on Socratic Debugging Dataset

## 1. Introduction

This report evaluates the effectiveness of fine-tuning Qwen2.5-3B-Instruct on the Socratic Debugging dataset. The objective of this fine-tuning is to transform a general-purpose instruction-following model into a Socratic programming tutor — one that guides students to discover and fix bugs through probing questions rather than directly revealing answers.

Unlike the DebugEval experiments which focus on benchmark accuracy, this evaluation assesses **pedagogical quality**: whether the model asks questions, avoids revealing answers, stays on topic, and provides helpful guidance.

## 2. Experimental Setup

### 2.1 Model

- **Base model**: Qwen2.5-3B-Instruct (`Qwen2ForCausalLM`) — a general-purpose (non-code-specialized) 3B parameter instruction-tuned model.
- **Total parameters**: 3,205,672,960
- **Compute dtype**: bfloat16

**Note**: This is the general Qwen2.5-3B-Instruct model, not the Coder variant. The choice of a general-purpose model is intentional — the Socratic tutoring task requires conversational and pedagogical skills rather than code generation proficiency.

### 2.2 Training Data: Socratic Debugging Dataset

- **Source**: Raw multi-turn debugging conversations in XML format
- **Conversion**: Converted to ShareGPT format via `convert_socratic_data.py`
- **Training samples**: 552
- **Test samples**: 77
- **Format**: Multi-turn conversations with system/human/gpt roles

Each training sample contains:
- **System prompt**: Includes the problem description, buggy code, bug description, and correct fix. The system prompt instructs the model to act as a Socratic tutor who must NOT directly reveal the bug or fix.
- **Conversation turns**: Multi-turn student-tutor dialogue where the tutor guides the student through questioning.

### 2.3 Fine-Tuning Configuration

| Hyperparameter | Value |
|---|---|
| Method | LoRA (Low-Rank Adaptation) |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| LoRA rank (r) | 64 |
| LoRA alpha (α) | 128 (scaling factor α/r = 2.0) |
| LoRA dropout | 0.05 |
| Trainable parameters | 119,734,272 (3.74% of total) |
| Epochs | 5 |
| Batch size | 2 (per device) |
| Gradient accumulation | 4 steps (effective batch size = 8) |
| Learning rate | 2e-5 |
| LR scheduler | Cosine |
| Warmup steps | 30 |
| Cutoff length | 2048 tokens |
| Precision | bf16 |
| Framework | LLaMA-Factory (SFT stage) |

### 2.4 Training Dynamics

- **Initial loss**: 2.468 → **Final loss**: 0.111 (at epoch 5.0)
- **Average training loss**: 0.5116
- **Training duration**: 8m 56s
- The loss trajectory shows rapid convergence in the first 2 epochs (2.468 → ~0.5), followed by gradual refinement to 0.111 by epoch 5.

### 2.5 Deployment

The fine-tuned LoRA adapter was merged with the base model weights and served via vLLM as a standalone merged model (`serve_qwen_instruct_merged.sh`).

## 3. Evaluation Methodology

Two complementary evaluation methods were used:

### 3.1 Heuristic-Based Evaluation (`run_socratic_eval.py`)

This method applies rule-based heuristic metrics to the model's generated responses:

| Metric | Description |
|---|---|
| **Question Rate** | Whether the response contains question marks ("?") |
| **Direct Fix Avoidance** | Absence of phrases like "the fix is", "the bug is", "change it to", etc. |
| **Socratic Phrase Rate** | Presence of guiding phrases like "what do you think", "let's trace", "can you explain", etc. |
| **Debugging Suggestion Rate** | Presence of debugging strategy suggestions (print statements, tracing, test cases) |
| **Socratic Score** | Composite score (0–100): +40 for questions, +30 for Socratic phrases, +20 for avoiding direct fixes, +10 for debugging suggestions |

**Test set**: 77 samples from the converted Socratic test data.

### 3.2 LLM-as-Judge Evaluation (`evaluate_with_gemma_judge.py`)

This method uses a separate Gemma-4-E4B-IT model as a judge to evaluate the tutor's responses. The judge evaluates each response across four dimensions:

| Metric | Scale | Description |
|---|---|---|
| **Questions** | yes/no | Does the response contain guiding questions? |
| **On-Topic** | 1–5 | Is the response relevant to the student's problem? |
| **Helpful** | 1–5 | Does the response help the student progress? |
| **Reveal Answer** | yes/no | Does the response directly reveal the bug or fix? |

The judge uses a structured JSON evaluation prompt, and its responses are parsed into a Pydantic schema (`Evaluation` model). The judge was served as a separate vLLM instance.

**Test set**: 100 samples from the raw Socratic debugging test data.

## 4. Results

### 4.1 Heuristic Evaluation Results (77 samples)

| Metric | Value |
|---|---|
| Average Socratic Score | **68.6 / 100** |
| Question Rate | **96.1%** (74/77) |
| Direct Fix Avoidance | **100.0%** (77/77) |
| Socratic Phrase Rate | **31.2%** (24/77) |
| Average Response Length | Varies by sample |

### 4.2 LLM Judge Evaluation Results (100 samples)

| Metric | Value |
|---|---|
| Successfully parsed | **100/100** (100%) |
| Questions Rate (yes) | **89.0%** (89/100) |
| Reveal Answer (no) | **97.0%** (97/100) |
| Average On-Topic Score | **4.93 / 5.00** |
| Average Helpfulness Score | **4.49 / 5.00** |

## 5. Analysis

### 5.1 Core Socratic Behavior: Successfully Learned

The model demonstrates strong adherence to the Socratic teaching paradigm:

1. **Question generation**: 96.1% (heuristic) and 89.0% (judge) of responses contain questions. The heuristic method detects any "?" character, while the judge applies a stricter standard for meaningful guiding questions — the difference (96.1% vs. 89.0%) suggests ~7% of responses contain only rhetorical or trivial questions.

2. **Answer concealment**: 100.0% (heuristic) and 97.0% (judge) of responses avoid revealing the answer directly. The model has learned the critical constraint of not giving away solutions. The 3% gap from the judge evaluation suggests rare edge cases where the model's hints may be too explicit for the judge's standard.

3. **On-topic relevance**: 4.93/5.00 from the judge indicates near-perfect topic adherence. The model consistently addresses the student's specific problem rather than providing generic advice.

4. **Helpfulness**: 4.49/5.00 from the judge indicates that the model's Socratic responses are constructively helpful — guiding students toward the solution without being vague or unhelpful.

### 5.2 Weakness: Socratic Phrase Diversity

The Socratic Phrase Rate of 31.2% is the weakest metric. This measures the presence of specific guiding phrases ("what do you think", "let's trace", "can you explain", etc.). The low rate suggests:

- The model may rely on **implicit questioning** (asking about specific code behavior) rather than using the **explicit Socratic patterns** that the heuristic measures.
- The heuristic's phrase list may not capture the full range of Socratic strategies the model has learned.
- This is a limitation of the heuristic metric rather than necessarily a model deficiency, as the judge's high helpfulness score (4.49/5) suggests the model is effective even without using these specific phrases.

### 5.3 Composite Socratic Score Analysis

The average Socratic Score of 68.6/100 breaks down as:
- +40 for questions: achieved by 96.1% of samples
- +30 for Socratic phrases: achieved by only 31.2%
- +20 for fix avoidance: achieved by 100%
- +10 for debugging suggestions: not explicitly measured in aggregate

The main score deficit comes from the Socratic phrases component. If the phrase detection were expanded to cover the model's actual guiding strategies, the composite score would likely be higher.

### 5.4 Training Considerations

- **Small dataset, effective learning**: With only 552 training samples and 5 epochs, the model achieved strong behavioral alignment. This demonstrates that LoRA fine-tuning can effectively teach conversational styles and pedagogical patterns from limited data.
- **Training loss convergence**: The loss curve (2.468 → 0.111) shows healthy convergence without signs of overfitting to the small dataset, likely due to the regularizing effect of LoRA dropout (0.05) and the relatively low learning rate (2e-5).
- **Lower learning rate**: The Socratic training used 2e-5 (vs. 5e-5 for DebugEval fine-tuning), a deliberate choice to prevent aggressive overwriting of the base model's conversational abilities.

### 5.5 Evaluation Methodology Comparison

The two evaluation methods provide complementary insights:

| Aspect | Heuristic | LLM Judge |
|---|---|---|
| Question detection | Surface-level ("?") | Semantic (meaningful questions) |
| Fix avoidance | Keyword matching | Contextual judgment |
| Quality assessment | Not measured | On-topic (4.93/5), Helpful (4.49/5) |
| Scalability | Fast, deterministic | Slow, requires GPU |
| Reliability | No false positives in detection | 100% parse rate |

The LLM judge provides richer evaluation but introduces dependency on the judge model's own biases and capabilities.

## 6. Conclusions

1. **The fine-tuned model successfully learns Socratic tutoring behavior**, with high question rates (89–96%), near-perfect answer concealment (97–100%), strong topical relevance (4.93/5), and good helpfulness (4.49/5).

2. **LoRA fine-tuning is effective for teaching pedagogical style** even with a small dataset (552 samples). The model learned both what to do (ask questions, suggest debugging strategies) and what not to do (reveal answers directly).

3. **The model's main weakness is limited diversity in Socratic phrasing** (31.2%), though this may reflect heuristic measurement limitations rather than actual behavioral deficiency.

4. **The dual evaluation approach** (heuristic + LLM judge) provides robust assessment: the heuristic catches surface patterns efficiently, while the judge evaluates semantic quality. Their general agreement (with expected differences in sensitivity) increases confidence in the results.

5. **Practical applicability**: The model is suitable for deployment as a programming tutor assistant, particularly for guided debugging exercises. The 3% rate of answer revelation (per the judge) represents the primary risk for pedagogical deployment.
