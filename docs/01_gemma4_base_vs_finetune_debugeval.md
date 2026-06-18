# Gemma-4-E4B-IT: Base vs. LoRA Fine-Tuned on DebugEval

## 1. Introduction

This report presents an empirical comparison between the base Gemma-4-E4B-IT model and its LoRA-fine-tuned variant on the DebugEval benchmark suite. The objective is to evaluate whether parameter-efficient fine-tuning (PEFT) via LoRA on a code-debugging dataset improves the model's ability to perform four core debugging tasks: bug localization, bug type identification, code repair, and code review.

## 2. Experimental Setup

### 2.1 Model

- **Base model**: Gemma-4-E4B-IT (google/gemma-4-E4B-it), a ~8B parameter multimodal model (text-only mode via `--language-model-only`), with architecture `Gemma4ForConditionalGeneration`.
- **Total parameters**: 7,959,254,304
- **Compute dtype**: bfloat16

### 2.2 Fine-Tuning Configuration

| Hyperparameter | Value |
|---|---|
| Method | LoRA (Low-Rank Adaptation) |
| Target modules | `q_proj`, `v_proj` |
| LoRA rank (r) | 64 |
| LoRA alpha (α) | 128 (scaling factor α/r = 2.0) |
| LoRA dropout | 0.05 |
| Trainable parameters | 18,153,472 (0.23% of total) |
| Epochs | 2 (configured); 1 completed |
| Batch size | 1 (per device) |
| Gradient accumulation | 16 steps (effective batch size = 16) |
| Learning rate | 5e-5 |
| LR scheduler | Cosine |
| Warmup steps | 50 |
| Cutoff length | 2048 tokens |
| Precision | bf16 |
| Thinking mode | Disabled (`--no_enable_thinking`) |
| Framework | LLaMA-Factory (SFT stage) |

### 2.3 Training Data

- **Dataset**: DebugEval SFT dataset (`fine-tune-dataset.json`)
- **Total training samples**: 24,892
- **Format**: Instruction-Output pairs covering localization, identification, repair, and review tasks derived from the DebugEval benchmark.

### 2.4 Training Dynamics

- **Initial loss**: 1.845 → **Final loss**: 0.239 (at epoch ~1.0)
- **Average training loss**: 0.303
- **Training duration**: 2h 31m 48s (~9,108 seconds)
- **Note**: Despite configuring 2 epochs, only 1 epoch was completed (1,556 update steps).

### 2.5 Evaluation Setup

- **Inference engine**: vLLM with OpenAI-compatible API
- **Temperature**: 0.2 | **Top-p**: 0.95
- **Max tokens**: 8,096 (dynamically adjusted)
- **Prompt format**: Gemma chat template with `<bos><|turn>` markers. The SFT variant uses an additional `<|channel>thought` marker.
- **Prompts**: COAST benchmark prompt templates (zero-shot)
  - Base model: standard prompts
  - Fine-tuned model: `llama_fine_tune` variant prompts (slightly more constrained format)

### 2.6 Benchmark: DebugEval

DebugEval consists of four tasks:

| Task | Description | Metric | Samples |
|---|---|---|---|
| Task 1: Bug Localization | Select the erroneous code snippet from options (A–D) | Accuracy | 578 |
| Task 2: Bug Identification | Classify the type of bug from options (A–D) | Accuracy | 2,320 |
| Task 3: Code Repair | Generate corrected source code, judged by test-case execution | Pass rate | 414 (138 per language: Python, C++, Java) |
| Task 4: Code Review | Identify buggy snippet between two alternatives (normal + reversed) | Accuracy | 4,800 (2,400 × 2 directions) |

Task 3 uses the OJ evaluation framework (`judgeLib`) which compiles and runs the repaired code against AtCoder test cases with a 10-second time limit and 1024MB memory limit.

## 3. Results

| Task | Base Model | Fine-Tuned | Δ (absolute) |
|---|---|---|---|
| Task 1: Bug Localization | **79.24%** (458/578) | 73.01% (422/578) | −6.23 pp |
| Task 2: Bug Identification | **55.43%** (1286/2320) | 44.61% (1035/2320) | −10.82 pp |
| Task 3: Code Repair | 8.94% (37/414) | **41.30%** (171/414) | +32.36 pp |
| Task 4: Code Review | Incomplete* | **86.65%** (4159/4800) | N/A |

\* *The base model evaluation for Task 4 was interrupted during inference at 1,593/4,800 prompts and did not produce a final accuracy score. No partial evaluation results are available for this task.*

## 4. Analysis

### 4.1 Trade-Off Pattern: Understanding vs. Generation

The results reveal a clear divergence between **comprehension tasks** (Tasks 1, 2) and **generation tasks** (Tasks 3, 4):

- **Comprehension tasks degraded**: The fine-tuned model lost 6.23 and 10.82 percentage points on localization and identification respectively. These tasks require selecting from predefined options — a capability where the base model's general reasoning and broad pretraining knowledge provided an advantage.
- **Code repair dramatically improved**: The most significant result is Task 3, where fine-tuning increased pass rate from 8.94% to 41.30% — a **4.6× improvement**. This indicates that LoRA adaptation successfully taught the model to generate syntactically and semantically correct code repairs that pass automated test cases.
- **Code review performance**: The fine-tuned model achieved 86.65% on Task 4. Without a complete base model result for comparison, a direct conclusion cannot be drawn.

### 4.2 Possible Causes of Comprehension Degradation

1. **Prompt template mismatch**: The SFT variant uses different prompt templates (`llama_fine_tune`) with stricter formatting constraints. This may interfere with the model's natural reasoning patterns for multiple-choice tasks.
2. **Catastrophic forgetting**: LoRA fine-tuning on Task 3-heavy data (code repair examples) may have biased the model toward code generation behaviors at the expense of analytical selection tasks.
3. **LoRA target limitation**: Only `q_proj` and `v_proj` were targeted (2 of 7 possible linear layers), limiting the adaptation surface. This narrow targeting may be sufficient for learning generation patterns but insufficient for preserving comprehension capabilities.

### 4.3 Training Observations

- The model trained for only 1 epoch despite a 2-epoch configuration, possibly due to early stopping or an interruption. Whether completing the second epoch would have changed the comprehension-generation trade-off remains unknown.
- The training loss curve shows rapid convergence (1.845 → ~0.3 in 1 epoch), suggesting the model adapted quickly to the SFT data distribution.

## 5. Conclusions

1. **LoRA fine-tuning significantly improves code repair capability** on Gemma-4-E4B-IT, achieving a 4.6× improvement on Task 3.
2. **Comprehension tasks (localization, identification) degrade** after fine-tuning, with reductions of 6–11 percentage points.
3. The results suggest a **specialization trade-off**: fine-tuning for code generation appears to come at the cost of multiple-choice analytical reasoning.
4. The incomplete Task 4 base evaluation limits the ability to draw full conclusions about code review performance.
5. Future work should explore whether targeting additional LoRA modules or adjusting the training data distribution could mitigate comprehension degradation while preserving repair gains.
