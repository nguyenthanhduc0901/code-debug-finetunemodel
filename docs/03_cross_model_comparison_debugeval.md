# Cross-Model Comparison: Gemma-4 vs. Qwen Coder on DebugEval

## 1. Introduction

This report presents a comprehensive cross-model comparison of four model configurations evaluated on the DebugEval benchmark:

1. **Gemma-4-E4B-IT Base** (~8B parameters)
2. **Gemma-4-E4B-IT + LoRA** (fine-tuned on DebugEval)
3. **Qwen2.5-Coder-3B-Instruct Base** (~3B parameters)
4. **Qwen2.5-Coder-3B-Instruct + LoRA** (fine-tuned on DebugEval)

The analysis examines how model architecture, model scale, and fine-tuning interact across four debugging tasks.

## 2. Model Specifications

| Property | Gemma-4-E4B-IT | Qwen2.5-Coder-3B-Instruct |
|---|---|---|
| Architecture | Gemma4ForConditionalGeneration | Qwen2ForCausalLM |
| Total parameters | 7.96B | 3.21B |
| Model type | General multimodal (text-only mode) | Code-specialized |
| LoRA targets | `q_proj`, `v_proj` (2 modules) | All 7 linear layers |
| Trainable params | 18.2M (0.23%) | 119.7M (3.74%) |
| LoRA rank / alpha | 64 / 128 | 64 / 128 |
| Training epochs | 1 (completed) | 2 (completed) |
| Final training loss | 0.303 | 0.014 |
| Training data | DebugEval SFT (24,892 samples) | DebugEval SFT (24,892 samples) |
| Effective batch size | 16 | 16 |
| Learning rate | 5e-5 | 5e-5 |

Both models used the same training dataset and the same LLaMA-Factory SFT pipeline. Evaluation used the same COAST prompt templates and vLLM inference engine (temperature=0.2, top_p=0.95).

## 3. Results

### 3.1 Full Results Table

| Task | Gemma4 Base | Gemma4 SFT | Qwen Base | Qwen SFT |
|---|---|---|---|---|
| **Task 1: Bug Localization** | **79.24%** | 73.01% | 44.29% | 71.45% |
| **Task 2: Bug Identification** | **55.43%** | 44.61% | 25.43% | 43.79% |
| **Task 3: Code Repair** | 8.94% | 41.30% | 34.30% | **44.69%** |
| **Task 4: Code Review** | N/A* | **86.65%** | 83.38% | 83.75% |

\* *Gemma4 Base Task 4 evaluation was interrupted during inference (1,593/4,800 prompts) and did not produce a valid result.*

### 3.2 Fine-Tuning Effect (Δ from Base)

| Task | Gemma4 Δ | Qwen Δ |
|---|---|---|
| Task 1 | −6.23 pp | **+27.16 pp** |
| Task 2 | −10.82 pp | **+18.36 pp** |
| Task 3 | **+32.36 pp** | +10.39 pp |
| Task 4 | N/A | +0.37 pp |

## 4. Analysis

### 4.1 Base Model Comparison: Scale vs. Specialization

The base model results reveal a sharp contrast between comprehension and generation tasks:

**Comprehension tasks (Tasks 1 & 2):** Gemma4 Base (8B, general-purpose) substantially outperforms Qwen Coder Base (3B, code-specialized):
- Task 1: 79.24% vs. 44.29% (+34.95 pp advantage for Gemma4)
- Task 2: 55.43% vs. 25.43% (+30.00 pp advantage for Gemma4)

This suggests that for multiple-choice analytical reasoning about bugs, **model scale and general reasoning capability** (Gemma4 at 8B) dominate over code specialization (Qwen Coder at 3B).

**Code repair (Task 3):** The relationship reverses — Qwen Coder Base (34.30%) dramatically outperforms Gemma4 Base (8.94%). The code-specialized pretraining of Qwen Coder provides a clear advantage for **code generation** tasks, even at a smaller scale.

**Code review (Task 4):** Qwen Coder Base achieves 83.38%, demonstrating strong performance. The Gemma4 Base result is unavailable for comparison.

### 4.2 Fine-Tuning Response: Divergent Patterns

The two models respond to identical LoRA fine-tuning in opposite ways:

**Gemma4**: Fine-tuning **degrades comprehension** (−6.23 pp, −10.82 pp) while **massively boosting generation** (+32.36 pp on repair). This is a classic specialization trade-off.

**Qwen Coder**: Fine-tuning **improves all tasks uniformly**, with the largest gains in comprehension (+27.16 pp, +18.36 pp) and moderate improvement in generation (+10.39 pp). No degradation occurs.

### 4.3 Post-Fine-Tuning Convergence

After fine-tuning, the performance gap between the two models narrows dramatically:

| Task | Gap (Base) | Gap (SFT) |
|---|---|---|
| Task 1 | 34.95 pp (Gemma4 leads) | 1.56 pp (Gemma4 leads) |
| Task 2 | 30.00 pp (Gemma4 leads) | 0.82 pp (Gemma4 leads) |
| Task 3 | 25.36 pp (Qwen leads) | 3.39 pp (Qwen leads) |
| Task 4 | N/A | 2.90 pp (Gemma4 leads) |

After fine-tuning, both models converge to **similar performance levels** across all tasks, with differences of less than 4 percentage points. This suggests that for this benchmark and dataset, the SFT data distribution — rather than base model capabilities — becomes the dominant factor.

### 4.4 Factors Explaining the Divergent Fine-Tuning Response

| Factor | Gemma4 | Qwen Coder |
|---|---|---|
| LoRA coverage | 2 modules (q_proj, v_proj) | 7 modules (all linear) |
| Trainable % | 0.23% | 3.74% |
| Completed epochs | 1 | 2 |
| Base comprehension | Already strong (79%, 55%) | Weak (44%, 25%) |
| Base generation | Very weak (8.94%) | Moderate (34.30%) |

The divergent responses can be attributed to:

1. **Room for improvement**: Qwen Coder Base had near-chance comprehension performance, providing substantial room for improvement. Gemma4 Base was already strong on comprehension, leaving less room for gain and more risk of regression.

2. **LoRA adaptation capacity**: Qwen Coder's broader LoRA coverage (3.74% trainable) provided more capacity for comprehensive adaptation. Gemma4's narrow coverage (0.23%) may have forced a redistribution of capabilities rather than an addition.

3. **Training completeness**: Qwen Coder completed 2 full epochs (final loss 0.014), while Gemma4 completed only 1 epoch (final loss 0.303). More training may have allowed Qwen Coder to learn a more balanced representation.

### 4.5 Best Configuration by Task

| Task | Best Model | Accuracy |
|---|---|---|
| Task 1: Bug Localization | Gemma4 Base | 79.24% |
| Task 2: Bug Identification | Gemma4 Base | 55.43% |
| Task 3: Code Repair | Qwen Coder SFT | 44.69% |
| Task 4: Code Review | Gemma4 SFT | 86.65% |

## 5. Conclusions

1. **Model scale vs. specialization**: For comprehension tasks, the larger general-purpose model (Gemma4 8B) outperforms the smaller code-specialized model (Qwen 3B) in the base configuration. For code generation, code specialization outweighs scale.

2. **Fine-tuning responses differ by architecture**: Gemma4 exhibits a comprehension-generation trade-off, while Qwen Coder achieves uniform improvement. This is likely due to the broader LoRA coverage and lower baseline comprehension of Qwen Coder.

3. **Post-SFT convergence**: After fine-tuning, both models reach similar performance levels (within ~3 pp), suggesting that the fine-tuning data becomes the dominant factor in performance regardless of the base model.

4. **Practical recommendation**: For a system requiring balanced performance across all debugging tasks, **Qwen2.5-Coder-3B-Instruct + LoRA** offers the best overall profile — competitive accuracy on all tasks with no degradation, at a fraction of the compute cost of the 8B Gemma4 model. If a single task must be maximized, Gemma4 Base remains strongest for comprehension, while Qwen Coder SFT leads on code repair.
