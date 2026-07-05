# Socratic Debugging Benchmark: Multi-Model Evaluation

## 1. Introduction

This report presents a comparative evaluation between **Gemma-4 Base (`gemma-4-e4b-it`)**, **DeepSeek v4 Flash**, **Gemini 3.1 Flash Lite**, and the **Fine-Tuned Socratic Tutor (`qwen2.5-finetuned`)** using the automated evaluation framework of the [socratic-debugging-benchmark](file:///workspace/finetune_gemma/socratic-debugging-benchmark/) repository. 

While previous evaluations (Document 4) focused on heuristic rules and LLM-as-a-Judge ratings, this benchmark evaluates models using standard lexical and semantic similarity metrics against a multi-reference gold-standard corpus of Socratic debugging dialogues.

---

## 2. Experimental Setup

### 2.1 Dataset
* **Source**: `socratic_debugging_benchmark/v3/testset`
* **Size**: 92 conversational threads (multi-turn student-tutor debugging dialogues).
* **Format**: XML dialogues parsed into turn-by-turn prompts containing student code state, buggy descriptions, and target Socratic tutor references.

### 2.2 Models under Evaluation
1. **Gemma-4 Base (`gemma-4-e4b-it`)**
2. **DeepSeek v4 Flash**
3. **Gemini 3.1 Flash Lite**
4. **Qwen SFT (`qwen2.5-finetuned`)**

### 2.3 Evaluation Parameters
* **Generation Mode**: `single` (the model generates a single tutor response turn).
* **Evaluation Method**: `overall` (evaluates performance against the closest matching human reference).
* **Inference Endpoint**: Updated `inference/gpt_inference.py` to use the modern `OpenAI` python wrapper calling local vLLM ports.
* **BERTScore Encoder**: Switched to `roberta-large` (from `deberta-xlarge-mnli`) inside `metrics/multi_reference_metrics.py` to bypass Hugging Face tokenizer token-overflow configuration issues.

---

## 3. Metric Definitions & Configuration Selection

The benchmark utilizes five metrics to gauge how closely the generated tutor response matches gold-standard human Socratic responses:
1. **BLEU-4**: Measures exact 4-gram overlaps between model output and human references.
2. **ROUGE-1 / ROUGE-2**: Computes unigram and bigram overlap (recall/precision harmonic mean).
3. **ROUGE-L**: Evaluates the Longest Common Subsequence, measuring structural similarity.
4. **BERTScore F1**: Leverages contextual embeddings to measure semantic similarity, capturing synonymous guiding phrases even when lexical overlap is low.

### The "Overall" vs. "Thoroughness" Scoring Decision
* **Thoroughness Method**: Employs a bipartite matching algorithm (NetworkX) to match a list of generated suggestions against a list of reference suggestions. When running in `single` response mode, the model generates exactly 1 suggestion, whereas the reference has 5–6 Socratic options. This causes an artificial cap on Recall (~20%), making thoroughness scores unfairly low.
* **Overall Method**: Evaluates the model’s single generated suggestion against the most similar reference suggestion. This is the optimal configuration for a fair turn-by-turn evaluation in `single` generation mode.

---

## 4. Evaluation Results

The evaluation script `run_socratic_benchmark_metrics.py` was executed across all 92 test cases. The resulting average scores across all test samples are summarized below:

| Metric | Gemma-4 Base | DeepSeek v4 Flash | Gemini 3.1 Flash Lite | Qwen SFT (`qwen2.5-finetuned`) |
| :--- | :---: | :---: | :---: | :---: |
| **BLEU-4** | 0.038150 | **0.044580** | 0.043950 | 0.044044 |
| **ROUGE-1** | 0.245120 | 0.267760 | 0.263850 | **0.283333** |
| **ROUGE-2** | 0.081520 | 0.086480 | 0.074260 | **0.103338** |
| **ROUGE-L** | 0.171200 | 0.190890 | 0.179490 | **0.237670** |
| **BERTScore F1** | 0.835120 | 0.871840 | 0.870670 | **0.886699** |


