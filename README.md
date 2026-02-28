![image](https://github.com/NEUIR/COAST/blob/main/Figure/title.png) 

<p align="center">
    <a href="https://arxiv.org/pdf/2408.05006">📜 Paper</a> •
    <a href="https://huggingface.co/datasets/yangweiqing/DebugEval">🤗 Data </a> •
    <a href="https://huggingface.co/yangweiqing">🤖 Model </a>
</p>

## 1. Introduction
This paper presents a benchmark, DebugEval, which is used to evaluate the code debugging ability of LLMs (Large Language Models) and proposals a framework for synthesizing training data using multiple agents, COAST.

### 1.1 DEBUGEVAL
#### DebugEval designs four task scenarios: BUG Localization, BUG Identification, Code Repair, and Code Recognition to comprehensively evaluate the code debugging capability of LLMs.

![image](https://github.com/NEUIR/COAST/blob/main/Figure/benchmark_00.png).
### 1.2 COAST
#### COAST is a framework for making use of multiple agents working together to synthesize training data to improve code debugging capability of LLMs.

![image](https://github.com/NEUIR/COAST/blob/main/Figure/COAST_00.png).
## 2. Installation

```bash
git clone https://github.com/nguyenthanhduc0901/code-debug-finetunemodel
cd code-debug-finetunemodel
```

### Step 1 — Install dependencies
```bash
bash setup_env.sh
```
> Automatically detects your CUDA version and installs the correct PyTorch build,
> then installs all remaining packages (transformers, deepspeed, peft, etc.)

### Step 2 — Download model & dataset
```bash
# Download both model weights (~13.5 GB) and DebugEval dataset
bash download_assets.sh

# Or separately:
bash download_assets.sh --model-only   # only DeepSeek-Coder-6.7B-Instruct
bash download_assets.sh --data-only    # only DebugEval dataset
```

| Asset | Source | Local path |
|-------|--------|-----------|
| DeepSeek-Coder-6.7B-Instruct | [HuggingFace](https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-instruct) | `models/deepseek-coder-6.7b-instruct/` |
| DebugEval dataset | [HuggingFace](https://huggingface.co/datasets/yangweiqing/DebugEval) | `Data/` |

## 3. Inference and Evaluation
Download the dataset we provide.

```
cd src
```
Please refer to `src/README.md` for more details.
## 4. Fine-Tuning
We use DeepSeek-Coder-6.7B-Ins and Llama3-8B-Ins as the base model, and train the models with COAST framework.

### 4.1 For DeepSeek-Coder-6.7B-Ins
```
cd neural_compiler
```
Please refer to `neural_compiler/README.md` for more details.
### 4.2 For Llama3-8B-Ins
```
cd LLaMA-Factory
```
Please refer to `LLaMA-Factory/README.md` for more details.

We provide the trained NeuDebugger models.

## 5. Result

![image](https://github.com/NEUIR/DebugEval/blob/main/Figure/performance_00.png)
## 6. Citation
Please cite the paper and star the repo if you use DebugEval and find it helpful.

Feel free to contact 2301983@stu.neu.edu.cn or open an issue if you have any questions.
```
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
# code-debug-finetunemodel
