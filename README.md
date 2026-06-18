# Finetune LLM — DebugEval & Socratic Tutor

Dự án fine-tune và đánh giá các mô hình ngôn ngữ lớn trên hai tác vụ:

1. **DebugEval (COAST)** — Đánh giá khả năng phát hiện, định vị, sửa lỗi code
2. **Socratic Tutor** — Đào tạo mô hình hướng dẫn debug theo phương pháp Socrates

---

## Cấu trúc dự án

```
.
├── data/                   Dữ liệu tập trung
│   ├── debugeval/          Tập dữ liệu DebugEval (COAST)
│   │   ├── raw/            Dữ liệu thô (.jsonl)
│   │   ├── sft/            Dữ liệu SFT (ShareGPT format)
│   │   └── atcoder_cases/  Test cases (Input/Output)
│   └── socratic/           Tập dữ liệu Socratic Tutor
│       ├── raw/            Dữ liệu thô (.json)
│       └── sft/            Dữ liệu SFT (ShareGPT format)
│
├── models/                 Mô hình (Base + Finetuned)
│   ├── base/
│   │   ├── Qwen2.5-Coder-3B-Instruct/   (DebugEval)
│   │   ├── Qwen2.5-3B-Instruct/         (Socratic)
│   │   └── gemma-4-E4B-it → HF cache    (DebugEval + Judge)
│   └── finetuned/
│       ├── debugeval/      Kết quả finetune DebugEval
│       │   ├── gemma4-sft/
│       │   ├── qwen-coder-3b-sft/
│       │   └── qwen-coder-3b-merged/
│       └── socratic/       Kết quả finetune Socratic
│           ├── qwen-3b-sft/
│           └── qwen-3b-merged/
│
├── configs/                Cấu hình huấn luyện (SFT)
│   ├── debugeval/          Qwen Coder & Gemma4 training configs
│   └── socratic/           Qwen Instruct training configs
│
├── scripts/                Scripts vận hành & đánh giá
│   ├── serve/              vLLM serve scripts
│   ├── eval/               Evaluation scripts
│   └── utils/              Tiện ích (convert data, download model)
│
├── logs/                   Logs tập trung
│   ├── training/           Logs quá trình train
│   ├── serving/            Logs vLLM serve
│   └── evaluation/         Kết quả evaluation & reports
│
├── COAST/                  Benchmark COAST gốc (read-only)
├── tools/                  Framework huấn luyện (LLaMA-Factory)
└── scratch/                File nháp / thử nghiệm
```

---

## Mô hình

| Mô hình | Mục đích | Tập dữ liệu |
|:---|:---|:---|
| **Qwen2.5-Coder-3B-Instruct** | Fine-tune DebugEval (code debugging) | `data/debugeval/` |
| **Qwen2.5-3B-Instruct** | Fine-tune Socratic Tutor (sư phạm) | `data/socratic/` |
| **Gemma-4-E4B-it** | Fine-tune DebugEval + Judge cho Socratic | `data/debugeval/` |

---

## Chuẩn bị (Setup)

Vì các tệp tin mô hình nặng, checkpoints, và dataset lớn không được push lên Git (theo cấu hình `.gitignore`), bạn cần chạy các script tiện ích sau để chuẩn bị môi trường trước khi chạy dự án:

### 1. Thiết lập Token Hugging Face
Vì các mô hình fine-tuned được lưu ở chế độ Private trên Hugging Face của bạn, hãy xuất token trước khi tải:
```bash
export HF_TOKEN="your_huggingface_write_token"
```

### 2. Tải mô hình Base
Tải cả 3 mô hình base (`Qwen2.5-Coder-3B-Instruct`, `Qwen2.5-3B-Instruct`, `gemma-4-E4B-it`):
```bash
bash scripts/utils/download_base_models.sh
```

### 3. Tải các bản Fine-tuned LoRA Adapters
Tải các trọng số LoRA đã huấn luyện từ Hugging Face về các thư mục tương ứng:
```bash
bash scripts/utils/download_lora_adapters.sh
```

### 4. Chuẩn bị Dữ liệu DebugEval
Giải nén hoặc tải tập dữ liệu DebugEval (ở local sẽ tự động giải nén từ tệp sao lưu `.tar.gz` nếu có):
```bash
bash scripts/utils/download_debugeval_data.sh
```

---

## Sử dụng


### Huấn luyện
```bash
# DebugEval - Qwen Coder
bash configs/debugeval/qwen_coder_sft.sh

# DebugEval - Gemma4
bash configs/debugeval/gemma4_sft.sh

# Socratic - Qwen Instruct
bash configs/socratic/qwen_instruct_sft.sh
```

### Serve mô hình
```bash
# Serve Qwen Coder (merged)
bash scripts/serve/serve_qwen_coder_merged.sh

# Serve Socratic Tutor (merged)
bash scripts/serve/serve_qwen_instruct_merged.sh

# Serve Gemma4 Judge
bash scripts/serve/serve_gemma4_judge.sh
```

### Đánh giá
```bash
# DebugEval evaluation
python scripts/eval/run_debugeval.py --model-type qwen-sft --port 8888

# Socratic evaluation (with Gemma Judge)
python scripts/eval/evaluate_with_gemma_judge.py
```
