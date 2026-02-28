"""
Comparison test: Base model vs Fine-tuned (LoRA) model
Tests all 4 DebugEval task types side by side.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL  = "/home/ubuntu/COAST/models/deepseek-coder-6.7b-instruct"
ADAPTER_DIR = "/home/ubuntu/COAST/output/deepseek-coder-6.7b-finetuned"

# ── TEST CASES ────────────────────────────────────────────────────────────────
tests = [
    {
        "task": "BUG Localization",
        "correct_answer": "(B)",
        "instruction": """Given a programming task, its incorrect solution and the options. \
The code snippets in the options are all from the Incorrect solution. \
Only one option belongs to the error code snippet(s). Please select the option for error code snippet(s).

Note:
- This task requires you to choose just one option.
- In your response, the final answer should be in the format: <Answer>(Option)</Answer>.

Task:
Write a function to return the sum of two numbers.

Incorrect_Solution:
def add(a, b):
    return a - b

Options:
(A) def add(a, b):
(B) return a - b
(C) return a + b
(D) return a * b
""",
    },
    {
        "task": "BUG Identification",
        "correct_answer": "(C)",
        "instruction": """Given the following buggy code, identify the type of error.
Choose one from: (A) Syntax Error  (B) Reference Error  (C) Logical Error  (D) Multiple Errors

Note: In your response, the final answer should be in the format: <Answer>(Option)</Answer>.

Buggy code:
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 2)   # bug here
""",
    },
    {
        "task": "Code Repair",
        "correct_answer": "factorial(n - 1)",
        "instruction": """Please fix the following buggy Python code so that it correctly \
computes the factorial of n.

Buggy code:
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 2)
""",
    },
    {
        "task": "Code Recognition",
        "correct_answer": "(B)",
        "instruction": """Given two code snippets, identify which one contains a bug.
Choose (A) or (B). Final answer format: <Answer>(Option)</Answer>.

(A)
def is_palindrome(s):
    return s == s[::-1]

(B)
def is_palindrome(s):
    return s == s[::1]
""",
    },
]


def build_prompt(instruction: str) -> str:
    return (
        "\nYou are an AI programming assistant, developed by NEUIR, "
        "and you only answer questions related to computer science. "
        "For politically sensitive questions, security and privacy issues, "
        "and other non-computer science questions, you will refuse to answer.\n"
        "### Instruction:\n"
        f"{instruction.strip()}\n"
        "### Response:\n"
    )


def ask(model, tokenizer, instruction: str, max_new_tokens: int = 256) -> str:
    prompt = build_prompt(instruction)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            top_p=0.95,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
    ).strip()


# ── LOAD TOKENIZER ────────────────────────────────────────────────────────────
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

# ── RUN BASE MODEL ────────────────────────────────────────────────────────────
print("\nLoading BASE model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
)
base_model.eval()
print("Base model ready. Running tests...")

base_answers = []
for t in tests:
    ans = ask(base_model, tokenizer, t["instruction"])
    base_answers.append(ans)
    print(f"  [{t['task']}] done")

# free GPU memory before loading fine-tuned model
del base_model
torch.cuda.empty_cache()

# ── RUN FINE-TUNED MODEL ──────────────────────────────────────────────────────
print("\nLoading FINE-TUNED model (base + LoRA)...")
ft_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
)
ft_model = PeftModel.from_pretrained(ft_model, ADAPTER_DIR)
ft_model.eval()
print("Fine-tuned model ready. Running tests...")

ft_answers = []
for t in tests:
    ans = ask(ft_model, tokenizer, t["instruction"])
    ft_answers.append(ans)
    print(f"  [{t['task']}] done")

del ft_model
torch.cuda.empty_cache()

# ── PRINT COMPARISON ──────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  COMPARISON: Base Model  vs  Fine-tuned (NeuDebugger)")
print("=" * 70)

correct_base = 0
correct_ft   = 0

for i, (t, base_ans, ft_ans) in enumerate(zip(tests, base_answers, ft_answers), 1):
    expected = t["correct_answer"]

    base_ok = expected.lower() in base_ans.lower()
    ft_ok   = expected.lower() in ft_ans.lower()
    if base_ok: correct_base += 1
    if ft_ok:   correct_ft   += 1

    print(f"\n[{i}/4] Task: {t['task']}  |  Expected: {expected}")
    print(f"  Base  {'✅' if base_ok else '❌'}: {base_ans[:300]}")
    print(f"  FT    {'✅' if ft_ok   else '❌'}: {ft_ans[:300]}")
    print("-" * 70)

print(f"\n{'='*70}")
print(f"  SCORE  →  Base: {correct_base}/4   |   Fine-tuned: {correct_ft}/4")
print(f"{'='*70}\n")
