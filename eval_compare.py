"""
Extended evaluation: Base model vs Fine-tuned (NeuDebugger)
- 20 test cases across all 4 DebugEval tasks
- Languages: Python, C++, Java
- Bug types: Syntax, Reference, Logical, Multiple
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL  = "/home/ubuntu/COAST/models/deepseek-coder-6.7b-instruct"
ADAPTER_DIR = "/home/ubuntu/COAST/output/deepseek-coder-6.7b-finetuned"

# ── TEST CASES ────────────────────────────────────────────────────────────────
tests = [

    # ── BUG LOCALIZATION (6 cases) ───────────────────────────────────────────
    {
        "task": "BUG Localization", "lang": "Python",
        "expected": "(B)",
        "instruction": """Given a programming task, its incorrect solution and the options. Only one option belongs to the error code snippet(s). Please select it.
Note: Final answer format: <Answer>(Option)</Answer>.

Task: Return the maximum element in a list.

Incorrect_Solution:
def find_max(lst):
    max_val = lst[0]
    for x in lst:
        if x < max_val:
            max_val = x
    return max_val

Options:
(A) max_val = lst[0]
(B) if x < max_val:
(C) for x in lst:
(D) return max_val
""",
    },
    {
        "task": "BUG Localization", "lang": "Python",
        "expected": "(C)",
        "instruction": """Given a programming task, its incorrect solution and the options. Only one option belongs to the error code snippet(s). Please select it.
Note: Final answer format: <Answer>(Option)</Answer>.

Task: Check if a number is prime.

Incorrect_Solution:
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, n):
        if n % i == 0:
            return True
    return False

Options:
(A) if n < 2:
(B) for i in range(2, n):
(C) return True
(D) return False
""",
    },
    {
        "task": "BUG Localization", "lang": "C++",
        "expected": "(B)",
        "instruction": """Given a programming task, its incorrect solution and the options. Only one option belongs to the error code snippet(s). Please select it.
Note: Final answer format: <Answer>(Option)</Answer>.

Task: Reverse a string in C++.

Incorrect_Solution:
string reverseStr(string s) {
    int n = s.length();
    for (int i = 0; i < n / 2; i++) {
        swap(s[i], s[i + 1]);
    }
    return s;
}

Options:
(A) int n = s.length();
(B) swap(s[i], s[i + 1]);
(C) for (int i = 0; i < n / 2; i++)
(D) return s;
""",
    },
    {
        "task": "BUG Localization", "lang": "Java",
        "expected": "(C)",
        "instruction": """Given a programming task, its incorrect solution and the options. Only one option belongs to the error code snippet(s). Please select it.
Note: Final answer format: <Answer>(Option)</Answer>.

Task: Compute the sum of elements in an array.

Incorrect_Solution:
public int sumArray(int[] arr) {
    int sum = 1;
    for (int x : arr) {
        sum += x;
    }
    return sum;
}

Options:
(A) for (int x : arr)
(B) sum += x;
(C) int sum = 1;
(D) return sum;
""",
    },
    {
        "task": "BUG Localization", "lang": "Python",
        "expected": "(A)",
        "instruction": """Given a programming task, its incorrect solution and the options. Only one option belongs to the error code snippet(s). Please select it.
Note: Final answer format: <Answer>(Option)</Answer>.

Task: Binary search — return index of target in sorted list, or -1.

Incorrect_Solution:
def binary_search(arr, target):
    left, right = 0, len(arr)
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

Options:
(A) left, right = 0, len(arr)
(B) mid = (left + right) // 2
(C) left = mid + 1
(D) right = mid - 1
""",
    },
    {
        "task": "BUG Localization", "lang": "C++",
        "expected": "(D)",
        "instruction": """Given a programming task, its incorrect solution and the options. Only one option belongs to the error code snippet(s). Please select it.
Note: Final answer format: <Answer>(Option)</Answer>.

Task: Compute n-th Fibonacci number (0-indexed, fib(0)=0, fib(1)=1).

Incorrect_Solution:
int fib(int n) {
    if (n <= 1) return n;
    int a = 0, b = 1;
    for (int i = 2; i <= n; i++) {
        int c = a + b;
        a = b;
        b = c;
    }
    return a;
}

Options:
(A) int a = 0, b = 1;
(B) int c = a + b;
(C) a = b;
(D) return a;
""",
    },

    # ── BUG IDENTIFICATION (5 cases) ─────────────────────────────────────────
    {
        "task": "BUG Identification", "lang": "Python",
        "expected": "(A)",
        "instruction": """Identify the type of error in the following buggy code.
Choose one: (A) Syntax Error  (B) Reference Error  (C) Logical Error  (D) Multiple Errors
Final answer format: <Answer>(Option)</Answer>.

Buggy code:
def greet(name)
    print("Hello, " + name)
""",
    },
    {
        "task": "BUG Identification", "lang": "Python",
        "expected": "(B)",
        "instruction": """Identify the type of error in the following buggy code.
Choose one: (A) Syntax Error  (B) Reference Error  (C) Logical Error  (D) Multiple Errors
Final answer format: <Answer>(Option)</Answer>.

Buggy code:
def compute():
    result = x + 10
    return result
""",
    },
    {
        "task": "BUG Identification", "lang": "Python",
        "expected": "(C)",
        "instruction": """Identify the type of error in the following buggy code.
Choose one: (A) Syntax Error  (B) Reference Error  (C) Logical Error  (D) Multiple Errors
Final answer format: <Answer>(Option)</Answer>.

Buggy code:
def count_vowels(s):
    count = 0
    for ch in s:
        if ch in "aeiou":
            count += 1
    return count + 1
""",
    },
    {
        "task": "BUG Identification", "lang": "Java",
        "expected": "(C)",
        "instruction": """Identify the type of error in the following buggy code.
Choose one: (A) Syntax Error  (B) Reference Error  (C) Logical Error  (D) Multiple Errors
Final answer format: <Answer>(Option)</Answer>.

Buggy code:
public boolean isPalindrome(String s) {
    int l = 0, r = s.length() - 1;
    while (l < r) {
        if (s.charAt(l) != s.charAt(r)) return false;
        l++;
        l++;   // should be r--
    }
    return true;
}
""",
    },
    {
        "task": "BUG Identification", "lang": "C++",
        "expected": "(D)",
        "instruction": """Identify the type of error in the following buggy code.
Choose one: (A) Syntax Error  (B) Reference Error  (C) Logical Error  (D) Multiple Errors
Final answer format: <Answer>(Option)</Answer>.

Buggy code:
int divide(int a, int b) {
    int result = a / b      // missing semicolon
    return ressult;         // typo: ressult
}
""",
    },

    # ── CODE REPAIR (5 cases) ─────────────────────────────────────────────────
    {
        "task": "Code Repair", "lang": "Python",
        "expected": "x * 2",
        "instruction": """Fix the following buggy Python code so that it doubles each element in a list.

Buggy code:
def double_list(lst):
    return [x + 2 for x in lst]
""",
    },
    {
        "task": "Code Repair", "lang": "Python",
        "expected": "left, right = 0, len(arr) - 1",
        "instruction": """Fix the following buggy Python binary search so it works correctly on a sorted list.

Buggy code:
def binary_search(arr, target):
    left, right = 0, len(arr)
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
""",
    },
    {
        "task": "Code Repair", "lang": "C++",
        "expected": "swap(s[i], s[n - 1 - i])",
        "instruction": """Fix the following buggy C++ function that is supposed to reverse a string in-place.

Buggy code:
string reverseStr(string s) {
    int n = s.length();
    for (int i = 0; i < n / 2; i++) {
        swap(s[i], s[i + 1]);
    }
    return s;
}
""",
    },
    {
        "task": "Code Repair", "lang": "Java",
        "expected": "int sum = 0",
        "instruction": """Fix the following buggy Java function that computes the sum of an array.

Buggy code:
public int sumArray(int[] arr) {
    int sum = 1;
    for (int x : arr) {
        sum += x;
    }
    return sum;
}
""",
    },
    {
        "task": "Code Repair", "lang": "Python",
        "expected": "return True",
        "instruction": """Fix the following buggy Python function that checks if a number is prime.

Buggy code:
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, n):
        if n % i == 0:
            return True
    return False
""",
    },

    # ── CODE RECOGNITION (4 cases) ────────────────────────────────────────────
    {
        "task": "Code Recognition", "lang": "Python",
        "expected": "(B)",
        "instruction": """Two code snippets are given. One contains a bug. Identify which.
Final answer format: <Answer>(Option)</Answer>.

(A)
def square_sum(n):
    return sum(i * i for i in range(1, n + 1))

(B)
def square_sum(n):
    return sum(i * i for i in range(1, n))
""",
    },
    {
        "task": "Code Recognition", "lang": "Python",
        "expected": "(A)",
        "instruction": """Two code snippets are given. One contains a bug. Identify which.
Final answer format: <Answer>(Option)</Answer>.

(A)
def remove_duplicates(lst):
    seen = []
    for x in lst:
        if x not in seen:
            seen.append(x)
    return lst

(B)
def remove_duplicates(lst):
    seen = []
    for x in lst:
        if x not in seen:
            seen.append(x)
    return seen
""",
    },
    {
        "task": "Code Recognition", "lang": "C++",
        "expected": "(B)",
        "instruction": """Two code snippets are given. One contains a bug. Identify which.
Final answer format: <Answer>(Option)</Answer>.

(A)
int fibonacci(int n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

(B)
int fibonacci(int n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 3);
}
""",
    },
    {
        "task": "Code Recognition", "lang": "Java",
        "expected": "(A)",
        "instruction": """Two code snippets are given. One contains a bug. Identify which.
Final answer format: <Answer>(Option)</Answer>.

(A)
public int[] twoSum(int[] nums, int target) {
    for (int i = 0; i < nums.length; i++)
        for (int j = i; j < nums.length; j++)  // bug: j should start at i+1
            if (nums[i] + nums[j] == target)
                return new int[]{i, j};
    return new int[]{};
}

(B)
public int[] twoSum(int[] nums, int target) {
    for (int i = 0; i < nums.length; i++)
        for (int j = i + 1; j < nums.length; j++)
            if (nums[i] + nums[j] == target)
                return new int[]{i, j};
    return new int[]{};
}
""",
    },
]


# ── INFERENCE HELPER ──────────────────────────────────────────────────────────
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


def run_all(model, tokenizer, label: str):
    print(f"\n{'─'*60}")
    print(f"  Running {label} ({len(tests)} test cases)...")
    print(f"{'─'*60}")
    answers = []
    for i, t in enumerate(tests, 1):
        ans = ask(model, tokenizer, t["instruction"])
        answers.append(ans)
        hit = "✅" if t["expected"].lower() in ans.lower() else "❌"
        print(f"  [{i:02d}] {hit} {t['task']:20s} [{t['lang']:6s}]  expected={t['expected']}")
    return answers


# ── LOAD TOKENIZER ────────────────────────────────────────────────────────────
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

# ── BASE MODEL ────────────────────────────────────────────────────────────────
print("\nLoading BASE model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
)
base_model.eval()
base_answers = run_all(base_model, tokenizer, "BASE model")
del base_model
torch.cuda.empty_cache()

# ── FINE-TUNED MODEL ──────────────────────────────────────────────────────────
print("\nLoading FINE-TUNED model (base + LoRA)...")
ft_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
)
ft_model = PeftModel.from_pretrained(ft_model, ADAPTER_DIR)
ft_model.eval()
ft_answers = run_all(ft_model, tokenizer, "FINE-TUNED model")
del ft_model
torch.cuda.empty_cache()

# ── DETAILED RESULTS ──────────────────────────────────────────────────────────
TASK_TYPES = ["BUG Localization", "BUG Identification", "Code Repair", "Code Recognition"]

print("\n" + "="*70)
print("  DETAILED COMPARISON")
print("="*70)

for i, (t, b_ans, f_ans) in enumerate(zip(tests, base_answers, ft_answers), 1):
    exp = t["expected"]
    b_ok = exp.lower() in b_ans.lower()
    f_ok = exp.lower() in f_ans.lower()
    print(f"\n[{i:02d}] {t['task']} | {t['lang']} | Expected: {exp}")
    print(f"  Base {'✅' if b_ok else '❌'}: {b_ans[:200].replace(chr(10),' ')}")
    print(f"  FT   {'✅' if f_ok else '❌'}: {f_ans[:200].replace(chr(10),' ')}")

# ── SCORE SUMMARY ─────────────────────────────────────────────────────────────
print("\n" + "="*70)
print(f"  {'Task':<22} {'Base':>8} {'Fine-tuned':>12}")
print("="*70)

total_b = total_f = 0
for task in TASK_TYPES:
    subset  = [(t, b, f) for t, b, f in zip(tests, base_answers, ft_answers)
               if t["task"] == task]
    b_score = sum(1 for t, b, _ in subset if t["expected"].lower() in b.lower())
    f_score = sum(1 for t, _, f in subset if t["expected"].lower() in f.lower())
    n       = len(subset)
    total_b += b_score
    total_f += f_score
    bar_b = "█" * b_score + "░" * (n - b_score)
    bar_f = "█" * f_score + "░" * (n - f_score)
    print(f"  {task:<22} {b_score}/{n}  {bar_b}   {f_score}/{n}  {bar_f}")

n = len(tests)
print("="*70)
print(f"  {'TOTAL':<22} {total_b}/{n}  ({100*total_b//n}%)   {total_f}/{n}  ({100*total_f//n}%)")
print("="*70 + "\n")
