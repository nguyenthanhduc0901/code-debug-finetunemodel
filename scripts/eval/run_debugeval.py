import os
import sys
import json
import argparse
import re
import time
import threading
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import shutil

INPUT_BASE_DIR = "/workspace/finetune_gemma/data/debugeval/atcoder_cases/Input"
OUTPUT_BASE_DIR = "/workspace/finetune_gemma/data/debugeval/atcoder_cases/Output"
TASK124_DATA_PATH = "/workspace/finetune_gemma/data/debugeval/raw/debugevalsuite_task124.jsonl"
TASK3_DATA_PATH = "/workspace/finetune_gemma/data/debugeval/raw/debugevalsuite_task3.jsonl"

# Add OJ evaluation path
sys.path.append(os.path.abspath('/workspace/finetune_gemma/COAST/OJ_Evaluation/CodeError_Judge-main'))
import judgeLib

def load_prompt_template(task_name, is_reverse=False, is_sft=False):
    prompts_dir = "/workspace/finetune_gemma/COAST/src/prompts"
    if is_sft:
        if task_name == "localization":
            path = os.path.join(prompts_dir, "error_code_localization", "llama_fine_tune", "prompt_zero_shot.txt")
        elif task_name == "identification":
            path = os.path.join(prompts_dir, "error_type_identification", "llama_fine_tune", "prompt_zero_shot.txt")
        elif task_name == "repair":
            path = os.path.join(prompts_dir, "code_repair", "NO_COT", "llama_fine_tune", "prompt_zero_shot.txt")
        elif task_name == "review":
            if is_reverse:
                path = os.path.join(prompts_dir, "code_review_reverse", "llama_fine_tune", "prompt_zero_shot.txt")
            else:
                path = os.path.join(prompts_dir, "code_review", "llama_fine_tune", "prompt_zero_shot.txt")
    else:
        if task_name == "localization":
            path = os.path.join(prompts_dir, "error_code_localization", "prompt_zero_shot.txt")
        elif task_name == "identification":
            path = os.path.join(prompts_dir, "error_type_identification", "prompt_zero_shot.txt")
        elif task_name == "repair":
            path = os.path.join(prompts_dir, "code_repair", "prompt_zero_shot.txt")
        elif task_name == "review":
            if is_reverse:
                path = os.path.join(prompts_dir, "code_review_reverse", "prompt_zero_shot.txt")
            else:
                path = os.path.join(prompts_dir, "code_review", "prompt_zero_shot.txt")
    
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def format_prompt(template, item):
    prompt = template
    if '%%%lang%%%' in prompt:
        prompt = prompt.replace('%%%lang%%%', item.get('language', ''))
    if '%%%Task%%%' in prompt:
        prompt = prompt.replace('%%%Task%%%', item.get('question_content', ''))
    if '%%%Incorrect_Solution%%%' in prompt:
        prompt = prompt.replace('%%%Incorrect_Solution%%%', item.get('buggy_code', ''))
    if '%%%Options%%%' in prompt:
        prompt = prompt.replace('%%%Options%%%', item.get('task1_options', ''))
    if '%%%buggy_code%%%' in prompt:
        prompt = prompt.replace('%%%buggy_code%%%', item.get('buggy_code', ''))
    if '%%%correct_code%%%' in prompt:
        prompt = prompt.replace('%%%correct_code%%%', item.get('correct_code', ''))
    return prompt

# Post-processing functions
def post_process_error_code_localization_zero_shot(response):
    match = re.search(r'<Answer>(.*?)</Answer>', response, re.DOTALL)
    content = match.group(1).strip() if match else response.strip()
    
    opt_match = re.search(r'\(([A-D])\)', content)
    if opt_match:
        return f"({opt_match.group(1)})"
    opt_match = re.search(r'\b([A-D])\b', content)
    if opt_match:
        return f"({opt_match.group(1)})"
    return content.replace("\n", " ").strip()

def post_process_error_type_identification_zero_shot(response):
    match = re.search(r'<Answer>(.*?)</Answer>', response, re.DOTALL)
    content = match.group(1).strip() if match else response.strip()
    
    opt_match = re.search(r'\(([A-D])\)', content)
    if opt_match:
        return f"({opt_match.group(1)})"
    opt_match = re.search(r'\b([A-D])\b', content)
    if opt_match:
        return f"({opt_match.group(1)})"
    return content.replace("\n", " ").strip()

def post_process_code_repair_zero_shot(response):
    # Strategy 1: Extract code after "Correct_Solution:" label
    for label in ['Correct_Solution:', 'Corrected_Solution:', 'Corrected Solution:', 'CorrectSolution:', '_Correct_Solution:', '_Correct_Solution{', '_Correct_:', '_FIX_:', '_FIX_', 'Answer_Code:', 'Lowered_Code:', 'Answer:']:
        if label in response:
            code_part = response.split(label, 1)[1].strip()
            inner_blocks = re.findall(r'```(?:code|cpp|java|python|Python)?\s*(.*?)\s*```', code_part, re.DOTALL)
            if inner_blocks:
                return inner_blocks[0].strip()
            return code_part
    
    # Strategy 2: Extract from code blocks
    code_blocks = re.findall(r'```(?:code|cpp|java|python|Python)?\s*(.*?)\s*```', response, re.DOTALL)
    valid_blocks = [
        b.strip() for b in code_blocks 
        if b.strip() and "correct_solution" not in b.lower() and b.strip() != "and"
    ]
    
    if valid_blocks:
        return valid_blocks[0]
    elif code_blocks:
        return code_blocks[0].strip()
    elif '```' in response:
        parts = response.split('```')
        return parts[1].strip() if len(parts) > 1 else response.strip()
    
    return response.strip()

def post_process_code_review_zero_shot(response):
    patterns = [
        r"buggy code snippet is \*\*?(Code-[AB])\*\*?",
        r"incorrect code snippet is \*\*?(Code-[AB])\*\*?",
        r"buggy snippet is \*\*?(Code-[AB])\*\*?",
        r"\*\*?(Code-[AB])\*\*? is the buggy snippet",
        r"\*\*?(Code-[AB])\*\*? is the incorrect snippet",
        r"buggy code is \*\*?(Code-[AB])\*\*?",
        r"incorrect code is \*\*?(Code-[AB])\*\*?",
        r"buggy is \*\*?(Code-[AB])\*\*?",
        r"incorrect is \*\*?(Code-[AB])\*\*?",
        r"bug is in \*\*?(Code-[AB])\*\*?",
        r"(Code-[AB]) is the buggy",
    ]
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            found = match.group(1)
            return "Code-A" if found.upper().endswith('A') else "Code-B"
            
    sentences = re.split(r'[.!?\n]', response)
    keywords = ["buggy", "incorrect", "bug", "error", "flaw", "wrong", "mistake", "bugged", "flawed"]
    for sentence in sentences:
        if any(kw in sentence.lower() for kw in keywords):
            has_a = "Code-A" in sentence
            has_b = "Code-B" in sentence
            if has_a and not has_b:
                return "Code-A"
            if has_b and not has_a:
                return "Code-B"
                
    if 'Code-A' in response and 'Code-B' not in response:
        return 'Code-A'
    if 'Code-B' in response and 'Code-A' not in response:
        return 'Code-B'
    return "don't know"

# Concurrent Query Batching
def query_model_batch(client, model_type, model_name, prompts, max_workers=32, max_tokens=8096):
    total = len(prompts)
    results = [None] * total
    counter = {'done': 0}
    lock = threading.Lock()
    start_time = time.time()

    def single_query(idx_prompt):
        idx, prompt = idx_prompt
        est_prompt_tokens = len(prompt) // 3
        safe_max_tokens = 16384 - est_prompt_tokens - 50
        safe_max_tokens = max(128, min(max_tokens, safe_max_tokens))
        try:
            system_msg = "You are a helpful assistant."
            
            # Format prompt based on model-type template
            if model_type in ["qwen", "qwen-sft"]:
                chat_prompt = f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            elif model_type == "gemma":
                chat_prompt = f"<bos><|turn>system\n<|think|>{system_msg}<turn|>\n<|turn>user\n{prompt}<turn|>\n<|turn>model\n"
            elif model_type == "gemma-sft":
                chat_prompt = f"<bos><|turn>system\n<|think|>{system_msg}<turn|>\n<|turn>user\n{prompt}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
            else:
                # Fallback to plain prompt
                chat_prompt = prompt
            
            response = client.completions.create(
                model=model_name,
                prompt=chat_prompt,
                max_tokens=safe_max_tokens,
                temperature=0.2,
                top_p=0.95
            )
            result = response.choices[0].text
            
            # Strip Gemma thinking channel if present
            if model_type == "gemma" and "<|channel>thought" in result:
                parts = result.split("<channel|>", 1)
                if len(parts) > 1:
                    result = parts[1]

        except Exception as e:
            result = f"Error: {str(e)}"
        
        with lock:
            counter['done'] += 1
            done = counter['done']
            elapsed = time.time() - start_time
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / rate if rate > 0 else 0
            print(f"  ⏳ Inference: {done}/{total} ({done*100/total:.1f}%) | {elapsed:.0f}s elapsed | ETA {eta:.0f}s", flush=True)
        
        return idx, result

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(single_query, (i, p)) for i, p in enumerate(prompts)]
        for future in as_completed(futures):
            idx, result = future.result()
            results[idx] = result
    
    return results

# Task execution functions
def run_task1_localization(client, model_type, model_name, data, limit, verbose):
    print("\n--- Running Task 1: BUG Localization ---")
    filtered_data = [d for d in data if d.get('task1_options') != '']
    if limit:
        filtered_data = filtered_data[:limit]
    
    correct_count = 0
    total_count = len(filtered_data)
    print(f"  Total samples: {total_count}")
    prompt_template = load_prompt_template("localization", is_sft=(model_type in ["qwen-sft", "gemma-sft"]))
    
    prompts = [format_prompt(prompt_template, item) for item in filtered_data]
    print(f"  Querying model...")
    raw_outputs = query_model_batch(client, model_type, model_name, prompts, max_workers=32)
    
    print(f"  Evaluating responses...")
    for i, (item, raw_output, prompt) in enumerate(zip(filtered_data, raw_outputs, prompts)):
        prediction = post_process_error_code_localization_zero_shot(raw_output)
        expected = item['task1_answer']
        is_correct = (prediction == expected)
        if is_correct:
            correct_count += 1
            
        if verbose:
            print(f"\n[Sample] ID: {item['question_id']}")
            print(f"--- Prompt ---\n{prompt}\n--------------")
            print(f"--- Raw Output ---\n{raw_output}\n------------------")
            print(f"Prediction: '{prediction}' | Expected: '{expected}' | Correct: {is_correct}")
            print("="*60)
        elif (i + 1) % 50 == 0 or i == total_count - 1:
            running_acc = correct_count / (i + 1) * 100
            print(f"  📊 Evaluated: {i+1}/{total_count} ({(i+1)*100/total_count:.1f}%) | Running Acc: {running_acc:.1f}%", flush=True)
            
    acc = (correct_count / total_count * 100) if total_count > 0 else 0.0
    print(f"  ✅ Task 1 Results: {correct_count}/{total_count} correct (Accuracy: {acc:.2f}%)")
    return {"accuracy": acc, "correct": correct_count, "total": total_count}

def run_task2_identification(client, model_type, model_name, data, limit, verbose):
    print("\n--- Running Task 2: BUG Identification ---")
    filtered_data = [d for d in data if d.get('task2_choice') != '']
    if limit:
        filtered_data = filtered_data[:limit]
        
    correct_count = 0
    total_count = len(filtered_data)
    print(f"  Total samples: {total_count}")
    prompt_template = load_prompt_template("identification", is_sft=(model_type in ["qwen-sft", "gemma-sft"]))
    
    prompts = [format_prompt(prompt_template, item) for item in filtered_data]
    print(f"  Querying model...")
    raw_outputs = query_model_batch(client, model_type, model_name, prompts, max_workers=32)
    
    print(f"  Evaluating responses...")
    for i, (item, raw_output, prompt) in enumerate(zip(filtered_data, raw_outputs, prompts)):
        prediction = post_process_error_type_identification_zero_shot(raw_output)
        expected = item['task2_choice']
        is_correct = (prediction == expected)
        if is_correct:
            correct_count += 1
            
        if verbose:
            print(f"\n[Sample] ID: {item['question_id']}")
            print(f"--- Prompt ---\n{prompt}\n--------------")
            print(f"--- Raw Output ---\n{raw_output}\n------------------")
            print(f"Prediction: '{prediction}' | Expected: '{expected}' | Correct: {is_correct}")
            print("="*60)
        elif (i + 1) % 50 == 0 or i == total_count - 1:
            running_acc = correct_count / (i + 1) * 100
            print(f"  📊 Evaluated: {i+1}/{total_count} ({(i+1)*100/total_count:.1f}%) | Running Acc: {running_acc:.1f}%", flush=True)
            
    acc = (correct_count / total_count * 100) if total_count > 0 else 0.0
    print(f"  ✅ Task 2 Results: {correct_count}/{total_count} correct (Accuracy: {acc:.2f}%)")
    return {"accuracy": acc, "correct": correct_count, "total": total_count}

def judge_single_problem(idx_item_raw_prompt):
    idx, item, raw_output, prompt = idx_item_raw_prompt
    repaired_code = post_process_code_repair_zero_shot(raw_output)
    lang = item['language']
    q_id = item['question_id']
    lang_ext = 'py' if lang == 'python' else ('cpp' if lang == 'cpp' else 'java')
    
    sub_dir = f"/workspace/finetune_gemma/temp_q_{q_id}_{idx}"
    os.makedirs(sub_dir, exist_ok=True)
    
    source_file = os.path.join(sub_dir, "Main.java" if lang == 'java' else f"temp.{lang_ext}")
    with open(source_file, "w", encoding="utf-8") as f:
        f.write(repaired_code)
        
    input_path = os.path.join(INPUT_BASE_DIR, q_id)
    answer_path = os.path.join(OUTPUT_BASE_DIR, q_id)
    is_correct = False
    judge_res = "Input/Output cases not found"
    
    if os.path.exists(input_path) and os.path.exists(answer_path):
        test_cnt = len(os.listdir(input_path))
        
        res_first, _, _, _ = judgeLib.judge(
            file_dir=source_file,
            input_dir=os.path.join(input_path, "1.in"),
            answer_dir=os.path.join(answer_path, "1.out"),
            timeLimit=10,
            memoryLimit=1024,
            test_id=0,
            cleanup=False
        )
        
        if res_first == 'CE':
            passed_all = False
            judge_res = "Failed at CE"
        else:
            passed_all = (res_first == 'AC')
            judge_res_list = [res_first]
            
            if test_cnt > 1:
                def run_single_test(t_idx):
                    res, _, _, _ = judgeLib.judge(
                        file_dir=source_file,
                        input_dir=os.path.join(input_path, f"{t_idx+1}.in"),
                        answer_dir=os.path.join(answer_path, f"{t_idx+1}.out"),
                        timeLimit=10,
                        memoryLimit=1024,
                        test_id=t_idx,
                        cleanup=False
                    )
                    return res
                
                with ThreadPoolExecutor(max_workers=16) as test_executor:
                    remaining_res = list(test_executor.map(run_single_test, range(1, test_cnt)))
                
                judge_res_list.extend(remaining_res)
                for r in remaining_res:
                    if r != 'AC':
                        passed_all = False
                        
            is_correct = passed_all
            judge_res = "AC" if passed_all else f"Failed at {judge_res_list}"
            
    shutil.rmtree(sub_dir, ignore_errors=True)
    return idx, q_id, lang, is_correct, judge_res, prompt, raw_output, repaired_code

def run_task3_repair(client, model_type, model_name, data, limit, verbose):
    print("\n--- Running Task 3: Code Repair ---")
    languages = ['python', 'cpp', 'java']
    filtered_data = []
    if limit:
        for lang in languages:
            lang_data = [d for d in data if d.get('language') == lang]
            filtered_data.extend(lang_data[:limit])
    else:
        filtered_data = data
    
    correct_count = 0
    total_count = len(filtered_data)
    lang_counts = {}
    for d in filtered_data:
        l = d.get('language', 'unknown')
        lang_counts[l] = lang_counts.get(l, 0) + 1
    print(f"  Total samples: {total_count} ({', '.join(f'{l}:{c}' for l, c in lang_counts.items())})")
    prompt_template = load_prompt_template("repair", is_sft=(model_type in ["qwen-sft", "gemma-sft"]))
    
    prompts = [format_prompt(prompt_template, item) for item in filtered_data]
    print(f"  Querying model...")
    raw_outputs = query_model_batch(client, model_type, model_name, prompts, max_workers=32)
    
    print(f"  Judging repaired code in parallel (ProcessPoolExecutor)...")
    judge_start = time.time()
    inputs = list(zip(range(total_count), filtered_data, raw_outputs, prompts))
    
    with ProcessPoolExecutor(max_workers=16) as outer_executor:
        results = list(outer_executor.map(judge_single_problem, inputs))
        
    judge_duration = time.time() - judge_start
    print(f"  ✨ Judging completed in {judge_duration:.2f} seconds.")
    
    for idx, q_id, lang, is_correct, judge_res, prompt, raw_output, repaired_code in sorted(results, key=lambda x: x[0]):
        if is_correct:
            correct_count += 1
            
        if verbose:
            print(f"\n[Sample] ID: {q_id} | Language: {lang}")
            print(f"--- Prompt ---\n{prompt}\n--------------")
            print(f"--- Raw Output ---\n{raw_output}\n------------------")
            print(f"--- Post-processed Code ---\n{repaired_code}\n---------------------------")
            print(f"Judge Result: {judge_res} | Correct: {is_correct}")
            print("="*60)
            
        running_acc = correct_count / (idx + 1) * 100
        print(f"  📊 Evaluated: {idx+1}/{total_count} ({(idx+1)*100/total_count:.1f}%) | Running Acc: {running_acc:.1f}%", flush=True)
        
    acc = (correct_count / total_count * 100) if total_count > 0 else 0.0
    print(f"  ✅ Task 3 Results: {correct_count}/{total_count} correct (Accuracy: {acc:.2f}%)")
    return {"accuracy": acc, "correct": correct_count, "total": total_count}

def run_task4_review(client, model_type, model_name, data, limit, verbose):
    print("\n--- Running Task 4: Code Review ---")
    filtered_data = [d for d in data if d.get('task4') == 'True' or d.get('task4') == True]
    if limit:
        filtered_data = filtered_data[:limit]
        
    correct_count = 0
    total_count = len(filtered_data) * 2
    print(f"  Total samples: {len(filtered_data)} (×2 directions = {total_count})")
    review_prompt_template = load_prompt_template("review", is_reverse=False, is_sft=(model_type in ["qwen-sft", "gemma-sft"]))
    reverse_prompt_template = load_prompt_template("review", is_reverse=True, is_sft=(model_type in ["qwen-sft", "gemma-sft"]))
    
    prompts_normal = []
    prompts_reverse = []
    for item in filtered_data:
        prompt_normal = format_prompt(review_prompt_template, item)
        prompt_reverse = format_prompt(reverse_prompt_template, item)
        prompts_normal.append(prompt_normal)
        prompts_reverse.append(prompt_reverse)
        
    all_prompts = prompts_normal + prompts_reverse
    print(f"  Querying model ({len(all_prompts)} prompts)...")
    raw_outputs = query_model_batch(client, model_type, model_name, all_prompts, max_workers=32)
    
    raw_outputs_normal = raw_outputs[:len(filtered_data)]
    raw_outputs_reverse = raw_outputs[len(filtered_data):]
    
    print(f"  Evaluating responses...")
    for i, (item, raw_output_normal, raw_output_reverse, prompt_normal, prompt_reverse) in enumerate(zip(
        filtered_data, raw_outputs_normal, raw_outputs_reverse, prompts_normal, prompts_reverse
    )):
        prediction_normal = post_process_code_review_zero_shot(raw_output_normal)
        is_correct_normal = (prediction_normal == 'Code-A')
        if is_correct_normal:
            correct_count += 1
            
        prediction_reverse = post_process_code_review_zero_shot(raw_output_reverse)
        is_correct_reverse = (prediction_reverse == 'Code-B')
        if is_correct_reverse:
            correct_count += 1
            
        if verbose:
            print(f"\n[Sample] ID: {item['question_id']}")
            print(f"--- Normal Prompt ---\n{prompt_normal}\n--------------")
            print(f"--- Normal Raw Output ---\n{raw_output_normal}\n------------------")
            print(f"Prediction Normal: '{prediction_normal}' | Expected: 'Code-A' | Correct: {is_correct_normal}")
            print(f"--- Reverse Prompt ---\n{prompt_reverse}\n--------------")
            print(f"--- Reverse Raw Output ---\n{raw_output_reverse}\n------------------")
            print(f"Prediction Reverse: '{prediction_reverse}' | Expected: 'Code-B' | Correct: {is_correct_reverse}")
            print("="*60)
        elif (i + 1) % 50 == 0 or i == len(filtered_data) - 1:
            running_acc = correct_count / ((i + 1) * 2) * 100
            print(f"  📊 Evaluated: {i+1}/{len(filtered_data)} ({(i+1)*100/len(filtered_data):.1f}%) | Running Acc: {running_acc:.1f}%", flush=True)
            
    acc = (correct_count / total_count * 100) if total_count > 0 else 0.0
    print(f"  ✅ Task 4 Results: {correct_count}/{total_count} correct (Accuracy: {acc:.2f}%)")
    return {"accuracy": acc, "correct": correct_count, "total": total_count}

class Logger(object):
    def __init__(self, filename="evaluation.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM on DebugEval Benchmark Tasks using COAST prompts")
    parser.add_argument("--model-type", type=str, default="qwen", choices=["qwen", "qwen-sft", "gemma", "gemma-sft"], help="Model type template to use")
    parser.add_argument("--model-name", type=str, default="", help="Served model name in vLLM (defaults match model-type)")
    parser.add_argument("--port", type=int, default=8888, help="vLLM server port")
    parser.add_argument("--tasks", type=str, default="1,2,3,4", help="Tasks to run (1,2,3,4)")
    parser.add_argument("--limit", type=int, default=0, help="Limit per task")
    parser.add_argument("--log-file", type=str, default="/workspace/finetune_gemma/logs/evaluation/debugeval.log", help="Log path")
    args = parser.parse_args()
    
    sys.stdout = Logger(args.log_file)
    print(f"Logging execution to: {args.log_file}")
    
    # Auto-resolve model name if not provided
    model_name = args.model_name
    if not model_name:
        if args.model_type == "qwen":
            model_name = "qwen2.5-coder-3b-instruct"
        elif args.model_type == "qwen-sft":
            model_name = "qwen-sft"
        elif args.model_type in ["gemma", "gemma-sft"]:
            model_name = "gemma-4-e4b-it"
        else:
            model_name = "model"
            
    client = OpenAI(
        api_key="EMPTY",
        base_url=f"http://localhost:{args.port}/v1",
    )
    
    def load_jsonl(path):
        with open(path, 'r', encoding='utf-8') as f:
            return [json.loads(line.strip()) for line in f if line.strip()]

    print("Loading datasets...")
    data_124 = load_jsonl(TASK124_DATA_PATH)
    data_3 = load_jsonl(TASK3_DATA_PATH)
    
    selected_tasks = [int(t.strip()) for t in args.tasks.split(",")]
    limit_value = args.limit
    verbose = (limit_value > 0)
    
    results = {}
    if 1 in selected_tasks:
        results[1] = run_task1_localization(client, args.model_type, model_name, data_124, limit_value, verbose)
    if 2 in selected_tasks:
        results[2] = run_task2_identification(client, args.model_type, model_name, data_124, limit_value, verbose)
    if 3 in selected_tasks:
        results[3] = run_task3_repair(client, args.model_type, model_name, data_3, limit_value, verbose)
    if 4 in selected_tasks:
        results[4] = run_task4_review(client, args.model_type, model_name, data_124, limit_value, verbose)
        
    print("\n" + "="*40 + "\nSUMMARY EVALUATION RESULTS\n" + "="*40)
    for task_num, res in results.items():
        task_name = {
            1: "BUG Localization",
            2: "BUG Identification",
            3: "Code Repair",
            4: "Code Review"
        }[task_num]
        print(f"Task {task_num} ({task_name}): Accuracy: {res['accuracy']:.2f}% ({res['correct']}/{res['total']})")
    print("="*40)

if __name__ == "__main__":
    main()
