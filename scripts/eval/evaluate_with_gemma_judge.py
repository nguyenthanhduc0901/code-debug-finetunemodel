import os
import sys
import json
import re
import argparse
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
REPO_SRC = Path('/workspace/finetune_gemma/scripts/eval/socratic_schemas')
if REPO_SRC.exists():
    sys.path.append(str(REPO_SRC))
else:
    print(f'Error: {REPO_SRC} does not exist.')
    sys.exit(1)
try:
    from data import Evaluation, Scores, Example
except ImportError as e:
    print(f'Error: Cannot import data structures from {REPO_SRC}: {e}')
    sys.exit(1)

def escape_template(str_template: str) -> str:
    pattern = re.compile('{[a-zA-Z0-9_]+}')
    keys = pattern.findall(str_template)
    placeholders = []
    protected_string = str_template
    for i, key in enumerate(keys):
        placeholder = f'__SPECIAL_CASE_{i}__'
        placeholders.append((placeholder, key))
        protected_string = protected_string.replace(key, placeholder)
    escaped_template = protected_string.replace('{', '{{').replace('}', '}}')
    final_template = escaped_template
    for placeholder, key in placeholders:
        final_template = final_template.replace(placeholder, key)
    return final_template

def clean_tutor_response(answer: str) -> str:
    for marker in ['Student:', 'User:', 'Teacher:', 'Assistant:', '\nStudent', '\nUser', '\nTeacher', '\nAssistant']:
        if marker in answer:
            answer = answer.split(marker)[0]
    return answer.strip()

def extract_and_parse_json(text: str) -> dict:
    cleaned = text.strip()
    match = re.search('(\\{.*\\})', cleaned, re.DOTALL)
    if match:
        cleaned = match.group(1)
    return json.loads(cleaned)

def main():
    parser = argparse.ArgumentParser(description='Evaluate Socratic Tutor using Gemma-4 Judge')
    parser.add_argument('--tutor-url', default='http://localhost:8001/v1', help='Socratic Tutor API base URL')
    parser.add_argument('--judge-url', default='http://localhost:8002/v1', help='Gemma-4 Judge API base URL')
    parser.add_argument('--tutor-model', default='socratic-tutor', help='Tutor model name')
    parser.add_argument('--judge-model', default='gemma-4-base', help='Judge model name')
    parser.add_argument('--dataset', default='/workspace/finetune_gemma/data/socratic/raw/debugging_test.json', help='Path to evaluation dataset')
    parser.add_argument('--inference-prompt', default='/workspace/finetune_gemma/scripts/eval/templates/inference.txt', help='Path to inference prompt template')
    parser.add_argument('--eval-prompt', default='/workspace/finetune_gemma/scripts/eval/templates/judge_llm.txt', help='Path to judge evaluation prompt template')
    parser.add_argument('--output', default='/workspace/finetune_gemma/logs/evaluation/socratic_evaluation_results_gemma.json', help='Path to save evaluation output')
    parser.add_argument('--num-samples', type=int, default=None, help='Number of samples to evaluate (default: all)')
    args = parser.parse_args()
    dataset_path = Path(args.dataset)
    inf_prompt_path = Path(args.inference_prompt)
    eval_prompt_path = Path(args.eval_prompt)
    if not dataset_path.exists():
        print(f'Dataset not found at: {dataset_path}')
        sys.exit(1)
    if not inf_prompt_path.exists():
        print(f'Inference prompt template not found at: {inf_prompt_path}')
        sys.exit(1)
    if not eval_prompt_path.exists():
        print(f'Eval prompt template not found at: {eval_prompt_path}')
        sys.exit(1)
    print('=' * 60)
    print('Socratic Tutor Evaluation with Gemma-4 Judge')
    print('=' * 60)
    print(f'Tutor Endpoint:  {args.tutor_url} (Model: {args.tutor_model})')
    print(f'Judge Endpoint:  {args.judge_url} (Model: {args.judge_model})')
    print(f'Dataset Path:    {args.dataset}')
    print(f'Output Path:     {args.output}')
    with open(inf_prompt_path, 'r', encoding='utf-8') as f:
        inference_prompt_template = f.read()
    with open(eval_prompt_path, 'r', encoding='utf-8') as f:
        judge_llm_prompt = escape_template(f.read())
    with open(dataset_path, 'r', encoding='utf-8') as f:
        eval_prompts = json.load(f)
    if args.num_samples:
        eval_prompts = eval_prompts[:args.num_samples]
    print(f'Total samples to evaluate: {len(eval_prompts)}')
    tutor_client = OpenAI(base_url=args.tutor_url, api_key='dummy')
    judge_client = OpenAI(base_url=args.judge_url, api_key='dummy')
    scores = Scores()
    for idx, prompt in enumerate(tqdm(eval_prompts, desc='Evaluating')):
        _prompt = inference_prompt_template.format(input=prompt)
        try:
            tutor_resp = tutor_client.chat.completions.create(model=args.tutor_model, messages=[{'role': 'user', 'content': _prompt}], max_tokens=250, temperature=0.2)
            raw_answer = tutor_resp.choices[0].message.content
            tutor_answer = clean_tutor_response(raw_answer)
        except Exception as e:
            print(f'\n Error querying tutor for sample {idx}: {e}')
            tutor_answer = f'[ERROR] {e}'
            raw_answer = tutor_answer
        formatted_judge_prompt = judge_llm_prompt.format(conversation=prompt, answer=tutor_answer)
        raw_evaluation = ''
        error_msg = None
        evaluation = None
        if not tutor_answer.startswith('[ERROR]'):
            try:
                judge_resp = judge_client.chat.completions.create(model=args.judge_model, messages=[{'role': 'user', 'content': formatted_judge_prompt}], temperature=0.2, seed=0)
                raw_evaluation = judge_resp.choices[0].message.content
                parsed_json = extract_and_parse_json(raw_evaluation)
                evaluation = Evaluation.model_validate(parsed_json)
            except Exception as e:
                error_msg = str(e)
                try:
                    parsed_json = extract_and_parse_json(raw_evaluation)
                    evaluation = Evaluation.model_validate(parsed_json)
                    error_msg = None
                except Exception as retry_err:
                    error_msg = f'Parsing error: {e}. Retry error: {retry_err}'
        else:
            error_msg = 'Tutor query failed.'
        scores.root.append(Example(prompt=prompt, output=tutor_answer, raw_evaluation=raw_evaluation, evaluation_error=error_msg, evaluation=evaluation))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(scores.model_dump_json(indent=2))
    os.chmod(output_path, 493)
    valid_examples = scores.get_valid()
    total_valid = len(valid_examples)
    total_examples = len(scores)
    print('\n' + '=' * 60)
    print('AGGREGATE RESULTS (Socratic Judge Gemma-4)')
    print('=' * 60)
    print(f'Total evaluated:      {total_examples}')
    print(f'Successfully parsed:  {total_valid}/{total_examples}')
    if total_valid > 0:
        print(f'Avg Socratic Score:   {scores.avg_summary_score()}/1.00')
        print(f'Question Rate (Yes):  {scores.avg_questions()}')
        print(f'On-Topic Rate:        {scores.avg_on_topic()}/1.00')
        print(f'Helpfulness Rate:     {scores.avg_helpfulness()}/1.00')
        print(f'Reveal Answer (Yes):  {scores.avg_reveal_answer()}')
    else:
        print('No valid evaluations parsed successfully.')
    print('=' * 60)
if __name__ == '__main__':
    main()
