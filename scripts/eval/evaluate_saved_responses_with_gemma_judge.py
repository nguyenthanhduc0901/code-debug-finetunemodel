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

def extract_and_parse_json(text: str) -> dict:
    cleaned = text.strip()
    match = re.search('(\\{.*\\})', cleaned, re.DOTALL)
    if match:
        cleaned = match.group(1)
    return json.loads(cleaned)

def main():
    parser = argparse.ArgumentParser(description='Evaluate saved Socratic Tutor responses using Gemma-4 Judge')
    parser.add_argument('--judge-url', default='http://localhost:8002/v1', help='Gemma-4 Judge API base URL')
    parser.add_argument('--judge-model', default='gemma-4-base', help='Judge model name')
    parser.add_argument('--heuristic-results', default='/workspace/finetune_gemma/logs/evaluation/socratic_2modules_heuristic_results.json', help='Path to saved heuristic results JSON')
    parser.add_argument('--dataset', default='/workspace/finetune_gemma/data/socratic/sft/socratic_test.json', help='Path to evaluation dataset')
    parser.add_argument('--eval-prompt', default='/workspace/finetune_gemma/scripts/eval/templates/judge_llm.txt', help='Path to judge evaluation prompt template')
    parser.add_argument('--output', default='/workspace/finetune_gemma/logs/evaluation/socratic_2modules_gemma_judge_results.json', help='Path to save evaluation output')
    args = parser.parse_args()

    judge_prompt_path = Path(args.eval_prompt)
    heuristic_path = Path(args.heuristic_results)

    if not judge_prompt_path.exists():
        print(f'Eval prompt template not found at: {judge_prompt_path}')
        sys.exit(1)
    if not heuristic_path.exists():
        print(f'Heuristic results not found at: {heuristic_path}')
        sys.exit(1)

    print('=' * 60)
    print('Socratic Tutor Evaluation with Gemma-4 Judge')
    print('=' * 60)
    print(f'Judge Endpoint:     {args.judge_url} (Model: {args.judge_model})')
    print(f'Heuristic Results: {args.heuristic_results}')
    print(f'Output Path:        {args.output}')

    with open(judge_prompt_path, 'r', encoding='utf-8') as f:
        judge_llm_prompt = escape_template(f.read())

    with open(heuristic_path, 'r', encoding='utf-8') as f:
        heuristic_data = json.load(f)

    samples = heuristic_data.get('samples', [])
    print(f'Total samples to evaluate: {len(samples)}')

    judge_client = OpenAI(base_url=args.judge_url, api_key='dummy')
    scores = Scores()

    for item in tqdm(samples, desc='Judging'):
        tutor_answer = item['generated']
        prompt_conv = item.get('expected', '')
        
        formatted_judge_prompt = judge_llm_prompt.format(conversation=prompt_conv, answer=tutor_answer)
        raw_evaluation = ''
        error_msg = None
        evaluation = None

        try:
            judge_resp = judge_client.chat.completions.create(
                model=args.judge_model,
                messages=[{'role': 'user', 'content': formatted_judge_prompt}],
                temperature=0.2,
                seed=0
            )
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

        scores.root.append(Example(
            prompt=prompt_conv,
            output=tutor_answer,
            raw_evaluation=raw_evaluation,
            evaluation_error=error_msg,
            evaluation=evaluation
        ))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(scores.model_dump_json(indent=2))

    valid_examples = scores.get_valid()
    total_valid = len(valid_examples)
    total_examples = len(scores)

    print('\n' + '=' * 60)
    print('AGGREGATE RESULTS (Socratic Judge Gemma-4 - 2 Modules)')
    print('=' * 60)
    print(f'Total evaluated:      {total_examples}')
    print(f'Successfully parsed:  {total_valid}/{total_examples}')
    if total_valid > 0:
        print(f'Avg Socratic Score:   {scores.avg_summary_score():.3f} / 1.00')
        print(f'Question Rate (Yes):  {scores.avg_questions():.3f}')
        print(f'On-Topic Rate:        {scores.avg_on_topic():.3f} / 1.00')
        print(f'Helpfulness Rate:     {scores.avg_helpfulness():.3f} / 1.00')
        print(f'Reveal Answer (Yes):  {scores.avg_reveal_answer():.3f}')
    else:
        print('No valid evaluations parsed successfully.')
    print('=' * 60)

if __name__ == '__main__':
    main()
