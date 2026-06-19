import json
import re
import argparse
import sys
from pathlib import Path
import requests

def load_test_data(test_path: str, num_samples: int | None=None) -> list[dict]:
    with open(test_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if num_samples:
        data = data[:num_samples]
    return data

def build_messages(sample: dict) -> list[dict]:
    conversations = sample['conversations']
    messages = []
    for turn in conversations:
        if turn['from'] == 'system':
            messages.append({'role': 'system', 'content': turn['value']})
        elif turn['from'] == 'human':
            messages.append({'role': 'user', 'content': turn['value']})
        elif turn['from'] == 'gpt':
            messages.append({'role': 'assistant', 'content': turn['value']})
    if messages and messages[-1]['role'] == 'assistant':
        expected_response = messages.pop()['content']
    else:
        expected_response = None
    return (messages, expected_response)

def query_model(messages: list[dict], api_url: str, model_name: str, max_tokens: int=128, temperature: float=0.7) -> str:
    payload = {'model': model_name, 'messages': messages, 'max_tokens': max_tokens, 'temperature': temperature}
    try:
        response = requests.post(f'{api_url}/v1/chat/completions', json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']
    except Exception as e:
        return f'[ERROR] {e}'

def analyze_response(response: str) -> dict:
    metrics = {'has_question': '?' in response, 'question_count': response.count('?'), 'gives_direct_fix': any((phrase in response.lower() for phrase in ['the fix is', 'the bug is', 'you should change', 'replace it with', 'the error is on line', 'the answer is', 'the solution is', 'change it to', 'the problem is that you'])), 'uses_socratic_phrases': any((phrase in response.lower() for phrase in ['what do you think', 'can you explain', 'what happens when', "let's look at", 'have you tried', 'what would happen', 'can you tell me', 'do you see', "let's trace", "let's walk through", 'what does', 'why do you think', 'how many times', 'can you describe'])), 'suggests_debugging': any((phrase in response.lower() for phrase in ['print statement', 'print(', 'debugger', 'trace', 'test case', 'try running', "let's verify", 'step through'])), 'response_length': len(response), 'word_count': len(response.split())}
    score = 0
    if metrics['has_question']:
        score += 40
    if metrics['uses_socratic_phrases']:
        score += 30
    if not metrics['gives_direct_fix']:
        score += 20
    if metrics['suggests_debugging']:
        score += 10
    metrics['socratic_score'] = score
    return metrics

def print_sample_result(idx: int, messages: list[dict], expected: str | None, generated: str, metrics: dict):
    system_msg = messages[0]['content'] if messages else ''
    func_match = re.search('def (\\w+)\\(', system_msg)
    func_name = func_match.group(1) if func_match else 'unknown'
    last_user_msg = ''
    for msg in reversed(messages):
        if msg['role'] == 'user':
            last_user_msg = msg['content'][:80]
            break
    print(f'\n{'─' * 60}')
    print(f'Sample {idx + 1} | Function: {func_name}')
    print(f'{'─' * 60}')
    print(f'  Last student msg: {last_user_msg}...')
    print(f'  Generated:        {generated[:120]}...')
    if expected:
        print(f'  Expected:         {expected[:120]}...')
    print(f'  Socratic Score:   {metrics['socratic_score']}/100')
    print(f'  Has question:     {('Yes' if metrics['has_question'] else 'No')}')
    print(f'  Avoids direct fix:{('Yes' if not metrics['gives_direct_fix'] else 'No')}')
    print(f'  Socratic phrases: {('Yes' if metrics['uses_socratic_phrases'] else 'No')}')

def main():
    parser = argparse.ArgumentParser(description='Evaluate Socratic model')
    parser.add_argument('--api-url', default='http://localhost:8001', help='Model API base URL')
    parser.add_argument('--model-name', default='socratic-tutor', help='Model name in the API')
    parser.add_argument('--num-samples', type=int, default=None, help='Number of test samples to evaluate (default: all)')
    parser.add_argument('--test-data', default=None, help='Path to test data file')
    parser.add_argument('--output', default=None, help='Path to save results JSON')
    args = parser.parse_args()
    base_dir = Path('/workspace/finetune_gemma')
    test_path = args.test_data or str(base_dir / 'data' / 'socratic' / 'sft' / 'socratic_test.json')
    output_path = args.output or str(base_dir / 'logs' / 'evaluation' / 'socratic_evaluation_results.json')
    print('=' * 60)
    print('Socratic Debugging Model Evaluation')
    print('=' * 60)
    print(f'API URL:    {args.api_url}')
    print(f'Model:      {args.model_name}')
    print(f'Test data:  {test_path}')
    test_data = load_test_data(test_path, args.num_samples)
    print(f'Samples:    {len(test_data)}')
    all_metrics = []
    all_results = []
    for i, sample in enumerate(test_data):
        messages, expected = build_messages(sample)
        if not messages or messages[-1]['role'] != 'user':
            continue
        generated = query_model(messages, args.api_url, args.model_name)
        metrics = analyze_response(generated)
        print_sample_result(i, messages, expected, generated, metrics)
        all_metrics.append(metrics)
        all_results.append({'index': i, 'generated': generated, 'expected': expected, 'metrics': metrics})
    if all_metrics:
        n = len(all_metrics)
        print('\n' + '=' * 60)
        print('AGGREGATE RESULTS')
        print('=' * 60)
        print(f'Total evaluated:         {n}')
        print(f'Avg Socratic Score:      {sum((m['socratic_score'] for m in all_metrics)) / n:.1f}/100')
        print(f'Has questions:           {sum((m['has_question'] for m in all_metrics))}/{n} ({100 * sum((m['has_question'] for m in all_metrics)) / n:.1f}%)')
        print(f'Avoids direct fix:       {sum((not m['gives_direct_fix'] for m in all_metrics))}/{n} ({100 * sum((not m['gives_direct_fix'] for m in all_metrics)) / n:.1f}%)')
        print(f'Uses Socratic phrases:   {sum((m['uses_socratic_phrases'] for m in all_metrics))}/{n} ({100 * sum((m['uses_socratic_phrases'] for m in all_metrics)) / n:.1f}%)')
        print(f'Suggests debugging:      {sum((m['suggests_debugging'] for m in all_metrics))}/{n} ({100 * sum((m['suggests_debugging'] for m in all_metrics)) / n:.1f}%)')
        print(f'Avg response length:     {sum((m['word_count'] for m in all_metrics)) / n:.0f} words')
        results_output = {'config': {'api_url': args.api_url, 'model_name': args.model_name, 'num_samples': len(test_data)}, 'aggregate': {'total': n, 'avg_socratic_score': sum((m['socratic_score'] for m in all_metrics)) / n, 'question_rate': sum((m['has_question'] for m in all_metrics)) / n, 'direct_fix_avoidance': sum((not m['gives_direct_fix'] for m in all_metrics)) / n, 'socratic_phrase_rate': sum((m['uses_socratic_phrases'] for m in all_metrics)) / n}, 'samples': all_results}
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_output, f, indent=2, ensure_ascii=False)
        print(f'\n Results saved to: {output_path}')
    else:
        print('\nNo samples evaluated.')
if __name__ == '__main__':
    main()
