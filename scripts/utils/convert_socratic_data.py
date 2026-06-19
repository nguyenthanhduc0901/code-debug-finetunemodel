import json
import re
import sys
import os
from pathlib import Path
from collections import Counter
SYSTEM_TEMPLATE = 'You are a Socratic tutor for programming. A student has written buggy code for a given problem. You have access to the problem description, the buggy code, a description of the bug, and the correct fix. However, you must NOT directly reveal the bug or the fix to the student. Instead, guide the student to discover and fix the bug on their own by asking probing questions, suggesting debugging strategies (e.g., print statements, tracing), and giving hints that lead them in the right direction.\n\n{context}'

def extract_xml_block(text: str, tag: str) -> str | None:
    pattern = f'<{tag}>(.*?)</{tag}>'
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None

def build_system_prompt(raw_text: str) -> str:
    blocks = []
    for tag in ['problem', 'bug_code', 'bug_desc', 'bug_fixes']:
        content = extract_xml_block(raw_text, tag)
        if content:
            blocks.append(f'<{tag}>\n{content}\n</{tag}>')
    context = '\n\n'.join(blocks)
    return SYSTEM_TEMPLATE.format(context=context)

def parse_conversation(raw_text: str) -> list[dict]:
    marker = '</bug_fixes>'
    idx = raw_text.find(marker)
    if idx == -1:
        return []
    conv_text = raw_text[idx + len(marker):]
    parts = re.split('(?:^|[\\t\\n])\\s*(User|Assistant)\\s*:\\s*', conv_text)
    turns = []
    i = 1
    while i + 1 < len(parts):
        role = parts[i].strip()
        content = parts[i + 1].strip().rstrip('\t').rstrip('\n').strip()
        if content:
            from_tag = 'human' if role == 'User' else 'gpt'
            turns.append({'from': from_tag, 'value': content})
        i += 2
    return turns

def ensure_valid_turns(turns: list[dict]) -> list[dict]:
    if not turns:
        return []
    while turns and turns[0]['from'] != 'human':
        turns.pop(0)
    while turns and turns[-1]['from'] != 'gpt':
        turns.pop()
    if len(turns) < 2:
        return []
    valid = [turns[0]]
    for turn in turns[1:]:
        if turn['from'] == valid[-1]['from']:
            valid[-1]['value'] += '\n' + turn['value']
        else:
            valid.append(turn)
    while valid and valid[-1]['from'] != 'gpt':
        valid.pop()
    return valid if len(valid) >= 2 else []

def convert_dataset(input_path: str, output_path: str) -> dict:
    with open(input_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    converted = []
    skipped = 0
    turn_counts = []
    for i, raw_text in enumerate(raw_data):
        system_prompt = build_system_prompt(raw_text)
        turns = parse_conversation(raw_text)
        turns = ensure_valid_turns(turns)
        if not turns:
            skipped += 1
            continue
        conversations = [{'from': 'system', 'value': system_prompt}]
        conversations.extend(turns)
        converted.append({'conversations': conversations})
        turn_counts.append(len(turns))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(converted, f, indent=2, ensure_ascii=False)
    stats = {'input_samples': len(raw_data), 'output_samples': len(converted), 'skipped': skipped, 'avg_turns': sum(turn_counts) / len(turn_counts) if turn_counts else 0, 'min_turns': min(turn_counts) if turn_counts else 0, 'max_turns': max(turn_counts) if turn_counts else 0, 'turn_distribution': dict(Counter(turn_counts))}
    return stats

def main():
    base_dir = Path(__file__).resolve().parent.parent
    input_train = base_dir / 'data' / 'debugging_train.json'
    input_test = base_dir / 'data' / 'debugging_test.json'
    output_train = base_dir / 'data' / 'sft' / 'socratic_train.json'
    output_test = base_dir / 'data' / 'sft' / 'socratic_test.json'
    print('=' * 60)
    print('Converting Socratic Debugging Dataset to ShareGPT Format')
    print('=' * 60)
    print(f'\n📂 Input:  {input_train}')
    print(f'📂 Output: {output_train}')
    train_stats = convert_dataset(str(input_train), str(output_train))
    print(f'\n Training data converted:')
    print(f'   Input samples:  {train_stats['input_samples']}')
    print(f'   Output samples: {train_stats['output_samples']}')
    print(f'   Skipped:        {train_stats['skipped']}')
    print(f'   Avg turns:      {train_stats['avg_turns']:.1f}')
    print(f'   Turn range:     {train_stats['min_turns']}-{train_stats['max_turns']}')
    print(f'\n📂 Input:  {input_test}')
    print(f'📂 Output: {output_test}')
    test_stats = convert_dataset(str(input_test), str(output_test))
    print(f'\n Test data converted:')
    print(f'   Input samples:  {test_stats['input_samples']}')
    print(f'   Output samples: {test_stats['output_samples']}')
    print(f'   Skipped:        {test_stats['skipped']}')
    print(f'   Avg turns:      {test_stats['avg_turns']:.1f}')
    print(f'   Turn range:     {test_stats['min_turns']}-{test_stats['max_turns']}')
    print('\n' + '=' * 60)
    print('Sample verification (first converted sample):')
    print('=' * 60)
    with open(str(output_train), 'r', encoding='utf-8') as f:
        data = json.load(f)
    if data:
        sample = data[0]
        print(f'\nSystem prompt (first 200 chars):')
        print(f'  {sample['conversations'][0]['value'][:200]}...')
        print(f'\nNumber of turns: {len(sample['conversations']) - 1}')
        for turn in sample['conversations'][1:]:
            role = '👤 Student' if turn['from'] == 'human' else '🤖 Tutor'
            text = turn['value'][:100]
            print(f'  {role}: {text}...')
if __name__ == '__main__':
    main()
