import json
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Calculate Code Repair pass@k accuracy')
parser.add_argument('--result_path', default='output/eval_results/code_repair/python_result.jsonl', help='Path to OJ judge result JSONL')
parser.add_argument('--data_path', default='Data/eval/debugevalsuite_task3.jsonl', help='Path to eval JSONL data (supports python/java/cpp via --lang)')
parser.add_argument('--lang', default='python', choices=['python', 'java', 'cpp'], help='Language to evaluate')
args_parsed = parser.parse_args()

result_path = args_parsed.result_path
data_path = args_parsed.data_path
lang = args_parsed.lang

if lang == 'python':
    panduan = 'responses_success'
elif lang in ('java', 'cpp'):
    panduan = 'result'

question_error = {}

dict = {}

def estimate_pass_at_k(num_samples, num_correct, k):
    """Estimates pass@k of each problem and returns them in an array."""

    def estimator(n: int, c: int, k: int) -> float:
        """Calculates 1 - comb(n - c, k) / comb(n, k)."""
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    import itertools

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array(
        [estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)]
    )

def calculate_pass_at_k(results, k_list=[1, 5]):
    # Calculate pass@k.
    total, correct = [], []
    for result in results:
        passed = [result]
        total.append(len(passed))
        correct.append(sum(passed))
    total = np.array(total)
    correct = np.array(correct)

    ks = k_list
    pass_at_k = {f"pass@{k}": estimate_pass_at_k(total, correct, k).mean()
                 for k in ks if (total >= k).all()}
    # print(pass_at_k)
    return pass_at_k

def read_jsonl_file(path):
    data = []
    with open(path,'r',encoding='utf8') as f:
        for line in f:
            item = json.loads(line)
            if item['language'] == lang:
                data.append(item)
    return data

results = read_jsonl_file(result_path)
datas = read_jsonl_file(data_path)

for elem in datas:
    question_error[elem['question_id']] = elem['major_error_type']

acc = []

for i,line in enumerate(results):
    error = question_error[line['question_id']]
    if line[panduan] == 'True':
        acc.append(1)
        if error not in dict.keys():
            dict[error] = []
        dict[error].append(1)
    else:
        acc.append(0)
        if error not in dict.keys():
            dict[error] = []
        dict[error].append(0)



print(f'average:{calculate_pass_at_k(acc)}')
print(f'syntax:{calculate_pass_at_k(dict["syntax error"])}')
print(f'reference:{calculate_pass_at_k(dict["reference error"])}')
print(f'logic:{calculate_pass_at_k(dict["logic error"])}')
print(f'multiple:{calculate_pass_at_k(dict["multiple error"])}')

