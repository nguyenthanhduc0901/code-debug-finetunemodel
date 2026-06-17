from judgeLib.compile import compile
from judgeLib.file import readFile, writeFile
from judgeLib.run import run, runPy, runJava
import os
import re

def compare_output_and_answer(output, answer, tolerance=1e-4):
    # Split output and answer into parts
    output_parts = output.strip().split()
    answer_parts = answer.strip().split()

    # Check if the number of parts is the same
    if len(output_parts) != len(answer_parts):
        return False

    # Define the regular expression pattern for floating point numbers
    float_pattern = r'^[-+]?\d+\.\d+$'

    # Compare parts one by one
    for i in range(len(output_parts)):
        # Use regular expression to check if it's a floating point number
        if re.match(float_pattern, output_parts[i]) or re.match(float_pattern, answer_parts[i]):
            try:
                output_float = float(output_parts[i])
                answer_float = float(answer_parts[i])
                if abs(output_float - answer_float) > tolerance:
                    return False
            except ValueError:
                return False
        else:
            # If it's not a floating point number, compare the strings directly
            if output_parts[i] != answer_parts[i]:
                return False

    return True

# this func return the result of judging and IO and expect of the program
def judge(file_dir: str = None, code_text: str = None, language: str = None, input_dir: str = None, answer_dir: str = None, timeLimit: float = 1.0, memoryLimit: int = 512, test_id: int = 0, cleanup: bool = True):
    res = ''
    exe_dir = ''
    class_dir = ''
    input = ''
    output = ''
    answer = ''

    need_delete = False
    if file_dir is None:
        if language is None:
            res = 'language is needed'
        if code_text is None:
            res = 'code text is needed'
        if res == '':
            if language == 'java':
                file_dir = 'Main.java'
            else:
                file_dir = 'temp.' + language
            writeFile(file_dir, code_text)
            need_delete = True

    is_java = file_dir is not None and file_dir.endswith('.java')
    orig_file_dir = file_dir

    if is_java:
        file_directory = os.path.dirname(file_dir)
        new_file_dir = os.path.join(file_directory, "Main.java")
        if os.path.exists(file_dir) and file_dir != new_file_dir:
            os.rename(file_dir, new_file_dir)
            file_dir = new_file_dir

    # compile
    if res == '':
        exe_dir = os.path.splitext(file_dir)[0] + '.exe'
        class_dir = os.path.splitext(file_dir)[0] + '.class'
        
        need_compile = True
        if file_dir.endswith('.py'):
            need_compile = False
        elif test_id > 0:
            if file_dir.endswith('.cpp') and os.path.exists(exe_dir):
                need_compile = False
            elif file_dir.endswith('.java') and os.path.exists(class_dir):
                need_compile = False
                
        if need_compile:
            compileSuccess = compile(file_dir, test_id)
            if not compileSuccess:
                res = 'CE'

    if res == '':
        # read input and answer
        input = readFile(input_dir)
        answer = readFile(answer_dir)
        if input == 'Error: file not found':
            res = 'input not found'
        if answer == 'Error: file not found':
            res = 'answer not found'

    # run
    if res == '':
        # print(file_dir)
        # print('exe_dir:', exe_dir)
        if os.path.exists(exe_dir):
            output = run(exe_dir, input, timeLimit, memoryLimit)
        elif file_dir.endswith('.py'):
            output = runPy(file_dir, input, timeLimit, memoryLimit, test_id)
        elif file_dir.endswith('.java') and os.path.exists(class_dir):
            output = runJava(file_dir, input, timeLimit, memoryLimit)
        else:
            # print('False')
            output = 'CE'

        if output == 'TLE':
            res = 'TLE'
            output = ''
        if output == 'MLE':
            res = 'MLE'
            output = ''
        if output == 'RE':
            res = 'RE'
            output = ''
        if output == 'CE':
            res = 'CE'
            output = ''

    file_directory = os.path.dirname(file_dir) if file_dir else '.'
    txt_dir = os.path.join(file_directory, f'temp_{test_id}.txt')
    if res == '':
        # wash the output
        writeFile(txt_dir, output)
        output = readFile(txt_dir)

        if compare_output_and_answer(output, answer)==True:
            res = 'AC'
        else:
            res = 'WA'

    # clean up the file
    if cleanup:
        if os.path.exists(exe_dir):
            os.remove(exe_dir)
        if os.path.exists(class_dir):
            os.remove(class_dir)
        if os.path.exists(file_dir) and need_delete:
            os.remove(file_dir)
    if os.path.exists(txt_dir):
        os.remove(txt_dir)

    if is_java and orig_file_dir != file_dir:
        if os.path.exists(file_dir):
            os.rename(file_dir, orig_file_dir)

    return res, input, output, answer
