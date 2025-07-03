from math_verify import parse, verify
import pandas as pd
import os
import re
from collections import Counter
from latex2sympy2_extended import latex2sympy
from sympy import latex, simplify
import signal
import numpy as np
from timeout_decorator import timeout, TimeoutError

def _timeout_handler(signum, frame):
    raise TimeoutError

def check_duplication(input_str, thres=10):
    count = dict(Counter(input_str.split()))
    dup_w_count = sum([1 for k,v in count.items() if v >30])
    return dup_w_count > thres
    
def check_sentence_num(source_str, target_str, thres=10):
    source_s_count = len(source_str.split('.'))
    target_s_count = len(target_str.split('.'))
    return abs(target_s_count-source_s_count) > thres

def check_answer_len(source_str, target_str, thres=1000):
    return abs(len(source_str) - len(target_str)) > thres

def max_duplicated(input_str, thres=50):
    count = dict(Counter(input_str.split()))
    return any([True for k,v in count.items() if v > thres])

def match_digit_num(source_str, target_str, thres=10):
    source_n_count = len([1 for _ in source_str if _.isdigit()])
    target_n_count = len([1 for _ in target_str if _.isdigit()])
    return abs(target_n_count-source_n_count) > thres

@timeout(3)
def latex_expressions_equal(solution: str, answer: str) -> str:
    try:
        sympy_solution, sympy_answer = latex2sympy(solution), latex2sympy(answer)
        simplified_solution, simplified_answer = simplify(sympy_solution), simplify(sympy_answer)
        return latex(simplified_solution) == latex(simplified_answer)
    except (TimeoutError, Exception):
        return False

def safe_latex_equal(a, b):
    return latex_expressions_equal(a, b)

def mcqa_formatting(question, answer):
    number_list = ["\n1. ", "\n2. ", "\n3. ", "\n4. ", "\n5. "]
    check = {num[1]: question.split(num)[-1].split("\n")[0].strip() for num in number_list if num in question}
    
    try:
        answer_choice = float(answer)
        answer_content = check[str(answer)]
    except:
        answer_content = answer
        for k, v in check.items():
            if v == answer_content:
                answer_choice = float(k)

    return answer_choice, answer_content

@timeout(3)
def answer_in_last_sentence(input_string, answer):
    last_sentence = str(input_string).strip().split('\n')[-1]
    numbers_in_last_sentence = [float(num) for num in re.findall(r'\d+\.?\d*', last_sentence)]
    if len(numbers_in_last_sentence) > 0:
        return float(answer) == numbers_in_last_sentence[-1]
    else:
        return False
    
def convert_to_int_safe(string_number):
    cleaned = str(string_number).replace(',', '')
    if re.fullmatch(r'\d+(\.\d+)?', cleaned): 
        return float(cleaned)
    else:
        return string_number

@timeout(3)
def parse_boxed_value(text,answer):
    match = re.search(r'\\boxed\{\s*([\d,]+(?:\.\d+)?)\s*\}', str(text))
    if match:
        pred = convert_to_int_safe(match.group(1))
        c1 = float(pred) == answer
        c2 = str(pred) == str(answer)
        return any([c1,c2])
    return False

def parse_boxed_content_value(text, answer):
    match = re.search(r'\\boxed\{([a-zA-Z0-9가-힣\=\+\-\*/\^\_\(\)\{\}\[\]\\ ]+)\}', str(text))
    if match:
        return any([str(answer) == str(match.group(1)), safe_latex_equal(str(match.group(1)), str(answer))])
    return False

@timeout(3)
def parse_mcqa_value(question, text, answer):
    answer_choice, answer_content = mcqa_formatting(question, answer)

    return any([answer_in_last_sentence(text, answer_choice), parse_boxed_value(text, answer_choice), parse_boxed_content_value(text, answer_content)])

def safe_latex_equal(a: str, b: str, timeout_sec: int = 2) -> bool:
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_sec)
    try:
        result = safe_latex_equal(a, b)
    except TimeoutError:
        result = False
    except Exception:
        result = False
    finally:
        signal.alarm(0)
    return result

@timeout(3)
def parse_ksm_value(question, text, answer):
    if all(tag in question for tag in ["1.", "2.", "3.", "4."]):
        return parse_mcqa_value(question, text, answer)
    try:
        answer_f = float(answer)
        return any([answer_in_last_sentence(text, answer_f), parse_boxed_value(text, answer_f)])
    except ValueError:
        match = re.search(r'\\boxed\{([a-zA-Z0-9가-힣\=\+\-\*/\^\_\(\)\{\}\[\]\\ ]+)\}', str(text))
        if not match:
            return False
        else:
            boxed = match.group(1)
            return any([str(answer) == boxed, safe_latex_equal(boxed, str(answer))])


def scoring_func(score_type, prompt_id, n, output_path):
    file_list = [os.path.join(output_path, f) for f in os.listdir(output_path) if ".csv" in f]
    scores = {}

    for st in score_type:
        scores[st] = {}
        for file in file_list:
            k = file.split("/")[-1].replace(".csv", "")
            df = pd.read_csv(file)
            
            score_dict = {}
            for iteration in range(1, n+1):
                if st == "original":
                    if k in ["GSM8K", "MATH", "OMNI_MATH"]:
                        score = sum([1 for _,row in df.iterrows() if any([answer_in_last_sentence(row[f"solution_{iteration}"],row.answer),parse_boxed_value(row[f"solution_{iteration}"],row.answer)])])
                    elif k == "MMMLU":
                        score = sum([1 for _,row in df.iterrows() if any([parse_mcqa_value(row.question,row[f"solution_{iteration}"],row.answer)])])
                    elif k == "KSM":
                        score = sum([1 for _,row in df.iterrows() if parse_ksm_value(row.original,row[f"solution_{iteration}"],row.original_answer)]) if prompt_id == "en" else sum([1 for _,row in df.iterrows() if parse_ksm_value(row.question,row[f"solution_{iteration}"],row.answer)])

                elif st == "math_verify":
                    if k in ["GSM8K", "MATH", "OMNI_MATH"]:
                        score = sum([1 for _,row in df.iterrows() if verify(parse(str(row.answer)), parse(row[f"solution_{iteration}"]))])
                    elif k == "MMMLU":
                        score = sum([1 for _,row in df.iterrows() if any([parse_mcqa_value(row.question,row[f"solution_{iteration}"],row.answer)])])
                    elif k == "KSM":
                        score = sum([1 for _,row in df.iterrows() if verify(parse(str(row.original_answer)), parse(row[f"solution_{iteration}"]))]) if prompt_id == "en" else sum([1 for _,row in df.iterrows() if verify(parse(str(row.answer)), parse(row[f"solution_{iteration}"]))])

                if n == 1:
                    continue
                else:
                    score_dict[k][f"iteration_{iteration}"] = score / len(df) * 100
            
            if n == 1:
                scores[st][k] = sum(score) / len(df) * 100
            else:
                score_dict["average"] = np.mean([score_dict[f"iteration_{ite}"] for ite in range(1, n+1)])
                scores[st][k] = score_dict
            
    return scores