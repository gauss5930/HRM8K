import pandas as pd
import os
from score import parse_boxed_value, answer_in_last_sentence, parse_mcqa_value, parse_ksm_value
import json
import yaml
from typing import List
import argparse
from tqdm.auto import tqdm
import numpy as np
# [수정] timeout_decorator에서 TimeoutError도 import
from timeout_decorator import TimeoutError


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check_type", type=str, required=True)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--subsets", nargs="+", default=None)
    parser.add_argument("--prompt_id", nargs="+", default=None)
    return parser.parse_args()


def load_config(
        path: str
):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def check_func(
        checking: bool
):
    if checking:
        return "O"
    else:
        return "X"
    

def result_check(
        prompt_id: str,
        subsets: List[str]
):
    sub_scores = {}
    for subs in tqdm(subsets, desc="Processing subsets"):
        subset = subs.split("/")[-1].replace(".csv", "")
        if os.path.exists(subs):
            df_result = pd.read_csv(subs)
            iterations = len([l for l in list(df_result.keys()) if "solution" in l])

            score_dict = {}
            for it in range(1, iterations+1):
                checks, score = [], []
                
                for idx, row in tqdm(df_result.iterrows(), total=len(df_result), desc=f"Iter {it} on {subset}", leave=False):
                    try:
                        is_correct = False
                        if subset in ["GSM8K", "MATH", "OMNI_MATH"]:
                            is_correct = any([
                                answer_in_last_sentence(row[f"solution_{it}"], row.answer),
                                parse_boxed_value(row[f"solution_{it}"], row.answer)
                            ])
                        elif subset == "MMMLU":
                            is_correct = parse_mcqa_value(row.question, row[f"solution_{it}"], row.answer)
                        elif subset == "KSM":
                            if prompt_id == "en":
                                is_correct = parse_ksm_value(row.original, row[f"solution_{it}"], row.original_answer)
                            else:
                                is_correct = parse_ksm_value(row.question, row[f"solution_{it}"], row.answer)
                        
                        checks.append(check_func(is_correct))
                        if is_correct:
                            score.append(1)
                    
                    except TimeoutError:
                        checks.append("T") # Timeout
                    except Exception as e:
                        print(f"Error on row {idx} in {subset}: {e}")
                        checks.append("E") # Error
                
                score_dict[f"iteration_{it}"] = (sum(score) / len(df_result) * 100) if len(df_result) > 0 else 0
                df_result[f"check_{it}"] = checks
            
            score_dict["average"] = np.mean([score_dict[f"iteration_{it}"] for it in range(1, iterations+1)]) if iterations > 0 else 0
            sub_scores[subset] = score_dict
            
            output_dir = os.path.join("check_results", *subs.split("/")[1:-1])
            os.makedirs(output_dir, exist_ok=True)
            df_result.to_csv(os.path.join(output_dir, f"{subset}_check.csv"), index=False)
        else:
            print(f"{subs} does not exist!")
            sub_scores[subset] = None

    return sub_scores


def main(
        check_type: str,
        model_list: List[str],
        subsets: List[str],
        prompt_id: List[str]
):
    os.makedirs("check_results", exist_ok=True)

    scores = {}

    prompt_id = prompt_id if check_type == "config" else [f for f in os.listdir("results") if "." not in f]

    for pi in prompt_id:
        scores[pi] = {}
        os.makedirs(f"check_results/{pi}", exist_ok=True)
        model_list = model_list if check_type == "config" else os.listdir(f"results/{pi}")
        for model_name in model_list:
            model_path = model_name.replace("/", "_")
            os.makedirs(f"check_results/{pi}/{model_path}", exist_ok=True)
            check_subsets = [f"results/{pi}/{model_path}/{s}.csv" for s in subsets] if check_type == "config" else [os.path.join(f"results/{pi}/{model_path}", f) for f in os.listdir(f"results/{pi}/{model_path}") if ".csv" in f]
            print(check_subsets)
            scores[pi][model_name] = result_check(pi, check_subsets)

    with open(f"check_score.json", "w") as f:
        json.dump(scores, f, indent=4)


if __name__ == "__main__":
    args = args_parse()
    main(args.check_type, args.models, args.subsets, args.prompt_id)