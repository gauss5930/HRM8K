import pandas as pd
import os
from score import parse_boxed_value, answer_in_last_sentence, parse_mcqa_value, parse_ksm_value
import json
import yaml
from typing import List
import argparse
from tqdm.auto import tqdm


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
    for subs in tqdm(subsets):
        subset = subs.split("/")[-1].replace(".csv", "")
        df_result = pd.read_csv(subs)
        checks = []
        if subset in ["GSM8K", "MATH", "OMNI_MATH"]:
            score = sum([1 for _,row in df_result.iterrows() if any([answer_in_last_sentence(row.solution,row.answer), parse_boxed_value(row.solution,row.answer)])]) / len(df_result) * 100
            for _,row in tqdm(df_result.iterrows(), total=len(df_result)):
                checks.append(check_func(any([answer_in_last_sentence(row.solution, row.answer), parse_boxed_value(row.solution, row.answer)])))
        elif subset == "MMMLU":
            score = sum([1 for _,row in df_result.iterrows() if any([parse_mcqa_value(row.question,row.solution,row.answer)])]) / len(df_result) * 100
            for _,row in tqdm(df_result.iterrows(), total=len(df_result)):
                checks.append(check_func(parse_mcqa_value(row.question, row.solution, row.answer)))
        elif subset == "KSM":
            score = sum([1 for _,row in df_result.iterrows() if parse_ksm_value(row.original,row.solution,row.original_answer)]) if prompt_id == "en" else sum([1 for _,row in df_result.iterrows() if parse_ksm_value(row.question,row.solution,row.answer)])
            score = score / len(df_result) * 100
            for _,row in tqdm(df_result.iterrows(), total=len(df_result)):
                checks.append(check_func(parse_ksm_value(row.question, row.solution, row.answer)) if prompt_id == "en" else check_func(parse_ksm_value(row.question, row.solution, row.answer)))
        sub_scores[subset] = score
        df_result["check"] = checks
        df_result.to_csv(os.path.join("check_results", "/".join(subs.split("/")[1:-1]), f"{subset}_check.csv"), index=False)

    return sub_scores


def main(
        check_type: str,
        subsets: List[str],
        model_list: List[str],
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--check_type", type=str, default="config", help="The type of result check. [`all`, `config`]")
    parse_args = parser.parse_args()
    yaml_args = load_config("eval_config.yaml")
    main(parse_args.check_type, yaml_args["subsets"], yaml_args["models"], yaml_args["prompt_id"])