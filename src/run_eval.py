import os
from openai import OpenAI
import google.generativeai as genai
from score import scoring_func
from generate import generate_solution
from datasets import load_dataset
from models import thinking_model_list
from typing import List
import pandas as pd
import yaml
import json
import asyncio

os.makedirs('results', exist_ok=True)
os.makedirs('score_results', exist_ok=True)


def load_config(
        path: str
):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    

def thinking_model_check(model_name, pi, reasoning, thinking_model_list=thinking_model_list):
    for model in thinking_model_list:
        if model_name in model:
            if reasoning == True:
                return pi + "_reasoning"
            else:
                return pi
    return pi


def main(
        subsets: List[str],
        model_list: List[str],
        prompt_id: List[str],
        reasoning: bool,
        score_type: List[str],
        temperature: float,
        top_p: float,
        max_tokens: int,
        batch: bool
):
    dfs = {subset: pd.DataFrame(load_dataset('HAERAE-HUB/HRM8K', subset)['test']) for subset in subsets}

    for p in prompt_id:
        for model_name in model_list:
            pi = thinking_model_check(model_name, p, reasoning)
            model_path = model_name.replace('/', '_')
            os.makedirs(f"results/{pi}", exist_ok=True)
            os.makedirs(f"score_results/{pi}", exist_ok=True)
            os.makedirs(f"results/{pi}/{model_path}", exist_ok=True)
            print(f"{model_name} - {prompt_id} Evaluation is starting..")

            results = asyncio.run(generate_solution(pi, model_name, reasoning, temperature, top_p, max_tokens, dfs, batch))
            for k in results.keys():
                results[k].to_csv(f"results/{pi}/{model_path}/{k}.csv", index=False)
                
            scores = scoring_func(score_type, pi, f"results/{pi}/{model_path}")
            print("----------------------- Score Board -----------------------")
            for key in scores.keys():
                print(f"{key}: {scores[key]}")
            print("-----------------------------------------------------------")
            
            with open(f"score_results/{pi}/{model_path}.json", "w") as f:
                json.dump(scores, f, indent=4)


if __name__ == "__main__":
    args = load_config("eval_config.yaml")
    main(args["subsets"], args["models"], args["prompt_id"], args["reasoning"], args["score_type"], args["temperature"], args["top_p"], args["max_tokens"], args["batch"])