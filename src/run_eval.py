import os
from prompts import prompts
from score import scoring_func
from generate import generate_solution
from datasets import load_dataset
from typing import List
import pandas as pd
import argparse
import yaml
import json


os.makedirs('results', exist_ok=True)
os.makedirs('score_results', exist_ok=True)


def str2bool(v):
    if v.lower() in ['true']:
        return True
    elif v.lower() in ['false']:
        return False
    else:
        raise ValueError(f"The given '{v}' is not supported! Choose between ['true', 'false']")


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="The model name to evaluate")
    parser.add_argument("--n", type=int, help="The number of iteration")
    parser.add_argument("--temperature", type=float, help="The generation hyperparameter for `temperature`")
    parser.add_argument("--top_p", type=float, help="The generation hyperparameter for `top_p`. This value will be ignored when `terperature` set to 0.")
    parser.add_argument("--max_tokens", type=int, help="The hyperparameter for `max_tokens`")
    parser.add_argument("--reasoning", type=str2bool, help="The key to trigger the long thinking mode of language model. (e.g., Qwen3)")
    parser.add_argument("--batch", type=str2bool, help="Option for batch generation. Only supported for litellm models.")
    parser.add_argument("--subsets", nargs='+')
    parser.add_argument("--prompt_id", nargs='+')
    parser.add_argument("--score_type", nargs='+')
    return parser.parse_args()


def load_config(
        path: str
):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(
        subsets: List[str],
        model_name: str,
        prompt_id: List[str],
        reasoning: bool,
        score_type: List[str],
        n: int,
        temperature: float,
        top_p: float,
        max_tokens: int,
        batch: bool
):
    if temperature == 0.0 and n > 1:
        raise ValueError(f"You have tried {n} iterations with temperature of 0.0! You should ser temperature over than 0.0!")
    
    print("---------------Evaluation Setup---------------")
    print("Model:", model_name)
    print("\nGeneration Parameters:")
    print("  - n:", n)
    print("  - temperature:", temperature)
    print("  - top_p:", top_p)
    print("  - max_tokens:", max_tokens)
    print("  - reasoning:", reasoning)
    print("  - batch:", batch)
    print("\nSubsets:", subsets)
    print("Prompt ID:", prompt_id)
    print("Score Type:", score_type)

    dfs = {subset: pd.DataFrame(load_dataset('HAERAE-HUB/HRM8K', subset)['test']) for subset in subsets}

    for p in prompt_id:
        if p not in prompts.keys():
            continue
        pi = p
        model_path = model_name.replace('/', '_')
        os.makedirs(f"results/{pi}", exist_ok=True)
        os.makedirs(f"score_results/{pi}", exist_ok=True)
        os.makedirs(f"results/{pi}/{model_path}", exist_ok=True)
        print(f"{model_name} - {prompt_id} Evaluation is starting..")

        try:
            results = generate_solution(pi, model_name, reasoning, n, temperature, top_p, max_tokens, dfs, batch)
            for k in results.keys():
                results[k].to_csv(f"results/{pi}/{model_path}/{k}.csv", index=False)
                
            scores = scoring_func(score_type, pi, n, f"results/{pi}/{model_path}")
            print("----------------------- Score Board -----------------------")
            for key in scores.keys():
                print(f"{key}: {scores[key]}")
            print("-----------------------------------------------------------")
            
            with open(f"score_results/{pi}/{model_path}.json", "w") as f:
                json.dump(scores, f, indent=4)
        except Exception as e:
            print(e)
            continue


if __name__ == "__main__":
    args = args_parse()
    main(args.subsets, args.model_name, args.prompt_id, args.reasoning, args.score_type, args.n, args.temperature, args.top_p, args.max_tokens, args.batch)