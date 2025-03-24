import os
import argparse
from score import scoring_func
from generate import generate_solution
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
import json

os.makedirs('results', exist_ok=True)
os.makedirs('score_results', exist_ok=True)

def eval_method_cls(eval_method, prompt_id):
    if eval_method == "normal":
        return prompt_id
    elif eval_method == "clp":
        return ["clp"]
    elif eval_method == "plug":
        return ["plug"]


def main(cats, model_name, prompt_id, eval_method, score_type, temperature, p):
    prompt_id = eval_method_cls(eval_method, prompt_id)

    # Load datasets
    dfs = {cat: pd.DataFrame(load_dataset('HAERAE-HUB/HRM8K', cat)['test']) for cat in cats}

    model_path = model_name.replace('/', '_')
    
    os.makedirs(f"results/temp_{str(temperature).replace('.', '_')}", exist_ok=True)
    os.makedirs(f"results/temp_{str(temperature).replace('.', '_')}/{model_path}", exist_ok=True)
    os.makedirs(f"score_results/temp_{str(temperature).replace('.', '_')}", exist_ok=True)
    os.makedirs(f"score_results/temp_{str(temperature).replace('.', '_')}/{model_path}", exist_ok=True)

    for pi in prompt_id:
        os.makedirs(f"results/temp_{str(temperature).replace('.', '_')}/{model_path}/{pi}", exist_ok=True)
        print(f"{model_name} - {prompt_id} Evaluation is starting..")

        results = generate_solution(pi, model_name, temperature, p, dfs)
        for k in results.keys():
            results[k].to_csv(f"results/temp_{str(temperature).replace('.', '_')}/{model_path}/{pi}/{k}.csv", index=False)
            
        scores = scoring_func(score_type, pi, f"results/temp_{str(temperature).replace('.', '_')}/{model_path}/{pi}")
        print("----------------------- Score Board -----------------------")
        for key in scores.keys():
            print(f"{key}: {scores[key]}")
        print("-----------------------------------------------------------")
        
        with open(f"score_results/temp_{str(temperature).replace('.', '_')}/{model_path}/{pi}.json", "w") as f:
            json.dump(scores, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate predictions and save results to CSV.")
    parser.add_argument('--cats', nargs='+', default=['MATH', 'GSM8K', 'OMNI_MATH', "MMMLU", "KSM"], help="List of dataset categories to process, separated by spaces.")
    parser.add_argument('--model_name', type=str, default='gpt-4o', help="Name of the model to use for generating predictions.")
    parser.add_argument('--prompt_id', nargs="+", default=["default"], help="Prompt to use for eval.")
    parser.add_argument('--temperature', type=float, default=0.0, help="Temperature value used for model inference")
    parser.add_argument('--p', type=float, default=0.95, help="Top-p value used for model inference")
    parser.add_argument('--max_tokens', type=int, default=8192, help="Max generation tokens for model")
    parser.add_argument('--eval_method', type=str, default="normal", help="Evaluation method")
    parser.add_argument('--score_type', nargs="+", default=["original", "math_verify"], help="Scoring type")
    args = parser.parse_args()
    main(args.cats, args.model_name, args.prompt_id, args.eval_method, args.score_type, args.temperature, args.p)