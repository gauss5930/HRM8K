from prompts import prompts
from models import load_model
from litellm import batch_completion, completion
from tqdm.auto import tqdm
import litellm
from openai import OpenAI
import google.generativeai as genai
import torch
import os


litellm.drop_params=True


def safe_parse_litellm(text):
    try:
        return text.choices[0].message.content
    except:
        return None


def safe_parse_vllm(text):
    try:
        return text.outputs[0].text
    except:
        return None


def generate_queries(df, model_name, tokenizer, prompt_id, reasoning, litellm_models):
    qrys = []
    
    for _,row in df.iterrows():
        question = row.original if prompt_id == "en" else row.question
        msg = " ".join([question, prompts[prompt_id]]).strip()

        if model_name not in litellm_models:
            qry = tokenizer.apply_chat_template([{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True, enable_thinking=reasoning)
        else:
            qry = [{"role": "user", "content": msg}]

        qrys.append(qry)

    return qrys


def generate_solution(prompt_id, model_name, reasoning, temperature, p, max_tokens, dfs, batch):
    litellm_models = []
    if os.environ.get("OPENAI_API_KEY") != None:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        litellm_models.extend([c.id for c in client.models.list()])
    if os.environ.get("GEMINI_API_KEY") != None:
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
        litellm_models.extend([m.name.replace("models/", "gemini/") for m in list(genai.list_models())])
    if "openrouter" in model_name:
        litellm_models.append(model_name)
    
    df_results = {}
    if model_name not in litellm_models:
        model, tokenizer, params = load_model(model_name, temperature, p, max_tokens)
    else:
        model, tokenizer, params = model_name, None, None

    for k, df in tqdm(dfs.items(), total=len(dfs)):
        prompts = generate_queries(df, model_name, tokenizer, prompt_id, reasoning, litellm_models=litellm_models)
        if model_name in litellm_models:
            if batch == True:
                responses = batch_completion(model=model_name, messages=prompts, temperature=temperature, top_p=p, max_tokens=max_tokens)
                outputs = [safe_parse_litellm(resp) for resp in responses]
            else:
                outputs = []
                for qry in tqdm(prompts):
                    response = completion(model=model_name, messages=qry, temperature=temperature, top_p=p, max_tokens=max_tokens)
                    outputs.append(safe_parse_litellm(response))
        else:
            responses = model.generate(prompts, params)
            outputs = [safe_parse_vllm(output) for output in responses]

        df["solution"] = outputs
            
        df_results[k] = df

    if model:
        del model
    if tokenizer:
        del tokenizer
        torch.cuda.empty_cache()

    return df_results
    