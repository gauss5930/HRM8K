from prompts import prompts, magistral_system
from models import load_model
from litellm import batch_completion, completion
from tqdm.auto import tqdm, trange
import litellm
from openai import OpenAI
import multiprocessing
import google.generativeai as genai
import cohere
import os
import time


litellm.drop_params=True
# litellm._turn_on_debug()


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

def safe_parse_gemini(text):
    try:
        return text.text
    except:
        return None

def process_request(task_args, max_retries=3):
    retry_count = 0
    while retry_count < max_retries:
        try:
            response = completion(**task_args)
            return response
        except Exception as e:
            retry_count += 1
            if retry_count >= max_retries:
                print(f"Rate limit error: Max retries ({max_retries}) reached. Aborting for this request. Error: {e}")
                return None
            
            # 10분(600초) 대기
            wait_time = 600
            print(f"Rate limit error encountered. Waiting for {wait_time/60:.0f} minutes before retrying... (Attempt {retry_count}/{max_retries})")
            print(f"Error details: {e}")
            time.sleep(wait_time)
    
    print(f"Failed to get a response after {max_retries} attempts.")
    return None


def multi_completion(tasks):
    num_processes = min(len(tasks), os.cpu_count(), 16)
    results = []
    with multiprocessing.Pool(processes=num_processes) as pool:
        pbar = tqdm(pool.imap(process_request, tasks), total=len(tasks))
        for result in pbar:
            results.append(result)
    return results


def gemini_retry(model, qry, generation_config, max_retry=3):
    retry_count = 0

    while retry_count < max_retry:
        retry_count += 1
        try:
            result = model.generate_content(qry, generation_config=generation_config)
            text = safe_parse_gemini(result)
            if text is None:
                raise ValueError("Empty / malformed model output")

            retry_count = 0
            return text

        except Exception as e:
            retry_count += 1
            wait_seconds = 5 * 60 if retry_count < 3 else 60 * 60
            print(
                f"[WARN] Generation failed ({e}). "
                f"{retry_count} consecutive error(s). "
                f"Sleeping {wait_seconds // 60} min and retrying..."
            )
            time.sleep(wait_seconds)

    print(f"Exceeded {max_retry} retries for a single prompt; aborting.")
    return None


def generate_queries(df, model_name, tokenizer, prompt_id, reasoning, temperature, top_p, max_tokens, litellm_models, gemini_models):
    qrys = []
    common_headers = {"HTTP-Referer": "https://github.com/BerriAI/litellm"}
    
    for _,row in df.iterrows():
        question = row.original if prompt_id == "en" else row.question
        msg = " ".join([question, prompts[prompt_id]]).strip()
        msg_prompt = [{"role": "system", "content": magistral_system}, {"role": "user", "content": msg}] if "magistral" in model_name.lower() else [{"role": "user", "content": msg}]

        if model_name in litellm_models:
            qry = {
                "model": model_name,
                "messages": msg_prompt,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "custom_header": common_headers,
                "num_retries": 3,
                "timeout": 600
            }
        elif model_name in gemini_models:
            qry = msg
        else:
            qry = tokenizer.apply_chat_template(msg_prompt, tokenize=False, add_generation_prompt=True, enable_thinking=reasoning)

        qrys.append(qry)

    return qrys


def generate_solution(prompt_id, model_name, reasoning, n, temperature, p, max_tokens, dfs, batch):
    litellm_models, gemini_models = [], []
    if os.environ.get("OPENAI_API_KEY") != None:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        litellm_models.extend([c.id for c in client.models.list()])
    if os.environ.get("GEMINI_API_KEY") != None:
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
        gemini_models.extend([m.name.replace("models/", "") for m in list(genai.list_models())])
    if os.environ.get("COHERE_API_KEY") != None:
        co = cohere.Client(os.environ.get("COHERE_API_KEY"))
        litellm_models.extend([m.name for m in co.models.list().models])
    if "openrouter" in model_name:
        litellm_models.append(model_name)
    
    df_results = {}
    if (model_name not in litellm_models) and (model_name not in gemini_models):
        model, tokenizer, params = load_model(model_name, temperature, p, max_tokens)
    else:
        model, tokenizer, params = model_name, None, None

    for k, df in tqdm(dfs.items(), total=len(dfs), desc="Processing subsets"):
        prompts_args = generate_queries(df, model_name, tokenizer, prompt_id, reasoning, temperature, p, max_tokens, litellm_models=litellm_models, gemini_models=gemini_models)
        for iteration in trange(n):
            print(f"{model} - {k} Generation #{iteration+1} Start!")
            outputs = []
            if model_name in litellm_models:
                if batch == True:
                    responses = batch_completion(model=model_name, messages=prompts_args, temperature=temperature, top_p=p, max_tokens=max_tokens)
                    outputs = [safe_parse_litellm(resp) for resp in responses]
                else:
                    responses = multi_completion(prompts_args)
                    outputs = [safe_parse_litellm(resp) for resp in responses]
            elif model_name in gemini_models:
                model = genai.GenerativeModel(model_name)
                generation_config = genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                    top_p=p
                )
                for qry in tqdm(prompts_args, desc=f"Generating with {model_name}"):
                    response = gemini_retry(model, qry, generation_config=generation_config)
                    outputs.append(response)
            else:
                responses = model.generate(prompts_args, params)
                outputs = [safe_parse_vllm(resp) for resp in responses]
            
            df[f"solution_{iteration+1}"] = outputs
                
            df_results[k] = df

    return df_results