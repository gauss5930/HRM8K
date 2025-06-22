from prompts import prompts
from models import load_model
from litellm import batch_completion, completion
from tqdm.auto import tqdm
import litellm
from openai import OpenAI
import google.generativeai as genai
import cohere
import torch
import os
import time
import asyncio
import gc
from vllm.utils import random_uuid


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


def litellm_retry(model, messages, temperature, top_p, max_tokens, max_retry=3):
    retry_count = 0

    while retry_count < max_retry:
        retry_count += 1
        try:
            result = completion(model=model, messages=messages, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
            text = safe_parse_litellm(result)
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


async def generate_single_vllm_request(model, prompt, params):
    """
    vLLM의 비동기 제너레이터를 소비하여 최종 결과만 반환하는 코루틴.
    """
    request_id = random_uuid()
    results_generator = model.generate(prompt, params, request_id)
    
    final_output = None
    async for request_output in results_generator:
        final_output = request_output
        
    return final_output


def generate_queries(df, model_name, tokenizer, prompt_id, reasoning, litellm_models, gemini_models):
    qrys = []
    
    for _,row in df.iterrows():
        question = row.original if prompt_id == "en" else row.question
        msg = " ".join([question, prompts[prompt_id]]).strip()

        if model_name in litellm_models:
            qry = [{"role": "user", "content": msg}]
        elif model_name in gemini_models:
            qry = msg
        else:
            qry = tokenizer.apply_chat_template([{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True, enable_thinking=reasoning)

        qrys.append(qry)

    return qrys


async def generate_solution(prompt_id, model_name, reasoning, temperature, p, max_tokens, dfs, batch):
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
        model, tokenizer, params = await load_model(model_name, temperature, p, max_tokens)
    else:
        model, tokenizer, params = model_name, None, None

    for k, df in tqdm(dfs.items(), total=len(dfs)):
        prompts = generate_queries(df, model_name, tokenizer, prompt_id, reasoning, litellm_models=litellm_models, gemini_models=gemini_models)
        outputs = []
        if model_name in litellm_models:
            if batch == True:
                responses = batch_completion(model=model_name, messages=prompts, temperature=temperature, top_p=p, max_tokens=max_tokens)
                outputs = [safe_parse_litellm(resp) for resp in responses]
            else:
                for qry in tqdm(prompts):
                    response = litellm_retry(model_name, qry, temperature, p, max_tokens)
                    outputs.append(response)
        elif model_name in gemini_models:
            model = genai.GenerativeModel(model_name)
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
                top_p=p
            )
            for qry in tqdm(prompts):
                response = gemini_retry(model, qry, generation_config=generation_config)
                outputs.append(response)
        else:
            tasks = [generate_single_vllm_request(model, qry, params) for qry in prompts]
            results = await asyncio.gather(*tasks)
            outputs = [safe_parse_vllm(result) for result in results]

        df["solution"] = outputs
            
        df_results[k] = df

    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    asyncio.sleep(5)

    return df_results
    