from prompts import prompts
from models import litellm_models, load_model
from litellm import batch_completion
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm.auto import tqdm
from jinja2.exceptions import TemplateError

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

def query_mapping(prompt_id):
    if prompt_id in ["k2k", "e2k"]:
        return prompts["ko"]
    elif prompt_id in ["e2e", "k2e"]:
        return prompts["en"]
    elif prompt_id == "plug":
        return ""
    else:
        return prompts[prompt_id]

def generate_queries_clp(df, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    qrys = []

    for _,row in df.iterrows():
        msg = [
            {"role": "system", "content": ""},
            {"role": "user", "content": prompts["clp_alignment"].replace("<source_lang>", "Korean").replace("<input_sentence>", row.question).replace("<target_lang>", "English")}
        ]
        if "alignment" in df.keys():
            msg += [
                {"role": "assistant", "content": row.alignment},
                {"role": "user", "content": prompts["clp_solve"].replace("<target_lang>", "English")}
            ]
        qrys.append(tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))
        
    return qrys

def generate_queries(df, model_name, prompt_id):
    try:        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except:
        tokenizer = ""
    qrys = []
    system_message = prompts["plug_system"] if prompt_id == "plug" else ""
    
    for _,row in df.iterrows():
        question = row.original if prompt_id in ["en",'oasst_en', 'e2e', 'e2k'] else row.question
        msg = prompts[prompt_id].replace("{instruction}", question) if ('oasst' in prompt_id) or ('qalign' in prompt_id) else " ".join([question, query_mapping(prompt_id)]).strip()
        if prompt_id in ["k2k", "k2e", "e2k", "e2e"]:
            msg = "".join([prompts["question_instruction"], msg])

        if model_name not in litellm_models:
            try:
                if "qalign" in prompt_id:
                    qry = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                else:
                    if system_message:
                        messages = [{"role": "system", "content": system_message}]
                    else:
                        messages = []
                    messages.append({"role": "user", "content": msg})
                    qry = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except TemplateError as e:
                if str(e) == 'System role not supported':
                    messages = [
                        {"role": "user", "content": system_message + '\n\n'+ msg}
                    ]
                    qry = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                else:
                    print(f"An error occurred: {e}")
        else:
            if system_message:
                qry = [{"role": "system", "content": system_message}]
            else:
                qry = []
            qry.append({"role": "user", "content": msg})

        qrys.append(qry)

    return qrys


def generate_solution(prompt_id, model_name, temperature, p, max_tokens, dfs):
    if model_name not in litellm_models:
        model, params = load_model(model_name, temperature, p)

    df_results = {}
    for k, df in tqdm(dfs.items(),total=len(dfs)):
        if prompt_id != "clp":
            prompts = generate_queries(df, model_name, prompt_id)
            if model_name in litellm_models:
                responses = batch_completion(model=model_name, messages=prompts, temperature=temperature, top_p=p, max_tokens=max_tokens)
                outputs = [safe_parse_litellm(resp) for resp in responses]
            else:
                responses = model.generate(prompts, params)
                outputs = [safe_parse_vllm(output) for output in responses]
            df["solution"] = outputs

        else:
            align_prompts = generate_queries_clp(df, model_name)
            if model_name in litellm_models:
                responses = batch_completion(model=model_name, messages=align_prompts, temperature=temperature, top_p=p, max_tokens=max_tokens)
                align_outputs = [safe_parse_litellm(resp) for resp in responses]
            else:
                responses = model.generate(align_prompts, params)
                align_outputs = [safe_parse_vllm(output) for output in responses]
            df["alignment"] = align_outputs

            solution_prompts = generate_queries_clp(df, model_name)
            if model_name in litellm_models:
                responses = batch_completion(model=model_name, messages=solution_prompts, temperature=temperature, top_p=p, max_tokens=max_tokens)
                solution_outputs = [safe_parse_litellm(resp) for resp in responses]
            else:
                responses = model.generate(solution_prompts, params)
                solution_outputs = [safe_parse_vllm(output) for output in responses]
            df["solution"] = solution_outputs
            
        df_results[k] = df

    return df_results