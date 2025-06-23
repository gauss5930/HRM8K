import torch
import os
from vllm import SamplingParams, LLM

thinking_model_list = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-32B",
    "Qwen/Qwen3-30B-A3B",
    "Qwen/Qwen3-235B-A22B"
]
    
async def load_model(model_name, temperature, p, max_tokens):
    if "exaone" in model_name.lower():
        os.environ["VLLM_USE_V1"] = "0"
    else:
        os.environ["VLLM_USE_V1"] = "1"
        
    try:
        model = LLM(model=model_name, tensor_parallel_size=torch.cuda.device_count(), trust_remote_code=True, max_model_len=max_tokens+2048)
    except Exception as e:
        print(e)
        print("Retry load model without `max_model_len`")
        model = LLM(model=model_name, tensor_parallel_size=torch.cuda.device_count(), trust_remote_code=True)
        
    tokenizer = model.get_tokenizer()

    if temperature == 0.0:
        params = SamplingParams(temperature=0.0, min_tokens=8, max_tokens=max_tokens)
    else:
        params = SamplingParams(temperature=temperature, top_p=p, min_tokens=8, max_tokens=max_tokens)

    return model, tokenizer, params
