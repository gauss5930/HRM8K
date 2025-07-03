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
    
def load_model(model_name, temperature, p, max_tokens):
    if "exaone" in model_name.lower():
        os.environ["VLLM_USE_V1"] = "0"
    else:
        os.environ["VLLM_USE_V1"] = "1"
        
    
    model = LLM(model=model_name, tensor_parallel_size=torch.cuda.device_count(), trust_remote_code=True)
        
    tokenizer = model.get_tokenizer()

    if temperature == 0.0:
        params = SamplingParams(temperature=0.0, min_tokens=8, max_tokens=max_tokens)
    else:
        params = SamplingParams(temperature=temperature, top_p=p, min_tokens=8, max_tokens=max_tokens)

    return model, tokenizer, params
