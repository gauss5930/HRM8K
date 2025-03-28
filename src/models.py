from vllm import LLM, SamplingParams
import torch

litellm_models = [
    'gpt-4o',
    'gpt-4o-mini',
    'gpt-4o-mini-2024-07-18'
]
    
def load_model(model_name, temperature, p, max_tokens=8192):
    try:
        llm = LLM(model_name, tensor_parallel_size=torch.cuda.device_count(),max_model_len=max_tokens+2048)
    except:
        llm = LLM(model_name, tensor_parallel_size=torch.cuda.device_count())

    if temperature == 0.0:
        params = SamplingParams(temperature=0.0, min_tokens=8, max_tokens=max_tokens)
    else:
        params = SamplingParams(temperature=temperature, top_p=p, min_tokens=8, max_tokens=2048)
        
    return llm, params
