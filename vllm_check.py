import asyncio
import gc
import torch
import time
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

# --- 모델 A 정보 ---
MODEL_A_PATH = "Qwen/Qwen2.5-1.5B-Instruct"

# --- 모델 B 정보 ---
MODEL_B_PATH = "Qwen/Qwen3-4B"

async def run_inference(engine, model_name, prompt):
    """주어진 엔진으로 추론을 실행하는 헬퍼 함수"""
    sampling_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=50)
    request_id = random_uuid()
    
    print(f"\n--- {model_name} 추론 시작 ---")
    print(f"Prompt: {prompt}")
    
    start_time = time.time()
    results_generator = engine.generate(prompt, sampling_params, request_id)
    
    final_output = None
    async for request_output in results_generator:
        # 스트리밍 출력이 아닌 최종 결과만 받기
        final_output = request_output

    end_time = time.time()
    
    if final_output:
        prompt_out = final_output.prompt
        text_outputs = [output.text for output in final_output.outputs]
        print(f"Output: {text_outputs[0]}")
    
    print(f"추론 시간: {end_time - start_time:.2f}초")
    print(f"--- {model_name} 추론 완료 ---\n")


async def main():
    # ==================================================================
    # 1. 모델 A 로드 및 사용
    # ==================================================================
    print("1. MODEL_A을 로드합니다...")
    engine_a_args = AsyncEngineArgs(model=MODEL_A_PATH, trust_remote_code=True, tensor_parallel_size=2)
    engine_a = AsyncLLMEngine.from_engine_args(engine_a_args)
    
    await run_inference(engine_a, "MODEL_A", "Hello, my name is")

    # ==================================================================
    # 2. 모델 A 언로드 (메모리 해제)
    # ==================================================================
    print("2. MODEL_A을 메모리에서 해제합니다...")
    del engine_a
    del engine_a_args
    
    # 가비지 컬렉션 실행
    gc.collect()
    
    # **핵심**: GPU 캐시 메모리 비우기
    # 이 코드가 없으면 "CUDA out of memory" 오류가 발생할 수 있습니다.
    torch.cuda.empty_cache()
    print("   GPU 캐시를 비웠습니다.")
    
    # 잠시 대기하여 시스템이 안정화될 시간을 줍니다.
    await asyncio.sleep(5)

    # ==================================================================
    # 3. 모델 B 로드 및 사용
    # ==================================================================
    print("\n3. 'MODEL_B'을 로드합니다...")
    engine_b_args = AsyncEngineArgs(model=MODEL_B_PATH, trust_remote_code=True, tensor_parallel_size=2)
    engine_b = AsyncLLMEngine.from_engine_args(engine_b_args)
    
    await run_inference(engine_b, "MODEL_B", "The capital of France is")
    
    # ==================================================================
    # 4. 모델 B 언로드
    # ==================================================================
    print("4. MODEL_B을 메모리에서 해제합니다...")
    del engine_b
    del engine_b_args
    gc.collect()
    torch.cuda.empty_cache()
    print("   모든 작업 완료.")


if __name__ == "__main__":
    # GPU 메모리 상태 확인 (시작 전)
    if torch.cuda.is_available():
        print(f"시작 전 GPU 메모리: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    asyncio.run(main())
    
    # GPU 메모리 상태 확인 (종료 후)
    if torch.cuda.is_available():
        print(f"종료 후 GPU 메모리: {torch.cuda.memory_allocated() / 1e9:.2f} GB")