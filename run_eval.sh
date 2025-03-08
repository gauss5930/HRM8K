#!/bin/bash

# List of model names to iterate over
models=(
    # "Qwen/Qwen2.5-1.5B-Instruct"
    # "Qwen/Qwen2.5-3B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
    # "Qwen/Qwen2.5-14B-Instruct"
    # "Qwen/Qwen2.5-32B-Instruct"
    # "Qwen/Qwen2.5-72B-Instruct"
    # "meta-llama/Llama-3.2-1B-Instruct"
    # "meta-llama/Llama-3.2-3B-Instruct"
    # "meta-llama/Llama-3.1-8B-Instruct"
    # "meta-llama/Llama-3.1-70B-Instruct"
    # "gpt-4o-mini"
    # "gpt-4o"
)

temp_list=(
    0.0
    0.7
)

# Set the categories
CATEGORIES="GSM8K MATH OMNI_MATH MMMLU KSM"

# Set the prompt ID
PROMPT_ID="k2k k2e e2k e2e clp"

# Set the evaluation method: ['normal', 'clp', 'plug']
EVAL_METHOD="normal"

# Set the scoring type: ['original', 'math_verify']
SCORE_TYPE="original math_verify"

# Optional: Set your tokens (uncomment and set as needed)
# export HF_TOKEN="YOUR_HF_TOKEN"
# export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"

# Loop through each model and run the Python script
for temp in "${temp_list[@]}"; do
    for model_name in "${models[@]}"; do
      echo "Running evaluation for model: $model_name / categories: $CATEGORIES / prompts: $PROMPT_ID"
      python src/run_eval.py --cats $CATEGORIES --model_name "$model_name" --prompt_id $PROMPT_ID --eval_method $EVAL_METHOD --score_type $SCORE_TYPE --temperature $temp
      # Uncomment the next line if you want to clear Hugging Face cache after each run
      # rm -rf ~/.cache/huggingface
    done
done
