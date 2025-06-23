#API KEY Setup
export HF_TOKEN="<HF_TOKEN>"
export OPENAI_API_KEY="<OPENAI_API_KEY>"
export GEMINI_API_KEY="<GEMINI_API_KEY>"
export COHERE_API_KEY="<COHERE_API_KEY>"
export OPENROUTER_API_KEY="<OPENROUTER_API_KEY>"

# Evaluation model list
models=(
	"MODEL_1"
	"MODEL_2"
	"MODEL_3"
)

# Generation parameters
n=3
temperature=0.6
top_p=0.95
max_tokens=8192
reasoning=false
batch=false

# Evaluation dataset subsets and prompts
subsets=("GSM8K" "MATH" "OMNI_MATH" "MMMLU" "KSM")
prompt_id=("ko")             # Korean(ko) and English(en) are supported
score_type=("original")      # or math_verify

# Model download (optional)
# python src/model_download.py --models $models

# Evaluation
for model in "${models[@]}"; do
  python src/run_eval.py \
	--model_name "$model" \
	--n $n \
	--temperature $temperature \
	--top_p $top_p \
	--max_tokens $max_tokens \
	--reasoning $reasoning \
	--batch $batch \
	--subsets "${subsets[@]}" \
	--prompt_id "${prompt_id[@]}" \
	--score_type "${score_type[@]}"
done

# Result Check
python src/check.py \
  --check_type config \
  --models "${models[@]}" \
  --subsets "${subsets[@]}" \
  --prompt_id "${prompt_id[@]}"