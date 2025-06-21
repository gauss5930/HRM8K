# Environment Variables
export HF_TOKEN="<HF_TOKEN>"
export OPENAI_API_KEY="<OPENAI_API_KEY>"
export GEMINI_API_KEY="<GEMINI_API_KEY>"
export COHERE_API_KEY="<COHERE_API_KEY>"

# Evaluation code
python src/run_eval.py

# Evaluation result check code
python src/check.py --check_type "config"