# API KEY Setup
export HF_TOKEN="<HF_TOKEN>"

# Model list
models=(
	"MODEL_1"
	"MODEL_2"
	"MODEL_3"
)

# Model Download
python src/model_download.py --models "${models[@]}"