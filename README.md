# HRM8K

<div align="center">
  <img src="assets/HAERAE_logo.png" alt="HAERAE_logo" width="250" height="250">
</div>

<p align="center">
  <b> HAERAE Team</b></a>
</p>

<div align="center">
  📝 <a href="https://arxiv.org/abs/2501.02448"><b>Paper</b></a>  
  |  📃 Blog (<a href="https://www.onelineai.com/blog/korean-reasoning-benchmarks"><b>EN</b></a>/<a href="https://www.onelineai.com/blog/%ED%95%9C%EA%B5%AD%EC%96%B4-%EC%B6%94%EB%A1%A0-%EB%B2%A4%EC%B9%98%EB%A7%88%ED%81%AC-hrm8k-hrmcr"><b>KO</b></a>)  
  |  <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" height="16" width="16" style="vertical-align:middle">  <a href="https://huggingface.co/datasets/HAERAE-HUB/HRM8K"><b>Dataset</b></a> 
</div>
<br>
<br>

HRM8K is a bilingual benchmark developed to explore multilingual mathematical reasoning, specifically focusing on Korean and English languages. 
It consists of 8,011 parallel math problems, available in both languages, designed to assess and compare the performance of language models in multilingual scenarios.

The problems included in HRM8K are sourced from various reputable examinations and competitions such as the Korean Mathematics Olympiads, university entrance exams, teacher certification tests, and popular English math benchmarks like GSM8K, MATH, Omni-MATH, and MMMLU. 
By providing parallel problems across two languages, HRM8K enables researchers and developers to systematically evaluate and identify differences in language model performance between English and Korean.

HRM8K offers a practical resource for advancing research in multilingual reasoning, helping improve the robustness and accuracy of language models across different linguistic contexts.

## 📄 Benchmark Construction

<div align="center">

||GSM8K|MATH|Omin-MATH|MMMLU|KSM|Sum|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|Counts|1,319|2,885|1,909|470|1,428|**8,011**|

</div>

The HRM8K consists of the following two subsets:

- `Korean School Math (KSM)`: This subset comprises 1,428 challenging Korean mathematical problems. We collect only from Olympiad or competition-level exams, regardless of the target age group. Consequently, even problems from younger curricula require a certain level of reasoning ability to solve. The `KSM` subset is sourced from the following diverse sources: KMO (한국수학올림피아드), KJMO (한국주니어수학올림피아드), CSAT (대학수학능력시험), KMS (한국대학수학경시대회), TQ (교원임용경쟁시험).
- `Prior Sets`: This subset comprises 6,583 problems from existing English mathematics benchmarks. We retain only instances with numeric answers for the MATH and Omni-MATH datasets, excluding those with text, equations, or proofs as final answers. In addition, we select only three math-related subsets, including abstract_algebra, college_mathematics, and high_school_mathematics from MMMLU datasets. The sources from which data was collected are as follows: GSM8K, MATH, Omni-MATH, MMMLU.

Each subset consists of questions formulated in both Korean and English.

## 🚀 Getting Started

**Prerequisites**
- Python ≥ 3.8
- vllm
- pandas
- openai
- litellm
- google-generativeai

**Requirements**
- `HF_TOKEN`
- `OPENAI_API_KEY` (optional)
- `GEMINI_API_KEY` (optional)
- `COHERE_API_KEY` (optional)
- `OPENROUTER_API_KEY` (optional)

### 1. Installation

```bash
git clone https://github.com/gauss5930/HRM8K.git
cd HRM8K
pip install -r requirements.txt
```

### 2. Config `run_eval.sh` file

Customize the evaluation config with your own evaluation setup.

**Models**
  - `models`: List of models to be evaluated. Currently, only `hf` and `openai` models are supported.

**Sampling Parameters**
  - `n`: A hyperparameter determining the number of responses to generate. When `n` is set to a value greater than 1, the `temperature` should be configured to a value greater than 0.0.
  - `temperature`: 
  - `top_p`: 
  - `max_tokens`: Default value is `8192`. 
  - `reasoning`: For models capable of switching into Thinking mode, setting the parameter to True activates the Thinking mode. (e.g., Qwen3 series)
  - `batch`: Batch generation is enabled for API-supported models such as OpenAI, Gemini, and OpenRouter by setting the corresponding parameter to True.

**Evaluation Setup**
  - `subsets`: List of HRM8K subsets for evaluation. Multiple selections are permitted from the following list: `[GSM8K, MATH, OMNI_MATH, MMMLU, KSM]`
  - `prompt_id`: Language prompt for evaluation. Multiple selections are permitted from the following list: `[ko, en]`
  - `score_type`: Scoring type for evaluation. Due to stability considerations, the `original` is recommended: `[original, math_verify]`
<br>

#### Setup Example
<details>
  <summary>Example Script</summary>
  
  ```bash
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

  ...
  ```
</details>
<details>
  <summary>Example 1: Single Sample Generation (Avg@1)</summary>

  ```bash
  ...
  n=1
  temperature=0.0
  top_p=0.95         # This value will be ignored
  max_tokens=8192
  ...
  ```
</details>
<details>
  <summary>Example 2: Multi Sample Generation (Avg@n)</summary>

  ```bash
  ...
  n=3
  temperature=0.6
  top_p=0.95
  max_tokens=8192
  ...
  ```
</details>
<br>

#### Prompt Example
<details>
  <summary>Korean Prompt</summary>

  ```
  {question} 주어진 문제를 한국어로 풀고 최종 답을 다음 형식으로 출력하시오: 최종 정답은 \\boxed{n}.
  ```
</details>
<details>
  <summary>English Prompt</summary>

  ```
  {question} Solve the given question in English and state the final answer in the following format: The final answer is \\boxed{n}.
  ```
</details>

### 3. Evaluation & Check

**0. Model download (Optional)**

Before starting evaluation, you can download the models with the `run_down.sh` script.
Ensure to set your `HF_TOKEN` in the `run_down.sh` script before run.

```bash
bash run_down.sh
```

**1. Evaluation**

Before starting evaluation, you need to set your `HF_TOKEN` in the `run_eval.sh` script. 
Additionally, if you plan to evaluate API supported models (not required), set your key as well: `[OPENAI_API_KEY, GEMINI_API_KEY, COHERE_API_KEY, OPENROUTER_API_KEY]`.
The following script performs evaluation based on the settings specified in `eval_config.yaml` and returns evaluation check results for all models, prompts, and subsets defined in the configuration.

```bash
bash run_eval.sh
```

**2. Check**

This script provides detailed information about correct and incorrect answers, as well as comprehensive scores for all completed evaluations. 
The results generated by running this script are saved in the `check_results` folder and the `check_score.json` file, reviewing the overall evaluation results.

```bash
bash run_check.sh
```

You can check the evaluation results in the following folders and files:

- `results`: It contains the models' responses to the questions.
- `score_results`: The final scores of the model for each subset are stored.
- `check_results`: Each response from the models is evaluated using a scoring function, and the results (correct/incorrect) are saved.
- `check_score.json`: The final scores of the model for each subset are stored.

## 📐 Result Timeline (KSM)

<details>
  <summary>Evaluation Configs</summary>

  - `n`: 3
  - `temperature`: 0.6
  - `top_p`: 0.95
  - `max_tokens`: 8192
  - `subset`: `KSM`
  - `Avg@3`: Generate responses three times, and compute the mean of the average scores obstained across each trial.
  
</details>
<details>
  <summary>English Models Result</summary>
  
  <div align="center">

| Models | Run 1 | Run 2 | Run 3 | Avg@3 | Pass@3 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| o4-mini | 75.84 | 76.75 | 76.61 | 76.40 | 85.22 |
| o3-mini | 73.11 | 72.97 | 72.20 | 72.76 | 80.11 |
| gpt-4.1 | 49.02 | 48.18 | 49.44 | 48.88 | 62.25 |
| gpt-4o | 26.33 | 26.61 | 25.84 | 26.26 | 35.85 |
| gemini-2.0-flash | 46.36 | 45.45 | 46.71 | 46.17 | 57.98 |
| gemini-2.0-flash-lite | 38.38 | 38.52 | 38.17 | 38.35 | 50.35 |
| gemini-1.5-flash | 28.29 | 28.43 | 28.71 | 28.48 | 37.46 |
| command-a-03-2025 | 24.09 | 24.79 | 23.18 | 24.02 | 34.45 |
| qwen3-32b | 54.69 | 54.62 | 54.20 | 54.51 | 65.20 |
| qwq-32B | 43.63 | 44.12 | 43.42 | 43.72 | 56.51 |
| qwen2.5-math-72B-instruct | 35.50 | 34.87 | 32.91 | 34.43 | 46.01 |
| qwen-2.5-72b-instruct | 28.78 | 29.83 | 28.50 | 29.04 | 39.71 |
| qwen-2.5-32b-instruct | 24.86 | 25.77 | 23.95 | 24.86 | 34.94 |
| llama-4-maverick | 39.29 | 38.52 | 40.13 | 39.31 | 51.05 |
| llama-4-scout | 30.81 | 32.07 | 30.88 | 31.26 | 40.97 |
| llama-3.1-405b-instruct | 16.67 | 15.27 | 16.32 | 16.08 | 26.40 |
| llama-3.3-70b-instruct | 19.12 | 18.14 | 17.65 | 18.30 | 28.71 |

  </div>
</details>
<details>
  <summary>Korean Models Result</summary>

  <div align="center">

| Models | Run 1 | Run 2 | Run 3 | Avg@3 | Pass@3 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| EXAONE-3.5-2.4B-Instruct | 9.80 | 10.01 | 8.89 | 9.57 | 17.51 |
| EXAONE-3.5-7.8B-Instruct | 16.32 | 16.81 | 17.23 | 16.78 | 25.56 |
| EXAONE-3.5-32B-Instruct | 18.63 | 18.21 | 18.77 | 18.53 | 26.82 |
| EXAONE-Deep-2.4B | 23.18 | 21.92 | 23.11 | 22.74 | 33.75 |
| EXAONE-Deep-7.8B | 29.97 | 33.96 | 30.39 | 31.44 | 44.12 |
| EXAONE-Deep-32B | 36.48 | 36.83 | 36.55 | 36.62 | 48.18 |
| kanana-1.5-2.1b-instruct-2505 | 8.68 | 8.82 | 9.17 | 8.89 | 15.27 |
| kanana-1.5-8b-instruct-2505 | 12.18 | 11.48 | 13.03 | 12.23 | 20.03 |
| HyperCLOVAX-SEED-Text-Instruct-0.5B | 4.06 | 3.15 | 3.29 | 3.50 | 7.21 |
| HyperCLOVAX-SEED-Text-Instruct-1.5B | 4.97 | 4.83 | 5.60 | 5.14 | 9.80 |
| Trillion-7B-preview | 5.11 | 5.39 | 5.39 | 5.07 | 9.17 |
| Llama-DNA-1.0-8B-Instruct | 5.39 | 5.60 | 6.02 | 5.67 | 10.57 |
| DNA-R1 | 36.62 | 35.64 | 35.85 | 36.04 | 48.67 |
| A.X-4.0-Light | 19.47 | 19.12 | 19.54 | 19.37 | 29.97 |
| A.X-4.0 | 29.06 | 29.69 | 29.83 | 29.53 | 41.25 |
    
  </div>
  
</details>

**English Models**

<div align="center">
  <img src="assets/en_model_result.png" alt="English Model Result Timeline">
</div>

**Korean Models**

<div align="center">
  <img src="assets/ko_model_result.png" alt="Korean Model Result Timeline">
</div>

## 🧑‍🔬 Contributors
```
Hyunwoo Ko, Guijin Son, and Dasol Choi
```

## 📨 Contact

Feel free to contact us via the following email!

```
kopilot100@gmail.com
```

## 📝 Citation
```
@article{ko2025understand,
  title={Understand, Solve and Translate: Bridging the Multilingual Mathematical Reasoning Gap},
  author={Ko, Hyunwoo and Son, Guijin and Choi, Dasol},
  journal={arXiv preprint arXiv:2501.02448},
  year={2025}
}
```
