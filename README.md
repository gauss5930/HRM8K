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

## Benchmark Construction

||GSM8K|MATH|Omin-MATH|MMMLU|KSM|Sum|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|Counts|1,319|2,885|1,909|470|1,428|**8,011**|

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

### 2. Config YAML file

Customize the evaluation config with `eval_config.yaml`.
- `models`: List of models to be evaluated. Currently, only `hf` and `openai` models are supported.
- `reasoning`: For models capable of switching into Thinking mode, setting the parameter to True activates the Thinking mode. (e.g., Qwen3 series)
- `subsets`: List of HRM8K subsets for evaluation. Multiple selections are permitted from the following list: [GSM8K, MATH, OMNI_MATH, MMMLU, KSM]
- `prompt_id`: Language prompt for evaluation. Multiple selections are permitted from the following list: [ko, en]
- `score_type`: Scoring type for evaluation. Due to stability considerations, the `original` is recommended.
<br>

```yaml
# Evaluation model list
models:
  - MODEL_1
  - MODEL_2
  - MODEL_3

# Generation parameters
temperature: 0.0
# This value will be ignored when `temperature` set to 0.0
top_p: 0.95
max_tokens: 8192
# The key to trigger the long thinking mode of language model (e.g., Qwen3 series)
reasoning: false
# Option for batch generation. (Only applied while evaluation litellm models.)
batch: false

# Evaluation dataset subsets & prompts
subsets: [GSM8K, MATH, OMNI_MATH, MMMLU, KSM]
prompt_id: [ko]         # Korean(ko) and English(en) are supported!
score_type: [original]    # or math_verify (math_verify does not work well on `MMMLU` subset)
```
<br>

**Korean Prompt**
```
{question} 주어진 문제를 한국어로 풀고 최종 답을 다음 형식으로 출력하시오: 최종 정답은 \\boxed{n}.
```
<br>

**English Prompt**
```
{question} Solve the given question in English and state the final answer in the following format: The final answer is \\boxed{n}.
```

### 3. Evaluation & Check

**0. Model download (Optional)**

Before starting evaluation, you can download the models with the `run_down.sh` script.
Ensure to set your `HF_TOKEN` in the `run_down.sh` script before run.

```bash
bash run_down.sh
```

**1. Evaluation**

Before starting evaluation, you need to set your `HF_TOKEN` in the `run_eval.sh` script. 
Additionally, if you plan to evaluate OpenAI models, set your `OPENAI_API_KEY` as well.
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

## Contributors
```
Hyunwoo Ko, Guijin Son, and Dasol Choi
```

## Contact

Feel free to contact us via the following email!

```
hyunwooko@onelineai.com
```

## Citation
```
@article{ko2025understand,
  title={Understand, Solve and Translate: Bridging the Multilingual Mathematical Reasoning Gap},
  author={Ko, Hyunwoo and Son, Guijin and Choi, Dasol},
  journal={arXiv preprint arXiv:2501.02448},
  year={2025}
}
```