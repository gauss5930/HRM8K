from huggingface_hub import snapshot_download
import argparse


def load_config(
        path: str
):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+")
    args = parser.parse_args()
    
    success_list, fail_dict = [], {}

    for model_name in args.models:
        try:
            snapshot_download(
                repo_id=model_name,
                repo_type="model"
            )
            success_list.append(model_name)
        except Exception as e:
            fail_dict[model_name] = e

    print("#################### Model Download Result ####################")
    print(f"- Count: {len(args.models)}\n- Succeed: {len(success_list)}\n- Failed: {len(fail_dict)}\n")
    print("**Succeed Models**")
    for s in success_list:
        print(f"- {s}")
    print("\n**Failed Models**")
    for f,e in fail_dict.items():
        print(f"- {f}: {e}")