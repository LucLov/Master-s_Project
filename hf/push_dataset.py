import argparse
import os
from datasets import load_dataset, DatasetDict

def push_json(repo_id: str, json_path: str, config_name: str, private: bool) -> bool:
    if not json_path or not os.path.exists(json_path):
        print(f"⚠ {config_name} not found, skipping: {json_path}")
        return False

    ds = load_dataset("json", data_files={"train": json_path})
    ds = DatasetDict({"train": ds["train"]})
    ds.push_to_hub(repo_id, config_name=config_name, private=private)
    print(f"✓ Pushed {config_name} -> {repo_id} from {json_path}")
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", required=True, help="npr. lucijalovric/vidi-news-dataset")
    parser.add_argument("--raw_path", default=os.path.join("data", "raw", "vidi_raw.json"))
    parser.add_argument("--clean_path", default=os.path.join("data", "processed", "vidi_clean.json"))
    parser.add_argument("--private", action="store_true")

    # ✅ novo: omogući da eksplicitno preskočiš raw
    parser.add_argument("--skip-raw", action="store_true", help="ne pushaj raw config")
    args = parser.parse_args()

    if not args.skip_raw:
        push_json(args.repo_id, args.raw_path, config_name="raw", private=args.private)

    push_json(args.repo_id, args.clean_path, config_name="clean", private=args.private)

if __name__ == "__main__":
    main()
