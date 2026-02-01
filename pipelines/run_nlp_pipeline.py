import argparse
import subprocess
import sys

def run_step(description: str, args_list: list[str]):
    print(f"=== Running: {description} ===")
    r = subprocess.run(args_list)
    if r.returncode != 0:
        print(f"❌ ERROR in step: {description}")
        sys.exit(1)
    print(f"✓ DONE: {description}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--config", default="clean")

    # Accept unknown args so we can forward them to build_labeled_dataset.py
    args, unknown = parser.parse_known_args()

    print("=== NLP PIPELINE START ===\n")
    py = sys.executable  # <-- KLJUČNO

    run_step(
        "pull_dataset.py",
        [
            py,
            "hf/pull_dataset.py",
            "--repo_id", args.repo_id,
            "--config", args.config,
            "--out_path", "data/processed/vidi_clean.json",
        ]
    )

    run_step("extract_ner.py", [py, "ner/extract_ner.py"])

    # Forward extra args ONLY to build_labeled_dataset.py
    run_step(
        "build_labeled_dataset.py",
        [py, "processing/build_labeled_dataset.py"] + unknown
    )

    run_step("train_classifier.py", [py, "classification/train_classifier.py"])
    run_step("evaluate.py", [py, "classification/evaluate.py", "--save_reports"])
    run_step("predict.py", [py, "classification/predict.py"])

    print("\n=== NLP PIPELINE COMPLETE ===")

if __name__ == "__main__":
    main()
