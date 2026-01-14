import argparse
import os
import subprocess
import sys

def run(desc: str, cmd: list[str]):
    print(f"=== {desc} ===")
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f"❌ ERROR: {desc}")
        sys.exit(1)
    print(f"✓ DONE: {desc}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--update-data", action="store_true")
    parser.add_argument("--repo-id", default="lucijalovric/vidi-news-dataset")
    parser.add_argument("--config", default="clean", help="raw ili clean (koristi se u NLP pull)")
    parser.add_argument("--push-after", action="store_true", help="pushaj HF dataset na kraju")
    args, unknown = parser.parse_known_args()

    py_data = r"venv_data\Scripts\python.exe"
    py_nlp  = r"venv_nlp\Scripts\python.exe"

    if not os.path.exists(py_data):
        print(f"❌ Missing: {py_data}")
        sys.exit(1)
    if not os.path.exists(py_nlp):
        print(f"❌ Missing: {py_nlp}")
        sys.exit(1)

    if args.update_data:
        run(
            "DATA pipeline (scrape/preprocess/push)",
            [py_data, r"pipelines\run_data_pipeline.py", "--repo-id", args.repo_id]
        )
    else:
        print("=== Skipping DATA pipeline (using Hugging Face dataset) ===\n")

    cmd = [py_nlp, r"pipelines\run_nlp_pipeline.py", "--repo-id", args.repo_id, "--config", args.config] + unknown
    run("NLP pipeline (pull/NER/train/predict)", cmd)

    if args.push_after:
        push_cmd = [py_data, r"hf/push_dataset.py", "--repo_id", args.repo_id]
        if not args.update_data:
            push_cmd += ["--skip-raw"]  # ✅ bitno u HF-only modu
        run("PUSH updated dataset to Hugging Face (after NLP)", push_cmd)

    print("=== PIPELINE COMPLETE ===")

if __name__ == "__main__":
    main()
