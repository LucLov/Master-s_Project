import argparse
import os
import shutil
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
    args = parser.parse_args()

    print("=== DATA PIPELINE START ===\n")
    py = sys.executable  # <-- KLJUČNO

    run_step("scrape_vidi.py", [py, "scraping/scrape_vidi.py"])

    src = os.path.join("scraping", "outputs", "vidi_articles.json")
    dst = os.path.join("data", "raw", "vidi_raw.json")
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    if not os.path.exists(src):
        print(f"❌ ERROR: {src} ne postoji — scraping nije uspio!")
        sys.exit(1)

    shutil.copy(src, dst)
    print(f"✓ DONE: Copied {src} → {dst}\n")

    run_step("preprocess_articles.py", [py, "processing/preprocess_articles.py"])
    run_step("push_dataset.py", [py, "hf/push_dataset.py", "--repo_id", args.repo_id])

    print("\n=== DATA PIPELINE COMPLETE ===")

if __name__ == "__main__":
    main()
