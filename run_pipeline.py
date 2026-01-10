import os
import shutil
import subprocess
import argparse
import sys


def run_step(description, command):
    print(f"=== Running: {description} ===")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"❌ ERROR in step: {description}")
        sys.exit(1)
    print(f"✓ DONE: {description}\n")


def main():
    parser = argparse.ArgumentParser(description="Run full pipeline. By default the pipeline will use already-pulled data. Use --scrape or --force to fetch new data from the web.")
    parser.add_argument('--scrape', action='store_true', help='Fetch new data from web (default: use existing data)')
    parser.add_argument('--force', action='store_true', help='Alias for --scrape (forces fetching new data)')
    args = parser.parse_args()
    use_scrape = args.scrape or args.force

    print("=== PIPELINE START ===\n")

    # -------------------------
    # 1. SCRAPING (VIDI)
    # -------------------------
    dst = "data/raw/vidi_raw.json"

    if use_scrape:
        run_step("scrape_vidi.py", "python scraping/scrape_vidi.py")

        # 1.5 Copy scraped file to data/raw/
        src = "scraping/outputs/vidi_articles.json"

        if not os.path.exists(src):
            print(f"❌ ERROR: {src} ne postoji — scraping nije uspio!")
            sys.exit(1)

        shutil.copy(src, dst)
        print(f"✓ DONE: Copied {src} → {dst}\n")
    else:
        if not os.path.exists(dst):
            print(f"❌ ERROR: {dst} does not exist. Run with --scrape to fetch new data from the web.")
            sys.exit(1)
        print(f"✓ SKIP: Using existing {dst}\n")

    # -------------------------
    # 2. PREPROCESS
    # -------------------------
    run_step("preprocess_articles.py", "python processing/preprocess_articles.py")

    # -------------------------
    # 3. NER
    # -------------------------
    run_step("extract_ner.py", "python ner/extract_ner.py")

    # -------------------------
    # 4. BUILD LABELED DATASET (AUTO-LABELING)
    # -------------------------
    run_step("build_labeled_dataset.py", "python processing/build_labeled_dataset.py")

    # -------------------------
    # 5. TRAIN CLASSIFIER (EMBEDDING MODEL)
    # -------------------------
    run_step("train_classifier.py", "python classification/train_classifier.py")

    # -------------------------
    # 6. PREDICT
    # -------------------------
    run_step("predict.py", "python classification/predict.py")

    print("\n=== PIPELINE COMPLETE ===")


if __name__ == "__main__":
    main()
