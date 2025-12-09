import os
import shutil
import subprocess


def run_step(description, command):
    print(f"=== Running: {description} ===")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"❌ ERROR in step: {description}")
        exit(1)
    print(f"✓ DONE: {description}\n")


print("=== PIPELINE START ===\n")

# -------------------------
# 1. SCRAPING (VIDI)
# -------------------------
run_step("scrape_vidi.py", "python scraping/scrape_vidi.py")

# 1.5 Copy scraped file to data/raw/
src = "scraping/outputs/vidi_articles.json"
dst = "data/raw/vidi_raw.json"

if not os.path.exists(src):
    print(f"❌ ERROR: {src} ne postoji — scraping nije uspio!")
    exit(1)

shutil.copy(src, dst)
print(f"✓ DONE: Copied {src} → {dst}\n")

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
