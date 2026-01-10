import os
import sys
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))


def find_python(venv_name):
    candidates = [
        os.path.join(HERE, venv_name, "Scripts", "python.exe"),
        os.path.join(HERE, venv_name, "Scripts", "python"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def running_in_venv(venv_name):
    exe = os.path.abspath(sys.executable).lower()
    venv_path = os.path.abspath(os.path.join(HERE, venv_name)).lower()
    return exe.startswith(venv_path)


def run_step(description, cmd_list):
    print(f"=== Running: {description} ===")
    rc = subprocess.call(cmd_list)
    if rc != 0:
        print(f"❌ ERROR in step: {description}")
        sys.exit(rc)
    print(f"✓ DONE: {description}\n")


def pipeline_main():
    import argparse

    parser = argparse.ArgumentParser(description="Data-only pipeline: scraping and preprocessing.")
    parser.add_argument('--scrape', '--force', action='store_true', dest='scrape', help='Fetch new data from web (overwrite data/raw/vidi_raw.json)')
    args = parser.parse_args()

    python = sys.executable

    # 1. SCRAPE (optional)
    dst = os.path.join('data', 'raw', 'vidi_raw.json')
    if args.scrape:
        run_step('scrape_vidi.py', [python, os.path.join('scraping', 'scrape_vidi.py')])
        src = os.path.join('scraping', 'outputs', 'vidi_articles.json')
        if not os.path.exists(src):
            print(f"❌ ERROR: {src} not found after scraping")
            sys.exit(1)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        import shutil
        shutil.copy(src, dst)
        print(f"✓ DONE: Copied {src} → {dst}\n")
    else:
        if not os.path.exists(dst):
            print(f"❌ ERROR: {dst} does not exist. Re-run with --scrape to fetch data.")
            sys.exit(1)
        print(f"✓ SKIP: Using existing {dst}\n")

    # 2. PREPROCESS
    run_step('preprocess_articles.py', [python, os.path.join('processing', 'preprocess_articles.py')])

    print('Data pipeline finished. Next: run the NLP pipeline to build labels and train models ("run_pipeline_nlp.bat").')


def main():
    venv = "venv_data"

    # If we're not already running inside the target venv, re-exec with that venv's python
    if not running_in_venv(venv):
        py = find_python(venv)
        if not py:
            print(f"ERROR: could not find Python executable for {venv} (looked in Scripts/)")
            sys.exit(2)
        os.execv(py, [py, __file__] + sys.argv[1:])

    # Running inside venv_data -> run pipeline
    pipeline_main()


if __name__ == "__main__":
    main()
