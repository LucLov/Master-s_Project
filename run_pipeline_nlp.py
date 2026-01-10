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

    parser = argparse.ArgumentParser(description='NLP pipeline: NER, labeling, training. Requires processed input from data pipeline.')
    parser.add_argument('--input', default=os.path.join('data', 'processed', 'vidi_clean.json'), help='Processed input file')
    parser.add_argument('--ner', action='store_true', help='Run NER')
    parser.add_argument('--label', action='store_true', help='Build labeled dataset (uses sentence_transformers)')
    parser.add_argument('--train', action='store_true', help='Train classifier')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ ERROR: processed input not found: {args.input}")
        print('Run the data pipeline first to produce processed data.')
        sys.exit(1)

    python = sys.executable

    # If no specific flags were provided, run all NLP steps
    run_all = not (args.ner or args.label or args.train)

    if run_all or args.ner:
        run_step('extract_ner.py', [python, os.path.join('ner', 'extract_ner.py')])

    if run_all or args.label:
        run_step('build_labeled_dataset.py', [python, os.path.join('processing', 'build_labeled_dataset.py')])

    if run_all or args.train:
        run_step('train_classifier.py', [python, os.path.join('classification', 'train_classifier.py')])

    print('NLP pipeline finished. Outputs produced: data/processed/vidi_ner.json, data/labeled/final_dataset.json, classification/model.pkl')


def main():
    venv = "venv_nlp"

    # If we're not already running inside the target venv, re-exec with that venv's python
    if not running_in_venv(venv):
        py = find_python(venv)
        if not py:
            print(f"ERROR: could not find Python executable for {venv} (looked in Scripts/)")
            sys.exit(2)
        os.execv(py, [py, __file__] + sys.argv[1:])

    # Running inside venv_nlp -> run pipeline
    pipeline_main()


if __name__ == "__main__":
    main()
