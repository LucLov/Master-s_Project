import argparse
import json
import os
from collections import Counter, defaultdict

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    hamming_loss,
    accuracy_score,
    classification_report,
)

DATA_PATH = os.path.join("data", "labeled", "final_dataset.json")
MODEL_PATH = os.path.join("classification", "model.pkl")
REPORT_DIR = "reports"
REPORT_JSON = os.path.join(REPORT_DIR, "metrics.json")
REPORT_TXT = os.path.join(REPORT_DIR, "metrics.txt")


def load_data(path: str):
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)

    texts = []
    labels = []
    for r in rows:
        title = (r.get("title") or "").strip()
        content = (r.get("content") or "").strip()
        text = (title + "\n" + content).strip() if content else title
        texts.append(text)
        labels.append(r.get("labels", []) or [])
    return texts, labels


def label_distribution(labels_list):
    c = Counter()
    for labs in labels_list:
        for l in labs:
            c[l] += 1
    return dict(c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA_PATH)
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_reports", action="store_true", help="spremi reports/metrics.*")
    args = ap.parse_args()

    texts, y_labels = load_data(args.data)

    pack = joblib.load(args.model)
    vectorizer = pack["vectorizer"]
    mlb = pack["binarizer"]
    clf = pack["model"]

    # y_true: multi-hot
    y_true = mlb.transform(y_labels)
    X = vectorizer.transform(texts)

    # Holdout split (proxy eval: heuristika kao ground truth)
    idx = np.arange(len(texts))
    train_idx, test_idx = train_test_split(
        idx, test_size=args.test_size, random_state=args.seed, shuffle=True
    )

    X_test = X[test_idx]
    y_test = y_true[test_idx]

    y_pred = clf.predict(X_test)

    # --- Metrics ---
    metrics = {}
    metrics["n_total"] = int(len(texts))
    metrics["n_test"] = int(len(test_idx))
    metrics["label_distribution_total"] = label_distribution(y_labels)

    # Micro/macro for multilabel
    metrics["f1_micro"] = float(f1_score(y_test, y_pred, average="micro", zero_division=0))
    metrics["f1_macro"] = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    metrics["precision_micro"] = float(precision_score(y_test, y_pred, average="micro", zero_division=0))
    metrics["recall_micro"] = float(recall_score(y_test, y_pred, average="micro", zero_division=0))
    metrics["hamming_loss"] = float(hamming_loss(y_test, y_pred))

    # Subset accuracy: strogo (svi labeli moraju biti identični)
    metrics["subset_accuracy"] = float(accuracy_score(y_test, y_pred))

    # Per-label report (tekst)
    report_txt = classification_report(
        y_test, y_pred, target_names=list(mlb.classes_), zero_division=0
    )

    # --- Diagnostics without "true" labels (confidence-ish) ---
    diag = {}
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(X_test)
        max_prob = np.max(probs, axis=1)
        diag["avg_max_prob"] = float(np.mean(max_prob))
        diag["median_max_prob"] = float(np.median(max_prob))
        diag["min_max_prob"] = float(np.min(max_prob))
        diag["max_max_prob"] = float(np.max(max_prob))
    metrics["diagnostics"] = diag

    # Print summary
    print("\n=== EVALUATION (proxy ground truth = heurističke labele) ===")
    print(f"Total samples: {metrics['n_total']}, test samples: {metrics['n_test']}")
    print(f"F1 micro: {metrics['f1_micro']:.3f}")
    print(f"F1 macro: {metrics['f1_macro']:.3f}")
    print(f"Precision micro: {metrics['precision_micro']:.3f}")
    print(f"Recall micro: {metrics['recall_micro']:.3f}")
    print(f"Hamming loss: {metrics['hamming_loss']:.3f}")
    print(f"Subset accuracy: {metrics['subset_accuracy']:.3f}")

    if diag:
        print("\n=== Diagnostics ===")
        print(f"Avg max prob: {diag['avg_max_prob']:.3f}")
        print(f"Median max prob: {diag['median_max_prob']:.3f}")

    print("\n=== Per-label report ===")
    print(report_txt)

    if args.save_reports:
        os.makedirs(REPORT_DIR, exist_ok=True)
        with open(REPORT_JSON, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        with open(REPORT_TXT, "w", encoding="utf-8") as f:
            f.write("EVALUATION (proxy ground truth = heurističke labele)\n\n")
            f.write(json.dumps(metrics, ensure_ascii=False, indent=2))
            f.write("\n\nPER-LABEL REPORT\n\n")
            f.write(report_txt)

        print(f"\n✓ Saved: {REPORT_JSON}")
        print(f"✓ Saved: {REPORT_TXT}")


if __name__ == "__main__":
    main()
