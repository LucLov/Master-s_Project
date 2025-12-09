import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
import joblib

INPUT_FILE = "data/labeled/final_dataset.json"
MODEL_PATH = "classification/model.pkl"


def main():
    print("[TRAIN] Loading labeled dataset...")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = [d["content"] for d in data]
    labels = [d["labels"] for d in data]

    # TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(texts)

    # Label binarizer
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(labels)

    clf = OneVsRestClassifier(LogisticRegression(max_iter=300))
    clf.fit(X, Y)

    joblib.dump({
        "vectorizer": vectorizer,
        "binarizer": mlb,
        "model": clf
    }, MODEL_PATH)

    print(f"[DONE] Model saved → {MODEL_PATH}")


if __name__ == "__main__":
    main()
