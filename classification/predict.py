import os
import sys
import joblib
import numpy as np
import classla

MODEL_PATH = "classification/model.pkl"

# ---------- flags (hardcoded, jednostavno za palit/gasit) ----------
SHOW_ML = True
SHOW_TOP_PROBS = True
SHOW_ML_TOKEN_EXPLAIN = True

SHOW_HEURISTIC_EXPLAIN = True
SHOW_LEMMAS_DEBUG = False          # <-- uključi ako želiš vidjeti lemme iz unosa

SHOW_DISAGREEMENT_NOTE = True

# ML fallback: ako je max proba preniska -> OTHER
ENABLE_OTHER_FALLBACK = True
OTHER_THRESHOLD = 0.55  # po potrebi podesi (0.50-0.70)
# ---------------------------------------------------------------

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from processing.build_labeled_dataset import explain_labels  # noqa: E402


# ----------------------------
# Classla for lemmatizing user input (title/content)
# ----------------------------
_NLP = None


def get_nlp():
    global _NLP
    if _NLP is None:
        _NLP = classla.Pipeline(
            lang="hr",
            processors="tokenize,pos,lemma",
            logging_level="WARNING",
        )
    return _NLP


def lemmas_from_text(text: str) -> list[str]:
    """Extract lemmas from full text (all tokens), lowercase, no punct."""
    text = text or ""
    if not text.strip():
        return []
    doc = get_nlp()(text)
    out = []
    for sent in doc.sentences:
        for w in sent.words:
            if w.upos != "PUNCT" and w.lemma:
                out.append(w.lemma.lower())
    return out


# ----------------------------
# ML helpers
# ----------------------------
def top_label_probabilities(clf, mlb, X, top_k=6):
    """Returns list of (label, prob) sorted desc."""
    if not hasattr(clf, "predict_proba"):
        return []
    probs = clf.predict_proba(X)[0]
    pairs = list(zip(mlb.classes_, probs))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:top_k]


def max_label_probability(clf, X) -> float | None:
    """Return max probability across labels for current sample (OneVsRest)."""
    if not hasattr(clf, "predict_proba"):
        return None
    probs = clf.predict_proba(X)[0]
    return float(np.max(probs)) if probs.size else None


def explain_ml_for_label(pack, X, label: str, top_k=10):
    """
    Explains ML decision for a given label using LR coefficients.
    Shows top contributing tokens from the input text.
    """
    vectorizer = pack["vectorizer"]
    mlb = pack["binarizer"]
    clf = pack["model"]

    if label not in mlb.classes_:
        return []

    idx = int(np.where(mlb.classes_ == label)[0][0])
    est = clf.estimators_[idx]  # LogisticRegression for that label

    try:
        feature_names = vectorizer.get_feature_names_out()
    except Exception:
        feature_names = np.array(vectorizer.get_feature_names())

    row = X.tocsr()[0]
    indices = row.indices
    values = row.data
    if len(indices) == 0:
        return []

    coefs = est.coef_.ravel()

    contribs = []
    for j, v in zip(indices, values):
        contrib = float(v * coefs[j])
        contribs.append((feature_names[j], contrib))

    contribs.sort(key=lambda x: x[1], reverse=True)
    contribs = [c for c in contribs if c[1] > 0]
    return contribs[:top_k]


def read_multiline(prompt: str) -> str:
    """Read pasted multi-line text until empty line."""
    print(prompt)
    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)
    return "\n".join(lines).strip()


def main():
    pack = joblib.load(MODEL_PATH)
    vectorizer = pack["vectorizer"]
    mlb = pack["binarizer"]
    model = pack["model"]

    print("\n[PREDICT] Enter title + paste content (finish content with empty line).")
    print("Type 'exit' as title to quit.\n")

    while True:
        title = input("Title: ").strip()
        if title.lower() == "exit":
            break
        if not title:
            print("⚠ Title is empty. Try again (or type 'exit').")
            continue

        content = read_multiline("Paste content (finish with empty line):")

        # Combine for ML
        full_text = f"{title}\n{content}".strip() if content else title

        # ---------- ML prediction ----------
        ml_labels = tuple()
        probs_top = []
        max_p = None
        X = None

        if SHOW_ML:
            X = vectorizer.transform([full_text])
            pred = model.predict(X)
            ml_labels = tuple(mlb.inverse_transform(pred)[0])

            max_p = max_label_probability(model, X)
            probs_top = top_label_probabilities(model, mlb, X, top_k=8) if SHOW_TOP_PROBS else []

            if ENABLE_OTHER_FALLBACK and (max_p is not None) and (max_p < OTHER_THRESHOLD):
                ml_labels = ("OTHER",)

            print("\n=== ML prediction ===")
            if max_p is not None:
                print(f"Max label probability: {max_p:.3f}")
                if ENABLE_OTHER_FALLBACK:
                    print(f"OTHER fallback threshold: {OTHER_THRESHOLD:.2f}")
            print("Predicted labels:", ml_labels)

            if probs_top:
                print("\nTop label probabilities:")
                for lab, p in probs_top:
                    print(f"  - {lab}: {p:.3f}")

            if SHOW_ML_TOKEN_EXPLAIN and ml_labels and ml_labels != ("OTHER",) and X is not None:
                print("\nML explanation (top contributing tokens):")
                for lab in ml_labels:
                    feats = explain_ml_for_label(pack, X, lab, top_k=10)
                    print(f"  • {lab}:")
                    if not feats:
                        print("      (no informative tokens / input too short)")
                    else:
                        for tok, score in feats:
                            print(f"      {tok}: +{score:.4f}")

        # ---------- Heuristic debug_explain-style output ----------
        heur_selected = tuple()
        heur_text = None

        if SHOW_HEURISTIC_EXPLAIN:
            lem_title = lemmas_from_text(title)
            lem_content = lemmas_from_text(content if content else title)

            if SHOW_LEMMAS_DEBUG:
                print("\n[LEMMA DEBUG]")
                print("title lemmas:", lem_title[:60], ("..." if len(lem_title) > 60 else ""))
                print("content lemmas:", lem_content[:60], ("..." if len(lem_content) > 60 else ""))

            pseudo_article = {
                "url": "",
                "title": title,
                "content": content if content else title,
                "lemmas_title": lem_title,
                "lemmas_content": lem_content,
            }

            # izračunaj jednom (da nema dupliranja/side-effecta)
            heur_obj = explain_labels(
                pseudo_article,
                threshold=0.35,
                max_labels=3,
                as_text=False
            )
            heur_selected = tuple(heur_obj.get("selected_labels", []))

            heur_text = explain_labels(
                pseudo_article,
                threshold=0.35,
                max_labels=3,
                as_text=True
            )

            print("\n=== Heuristic explanation ===")
            print(heur_text)

        # ---------- Disagreement note ----------
        if SHOW_DISAGREEMENT_NOTE and SHOW_ML and SHOW_HEURISTIC_EXPLAIN:
            if ml_labels and heur_selected and (ml_labels != heur_selected):
                print("\n⚠ NOTE: ML prediction and heuristic labels disagree.")
                print("   ML:", ml_labels)
                print("   Heuristic:", heur_selected)

        print("\n" + "-" * 80 + "\n")

    print("✓ DONE: predict.py")


if __name__ == "__main__":
    main()
