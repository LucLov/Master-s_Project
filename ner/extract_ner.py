import json
import classla
import re
from collections import defaultdict

INPUT_FILE = "data/processed/vidi_clean.json"
OUTPUT_FILE = "data/processed/vidi_ner.json"


TECH_KEYWORDS = [
    "AI", "umjetna inteligencija", "LLM", "model", "neuronska mreža",
    "strojno učenje", "algoritam", "procesor", "računalo", "server",
    "GPU", "CPU", "robot", "aplikacija", "softver", "hardver",
    "čip", "platforma", "kvantno", "tehnologija"
]

PRODUCT_PATTERNS = [
    r"[A-Za-z]+\s?\d{2,4}",
    r"\bmodel\b",
    r"\bserija\b",
    r"[A-Z][A-Za-z0-9\-]+\s?(Pro|Ultra|Max|Lite)"
]

EVENT_KEYWORDS = [
    "konferencija", "sajam", "summit", "forum", "dani", "expo"
]


def guess_custom_category(ent_type, text):
    """Mapira standardne entitete u tvoje custom kategorije."""

    # --- PRODUCT ---
    for patt in PRODUCT_PATTERNS:
        if re.search(patt, text):
            return "PRODUCT"

    # --- TECH ---
    for kw in TECH_KEYWORDS:
        if kw.lower() in text.lower():
            return "TECH"

    # --- EVENT ---
    for ew in EVENT_KEYWORDS:
        if ew.lower() in text.lower():
            return "EVENT"

    # Ako se ništa ne uklapa → vrati None
    return None


# ------------------------------
# MAIN PIPELINE
# ------------------------------

def main():
    print("[NER] Loading Classla pipeline...")

    nlp = classla.Pipeline(
        lang="hr",
        processors="tokenize,ner",
        logging_level="WARNING"
    )

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        articles = json.load(f)

    results = []

    for art in articles:
        text = art["content"]
        doc = nlp(text)

        standard = defaultdict(set)  # PER, ORG, LOC, MISC...
        custom = defaultdict(set)    # TECH, PRODUCT, EVENT...

        for ent in doc.ents:
            raw = ent.text.strip()
            ent_type = ent.type

            if len(raw) < 2:
                continue

            # ------------------
            # STANDARD ENTITIES
            # ------------------
            standard[ent_type].add(raw)

            # ------------------
            # CUSTOM ENTITIES
            # ------------------
            mycat = guess_custom_category(ent_type, raw)
            if mycat:
                custom[mycat].add(raw)

        # Convert sets → lists
        standard = {k: list(v) for k, v in standard.items()}
        custom = {k: list(v) for k, v in custom.items()}

        results.append({
            "url": art["url"],
            "title": art["title"],
            "entities": {
                "standard": standard,
                "custom": custom
            }
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[DONE] NER extracted → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()