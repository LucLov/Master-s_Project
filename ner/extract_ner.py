import json
import classla
import re
from collections import defaultdict

INPUT_FILE = "data/processed/vidi_clean.json"
OUTPUT_FILE = "data/processed/vidi_lemmas.json"

TECH_KEYWORDS = [
    "AI", "UI", "artificial intelligence", "umjetna inteligencija", "LLM", "model", "neuronska mreža",
    "strojno učenje", "algoritam", "procesor", "računalo", "server",
    "GPU", "CPU", "robot", "dron", "aplikacija", "softver", "hardver",
    "čip", "platforma", "kvantno", "tehnologija", "računanje", "optimizacija"
]

PRODUCT_PATTERNS = [
    r"[A-Za-z]+\s?\d{2,4}",
    r"\bmodel\b",
    r"\bserija\b",
    r"[A-Z][A-Za-z0-9\-]+\s?(Pro|Ultra|Max|Lite)"
]

EVENT_KEYWORDS = ["konferencija", "sajam", "summit", "forum", "dani", "expo"]

def guess_custom_category(ent_type, text):
    for patt in PRODUCT_PATTERNS:
        if re.search(patt, text):
            return "PRODUCT"
    for kw in TECH_KEYWORDS:
        if kw.lower() in text.lower():
            return "TECH"
    for ew in EVENT_KEYWORDS:
        if ew.lower() in text.lower():
            return "EVENT"
    return None

def entity_lemma(ent):
    try:
        lemmas = []
        for tok in ent.tokens:
            if tok.words and tok.words[0].lemma:
                lemmas.append(tok.words[0].lemma)
            else:
                lemmas.append(tok.text)
        lemma = " ".join(lemmas).strip()
        return lemma if lemma else ent.text.strip()
    except Exception:
        return ent.text.strip()

def add_ent(store: dict, ent_type: str, surface: str, lemma: str):
    key = (surface, lemma)
    store[ent_type].add(key)

def doc_lemmas(doc) -> list[str]:
    """Extract lemmas from whole doc (all tokens), lowercase, no punct."""
    out = []
    for sent in doc.sentences:
        for w in sent.words:
            if w.upos != "PUNCT" and w.lemma:
                out.append(w.lemma.lower())
    return out

def main():
    print("[LEMMA+NER] Loading Classla pipeline...")

    nlp = classla.Pipeline(
        lang="hr",
        processors="tokenize,pos,lemma,ner",
        logging_level="WARNING"
    )

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        articles = json.load(f)

    results = []

    for art in articles:
        title = art.get("title", "") or ""
        content = art.get("content", "") or ""

        # process title + content separately (title is short)
        doc_title = nlp(title)
        doc_content = nlp(content)

        standard = defaultdict(set)
        custom = defaultdict(set)

        # NER over content (obično je dovoljno)
        for ent in doc_content.ents:
            surface = ent.text.strip()
            if len(surface) < 2:
                continue
            lemma = entity_lemma(ent)

            add_ent(standard, ent.type, surface, lemma)

            mycat = guess_custom_category(ent.type, surface)
            if mycat:
                add_ent(custom, mycat, surface, lemma)

        standard_out = {
            k: [{"surface": s, "lemma": l} for (s, l) in sorted(v)]
            for k, v in standard.items()
        }
        custom_out = {
            k: [{"surface": s, "lemma": l} for (s, l) in sorted(v)]
            for k, v in custom.items()
        }

        results.append({
            "url": art.get("url", ""),
            "title": title,
            "content": content,
            "lemmas_title": doc_lemmas(doc_title),
            "lemmas_content": doc_lemmas(doc_content),
            "entities": {
                "standard": standard_out,
                "custom": custom_out
            }
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Lemmas+NER extracted → {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
