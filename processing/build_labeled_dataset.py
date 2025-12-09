import json
from sentence_transformers import SentenceTransformer, util

INPUT_FILE = "data/processed/vidi_clean.json"
OUTPUT_FILE = "data/labeled/final_dataset.json"

# ------------------------------------------
# DEFINICIJE LABELA
# ------------------------------------------

LABELS = {
    "TECH": [
        "tehnologija", "procesor", "računalo", "hardver", "softver",
        "inovacija", "digitalno", "sustav", "aplikacija", "chip",
        "GPU", "CPU", "smartphone", "robot"
    ],
    "AI": [
        "AI", "umjetna inteligencija", "model", "LLM", "neuronska mreža",
        "machine learning", "deep learning"
    ],
    "BUSINESS": [
        "poslovanje", "investicija", "ulaganje", "akvizicija", "tržište",
        "upravljanje", "ekonomija", "financije", "profit"
    ],
    "SCIENCE": [
        "znanost", "istraživanje", "eksperiment", "fizika", "kemija",
        "biologija", "laboratorij"
    ],
    "SPACE": [
        "NASA", "ESA", "svemir", "raketa", "misija", "astronaut", "teleskop",
        "planet", "Mjesec", "Mars"
    ],
    "CYBERSEC": [
        "sigurnost", "cyber", "kibernetički", "napad", "haker", "enkripcija",
        "ranjivost"
    ]
}

# embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding_label(text):
    """Embedding klasifikacija – bira NAJBLIŽU kategoriju samo ako je vrlo sigurna."""
    text_emb = model.encode(text, convert_to_tensor=True)

    best_label = None
    best_score = 0.0

    for label, keywords in LABELS.items():
        group_embedding = model.encode(" ".join(keywords), convert_to_tensor=True)
        score = float(util.cos_sim(text_emb, group_embedding))

        if score > best_score:
            best_score = score
            best_label = label

    # prag sigurnosti
    if best_score >= 0.30:
        return best_label
    return None


def auto_label(article):
    """Dodjeljuje više labela kombinacijom heuristike, NER-a i embeddingsa."""

    text = (article["title"] + " " + article["content"]).lower()
    assigned = set()

    # 1) heuristika po ključnim riječima
    for label, kws in LABELS.items():
        for kw in kws:
            if kw.lower() in text:
                assigned.add(label)
                break

    # 2) embedding klasifikacija
    emb_label = get_embedding_label(article["content"])
    if emb_label:
        assigned.add(emb_label)

    # 3) ako ništa nije pogodilo → OTHER
    if not assigned:
        assigned.add("OTHER")

    return list(assigned)


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        articles = json.load(f)

    output = []
    for art in articles:
        labels = auto_label(art)

        output.append({
            "url": art["url"],
            "title": art["title"],
            "content": art["content"],
            "labels": labels
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Dataset with labels saved → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
