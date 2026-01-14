import json
import os
from clean_text import clean_text

RAW_PATH = os.path.join("data", "raw", "vidi_raw.json")
OUTPUT_PATH = os.path.join("data", "processed", "vidi_clean.json")

def main():
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(f"Raw file not found: {RAW_PATH}")

    os.makedirs(os.path.dirname(RAW_PATH), exist_ok=True)
    with open(RAW_PATH, "r", encoding="utf-8") as f:
        articles = json.load(f)

    cleaned_articles = []
    for art in articles:
        cleaned_content = clean_text(art["content"])
        cleaned_articles.append({
            "url": art["url"],
            "title": art["title"],
            "content": cleaned_content
        })

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(cleaned_articles, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Cleaned articles → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
