import argparse
import json
import os


def norm_url(u: str) -> str:
    u = (u or "").strip()
    return u[:-1] if u.endswith("/") else u


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_entity_lemmas(article: dict) -> set[str]:
    """
    From data/processed/vidi_lemmas.json (extract_ner.py output):
    collect entity lemma (preferred) else surface; lowercase.
    """
    out = set()
    ents = article.get("entities") or {}

    for block_name in ["standard", "custom"]:
        block = ents.get(block_name) or {}
        for _, items in block.items():
            for it in items or []:
                lemma = (it.get("lemma") or "").strip().lower()
                surf = (it.get("surface") or "").strip().lower()
                term = lemma or surf
                if term and len(term) >= 2:
                    out.add(term)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--links_json", required=True, help="label -> exactly 2 urls")
    ap.add_argument("--ner_path", default=os.path.join("data", "processed", "vidi_lemmas.json"))
    ap.add_argument("--out_path", default=os.path.join("data", "analysis", "strong_indicators.json"))
    args = ap.parse_args()

    label_links = load_json(args.links_json)
    data = load_json(args.ner_path)

    by_url = {}
    for art in data:
        u = norm_url(art.get("url", ""))
        if u:
            by_url[u] = art

    out = {}

    for label, urls in (label_links or {}).items():
        label_u = str(label).strip().upper()
        urls = [norm_url(u) for u in (urls or []) if norm_url(u)]

        if len(urls) != 2:
            out[label_u] = {"urls": urls, "strong_terms": [], "error": "Provide exactly 2 URLs"}
            continue

        a1 = by_url.get(urls[0])
        a2 = by_url.get(urls[1])

        if a1 is None or a2 is None:
            out[label_u] = {
                "urls": urls,
                "strong_terms": [],
                "error": "URL not found in vidi_lemmas.json",
                "missing": [u for u in urls if u not in by_url]
            }
            continue

        s1 = extract_entity_lemmas(a1)
        s2 = extract_entity_lemmas(a2)

        strong = sorted(s1.intersection(s2))

        out[label_u] = {
            "urls": urls,
            "count": len(strong),
            "strong_terms": strong
        }

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Saved -> {args.out_path}")


if __name__ == "__main__":
    main()
