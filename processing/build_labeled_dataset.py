import argparse
import json
import re
import unicodedata
from collections import defaultdict
from functools import lru_cache

import classla
from sentence_transformers import SentenceTransformer, util

INPUT_FILE = "data/processed/vidi_lemmas.json"
OUTPUT_FILE = "data/labeled/final_dataset.json"
DEBUG_EXPLAIN = True
DEBUG_EXPLAIN_FILE = "debug_explain.txt"


# ----------------------------
# STRICT surface matching (no substrings) + folding
# ----------------------------
def _fold(s: str) -> str:
    s = s or ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.lower()


def match_kw(text: str, kw: str) -> bool:
    """
    Case-insensitive, whole-word / whole-phrase match on surface text.
    - prevents: 'esa' matching inside 'mesa'
    - phrase: 'pametni sat' must appear as a separate phrase
    """
    kw = (kw or "").strip()
    if not text or not kw:
        return False

    # diacritic-insensitive
    text_n = _fold(text)
    kw_n = _fold(kw)

    if " " in kw_n:
        pattern = rf"(?<!\w){re.escape(kw_n)}(?!\w)"
    else:
        pattern = rf"\b{re.escape(kw_n)}\b"

    return re.search(pattern, text_n, flags=re.IGNORECASE) is not None


# ----------------------------
# Keyword lemmatization via Classla (NO hardcoding)
# ----------------------------
_NLP = None


def get_kw_nlp():
    global _NLP
    if _NLP is None:
        _NLP = classla.Pipeline(
            lang="hr",
            processors="tokenize,pos,lemma",
            logging_level="WARNING",
        )
    return _NLP


@lru_cache(maxsize=4096)
def kw_to_lemma_parts(kw: str) -> list[str]:
    """
    Lemmatize a keyword (word or phrase) using Classla, to match article lemmas.
    No overrides, no hardcoded special cases.
    """
    kw = (kw or "").strip()
    if not kw:
        return []

    doc = get_kw_nlp()(kw)
    out: list[str] = []
    for sent in doc.sentences:
        for w in sent.words:
            if w.upos != "PUNCT" and w.lemma:
                out.append(w.lemma.lower())
    return out


def match_kw_lemmas(lemmas: list[str], kw: str) -> bool:
    """
    Strict match over lemmas:
    - word: must exist as lemma token
    - phrase: contiguous sequence of lemma tokens must exist
    Keyword is lemmatized via classla.
    """
    if not lemmas:
        return False

    parts = kw_to_lemma_parts(kw)
    if not parts:
        return False

    lem = [x.lower() for x in lemmas if x]

    if len(parts) == 1:
        return parts[0] in lem

    n = len(parts)
    for i in range(len(lem) - n + 1):
        if lem[i:i + n] == parts:
            return True
    return False


def match_kw_any(title: str, content: str, title_lem: list[str], content_lem: list[str], kw: str) -> bool:
    """
    Match keyword either:
    - surface strict, OR
    - lemma strict (classla keyword lemmatization)
    """
    if match_kw(title, kw) or match_kw(content, kw):
        return True
    if match_kw_lemmas(title_lem, kw) or match_kw_lemmas(content_lem, kw):
        return True
    return False


def count_hits(title: str, content: str, title_lem: list[str], content_lem: list[str], kws: list[str]) -> int:
    return sum(1 for w in kws if match_kw_any(title, content, title_lem, content_lem, w))


# ----------------------------
# Evidence helper: show actual surface forms for lemma matches
# (keeps your output structure; only improves "hit list")
# ----------------------------
def _kw_to_flex_surface_pattern(kw: str) -> str | None:
    """
    Build a surface pattern that matches inflected Croatian forms.
    Used ONLY to display what matched in evidence (not for scoring).
    Example: "računalni virus" matches "računalnih virusa".
    """
    kw = (kw or "").strip()
    if not kw:
        return None

    parts = _fold(kw).split()
    if not parts:
        return None

    stems = [(p[:6] if len(p) >= 6 else p) for p in parts]

    if len(stems) == 1:
        return rf"\b{re.escape(stems[0])}\w*\b"

    return r"\b" + r"\s+".join([rf"{re.escape(st)}\w*" for st in stems]) + r"\b"


def find_surface_hit(text: str, kw: str) -> str | None:
    """
    Return first matched surface substring from the ORIGINAL text (so you see real form).
    Works on folded text for matching, uses spans to slice original.
    (Note: with unicodedata folding, spans can be off in rare cases; usually OK for cro diacritics.)
    """
    text = text or ""
    patt = _kw_to_flex_surface_pattern(kw)
    if not text or not patt:
        return None

    text_n = _fold(text)
    m = re.search(patt, text_n, flags=re.IGNORECASE)
    if not m:
        return None

    frag = text[m.start():m.end()].strip()
    return frag if frag else None


# ----------------------------
# Keyword sets
# ----------------------------
LABELS = {
    "TECH": [
        "tehnologija", "procesor", "računalo", "hardver", "softver",
        "inovacija", "digitalno", "aplikacija", "chip",
        "GPU", "CPU", "smartphone", "robot", "gaming",
        "pametni sat", "pametni telefon", "smartwatch"
    ],
    "AI": [
        "AI", "artificial intelligence", "UI", "umjetna inteligencija",
        "LLM", "neuronska mreža", "deep seek", "generativni model",
        "machine learning", "deep learning"
    ],
    "BUSINESS": [
        "poslovanje", "investicija", "ulaganje", "akvizicija", "tržište",
        "upravljanje", "ekonomija", "financije", "profit"
    ],
    "SCIENCE": [
        "znanost", "istraživanje", "eksperiment", "fizika", "kemija", "biologija", 
        "geofizika", "seizmologija", "laboratorij", "proučavanje", "istraživač", "promatrač"
    ],
    "SPACE": [
        "NASA", "ESA", "svemir", "raketa", "svemirska misija", "astronaut", "teleskop", "astronomija",
        "satelit", "orbita", "svemirski rover", "svemirska sonda", "Mars", "Venera", "Jupiter", "Saturn",
        "zvijezda", "planet", "galaksija", "sunčev sustav", "asteroid", "svemirski brod"
    ],
    "CYBERSEC": [
        "informatička sigurnost", "cyber", "kibernetički", "cybernapad",
        "haker", "enkripcija", "dekripcija", "ranjivost", "prijetnja", "protokoli",
        "zaštita od virusa", "računalni virus", "malware", "ransomware", "phishing",
        "firewall", "antivirus", "antivirusni"
    ],
}

AI_STRONG = [
    "AI", "umjetna inteligencija", "artificial intelligence",
    "LLM", "neuronska mreža", "machine learning", "deep learning",
    "generativni", "ChatGPT", "transformer", "Transformers"
]

BUSINESS_STRONG = [
    "investicija", "ulaganje", "akvizicija", "preuzimanje", "spajanje",
    "profit", "prihod", "zarada", "gubitak", "dionica", "dionice",
    "burza", "IPO", "tržište", "financije", "kredit", "kapital",
    "poslovanje", "tvrtka", "kompanija"
]

SPACE_STRONG = [
    "NASA", "ESA", "svemir", "raketa", "astronaut", "međunarodna svemirska stanica",
    "teleskop", "satelit", "orbita", "svemirski", "svemirski rover", "svemirska sonda"
]

SPACE_WEAK = [
    "planet", "zvijezda", "galaksija", "lansiranje",
    "Mars", "Venera", "Zemlja", "Jupiter", "Saturn", "Mjesec"
]


# ----------------------------
# Embedding model
# ----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")
LABEL_TEXT = {lab: " ".join(kws) for lab, kws in LABELS.items()}
LABEL_EMB = {lab: model.encode(txt, convert_to_tensor=True) for lab, txt in LABEL_TEXT.items()}


# ----------------------------
# Embedding helpers
# ----------------------------
def _embedding_scores(text: str) -> dict[str, float]:
    text = text or ""
    text_emb = model.encode(text, convert_to_tensor=True)
    return {label: float(util.cos_sim(text_emb, emb)) for label, emb in LABEL_EMB.items()}


def get_embedding_label(text: str, threshold: float) -> str | None:
    scores = _embedding_scores(text)
    if not scores:
        return None
    best_label, best_score = max(scores.items(), key=lambda x: x[1])
    return best_label if best_score >= threshold else None


# ----------------------------
# Explainability helpers
# ----------------------------
def _top_kw_hits_any(title: str, content: str, title_lem: list[str], content_lem: list[str], kws: list[str]) -> dict:
    """
    Keep your previous evidence structure (lists of strings),
    but if a hit happens only via lemmas, try to display the real surface form.
    """
    title_hits = []
    content_hits = []

    for kw in kws:
        if match_kw(title, kw):
            title_hits.append(kw)
        elif match_kw_lemmas(title_lem, kw):
            # show actual form if possible (e.g. "računalnih virusa")
            surf = find_surface_hit(title, kw)
            title_hits.append(surf if surf else kw)

        if match_kw(content, kw):
            content_hits.append(kw)
        elif match_kw_lemmas(content_lem, kw):
            surf = find_surface_hit(content, kw)
            content_hits.append(surf if surf else kw)

    return {
        "title_hits": title_hits,
        "content_hits": content_hits,
        "title_count": len(title_hits),
        "content_count": len(content_hits),
        "total": len(set(title_hits + content_hits)),
    }


def explain_labels(
    article: dict,
    threshold: float = 0.35,
    max_labels: int = 3,
    include_embedding_ranking: bool = True,
    as_text: bool = False,
) -> dict | str:
    title = (article.get("title") or "")
    content = (article.get("content") or "")

    title_lem = article.get("lemmas_title") or []
    content_lem = article.get("lemmas_content") or []

    scores = defaultdict(float)

    # 1) keyword heuristics
    keyword_evidence: dict[str, dict] = {}
    for label, kws in LABELS.items():
        keyword_evidence[label] = _top_kw_hits_any(title, content, title_lem, content_lem, kws)

        scored = False
        for kw in kws:
            # title stronger
            if match_kw(title, kw) or match_kw_lemmas(title_lem, kw):
                scores[label] += 2.0
                scored = True
                break
            if match_kw(content, kw) or match_kw_lemmas(content_lem, kw):
                scores[label] += 1.0
                scored = True
                break

        if scored:
            trigger_kw, trigger_where = None, None
            for kw in kws:
                if match_kw(title, kw) or match_kw_lemmas(title_lem, kw):
                    trigger_kw, trigger_where = kw, "title"
                    break
                if match_kw(content, kw) or match_kw_lemmas(content_lem, kw):
                    trigger_kw, trigger_where = kw, "content"
                    break
            keyword_evidence[label]["scoring_trigger"] = {
                "keyword": trigger_kw,
                "where": trigger_where,
                "points": 2.0 if trigger_where == "title" else 1.0
            }
        else:
            keyword_evidence[label]["scoring_trigger"] = None

    # 2) embedding boost (UNCHANGED output structure)
    emb_scores = _embedding_scores(content)
    best_label = max(emb_scores.items(), key=lambda x: x[1])[0] if emb_scores else None
    best_score = emb_scores.get(best_label, None) if best_label else None

    embedding_evidence = {
        "best_label": best_label,
        "best_score": best_score,
        "threshold": threshold,
        "boost_applied": False,
        "boost_points": 1.5,
        "ranking": None
    }

    if include_embedding_ranking and emb_scores:
        embedding_evidence["ranking"] = sorted(emb_scores.items(), key=lambda x: x[1], reverse=True)

    if best_label is not None and best_score is not None and best_score >= threshold:
        scores[best_label] += 1.5
        embedding_evidence["boost_applied"] = True

    # 3) gating / penalties (UNCHANGED)
    gating_penalties: list[dict] = []

    if "AI" in scores:
        before = scores["AI"]
        ai_hits = count_hits(title, content, title_lem, content_lem, AI_STRONG)
        if ai_hits < 1:
            scores["AI"] *= 0.2
            gating_penalties.append({
                "label": "AI",
                "rule": "AI gating (soft): requires >=1 strong AI hit",
                "ai_strong_hits": ai_hits,
                "before": before,
                "after": scores["AI"],
            })
        else:
            gating_penalties.append({
                "label": "AI",
                "rule": "AI gating (soft): passed (>=1 strong AI hit)",
                "ai_strong_hits": ai_hits,
                "before": before,
                "after": scores["AI"],
            })

    if "TECH" in scores:
        before = scores["TECH"]
        scores["TECH"] *= 0.85
        gating_penalties.append({
            "label": "TECH",
            "rule": "TECH penalty: broad label downweighted",
            "before": before,
            "after": scores["TECH"],
        })

    if "BUSINESS" in scores:
        before = scores["BUSINESS"]
        biz_hits = count_hits(title, content, title_lem, content_lem, BUSINESS_STRONG)
        if biz_hits < 1:
            scores["BUSINESS"] *= 0.2
            gating_penalties.append({
                "label": "BUSINESS",
                "rule": "BUSINESS gating (soft): requires >=1 strong business hit",
                "business_strong_hits": biz_hits,
                "before": before,
                "after": scores["BUSINESS"],
            })
        else:
            gating_penalties.append({
                "label": "BUSINESS",
                "rule": "BUSINESS gating (soft): passed (>=1 strong business hit)",
                "business_strong_hits": biz_hits,
                "before": before,
                "after": scores["BUSINESS"],
            })

    if "SPACE" in scores:
        before = scores["SPACE"]
        strong_hits = count_hits(title, content, title_lem, content_lem, SPACE_STRONG)
        weak_hits = count_hits(title, content, title_lem, content_lem, SPACE_WEAK)
        if strong_hits < 1 and weak_hits < 2:
            scores["SPACE"] = 0.0
            gating_penalties.append({
                "label": "SPACE",
                "rule": "SPACE gating: requires (>=1 strong) OR (>=2 weak) hits",
                "space_strong_hits": strong_hits,
                "space_weak_hits": weak_hits,
                "before": before,
                "after": scores["SPACE"],
            })
        else:
            gating_penalties.append({
                "label": "SPACE",
                "rule": "SPACE gating: passed (>=1 strong OR >=2 weak)",
                "space_strong_hits": strong_hits,
                "space_weak_hits": weak_hits,
                "before": before,
                "after": scores["SPACE"],
            })

    # 4) select labels
    items = [(lab, sc) for lab, sc in scores.items() if sc > 0.0]
    if not items:
        result = {
            "selected_labels": ["OTHER"],
            "final_scores": dict(scores),
            "keyword_evidence": keyword_evidence,
            "embedding_evidence": embedding_evidence,
            "gating_penalties": gating_penalties,
        }
        if as_text:
            return "Selected labels: OTHER\nReason: no label achieved positive score after rules.\n"
        return result

    top = sorted(items, key=lambda x: x[1], reverse=True)[:max_labels]
    selected = [lab for lab, _ in top]

    result = {
        "selected_labels": selected,
        "final_scores": dict(scores),
        "keyword_evidence": keyword_evidence,
        "embedding_evidence": embedding_evidence,
        "gating_penalties": gating_penalties,
    }

    if not as_text:
        return result

    # Pretty text output (KEEP SAME STRUCTURE AS YOUR CURRENT ONE)
    lines = []
    lines.append("=" * 80)
    lines.append(f"TITLE: {title}")
    if article.get("url"):
        lines.append(f"URL:   {article.get('url')}")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Selected labels: {', '.join(selected)}")
    lines.append("")
    lines.append("Final scores (top):")
    for lab, sc in sorted(items, key=lambda x: x[1], reverse=True)[:10]:
        lines.append(f"  - {lab}: {sc:.3f}")
    lines.append("")

    lines.append("Keyword evidence (labels with hits):")
    for lab in LABELS.keys():
        ev = keyword_evidence.get(lab, {})
        if not ev:
            continue
        if ev.get("title_count", 0) == 0 and ev.get("content_count", 0) == 0:
            continue

        trig = ev.get("scoring_trigger")
        marker = " (SELECTED)" if lab in selected else ""
        if trig:
            lines.append(f"  - {lab}{marker}: trigger='{trig['keyword']}' in {trig['where']} (+{trig['points']})")
        else:
            lines.append(f"  - {lab}{marker}: hits exist (no trigger?)")
        if ev.get("title_hits"):
            lines.append(f"      title hits: {', '.join(ev['title_hits'][:12])}" + (" ..." if len(ev['title_hits']) > 12 else ""))
        if ev.get("content_hits"):
            lines.append(f"      content hits: {', '.join(ev['content_hits'][:12])}" + (" ..." if len(ev['content_hits']) > 12 else ""))
    lines.append("")

    lines.append("Embedding evidence:")
    if embedding_evidence["best_label"] is None or embedding_evidence["best_score"] is None:
        lines.append("  - no embedding scores computed")
    else:
        lines.append(
            f"  - best_label={embedding_evidence['best_label']}, best_score={embedding_evidence['best_score']:.3f}, threshold={embedding_evidence['threshold']:.3f}"
        )
        lines.append(
            f"  - boost_applied={embedding_evidence['boost_applied']} (+{embedding_evidence['boost_points']})"
        )
        if include_embedding_ranking and embedding_evidence["ranking"]:
            top5 = embedding_evidence["ranking"][:5]
            lines.append("  - top similarities:")
            for lab, sc in top5:
                lines.append(f"      {lab}: {sc:.3f}")
    lines.append("")

    if gating_penalties:
        lines.append("Gating / penalties:")
        for g in gating_penalties:
            lines.append(f"  - {g.get('label')}: {g.get('rule')} ({g.get('before'):.3f} → {g.get('after'):.3f})")
    else:
        lines.append("Gating / penalties: none")

    return "\n".join(lines)


# ----------------------------
# Labeling logic
# ----------------------------
def auto_label(article: dict, threshold: float, max_labels: int = 3) -> list[str]:
    title = (article.get("title") or "")
    content = (article.get("content") or "")

    title_lem = article.get("lemmas_title") or []
    content_lem = article.get("lemmas_content") or []

    scores = defaultdict(float)

    # 1) keyword heuristics: title stronger
    for label, kws in LABELS.items():
        for kw in kws:
            if match_kw(title, kw) or match_kw_lemmas(title_lem, kw):
                scores[label] += 2.0
                break
            if match_kw(content, kw) or match_kw_lemmas(content_lem, kw):
                scores[label] += 1.0
                break

    # 2) embedding boost
    emb_label = get_embedding_label(content, threshold=threshold)
    if emb_label:
        scores[emb_label] += 1.5

    # 3) gating / penalties
    if "AI" in scores:
        ai_hits = count_hits(title, content, title_lem, content_lem, AI_STRONG)
        if ai_hits < 1:
            scores["AI"] *= 0.2

    if "TECH" in scores:
        scores["TECH"] *= 0.85

    if "BUSINESS" in scores:
        biz_hits = count_hits(title, content, title_lem, content_lem, BUSINESS_STRONG)
        if biz_hits < 1:
            scores["BUSINESS"] *= 0.2

    if "SPACE" in scores:
        strong_hits = count_hits(title, content, title_lem, content_lem, SPACE_STRONG)
        weak_hits = count_hits(title, content, title_lem, content_lem, SPACE_WEAK)
        if strong_hits < 1 and weak_hits < 2:
            scores["SPACE"] = 0.0

    # 4) select labels
    items = [(lab, sc) for lab, sc in scores.items() if sc > 0.0]
    if not items:
        return ["OTHER"]

    top = sorted(items, key=lambda x: x[1], reverse=True)[:max_labels]
    return [lab for lab, _ in top]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--max_labels", type=int, default=3)
    args = parser.parse_args()

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        articles = json.load(f)

    output = []

    if DEBUG_EXPLAIN:
        with open(DEBUG_EXPLAIN_FILE, "w", encoding="utf-8") as dbg:
            dbg.write("[DEBUG EXPLAIN LOG]\n\n")

            for art in articles:
                if not (art.get("title") or "").strip():
                    art["title"] = "(no title)"

                labels = auto_label(art, threshold=args.threshold, max_labels=args.max_labels)

                dbg.write(explain_labels(art, threshold=args.threshold, max_labels=args.max_labels, as_text=True))
                dbg.write("\n\n")

                output.append({
                    "url": art.get("url", ""),
                    "title": art.get("title", ""),
                    "content": art.get("content", ""),
                    "labels": labels
                })
    else:
        for art in articles:
            if not (art.get("title") or "").strip():
                art["title"] = "(no title)"

            labels = auto_label(art, threshold=args.threshold, max_labels=args.max_labels)

            output.append({
                "url": art.get("url", ""),
                "title": art.get("title", ""),
                "content": art.get("content", ""),
                "labels": labels
            })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Dataset with labels saved → {OUTPUT_FILE} [{len(output)} rows]")


if __name__ == "__main__":
    main()
