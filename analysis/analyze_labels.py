import json
from collections import Counter, defaultdict

DATASET_PATH = "data/labeled/final_dataset.json"

def main():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"\n📊 Total articles: {len(data)}\n")

    label_counter = Counter()
    multi_label = []
    other_only = []

    for art in data:
        labels = art.get("labels", [])
        label_counter.update(labels)

        if len(labels) > 1:
            multi_label.append((art["title"], labels))

        if labels == ["OTHER"]:
            other_only.append(art["title"])

    print("🏷 Label distribution:")
    for label, cnt in label_counter.most_common():
        print(f"  {label:10s} : {cnt}")

    print("\n🔀 Articles with multiple labels:")
    for title, labels in multi_label[:10]:
        print(f"  - {title} → {labels}")
    if len(multi_label) > 10:
        print(f"  ... and {len(multi_label) - 10} more")

    print("\n❓ Articles labeled ONLY as OTHER:")
    for title in other_only[:10]:
        print(f"  - {title}")
    if len(other_only) > 10:
        print(f"  ... and {len(other_only) - 10} more")

if __name__ == "__main__":
    main()
