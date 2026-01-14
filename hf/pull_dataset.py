import argparse
import os
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", required=True, help="lucijalovric/vidi-news-dataset")
    parser.add_argument("--config", default="clean", help="raw or clean")
    parser.add_argument("--split", default="train")
    parser.add_argument("--out_path", default=os.path.join("data", "processed", "vidi_clean.json"))
    args = parser.parse_args()

    ds = load_dataset(
        args.repo_id,
        name=args.config,
        split=args.split,
        download_mode="force_redownload"
    )

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    # spremi kao JSON (lista zapisa) da ti postojeći kod radi bez promjena
    ds.to_json(args.out_path, orient="records", lines=False, force_ascii=False)

    print(f"✓ Pulled {args.repo_id} ({args.config}/{args.split}) -> {args.out_path} [{len(ds)} rows]")

if __name__ == "__main__":
    main()
