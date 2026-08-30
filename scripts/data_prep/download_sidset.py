"""
Pull a subset of SID-Set WITHOUT downloading the full 140GB / 240k rows.

Streams the parquet shards row by row, keeps only what's needed, stops as soon as
targets are hit. Tampered images (label 2) are skipped entirely -- per
docs/pipeline-decisions.md, they're a different task (localized edit detection),
not whole-image AIGC detection.

Usage:
    python scripts/download_sidset.py
    python scripts/download_sidset.py --n-real 10000 --n-fake 2000
"""
import argparse
from pathlib import Path

from datasets import load_dataset


def download_sidset(out_root: Path, n_real: int, n_fake: int, seed: int = 0) -> None:
    real_dir = out_root / "real"
    fake_dir = out_root / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("saberzl/SID_Set", split="train", streaming=True)
    ds = ds.shuffle(buffer_size=10_000, seed=seed)  # so we're not just taking file order

    n_seen_real, n_seen_fake, n_skipped_tampered = 0, 0, 0

    for row in ds:
        if n_seen_real >= n_real and n_seen_fake >= n_fake:
            break

        label = row["label"]
        if label == 0 and n_seen_real < n_real:
            row["image"].convert("RGB").save(real_dir / f"{row['img_id']}.png")
            n_seen_real += 1
        elif label == 1 and n_seen_fake < n_fake:
            row["image"].convert("RGB").save(fake_dir / f"{row['img_id']}.png")
            n_seen_fake += 1
        elif label == 2:
            n_skipped_tampered += 1
            continue

        if (n_seen_real + n_seen_fake) % 500 == 0:
            print(f"  ...{n_seen_real} real, {n_seen_fake} fake so far")

    print(f"\nDone: {n_seen_real} real, {n_seen_fake} fake saved to {out_root}")
    print(f"({n_skipped_tampered} tampered rows skipped along the way)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/raw/sidset")
    ap.add_argument("--n-real", type=int, default=12_000)
    ap.add_argument("--n-fake", type=int, default=2_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    download_sidset(Path(args.out), args.n_real, args.n_fake, args.seed)


if __name__ == "__main__":
    main()
