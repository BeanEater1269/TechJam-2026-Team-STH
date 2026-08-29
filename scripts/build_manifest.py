"""
Scans data/raw/, tags every image (source, generator_family, label), then does the
stratified 80/10/10 split -- writes data/manifest.csv.

Run this AFTER downloading data, BEFORE running build_cache.py. This is where the
train/val/test decision gets made -- once, on the raw images, before any transform exists.
build_cache.py just reads the "split" column this script writes; it makes no split
decisions of its own.

Usage:
    python scripts/build_manifest.py
    python scripts/build_manifest.py --raw-root data/raw --out data/manifest.csv
"""
import argparse
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


# ---------------------------------------------------------------------------
# One scan_*() function per source. Each just needs to return the same shape:
# a list of dicts with path / source_dataset / generator_family / label.
# ---------------------------------------------------------------------------

def scan_cifake(root: Path) -> list[dict]:
    """CIFAKE: train/ and test/ splits, each with REAL/ and FAKE/ subfolders.
    We ignore CIFAKE's own train/test split -- we're building one 80/10/10 split
    across every source combined, so we just want every image, tagged."""
    rows = []
    for split_dir in ("train", "test"):
        for label_name, label in (("REAL", 0), ("FAKE", 1)):
            folder = root / split_dir / label_name
            if not folder.exists():
                continue
            for p in folder.rglob("*"):
                if p.suffix.lower() in IMG_EXTS:
                    rows.append({
                        "path": str(p),
                        "source_dataset": "cifake",
                        "generator_family": "stable_diffusion" if label == 1 else "real",
                        "label": label,  # 0 = real, 1 = fake
                    })
    return rows


# TODO (teammate): scan_wildfake(), scan_aigibench() -- write these once each is actually
# downloaded and you can see the real folder layout. Each just needs to return the same
# shape of dict as scan_cifake() above: path / source_dataset / generator_family / label.
#
# TODO (teammate): if any additional source gets decided later (still open as of tonight),
# add its own scan_<name>() here too, same shape, same pattern -- then call it in
# build_manifest() below, same as every other source.


def scan_sidset(root: Path) -> list[dict]:
    """SID-Set, as saved by download_sidset.py: real/ and fake/ subfolders directly
    (tampered was never downloaded in the first place, so there's nothing to skip here)."""
    rows = []
    for folder_name, label, family in (("real", 0, "real"), ("fake", 1, "full_synthetic")):
        folder = root / folder_name
        if not folder.exists():
            continue
        for p in folder.rglob("*"):
            if p.suffix.lower() in IMG_EXTS:
                rows.append({
                    "path": str(p),
                    "source_dataset": "sidset",
                    "generator_family": family,
                    "label": label,
                })
    return rows


def add_pixel_size(df: pd.DataFrame) -> pd.DataFrame:
    """Reads each image's native width/height directly off disk. Lazy read -- PIL only
    parses the file header for .size, it doesn't decode the full pixel data, so this stays
    fast even across tens of thousands of images. Applied once, here, to every row --
    individual scan_*() functions don't need to do this themselves."""
    widths, heights = [], []
    for p in df["path"]:
        try:
            with Image.open(p) as im:
                w, h = im.size
        except Exception:
            w, h = None, None
        widths.append(w)
        heights.append(h)
    df = df.copy()
    df["width"] = widths
    df["height"] = heights
    return df


def build_manifest(raw_root: Path) -> pd.DataFrame:
    rows: list[dict] = []

    cifake_root = raw_root / "cifake"
    if cifake_root.exists():
        rows.extend(scan_cifake(cifake_root))
    else:
        print(f"  (skipping cifake -- {cifake_root} not found)")

    sidset_root = raw_root / "sidset"
    if sidset_root.exists():
        rows.extend(scan_sidset(sidset_root))
    else:
        print(f"  (skipping sidset -- {sidset_root} not found)")

    # TODO (teammate): call scan_wildfake(), scan_aigibench(), and any other scan_*()
    # you add, here -- same pattern as cifake/sidset above.

    if not rows:
        raise SystemExit(f"No images found under {raw_root} -- check your downloads landed there")

    df = pd.DataFrame(rows)
    df = add_pixel_size(df)
    print(f"\nFound {len(df)} images total:")
    print(df.groupby(["source_dataset", "generator_family", "label"]).size().to_string())
    print(f"\nPixel-size range: width {df['width'].min()}-{df['width'].max()}, "
          f"height {df['height'].min()}-{df['height'].max()}")
    return df


def add_split(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """80/10/10, stratified by source + generator_family + label together, so no split
    ends up accidentally all-one-source or all-one-label. If this errors with a
    "least populated class" message, some (source, generator_family, label) group is too
    small to stratify on -- either that source needs more images, or the stratification
    needs to fall back to something coarser for that group."""
    strat_key = (
        df["source_dataset"].astype(str) + "_"
        + df["generator_family"].astype(str) + "_"
        + df["label"].astype(str)
    )

    train_idx, temp_idx = train_test_split(
        df.index, test_size=0.2, stratify=strat_key, random_state=seed
    )
    temp_strat = strat_key.loc[temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=temp_strat, random_state=seed
    )

    df = df.copy()
    df["split"] = "train"
    df.loc[val_idx, "split"] = "val"
    df.loc[test_idx, "split"] = "test"
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-root", default="data/raw")
    ap.add_argument("--out", default="data/manifest.csv")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = build_manifest(Path(args.raw_root))
    df = add_split(df, seed=args.seed)

    print("\nSplit sizes:")
    print(df["split"].value_counts().to_string())
    print("\nSplit x label (should look proportional across splits):")
    print(df.groupby(["split", "label"]).size().to_string())

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nWrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
