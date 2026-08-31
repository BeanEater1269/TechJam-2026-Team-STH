"""
Scans all dataset directories under data/raw/, tags every image (source,
generator_family, label), and builds a stratified 80/10/10 split in data/manifest.csv.

Supported directory format per dataset under data/raw/<dataset_name>/:
    - Must contain 'real' (or 'REAL') and/or 'fake' (or 'FAKE') subfolders
      (can be directly inside or nested under split subfolders like train/REAL).
    - If a subfolder is missing or empty, that category counts as 0 images.

Usage:
    python scripts/build_manifest.py
    python scripts/build_manifest.py --raw-dir data/raw --out data/manifest.csv
"""
import argparse
import random
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# Optional per-source sampling caps.
# Datasets not listed here will keep 100% of their valid scanned images.
SAMPLE_CAPS = {
    "cifake": {0: 3000, 1: 3000},  # 3000 real (0), 3000 fake (1)
    "stylegan_xl": 2500,
    "gigagan": 2500,
    "dalle": 2500,
    "imagen3": 2500,
}

# Optional explicit generator family mapping (falls back to dataset folder name if missing)
GENERATOR_FAMILIES = {
    "cifake": "stable_diffusion",
    "stylegan_xl": "stylegan_xl",
    "gigagan": "gigagan",
    "dalle": "dalle",
    "imagen3": "imagen3",
    "sidset": "full_synthetic",
}


def scan_dataset_dir(dataset_dir: Path) -> list[dict]:
    """Scans a single dataset folder. Inspects subfolder path parts for 'real' or 'fake'
    case-insensitively. Images outside both folders are ignored."""
    dataset_name = dataset_dir.name.lower()
    gen_family = GENERATOR_FAMILIES.get(dataset_name, dataset_name)
    rows = []

    for p in dataset_dir.rglob("*"):
        if p.suffix.lower() not in IMG_EXTS:
            continue

        # Extract parent subfolder names relative to the dataset root
        rel_parts = [part.lower() for part in p.relative_to(dataset_dir).parts[:-1]]

        if "real" in rel_parts:
            label = 0
            family = "real"
        elif "fake" in rel_parts:
            label = 1
            family = gen_family
        else:
            # Not under a real/ or fake/ subfolder -- skip
            continue

        rows.append({
            "path": str(p.resolve()),
            "source_dataset": dataset_name,
            "generator_family": family,
            "label": label,
        })

    return rows


def make_unique_img_ids(df: pd.DataFrame) -> pd.Series:
    """Builds a collision-safe img_id unique across all datasets."""
    base = df["source_dataset"] + "_" + df["path"].apply(lambda p: Path(p).stem)
    seen: dict[str, int] = {}
    ids = []
    for b in base:
        n = seen.get(b, 0)
        seen[b] = n + 1
        ids.append(b if n == 0 else f"{b}_{n}")
    return pd.Series(ids, index=df.index)


def is_readable(path: str) -> bool:
    """Decodes image fully to catch corrupted files prior to caching."""
    try:
        with Image.open(path) as im:
            im.load()
        return True
    except Exception:
        return False


def _take_valid_sample(rows: list[dict], cap: int | None, rng: random.Random) -> tuple[list[dict], int]:
    shuffled = rows[:]
    rng.shuffle(shuffled)

    kept: list[dict] = []
    skipped = 0
    for r in shuffled:
        if cap is not None and len(kept) >= cap:
            break
        if is_readable(r["path"]):
            kept.append(r)
        else:
            skipped += 1
    return kept, skipped


def add_pixel_size(df: pd.DataFrame) -> pd.DataFrame:
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


def build_manifest(raw_root: Path, sample_seed: int = 0) -> pd.DataFrame:
    rows: list[dict] = []
    rng = random.Random(sample_seed)

    if not raw_root.exists():
        raise SystemExit(f"Raw root folder not found: {raw_root}")

    dataset_dirs = sorted([d for d in raw_root.iterdir() if d.is_dir()])
    if not dataset_dirs:
        raise SystemExit(f"No dataset subdirectories found inside {raw_root}")

    print(f"Scanning raw dataset directory: {raw_root}\n")

    for dataset_dir in dataset_dirs:
        name = dataset_dir.name.lower()
        source_rows = scan_dataset_dir(dataset_dir)
        if not source_rows:
            print(f"  (skipping {name} -- no images in real/ or fake/ subfolders)")
            continue

        cap = SAMPLE_CAPS.get(name)

        if isinstance(cap, dict):
            capped_rows = []
            for label, n in cap.items():
                label_rows = [r for r in source_rows if r["label"] == label]
                valid_rows, skipped = _take_valid_sample(label_rows, n, rng)
                if skipped:
                    print(f"  {name} (label={label}): skipped {skipped} unreadable file(s)")
                print(f"  {name} (label={label}): found {len(label_rows)}, keeping {len(valid_rows)} readable")
                capped_rows.extend(valid_rows)
            source_rows = capped_rows
        else:
            valid_rows, skipped = _take_valid_sample(source_rows, cap, rng)
            if skipped:
                print(f"  {name}: skipped {skipped} unreadable file(s)")
            print(f"  {name}: found {len(source_rows)}, keeping {len(valid_rows)} readable")
            source_rows = valid_rows

        rows.extend(source_rows)

    if not rows:
        raise SystemExit("No valid images found across any raw subfolders.")

    df = pd.DataFrame(rows)
    df["img_id"] = make_unique_img_ids(df)
    df = add_pixel_size(df)

    print(f"\nFound {len(df)} images total:")
    print(df.groupby(["source_dataset", "generator_family", "label"]).size().to_string())
    print(f"\nPixel-size range: width {df['width'].min()}-{df['width'].max()}, "
          f"height {df['height'].min()}-{df['height'].max()}")
    return df


def add_split(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
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
    project_root = Path(__file__).resolve().parents[1]
    default_raw = project_root / "data" / "raw"
    default_out = project_root / "data" / "manifest.csv"

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", default=str(default_raw), help="Path to data/raw directory")
    ap.add_argument("--out", default=str(default_out), help="Output manifest.csv path")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = build_manifest(raw_root=Path(args.raw_dir), sample_seed=args.seed)
    df = add_split(df, seed=args.seed)

    print("\nSplit sizes:")
    print(df["split"].value_counts().to_string())
    print("\nSplit x label:")
    print(df.groupby(["split", "label"]).size().to_string())

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nWrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()