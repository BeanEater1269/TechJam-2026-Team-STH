"""
Reads manifest.csv, and for every image: applies the right shape-fix, resizes to 512, and
writes clean + all 15 robustness variants into cache/<split>/<image_id>/.

This is the step that turns "downloaded raw images + a split decision" into the actual
files training will read. Run this AFTER build_manifest.py.

Usage:
    python scripts/build_cache.py
    python scripts/build_cache.py --manifest data/manifest.csv --cache-root data/cache/clean
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from transforms import (  # noqa: E402
    WORKING_RES,
    build_all_variants,
    crop_to_square,
    jitter_crop_square,
    resize_to_working,
    t_jpeg_compress,
)


def make_clean_baseline(image: Image.Image, is_real: bool) -> Image.Image:
    """Shape-fix crop applies to ANY non-square image, real or fake -- not just reals.
    Fakes are assumed square by construction (generators default to square canvases), so
    this is a safe no-op for every case tested so far. It exists as a defensive fallback:
    if a future source ever produces a non-square fake, this crops it instead of letting
    resize_to_working() silently squash it (non-uniform stretch -- exactly the visible
    distortion tell the whole crop-not-squash decision was about in the first place).

    From there, real and fake follow the exact same rule: if the (now-square) shortest
    side is under WORKING_RES, resize_to_working() upsamples; at or above, it downsamples.
    One resize call handles both directions.

    The one asymmetry that stays, on purpose: only fakes shrinking down from bigger than
    WORKING_RES get the jitter crop first (breaks the identical-resize-ratio tell). Reals
    never get it -- they already have natural per-photo variation from their own native
    sizes, nothing uniform to break in the first place."""
    w, h = image.size
    if w != h:
        image = crop_to_square(image)

    w, _ = image.size  # square now, either way
    if not is_real and w > WORKING_RES:
        image = jitter_crop_square(image)

    return resize_to_working(image)  # same call, handles upsample and downsample alike


def save_variant(image: Image.Image, path: Path, variant_name: str) -> None:
    """JPEG-100 for everything, including the 4 jpeg_q* variants.

    This used to call image.save(path) with no quality for jpeg_q* names, assuming the
    quality from t_jpeg_compress() would carry through. It doesn't -- PIL Images don't
    remember their own quality after being decoded, so that line was silently falling back
    to PIL's own default (~75) instead of the intended target. Confirmed by testing: q90,
    q50, and q30 all came out 44-50% off from a true single encode at that quality; q70
    only looked right by coincidence, since 75 (the default) happens to sit close to 70.

    The fix: the target-quality compression already happened once, correctly, inside
    t_jpeg_compress() -- that's what shaped these exact pixels. Saving them again at a
    high, faithful quality here preserves that shaping without stacking a second,
    uncontrolled lossy pass on top."""
    image.convert("RGB").save(path, format="JPEG", quality=100)


def process_row(row: pd.Series, cache_root: Path) -> None:
    image = Image.open(row["path"]).convert("RGB")
    is_real = row["label"] == 0

    clean = make_clean_baseline(image, is_real)
    variants = build_all_variants(clean)
    variants["clean"] = clean

    out_dir = cache_root / row["split"] / str(row["img_id"])
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, variant_img in variants.items():
        save_variant(variant_img, out_dir / f"{name}.jpg", name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--cache-root", default="data/cache/clean")
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    if "img_id" not in df.columns:
        # Should not happen with a manifest from the current build_manifest.py -- it
        # writes a source-prefixed, collision-checked img_id itself. This bare-stem
        # fallback only exists for an old/manually-edited manifest missing that column,
        # and is NOT collision-safe: two different sources can share a bare filename
        # stem (e.g. "00048.png" showing up in both gigagan/ and dalle/), which is
        # exactly the bug that caused cached images to silently overwrite each other
        # and manifest.loc[img_id] lookups to return multiple rows. Regenerate the
        # manifest with build_manifest.py instead of relying on this path.
        print("WARNING: manifest has no img_id column -- falling back to bare filename "
              "stems, which are NOT guaranteed unique across sources. Recommend "
              "regenerating the manifest with build_manifest.py instead.")
        df["img_id"] = [Path(p).stem for p in df["path"]]

    cache_root = Path(args.cache_root)
    failed = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="caching"):
        try:
            process_row(row, cache_root)
        except Exception as e:
            failed.append((row["path"], str(e)))

    print(f"\nDone: {len(df) - len(failed)}/{len(df)} images cached "
          f"({16} files each: clean + 15 variants)")
    if failed:
        print(f"{len(failed)} failed:")
        for path, err in failed[:10]:
            print(f"  {path}: {err}")

        # Drop the failed rows from the manifest itself (not just skip them this run) --
        # these are almost always corrupted/truncated source files (bad downloads), not
        # something a retry will fix, so leaving them in manifest.csv just means every
        # future build_cache.py run hits the exact same failures again. Match on "path"
        # since that's the only column guaranteed unique per row across sources.
        failed_paths = {path for path, _ in failed}
        before = len(df)
        df = df[~df["path"].isin(failed_paths)].reset_index(drop=True)
        removed = before - len(df)
        df.to_csv(args.manifest, index=False)
        print(f"\nRemoved {removed} failed row(s) from {args.manifest} "
              f"({len(df)} rows remain)")


if __name__ == "__main__":
    main()