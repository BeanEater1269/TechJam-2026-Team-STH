"""
Builds the robustness cache (clean + 15 transform variants) from an arbitrary
folder of images -- NOT from manifest.csv. For external / inference-time image
sets (e.g. a judge's submitted photos, an external validation folder) that were
never part of the manifest -> build_cache training pipeline.

Recursively finds every image under --input-dir, at ANY subfolder depth.

LABEL / is_real HANDLING -- this matters for correctness, not just bookkeeping:
build_cache.py's make_clean_baseline() applies jitter_crop_square() ONLY to fakes
larger than WORKING_RES, specifically to break the identical-resize-ratio
fingerprint a generator's fixed-size fake images would otherwise carry. That step
is a FAKES-ONLY anti-fingerprint step -- it must NOT be applied to an image whose
true label is unknown, since blindly jitter-cropping an unknown image applies
training-time fake-specific preprocessing to something that might actually be
real, mismatching what the classifier was trained to expect at inference time.

So:
  - If --input-dir has real/ and fake/ subfolders (same convention used
    elsewhere in this project), images are labeled accordingly and preprocessed
    exactly like training data of that label.
  - Otherwise (a flat folder -- e.g. a judge's raw phone photos, true label
    unknown), EVERY image goes through the "real" branch: shape-fix crop only,
    no jitter_crop_square. This keeps preprocessing parity with training without
    ever guessing a label the pipeline doesn't actually have.

Writes an id_mapping.csv into the output folder (img_id -> original path, label)
since there's no manifest.csv here to trace img_id back to a source file.

Run this instead of build_cache.py for ad-hoc / external image folders.

Usage:
    python scripts/test/build_cache_external.py --input-dir "C:\\path\\to\\images"
    python scripts/test/build_cache_external.py --input-dir "..." --split-name judge_photos --cache-root data/cache/clean
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm


def _find_src_dir() -> Path:
    """Walks up from this file looking for a 'src' directory, rather than
    assuming a fixed number of parent levels -- build_cache.py's own import
    depth already changed once (parent.parent -> parent.parent.parent) as the
    repo's scripts/ layout evolved, so a hardcoded depth here would be fragile
    to wherever this script ends up living."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "src"
        if candidate.exists():
            return candidate
    raise SystemExit("Could not locate a 'src' directory above this script -- "
                      "adjust the sys.path.insert line manually.")


SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from transforms import (  # noqa: E402
    WORKING_RES,
    build_all_variants,
    crop_to_square,
    jitter_crop_square,
    resize_to_working,
)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def make_clean_baseline(image: Image.Image, is_real: bool) -> Image.Image:
    """Identical logic to build_cache.py's make_clean_baseline() -- duplicated
    here rather than imported, since this script is meant to run standalone
    against any folder without depending on build_cache.py's manifest-row-based
    structure. Keep both in sync if the shape-fix logic ever changes."""
    w, h = image.size
    if w != h:
        image = crop_to_square(image)
    w, _ = image.size
    if not is_real and w > WORKING_RES:
        image = jitter_crop_square(image)
    return resize_to_working(image)


def save_variant(image: Image.Image, path: Path) -> None:
    image.convert("RGB").save(path, format="JPEG", quality=100)


def scan_input_dir(root: Path) -> list[tuple[Path, bool | None]]:
    """Returns (path, is_real) pairs. is_real is True/False if real/fake
    subfolders are found; otherwise None for every image in a flat/unlabeled
    folder. None is handled explicitly in process_image() as "unknown label ->
    use the real branch" -- not silently defaulted here, so the decision stays
    visible and traceable rather than buried in a fallback."""
    real_dir, fake_dir = root / "real", root / "fake"
    if real_dir.exists() or fake_dir.exists():
        pairs = []
        for folder, is_real in ((real_dir, True), (fake_dir, False)):
            if not folder.exists():
                continue
            for p in sorted(folder.rglob("*")):
                if p.suffix.lower() in IMG_EXTS:
                    pairs.append((p, is_real))
        return pairs

    paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS)
    return [(p, None) for p in paths]


def make_unique_img_ids(paths: list[Path], root: Path) -> list[str]:
    """Path-relative-to-root with separators flattened to underscores as the
    base id, plus a collision-safe counter suffix -- same defensive pattern as
    build_manifest.py's make_unique_img_ids(), since nested subfolders can still
    reuse filenames across different subfolders (e.g. two "IMG_0001.jpg" from
    different phone albums dumped into the same parent folder)."""
    seen: dict[str, int] = {}
    ids = []
    for p in paths:
        base = "_".join(p.relative_to(root).with_suffix("").parts)
        n = seen.get(base, 0)
        seen[base] = n + 1
        ids.append(base if n == 0 else f"{base}_{n}")
    return ids


def process_image(path: Path, is_real: bool | None, img_id: str, out_root: Path) -> None:
    image = Image.open(path).convert("RGB")
    # Unknown label -> "real" branch (no jitter_crop_square). See module docstring.
    effective_is_real = True if is_real is None else is_real

    clean = make_clean_baseline(image, effective_is_real)
    variants = build_all_variants(clean)
    variants["clean"] = clean

    out_dir = out_root / img_id
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, variant_img in variants.items():
        save_variant(variant_img, out_dir / f"{name}.jpg")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--cache-root", default="data/cache/clean")
    ap.add_argument("--split-name", default="external",
                     help="Subfolder name under --cache-root this gets written to. "
                          "NOT a real train/val/test split -- just a label so this "
                          "doesn't collide with the manifest-driven cache.")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise SystemExit(f"--input-dir not found: {input_dir}")

    pairs = scan_input_dir(input_dir)
    if not pairs:
        raise SystemExit(f"No images found under {input_dir}")

    paths = [p for p, _ in pairs]
    labels = [is_real for _, is_real in pairs]
    img_ids = make_unique_img_ids(paths, input_dir)

    if all(l is None for l in labels):
        print(f"Found {len(paths)} images, flat/unlabeled folder -- ALL will use "
              f"the 'real' preprocessing branch (no jitter_crop_square), since "
              f"true label is unknown. See module docstring for why.")
    else:
        print(f"Found {len(paths)} images with real/fake subfolders detected -- "
              f"preprocessed per their actual label.")

    out_root = Path(args.cache_root) / args.split_name
    out_root.mkdir(parents=True, exist_ok=True)

    failed = []
    for path, is_real, img_id in tqdm(list(zip(paths, labels, img_ids)), desc=args.split_name):
        try:
            process_image(path, is_real, img_id, out_root)
        except Exception as e:
            failed.append((str(path), str(e)))

    print(f"\nDone: {len(paths) - len(failed)}/{len(paths)} images cached "
          f"(16 files each: clean + 15 variants)")
    if failed:
        print(f"{len(failed)} failed:")
        for path, err in failed[:10]:
            print(f"  {path}: {err}")

    # No manifest.csv exists for this folder -- write a small mapping so img_id
    # can still be traced back to its original file (and detected label, if any).
    kept = [(i, p, l) for i, p, l in zip(img_ids, paths, labels)
            if str(p) not in {fp for fp, _ in failed}]
    mapping = pd.DataFrame({
        "img_id": [i for i, _, _ in kept],
        "original_path": [str(p) for _, p, _ in kept],
        "label": ["real" if l is True else "fake" if l is False else "unknown"
                  for _, _, l in kept],
    })
    mapping_path = out_root / "id_mapping.csv"
    mapping.to_csv(mapping_path, index=False)
    print(f"Wrote {mapping_path} ({len(mapping)} rows) for img_id -> original "
          f"file traceability")


if __name__ == "__main__":
    main()