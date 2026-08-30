"""
Stage 0, check #1: sample native image resolutions across your datasets.

Run this before locking a working (caching) resolution. The rule of thumb:
if native sizes cluster around 800-1200px, 1024x1024 is a justified, non-
arbitrary choice. If most images are meaningfully smaller than that,
upsampling to 1024 just burns compute on detail that was never there --
pick a working resolution closer to what the data actually contains.

Usage:
    python check_native_resolution.py --root path/to/wildfake --root path/to/sidset
    python check_native_resolution.py --root path/to/wildfake --n 500
"""
import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def sample_sizes(root: Path, n: int, seed: int = 0) -> list[tuple[int, int]]:
    all_paths = [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS]
    if not all_paths:
        print(f"  no images found under {root}")
        return []
    rng = random.Random(seed)
    sample = rng.sample(all_paths, min(n, len(all_paths)))
    sizes = []
    for p in sample:
        try:
            with Image.open(p) as im:
                sizes.append(im.size)  # (width, height) -- lazy, no full decode
        except Exception as e:
            print(f"  skipped {p.name}: {e}")
    return sizes


def summarize(name: str, sizes: list[tuple[int, int]]) -> None:
    if not sizes:
        return
    widths = np.array([w for w, h in sizes])
    heights = np.array([h for w, h in sizes])
    longest = np.maximum(widths, heights)
    print(f"\n{name}  (n={len(sizes)})")
    print(
        f"  width  - min {widths.min()}, median {int(np.median(widths))}, "
        f"p25 {int(np.percentile(widths, 25))}, p75 {int(np.percentile(widths, 75))}, "
        f"max {widths.max()}"
    )
    print(
        f"  height - min {heights.min()}, median {int(np.median(heights))}, "
        f"p25 {int(np.percentile(heights, 25))}, p75 {int(np.percentile(heights, 75))}, "
        f"max {heights.max()}"
    )
    print(
        f"  longest side - median {int(np.median(longest))}, "
        f"p25 {int(np.percentile(longest, 25))}, p75 {int(np.percentile(longest, 75))}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root",
        action="append",
        required=True,
        help="dataset root folder, repeatable, e.g. --root wildfake/ --root sidset/",
    )
    ap.add_argument("--n", type=int, default=300, help="images to sample per dataset")
    args = ap.parse_args()

    all_sizes: list[tuple[int, int]] = []
    for root in args.root:
        root_path = Path(root)
        sizes = sample_sizes(root_path, args.n)
        summarize(root_path.name or str(root_path), sizes)
        all_sizes.extend(sizes)

    if all_sizes:
        summarize("COMBINED", all_sizes)
        longest = np.array([max(w, h) for w, h in all_sizes])
        median = float(np.median(longest))
        print(f"\nMedian longest side across everything sampled: {median:.0f}px")
        if 800 <= median <= 1200:
            print("-> In the 800-1200 range: 1024x1024 is a justified working resolution.")
        else:
            print(
                f"-> Outside 800-1200: consider a working resolution closer to "
                f"~{median:.0f}px instead of upsampling to 1024 for nothing."
            )


if __name__ == "__main__":
    main()
