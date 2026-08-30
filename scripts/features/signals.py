"""
Computes B -- Classical Degradation Stats -- over every cached image variant:
    - Laplacian variance (blur)
    - DCT energy, low-band and high-band (split, not blended -- SPAI-inspired)
    - Noise variance

Walks the SAME cache folders extract_embeddings.py walks (cache/<split>/<img_id>/*.jpg,
all 16 files -- clean + 15 variants), in the SAME sorted order, so this script's output
lines up 1:1 with extract_embeddings.py's .npz files by (img_id, variant). No manifest
lookup needed here -- these stats are pure pixel measurements, no label required.

Run this AFTER build_cache.py. Can run independently of / in parallel with
extract_embeddings.py (they don't depend on each other, only on the same cache).

Usage:
    python scripts/classical_degradation_stats.py
    python scripts/classical_degradation_stats.py --cache-root data/cache/clean --workers 8
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import scipy.fft
import scipy.ndimage
from PIL import Image
from tqdm import tqdm

# Fraction of each dimension (from the DC corner) counted as "low band" in the DCT
# split. 0.25 means the top-left 25%-by-25% block of frequency coefficients is low,
# everything else is high. This is a fraction, not a fixed pixel count, so it scales
# correctly regardless of image size (all cached images are WORKING_RES x WORKING_RES,
# but this keeps the script correct even if that ever changes).
DCT_LOW_BAND_FRAC = 0.25

# Immerkaer's noise-variance estimator kernel (Immerkaer 1996, "Fast Noise Variance
# Estimation") -- a discrete Laplacian-of-Laplacian operator whose response is
# (almost) pure white-noise energy, largely insensitive to real image structure.
_NOISE_KERNEL = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)


def laplacian_variance(gray: np.ndarray) -> float:
    """Classic blur metric: variance of the image's discrete Laplacian. Sharp images
    have strong edges -> high-variance Laplacian response; blurry images don't."""
    lap = scipy.ndimage.laplace(gray)
    return float(lap.var())


def dct_band_energies(gray: np.ndarray) -> tuple[float, float]:
    """2D DCT of the image, split into low-band and high-band energy (sum of squared
    coefficients), normalized by pixel count so the numbers are comparable regardless
    of resolution. Same DCT computation either way -- this just reads out two sums
    from one transform instead of blending everything into one number."""
    h, w = gray.shape
    dct = scipy.fft.dctn(gray, type=2, norm="ortho")
    lh, lw = max(1, int(h * DCT_LOW_BAND_FRAC)), max(1, int(w * DCT_LOW_BAND_FRAC))

    total_energy = float(np.sum(dct ** 2))
    low_energy = float(np.sum(dct[:lh, :lw] ** 2))
    high_energy = total_energy - low_energy

    n_px = h * w
    return low_energy / n_px, high_energy / n_px


def noise_variance(gray: np.ndarray) -> float:
    """Immerkaer's fast noise-variance estimator. sigma = sqrt(pi/2) * mean(|I * M|)
    over valid convolution positions; we square it to report variance rather than
    standard deviation, to match "Noise Variance" as named in the architecture doc."""
    h, w = gray.shape
    conv = scipy.ndimage.convolve(gray, _NOISE_KERNEL, mode="reflect")
    sigma = np.sqrt(np.pi / 2) * np.sum(np.abs(conv)) / (6 * max(h - 2, 1) * max(w - 2, 1))
    return float(sigma ** 2)


def compute_stats_for_path(path_str: str) -> tuple[float, float, float, float]:
    """Runs all 4 stats for one cached image file. Module-level (not a closure) so it
    can be pickled for ProcessPoolExecutor."""
    with Image.open(path_str) as im:
        gray = np.asarray(im.convert("L"), dtype=np.float64)
    lap_var = laplacian_variance(gray)
    dct_low, dct_high = dct_band_energies(gray)
    noise_var = noise_variance(gray)
    return lap_var, dct_low, dct_high, noise_var


def extract_split(split_dir: Path, workers: int) -> dict | None:
    # Same glob + same sort as extract_embeddings.py's extract_split() -- this is
    # what guarantees row i here describes the same (img_id, variant) as row i in
    # that script's output, without needing an explicit join.
    image_paths = sorted(split_dir.glob("*/*.jpg"))
    if not image_paths:
        return None

    path_strs = [str(p) for p in image_paths]
    results = [None] * len(path_strs)

    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, stats in enumerate(
            tqdm(pool.map(compute_stats_for_path, path_strs, chunksize=32),
                 total=len(path_strs), desc=f"  {split_dir.name}")
        ):
            results[i] = stats

    stats_arr = np.array(results, dtype=np.float64)  # (N, 4)
    return {
        "img_ids": np.array([p.parent.name for p in image_paths]),
        "variant": np.array([p.stem for p in image_paths]),
        "laplacian_var": stats_arr[:, 0],
        "dct_low_energy": stats_arr[:, 1],
        "dct_high_energy": stats_arr[:, 2],
        "noise_variance": stats_arr[:, 3],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-root", default="data/cache/clean")
    ap.add_argument("--out-root", default="data/cache/stats")
    ap.add_argument("--workers", type=int, default=8,
                     help="CPU-bound work -- process pool, not threads, since numpy/"
                          "scipy release the GIL inconsistently across ops.")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        split_dir = Path(args.cache_root) / split
        if not split_dir.exists():
            print(f"  (skipping {split} -- {split_dir} not found)")
            continue
        result = extract_split(split_dir, args.workers)
        if result is None:
            print(f"  (skipping {split} -- no images found)")
            continue
        out_path = out_root / f"{split}.npz"
        np.savez(out_path, **result)
        print(f"  wrote {out_path}: {len(result['img_ids'])} rows")

    print(f"\nDone. {out_root}/train.npz + val.npz + test.npz are ready to merge "
          f"with the CLIP embeddings in concatenate_training.py.")


if __name__ == "__main__":
    main()
