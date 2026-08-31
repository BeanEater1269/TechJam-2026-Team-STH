"""
Computes the LIQE quality score (signal D, "if time allows" tier): a pretrained
image-quality model that specifically covers color jitter -- the one degradation
type nothing else on the signal list (Laplacian/DCT/noise, CLIP-drift) touches.

Unlike CLIP-drift, this needs no reference embedding to compare against -- LIQE looks at
ONE image and directly scores it, so it naturally runs across all 16 cached variants per
image (clean + 15 damaged), same as the Laplacian/DCT/noise signals -- not restricted to
clean-only like the CLIP-drift probe had to be for cost reasons.

No training required -- loads a pretrained model, runs inference. requirements.txt's
pyiqa entry was included specifically for this (needs setuptools<81 pinned too --
pyiqa's LIQE pulls in openai's own clip package, which still imports pkg_resources).

Run this AFTER build_cache.py.

Usage:
    python scripts/liqe.py
    python scripts/liqe.py --device cpu
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

try:
    import pyiqa
except ImportError:
    raise SystemExit("pyiqa not installed -- pip install pyiqa (already in requirements.txt)")


def process_split(split_dir: Path, metric):
    image_paths = sorted(split_dir.glob("*/*.jpg"))
    if not image_paths:
        return None

    img_ids, variants, scores = [], [], []
    for p in tqdm(image_paths, desc=f"  {split_dir.name}"):
        score = metric(str(p)).item()
        img_ids.append(p.parent.name)
        variants.append(p.stem)
        scores.append(score)

    return {
        "img_ids": np.array(img_ids),
        "variant": np.array(variants),
        "liqe_score": np.array(scores, dtype=np.float32),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-root", default="data/cache/clean")
    ap.add_argument("--out-root", default="data/cache/stats")
    ap.add_argument("--device", choices=["cuda", "cpu"], default=None,
                     help="Defaults to cuda if available, else cpu. LIQE is cheap "
                          "enough that --device cpu is fine if the GPU is busy training.")
    args = ap.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    metric = pyiqa.create_metric("liqe", device=device, as_loss=False)

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        split_dir = Path(args.cache_root) / split
        if not split_dir.exists():
            print(f"  (skipping {split} -- {split_dir} not found)")
            continue
        result = process_split(split_dir, metric)
        if result is None:
            print(f"  (skipping {split} -- no images found)")
            continue
        out_path = out_root / f"{split}_liqe.npz"
        np.savez(out_path, **result)
        print(f"  wrote {out_path}: {len(result['img_ids'])} rows")


if __name__ == "__main__":
    main()
