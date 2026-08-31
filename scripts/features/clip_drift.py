"""
Computes the CLIP-drift probe (signal C): nudge each cached image variant with a small,
fixed pixel perturbation, re-embed it, and measure how far the embedding moved --
cosine similarity against that SAME variant's embedding extract_embeddings.py already
computed (reused, not recomputed -- only the nudged version needs a fresh CLIP pass).

Real photos are expected to stay more stable under the nudge than fakes (RA-Det's core
finding; this is a simplified, non-learned version of their method -- a small FIXED
perturbation, not their trainable UNet -- see pipeline-decisions.md).

Full version: computed on ALL 16 variants (clean + 15 damaged), not just clean -- a
jpeg_q30 row measures stability of the jpeg_q30 embedding under a further tiny nudge,
not stability of clean. This is the "full 2x-cost version" (16x the CLIP forward passes
of the clean-only fallback) -- deliberately chosen over the cheaper clean-only variant
so the output shape (img_ids + variant + one score column) matches signals.py's
{split}_signals.npz exactly, which is what lets train_concat_normalize_drift.py /
evaluate_concat_drift.py merge this file in by the same (img_id, variant) key they
already use for the 4 B-signals, instead of needing special-case broadcast logic.

Run this AFTER build_cache.py (needs the cached pixel files) and AFTER
extract_embeddings.py (needs its saved embeddings, reused here).

Usage:
    python scripts/features/clip_drift.py
    python scripts/features/clip_drift.py --backbone l14
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

BACKBONES = {
    "b32": "openai/clip-vit-base-patch32",   # 512-dim
    "l14": "openai/clip-vit-large-patch14",  # 768-dim
}

# Half the smallest robustness noise-transform setting (0.02 in transforms.py) --
# meant to be a subtle probe, not a meaningful degradation in its own right.
NUDGE_SIGMA_FRAC = 0.01


def nudge_image(image: Image.Image, sigma_frac: float = NUDGE_SIGMA_FRAC) -> Image.Image:
    """The 'small fixed perturbation'. Same shape as transforms.py's noise transform,
    deliberately smaller than its smallest setting."""
    arr = np.array(image).astype(np.float32)
    sigma = sigma_frac * 255.0
    noise = np.random.normal(0.0, sigma, arr.shape)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def process_split(
    split_dir: Path, emb_data, model, processor, device, batch_size: int
):
    """emb_data is the already-loaded {split}.npz from extract_embeddings.py -- every
    row (all 16 variants per image) gets its own drift score, each nudge compared
    against THAT row's own cached embedding, not always against clean's."""
    img_ids_all = emb_data["img_ids"]
    variant_all = emb_data["variant"]
    embeds_all = emb_data["embeddings"]

    if len(img_ids_all) == 0:
        return None

    img_ids_out, variant_out, drift_scores = [], [], []

    for i in tqdm(range(0, len(img_ids_all), batch_size), desc=f"  {split_dir.name}"):
        batch_ids = img_ids_all[i : i + batch_size]
        batch_variants = variant_all[i : i + batch_size]
        batch_embeds = embeds_all[i : i + batch_size]

        batch_nudged = []
        for img_id, variant in zip(batch_ids, batch_variants):
            img_path = split_dir / str(img_id) / f"{variant}.jpg"
            orig = Image.open(img_path).convert("RGB")
            batch_nudged.append(nudge_image(orig))

        inputs = processor(images=batch_nudged, return_tensors="pt").to(device)
        with torch.no_grad():
            # transformers>=5's get_image_features() returns a BaseModelOutputWithPooling,
            # not a bare tensor -- .pooler_output is the correct embedding accessor
            # (confirmed bit-identical to the classic projected CLIP embedding).
            output = model.get_image_features(**inputs)
            if not isinstance(output, torch.Tensor):
                output = output.pooler_output
            nudged_embeds = output.cpu().numpy()

        for img_id, variant, embed, nudged_emb in zip(batch_ids, batch_variants, batch_embeds, nudged_embeds):
            img_ids_out.append(img_id)
            variant_out.append(variant)
            drift_scores.append(cosine_similarity(embed, nudged_emb))

    return {
        "img_ids": np.array(img_ids_out),
        "variant": np.array(variant_out),
        "clip_drift": np.array(drift_scores, dtype=np.float32),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-root", default="data/cache/clean")
    ap.add_argument("--embeddings-root", default="data/cache/embeddings")
    ap.add_argument("--out-root", default="data/cache/stats")
    ap.add_argument("--backbone", choices=["b32", "l14"], default="l14")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    checkpoint = BACKBONES[args.backbone]
    model = CLIPModel.from_pretrained(checkpoint).to(device).eval()
    processor = CLIPProcessor.from_pretrained(checkpoint)

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        emb_path = Path(args.embeddings_root) / f"{split}.npz"
        if not emb_path.exists():
            print(f"  (skipping {split} -- {emb_path} not found, run extract_embeddings.py first)")
            continue
        emb_data = np.load(emb_path, allow_pickle=True)
        split_dir = Path(args.cache_root) / split

        result = process_split(split_dir, emb_data, model, processor, device, args.batch_size)
        if result is None:
            print(f"  (skipping {split} -- no rows found in {emb_path})")
            continue

        out_path = out_root / f"{split}_drift.npz"
        np.savez(out_path, **result)
        print(f"  wrote {out_path}: {result['clip_drift'].shape[0]} scores "
              f"(mean={result['clip_drift'].mean():.4f}, std={result['clip_drift'].std():.4f})")

    print(f"\nDone. {out_root}/train_drift.npz + val_drift.npz + test_drift.npz -- "
          f"same (img_ids, variant) shape as {{split}}_signals.npz, ready to merge in by "
          f"train_concat_normalize_drift.py / evaluate_concat_drift.py.")


if __name__ == "__main__":
    main()
