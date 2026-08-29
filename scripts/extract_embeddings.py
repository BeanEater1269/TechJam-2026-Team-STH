"""
Runs frozen CLIP over every cached image and saves the resulting embeddings.

THIS is the file that actually gets shared with the team -- train.npz, val.npz,
test.npz. Nobody needs the pixel cache once this has run; this is what training
actually reads.

Run this AFTER build_cache.py.

Usage:
    python scripts/extract_embeddings.py
    python scripts/extract_embeddings.py --backbone b32 --cache-root data/cache/clean
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

BACKBONES = {
    "b32": "openai/clip-vit-base-patch32",
    "l14": "openai/clip-vit-large-patch14",
}


def extract_split(
    split_dir: Path, manifest: pd.DataFrame, model, processor, device, batch_size: int
) -> dict:
    image_paths = sorted(split_dir.glob("*/*.jpg"))
    if not image_paths:
        return None

    all_embeds, labels, img_ids, variants, sources, families = [], [], [], [], [], []

    for i in tqdm(range(0, len(image_paths), batch_size), desc=f"  {split_dir.name}"):
        batch_paths = image_paths[i : i + batch_size]
        batch_imgs = [Image.open(p).convert("RGB") for p in batch_paths]

        inputs = processor(images=batch_imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            batch_embeds = model.get_image_features(**inputs).cpu().numpy()

        for path, embed in zip(batch_paths, batch_embeds):
            img_id = path.parent.name
            row = manifest.loc[img_id]
            all_embeds.append(embed)
            labels.append(row["label"])
            img_ids.append(img_id)
            variants.append(path.stem)
            sources.append(row["source_dataset"])
            families.append(row["generator_family"])

    return {
        "embeddings": np.stack(all_embeds),
        "labels": np.array(labels),
        "img_ids": np.array(img_ids),
        "variant": np.array(variants),
        "source_dataset": np.array(sources),
        "generator_family": np.array(families),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--cache-root", default="data/cache/clean")
    ap.add_argument("--out-root", default="data/cache/embeddings")
    ap.add_argument("--backbone", choices=["b32", "l14"], default="b32")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest)
    if "img_id" not in manifest.columns:
        manifest["img_id"] = [Path(p).stem for p in manifest["path"]]
    manifest = manifest.set_index("img_id")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    checkpoint = BACKBONES[args.backbone]
    model = CLIPModel.from_pretrained(checkpoint).to(device).eval()
    processor = CLIPProcessor.from_pretrained(checkpoint)

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        split_dir = Path(args.cache_root) / split
        if not split_dir.exists():
            print(f"  (skipping {split} -- {split_dir} not found)")
            continue
        result = extract_split(split_dir, manifest, model, processor, device, args.batch_size)
        if result is None:
            print(f"  (skipping {split} -- no images found)")
            continue
        out_path = out_root / f"{split}.npz"
        np.savez(out_path, **result)
        print(f"  wrote {out_path}: {result['embeddings'].shape[0]} embeddings, "
              f"dim {result['embeddings'].shape[1]}")

    print(f"\nDone. Send {out_root}/train.npz + val.npz + test.npz to the team -- "
          f"zip or drive link, not git.")


if __name__ == "__main__":
    main()
