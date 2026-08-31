"""
Extracts CLIP embeddings from a plain external image folder -- NOT from the
manifest/build_cache pipeline. No labels, no train/val/test split, no 16 cached
variants: just every image under --image-dir, embedded once each.

This is for external validation sets (e.g. FakeXplain) that live outside your own
raw-data pipeline entirely -- a one-off embedding extraction, not part of the
manifest -> cache -> extract_embeddings flow the rest of the project uses.

IMPORTANT: --backbone MUST match whatever backbone your train/val/test embeddings
were extracted with (see extract_embeddings.py's --backbone). Embeddings from
different backbones have different dimensions (b32 -> 512, l14 -> 768) and are NOT
compatible with a classifier trained on the other one. Defaults to l14 here since
that's the current default in extract_embeddings.py -- change both together if you
ever switch backbones.

If --image-dir has real/ and fake/ subfolders, labels are attached (0=real,
1=fake), same convention as build_manifest.py's source scanners. If it's just a
flat folder of images (as FakeXplain/batch_b is), no labels are attached -- this is
an unlabeled external set, evaluated by inspecting predictions, not accuracy.

Usage:
    python scripts/extract_embeddings_external.py
    python scripts/extract_embeddings_external.py --image-dir "some/other/folder" --out "some/other/out.npz"
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

BACKBONES = {
    "b32": "openai/clip-vit-base-patch32",
    "l14": "openai/clip-vit-large-patch14",
}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def scan_folder(root: Path) -> tuple[list[Path], np.ndarray | None]:
    """Same real/fake-subfolder check used throughout build_manifest.py's source
    scanners -- attach labels if that structure exists, otherwise treat this as an
    unlabeled external set (which is what FakeXplain/batch_b actually is: a flat
    folder, no real/fake split)."""
    real_dir, fake_dir = root / "real", root / "fake"
    if real_dir.exists() or fake_dir.exists():
        paths, labels = [], []
        for folder, label in ((real_dir, 0), (fake_dir, 1)):
            if not folder.exists():
                continue
            for p in sorted(folder.rglob("*")):
                if p.suffix.lower() in IMG_EXTS:
                    paths.append(p)
                    labels.append(label)
        return paths, np.array(labels)

    paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS)
    return paths, None


def extract_folder(paths: list[Path], model, processor, device, batch_size: int) -> dict:
    all_embeds, filenames, failed = [], [], []

    for i in tqdm(range(0, len(paths), batch_size), desc="extracting"):
        batch_paths = paths[i : i + batch_size]
        batch_imgs, batch_ok_paths = [], []
        for p in batch_paths:
            try:
                batch_imgs.append(Image.open(p).convert("RGB"))
                batch_ok_paths.append(p)
            except Exception as e:
                failed.append((str(p), str(e)))

        if not batch_imgs:
            continue

        inputs = processor(images=batch_imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            batch_embeds = model.get_image_features(**inputs).cpu().numpy()

        for p, embed in zip(batch_ok_paths, batch_embeds):
            all_embeds.append(embed)
            filenames.append(p.name)

    if failed:
        print(f"\n{len(failed)} image(s) failed to open, skipped:")
        for path, err in failed[:10]:
            print(f"  {path}: {err}")

    return {
        "embeddings": np.stack(all_embeds),
        "filename": np.array(filenames),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image-dir", default=r"C:\Users\hunte\Downloads\FakeXplain\FakeXplain\images\batch_b")
    ap.add_argument("--out", default=r"D:\Coding\playground\TechJam-2026-Team-STH\data\cache\embeddings\fakexplain_test.npz")
    ap.add_argument("--backbone", choices=["b32", "l14"], default="l14")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        raise SystemExit(f"--image-dir not found: {image_dir}")

    checkpoint = BACKBONES[args.backbone]
    print(f"Backbone selected: {args.backbone.upper()} ({checkpoint})")

    paths, labels = scan_folder(image_dir)
    if not paths:
        raise SystemExit(f"No images found under {image_dir}")
    print(f"Found {len(paths)} images"
          f"{' (labeled real/fake subfolders detected)' if labels is not None else ' (flat folder, unlabeled)'}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    model = CLIPModel.from_pretrained(checkpoint).to(device).eval()
    processor = CLIPProcessor.from_pretrained(checkpoint)

    result = extract_folder(paths, model, processor, device, args.batch_size)
    if labels is not None:
        result["labels"] = labels

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **result)
    print(f"\nWrote {out_path}: {result['embeddings'].shape[0]} embeddings, "
          f"dim {result['embeddings'].shape[1]}")


if __name__ == "__main__":
    main()