"""
Stage 0, check #2: time a frozen-CLIP embedding pass with both backbones on
YOUR actual hardware, so ViT-B/32 vs ViT-L/14 gets picked by a number
instead of a guess.

Usage:
    python check_clip_backbone_timing.py --images path/to/some/jpgs --n 500
"""
import argparse
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

BACKBONES = {
    "ViT-B/32": "openai/clip-vit-base-patch32",
    "ViT-L/14": "openai/clip-vit-large-patch14",
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def load_images(root: Path, n: int) -> list[Image.Image]:
    paths = [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS][:n]
    imgs = []
    for p in paths:
        try:
            imgs.append(Image.open(p).convert("RGB"))
        except Exception:
            pass
    return imgs


def time_backbone(name: str, checkpoint: str, images, batch_size: int, device: str) -> float | None:
    print(f"\n{name}  ({checkpoint})")
    model = CLIPModel.from_pretrained(checkpoint).to(device).eval()
    processor = CLIPProcessor.from_pretrained(checkpoint)

    n_params = sum(p.numel() for p in model.parameters())
    n_vision_params = sum(p.numel() for p in model.vision_model.parameters())
    print(f"  params: {n_params / 1e6:.1f}M total ({n_vision_params / 1e6:.1f}M vision tower, "
          f"the part we actually run)")
    print(f"  processor's expected input size: {processor.image_processor.crop_size}")

    # warm-up: first CUDA call pays a fixed setup cost, don't let it skew the timing
    with torch.no_grad():
        warm = processor(images=images[: min(4, len(images))], return_tensors="pt").to(device)
        model.get_image_features(**warm)
        if device == "cuda":
            torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            inputs = processor(images=batch, return_tensors="pt").to(device)
            model.get_image_features(**inputs)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    ips = len(images) / elapsed
    print(f"  {len(images)} images in {elapsed:.1f}s -> {ips:.1f} images/sec")

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return ips


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", required=True, help="folder of sample images (ideally already at your working resolution)")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--dataset-size", type=int, default=50_000, help="rough total working-set size, to project full-caching time")
    ap.add_argument("--variants-per-image", type=int, default=16, help="clean + 15 variants")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    images = load_images(Path(args.images), args.n)
    print(f"loaded {len(images)} sample images")
    if not images:
        print("no images found -- point --images at a folder of jpg/png files")
        return

    results: dict[str, float] = {}
    for name, ckpt in BACKBONES.items():
        try:
            ips = time_backbone(name, ckpt, images, args.batch_size, device)
            if ips:
                results[name] = ips
        except Exception as e:
            print(f"  failed: {e}")

    if len(results) == 2:
        total_embeds = args.dataset_size * args.variants_per_image
        print(
            f"\nProjected full caching pass "
            f"({args.dataset_size:,} images x {args.variants_per_image} variants "
            f"= {total_embeds:,} embeddings):"
        )
        for name, ips in results.items():
            hours = total_embeds / ips / 3600
            print(f"  {name}: ~{hours:.1f} hours")


if __name__ == "__main__":
    main()
