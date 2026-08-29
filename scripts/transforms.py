"""
Shape/resolution standardization + the 6 robustness transforms.

Everything lands at WORKING_RES x WORKING_RES. Locked parameters, see
../docs/pipeline-decisions.md for the full reasoning:

- Real images: crop_to_square() BEFORE resizing to WORKING_RES
- Fake images, 1024-native: jitter_crop_square() BEFORE resizing to WORKING_RES
- Fake images, already at WORKING_RES natively: skip straight to caching, nothing to do
- All 15 variants below are generated FROM the clean, already-square, already-512 image --
  never from the original raw file directly.
- Save format (actually happens in build_cache.py's save_variant(), not here -- this file
  never writes to disk): JPEG-100 for everything, EXCEPT the 4 JPEG-compression variants,
  which save at their own target quality directly (90/70/50/30) -- that quality drop IS
  the transform.
"""
import io
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

WORKING_RES = 512


# ---------------------------------------------------------------------------
# Shape / resolution standardization -- runs ONCE per image, before any of the
# 15 variants exist. Builds the "clean" cached baseline.
# ---------------------------------------------------------------------------

def crop_to_square(image: Image.Image) -> Image.Image:
    """Real images only. Center-crop to the shorter side -- fixes rectangle-vs-square,
    never squash, never pad (see pipeline-decisions.md for why)."""
    w, h = image.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return image.crop((left, top, left + side, top + side))


def jitter_crop_square(image: Image.Image, min_keep: float = 0.90) -> Image.Image:
    """1024-native fake images only. Already square -- this isn't a shape fix, it stops
    every fake from being resized by the exact same fixed ratio (a DCT-visible fingerprint).
    Keeps a random keep_frac in [min_keep, 1.0] of the frame, centered."""
    w, h = image.size  # w == h going in
    keep_frac = random.uniform(min_keep, 1.0)
    side = int(w * keep_frac)
    left = (w - side) // 2
    top = (h - side) // 2
    return image.crop((left, top, left + side, top + side))


def resize_to_working(image: Image.Image, size: int = WORKING_RES) -> Image.Image:
    return image.resize((size, size), Image.BICUBIC)


# ---------------------------------------------------------------------------
# The 6 robustness transforms. Each takes the CLEAN, already-square,
# already-512 image and returns one variant. Dimensions in/out are noted.
# ---------------------------------------------------------------------------

def t_jpeg_compress(image: Image.Image, quality: int) -> Image.Image:
    """quality in {90, 70, 50, 30}. Size unchanged -- the quality drop IS the transform.
    Returns pixel data only; this function never writes to disk. The caller
    (build_cache.py's save_variant()) is responsible for the actual file write, at
    quality=100, which faithfully preserves whatever compression damage happened here."""
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def t_gaussian_blur(image: Image.Image, sigma: float) -> Image.Image:
    """sigma in {0.5, 1.0, 2.0}, in pixels. Size unchanged."""
    return image.filter(ImageFilter.GaussianBlur(radius=sigma))


def t_resize_down_up(image: Image.Image, scale: float) -> Image.Image:
    """scale in {0.5, 0.25}. Shrinks then stretches back to WORKING_RES so this variant
    matches every other variant's final size."""
    w, h = image.size
    small = image.resize((max(int(w * scale), 1), max(int(h * scale), 1)), Image.BICUBIC)
    return small.resize((w, h), Image.BICUBIC)


def t_gaussian_noise(image: Image.Image, sigma_frac: float) -> Image.Image:
    """sigma_frac in {0.02, 0.05, 0.10}, as a fraction of the 0-255 pixel range.
    Size unchanged."""
    arr = np.array(image.convert("RGB")).astype(np.float32)
    sigma = sigma_frac * 255.0
    arr += np.random.normal(0, sigma, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype("uint8"))


def t_color_jitter(image: Image.Image, pct: float) -> Image.Image:
    """pct in {+0.20, -0.20}. Brightness, contrast, and saturation each shifted
    independently by pct. Size unchanged."""
    out = image
    for Enhancer in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
        out = Enhancer(out).enhance(1.0 + pct)
    return out


def t_center_crop(image: Image.Image, keep: float = 0.80) -> Image.Image:
    """The robustness-test crop (NOT the shape-fix crop -- this runs on the already-square
    clean image, as one of the 15 variants). Crops to `keep` of each side, then stretches
    back to WORKING_RES to match everyone else's final size."""
    w, h = image.size
    cw, ch = int(w * keep), int(h * keep)
    left = (w - cw) // 2
    top = (h - ch) // 2
    cropped = image.crop((left, top, left + cw, top + ch))
    return cropped.resize((w, h), Image.BICUBIC)


# ---------------------------------------------------------------------------
# All 15 variants, one call each. 4 JPEG + 3 blur + 2 resize + 3 noise +
# 2 color + 1 crop = 15.
# ---------------------------------------------------------------------------

def build_all_variants(clean_image: Image.Image) -> dict[str, Image.Image]:
    """clean_image must already be square and at WORKING_RES (the standardized baseline)."""
    variants = {}
    for q in (90, 70, 50, 30):
        variants[f"jpeg_q{q}"] = t_jpeg_compress(clean_image, q)
    for s in (0.5, 1.0, 2.0):
        variants[f"blur_s{s}"] = t_gaussian_blur(clean_image, s)
    for scale in (0.5, 0.25):
        variants[f"resize_{scale}"] = t_resize_down_up(clean_image, scale)
    for s in (0.02, 0.05, 0.10):
        variants[f"noise_s{s}"] = t_gaussian_noise(clean_image, s)
    for pct in (0.20, -0.20):
        sign = "up" if pct > 0 else "down"
        variants[f"color_{sign}"] = t_color_jitter(clean_image, pct)
    variants["crop80"] = t_center_crop(clean_image, 0.80)
    return variants
