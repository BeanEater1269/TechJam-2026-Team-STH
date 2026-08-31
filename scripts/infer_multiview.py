"""
Multi-view variant of scripts/infer.py -- the pipeline's attempt at cropping tolerance.
See README.md's Limitations section: "nothing in this pipeline can *detect* cropping
(only tolerate it via multi-view averaging)." The model can't tell a cropped image
apart from an uncropped one; this doesn't fix that -- it just makes the FINAL score
less sensitive to whichever single crop happens to land in front of it, by scoring two
views and averaging.

The two views, per image:
  1. the whole standardized image (exactly what infer.py scores)
  2. its center-crop-80% view -- t_center_crop(image, 0.80), imported directly from
     src/transforms.py, NOT reimplemented -- this is the exact "crop80" robustness
     variant the model was already trained on (see transforms.py's build_all_variants),
     so view 2 isn't a novel transform, it's one the model has literally seen before.

Crop80 must be applied to the ALREADY-STANDARDIZED (square, WORKING_RES) image, same
order training used -- cropping the raw upload first and standardizing the crop
separately would NOT reproduce the same pixels crop80 produced at training time (a
crop of an arbitrary-aspect-ratio original resizes differently than a crop of an
already-square 512x512 image). Both views end up already exactly WORKING_RES x
WORKING_RES, so passing them straight into Predictor.predict() is safe -- its internal
standardize() call becomes a no-op pass-through on already-standardized input (crop_to_
square and resize are both skipped when the size already matches).

The two views' raw_prob get averaged into the final `pred`; the individual per-view
scores are also kept in the output so you can see how much the crop changed the answer,
not just the combined number.

Usage:
    python scripts/infer_multiview.py path/to/image_dir
    python scripts/infer_multiview.py path/to/image_dir --out my_predictions.json
    python scripts/infer_multiview.py path/to/single_image.jpg
"""
import argparse
import json
import sys
from pathlib import Path

from PIL import Image, UnidentifiedImageError

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "webdemo"))
sys.path.insert(0, str(REPO_ROOT / "src"))
from inference import get_predictor, standardize, THRESHOLD  # noqa: E402
from transforms import t_center_crop  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
BATCH_SIZE = 32  # outer chunk size (in IMAGES, not views) -- each image contributes 2
                 # views, so this caps actual CLIP-batch input at 2x this per chunk


def find_images(root: Path) -> list:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS)


def load_image(path: Path) -> Image.Image | None:
    try:
        return Image.open(path)
    except (UnidentifiedImageError, OSError) as e:
        print(f"  WARNING: skipping {path} -- not a readable image ({e})")
        return None


def combine(whole_result: dict, crop_result: dict) -> dict:
    """Averages the two views' raw_prob into a single combined verdict, re-deriving
    label/confidence from that combined number (not by averaging the two labels/
    confidences separately, which wouldn't be a well-defined operation)."""
    combined_pred = (whole_result["raw_prob"] + crop_result["raw_prob"]) / 2
    label = "FAKE" if combined_pred >= THRESHOLD else "REAL"
    confidence = combined_pred if label == "FAKE" else 1.0 - combined_pred
    return {
        "pred": combined_pred,
        "prediction": label,
        "confidence": confidence,
        "whole_image_pred": whole_result["raw_prob"],
        "crop80_pred": crop_result["raw_prob"],
    }


def predict_multiview(predictor, images: list) -> list:
    """images: list of raw PIL.Image. Returns one combined dict per image (see
    combine()). Standardizes each image once, builds its crop80 view, then scores
    ALL views (2 per image) in a single batched Predictor.predict() call -- same
    total CLIP work as calling predict() twice, fewer round trips."""
    std_images = [standardize(im) for im in images]
    crop_views = [t_center_crop(im, 0.80) for im in std_images]

    n = len(images)
    all_results = predictor.predict(std_images + crop_views)
    whole_results, crop_results = all_results[:n], all_results[n:]

    return [combine(w, c) for w, c in zip(whole_results, crop_results)]


def run_batch(image_dir: Path, out_path: Path) -> None:
    paths = find_images(image_dir)
    if not paths:
        raise SystemExit(f"no images found under {image_dir} (looked for {sorted(IMG_EXTS)})")
    print(f"found {len(paths)} image(s) under {image_dir}")

    predictor = get_predictor()
    predictions = []
    for i in range(0, len(paths), BATCH_SIZE):
        chunk_paths = paths[i : i + BATCH_SIZE]
        loaded = [(p, load_image(p)) for p in chunk_paths]
        ok_paths = [p for p, img in loaded if img is not None]
        ok_images = [img for _, img in loaded if img is not None]
        if not ok_images:
            continue

        results = predict_multiview(predictor, ok_images)
        for p, r in zip(ok_paths, results):
            predictions.append({"image_path": str(p), **r})
        print(f"  scored {min(i + BATCH_SIZE, len(paths))}/{len(paths)}")

    n = len(predictions)
    n_fake = sum(1 for p in predictions if p["prediction"] == "FAKE")
    summary = {
        "n": n,
        "n_fake": n_fake,
        "n_real": n - n_fake,
        "fake_rate": (n_fake / n) if n else 0.0,
        "mean_pred": (sum(p["pred"] for p in predictions) / n) if n else 0.0,
        "mean_confidence": (sum(p["confidence"] for p in predictions) / n) if n else 0.0,
    }

    out_path.write_text(json.dumps({"summary": summary, "predictions": predictions}, indent=2))
    print(f"\nWrote {n} prediction(s) to {out_path}")
    print(f"  {n_fake} flagged FAKE, {n - n_fake} flagged REAL "
          f"(fake rate {summary['fake_rate']:.1%}, mean pred {summary['mean_pred']:.4f})")


def run_single(image_path: Path) -> None:
    image = load_image(image_path)
    if image is None:
        raise SystemExit(1)

    predictor = get_predictor()
    result = predict_multiview(predictor, [image])[0]

    print(f"\nimage:              {image_path}")
    print(f"whole-image pred:   {result['whole_image_pred']:.4f}")
    print(f"crop80 pred:        {result['crop80_pred']:.4f}")
    print(f"inference result:   {result['pred']:.4f}  (P(AI-generated), averaged over 2 views)")
    print(f"model prediction:   {result['prediction']}")
    print(f"confidence score:   {result['confidence']:.4f}  ({result['confidence'] * 100:.1f}%)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="Path to an image directory (batch mode) or a single image file.")
    ap.add_argument("--out", default="predictions.json", help="Batch mode only: output JSON path.")
    args = ap.parse_args()

    path = Path(args.path)
    if not path.exists():
        raise SystemExit(f"path not found: {path}")

    if path.is_dir():
        run_batch(path, Path(args.out))
    else:
        run_single(path)


if __name__ == "__main__":
    main()
