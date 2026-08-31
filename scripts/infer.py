"""
CLI inference entrypoint -- the scripts/infer.py referenced in README.md's Reproduction
and Demo sections. Wraps webdemo/inference.py's Predictor (the exact same model/
preprocessing the web app serves: ConcatClassifier + CLIP ViT-L/14 + 5 signals, aka
"concat_drift", the model the team settled on) so a judge running this script and a
judge clicking through the web demo can never get a different answer for the same image.

Two modes, chosen by what <path> points at:

  Directory -> batch mode. Every image found under <path> (recursively) gets scored,
  written to --out as {"summary": {...}, "predictions": [...]}. Each entry in
  `predictions` is {"image_path", "pred", "prediction", "confidence"} -- `pred` alone
  (a bare float) doesn't tell a reader what the model actually concluded, so every
  entry also carries the same `prediction` (REAL/FAKE label) and `confidence` (how sure
  the model is of THAT label, always >= 0.5) the web app and single-image mode show.
  `pred` is the raw sigmoid output, P(image is AI-generated) -- distinct from
  `confidence`, which is always >= 0.5 regardless of which way it leans. `summary`
  gives the folder-level rollup: counts, fake rate, and the average pred/confidence
  across the whole batch.

  Single file -> prints the 3 things to stdout: the raw inference result (P(fake)), the
  model's prediction (REAL/FAKE), and the confidence score (confidence in that specific
  prediction). No JSON written for this mode -- read it off stdout.

Images that fail to open are skipped with a warning, not fatal -- one corrupt file in a
judge's test folder shouldn't kill the whole batch run.

Usage:
    python scripts/infer.py path/to/image_dir
    python scripts/infer.py path/to/image_dir --out my_predictions.json
    python scripts/infer.py path/to/single_image.jpg
"""
import argparse
import json
import sys
from pathlib import Path

from PIL import Image, UnidentifiedImageError

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "webdemo"))
from inference import get_predictor, to_record  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
BATCH_SIZE = 32  # outer chunk size for directory mode -- keeps memory bounded on large
                 # folders; Predictor.predict() further sub-batches CLIP calls internally


def find_images(root: Path) -> list:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS)


def load_image(path: Path) -> Image.Image | None:
    try:
        return Image.open(path)
    except (UnidentifiedImageError, OSError) as e:
        print(f"  WARNING: skipping {path} -- not a readable image ({e})")
        return None


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

        results = predictor.predict(ok_images)
        for p, r in zip(ok_paths, results):
            predictions.append({
                "image_path": str(p),
                "pred": r["raw_prob"],
                "prediction": r["label"],
                "confidence": r["confidence"],
            })
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
    result = to_record(predictor.predict([image])[0])

    print(f"\nimage:              {image_path}")
    print(f"inference result:   {result['raw_prob']:.4f}  (P(AI-generated))")
    print(f"model prediction:   {result['label']}")
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
