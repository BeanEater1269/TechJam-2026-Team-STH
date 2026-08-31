# Track 5 — Robust AIGC Image Detection

TODO: one-paragraph project overview once the pitch is finalized.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Before pulling any data: two checks

Both are cheap and gate real decisions below — run them first.

```bash
# 1. Backbone timing -- on YOUR actual GPU, using any ~100-500 photos already on your machine
python scripts/check_clip_backbone_timing.py --images <folder of any local jpg/png files> --n 100

# 2. Native resolution -- once you have a small sample (a few hundred images) from each source
python scripts/check_native_resolution.py --root <sample folder 1> --root <sample folder 2>
```

## Data

See `docs/dataset-plan.md` for the full source breakdown (WildFake, SID-Set, CIFAKE,
AIGIBench — 30,000 images, 15,000 real / 15,000 fake) and `docs/pipeline-decisions.md` for
the resolution (512×512) and shape-standardization (crop-to-square) reasoning, including
what was considered and rejected.

## Reproduction

1. Download data, build the manifest + 80/10/10 split, cache clean + 15 variants, and
   extract embeddings/signals — see `scripts/build_manifest.py`, `scripts/data_prep/
   build_cache.py`, `scripts/features/extract_embeddings.py`, `scripts/features/
   signals.py`, `scripts/features/clip_drift.py`, `scripts/features/liqe.py`.
2. Train: `python scripts/training/concat_drift_epoch_optimization.py --epochs <N>
   --backbone-dim 768` (produces `checkpoints/final_model.pt`). Compare candidates with
   `python scripts/evaluation/collect_results.py --backbone-dim 768` (writes
   `results_l14/`).
3. Run inference (this is the part judges need):

   **Setup** (once):
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
   `checkpoints/final_model.pt` and everything else `scripts/infer.py` needs
   (`data/cache/stats/*.npz`) are committed in this repository — a plain `git clone`
   (or unzip, if handed as an archive) is all that's needed, no separate model-weights
   download and no dataset download required to run inference.

   **Run:**
   ```bash
   # Whole folder -> predictions.json ({"summary": {...}, "predictions": [{"image_path", "pred", "prediction", "confidence"}, ...]})
   python scripts/infer.py path/to/image_dir --out predictions.json

   # Single image -> prints inference result / prediction / confidence to stdout
   python scripts/infer.py path/to/single_image.jpg
   ```
   `pred` is the raw P(AI-generated) probability; `prediction` is the REAL/FAKE label;
   `confidence` is confidence in that specific label (always >= 0.5). First run
   downloads the CLIP ViT-L/14 backbone (~1.7GB) from Hugging Face — needs internet
   access once, cached locally after that.

   For cropping-tolerance (scores the image twice — whole + center-crop-80% view —
   and averages), same interface: `python scripts/infer_multiview.py <path> [--out ...]`.

## Demo

Web app lives in `webdemo/`, built on the same `scripts/infer.py`/`webdemo/inference.py`
model (so the web demo and the CLI script can never disagree on a score). Same setup as
above (venv + `checkpoints/final_model.pt` in place), then:
```bash
.venv/Scripts/python.exe -m uvicorn webdemo.main:app --port 8000   # Windows
.venv/bin/python -m uvicorn webdemo.main:app --port 8000            # macOS/Linux
```
Open `http://localhost:8000` — `/` for the single-image detector, `/live` for
batch/folder analysis with a saved run history, `/results` for the model comparison
dashboard.

## Results

Final model (`checkpoints/final_model.pt`, ViT-L/14 + drift/quality signals) evaluated
on 3,000 held-out images per condition — clean plus 15 transformed variants, 48,000
images total. Full numbers in `results_l14/final_model_eval.json`.

**Clean vs. transformed (overall)**

| Condition | Accuracy | AUC |
|---|---|---|
| Clean | 93.6% | 0.977 |
| Mean of 15 transforms | 92.2% | 0.971 |
| **Gap** | **-1.3 pts** | **-0.006** |

**By transform family**

| Family | Accuracy range | Mean accuracy | Weakest case |
|---|---|---|---|
| Blur (σ0.5–2.0) | 92.0–93.6% | 93.0% | σ2.0 |
| Resize (0.25×–0.5×) | 92.4–93.4% | 92.9% | 0.25× |
| Crop (80% center) | 92.9% | 92.9% | — |
| JPEG (q30–q90) | 91.4–92.9% | 92.2% | q30 |
| Color jitter (±) | 91.2–92.7% | 92.0% | brighten |
| Gaussian noise (σ0.02–0.1) | 90.2–91.8% | 91.1% | σ0.1 |

AUC stays within ~1 point of clean (0.957–0.977) across every condition — degradation
under transformation shows up mainly as a threshold/accuracy effect, not a loss of
ranking ability. Heavy noise (σ0.1) is the worst case, at -3.4 pts accuracy vs. clean.

## Limitations

TODO — fill in honestly once trained: expected known weaknesses include the residual,
tagged upsample risk on a minority of SID-Set real crops, and the standing limitation that
nothing in this pipeline can *detect* cropping (only tolerate it via multi-view averaging).

## Team contributions

TODO.
