# Track 5 — Robust AIGC Image Detection

Our solution for Track 5 — Robust AIGC Image Detection is a highly resilient AI image classifier designed to survive real-world platform transformations like JPEG compression, blurring, and color jittering. Instead of relying solely on a base vision model, we augment a frozen CLIP backbone with purpose-built "robustness signals" (including Feature Drift Magnitude, frequency-domain artifacts, and perceptual quality scores). By fusing these signals and applying Degradation-Consistent Paired Training (DCPT)—a consistency regularization technique that forces the model to yield identical predictions for clean and degraded views of the same image—our lightweight MLP classifier maintains high real-vs-fake separability across diverse generator families without sacrificing accuracy on degraded media.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```


`checkpoints/final_model.pt` and everything else `scripts/infer.py` needs
(`data/cache/stats/*.npz`) are committed in this repository. 
To obtain the final model file, please use `git clone`

Download below link for embedded vectors files

Train: https://drive.google.com/file/d/1_7v-TGS5Wg7QK0mHzBx3ZIrnYpQ2G7kA/view?usp=drive_link
Validation: https://drive.google.com/file/d/1u4EJ5yPf6bk7t60aJxFKHpt2u1u7yS2W/view?usp=drive_link
Test: https://drive.google.com/file/d/1hvUixsPcWBm8feye4vG3_FuOE0-xNRKp/view?usp=drive_link

## Data

See `docs/dataset-plan.md` for the full source breakdown (WildFake, SID-Set, CIFAKE,
AIGIBench — 30,000 images, 15,000 real / 15,000 fake) and `docs/pipeline-decisions.md` for
the resolution (512×512) and shape-standardization (crop-to-square) reasoning, including
what was considered and rejected.

## Reproduction

1. Download data, build the manifest + 80/10/10 split, cache clean + 15 variants, and
   extract embeddings/signals. Run the script in the following sequence
   - `scripts/build_manifest.py`
   - `scripts/data_prep/ build_cache.py`
   - `scripts/features/extract_embeddings.py`
   - `scripts/features/ signals.py`
   - `scripts/features/clip_drift.py`
2. Train: run the code `python scripts/training/concat_drift_epoch_optimization.py --epochs <N>
   --backbone-dim 768` to produce `checkpoints/final_model.pt` for classifier model. 
3. Run inference:

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

AUC stays within ~2 points of clean (0.957–0.977) across every condition — degradation
under transformation shows up mainly as a threshold/accuracy effect, not a loss of
ranking ability. Heavy noise (σ0.1) is the worst case, at -3.4 pts accuracy vs. clean.

## Limitations

Our main limitation is a labeling error in our own data, not a modeling one. During manual
sorting of downloaded validation images, a batch of genuinely real photos got misfiled into
the fake bucket -- so some of what we scored our model against as "fake" was actually real,
and some of the "false positives" we measured are the model being right against a wrong label,
not the model being fooled. We caught this too late in the hackathon to re-sort the data and
retrain -- if we had more time, that's the first thing we'd fix, followed by re-running our
evaluation numbers with the corrected labels.

See `docs/error_analysis.md` for the concrete examples (including a labeled-fake image that is
visibly a real photo with a photographer's watermark on it) and `validation_result/` for the
underlying prediction data and the before/after numbers this produced.

Beyond that: the main trade-off this time around is the lack of prevention against semantic
bias, given our inherent focus on robustness to image transformations. Real images from one
dataset (SID-Set) result in larger false positives for scenery images, especially mountain
views. Another limitation in our architecture is the lack of direct cropping detection, as we
did not have enough time to properly implement the strategy into our system effectively. In
the future, on top of heavier emphasis on the dataset, we will be exploring new signals to
quantify a wider range of transformations, as well as fine-tuning the vector concatenation.

## Team contributions

**Kaiqiang** (team lead) — came up with the core architecture and big idea, implemented the
base CLIP model and Degradation-Consistent Paired Training (DCPT), led the team.

**Jiatai** — cleaned the data and set up the pipeline, contributed to the architecture,
troubleshot the signals, ran the pipeline end-to-end, and fine-tuned the signals and their
hypotheses.

Lionel and Terence worked on the FiLM-vs-concatenation fusion comparison and ablation testing.

Jaden worked on the web demo and helped set up the base CLIP model, testing, and ablations.

## Repository guide -- what to look at

A quick map of what's in this repo and why it's here, for judges working through the
deliverables:

- **`scripts/infer.py`** -- the required inference script. Takes an image directory (or a
  single image) and outputs a JSON file with `image_path` and `pred` (P(AI-generated)) per
  image, plus a `prediction`/`confidence` label. See [Reproduction](#reproduction) above.
- **`results_l14/`** -- our own held-out evaluation of the final model (`final_model_eval.json`
  is the main robustness table: clean vs. 15 transformed variants, by-source and
  by-generator-family breakdowns).
- **`final_model_fn_fp/`** -- representative false positive / false negative images from our
  own held-out test set, referenced in `docs/error_analysis.md`.
- **`validation_result/`** -- a supplementary out-of-distribution stress test we built
  ourselves: real COCO photos vs. independently-sourced DALLE-3 generations, entirely outside
  our training data. `coco_dalle3_auc_comparison.json` has the clean-vs-transformed AUC/accuracy
  comparison (both a balanced and an unbalanced version, clearly labeled).
- **`validation_fn_fp/`** -- representative false positive / false negative images from that
  out-of-distribution stress test, also referenced in `docs/error_analysis.md`.
- **`docs/error_analysis.md`** -- the full error analysis writeup: representative false
  positives/negatives from both evaluation sets above, with images embedded, plus the
  trade-offs made in our approach.
- **`DriftClip.pdf`** -- project poster / one-pager.
- **Robustness summary table (PDF)** -- compact clean-vs-transformed visual summary, to be
  added.
