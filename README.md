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

We evaluate FusionCLIP on clean images and **15 held-out transformation types** representing common image degradations, including blur, JPEG compression, noise, resizing, color shifts, and sharpening.

### FusionCLIP vs. DCPT baseline

The baseline uses the same frozen **CLIP ViT-L/14 + DCPT** training approach but without the four additional engineered signals.

| Metric                      | CLIP + DCPT Baseline | FusionCLIP + DCPT |           Change |
| --------------------------- | -------------------: | ----------------: | ---------------: |
| Clean Accuracy              |                93.2% |             92.9% |         -0.3 pts |
| Clean AUC                   |               0.9699 |            0.9699 |              ≈ 0 |
| Mean Transformed Accuracy   |               91.83% |            91.97% |        +0.14 pts |
| Mean Transformed AUC        |               0.9646 |            0.9651 |          +0.0005 |
| Clean → Transformed AUC Gap |               0.0053 |        **0.0048** | **0.0005 lower** |
| Clean AUC Retention         |               99.45% |        **99.50%** |        +0.05 pts |

FusionCLIP retains approximately 99.5% of its clean AUC after transformation, with mean AUC decreasing from 0.9699 to 0.9651 across the 15 held-out transformation types.

The key result is that FusionCLIP retains the robustness already established by DCPT while providing a **modest additional improvement** from the explicit degradation-aware signals. The clean-to-transformed AUC gap is reduced by approximately **9.4% relative to the DCPT baseline**, from 0.0053 to 0.0048, while clean AUC remains unchanged.

Across the 15 held-out transformation types, FusionCLIP achieves approximately **92% accuracy** and maintains a high AUC, showing that its ability to distinguish real from AI-generated images remains stable under common image degradations rather than relying solely on clean-image performance.

### Comparison with prior work

Robustness to post-processing is a known challenge in AIGC detection — prior work such as Raising the Bar of AI-generated Image Detection with CLIP (Cozzolino et al., CVPRW 2024) has documented substantial degradation in earlier, non-CLIP-based detectors under common post-processing operations. FusionCLIP achieves 99.5% clean AUC retention (0.9699 → 0.9651) across our 15 held-out transformation types, demonstrating that the combination of DCPT training and engineered signals is effective at preserving performance under degradation.

However, datasets and evaluation protocols differ, so this should be interpreted as a general comparison rather than a direct benchmark.

### Interpretation

These results suggest that **DCPT is the primary contributor to degradation robustness**, while the engineered signals provide an additional, smaller improvement by exposing low-level degradation information to the classifier.

The improvement is not uniform across every transformation type, so we do not claim that the engineered signals universally improve every degradation. Instead, our results support the more conservative conclusion that **explicit degradation-aware signals can complement DCPT and slightly reduce the overall robustness gap without sacrificing clean-image AUC**.

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

Additionally: The additional robustness gain from FusionCLIP is modest. Our DCPT baseline is already highly robust, so the engineered signals reduce the clean-to-transformed AUC gap only from **0.0053 to 0.0048**. We therefore do not claim that the engineered signals are solely responsible for the overall robustness of the system.

Beyond that: the main trade-off this time around is the lack of prevention against semantic
bias, given our inherent focus on robustness to image transformations. As we
did not have enough time to properly implement the strategy into our system effectively. In
the future, on top of heavier emphasis on the dataset, we will be exploring new signals to
quantify a wider range of transformations, as well as fine-tuning the vector concatenation.

To add on: Performance varies across transformation types. FusionCLIP improves performance on some degradations more than others, and the engineered signals do not consistently outperform the baseline on every individual transformation.

Last but not least: Our 15 held-out transformations represent common real-world changes such as JPEG compression, blur, noise, resizing, color shifts, and sharpening, but real-world images may undergo combinations of transformations that are not fully represented by our evaluation.

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
- **`validation_result/`** -- From Judge's Validation set a  real COCO photos vs. independently-sourced DALLE-3  generations, entirely outside
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


## Video Link
VideoLink:https://youtu.be/aOOY6EsNVPI
