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

TODO: fill in as the pipeline gets built —
1. Download data (`scripts/` — pending)
2. Build manifest + 80/10/10 split (`scripts/build_manifest.py` — pending)
3. Cache clean + 15 variants, extract embeddings (`scripts/build_cache.py` — pending)
4. Train (`scripts/train.py` — pending)
5. Run inference: `scripts/infer.py <image_dir> -> predictions.json` (pending)

## Demo

Jaden's building this — lives in `webdemo/`. Once `scripts/infer.py` exists, that's the
function the demo should call into.

## Limitations

TODO — fill in honestly once trained: expected known weaknesses include the residual,
tagged upsample risk on a minority of SID-Set real crops, and the standing limitation that
nothing in this pipeline can *detect* cropping (only tolerate it via multi-view averaging).

## Team contributions

TODO.
