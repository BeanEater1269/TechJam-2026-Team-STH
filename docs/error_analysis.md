# Error Analysis

Representative false positives, false negatives, and trade-offs observed in FusionCLIP's
predictions, drawn from our own evaluation and an out-of-distribution stress test (COCO
real photos vs. independently-sourced DALLE-3 generations, outside all four training sources).

## False Positives (real images misclassified as AI-generated)

Our clearest false-positive source is out-of-distribution real photography. On a held-out
set of COCO images, the model misflagged real photos as fake at near-certain confidence:

| Image | Predicted P(fake) | Ground truth |
|---|---|---|
| `img160422.jpg` | 1.0000 | Real (COCO) |
| `img159091.jpg` | 0.9999 | Real (COCO) |
| `img162639.jpg` | 0.9998 | Real (COCO) |

<!-- TODO: embed thumbnails here, e.g.
![false positive 1](assets/fp_img160422.jpg)
![false positive 2](assets/fp_img159091.jpg)
![false positive 3](assets/fp_img162639.jpg)
-->

**Hypothesis:** our real-image training data leans heavily on one source (SID-Set), which
has a specific compression and crop signature. The model may be keying on "looks like
SID-Set" rather than "looks real" in general, so genuine photos from a different pipeline
(different compression history, different crop conventions) don't register as real. See
[Limitations](../README.md#limitations) for the fuller discussion, including the ~30-40%
false-positive rate we measured on this out-of-distribution set and the caveat around a
labeling error discovered in how that stress-test set was assembled.

## False Negatives (AI-generated images misclassified as real)

Our most confident false negatives are DALLE-3 images the model scored as low as 0.01-0.05
P(fake) -- i.e. it was highly confident they were real:

| Image | Predicted P(fake) | Ground truth |
|---|---|---|
| `830e6b1cfa79...863.jpg` | 0.0125 | Fake (DALLE-3) |
| `d847bbb634ba...5ac.jpg` | 0.0284 | Fake (DALLE-3) |
| `c6d4b444184a...ffe.jpg` | 0.0293 | Fake (DALLE-3) |

<!-- TODO: embed thumbnails here, e.g.
![false negative 1](assets/fn_830e6b1c.jpg)
![false negative 2](assets/fn_d847bbb6.jpg)
![false negative 3](assets/fn_c6d4b444.jpg)
-->

**Hypothesis:** *(fill in once thumbnails are reviewed)* -- typically this is where a
diffusion model produces unusually clean, low-noise, photographically-lit output with
none of the high-frequency artifacts our DCT/noise/Laplacian signals are built to catch.
Worth confirming against what these specific images actually look like (busy natural
texture that masks artifacts? soft/diffuse lighting? simple composition?) before finalizing
this line.

## Trade-offs

- **Robustness vs. peak clean accuracy.** Our fusion ablation showed FiLM conditioning
  scores higher on clean images (93.3% vs. 92.8% accuracy) but has more than double
  concatenation's robustness gap under transforms (1.72pt vs. 0.98pt accuracy gap). We
  chose concatenation, trading ~0.5pt of clean accuracy for meaningfully better
  degradation robustness.
- **Signal complexity vs. marginal value.** Adding a LIQE no-reference quality signal made
  results slightly worse (92.03% -> 91.87% overall accuracy), likely redundant with our
  existing noise-variance and Laplacian-variance signals -- a reminder that more engineered
  features isn't free; each one needs to earn its place empirically.
- **Discrimination vs. calibration.** On our OOD real/fake stress test, AUC stayed strong
  (0.95-0.97) even as accuracy at a fixed 0.5 threshold looked weak -- meaning the model
  still *ranks* real vs. fake well, but its decision threshold, tuned on our training
  distribution, doesn't transfer cleanly to unseen real-photo sources. This is a
  calibration problem, not a total discrimination failure, and is in principle fixable
  without retraining (threshold recalibration on a target distribution).
- **Information loss vs. shortcut prevention.** Center-cropping to square throws away
  image content, but was necessary to stop aspect ratio from becoming a free shortcut
  against one training source's consistently non-square real images.
- **Downsampling vs. upsampling.** We prioritized downsampling to 512px over upsampling
  smaller images, since upsampling introduces a detectable softness that could otherwise
  become a shortcut correlated with the real/fake label rather than actual content.
