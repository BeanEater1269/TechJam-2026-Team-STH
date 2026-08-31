# Error Analysis

Two separate evaluation sets, kept explicitly distinct below:

- **Our own held-out test set** (`final_model_fn_fp/`) -- the 3,000-image test split from
  `data/manifest.csv`, drawn from our four training sources (SID-Set, CIFAKE, WildFake,
  StyleGan_XL,IMAGEN3), never trained on.
- **Judge's validation set** (`validation_fn_fp/`) -- our supplementary out-of-distribution
  stress test (COCO real photos + independently-sourced DALLE-3 generations), built to check
  generalization beyond anything in the training manifest. 
  and [`validation_result/coco_dalle3_auc_comparison.json`](../validation_result/coco_dalle3_auc_comparison.json).

## Our own held-out test set

### False positive

![false positive - child portrait](../final_model_fn_fp/false_positive_photo_fe0bec3214706d60.jpg)

A heavily blurred, desaturated, colorized portrait of a young child in a lace bonnet and
glasses, seated against a dark, vignetted background -- visually reads like a still from old
archival film footage or a hand-colorized antique photograph. The model flagged this real
image as AI-generated.

**Hypothesis:** this one is a much closer confound than a generic soft-focus photo. The image
combines heavy blur, a muted/desaturated color grade, and a dark vignette -- a specific
aesthetic that diffusion models frequently produce when generating "vintage portrait" or
"old photograph" style images, since that look is heavily represented in their training data.
Our Laplacian and DCT high-band signals measure *how sharp/detailed* the image is, not *why*
it's soft -- so a real photo that is soft, low-detail, and stylistically "vintage" for
genuine archival reasons lands in the same signal range as a diffusion model deliberately
generating that exact aesthetic. This is arguably our most informative false positive: it
isn't random noise, it's a specific, describable style collision between "genuinely old/blurry
real photo" and "AI-generated in an old-photo style."

### False negative

![false negative - racing photo](../final_model_fn_fp/false_negative_photo_fe0bec3214706d60.jpg)

A motorsport photo (marshal in a hi-vis vest, yellow car exiting a corner) carrying a visible
photographer credit watermark: "SLEEPYCAT, Photos by Glenda Clarke." The model predicted this
image as real; our manifest's ground-truth label says fake.

**This is not a model error -- it's a label error we introduced.** A third-party
photo-agency watermark crediting a named photographer is strong evidence this is authentic
photography, not a generated image. This is a concrete instance of the labeling mistake
described in [Limitations](../README.md#limitations): during manual sorting, some genuinely
real images were filed into the fake bucket. The model's "real" prediction here is arguably
*correct*; our scoring counted it as a miss because the label was wrong, not because the
model was fooled.

## Judge's validation set (COCO real vs. DALLE-3 fake, out-of-distribution)

### False positives

| | | |
|---|---|---|
| ![fp1](../validation_fn_fp/false_positive_validation_photo_img158957.jpg) | ![fp2](../validation_fn_fp/false_positive_validation_photo_img158959.jpg) | ![fp3](../validation_fn_fp/false_positive__validation_photo_img158960.jpg) |

Three real COCO photos flagged as fake: a yellow-walled living room, a bedroom with a window
and bookshelf, and an upside-down street STOP sign shot from below. Deliberately shown
together because there is **no shared visual property** across them -- different subjects,
different lighting, different composition, indoor and outdoor, no common color palette or
texture.

**Hypothesis:** the absence of a pattern is itself the finding. If these false positives shared
a visual trait (all blurry, all low-light, all one color palette), that would point to a real
signal-level blind spot. Instead, the scattershot nature across ~30-40% of this out-of-distribution
real-image set is more consistent with what we already flagged: a mismatch between our narrow
real-image training distribution (dominated by SID-Set's specific look) and general photography,
compounded by the labeling error in how this stress-test set itself was assembled

### False negative

![false negative - world map art](../validation_fn_fp/false_negative_validation_0763e21106d9bb5adb44fafdebba1337.jpg)

An AI-generated stylized world map made of dense, colorful dotted/pointillist texture --
clearly generative art, not a photograph. Shown here in its **clean** form for reference; the
model correctly classified this exact image as fake at clean resolution. The miss happened
specifically on its **blur (sigma=2.0)** transformed variant -- our most severe blur setting
(sigma in {0.5, 1.0, 2.0}, see `src/transforms.py`), which the clean version above does not
visually represent.

**Hypothesis:** this image's strongest tell is its fine, repetitive dot texture -- exactly the
kind of high-frequency structure our DCT high-band, Laplacian, and noise-variance signals are
built to catch, and exactly what heavy Gaussian blur erases first. At sigma=2.0, the dot
pattern is smoothed away almost entirely, so the signals that correctly caught this image at
clean resolution have essentially nothing left to measure -- while CLIP's semantic read
("stylized map artwork") isn't inherently indicative of real vs. fake on its own. This is a
case where robustness training can soften the drop but can't manufacture discriminative signal
that the transform has physically removed from the image -- a genuine ceiling, not a training
bug.

## Trade-offs

- **Robustness vs. peak clean accuracy.** Our fusion ablation showed FiLM conditioning scores
  higher on clean images (93.3% vs. 92.8% accuracy) but has more than double concatenation's
  robustness gap under transforms (1.71pt vs. 0.98pt accuracy gap). We chose concatenation,
  trading ~0.5pt of clean accuracy for meaningfully better degradation robustness.
- **Signal complexity vs. marginal value.** Adding a LIQE no-reference quality signal made
  results slightly worse (92.03% -> 91.87% overall accuracy), likely redundant with our
  existing noise-variance and Laplacian-variance signals -- a reminder that more engineered
  features isn't free; each one needs to earn its place empirically.
- **Discrimination vs. calibration.** On our OOD real/fake stress test, AUC stayed strong
  (0.95-0.97) even as accuracy at a fixed 0.5 threshold looked weak -- meaning the model still
  *ranks* real vs. fake well, but its decision threshold, tuned on our training distribution,
  doesn't transfer cleanly to unseen real-photo sources. This is a calibration problem, not a
  total discrimination failure, and is in principle fixable without retraining (threshold
  recalibration on a target distribution).
- **Signal strength vs. transform severity, concretely.** The world-map false negative above
  shows the limit case directly: at our most severe blur setting, the frequency/texture-based
  signals that catch this specific fake lose their signal almost entirely. Robustness training
  narrows this gap but can't eliminate it when the image's only strong tell is fine texture.
- **Information loss vs. shortcut prevention.** Center-cropping to square throws away image
  content, but was necessary to stop aspect ratio from becoming a free shortcut against one
  training source's consistently non-square real images.
- **Downsampling vs. upsampling.** We prioritized downsampling to 512px over upsampling smaller
  images, since upsampling introduces a detectable softness that could otherwise become a
  shortcut correlated with the real/fake label rather than actual content.
