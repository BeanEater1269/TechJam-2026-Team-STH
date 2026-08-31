# Dataset Plan

Base pool: **30,000 images** (before the 15 transform variants), split **15,000 real / 15,000 fake**.
Working resolution: **512x512** (see "Why 512" below -- this replaced the original 680 plan).

## Sources

Four sources, chosen for generator diversity -- native GAN and diffusion families, not just
one generator's fingerprint:

| Source | Real | Fake | Generator family | Role |
|---|---|---|---|---|
| **SID-Set** | 12,000 | 2,000 | `real`, `full_synthetic` | Primary real-image and social-media-realism source |
| **CIFAKE** | 3,000 | 3,000 | `real`, `stable_diffusion` | Native 32x32 images -- upsampling-robustness stress test (both real and fake get the same upsample softness, so it doesn't become a shortcut) |
| **WildFake** | -- | 5,000 | `gigagan` (2,500), `dalle` (2,500) | GAN + diffusion diversity |
| **AIGIBench** | -- | 5,000 | `stylegan_xl` (2,500), `imagen3` (2,500) | Modern-generator diversity (newer GAN + diffusion families) |
| **Total** | **15,000** | **15,000** | | **= 30,000** |

SID-Set's fakes are full-synthetic only (label 1); tampered images (label 2, mostly-real photo
with one edited region) are excluded -- see "Excluded, and why" below.

## Why 512, not 680 or 1024

CLIP itself resizes every input to a fixed size (224x224) regardless -- the working resolution
isn't for CLIP, it's so the robustness transforms (blur, JPEG, crop, etc.) get simulated at
something close to a real photo's actual size, and so the classical signals (Laplacian
variance, DCT split, noise variance) have real detail to measure.

The original plan targeted 680 (close to SID-Set's real-crop median) to minimize upsampling.
In practice we moved to 512 and prioritized **downsampling over upsampling** wherever possible:
upsampling introduces a detectable softness that risks becoming a shortcut correlated with the
real/fake label rather than actual content, so it was worth accepting more downsampling (which
loses detail but not label-correlated artifacts) over less upsampling.

## Excluded, and why

| Excluded | Reason |
|---|---|
| WildFake's real images | Risk of overlapping with the hackathon's own COCO-based demo validation set (leakage) |
| SID-Set's tampered images (label 2) | Mostly-real photo with one small edited region -- a different task than whole-image AIGC detection, and a confusing label if included |

## Known residual risks -- tagged in the manifest, checked after training where noted

| Risk | Manifest tag | Status |
|---|---|---|
| CIFAKE's extreme 32->512 upsample | `source_dataset == cifake` | Checked -- see `by_source_dataset` breakdown in `results_l14/final_model_eval.json` (cifake performs well, no anomalous accuracy) |
| Accuracy gap by generator family (WildFake/AIGIBench mix) | `generator_family` | Checked -- see `by_generator_family` in `results_l14/final_model_eval.json`. Confirmed real gap: strongest on sidset/cifake (95%+), weakest on imagen3 (81.2%) and stylegan_xl (83.6%) -- under-represented generator families, out-of-distribution relative to the rest of the mix |
| SID-Set real crops with native short side well under 512 | `native_short_side` | **Not checked** -- tagged in the manifest but the post-hoc verification script was never built. Open risk: a minority of SID-Set real images needed upsampling before being labeled real, which could bias the classifier toward resolution/blur cues on that subset. First thing to verify with more time. |

## Split

80/10/10, stratified by source + generator_family + label, done at the image level **before**
any of the 15 variants exist (splitting after would leak the same photo's content across
train/test in different disguises):

- Train: 24,000
- Val: 3,000 -- iterate against this for every ablation decision
- Test: 3,000 -- touched once, at the very end, for the numbers that go in the robustness table

## Storage

30,000 x 16 (clean + 15 variants) = 480,000 cached items. What actually needs keeping/sharing
is the CLIP embeddings + signal values per item -- a few GB total, not the raw pixels, which
get discarded after embedding extraction.

