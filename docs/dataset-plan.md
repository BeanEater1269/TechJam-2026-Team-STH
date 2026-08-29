# Track 5 — Dataset Plan (locked, as of tonight)

Base pool: **30,000 images** (before the 15 transform variants), split **15,000 real / 15,000 fake**.
Working resolution: **680×680**, saved as JPEG-95 (except the 4 JPEG-compression transform
variants, which save at their actual target quality: 90/70/50/30 -- that quality drop *is*
the transform, not an accident on top of it).

## Sources

| Source | Real | Fake | Native size / shape | Content |
|---|---|---|---|---|
| WildFake | -- | ~8,000 | 1024x1024 only -- the 512-native half is excluded | AI-generated images across GAN, diffusion, and other generator families. Hierarchically organized by generator type. |
| SID-Set | ~10,000 | ~2,000 | Real: not square natively -- long side ~1024, short side ~490-760px, median ~680. Cropped to square (matching the short side) before resizing. Fake: 1024x1024, full-synthetic only (label 1). Tampered (label 2) excluded. | Real: everyday/social-media-style photos, sourced from OpenImages V7. Fake: matched AI generations in the same style. |
| CIFAKE | 5,000 | 5,000 | 32x32, both sides, already square | Real: CIFAR-10 photos. Fake: Stable-Diffusion-generated matches. |
| **Total** | **15,000** | **15,000** | | **= 30,000** |

WildFake's ~8,000 fake, illustrative split pending an actual count in the file browser:
~3,500 GAN / ~3,500 diffusion / ~1,000 other.

## Why 680, not 1024 or 512

CLIP itself resizes every input to a fixed size (224x224) no matter what -- the working
resolution isn't for CLIP, it's so the robustness transforms (blur, JPEG, crop, etc.) get
simulated at something close to a real photo's actual size, and so the classical signals
(Laplacian variance, DCT split) have real detail to measure.

680 sits close to SID-Set's real-crop median, so most real photos need little or no
upsampling to reach it, while every fake (native 1024) safely downsamples. 512 would remove
the upsample risk entirely but costs real detail the B-signals rely on; 1024 would force
nearly every real photo to stretch. 680 is the balance point.

## Excluded, and why

| Excluded | Reason |
|---|---|
| WildFake's 512-native images (either side) | Would need heavy upsampling to reach 680 -- reintroduces the resolution-vs-label confound |
| WildFake's real images, by default | Risk of overlapping with the hackathon's own COCO-based demo validation set (leakage) |
| SID-Set's tampered images (label 2) | Mostly-real photo with one small edited region -- a different task than whole-image AIGC detection, and a confusing label if included |

## Known residual risks -- tagged in the manifest, checked after training, not silently ignored

| Risk | Manifest tag | What the post-training check looks for |
|---|---|---|
| CIFAKE's extreme 32->680 upsample | `source_dataset == cifake` | Suspiciously high/low accuracy on this slice specifically |
| SID-Set real crops with native short side well under 680 | `native_short_side` | Accuracy gap between upsampled vs. non-upsampled real photos |
| Generator-level resolution gaps inside WildFake's kept slice | `generator_family` | Accuracy gap by generator, once real GAN/diffusion counts are known |

## Optional upgrades -- real, worth doing with spare time, not required for a working pipeline

- A *modest* slice of WildFake's FFHQ/CelebA-HQ (natively 1024, non-COCO) swapped in for
  some of SID-Set's real allocation. Keep it genuinely minor -- both are face-only content,
  and leaning on them too hard trades the resolution problem for a face-vs-everything
  semantic bias.
- Filtering WildFake's real images down to just its non-COCO sources more broadly, if the
  file structure allows a clean split.
- Confirming the actual GAN/diffusion counts inside WildFake's 1024-native slice, to firm up
  the ~3,500/3,500/1,000 split above.
- CLIP backbone: ViT-B/32 vs ViT-L/14 -- pending the timing check. B/32 alone is the
  lighter, bandwidth-friendly default for tonight.

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
