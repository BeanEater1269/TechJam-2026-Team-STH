"""
The 6 robustness transforms + the shape/resolution standardization steps.

Locked parameters (see ../docs/pipeline-decisions.md for the full reasoning):
- Working resolution: 512x512
- Real images: center-crop to square (matching the short side) BEFORE resizing to 512
- Fake images, 1024-native: small random crop (keep ~90-100% of frame) BEFORE resizing to 512
- Fake images, 512-native: no processing needed, already at target
- All 15 variants get generated from the clean, already-standardized (square, 512) image
- Save format: JPEG-95 for everything, EXCEPT the 4 JPEG-compression variants, which save
  at their actual target quality (90/70/50/32) directly -- that quality drop IS the transform

Transform categories, per the brief's parameter table:
- JPEG Compression: quality = 90, 70, 50, 30
- Gaussian Blur: sigma = 0.5, 1.0, 2.0
- Resize: scale 0.5x / 0.25x, then upscale back
- Gaussian Noise: sigma = 0.02, 0.05, 0.10
- Color Jitter: brightness/contrast/saturation +/-20%
- Center Crop: crop 80% (this is the ROBUSTNESS transform -- separate from, and applied
  AFTER, the shape-fix crop above)

TODO: implement once data is in. This file is a placeholder so the repo structure and the
locked parameters are visible from the start.
"""
