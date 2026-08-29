"""
The classical signals fed to the classifier alongside the CLIP embedding.

- Laplacian variance (blur/sharpness)
- DCT energy, split low-band / high-band
- Noise variance
- CLIP-drift probe (nudge the image, measure embedding movement)

These run on the raw cached pixels directly -- they never touch CLIP's own preprocessing,
which is exactly why the shape/resolution standardization in transforms.py has to happen
before these are computed, not left to CLIP to handle. See ../docs/pipeline-decisions.md.

TODO: implement once data is in.
"""
