"""
Frozen CLIP + small trainable classifier head.

CLIP is never fine-tuned. Only the small head on top trains. Backbone choice (ViT-B/32 vs
ViT-L/14) is decided by scripts/check_clip_backbone_timing.py -- default to ViT-B/32 unless
that check says otherwise.

TODO: implement once embeddings are cached.
"""
