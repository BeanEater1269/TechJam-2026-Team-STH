# Image Standardization Pipeline -- Final Decisions

## What we decided

1. Working resolution: **512x512**
2. Every image gets **cropped to square** before its final resize
3. 1024-native fake images additionally get a **small random crop** during training/caching only
4. Inference: **crop to square -> resize to 512 -> feed to model.** Identical for every image, no randomness, no branching.

## Why 512x512

WildFake's fake images arrive at roughly two native sizes -- a slice near 512, a slice near
1024. (Whether 512-native skews toward older GAN checkpoints specifically is a pattern worth
confirming once we're actually in the file browser, not something we've counted yet.)

Picking 512 as our working resolution does two things at once:

- **The 512-native fakes need zero processing to reach 512 -- an exact match.** Our earlier
  plan excluded this entire slice to avoid an upsampling risk. At 512, that exclusion is no
  longer necessary -- we get the diversity back for free.
- **Every real photo, after cropping to square, safely downsamples to 512 -- no exceptions,**
  including our most extreme aspect-ratio outliers. Upsampling risk on the real side isn't
  reduced. It's eliminated by construction.

This matters specifically for robustness, not just general data hygiene: four of our six
required transforms -- blur, JPEG compression, Gaussian noise, and resize-down-then-up -- all
reduce sharpness further on top of whatever a real image already has. If any
upsampling-induced softness had survived into training, those four transforms would actively
trigger it during the exact evaluation this track is graded on. 512 removes that interaction
entirely, not just on average.

## The cost, stated plainly

1024-native fake images (WildFake and SID-Set) retain only about **25%** of their native
pixel area once resized to 512. This cost lands specifically on our two classical signals --
Laplacian variance and the DCT frequency split -- since both exist to read fine detail, and
that's exactly what a hard downsample removes first.

**This cost does not touch CLIP.** CLIP resizes every image to its own small fixed input
internally, regardless of what resolution we hand it, so 512 is invisible to CLIP either way.
The 25% loss is a cost to our supplementary classical signals only, not to our primary
detection pathway.

The actual trade: some strength in the classical signals, in exchange for removing a shortcut
that would specifically undermine robustness -- the thing Track 5 is scored on, not
clean-data accuracy.

## Why crop to square, not rectangle, not "leave it alone"

CLIP's own preprocessing crops every image to square internally, unconditionally, before it
computes anything -- this isn't something we configure, it happens regardless of what we feed
it.

Given that, cropping real photos to square ourselves is the version that costs the least and
is the one we actually control:

- Fakes need no shape processing at all -- already square, zero-cost no-op.
- Reals need exactly one crop, done our way: a center crop (not squash, not pad), so the
  result looks like an ordinary photo with no visible trace that anything was cut.
- Our classical signals see the standardized square pixels directly, since they run before
  CLIP touches anything.

We considered the reverse -- cropping fakes down to a rectangle to match real photos' natural
shape -- and rejected it, for two concrete reasons:

1. **It doesn't actually keep square images from reaching the model.** CLIP crops to square
   internally regardless of what shape we hand it, so a "rectangularized" fake would be
   cropped once by us and then again by CLIP -- content lost twice, for a shape that never
   survives to matter.
2. **Our classical signals never touch CLIP's preprocessing.** They run directly on our
   cached pixels. If we stopped cropping reals to square, those two signals would be computed
   on non-square reals versus square fakes -- reintroducing the exact aspect-ratio shortcut we
   built this whole step to remove, specifically in the one place CLIP's own behavior can't
   reach.

Crop-real-to-square: content lost once, deliberately, protects both halves of the
architecture. Crop-fake-to-rectangle: content lost twice, only protects the half CLIP was
already going to fix on its own.

## The real cost of cropping to square, acknowledged directly

A typical real photo in our data keeps roughly two-thirds of its area after the square crop;
our narrowest outliers keep closer to half. Real content lost, not a rounding error.

What makes it acceptable: the result looks like an ordinary square photo -- nothing about it
reveals that a crop happened. That's different from the two alternatives we ruled out earlier
(squashing, which visibly warps the image; padding, which leaves visible bars) -- both leave a
mark a classifier could learn to detect. A clean center crop doesn't.

## One step that's training-only, not part of the real pipeline

1024-native fake images get a small random crop (keeping roughly 90-100% of the frame, chosen
fresh per image) before resizing to 512 -- during caching, on training data only.

This exists so every 1024-native fake isn't resized by the exact same fixed ratio, which
would otherwise be a subtle, consistent fingerprint -- exactly what our DCT signal is built to
notice.

This step never appears at inference. The trained model doesn't need protecting from a
pattern that's no longer in its weights, and at inference time there's no label yet to decide
whether an incoming image should get "fake-only" treatment in the first place. The real
pipeline is unconditional and identical for every image:

**crop to square -> resize to 512 -> feed to model.**

No randomness, no branching, no dependence on an answer we don't have yet.
