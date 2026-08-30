"""
Scans the raw-data folders below, tags every image (source, generator_family, label),
then does the stratified 80/10/10 split -- writes data/manifest.csv.

Run this AFTER downloading data, BEFORE running build_cache.py. This is where the
train/val/test decision gets made -- once, on the raw images, before any transform exists.
build_cache.py just reads the "split" column this script writes; it makes no split
decisions of its own.

Paths are hardcoded below (RAW_SOURCES) instead of passed on the command line, since
right now the four sources live in four different places (two different drives,
Downloads folder, and the repo's own data/raw/). Update RAW_SOURCES if a path moves.

Usage:
    python scripts/build_manifest.py
    python scripts/build_manifest.py --out data/manifest.csv
"""
import argparse
import random
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


# ---------------------------------------------------------------------------
# Hardcoded raw-data locations. Everything gets tagged + merged in build_manifest().
# Add/remove/edit entries here as sources get downloaded or move -- nothing else in
# this file needs to change to pick up a new path.
# ---------------------------------------------------------------------------
CIFAKE_ROOT = Path(r"D:\Coding\playground\TechJam-2026-Team-STH\data\raw\archive")
STYLEGAN_ROOT = Path(r"C:\Users\hunte\Downloads\StyleGAN-XL")
GIGAGAN_ROOT = Path(r"C:\Users\hunte\Downloads\fake_images")
DALLE_ROOT = Path(r"C:\Users\hunte\Downloads\DALLE (1)")
IMAGEN3_ROOT = Path(r"C:\Users\hunte\Downloads\Imagen3")

# Set once SID-Set was downloaded; None disables it (see the `if SIDSET_ROOT is not None`
# check in build_manifest() below -- that's the single switch that turns this source on).
SIDSET_ROOT = Path(r"D:\Coding\playground\TechJam-2026-Team-STH\data\raw\sidset")

# Per-source cap on how many images to keep, applied as a random subsample AFTER
# scanning each source in full (so the discard is a uniform random draw, not just
# "whatever the OS lists first"). A source with no entry here (or set to None) keeps
# every image it finds -- e.g. sidset isn't listed because it's already randomized
# upstream and doesn't need capping here.
# Per-source cap on how many images to keep, applied as a random subsample AFTER
# scanning each source in full (so the discard is a uniform random draw, not just
# "whatever the OS lists first"). A source with no entry here (or set to None) keeps
# every image it finds -- e.g. sidset isn't listed because it's already randomized
# upstream and doesn't need capping here.
#
# Two forms are supported:
#   - a plain int: cap = uniform random sample over ALL of that source's rows,
#     regardless of label. Fine for stylegan/gigagan/dalle/imagen3 since those are
#     single-label (fake-only) sources -- there's no class to accidentally skew.
#   - a dict of {label: count}: cap PER LABEL, sampled independently, so the mix
#     comes out exactly as specified. Needed for cifake, which has real AND fake
#     images -- a plain int cap there would sample uniformly across the combined
#     REAL+FAKE pool and just inherit whatever ratio the folder happens to have,
#     not necessarily an even split.
SAMPLE_CAPS = {
    "cifake": {0: 3000, 1: 3000},  # 0 = real, 1 = fake -- 3000 of each, guaranteed
    "stylegan_xl": 2500,
    "gigagan": 2500,
    "dalle": 2500,
    "imagen3": 2500,
}


# ---------------------------------------------------------------------------
# One scan_*() function per source. Each just needs to return the same shape:
# a list of dicts with path / source_dataset / generator_family / label.
# ---------------------------------------------------------------------------

def scan_cifake(root: Path) -> list[dict]:
    """CIFAKE: train/ and test/ splits, each with REAL/ and FAKE/ subfolders.
    We ignore CIFAKE's own train/test split -- we're building one 80/10/10 split
    across every source combined, so we just want every image, tagged."""
    rows = []
    for split_dir in ("train", "test"):
        for label_name, label in (("REAL", 0), ("FAKE", 1)):
            folder = root / split_dir / label_name
            if not folder.exists():
                continue
            for p in folder.rglob("*"):
                if p.suffix.lower() in IMG_EXTS:
                    rows.append({
                        "path": str(p),
                        "source_dataset": "cifake",
                        "generator_family": "stable_diffusion" if label == 1 else "real",
                        "label": label,  # 0 = real, 1 = fake
                    })
    return rows


def _scan_flat_fake_folder(root: Path, source_dataset: str, generator_family: str) -> list[dict]:
    """Shared logic for raw dumps that are just a folder of images from one generator,
    with no real/fake split inside (e.g. StyleGAN-XL / GigaGAN / DALLE samples downloaded
    straight into a single folder). If real/ and fake/ subfolders DO exist under root,
    use those instead of assuming everything is fake -- checked first so we don't
    mislabel a source that actually ships its own real images alongside generated ones.
    """
    real_dir, fake_dir = root / "real", root / "fake"
    if real_dir.exists() or fake_dir.exists():
        rows = []
        for folder, label in ((real_dir, 0), (fake_dir, 1)):
            if not folder.exists():
                continue
            for p in folder.rglob("*"):
                if p.suffix.lower() in IMG_EXTS:
                    rows.append({
                        "path": str(p),
                        "source_dataset": source_dataset,
                        "generator_family": "real" if label == 0 else generator_family,
                        "label": label,
                    })
        return rows

    # No real/fake split found -- treat every image under root as fake, generated by
    # `generator_family`. Matches folders like "StyleGAN-XL" / "fake_images" / "DALLE (1)"
    # that only ever contained generated samples.
    rows = []
    for p in root.rglob("*"):
        if p.suffix.lower() in IMG_EXTS:
            rows.append({
                "path": str(p),
                "source_dataset": source_dataset,
                "generator_family": generator_family,
                "label": 1,
            })
    return rows


def scan_stylegan(root: Path) -> list[dict]:
    """StyleGAN-XL samples downloaded into Downloads/StyleGAN-XL."""
    return _scan_flat_fake_folder(root, source_dataset="stylegan_xl", generator_family="stylegan_xl")


def scan_gigagan(root: Path) -> list[dict]:
    """GigaGAN samples downloaded into Downloads/fake_images."""
    return _scan_flat_fake_folder(root, source_dataset="gigagan", generator_family="gigagan")


def scan_dalle(root: Path) -> list[dict]:
    """DALLE samples downloaded into Downloads/DALLE (1)."""
    return _scan_flat_fake_folder(root, source_dataset="dalle", generator_family="dalle")


def scan_imagen3(root: Path) -> list[dict]:
    """Imagen3 samples downloaded into Downloads/Imagen3."""
    return _scan_flat_fake_folder(root, source_dataset="imagen3", generator_family="imagen3")


def scan_sidset(root: Path) -> list[dict]:
    """SID-Set, as saved by download_sidset.py: real/ and fake/ subfolders directly
    (tampered was never downloaded in the first place, so there's nothing to skip here).
    Not wired into build_manifest() yet -- set SIDSET_ROOT above and add it to
    RAW_SOURCES once this source is actually downloaded."""
    rows = []
    for folder_name, label, family in (("real", 0, "real"), ("fake", 1, "full_synthetic")):
        folder = root / folder_name
        if not folder.exists():
            continue
        for p in folder.rglob("*"):
            if p.suffix.lower() in IMG_EXTS:
                rows.append({
                    "path": str(p),
                    "source_dataset": "sidset",
                    "generator_family": family,
                    "label": label,
                })
    return rows


def make_unique_img_ids(df: pd.DataFrame) -> pd.Series:
    """Builds an img_id that's guaranteed unique across the ENTIRE manifest.

    Source-prefixing the filename stem (e.g. "gigagan_00048") handles the main
    collision case -- stylegan_xl/gigagan/dalle/imagen3 are all just sequentially
    numbered downloads, so the same bare stem shows up in more than one of them.

    That alone isn't quite enough though: CIFAKE numbers its REAL/ and FAKE/ folders
    independently, so "cifake_0001" can still collide within a single source. The
    counter-suffix pass below catches any of those remaining clashes, whatever source
    they come from, without needing to special-case CIFAKE by name.

    This MUST stay unique: build_cache.py uses img_id as the cache output folder name
    (a collision means one image's cached files silently overwrite another's), and
    extract_embeddings.py uses manifest.loc[img_id] to look up metadata (a collision
    means that lookup returns two rows instead of one)."""
    base = df["source_dataset"] + "_" + df["path"].apply(lambda p: Path(p).stem)
    seen: dict[str, int] = {}
    ids = []
    for b in base:
        n = seen.get(b, 0)
        seen[b] = n + 1
        ids.append(b if n == 0 else f"{b}_{n}")
    return pd.Series(ids, index=df.index)


def is_readable(path: str) -> bool:
    """Full decode check -- catches the same failures build_cache.py hit last time
    ("cannot identify image file", "broken PNG file"). add_pixel_size()'s lazy
    im.size read only parses the file header, which isn't enough: some corrupted
    files pass a header read fine and only fail once something actually decodes the
    pixel data (exactly what build_cache.py's Image.open(...).convert("RGB") does).
    im.load() forces that same full decode here, so a corrupted file gets caught and
    excluded at manifest-build time instead of surfacing as a build_cache.py failure
    after the (much more expensive) caching step has already started."""
    try:
        with Image.open(path) as im:
            im.load()
        return True
    except Exception:
        return False


def _take_valid_sample(rows: list[dict], cap: int | None, rng: random.Random) -> tuple[list[dict], int]:
    """Randomly samples up to `cap` READABLE rows from `rows` (or all readable rows,
    if cap is None). Shuffles first, then validates one at a time and stops as soon
    as `cap` valid images are found -- this avoids fully decoding an entire multi-
    thousand-image pool just to keep a small capped sample; the corruption check only
    costs roughly `cap + a handful of skips`, not the whole source. Returns
    (kept_rows, num_skipped_as_unreadable)."""
    shuffled = rows[:]
    rng.shuffle(shuffled)

    kept: list[dict] = []
    skipped = 0
    for r in shuffled:
        if cap is not None and len(kept) >= cap:
            break
        if is_readable(r["path"]):
            kept.append(r)
        else:
            skipped += 1
    return kept, skipped


def add_pixel_size(df: pd.DataFrame) -> pd.DataFrame:
    """Reads each image's native width/height directly off disk. Lazy read -- PIL only
    parses the file header for .size, it doesn't decode the full pixel data, so this stays
    fast even across tens of thousands of images. Applied once, here, to every row --
    individual scan_*() functions don't need to do this themselves."""
    widths, heights = [], []
    for p in df["path"]:
        try:
            with Image.open(p) as im:
                w, h = im.size
        except Exception:
            w, h = None, None
        widths.append(w)
        heights.append(h)
    df = df.copy()
    df["width"] = widths
    df["height"] = heights
    return df


def build_manifest(sample_seed: int = 0) -> pd.DataFrame:
    rows: list[dict] = []
    rng = random.Random(sample_seed)

    # name -> (root path, scan function). This is the single place that links each
    # hardcoded path above to the scan_*() function that knows its folder layout.
    RAW_SOURCES = [
        ("cifake", CIFAKE_ROOT, scan_cifake),
        ("stylegan_xl", STYLEGAN_ROOT, scan_stylegan),
        ("gigagan", GIGAGAN_ROOT, scan_gigagan),
        ("dalle", DALLE_ROOT, scan_dalle),
        ("imagen3", IMAGEN3_ROOT, scan_imagen3),
    ]
    if SIDSET_ROOT is not None:
        RAW_SOURCES.append(("sidset", Path(SIDSET_ROOT), scan_sidset))

    for name, root, scan_fn in RAW_SOURCES:
        if not root.exists():
            print(f"  (skipping {name} -- {root} not found)")
            continue
        source_rows = scan_fn(root)
        cap = SAMPLE_CAPS.get(name)

        if isinstance(cap, dict):
            # Per-label cap: sample each label's rows independently so the kept mix
            # matches the requested counts exactly, instead of a single combined draw
            # that would just inherit whatever ratio the source folder happens to have.
            capped_rows = []
            for label, n in cap.items():
                label_rows = [r for r in source_rows if r["label"] == label]
                valid_rows, skipped = _take_valid_sample(label_rows, n, rng)
                if skipped:
                    print(f"  {name} (label={label}): skipped {skipped} unreadable/corrupted file(s)")
                if len(valid_rows) < n:
                    print(f"  {name} (label={label}): only found {len(valid_rows)} readable, "
                          f"wanted {n} -- keeping all readable ones")
                else:
                    print(f"  {name} (label={label}): found {len(label_rows)}, randomly keeping {n} readable")
                capped_rows.extend(valid_rows)
            source_rows = capped_rows
        else:
            valid_rows, skipped = _take_valid_sample(source_rows, cap, rng)
            if skipped:
                print(f"  {name}: skipped {skipped} unreadable/corrupted file(s)")
            if cap is not None:
                if len(valid_rows) < cap:
                    print(f"  {name}: only found {len(valid_rows)} readable, wanted {cap} "
                          f"-- keeping all readable ones")
                else:
                    print(f"  {name}: found {len(source_rows)}, randomly keeping {cap} readable")
            source_rows = valid_rows

        rows.extend(source_rows)

    # TODO (teammate): if any additional source gets decided later (still open as of
    # tonight), add its path near the top, write its scan_<name>(), and add a row to
    # RAW_SOURCES above -- same pattern as every other source.

    if not rows:
        raise SystemExit("No images found across any configured RAW_SOURCES path -- "
                          "check the hardcoded paths at the top of this file")

    df = pd.DataFrame(rows)
    # img_id must be unique across the WHOLE manifest, not just within one source.
    # A bare filename stem (e.g. "00048") is not enough -- multiple sources
    # (stylegan_xl/gigagan/dalle/imagen3 in particular, since they're all just
    # sequentially-numbered downloads) can and do reuse the same stem. See
    # make_unique_img_ids() above for the full collision story. This column is the
    # single source of truth downstream: build_cache.py uses it as the cache folder
    # name, extract_embeddings.py uses it to look metadata back up via
    # manifest.loc[img_id] -- both need it unique or images silently overwrite each
    # other / metadata lookups return the wrong row.
    df["img_id"] = make_unique_img_ids(df)
    df = add_pixel_size(df)
    print(f"\nFound {len(df)} images total:")
    print(df.groupby(["source_dataset", "generator_family", "label"]).size().to_string())
    print(f"\nPixel-size range: width {df['width'].min()}-{df['width'].max()}, "
          f"height {df['height'].min()}-{df['height'].max()}")
    return df


def add_split(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """80/10/10, stratified by source + generator_family + label together, so no split
    ends up accidentally all-one-source or all-one-label. If this errors with a
    "least populated class" message, some (source, generator_family, label) group is too
    small to stratify on -- either that source needs more images, or the stratification
    needs to fall back to something coarser for that group."""
    strat_key = (
        df["source_dataset"].astype(str) + "_"
        + df["generator_family"].astype(str) + "_"
        + df["label"].astype(str)
    )

    train_idx, temp_idx = train_test_split(
        df.index, test_size=0.2, stratify=strat_key, random_state=seed
    )
    temp_strat = strat_key.loc[temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=temp_strat, random_state=seed
    )

    df = df.copy()
    df["split"] = "train"
    df.loc[val_idx, "split"] = "val"
    df.loc[test_idx, "split"] = "test"
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/manifest.csv")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = build_manifest(sample_seed=args.seed)
    df = add_split(df, seed=args.seed)

    print("\nSplit sizes:")
    print(df["split"].value_counts().to_string())
    print("\nSplit x label (should look proportional across splits):")
    print(df.groupby(["split", "label"]).size().to_string())

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nWrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()