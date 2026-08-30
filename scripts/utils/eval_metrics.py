"""
Shared helpers for evaluate_base.py / evaluate_concat.py / evaluate_film.py's
false-positive/false-negative-rate-by-threshold table, confound-slice (source_dataset,
generator_family) breakdowns, and per-example FP/FN CSV dumps -- logic identical across
all three, only how each script gets its probs/labels/img_ids differs (handled by the
caller, not here).
"""
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

# 3 operating points, not just the default 0.5 -- a single threshold can hide a model
# that's only "accurate" because it's mis-calibrated in a way that happens to land on
# the right side of 0.5 most of the time.
FPR_THRESHOLDS = (0.3, 0.5, 0.7)

# Columns for the per-example FP/FN CSVs -- "path" is reconstructed, not read from disk,
# from the same (split, img_id, variant) -> file layout extract_embeddings.py uses
# (data/cache/clean/<split>/<img_id>/<variant>.jpg), so opening it doesn't require
# re-consulting the manifest.
ERROR_CSV_FIELDS = ["img_id", "variant", "source_dataset", "generator_family", "label", "prob", "path"]


def false_positive_rate(probs: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    """label 1 = fake (positive class), label 0 = real (negative class).
    FPR = FP / (FP + TN) -- fraction of REAL images wrongly flagged as fake at this
    threshold. nan (not 0) if there are no real images in the slice at all, since a
    0% FPR on zero real images means "undefined," not "perfect.\""""
    preds = (probs > threshold).astype(int)
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    return fp / (fp + tn) if (fp + tn) else float("nan")


def false_negative_rate(probs: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    """Mirror of false_positive_rate(): FNR = FN / (FN + TP) -- fraction of FAKE images
    wrongly passed off as real at this threshold. This is the more consequential miss
    for a fake-detector (a fake that slipped through) and is exactly what's well-defined
    on the fake-only generator_family slices where FPR is nan (no real images there to
    compute FPR against). nan if there are no fake images in the slice at all."""
    preds = (probs > threshold).astype(int)
    fn = int(((preds == 0) & (labels == 1)).sum())
    tp = int(((preds == 1) & (labels == 1)).sum())
    return fn / (fn + tp) if (fn + tp) else float("nan")


def print_fpr_table(title: str, groups: dict) -> None:
    """groups: {row_name: (probs, labels)}, e.g. one row per variant, or one row per
    source_dataset. Prints accuracy/auc (at the canonical 0.5 threshold) plus FPR and
    FNR at each of FPR_THRESHOLDS, one row per group, in the order given."""
    header = (f"{'':<14} {'n':>7} {'accuracy':>10} {'auc':>10}"
              + "".join(f"  fpr@{t:<4}" for t in FPR_THRESHOLDS)
              + "".join(f"  fnr@{t:<4}" for t in FPR_THRESHOLDS))
    print(f"\n{title}")
    print(header)
    print("-" * len(header))
    for name, (probs, labels) in groups.items():
        preds = (probs > 0.5).astype(int)
        acc = float((preds == labels).mean()) if len(labels) else float("nan")
        auc = float(roc_auc_score(labels, probs)) if len(set(labels.tolist())) > 1 else float("nan")
        row = f"{name:<14} {len(labels):>7} {acc:>10.4f} {auc:>10.4f}"
        for t in FPR_THRESHOLDS:
            row += f"  {false_positive_rate(probs, labels, t):>7.4f}"
        for t in FPR_THRESHOLDS:
            row += f"  {false_negative_rate(probs, labels, t):>7.4f}"
        print(row)


def collect_errors(
    probs: np.ndarray, labels: np.ndarray, img_ids: np.ndarray, variant: str,
    source_dataset: np.ndarray, generator_family: np.ndarray, cache_root: str,
    split: str = "test", threshold: float = 0.5,
) -> tuple:
    """Splits one variant's rows into (fp_rows, fn_rows) at the canonical 0.5 threshold
    -- FP: real image (label 0) predicted fake; FN: fake image (label 1) predicted real.
    Each row is a dict matching ERROR_CSV_FIELDS, ready to hand straight to
    csv.DictWriter. `path` is reconstructed (not verified to exist) from the same
    directory layout extract_embeddings.py reads from."""
    preds = (probs > threshold).astype(int)
    fp_mask = (preds == 1) & (labels == 0)
    fn_mask = (preds == 0) & (labels == 1)

    def rows_for(mask: np.ndarray) -> list:
        return [
            {
                "img_id": str(img_ids[i]),
                "variant": variant,
                "source_dataset": str(source_dataset[i]),
                "generator_family": str(generator_family[i]),
                "label": int(labels[i]),
                "prob": float(probs[i]),
                "path": str(Path(cache_root) / split / str(img_ids[i]) / f"{variant}.jpg"),
            }
            for i in np.nonzero(mask)[0]
        ]

    return rows_for(fp_mask), rows_for(fn_mask)


def write_error_csv(path: Path, rows: list) -> None:
    """Writes one FP or FN CSV, creating parent dirs as needed. Always writes the
    header, even for an empty `rows` (a clean "0 errors" file beats a missing one --
    missing could mean "no errors" or "never ran")."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ERROR_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
