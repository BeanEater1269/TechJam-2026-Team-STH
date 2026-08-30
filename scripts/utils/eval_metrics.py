"""
Shared helpers for evaluate_base.py / evaluate_concat.py / evaluate_film.py's
false-positive-rate-by-threshold table and confound-slice (source_dataset,
generator_family) breakdowns -- logic identical across all three, only how each
script gets its probs/labels differs (handled by the caller, not here).
"""
import numpy as np
from sklearn.metrics import roc_auc_score

# 3 operating points, not just the default 0.5 -- a single threshold can hide a model
# that's only "accurate" because it's mis-calibrated in a way that happens to land on
# the right side of 0.5 most of the time.
FPR_THRESHOLDS = (0.3, 0.5, 0.7)


def false_positive_rate(probs: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    """label 1 = fake (positive class), label 0 = real (negative class).
    FPR = FP / (FP + TN) -- fraction of REAL images wrongly flagged as fake at this
    threshold. nan (not 0) if there are no real images in the slice at all, since a
    0% FPR on zero real images means "undefined," not "perfect.\""""
    preds = (probs > threshold).astype(int)
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    return fp / (fp + tn) if (fp + tn) else float("nan")


def print_fpr_table(title: str, groups: dict) -> None:
    """groups: {row_name: (probs, labels)}, e.g. one row per variant, or one row per
    source_dataset. Prints accuracy/auc (at the canonical 0.5 threshold) plus FPR at
    each of FPR_THRESHOLDS, one row per group, in the order given."""
    header = f"{'':<14} {'n':>7} {'accuracy':>10} {'auc':>10}" + "".join(f"  fpr@{t:<4}" for t in FPR_THRESHOLDS)
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
        print(row)
