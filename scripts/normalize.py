"""
Shared z-score normalization for the classical B-signals (laplacian_var,
dct_low_energy, dct_high_energy, noise_variance) computed by signals.py.

Any script that concatenates or otherwise feeds these raw signals into a model should
import from here instead of re-deriving its own stats -- keeps every consumer using
the exact same normalization rule.

Stats (mean, std) are always computed from the TRAIN split's raw signal file ONLY,
then applied unchanged to whichever split is actually being loaded (train/val/test).
Computing separate stats per split would leak that split's own distribution into its
own scaling, and would mean val/test are being scored on a different feature scale
than the one the model was actually trained on -- always reuse train's mean/std,
never recompute per split.

Usage:
    from normalize import compute_train_stats, apply_normalization
    mean, std = compute_train_stats(stats_root / "train_signals.npz", SIGNAL_COLUMNS)
    normalized = apply_normalization(raw_signals, mean, std)
"""
from pathlib import Path

import numpy as np


def compute_train_stats(train_stats_path: Path, signal_columns: list) -> tuple:
    """Computes per-column mean and std from the TRAIN split's raw signal file only.
    Returns (mean, std), each shape (len(signal_columns),), in the same column order
    passed in -- callers must build their per-row signal arrays in that same order."""
    data = np.load(train_stats_path, allow_pickle=True)
    cols = np.stack([data[col] for col in signal_columns], axis=1).astype(np.float64)
    mean = cols.mean(axis=0)
    std = cols.std(axis=0)
    std[std == 0] = 1.0  # guard against a constant column producing a divide-by-zero
    return mean.astype(np.float32), std.astype(np.float32)


def apply_normalization(signals: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Z-scores a (N, len(signal_columns)) or (len(signal_columns),) signals array
    using stats already produced by compute_train_stats(). Same transform regardless
    of which split is being normalized -- train, val, and test all get the TRAIN
    split's mean/std, never their own."""
    return (signals - mean) / std
