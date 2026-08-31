"""
Evaluates a trained FiLMClassifier (from train_film_drift_liqe.py ONLY -- the 5-signal
train_film_normalize_drift.py/model_film_drift_normalized.pt path is a separate,
untouched baseline) on the TEST split, broken out per variant. Same table shape as
evaluate_film_drift.py -- same test images, same metrics -- except this model takes 6
signals (the 4 B-signals + clip_drift + liqe_score) instead of 5.

Signals are ALWAYS z-scored before evaluation, using TRAIN split stats (via
normalize.py) -- not optional, since FiLMClassifier uses the signals to generate a
scale/shift that modulates the CLIP embedding directly; unnormalized signals would
distort that modulation before the MLP ever sees it.

Also reports:
  - False positive rate at 3 thresholds (0.3 / 0.5 / 0.7), per variant.
  - Accuracy/AUC/FPR broken out by source_dataset and by generator_family, on the clean
    images only -- per dataset-plan.md's "Known residual risks" table.

Merges data/cache/embeddings/test.npz with data/cache/stats/test_signals.npz,
test_drift.npz, AND test_liqe.npz by matching (img_id, variant) as keys -- same
reasoning as train_film_drift_liqe.py's load_merged_split(), just grouped by variant
here instead of by img_id (the robustness table needs "every jpeg_q30 row" as one
batch, not paired for DCPT).

Run this AFTER train_film_drift_liqe.py has produced a checkpoint. Touches test.npz,
which nothing else in the pipeline reads -- meant to be run once, at the end.

Usage:
    python scripts/evaluation/evaluate_film_drift_liqe.py
    python scripts/evaluation/evaluate_film_drift_liqe.py --checkpoint checkpoints/model_film_drift_liqe_normalized.pt
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from eval_metrics import collect_errors, print_fpr_table, write_error_csv  # noqa: E402
from normalize import apply_normalization, compute_train_stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from model_film import FiLMClassifier  # noqa: E402

SIGNAL_COLUMNS = ["laplacian_var", "dct_low_energy", "dct_high_energy", "noise_variance"]
DRIFT_COLUMNS = ["clip_drift"]
LIQE_COLUMNS = ["liqe_score"]
ALL_SIGNAL_COLUMNS = SIGNAL_COLUMNS + DRIFT_COLUMNS + LIQE_COLUMNS
SIGNAL_DIM = len(ALL_SIGNAL_COLUMNS)


def _build_lookup(stats_data, columns: list) -> dict:
    """(img_id, variant) -> raw (unnormalized) column vector, for one stats npz. Keys
    are normalized to plain Python str on both sides of every comparison -- see
    train_film_drift_liqe.py's _build_lookup() for why that makes the three files'
    independently-derived id arrays safe to merge regardless of exact numpy dtype."""
    img_ids, variant = stats_data["img_ids"], stats_data["variant"]
    matrix = np.stack([stats_data[col] for col in columns], axis=1).astype(np.float32)
    return {(str(img_ids[i]), str(variant[i])): matrix[i] for i in range(len(img_ids))}


def load_by_variant(
    embeddings_path: Path, base_stats_path: Path, drift_stats_path: Path, liqe_stats_path: Path,
    mean: np.ndarray, std: np.ndarray,
) -> dict:
    """Merges embeddings + 4 B-signals + clip_drift + liqe_score by (img_id, variant),
    then groups by variant_name -> (embeddings_array, signals_array [6-dim, normalized],
    labels_array, source_dataset_array, generator_family_array, img_ids_array). Unlike
    train_film_drift_liqe.py's load_merged_split() (grouped by img_id, for DCPT
    pairing), this groups by variant, since the robustness table needs "every jpeg_q30
    row" as one batch, not paired. source_dataset/generator_family/img_ids come from the
    embeddings file and are carried through for every variant -- only the "clean"
    group's copies get used for the confound-slice breakdown, but the FP/FN CSV dump
    needs every variant's img_ids to name the exact example.

    Arrays are pulled out of every NpzFile ONCE, up front -- see
    train_film_normalize.py's load_merged_split() for why indexing data[key][i] in a
    loop is a correctness/memory bug, not just a style choice.

    mean/std are always the TRAIN split's (concatenated from three compute_train_stats()
    calls in main(), ALL_SIGNAL_COLUMNS order) -- signals are z-scored with that exact
    transform before being returned, unconditionally, since this script only ever
    evaluates a checkpoint that was trained on normalized signals."""
    emb_data = np.load(embeddings_path, allow_pickle=True)
    base_data = np.load(base_stats_path, allow_pickle=True)
    drift_data = np.load(drift_stats_path, allow_pickle=True)
    liqe_data = np.load(liqe_stats_path, allow_pickle=True)

    base_lookup = _build_lookup(base_data, SIGNAL_COLUMNS)
    drift_lookup = _build_lookup(drift_data, DRIFT_COLUMNS)
    liqe_lookup = _build_lookup(liqe_data, LIQE_COLUMNS)

    e_embeddings, e_img_ids, e_variant, e_labels = (
        emb_data["embeddings"], emb_data["img_ids"], emb_data["variant"], emb_data["labels"]
    )
    e_source, e_family = emb_data["source_dataset"], emb_data["generator_family"]
    by_variant: dict = defaultdict(lambda: ([], [], [], [], [], []))
    missing = 0
    for i in range(len(e_embeddings)):
        key = (str(e_img_ids[i]), str(e_variant[i]))
        if key not in base_lookup or key not in drift_lookup or key not in liqe_lookup:
            missing += 1
            continue
        raw_signals = np.concatenate([base_lookup[key], drift_lookup[key], liqe_lookup[key]])
        norm_signals = apply_normalization(raw_signals, mean, std).astype(np.float32)
        v = key[1]
        by_variant[v][0].append(e_embeddings[i])
        by_variant[v][1].append(norm_signals)
        by_variant[v][2].append(int(e_labels[i]))
        by_variant[v][3].append(str(e_source[i]))
        by_variant[v][4].append(str(e_family[i]))
        by_variant[v][5].append(key[0])
    if missing:
        print(f"  WARNING: {missing} embedding row(s) had no matching signals AND/OR "
              f"drift AND/OR liqe row -- dropped.")
    return {v: (np.stack(e), np.stack(s), np.array(l), np.array(src), np.array(fam), np.array(iid))
            for v, (e, s, l, src, fam, iid) in by_variant.items()}


def evaluate_variant(model, embeddings: np.ndarray, signals: np.ndarray, labels: np.ndarray, device: str) -> tuple:
    model.eval()
    with torch.no_grad():
        emb_t = torch.tensor(embeddings, dtype=torch.float32, device=device)
        sig_t = torch.tensor(signals, dtype=torch.float32, device=device)
        probs = torch.sigmoid(model(emb_t, sig_t)).cpu().numpy()
    preds = (probs > 0.5).astype(int)
    acc = float((preds == labels).mean())
    auc = float(roc_auc_score(labels, probs))
    return acc, auc, probs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embeddings-root", default="data/cache/embeddings")
    ap.add_argument("--stats-root", default="data/cache/stats")
    ap.add_argument("--checkpoint", default="checkpoints/model_film_drift_liqe_normalized.pt")
    ap.add_argument("--backbone-dim", type=int, default=512, help="512 for ViT-B/32, 768 for ViT-L/14")
    ap.add_argument("--cache-root", default="data/cache/clean",
                     help="Only used to reconstruct the `path` column in the FP/FN CSVs -- "
                          "must match extract_embeddings.py's --cache-root for those paths to resolve.")
    ap.add_argument("--errors-dir", default="results/errors",
                     help="Where film_drift_liqe_fp.csv / film_drift_liqe_fn.csv (every variant, threshold 0.5) get written.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}, signal_dim: {SIGNAL_DIM} ({', '.join(ALL_SIGNAL_COLUMNS)}), normalized: True")

    model = FiLMClassifier(clip_dim=args.backbone_dim, signal_dim=SIGNAL_DIM).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    emb_root, stats_root = Path(args.embeddings_root), Path(args.stats_root)
    base_mean, base_std = compute_train_stats(stats_root / "train_signals.npz", SIGNAL_COLUMNS)
    drift_mean, drift_std = compute_train_stats(stats_root / "train_drift.npz", DRIFT_COLUMNS)
    liqe_mean, liqe_std = compute_train_stats(stats_root / "train_liqe.npz", LIQE_COLUMNS)
    mean = np.concatenate([base_mean, drift_mean, liqe_mean])
    std = np.concatenate([base_std, drift_std, liqe_std])
    print(f"  signal mean (train): {mean}")
    print(f"  signal std  (train): {std}")
    by_variant = load_by_variant(
        emb_root / "test.npz", stats_root / "test_signals.npz",
        stats_root / "test_drift.npz", stats_root / "test_liqe.npz", mean, std)
    variant_order = ["clean"] + sorted(v for v in by_variant if v != "clean")

    print(f"\n{'variant':<14} {'n':>7} {'accuracy':>10} {'auc':>10}")
    print("-" * 44)

    results = {}
    all_probs, all_labels = [], []
    fpr_groups = {}
    fp_all, fn_all = [], []
    clean_source = clean_family = clean_probs_for_slices = clean_labels_for_slices = None
    for v in variant_order:
        embeddings, signals, labels, source_ds, gen_fam, img_ids = by_variant[v]
        acc, auc, probs = evaluate_variant(model, embeddings, signals, labels, device)
        results[v] = (acc, auc)
        print(f"{v:<14} {len(labels):>7} {acc:>10.4f} {auc:>10.4f}")
        all_probs.append(probs)
        all_labels.append(labels)
        fpr_groups[v] = (probs, labels)
        fp_rows, fn_rows = collect_errors(probs, labels, img_ids, v, source_ds, gen_fam, args.cache_root)
        fp_all.extend(fp_rows)
        fn_all.extend(fn_rows)
        if v == "clean":
            clean_source, clean_family = source_ds, gen_fam
            clean_probs_for_slices, clean_labels_for_slices = probs, labels

    print("-" * 44)
    overall_probs = np.concatenate(all_probs)
    overall_labels = np.concatenate(all_labels)
    overall_preds = (overall_probs > 0.5).astype(int)
    overall_acc = float((overall_preds == overall_labels).mean())
    overall_auc = float(roc_auc_score(overall_labels, overall_probs))
    print(f"{'ALL (16)':<14} {len(overall_labels):>7} {overall_acc:>10.4f} {overall_auc:>10.4f}")
    fpr_groups["ALL (16)"] = (overall_probs, overall_labels)

    clean_acc, clean_auc = results["clean"]
    transformed = [v for v in variant_order if v != "clean"]
    mean_trans_acc = sum(results[v][0] for v in transformed) / len(transformed)
    mean_trans_auc = sum(results[v][1] for v in transformed) / len(transformed)
    print(f"\nRobustness summary:")
    print(f"  clean:                 acc {clean_acc:.4f}, auc {clean_auc:.4f}")
    print(f"  mean over 15 variants: acc {mean_trans_acc:.4f}, auc {mean_trans_auc:.4f}")
    print(f"  gap (clean - mean):    acc {clean_acc - mean_trans_acc:+.4f}, auc {clean_auc - mean_trans_auc:+.4f}")

    print_fpr_table("False positive rate by threshold (per variant):", fpr_groups)

    source_groups = {
        s: (clean_probs_for_slices[clean_source == s], clean_labels_for_slices[clean_source == s])
        for s in sorted(set(clean_source))
    }
    print_fpr_table("By source_dataset (clean images only):", source_groups)

    family_groups = {
        g: (clean_probs_for_slices[clean_family == g], clean_labels_for_slices[clean_family == g])
        for g in sorted(set(clean_family))
    }
    print_fpr_table("By generator_family (clean images only):", family_groups)

    errors_dir = Path(args.errors_dir)
    write_error_csv(errors_dir / "film_drift_liqe_fp.csv", fp_all)
    write_error_csv(errors_dir / "film_drift_liqe_fn.csv", fn_all)
    print(f"\nWrote {len(fp_all)} false positive(s) to {errors_dir / 'film_drift_liqe_fp.csv'}")
    print(f"Wrote {len(fn_all)} false negative(s) to {errors_dir / 'film_drift_liqe_fn.csv'}")


if __name__ == "__main__":
    main()
