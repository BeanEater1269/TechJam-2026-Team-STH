"""
Evaluates a trained ConcatClassifier (from train_concat_normalize.py ONLY -- the
un-normalized train_concat.py/model_concat.pt path has been retired) on the TEST split,
broken out per variant. Reports accuracy + AUC for each of the 16 variants (clean + 15
robustness transforms) individually, so it's directly comparable to evaluate_base.py's
table -- same test images, same metrics, only the model differs.

Signals are ALWAYS z-scored before evaluation, using TRAIN split stats (via
normalize.py) -- this is not optional, since the only checkpoint this script supports
was trained on normalized signals and would get garbage predictions on raw ones.

Also reports:
  - False positive rate at 3 thresholds (0.3 / 0.5 / 0.7), per variant.
  - Accuracy/AUC/FPR broken out by source_dataset and by generator_family, on the clean
    images only -- per dataset-plan.md's "Known residual risks" table.

Merges data/cache/embeddings/test.npz with data/cache/stats/test_signals.npz by matching
(img_id, variant) as keys -- NOT by row position, same reasoning as
train_concat_normalize.py.

Run this AFTER train_concat_normalize.py has produced a checkpoint. Touches test.npz,
which nothing else in the pipeline reads -- meant to be run once, at the end.

Usage:
    python scripts/evaluate_concat.py
    python scripts/evaluate_concat.py --checkpoint checkpoints/model_concat_normalized.pt
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from eval_metrics import print_fpr_table  # noqa: E402
from normalize import apply_normalization, compute_train_stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from model_concat import ConcatClassifier  # noqa: E402

SIGNAL_COLUMNS = ["laplacian_var", "dct_low_energy", "dct_high_energy", "noise_variance"]
SIGNAL_DIM = len(SIGNAL_COLUMNS)


def load_by_variant(embeddings_path: Path, stats_path: Path, mean: np.ndarray, std: np.ndarray) -> dict:
    """Merges embeddings + signals by (img_id, variant), then groups by variant_name ->
    (embeddings_array, signals_array, labels_array, source_dataset_array,
    generator_family_array). Unlike train_concat_normalize.py's load_merged_split()
    (grouped by img_id, for DCPT pairing), this groups by variant, since the
    robustness table needs "every jpeg_q30 row" as one batch, not paired.
    source_dataset/generator_family come from the embeddings file (test_signals.npz
    has no such columns -- it's pure pixel measurements, no metadata) and are carried
    through unused for most variants -- only the "clean" group's copies get used, for
    the confound-slice breakdown.

    Arrays are pulled out of both NpzFiles ONCE, up front -- see
    train_concat_normalize.py's load_merged_split() for why indexing data[key][i] in a
    loop is a correctness/memory bug (re-reads the whole array from disk on every
    access), not just a style choice.

    mean/std are always the TRAIN split's (from normalize.py) -- signals are z-scored
    with that exact transform before being returned, unconditionally, since this
    script only ever evaluates a checkpoint that was trained on normalized signals."""
    emb_data = np.load(embeddings_path, allow_pickle=True)
    stats_data = np.load(stats_path, allow_pickle=True)

    s_img_ids, s_variant = stats_data["img_ids"], stats_data["variant"]
    s_matrix = np.stack([stats_data[col] for col in SIGNAL_COLUMNS], axis=1).astype(np.float32)
    s_matrix = apply_normalization(s_matrix, mean, std)

    stats_lookup = {}
    for i in range(len(s_img_ids)):
        key = (str(s_img_ids[i]), str(s_variant[i]))
        stats_lookup[key] = s_matrix[i]

    e_embeddings, e_img_ids, e_variant, e_labels = (
        emb_data["embeddings"], emb_data["img_ids"], emb_data["variant"], emb_data["labels"]
    )
    e_source, e_family = emb_data["source_dataset"], emb_data["generator_family"]
    by_variant: dict = defaultdict(lambda: ([], [], [], [], []))
    missing = 0
    for i in range(len(e_embeddings)):
        key = (str(e_img_ids[i]), str(e_variant[i]))
        if key not in stats_lookup:
            missing += 1
            continue
        v = key[1]
        by_variant[v][0].append(e_embeddings[i])
        by_variant[v][1].append(stats_lookup[key])
        by_variant[v][2].append(int(e_labels[i]))
        by_variant[v][3].append(str(e_source[i]))
        by_variant[v][4].append(str(e_family[i]))
    if missing:
        print(f"  WARNING: {missing} embedding row(s) had no matching stats row -- dropped.")
    return {v: (np.stack(e), np.stack(s), np.array(l), np.array(src), np.array(fam))
            for v, (e, s, l, src, fam) in by_variant.items()}


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
    ap.add_argument("--checkpoint", default="checkpoints/model_concat_normalized.pt")
    ap.add_argument("--backbone-dim", type=int, default=512, help="512 for ViT-B/32, 768 for ViT-L/14")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}, signal_dim: {SIGNAL_DIM} ({', '.join(SIGNAL_COLUMNS)}), normalized: True")

    model = ConcatClassifier(clip_dim=args.backbone_dim, signal_dim=SIGNAL_DIM).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    emb_root, stats_root = Path(args.embeddings_root), Path(args.stats_root)
    mean, std = compute_train_stats(stats_root / "train_signals.npz", SIGNAL_COLUMNS)
    print(f"  signal mean (train): {mean}")
    print(f"  signal std  (train): {std}")
    by_variant = load_by_variant(emb_root / "test.npz", stats_root / "test_signals.npz", mean, std)
    variant_order = ["clean"] + sorted(v for v in by_variant if v != "clean")

    print(f"\n{'variant':<14} {'n':>7} {'accuracy':>10} {'auc':>10}")
    print("-" * 44)

    results = {}
    all_probs, all_labels = [], []
    fpr_groups = {}
    clean_source = clean_family = clean_probs_for_slices = clean_labels_for_slices = None
    for v in variant_order:
        embeddings, signals, labels, source_ds, gen_fam = by_variant[v]
        acc, auc, probs = evaluate_variant(model, embeddings, signals, labels, device)
        results[v] = (acc, auc)
        print(f"{v:<14} {len(labels):>7} {acc:>10.4f} {auc:>10.4f}")
        all_probs.append(probs)
        all_labels.append(labels)
        fpr_groups[v] = (probs, labels)
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


if __name__ == "__main__":
    main()
