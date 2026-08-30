"""
Evaluates a trained BaseClassifier on the TEST split, broken out per variant -- this is
the robustness table train_base.py's own evaluate() docstring deferred to "a separate,
later script." Reports accuracy + AUC for each of the 16 variants (clean + 15 robustness
transforms) individually, not just clean, since the whole point of DCPT is measuring how
much performance holds up under degradation, not just on pristine images.

Also reports:
  - False positive rate at 3 thresholds (0.3 / 0.5 / 0.7), per variant -- one operating
    point isn't enough to judge a detector by.
  - Accuracy/AUC/FPR broken out by source_dataset and by generator_family, on the clean
    images only -- per dataset-plan.md's "Known residual risks" table (checks whether
    e.g. CIFAKE's 32->680 upsample, or a specific generator, produces suspiciously
    high/low accuracy -- a sign the model latched onto a resolution/generator shortcut
    instead of real-vs-fake content).

Run this AFTER train_base.py has produced a checkpoint. Touches test.npz, which nothing
else in the pipeline reads -- this is meant to be run once, at the end.

Usage:
    python scripts/evaluate_base.py
    python scripts/evaluate_base.py --checkpoint checkpoints/base_classifier.pt
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

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from base_classifier import BaseClassifier  # noqa: E402


def load_by_variant(npz_path: Path) -> dict:
    """Groups rows by variant_name -> (embeddings_array, labels_array,
    source_dataset_array, generator_family_array, img_ids_array), across ALL images --
    unlike train_base.py's group_by_image() (which groups by img_id, for DCPT pairing),
    this groups by variant, since the robustness table needs "every jpeg_q30 row" as one
    batch, not paired with anything. source_dataset/generator_family/img_ids are carried
    through for every variant (not just clean) so the FP/FN CSV dump can name the exact
    example for any variant, even though the confound-slice breakdown further down only
    uses the "clean" group's copies.

    Arrays are pulled out of the NpzFile ONCE, up front -- see group_by_image() in
    train_base.py for why indexing data[key][i] in a loop is a correctness/memory bug
    (re-reads the whole array from disk on every access), not just a style choice."""
    data = np.load(npz_path, allow_pickle=True)
    embeddings, variants, labels = data["embeddings"], data["variant"], data["labels"]
    source_dataset, generator_family = data["source_dataset"], data["generator_family"]
    img_ids = data["img_ids"]

    by_variant: dict = defaultdict(lambda: ([], [], [], [], []))
    for i in range(len(embeddings)):
        v = str(variants[i])
        by_variant[v][0].append(embeddings[i])
        by_variant[v][1].append(int(labels[i]))
        by_variant[v][2].append(str(source_dataset[i]))
        by_variant[v][3].append(str(generator_family[i]))
        by_variant[v][4].append(str(img_ids[i]))
    return {v: (np.stack(e), np.array(l), np.array(s), np.array(g), np.array(iid))
            for v, (e, l, s, g, iid) in by_variant.items()}


def evaluate_variant(model, embeddings: np.ndarray, labels: np.ndarray, device: str) -> tuple:
    model.eval()
    with torch.no_grad():
        emb_t = torch.tensor(embeddings, dtype=torch.float32, device=device)
        probs = torch.sigmoid(model(emb_t)).cpu().numpy()
    preds = (probs > 0.5).astype(int)
    acc = float((preds == labels).mean())
    auc = float(roc_auc_score(labels, probs))
    return acc, auc, probs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embeddings-root", default="data/cache/embeddings")
    ap.add_argument("--checkpoint", default="checkpoints/base_classifier.pt")
    ap.add_argument("--backbone-dim", type=int, default=512, help="512 for ViT-B/32, 768 for ViT-L/14")
    ap.add_argument("--cache-root", default="data/cache/clean",
                     help="Only used to reconstruct the `path` column in the FP/FN CSVs -- "
                          "must match extract_embeddings.py's --cache-root for those paths to resolve.")
    ap.add_argument("--errors-dir", default="results/errors",
                     help="Where base_fp.csv / base_fn.csv (every variant, threshold 0.5) get written.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    model = BaseClassifier(clip_dim=args.backbone_dim).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    by_variant = load_by_variant(Path(args.embeddings_root) / "test.npz")
    variant_order = ["clean"] + sorted(v for v in by_variant if v != "clean")

    print(f"\n{'variant':<14} {'n':>7} {'accuracy':>10} {'auc':>10}")
    print("-" * 44)

    results = {}
    all_probs, all_labels = [], []
    fpr_groups = {}
    fp_all, fn_all = [], []
    clean_source = clean_family = clean_probs_for_slices = clean_labels_for_slices = None
    for v in variant_order:
        embeddings, labels, source_ds, gen_fam, img_ids = by_variant[v]
        acc, auc, probs = evaluate_variant(model, embeddings, labels, device)
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

    # False positive rate at 3 thresholds, reusing the probs/labels already computed
    # above -- no extra forward pass needed.
    print_fpr_table("False positive rate by threshold (per variant):", fpr_groups)

    # Confound-slice breakdown, clean images only -- reuses clean's already-computed
    # probs, sliced by source_dataset / generator_family instead of re-scored.
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
    write_error_csv(errors_dir / "base_fp.csv", fp_all)
    write_error_csv(errors_dir / "base_fn.csv", fn_all)
    print(f"\nWrote {len(fp_all)} false positive(s) to {errors_dir / 'base_fp.csv'}")
    print(f"Wrote {len(fn_all)} false negative(s) to {errors_dir / 'base_fn.csv'}")


if __name__ == "__main__":
    main()
