"""
Identical to train_concat_normalize_drift.py (DCPT + ConcatClassifier, 4 z-scored
B-signals + clip_drift) -- this is a separately-named copy specifically for epoch-count
experimentation. --epochs already existed as a flag on the original script; this file
exists so epoch sweeps produce a distinctly-named checkpoint (checkpoints/final_model.pt
by default) instead of overwriting the already-settled-on model_concat_drift_normalized.pt
baseline every run.

Not wired into collect_results.py -- unlike the fixed JOBS list there, epoch count here
is meant to be chosen per run via --epochs, not pinned to one default for an automated
sweep.

Usage:
    python scripts/training/concat_drift_epoch_optimization.py --epochs 15
    python scripts/training/concat_drift_epoch_optimization.py --epochs 20 --backbone-dim 768
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from normalize import apply_normalization, compute_train_stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from model_concat import ConcatClassifier  # noqa: E402

SIGNAL_COLUMNS = ["laplacian_var", "dct_low_energy", "dct_high_energy", "noise_variance"]
DRIFT_COLUMNS = ["clip_drift"]
ALL_SIGNAL_COLUMNS = SIGNAL_COLUMNS + DRIFT_COLUMNS
SIGNAL_DIM = len(ALL_SIGNAL_COLUMNS)


def _build_lookup(stats_data, columns: list) -> dict:
    """(img_id, variant) -> raw (unnormalized) column vector, for one stats npz."""
    img_ids, variant = stats_data["img_ids"], stats_data["variant"]
    matrix = np.stack([stats_data[col] for col in columns], axis=1).astype(np.float32)
    return {(str(img_ids[i]), str(variant[i])): matrix[i] for i in range(len(img_ids))}


def load_merged_split(
    embeddings_path: Path, base_stats_path: Path, drift_stats_path: Path,
    mean: np.ndarray, std: np.ndarray,
) -> dict:
    """Returns img_id -> {variant_name: (embedding, NORMALIZED 5-dim signals_array, label)}.

    Merges THREE sources by (img_id, variant): the CLIP embeddings, the 4 B-signals
    (base_stats_path), and clip_drift (drift_stats_path) -- concatenated into one raw
    5-vector per row, THEN normalized in one shot with the 5-dim mean/std (concatenated
    in main() from two separate compute_train_stats() calls, in ALL_SIGNAL_COLUMNS
    order). A row missing from EITHER stats file is dropped and counted, not guessed at.

    All arrays are pulled out of every NpzFile ONCE, up front -- see
    train_concat_normalize.py's load_merged_split() for why indexing data[key][i] in a
    loop is a correctness/memory bug, not just a style choice."""
    emb_data = np.load(embeddings_path, allow_pickle=True)
    base_data = np.load(base_stats_path, allow_pickle=True)
    drift_data = np.load(drift_stats_path, allow_pickle=True)

    base_lookup = _build_lookup(base_data, SIGNAL_COLUMNS)
    drift_lookup = _build_lookup(drift_data, DRIFT_COLUMNS)

    e_embeddings, e_img_ids, e_variant, e_labels = (
        emb_data["embeddings"], emb_data["img_ids"], emb_data["variant"], emb_data["labels"]
    )
    by_img: dict = defaultdict(dict)
    missing = 0
    for i in range(len(e_embeddings)):
        img_id = str(e_img_ids[i])
        variant = str(e_variant[i])
        key = (img_id, variant)
        if key not in base_lookup or key not in drift_lookup:
            missing += 1
            continue
        raw_signals = np.concatenate([base_lookup[key], drift_lookup[key]])
        norm_signals = apply_normalization(raw_signals, mean, std)
        by_img[img_id][variant] = (
            e_embeddings[i],
            norm_signals.astype(np.float32),
            int(e_labels[i]),
        )
    if missing:
        print(f"  WARNING: {missing} embedding row(s) had no matching signals AND/OR "
              f"drift row -- dropped. Check clip_drift.py ran on the same cache/embeddings.")
    return by_img


def sample_pair(by_img: dict, img_id: str) -> tuple:
    variants = by_img[img_id]
    clean_emb, clean_sig, label = variants["clean"]
    other_names = [v for v in variants if v != "clean"]
    chosen = random.choice(other_names)
    trans_emb, trans_sig, _ = variants[chosen]
    return clean_emb, clean_sig, trans_emb, trans_sig, label


def symmetric_kl(logit_a: torch.Tensor, logit_b: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    p_a = torch.sigmoid(logit_a).clamp(eps, 1 - eps)
    p_b = torch.sigmoid(logit_b).clamp(eps, 1 - eps)
    kl_ab = p_a * torch.log(p_a / p_b) + (1 - p_a) * torch.log((1 - p_a) / (1 - p_b))
    kl_ba = p_b * torch.log(p_b / p_a) + (1 - p_b) * torch.log((1 - p_b) / (1 - p_a))
    return ((kl_ab + kl_ba) / 2).mean()


def consistency_weight(step: int, total_steps: int, max_weight: float, ramp_fraction: float = 0.25) -> float:
    ramp_steps = max(int(total_steps * ramp_fraction), 1)
    return min(step / ramp_steps, 1.0) * max_weight


def evaluate(model: torch.nn.Module, by_img: dict, device: str) -> tuple:
    """Accuracy AND AUC on each image's clean version. AUC is computed on the raw
    sigmoid probability, not the thresholded prediction, since it measures ranking
    quality across every threshold rather than just the fixed 0.5 cutoff accuracy
    uses."""
    model.eval()
    correct, total = 0, 0
    all_probs, all_labels = [], []
    with torch.no_grad():
        for variants in by_img.values():
            emb, sig, label = variants["clean"]
            emb_t = torch.tensor(emb, dtype=torch.float32, device=device).unsqueeze(0)
            sig_t = torch.tensor(sig, dtype=torch.float32, device=device).unsqueeze(0)
            logit = model(emb_t, sig_t)
            prob = torch.sigmoid(logit).item()
            pred = int(prob > 0.5)
            correct += int(pred == label)
            total += 1
            all_probs.append(prob)
            all_labels.append(label)
    model.train()
    acc = correct / total if total else 0.0
    auc = roc_auc_score(all_labels, all_probs) if total else 0.0
    return acc, auc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embeddings-root", default="data/cache/embeddings")
    ap.add_argument("--stats-root", default="data/cache/stats")
    ap.add_argument("--backbone-dim", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=10, help="Pick this per run -- that's the whole point of this script.")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--consistency-max-weight", type=float, default=1.0)
    ap.add_argument("--out", default="checkpoints/final_model.pt")
    ap.add_argument("--seed", type=int, default=42,
                     help="Seeds random/numpy/torch before model init and DCPT's random "
                          "per-step transform pick, so re-runs are reproducible.")
    ap.add_argument("--results-dir", default="Epoch_Result",
                     help="Where this run's JSON record gets written -- one file per run, "
                          "named train_epoch<N>_<UTC timestamp>.json so repeated epoch "
                          "sweeps never overwrite each other's results.")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}, signal_dim: {SIGNAL_DIM} ({', '.join(ALL_SIGNAL_COLUMNS)}), "
          f"normalized: True, seed: {args.seed}, epochs: {args.epochs}")

    emb_root, stats_root = Path(args.embeddings_root), Path(args.stats_root)

    base_mean, base_std = compute_train_stats(stats_root / "train_signals.npz", SIGNAL_COLUMNS)
    drift_mean, drift_std = compute_train_stats(stats_root / "train_drift.npz", DRIFT_COLUMNS)
    mean = np.concatenate([base_mean, drift_mean])
    std = np.concatenate([base_std, drift_std])
    print(f"  signal mean (train): {mean}")
    print(f"  signal std  (train): {std}")

    train_by_img = load_merged_split(
        emb_root / "train.npz", stats_root / "train_signals.npz", stats_root / "train_drift.npz", mean, std)
    val_by_img = load_merged_split(
        emb_root / "val.npz", stats_root / "val_signals.npz", stats_root / "val_drift.npz", mean, std)
    train_ids = list(train_by_img.keys())
    print(f"train images: {len(train_ids)}, val images: {len(val_by_img)}")

    model = ConcatClassifier(clip_dim=args.backbone_dim, signal_dim=SIGNAL_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    steps_per_epoch = max(len(train_ids) // args.batch_size, 1)
    total_steps = steps_per_epoch * args.epochs
    step = 0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    epoch_history = []
    for epoch in range(args.epochs):
        random.shuffle(train_ids)
        epoch_loss = 0.0
        for i in tqdm(range(0, len(train_ids), args.batch_size), desc=f"epoch {epoch + 1}/{args.epochs}"):
            batch_ids = train_ids[i : i + args.batch_size]
            clean_e, clean_s, trans_e, trans_s, labels = [], [], [], [], []
            for img_id in batch_ids:
                ce, cs, te, ts, l = sample_pair(train_by_img, img_id)
                clean_e.append(ce); clean_s.append(cs)
                trans_e.append(te); trans_s.append(ts)
                labels.append(l)

            clean_et = torch.tensor(np.stack(clean_e), dtype=torch.float32, device=device)
            clean_st = torch.tensor(np.stack(clean_s), dtype=torch.float32, device=device)
            trans_et = torch.tensor(np.stack(trans_e), dtype=torch.float32, device=device)
            trans_st = torch.tensor(np.stack(trans_s), dtype=torch.float32, device=device)
            labels_t = torch.tensor(labels, dtype=torch.float32, device=device)

            clean_logit = model(clean_et, clean_st)
            trans_logit = model(trans_et, trans_st)

            loss_clean = F.binary_cross_entropy_with_logits(clean_logit, labels_t)
            loss_trans = F.binary_cross_entropy_with_logits(trans_logit, labels_t)
            loss_consistency = symmetric_kl(clean_logit, trans_logit)
            w = consistency_weight(step, total_steps, args.consistency_max_weight)
            loss = loss_clean + loss_trans + w * loss_consistency

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            step += 1

        val_acc, val_auc = evaluate(model, val_by_img, device)
        avg_loss = epoch_loss / steps_per_epoch
        print(f"epoch {epoch + 1}: avg loss {avg_loss:.4f}, "
              f"val acc {val_acc:.4f}, val auc {val_auc:.4f}")
        epoch_history.append({"epoch": epoch + 1, "avg_loss": avg_loss, "val_acc": val_acc, "val_auc": val_auc})

    torch.save(model.state_dict(), args.out)
    print(f"\nSaved trained model to {args.out}")

    backbone_label = {512: "ViT-B/32", 768: "ViT-L/14"}.get(args.backbone_dim, f"dim={args.backbone_dim}")
    record = {
        "run_name": "concat_drift_epoch_optimization",
        "script": "training/concat_drift_epoch_optimization.py",
        "kind": "train",
        "model": "final_model",
        "signals": ALL_SIGNAL_COLUMNS,
        "signals_normalized": True,
        "backbone": backbone_label,
        "backbone_dim": args.backbone_dim,
        "seed": args.seed,
        "epochs": args.epochs,
        "device": device,
        "checkpoint": str(args.out),
        "results": {"epochs": epoch_history, "final": epoch_history[-1] if epoch_history else None},
    }
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = results_dir / f"train_epoch{args.epochs}_{ts}.json"
    out_path.write_text(json.dumps(record, indent=2))
    print(f"Saved run record to {out_path}")


if __name__ == "__main__":
    main()
