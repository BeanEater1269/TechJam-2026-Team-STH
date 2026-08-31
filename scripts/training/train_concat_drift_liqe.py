"""
Same as train_concat_normalize_drift.py (DCPT + ConcatClassifier, 4 z-scored B-signals +
clip_drift), plus a 6th signal: liqe_score (scripts/features/liqe.py's pretrained
image-quality probe, "signal D" -- covers color-jitter degradation, the one thing
nothing else on the signal list touches).

liqe_score lives in its OWN file (data/cache/stats/{split}_liqe.npz), separate from the
4 B-signals' {split}_signals.npz and clip_drift's {split}_drift.npz -- signals.py,
clip_drift.py, and liqe.py are three independent pipelines that happen to share a merge
key. All three files have the same shape (img_ids + variant + score column(s)) -- LIQE
needs no reference embedding to compare against, so (like signals.py, unlike the
clean-only version of clip_drift.py) it naturally scores all 16 variants directly, no
broadcast logic needed here either.

Normalization stats (mean, std) are computed separately per file via normalize.py's
existing compute_train_stats() (unmodified, still one-file-at-a-time) then concatenated
-- avoids touching normalize.py itself for what all three files already do
independently.

Usage:
    python scripts/training/train_concat_drift_liqe.py
    python scripts/training/train_concat_drift_liqe.py --epochs 10 --backbone-dim 768
"""
import argparse
import random
import sys
from collections import defaultdict
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
LIQE_COLUMNS = ["liqe_score"]
ALL_SIGNAL_COLUMNS = SIGNAL_COLUMNS + DRIFT_COLUMNS + LIQE_COLUMNS
SIGNAL_DIM = len(ALL_SIGNAL_COLUMNS)


def _build_lookup(stats_data, columns: list) -> dict:
    """(img_id, variant) -> raw (unnormalized) column vector, for one stats npz. Keys
    are normalized to plain Python str on both sides of every comparison -- img_ids come
    back as numpy string arrays from np.load(), but str(...) makes the exact numpy
    dtype/width irrelevant, so signals.py/clip_drift.py/liqe.py's three independently
    -derived id arrays all compare correctly even if their underlying dtypes differ."""
    img_ids, variant = stats_data["img_ids"], stats_data["variant"]
    matrix = np.stack([stats_data[col] for col in columns], axis=1).astype(np.float32)
    return {(str(img_ids[i]), str(variant[i])): matrix[i] for i in range(len(img_ids))}


def load_merged_split(
    embeddings_path: Path, base_stats_path: Path, drift_stats_path: Path, liqe_stats_path: Path,
    mean: np.ndarray, std: np.ndarray,
) -> dict:
    """Returns img_id -> {variant_name: (embedding, NORMALIZED 6-dim signals_array, label)}.

    Merges FOUR sources by (img_id, variant): the CLIP embeddings, the 4 B-signals
    (base_stats_path), clip_drift (drift_stats_path), and liqe_score (liqe_stats_path)
    -- concatenated into one raw 6-vector per row, THEN normalized in one shot with the
    6-dim mean/std (concatenated in main() from three separate compute_train_stats()
    calls, in ALL_SIGNAL_COLUMNS order). A row missing from ANY of the three stats files
    is dropped and counted, not guessed at -- same reasoning as
    train_concat_normalize_drift.py's load_merged_split().

    All arrays are pulled out of every NpzFile ONCE, up front -- see
    train_concat_normalize.py's load_merged_split() for why indexing data[key][i] in a
    loop is a correctness/memory bug, not just a style choice."""
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
    by_img: dict = defaultdict(dict)
    missing = 0
    for i in range(len(e_embeddings)):
        img_id = str(e_img_ids[i])
        variant = str(e_variant[i])
        key = (img_id, variant)
        if key not in base_lookup or key not in drift_lookup or key not in liqe_lookup:
            missing += 1
            continue
        raw_signals = np.concatenate([base_lookup[key], drift_lookup[key], liqe_lookup[key]])
        norm_signals = apply_normalization(raw_signals, mean, std)
        by_img[img_id][variant] = (
            e_embeddings[i],
            norm_signals.astype(np.float32),
            int(e_labels[i]),
        )
    if missing:
        print(f"  WARNING: {missing} embedding row(s) had no matching signals AND/OR "
              f"drift AND/OR liqe row -- dropped. Check clip_drift.py and liqe.py both "
              f"ran on the same cache/embeddings.")
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
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--consistency-max-weight", type=float, default=1.0)
    ap.add_argument("--out", default="checkpoints/model_concat_drift_liqe_normalized.pt")
    ap.add_argument("--seed", type=int, default=42,
                     help="Seeds random/numpy/torch before model init and DCPT's random "
                          "per-step transform pick, so re-runs are reproducible.")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}, signal_dim: {SIGNAL_DIM} ({', '.join(ALL_SIGNAL_COLUMNS)}), "
          f"normalized: True, seed: {args.seed}")

    emb_root, stats_root = Path(args.embeddings_root), Path(args.stats_root)

    base_mean, base_std = compute_train_stats(stats_root / "train_signals.npz", SIGNAL_COLUMNS)
    drift_mean, drift_std = compute_train_stats(stats_root / "train_drift.npz", DRIFT_COLUMNS)
    liqe_mean, liqe_std = compute_train_stats(stats_root / "train_liqe.npz", LIQE_COLUMNS)
    mean = np.concatenate([base_mean, drift_mean, liqe_mean])
    std = np.concatenate([base_std, drift_std, liqe_std])
    print(f"  signal mean (train): {mean}")
    print(f"  signal std  (train): {std}")

    train_by_img = load_merged_split(
        emb_root / "train.npz", stats_root / "train_signals.npz",
        stats_root / "train_drift.npz", stats_root / "train_liqe.npz", mean, std)
    val_by_img = load_merged_split(
        emb_root / "val.npz", stats_root / "val_signals.npz",
        stats_root / "val_drift.npz", stats_root / "val_liqe.npz", mean, std)
    train_ids = list(train_by_img.keys())
    print(f"train images: {len(train_ids)}, val images: {len(val_by_img)}")

    model = ConcatClassifier(clip_dim=args.backbone_dim, signal_dim=SIGNAL_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    steps_per_epoch = max(len(train_ids) // args.batch_size, 1)
    total_steps = steps_per_epoch * args.epochs
    step = 0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

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
        print(f"epoch {epoch + 1}: avg loss {epoch_loss / steps_per_epoch:.4f}, "
              f"val acc {val_acc:.4f}, val auc {val_auc:.4f}")

    torch.save(model.state_dict(), args.out)
    print(f"\nSaved trained model to {args.out}")


if __name__ == "__main__":
    main()
