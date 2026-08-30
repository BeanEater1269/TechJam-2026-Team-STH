"""
Same as train_film.py (DCPT + FiLMClassifier), except the 4 raw B-signals get z-score
normalized before they're used to generate FiLM's scale/shift.

This matters even more for FiLM than for concatenation: FiLM feeds the raw signals
through a Linear layer to produce a MULTIPLICATIVE scale and additive shift applied
directly to the CLIP embedding (modulated = clip_embedding * (1 + scale) + shift). If
the raw signal values are large/unnormalized, that scale/shift can distort the
embedding severely before the MLP ever sees it.

Normalization stats (mean, std) come from normalize.py, computed ONCE from the TRAIN
split's raw signal file, then applied unchanged to train, val, and (in a corresponding
evaluate script) test -- never recomputed per split.

Usage:
    python scripts/train_film_normalize.py
    python scripts/train_film_normalize.py --epochs 10 --backbone-dim 768
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
from model_film import FiLMClassifier  # noqa: E402

SIGNAL_COLUMNS = ["laplacian_var", "dct_low_energy", "dct_high_energy", "noise_variance"]
SIGNAL_DIM = len(SIGNAL_COLUMNS)


def load_merged_split(embeddings_path: Path, stats_path: Path, mean: np.ndarray, std: np.ndarray) -> dict:
    """Returns img_id -> {variant_name: (embedding, NORMALIZED signals_array, label)}.

    Merge key is (img_id, variant) -- explicit, not positional. If a row exists in one
    file but not the other, it's dropped and counted, not silently guessed at.

    All arrays are pulled out of both NpzFiles ONCE, up front -- data[key] re-reads the
    whole array fresh from the zip archive on every access, so indexing data[key][i]
    inside a loop was re-loading the entire array on every single row instead of once
    (catastrophically slow, and the direct cause of the OOM crash on train.npz).

    Signals are normalized vectorized, in one shot, right after loading -- mean/std are
    always the TRAIN split's, passed in from main(), never recomputed here."""
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
    by_img: dict = defaultdict(dict)
    missing = 0
    for i in range(len(e_embeddings)):
        img_id = str(e_img_ids[i])
        variant = str(e_variant[i])
        key = (img_id, variant)
        if key not in stats_lookup:
            missing += 1
            continue
        by_img[img_id][variant] = (
            e_embeddings[i],
            stats_lookup[key],
            int(e_labels[i]),
        )
    if missing:
        print(f"  WARNING: {missing} embedding row(s) had no matching stats row -- "
              f"dropped. Check classical_degradation_stats.py ran on the same cache.")
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
    ap.add_argument("--out", default="checkpoints/model_film_normalized.pt")
    ap.add_argument("--seed", type=int, default=42,
                     help="Seeds random/numpy/torch before model init and DCPT's random "
                          "per-step transform pick, so re-runs are reproducible.")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}, signal_dim: {SIGNAL_DIM} ({', '.join(SIGNAL_COLUMNS)}), normalized: True, seed: {args.seed}")

    emb_root, stats_root = Path(args.embeddings_root), Path(args.stats_root)

    mean, std = compute_train_stats(stats_root / "train_signals.npz", SIGNAL_COLUMNS)
    print(f"  signal mean (train): {mean}")
    print(f"  signal std  (train): {std}")

    train_by_img = load_merged_split(emb_root / "train.npz", stats_root / "train_signals.npz", mean, std)
    val_by_img = load_merged_split(emb_root / "val.npz", stats_root / "val_signals.npz", mean, std)
    train_ids = list(train_by_img.keys())
    print(f"train images: {len(train_ids)}, val images: {len(val_by_img)}")

    model = FiLMClassifier(clip_dim=args.backbone_dim, signal_dim=SIGNAL_DIM).to(device)
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
