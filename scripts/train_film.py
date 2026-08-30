"""
Trains the FiLM classifier using DCPT, WITH real signals -- REQUIRES signals, no zero-signal fallback (use train_base.py for that) -- Laplacian variance, DCT low/high,
noise variance, merged in from classical_degradation_stats.py's output.

This is the file train_base.py deliberately isn't: train_base.py stays as the
zero-signal safety net (still fully working, unchanged). This one is the real
model_concat.py path, once signals actually exist.

Merges data/cache/embeddings/<split>.npz with data/cache/stats/<split>.npz by matching
(img_id, variant) as keys -- NOT by row position. Both files are independently sorted/
produced by different scripts; trusting position here would be the exact same class of
bug as the img_id collision from earlier tonight. Matching by key is slightly more code
and is not something to shortcut.

Usage:
    python scripts/train.py
    python scripts/train.py --epochs 10 --backbone-dim 768
"""
import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from model_film import FiLMClassifier  # noqa: E402

SIGNAL_COLUMNS = ["laplacian_var", "dct_low_energy", "dct_high_energy", "noise_variance"]
SIGNAL_DIM = len(SIGNAL_COLUMNS)


def load_merged_split(embeddings_path: Path, stats_path: Path) -> dict:
    """Returns img_id -> {variant_name: (embedding, signals_array, label)}.

    Merge key is (img_id, variant) -- explicit, not positional. If a row exists in one
    file but not the other, it's dropped and counted, not silently guessed at."""
    emb_data = np.load(embeddings_path, allow_pickle=True)
    stats_data = np.load(stats_path, allow_pickle=True)

    stats_lookup = {}
    for i in range(len(stats_data["img_ids"])):
        key = (str(stats_data["img_ids"][i]), str(stats_data["variant"][i]))
        stats_lookup[key] = np.array([stats_data[col][i] for col in SIGNAL_COLUMNS], dtype=np.float32)

    by_img: dict = defaultdict(dict)
    missing = 0
    for i in range(len(emb_data["embeddings"])):
        img_id = str(emb_data["img_ids"][i])
        variant = str(emb_data["variant"][i])
        key = (img_id, variant)
        if key not in stats_lookup:
            missing += 1
            continue
        by_img[img_id][variant] = (
            emb_data["embeddings"][i],
            stats_lookup[key],
            int(emb_data["labels"][i]),
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


def evaluate(model: torch.nn.Module, by_img: dict, device: str) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for variants in by_img.values():
            emb, sig, label = variants["clean"]
            emb_t = torch.tensor(emb, dtype=torch.float32, device=device).unsqueeze(0)
            sig_t = torch.tensor(sig, dtype=torch.float32, device=device).unsqueeze(0)
            logit = model(emb_t, sig_t)
            pred = int(torch.sigmoid(logit).item() > 0.5)
            correct += int(pred == label)
            total += 1
    model.train()
    return correct / total if total else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embeddings-root", default="data/cache/embeddings")
    ap.add_argument("--stats-root", default="data/cache/stats")
    ap.add_argument("--backbone-dim", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--consistency-max-weight", type=float, default=1.0)
    ap.add_argument("--out", default="checkpoints/model_film.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}, signal_dim: {SIGNAL_DIM} ({', '.join(SIGNAL_COLUMNS)})")

    emb_root, stats_root = Path(args.embeddings_root), Path(args.stats_root)
    train_by_img = load_merged_split(emb_root / "train.npz", stats_root / "train_signals.npz")
    val_by_img = load_merged_split(emb_root / "val.npz", stats_root / "val_signals.npz")
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

        val_acc = evaluate(model, val_by_img, device)
        print(f"epoch {epoch + 1}: avg loss {epoch_loss / steps_per_epoch:.4f}, val acc {val_acc:.4f}")

    torch.save(model.state_dict(), args.out)
    print(f"\nSaved trained model to {args.out}")


if __name__ == "__main__":
    main()
