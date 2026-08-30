"""
Trains the classifier using DCPT (Degradation-Consistent Paired Training).

For each step: pick clean + one random transformed variant of the SAME image, run both
through the same classifier, compute a 3-part loss (clean-vs-label, transformed-vs-label,
and a symmetric-KL consistency term between the two predictions), backprop.

This version runs WITHOUT signals.py -- uses model_concat.py with signal_dim=0, which is
the "base gauge" path: a plain CLIP-embedding classifier, no signals involved. Get DCPT
working end to end here first. Once signals.py exists, this becomes a small edit (build
the classifier with signal_dim=<real number>, compute+pass real signals per batch) --
nothing about the training loop itself needs to change.

Usage:
    python train.py
    python train.py --epochs 10 --backbone-dim 768
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

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from base_classifier import BaseClassifier  # noqa: E402


def group_by_image(npz_path: Path) -> dict:
    """Loads one split's npz, groups rows by img_id -> {variant_name: (embedding, label)}.
    This is the "pairing" reconstruction -- nothing pre-paired in the file itself, just
    img_id + variant columns sitting next to each embedding.

    Arrays are pulled out of the NpzFile ONCE, up front -- data[key] re-reads the whole
    array fresh from the zip archive on every access, so indexing data[key][i] inside the
    loop was re-loading the entire (multi-hundred-MB) array on every single row instead of
    once, which is both why this was catastrophically slow and why it ran out of memory."""
    data = np.load(npz_path, allow_pickle=True)
    embeddings, img_ids, variants, labels = (
        data["embeddings"], data["img_ids"], data["variant"], data["labels"]
    )
    by_img = defaultdict(dict)
    for i in range(len(embeddings)):
        by_img[str(img_ids[i])][str(variants[i])] = (embeddings[i], int(labels[i]))
    return by_img


def sample_pair(by_img: dict, img_id: str) -> tuple:
    """clean is always one half; the other is picked fresh from the remaining 15 each call."""
    variants = by_img[img_id]
    clean_emb, label = variants["clean"]
    other_names = [v for v in variants if v != "clean"]
    chosen = random.choice(other_names)
    transformed_emb, _ = variants[chosen]  # same photo, same label as clean
    return clean_emb, transformed_emb, label


def symmetric_kl(logit_a: torch.Tensor, logit_b: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """The actual DCPT consistency term -- symmetric KL between the two Bernoulli
    (real/fake) distributions implied by each logit. Penalizes the model for answering
    differently on clean vs. transformed, regardless of whether either answer is correct."""
    p_a = torch.sigmoid(logit_a).clamp(eps, 1 - eps)
    p_b = torch.sigmoid(logit_b).clamp(eps, 1 - eps)
    kl_ab = p_a * torch.log(p_a / p_b) + (1 - p_a) * torch.log((1 - p_a) / (1 - p_b))
    kl_ba = p_b * torch.log(p_b / p_a) + (1 - p_b) * torch.log((1 - p_b) / (1 - p_a))
    return ((kl_ab + kl_ba) / 2).mean()


def consistency_weight(step: int, total_steps: int, max_weight: float, ramp_fraction: float = 0.25) -> float:
    """Ramp-up: near zero at the start, full weight after the first ~quarter of training.
    Without this, early training can cheaply "satisfy" the consistency term by having both
    branches agree on a shared WRONG answer, before the classifier's learned anything real."""
    ramp_steps = max(int(total_steps * ramp_fraction), 1)
    return min(step / ramp_steps, 1.0) * max_weight


def evaluate(model: torch.nn.Module, by_img: dict, device: str) -> tuple[float, float]:
    """Accuracy AND AUC on each image's clean version -- a quick per-epoch val check,
    not the full robustness table (that's a separate, later script). AUC is computed
    on the raw sigmoid probability, not the thresholded prediction, since it measures
    ranking quality across every threshold rather than just the fixed 0.5 cutoff
    accuracy uses."""
    model.eval()
    correct, total = 0, 0
    all_probs, all_labels = [], []
    with torch.no_grad():
        for variants in by_img.values():
            emb, label = variants["clean"]
            emb_t = torch.tensor(emb, dtype=torch.float32, device=device).unsqueeze(0)
            logit = model(emb_t)
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
    ap.add_argument("--backbone-dim", type=int, default=512, help="512 for ViT-B/32, 768 for ViT-L/14")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--consistency-max-weight", type=float, default=1.0)
    ap.add_argument("--out", default="checkpoints/base_classifier.pt")
    ap.add_argument("--seed", type=int, default=42,
                     help="Seeds random/numpy/torch before model init and DCPT's random "
                          "per-step transform pick, so re-runs are reproducible.")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}, seed: {args.seed}")

    root = Path(args.embeddings_root)
    train_by_img = group_by_image(root / "train.npz")
    val_by_img = group_by_image(root / "val.npz")
    train_ids = list(train_by_img.keys())
    print(f"train images: {len(train_ids)}, val images: {len(val_by_img)}")

    model = BaseClassifier(clip_dim=args.backbone_dim).to(device)
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
            clean_embs, trans_embs, labels = [], [], []
            for img_id in batch_ids:
                c, t, l = sample_pair(train_by_img, img_id)
                clean_embs.append(c)
                trans_embs.append(t)
                labels.append(l)

            clean_t = torch.tensor(np.stack(clean_embs), dtype=torch.float32, device=device)
            trans_t = torch.tensor(np.stack(trans_embs), dtype=torch.float32, device=device)
            labels_t = torch.tensor(labels, dtype=torch.float32, device=device)

            clean_logit = model(clean_t)
            trans_logit = model(trans_t)

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
