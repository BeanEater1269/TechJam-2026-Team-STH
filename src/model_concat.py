"""
Concatenation classifier. Owned by: Jia Tai.

CLIP embedding and B-signals get stuck together end to end, fed through a small MLP.

Gracefully supports signal_dim=0 (or signals=None at call time) -- in that case this is
just a plain CLIP-embedding classifier, no signals involved at all. That's the intended
"base gauge" path: get DCPT working end to end with this file alone, before signals.py
exists. Once signals.py is real, pass signal_dim=<however many numbers it outputs> and
start passing real signals at call time -- nothing else about this file needs to change.
"""
import torch
import torch.nn as nn


class ConcatClassifier(nn.Module):
    def __init__(self, clip_dim: int, signal_dim: int = 0, hidden_dim: int = 128):
        super().__init__()
        self.signal_dim = signal_dim
        self.net = nn.Sequential(
            nn.Linear(clip_dim + signal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, clip_embedding: torch.Tensor, signals: torch.Tensor | None = None) -> torch.Tensor:
        if self.signal_dim > 0:
            if signals is None:
                raise ValueError(f"signal_dim={self.signal_dim} but no signals were passed")
            x = torch.cat([clip_embedding, signals], dim=-1)
        else:
            x = clip_embedding  # the "base gauge" path -- no signals to concatenate at all
        return self.net(x).squeeze(-1)  # raw logit -- apply sigmoid outside for a probability
