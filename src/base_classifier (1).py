"""
The bare classifier -- no FiLM, no concatenation, no signals. Just the CLIP embedding
straight into a small MLP. This IS the base gauge.

This is functionally identical to model_concat.py called with signal_dim=0 -- same math,
same output. This file exists so there's a name that can't be mistaken for "doing
concatenation" -- BaseClassifier never even has a signals argument to be confused about.
"""
import torch
import torch.nn as nn


class BaseClassifier(nn.Module):
    def __init__(self, clip_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(clip_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, clip_embedding: torch.Tensor) -> torch.Tensor:
        return self.net(clip_embedding).squeeze(-1)  # raw logit
