"""
FiLM classifier. Owned by: Lionel + Terence.

The B-signals don't get appended -- they generate a per-dimension scale and shift that
modulate the CLIP embedding directly, before the MLP ever sees it.

This file REQUIRES signal_dim > 0 -- FiLM's entire mechanism is using signals to reshape
the embedding, so with zero signals there's nothing for it to compute. It is not a
smaller version of FiLM, it isn't FiLM at all. If you want a working baseline before
signals.py exists, use model_concat.py instead (signal_dim=0 there is a real, supported,
graceful case) -- come back to this file once real signals exist to feed it.
"""
import torch
import torch.nn as nn


class FiLMClassifier(nn.Module):
    def __init__(self, clip_dim: int, signal_dim: int, hidden_dim: int = 128):
        super().__init__()
        if signal_dim <= 0:
            raise ValueError(
                "FiLM needs signal_dim > 0 -- it has nothing to modulate the embedding "
                "with otherwise. Use model_concat.py (signal_dim=0) for a signal-free baseline."
            )
        self.clip_dim = clip_dim
        self.film_generator = nn.Linear(signal_dim, clip_dim * 2)  # outputs [scale | shift]
        self.net = nn.Sequential(
            nn.Linear(clip_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, clip_embedding: torch.Tensor, signals: torch.Tensor) -> torch.Tensor:
        film_params = self.film_generator(signals)
        scale, shift = film_params[:, : self.clip_dim], film_params[:, self.clip_dim :]
        modulated = clip_embedding * (1 + scale) + shift
        return self.net(modulated).squeeze(-1)
