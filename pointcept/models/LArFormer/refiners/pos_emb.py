"""Position-embedding modules shared across refiners.

Two flavors, matching the decoder's `pos_emb_kind` option:
  - "mlp":         (3) → hidden → (D) learnable MLP
  - "sinusoidal":  NeRF-style log-spaced fixed sin/cos features → Linear(D)

Both modules consume per-token coords in the same normalized frame as
spacepoint coord_norm (LArFormerDataset uses
  coord_norm = (coord_cm - coord_center) / coord_scale
i.e. roughly [-1, +1] inside the TPC).

Kept separate from decoder.py so refiners don't depend on the decoder
module — that way an ablation that wants different pos_emb flavors on
either side (refiner sinusoidal + decoder MLP, say) just sets two
unrelated config keys.
"""

import math
from typing import Optional

import torch
import torch.nn as nn


class MLPPosEmb(nn.Module):
    def __init__(self, dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.net(coords)


class SinusoidalPosEmb3D(nn.Module):
    def __init__(self, dim: int,
                 num_freq: Optional[int] = None,
                 max_freq: float = 256.0):
        super().__init__()
        if num_freq is None:
            num_freq = max(1, dim // 12)
        freqs = 2.0 ** torch.linspace(0.0, math.log2(float(max_freq)),
                                       int(num_freq))
        self.register_buffer("freqs", freqs, persistent=False)
        self.num_freq = int(num_freq)
        self.raw_dim = 3 * 2 * self.num_freq
        self.proj = nn.Linear(self.raw_dim, int(dim))

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        if coords.shape[-1] != 3:
            raise ValueError(
                f"SinusoidalPosEmb3D expects last dim == 3, got "
                f"shape {tuple(coords.shape)}"
            )
        scaled = coords.unsqueeze(-1) * self.freqs
        emb = torch.stack([torch.sin(scaled), torch.cos(scaled)], dim=-1)
        emb = emb.flatten(-3)
        return self.proj(emb)


def build_pos_emb(kind: str, dim: int,
                  hidden_dim: Optional[int] = None,
                  num_freq: Optional[int] = None,
                  max_freq: float = 256.0) -> nn.Module:
    kind = str(kind).lower()
    if kind == "mlp":
        return MLPPosEmb(dim, hidden_dim=hidden_dim)
    if kind == "sinusoidal":
        return SinusoidalPosEmb3D(dim, num_freq=num_freq, max_freq=max_freq)
    raise ValueError(
        f"pos_emb_kind must be 'mlp' or 'sinusoidal'; got {kind!r}"
    )
