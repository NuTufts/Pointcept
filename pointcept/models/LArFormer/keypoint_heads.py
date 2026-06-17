"""Dense per-spacepoint keypoint heads for the LArFormer keypoint module.

Phase 1 (see lartpc_data_prep/larformer_keypoint/README.md):

    KeypointScoreHead — per-SP MLP predicting the (N, n_types) keypoint
    proximity-score field (the `kpscores` target). Trained against the
    Gaussian-proximity GT with MSE (LArMatch precedent) on the sigmoid of
    the logits, so the output is naturally bounded to [0, 1].

    KeypointOffsetHead — per-SP MLP predicting, for each keypoint type, a
    3D offset (vote) from the spacepoint toward the nearest keypoint of
    that type (VoteNet/PPN-style). Supervised only at points NEAR a
    keypoint (score > threshold). Lets the model localize a keypoint that
    sits BETWEEN measured points: predicted_kp = sp_coord + predicted_offset.
"""

from typing import Optional

import torch
import torch.nn as nn

from .decoder import build_coord_pos_emb


class KeypointScoreHead(nn.Module):
    """Per-spacepoint keypoint-score regressor.

    Args:
        in_dim:      per-SP backbone feature dim.
        n_types:     number of keypoint types / output channels (default 6).
        hidden_dim:  optional MLP width; 0 → a single linear layer.
        dropout:     dropout between hidden layers (only if hidden_dim > 0).

    Output: raw logits (N, n_types). Apply sigmoid for the score in [0, 1].
    The output projection is NOT zero-initialized — unlike the per-token cls
    head, we WANT the head to express sharp peaks from iteration 1; the
    bounded sigmoid + MSE keeps it stable without zero-init.
    """

    def __init__(self, in_dim, n_types=6, hidden_dim=256, dropout=0.0,
                 pos_emb_kind: Optional[str] = None,
                 pos_emb_dim: Optional[int] = None,
                 pos_emb_num_freq: Optional[int] = None,
                 pos_emb_max_freq: float = 256.0,
                 pos_emb_hidden_dim: Optional[int] = None):
        super().__init__()
        self.in_dim = int(in_dim)
        self.n_types = int(n_types)
        # Opt-in coordinate pos-emb concatenated onto the per-point feature, so
        # the MLP gets an explicit, undistorted position handle (the backbone
        # feature carries position only implicitly). Off by default → identical
        # to the original feature-only head.
        pe_dim = int(pos_emb_dim) if pos_emb_dim else self.in_dim
        self.pos_emb = build_coord_pos_emb(
            pos_emb_kind, pe_dim, num_freq=pos_emb_num_freq,
            max_freq=pos_emb_max_freq, hidden_dim=pos_emb_hidden_dim)
        net_in = self.in_dim + (pe_dim if self.pos_emb is not None else 0)
        if hidden_dim and hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(net_in, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(hidden_dim, self.n_types),
            )
        else:
            self.net = nn.Linear(net_in, self.n_types)

    def forward(self, feat: torch.Tensor,
                coord: Optional[torch.Tensor] = None) -> torch.Tensor:
        """feat: (N, in_dim), coord: (N, 3) → (N, n_types) logits."""
        if self.pos_emb is not None:
            if coord is None:
                raise ValueError("KeypointScoreHead has a pos-emb but got "
                                 "coord=None")
            feat = torch.cat([feat, self.pos_emb(coord)], dim=-1)
        return self.net(feat)


class KeypointOffsetHead(nn.Module):
    """Per-spacepoint offset/vote regressor (VoteNet/PPN-style).

    Predicts, for each keypoint type, a 3D displacement from the spacepoint
    toward the nearest keypoint of that type, in the SAME (normalized) frame
    as `coord_norm`. The predicted keypoint location for a point is
    `coord_norm + offset`. Supervised only where the point is near a keypoint
    (handled by the loss mask in the model), so off-cloud keypoints can be
    voted for from the nearby measured points.

    Args:
        in_dim:      per-SP backbone feature dim.
        n_types:     number of keypoint types (default 6).
        hidden_dim:  MLP width; 0 → single linear.
        dropout:     dropout between hidden layers (only if hidden_dim > 0).

    Output: (N, n_types, 3) offsets. The output projection is zero-init so
    the model starts by voting "no displacement" (predicted_kp = sp itself)
    — a sensible, stable prior that the supervised points then refine.
    """

    def __init__(self, in_dim, n_types=6, hidden_dim=256, dropout=0.0,
                 pos_emb_kind: Optional[str] = None,
                 pos_emb_dim: Optional[int] = None,
                 pos_emb_num_freq: Optional[int] = None,
                 pos_emb_max_freq: float = 256.0,
                 pos_emb_hidden_dim: Optional[int] = None):
        super().__init__()
        self.in_dim = int(in_dim)
        self.n_types = int(n_types)
        out_dim = self.n_types * 3
        # Opt-in coordinate pos-emb (see KeypointScoreHead). The offset head
        # benefits most: regressing a geometric displacement is easier with an
        # explicit coordinate than from a position-entangled feature. Off by
        # default → identical to the original feature-only head.
        pe_dim = int(pos_emb_dim) if pos_emb_dim else self.in_dim
        self.pos_emb = build_coord_pos_emb(
            pos_emb_kind, pe_dim, num_freq=pos_emb_num_freq,
            max_freq=pos_emb_max_freq, hidden_dim=pos_emb_hidden_dim)
        net_in = self.in_dim + (pe_dim if self.pos_emb is not None else 0)
        if hidden_dim and hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(net_in, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(hidden_dim, out_dim),
            )
            out_lin = self.net[-1]
        else:
            self.net = nn.Linear(net_in, out_dim)
            out_lin = self.net
        nn.init.zeros_(out_lin.weight)
        nn.init.zeros_(out_lin.bias)

    def forward(self, feat: torch.Tensor,
                coord: Optional[torch.Tensor] = None) -> torch.Tensor:
        """feat: (N, in_dim), coord: (N, 3) → (N, n_types, 3) offsets."""
        if self.pos_emb is not None:
            if coord is None:
                raise ValueError("KeypointOffsetHead has a pos-emb but got "
                                 "coord=None")
            feat = torch.cat([feat, self.pos_emb(coord)], dim=-1)
        return self.net(feat).view(-1, self.n_types, 3)
