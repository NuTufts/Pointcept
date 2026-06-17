"""KeypointRefinementDecoder — Phase-3 (design 2B) point-refinement decoder.

Sharpens the Phase-2 query keypoints (start/end) by letting each particle
query look at ITS OWN spacepoints, instead of regressing the point from a
single global query embedding (the 2A limitation — coarse track-end).

Per layer (L = 2-3), reusing the main decoder's `_MaskedDecoderLayer`:
    masked cross-attention: query → spacepoint-level tokens, gated by the
        query's own predicted spacepoint mask (so it refines from the right
        points; off-cloud keypoints stay expressible — we regress an offset);
    self-attention among queries (vertex/consensus + endpoint consistency);
    FFN;
    point head: regress an OFFSET to the current start/end estimate
        (iterative refinement, Deformable-DETR / PPN style).

Identity at init: the per-layer point head's output projection is zero-init,
so `refined == initial` at iteration 0 — adding this decoder on top of a
trained 2A checkpoint changes nothing until it learns, which keeps fine-tuning
stable.

Returns one dict per layer (deep supervision), each shaped like the 2A head:
    {origin (=start), end, start_logvar, end_logvar, end_logit}
all (Q, ...). The model uses the LAST layer as the final keypoint prediction
and supervises every layer.
"""

from typing import List, Optional

import torch
import torch.nn as nn

from .decoder import _MaskedDecoderLayer, SinusoidalPosEmb3D


class KeypointRefinementDecoder(nn.Module):
    """Iterative, mask-gated point-refinement decoder for keypoints.

    Args:
        dim:              query/token dim (= decoder token_dim).
        num_layers:       number of refinement layers (2-3 typical).
        num_heads:        MHA heads.
        mlp_ratio:        FFN expansion in each layer.
        dropout:          MHA/FFN dropout.
        mask_threshold:   cross-attention is blocked where a query's
                          spacepoint mask logit <= this (sigmoid 0.5 at 0).
        pos_emb_kind:     "mlp" (default) or "sinusoidal" — position embedding
                          for the anchor (current start) + spacepoint coords.
        pos_emb_*:        sinusoidal params (when chosen).
    """

    def __init__(self, dim: int, num_layers: int = 2, num_heads: int = 4,
                 mlp_ratio: float = 4.0, dropout: float = 0.0,
                 mask_threshold: float = 0.0, pos_emb_kind: str = "mlp",
                 pos_emb_num_freq: Optional[int] = None,
                 pos_emb_max_freq: float = 256.0):
        super().__init__()
        self.dim = int(dim)
        self.num_layers = int(num_layers)
        self.mask_threshold = float(mask_threshold)

        if pos_emb_kind == "sinusoidal":
            self.pos_emb = SinusoidalPosEmb3D(
                dim, num_freq=pos_emb_num_freq, max_freq=pos_emb_max_freq)
        else:
            self.pos_emb = nn.Sequential(
                nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))

        self.layers = nn.ModuleList([
            _MaskedDecoderLayer(dim, num_heads, mlp_ratio=mlp_ratio,
                                dropout=dropout)
            for _ in range(self.num_layers)
        ])
        # Per-layer point head: delta_start(3), delta_end(3), start_logvar(3),
        # end_logvar(3), delta_end_logit(1) = 13. Output zero-init → identity
        # refinement + logvar≈0 (var≈1) + end_logit unchanged at init.
        self.point_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 13))
            for _ in range(self.num_layers)
        ])
        for ph in self.point_heads:
            nn.init.zeros_(ph[-1].weight)
            nn.init.zeros_(ph[-1].bias)

    def _attn_mask(self, sp_mask_logits: torch.Tensor) -> Optional[torch.Tensor]:
        """(Q, M) bool — True blocks attention. Mirrors the main decoder:
        block points below threshold; un-block any all-blocked query row so
        it can still attend (avoids NaN)."""
        if sp_mask_logits.shape[1] == 0:
            return None
        am = (sp_mask_logits.detach() <= self.mask_threshold)
        all_masked = am.all(dim=-1)
        if all_masked.any():
            am = am.clone()
            am[all_masked] = False
        return am

    def forward(
        self,
        query_embed: torch.Tensor,      # (Q, D)
        init_start: torch.Tensor,        # (Q, 3)
        init_end: torch.Tensor,          # (Q, 3)
        init_end_logit: torch.Tensor,    # (Q,)
        sp_tokens: torch.Tensor,         # (M, D)
        sp_coords: torch.Tensor,         # (M, 3)
        sp_mask_logits: torch.Tensor,    # (Q, M)
    ) -> List[dict]:
        Q = query_embed.shape[0]
        if Q == 0 or sp_tokens.shape[0] == 0:
            # Nothing to refine against — return the 2A estimate unchanged so
            # downstream supervision/eval still has a (single) layer.
            return [{
                "origin": init_start, "end": init_end,
                "start_logvar": init_start.new_zeros(Q, 3),
                "end_logvar": init_end.new_zeros(Q, 3),
                "end_logit": init_end_logit,
            }]

        queries = query_embed.unsqueeze(0)                  # (1, Q, D)
        keys = sp_tokens.unsqueeze(0)                       # (1, M, D)
        key_pos = self.pos_emb(sp_coords)                  # (M, D)
        attn_mask = self._attn_mask(sp_mask_logits)

        cur_start = init_start
        cur_end = init_end
        cur_end_logit = init_end_logit
        out_layers: List[dict] = []
        for li in range(self.num_layers):
            # Anchor the query position at the current start estimate (the
            # origin-feedback trick from the main decoder).
            query_pos = self.pos_emb(cur_start)             # (Q, D)
            queries = self.layers[li](
                queries, query_pos, keys, key_pos, attn_mask=attn_mask)
            h = queries.squeeze(0)                          # (Q, D)
            d = self.point_heads[li](h)                     # (Q, 13)
            cur_start = cur_start + d[:, 0:3]
            cur_end = cur_end + d[:, 3:6]
            start_logvar = d[:, 6:9]
            end_logvar = d[:, 9:12]
            cur_end_logit = cur_end_logit + d[:, 12]
            out_layers.append({
                "origin": cur_start, "end": cur_end,
                "start_logvar": start_logvar, "end_logvar": end_logvar,
                "end_logit": cur_end_logit,
            })
        return out_layers
