"""
FragmentBuilder — token per pre-computed (DBSCAN) fragment.

Reads `event_dict["fragment_indices"]` (length-F list of LongTensors of
per-fragment spacepoint indices) and produces:

  - tokens (F, D):   FragmentPool (mini set-transformer over a fragment's
                      spacepoint features) + FragmentContentEnricher
                      (per-fragment PCA axis, bbox extent, log-count,
                      mean strength), added.
  - coords (F, 3):   per-fragment centroid in normalized coords.
  - sp_to_level_id (N,): which fragment a spacepoint belongs to, or −1 if
                          it's not in any fragment (the common case — DBSCAN
                          covers only shower-tagged spacepoints).

The pool / enricher code is **lifted verbatim** from
`pointcept/models/shower_clustering/tokenizer.py` so LArFormer is fully
self-contained. Any future divergence between the two implementations is
intentional and should be commented.

`event_dict` keys this builder reads:
    fragment_indices: list[F] of (M_f,) LongTensor   (REQUIRED)
    coord_norm:        (N, 3)                        (REQUIRED)
    feat:              (N, ≥6) — strength = feat[:, 3:6]
                                 (REQUIRED; falls back to zeros if missing)

If `event_dict` has no `fragment_indices`, returns an empty level (F=0),
which the decoder + loss handle gracefully (synthesized dummy keys).
"""

from typing import List, Optional, Sequence

import torch
import torch.nn as nn

from .base import BUILDERS, LevelBuilder, LevelOutput


# ---------------------------------------------------------------------------
# Mini set-transformer pool (copy from shower_clustering/tokenizer.py)
# ---------------------------------------------------------------------------

class _FragAttnBlock(nn.Module):
    """Pre-norm self-attention block with MLP residual."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(
            h, h, h, key_padding_mask=key_padding_mask, need_weights=False,
        )
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class FragmentPool(nn.Module):
    """Mini set-transformer pool over spacepoints in each fragment.

    Per-fragment pool: prepend a learnable query, run K self-attention layers
    over (1 + M_frag) tokens, take the pool query out. Empty fragment list
    (F=0) returns a (0, out_dim) tensor on the same device/dtype as `feat`.

    To bound peak memory, fragments larger than `max_pool_points` are
    randomly subsampled before pooling.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        max_pool_points: int = 512,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.max_pool_points = int(max_pool_points)

        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.pool_query = nn.Parameter(torch.empty(1, 1, hidden_dim))
        nn.init.trunc_normal_(self.pool_query, std=0.02)

        self.blocks = nn.ModuleList([
            _FragAttnBlock(hidden_dim, num_heads,
                           mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def _maybe_subsample(self, idx: torch.Tensor) -> torch.Tensor:
        if self.max_pool_points <= 0 or idx.shape[0] <= self.max_pool_points:
            return idx
        perm = torch.randperm(idx.shape[0], device=idx.device)
        return idx[perm[:self.max_pool_points]]

    def forward(
        self,
        feat: torch.Tensor,
        fragment_indices: Sequence,
    ) -> torch.Tensor:
        device = feat.device
        n_frags = len(fragment_indices)
        if n_frags == 0:
            return feat.new_zeros(0, self.out_dim)

        idx_tensors: List[torch.Tensor] = []
        sizes: List[int] = []
        for idx in fragment_indices:
            if not isinstance(idx, torch.Tensor):
                idx = torch.as_tensor(idx, dtype=torch.long, device=device)
            else:
                idx = idx.to(device=device, dtype=torch.long)
            idx = self._maybe_subsample(idx)
            idx_tensors.append(idx)
            sizes.append(int(idx.shape[0]))
        max_m = max(sizes)

        padded = feat.new_zeros(n_frags, max_m, self.hidden_dim)
        pad_mask = torch.ones(n_frags, max_m, dtype=torch.bool, device=device)
        for fi, (idx, m) in enumerate(zip(idx_tensors, sizes)):
            padded[fi, :m] = self.in_proj(feat[idx])
            pad_mask[fi, :m] = False

        pool_q = self.pool_query.expand(n_frags, -1, -1)
        tokens = torch.cat([pool_q, padded], dim=1)
        full_mask = torch.cat(
            [torch.zeros(n_frags, 1, dtype=torch.bool, device=device), pad_mask],
            dim=1,
        )

        for blk in self.blocks:
            tokens = blk(tokens, key_padding_mask=full_mask)

        pooled = tokens[:, 0, :]
        return self.out_proj(pooled)


# ---------------------------------------------------------------------------
# Per-fragment content enricher (copy from shower_clustering/tokenizer.py)
# ---------------------------------------------------------------------------

class FragmentContentEnricher(nn.Module):
    """Adds per-fragment geometric / strength features into the fragment tokens.

    Per-fragment features (centroid is excluded — the decoder's pos_emb on
    fragment_coords already encodes it):
        pca_axis (3) — first principal-axis direction (unit length)
        bbox_extent (3) — max - min of normalized spacepoint coords
        log_count (1) — log(1 + n_points_in_fragment)
        mean_strength (3) — mean of strength features
    Total: 10 dim → MLP → out_dim, ADDED to the pooled fragment tokens.
    """

    INPUT_DIM = 3 + 3 + 1 + 3

    def __init__(self, out_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.out_dim = out_dim
        self.mlp = nn.Sequential(
            nn.Linear(self.INPUT_DIM, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    @staticmethod
    def _per_fragment_features(
        coord_norm: torch.Tensor,
        strength: torch.Tensor,
        fragment_indices: Sequence,
    ) -> torch.Tensor:
        device = coord_norm.device
        n_frags = len(fragment_indices)
        feats = coord_norm.new_zeros(n_frags, FragmentContentEnricher.INPUT_DIM)
        if n_frags == 0:
            return feats
        for fi, idx in enumerate(fragment_indices):
            if not isinstance(idx, torch.Tensor):
                idx = torch.as_tensor(idx, dtype=torch.long, device=device)
            else:
                idx = idx.to(device=device, dtype=torch.long)
            c = coord_norm[idx]
            s = strength[idx]
            m = c.shape[0]
            centroid = c.mean(dim=0)
            extent = c.amax(dim=0) - c.amin(dim=0)
            mean_str = s.mean(dim=0)
            log_count = torch.log1p(c.new_tensor(float(m)))
            if m >= 2:
                c_centered = c - centroid
                cov = (c_centered.transpose(0, 1) @ c_centered) / float(m - 1)
                # eigh on CUDA isn't implemented for Half; cast to float32.
                eigvals, eigvecs = torch.linalg.eigh(cov.float())
                axis = eigvecs[:, -1].to(c.dtype)
                if (axis @ centroid) < 0:
                    axis = -axis
            else:
                axis = c.new_zeros(3)
            feats[fi, 0:3] = axis
            feats[fi, 3:6] = extent
            feats[fi, 6] = log_count
            feats[fi, 7:10] = mean_str
        return feats

    def forward(
        self,
        coord_norm: torch.Tensor,
        strength: torch.Tensor,
        fragment_indices: Sequence,
    ) -> torch.Tensor:
        feats = self._per_fragment_features(coord_norm, strength, fragment_indices)
        return self.mlp(feats)


# ---------------------------------------------------------------------------
# Centroid helper
# ---------------------------------------------------------------------------

def _fragment_centroids(
    coord_norm: torch.Tensor,
    fragment_indices: Sequence,
) -> torch.Tensor:
    device = coord_norm.device
    n_frags = len(fragment_indices)
    out = coord_norm.new_zeros(n_frags, 3)
    for fi, idx in enumerate(fragment_indices):
        if not isinstance(idx, torch.Tensor):
            idx = torch.as_tensor(idx, dtype=torch.long, device=device)
        else:
            idx = idx.to(device=device, dtype=torch.long)
        if idx.numel() > 0:
            out[fi] = coord_norm[idx].mean(dim=0)
    return out


# ---------------------------------------------------------------------------
# FragmentBuilder
# ---------------------------------------------------------------------------

@BUILDERS.register_module()
class FragmentBuilder(LevelBuilder):
    """Produces one token per pre-computed fragment.

    Args:
        in_dim:           backbone feature dim
        token_dim:        output token dim
        pool_hidden_dim:  hidden width of FragmentPool (default = token_dim)
        pool_layers:      depth of the per-fragment self-attention stack
        pool_heads:       MHA heads in the pool
        pool_mlp_ratio:   FFN expansion in the pool
        pool_dropout:     dropout in pool attention + FFN
        pool_max_points:  per-fragment random subsample cap (bounds O(M²) attn)
        enricher_hidden_dim: hidden width of FragmentContentEnricher
        strength_offset:  start column of per-SP strength inside `feat`
        strength_dim:     number of strength dims after `strength_offset`
                          (set to 0 to skip the content enricher entirely
                          when no strength is available)
    """

    def __init__(
        self,
        in_dim: int,
        token_dim: int,
        pool_hidden_dim: Optional[int] = None,
        pool_layers: int = 2,
        pool_heads: int = 8,
        pool_mlp_ratio: float = 4.0,
        pool_dropout: float = 0.0,
        pool_max_points: int = 512,
        enricher_hidden_dim: int = 128,
        strength_offset: int = 3,
        strength_dim: int = 3,
    ):
        super().__init__(in_dim=in_dim, token_dim=token_dim)
        if pool_hidden_dim is None:
            pool_hidden_dim = token_dim
        self.pool = FragmentPool(
            in_dim=in_dim, hidden_dim=pool_hidden_dim, out_dim=token_dim,
            num_layers=pool_layers, num_heads=pool_heads,
            mlp_ratio=pool_mlp_ratio, dropout=pool_dropout,
            max_pool_points=pool_max_points,
        )
        self.strength_offset = int(strength_offset)
        self.strength_dim = int(strength_dim)
        self.enricher = (FragmentContentEnricher(
            out_dim=token_dim, hidden_dim=enricher_hidden_dim,
        ) if self.strength_dim > 0 else None)

    def _build_sp_to_level_id(
        self,
        fragment_indices: Sequence,
        n_sp: int,
        device: torch.device,
    ) -> torch.Tensor:
        """SP → fragment idx, with −1 for SPs in no fragment.

        DBSCAN clusters are typically disjoint, but if a SP appears in two
        fragments (rare), last-write wins. Note this only matters for the
        loss's per-level GT-mask scatter; the matcher/decoder are unaffected.
        """
        out = torch.full((n_sp,), -1, dtype=torch.long, device=device)
        for fi, idx in enumerate(fragment_indices):
            if not isinstance(idx, torch.Tensor):
                idx = torch.as_tensor(idx, dtype=torch.long, device=device)
            else:
                idx = idx.to(device=device, dtype=torch.long)
            if idx.numel() > 0:
                out[idx] = fi
        return out

    def forward(
        self,
        sp_feat: torch.Tensor,
        coord_norm: torch.Tensor,
        event_dict: dict,
    ) -> LevelOutput:
        device = sp_feat.device
        n_sp = sp_feat.shape[0]
        fragment_indices = event_dict.get("fragment_indices", None) or []

        if len(fragment_indices) == 0:
            return LevelOutput(
                tokens=sp_feat.new_zeros(0, self.token_dim),
                coords=sp_feat.new_zeros(0, 3),
                sp_to_level_id=torch.full((n_sp,), -1, dtype=torch.long,
                                          device=device),
                name=self.name,
            )

        tokens = self.pool(sp_feat, fragment_indices)
        if self.enricher is not None:
            feat = event_dict.get("feat", None)
            if feat is not None and feat.shape[1] >= (self.strength_offset
                                                     + self.strength_dim):
                strength = feat[:, self.strength_offset:
                                self.strength_offset + self.strength_dim]
            else:
                strength = sp_feat.new_zeros(n_sp, self.strength_dim)
            tokens = tokens + self.enricher(coord_norm, strength, fragment_indices)

        coords = _fragment_centroids(coord_norm, fragment_indices)
        sp_to_level_id = self._build_sp_to_level_id(
            fragment_indices, n_sp, device,
        )
        return LevelOutput(
            tokens=tokens,
            coords=coords,
            sp_to_level_id=sp_to_level_id,
            name=self.name,
        )
