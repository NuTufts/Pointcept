"""
Tokenizer for the shower-clustering Mask2Former model.

Takes per-spacepoint features from the frozen backbone and produces three
token sets the decoder will cross-attend to (see design doc §3):

    spacepoint_tokens (N, D) — projected backbone features (mask-refinement)
    fragment_tokens   (F, D) — pooled features per DBSCAN fragment + per-frag
                                geometric content enrichment
    voxel_tokens      (V, D) — mean-pooled features per 5 cm voxel

For each scale we ALSO return a (•, 3) coord tensor so the decoder can apply
its shared positional-embedding MLP. The PE is NOT added inside the tokenizer
anymore — the decoder owns position embedding so the same (3)→(D) function is
applied at every attention call (DETR / Mask2Former canonical pattern).

Returned dict keys (per event, no batch dim):

    spacepoint_tokens (N, D)        spacepoint_coords (N, 3)  normalized
    voxel_tokens      (V, D)        voxel_coords      (V, 3)  normalized
    fragment_tokens   (F, D)        fragment_coords   (F, 3)  normalized

Fragment pool is a mini set-transformer: a learnable pool query plus K
self-attention layers over the fragment's spacepoints. Voxel pool is a
deterministic mean (we don't have enough per-voxel structure to justify a
learnable pool, and voxel scale is supposed to give cheap global context).

NOTE on memory: FragmentPool pads fragments in a batch to max-fragment-size,
and MultiheadAttention is O(M²) in memory. To bound peak memory, fragments
larger than `max_pool_points` (default 512) are randomly subsampled before
pooling. The pool query still sees a representative subset of the fragment;
the output token is unbiased for the model's purpose (representing fragment
identity, not enumerating every point). At max_pool_points=512, peak working
memory for a 100-fragment event with 8 heads is ~800 MB.
"""

from typing import List, Optional, Sequence

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Fragment pool (mini set-transformer)
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
        # x: (B, T, D); key_padding_mask: (B, T) bool, True = pad
        h = self.norm1(x)
        attn_out, _ = self.attn(
            h, h, h, key_padding_mask=key_padding_mask, need_weights=False,
        )
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class FragmentPool(nn.Module):
    """
    Mini set-transformer pool over spacepoints in each DBSCAN fragment.

    Procedure:
        1. Project per-spacepoint features (in_dim -> hidden_dim).
        2. Per fragment, prepend a learnable pool query, then run K
           self-attention layers over (1 + M_frag) tokens.
        3. Take the pool query output as the per-fragment token.
        4. Project hidden_dim -> out_dim.

    Forward arg `fragment_indices` is a list of per-fragment 1-D LongTensors
    (or numpy arrays) indexing into `feat`. Variable fragment sizes are
    padded to the batch's max with a key-padding mask.

    Empty fragment list (F=0) returns a (0, out_dim) tensor on the same
    device/dtype as `feat`.
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
        """Random subsample if the fragment exceeds max_pool_points.

        Uses torch.randperm so behavior follows the calling code's RNG seed.
        On train this gives augmentation; on val a fixed seed gives determinism.
        """
        if self.max_pool_points <= 0 or idx.shape[0] <= self.max_pool_points:
            return idx
        perm = torch.randperm(idx.shape[0], device=idx.device)
        return idx[perm[:self.max_pool_points]]

    def forward(
        self,
        feat: torch.Tensor,
        fragment_indices: Sequence,
    ) -> torch.Tensor:
        """
        Args:
            feat: (N, in_dim) per-spacepoint features
            fragment_indices: length-F list of per-fragment index arrays
        Returns:
            (F, out_dim) per-fragment pooled tokens
        """
        device = feat.device
        n_frags = len(fragment_indices)
        if n_frags == 0:
            return feat.new_zeros(0, self.out_dim)

        # Move/coerce all index arrays to LongTensor on `feat`'s device, then
        # subsample if oversized. Cap on per-fragment input is needed to bound
        # MultiheadAttention's O(M²) memory.
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

        # Pad to (F, max_m, hidden_dim). False = valid, True = padding.
        padded = feat.new_zeros(n_frags, max_m, self.hidden_dim)
        pad_mask = torch.ones(n_frags, max_m, dtype=torch.bool, device=device)
        for fi, (idx, m) in enumerate(zip(idx_tensors, sizes)):
            padded[fi, :m] = self.in_proj(feat[idx])
            pad_mask[fi, :m] = False

        # Prepend learnable pool query at position 0 (always valid)
        pool_q = self.pool_query.expand(n_frags, -1, -1)  # (F, 1, hidden_dim)
        tokens = torch.cat([pool_q, padded], dim=1)
        full_mask = torch.cat(
            [torch.zeros(n_frags, 1, dtype=torch.bool, device=device), pad_mask],
            dim=1,
        )  # (F, 1+max_m)

        for blk in self.blocks:
            tokens = blk(tokens, key_padding_mask=full_mask)

        pooled = tokens[:, 0, :]  # (F, hidden_dim)
        return self.out_proj(pooled)


# ---------------------------------------------------------------------------
# Voxel pool (deterministic mean + linear projection)
# ---------------------------------------------------------------------------

class VoxelPool(nn.Module):
    """Mean-pool spacepoint features into voxels via scatter-add, then project.

    Empty voxels (no spacepoints) are not produced — n_voxels equals the count
    of unique voxel_ids in the input. Caller guarantees voxel_id ∈ [0, V).
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(
        self,
        feat: torch.Tensor,
        voxel_id: torch.Tensor,
        n_voxels: int,
    ) -> torch.Tensor:
        """
        Args:
            feat: (N, in_dim)
            voxel_id: (N,) long, in [0, n_voxels)
            n_voxels: V
        Returns:
            (V, out_dim)
        """
        if n_voxels == 0:
            return feat.new_zeros(0, self.out_dim)
        v_sum = feat.new_zeros(n_voxels, self.in_dim)
        v_sum.index_add_(0, voxel_id, feat)
        ones = feat.new_ones(feat.shape[0])
        v_count = feat.new_zeros(n_voxels)
        v_count.index_add_(0, voxel_id, ones)
        v_mean = v_sum / v_count.clamp(min=1.0).unsqueeze(-1)
        return self.proj(v_mean)


# ---------------------------------------------------------------------------
# Per-fragment content enricher
# ---------------------------------------------------------------------------

class FragmentContentEnricher(nn.Module):
    """
    Adds per-fragment geometric / strength features into the fragment tokens
    as CONTENT (not position — the decoder's shared pos_emb provides position).

    Per-fragment features (centroid is intentionally left out since the
    decoder's pos_emb on `fragment_coords` already encodes it):

        pca_axis (3) — first principal-axis direction (unit length)
        bbox_extent (3) — max - min of normalized spacepoint coords
        log_count (1) — log(1 + n_points_in_fragment)
        mean_strength (3) — mean of strength features

        total = 10 dim → MLP → out_dim, then ADDED to the fragment_tokens.
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
        """Compute the 10-dim feature vector per fragment + return centroids."""
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
            c = coord_norm[idx]                # (m, 3)
            s = strength[idx]                  # (m, 3)
            m = c.shape[0]
            centroid = c.mean(dim=0)
            extent = c.amax(dim=0) - c.amin(dim=0)
            mean_str = s.mean(dim=0)
            log_count = torch.log1p(c.new_tensor(float(m)))
            # PCA: covariance + eigendecomposition. Robust for tiny m.
            if m >= 2:
                c_centered = c - centroid
                cov = (c_centered.transpose(0, 1) @ c_centered) / float(m - 1)
                # eigh on CUDA isn't implemented for Half; cast the
                # symmetric 3x3 cov to float32 just for this op so the
                # tokenizer is AMP-compatible.
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
# Per-scale coord builders
# ---------------------------------------------------------------------------

def _voxel_centers(voxel_keys: torch.Tensor, voxel_size_norm: float) -> torch.Tensor:
    """Voxel-grid integer coords → voxel center in normalized coords (V, 3)."""
    if voxel_keys.shape[0] == 0:
        return voxel_keys.new_zeros(0, 3, dtype=torch.float32)
    return (voxel_keys.float() + 0.5) * float(voxel_size_norm)


def _fragment_centroids(
    coord_norm: torch.Tensor,
    fragment_indices: Sequence,
) -> torch.Tensor:
    """Per-fragment mean of normalized spacepoint coords (F, 3)."""
    device = coord_norm.device
    n_frags = len(fragment_indices)
    out = coord_norm.new_zeros(n_frags, 3)
    for fi, idx in enumerate(fragment_indices):
        if not isinstance(idx, torch.Tensor):
            idx = torch.as_tensor(idx, dtype=torch.long, device=device)
        else:
            idx = idx.to(device=device, dtype=torch.long)
        out[fi] = coord_norm[idx].mean(dim=0)
    return out


# ---------------------------------------------------------------------------
# Top-level tokenizer module
# ---------------------------------------------------------------------------

class ShowerClusteringTokenizer(nn.Module):
    """
    Wraps fragment / voxel / spacepoint token construction into one module.

    Forward inputs: backbone per-spacepoint features and the per-event
    bookkeeping tensors emitted by `ShowerClusteringDataset`.

    Forward output dict:
        spacepoint_tokens (N, token_dim)   spacepoint_coords (N, 3)
        fragment_tokens   (F, token_dim)   fragment_coords   (F, 3)
        voxel_tokens      (V, token_dim)   voxel_coords      (V, 3)

    Coords are in the same normalized frame as `coord_norm` so the decoder
    can apply one shared (3)→(D) position-embedding MLP across all scales
    and across queries.

    Per-event semantics. Multi-event batches are processed by calling forward
    once per event (see Phase 4 of the design doc); the decoder ties the
    three token sets together.
    """

    def __init__(
        self,
        in_dim: int,
        token_dim: int,
        voxel_size_norm: float,
        frag_pool_hidden_dim: Optional[int] = None,
        frag_pool_layers: int = 2,
        frag_pool_heads: int = 8,
        frag_pool_mlp_ratio: float = 4.0,
        frag_pool_dropout: float = 0.0,
        frag_pool_max_points: int = 512,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.token_dim = token_dim
        self.voxel_size_norm = float(voxel_size_norm)

        if frag_pool_hidden_dim is None:
            frag_pool_hidden_dim = token_dim

        self.fragment_pool = FragmentPool(
            in_dim=in_dim,
            hidden_dim=frag_pool_hidden_dim,
            out_dim=token_dim,
            num_layers=frag_pool_layers,
            num_heads=frag_pool_heads,
            mlp_ratio=frag_pool_mlp_ratio,
            dropout=frag_pool_dropout,
            max_pool_points=frag_pool_max_points,
        )
        # Geometric / strength features are content (not position) now.
        self.fragment_content = FragmentContentEnricher(out_dim=token_dim)

        self.voxel_pool = VoxelPool(in_dim=in_dim, out_dim=token_dim)

        self.spacepoint_proj = nn.Linear(in_dim, token_dim)

    def forward(
        self,
        sp_feat: torch.Tensor,
        coord_norm: torch.Tensor,
        strength: torch.Tensor,
        voxel_id: torch.Tensor,
        voxel_keys: torch.Tensor,
        n_voxels: int,
        fragment_indices: Sequence,
    ) -> dict:
        """
        Args:
            sp_feat: (N, in_dim) per-spacepoint backbone features
            coord_norm: (N, 3) normalized spacepoint coords
            strength: (N, 3) input strength features
            voxel_id: (N,) long — voxel index per spacepoint
            voxel_keys: (V, 3) long — integer grid coords for each voxel
            n_voxels: V
            fragment_indices: length-F list of per-fragment index arrays
        Returns:
            dict with `*_tokens` and `*_coords` for spacepoint / voxel / fragment.
        """
        sp_tokens = self.spacepoint_proj(sp_feat)
        sp_coords = coord_norm

        if len(fragment_indices) > 0:
            frag_pooled = self.fragment_pool(sp_feat, fragment_indices)
            frag_content = self.fragment_content(
                coord_norm, strength, fragment_indices,
            )
            fragment_tokens = frag_pooled + frag_content
            fragment_coords = _fragment_centroids(coord_norm, fragment_indices)
        else:
            fragment_tokens = sp_feat.new_zeros(0, self.token_dim)
            fragment_coords = sp_feat.new_zeros(0, 3)

        if n_voxels > 0:
            voxel_tokens = self.voxel_pool(sp_feat, voxel_id, n_voxels)
            voxel_coords = _voxel_centers(voxel_keys, self.voxel_size_norm)
            # voxel_coords inherits voxel_keys' device but should match
            # tokens dtype for downstream additions.
            voxel_coords = voxel_coords.to(dtype=voxel_tokens.dtype,
                                           device=voxel_tokens.device)
        else:
            voxel_tokens = sp_feat.new_zeros(0, self.token_dim)
            voxel_coords = sp_feat.new_zeros(0, 3)

        return {
            "spacepoint_tokens": sp_tokens,
            "spacepoint_coords": sp_coords,
            "fragment_tokens": fragment_tokens,
            "fragment_coords": fragment_coords,
            "voxel_tokens": voxel_tokens,
            "voxel_coords": voxel_coords,
        }
