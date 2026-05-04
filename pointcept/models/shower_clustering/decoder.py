"""
Mask2Former-style decoder for the shower-clustering model. Phase 4 of the
design (see pointcept/docs/shower_clustering_design.md §3, Phase 4).

Inputs (per event, no batch dim — model wraps batching):
    voxel_tokens       (V, D)  — coarse-scale tokens (mean-pooled features)
    fragment_tokens    (F, D)  — DBSCAN-fragment tokens (set-transformer pool)
    spacepoint_tokens  (N, D)  — projected per-spacepoint backbone features

Outputs (list of L layer dicts):
    class_logits       (Q, num_classes) — per-query class score
                                          (last entry is "no object")
    origin             (Q, 3)          — per-query predicted origin
                                          (normalized coord)
    mask_logits["voxel"]      (Q, V)
    mask_logits["fragment"]   (Q, F)
    mask_logits["spacepoint"] (Q, N)

Architecture per layer:
    1. masked cross-attention to ONE scale (rotating pattern)
    2. query self-attention
    3. FFN (pre-norm everywhere)

Default scale pattern over 6 layers:
    [voxel, fragment, voxel, fragment, spacepoint, spacepoint]

This biases compute toward the cheap scales (voxel ~2k, fragment ~100) and
puts the expensive spacepoint cross-attention (N up to ~250k) only after
we already have a good mask predicted at that scale. Configurable via the
`scale_pattern` arg.

Mask gating: at every layer we compute mask logits for all three scales
(used by deep supervision in Phase 6) and use them to gate the next
cross-attention at the matching scale. Mask2Former-style: each (q, k)
pair where the previous mask logit ≤ 0 is masked out of attention. If a
query ends up with all keys masked, we drop its mask for that step
(otherwise softmax NaNs).
"""

from typing import List, Optional, Sequence

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Query positional encoder: static learned embedding + dynamic origin MLP
# ---------------------------------------------------------------------------

class QueryPositionalEncoder(nn.Module):
    """Per-query positional encoding.

    Static component: learnable embedding per query (Q, D).
    Dynamic component: MLP on per-layer predicted origin coord (3D
    normalized) — feeds the auxiliary regression head's prediction back as
    positional information for the next layer's queries.
    """

    def __init__(self, num_queries: int, dim: int):
        super().__init__()
        self.num_queries = num_queries
        self.dim = dim
        self.static = nn.Parameter(torch.empty(num_queries, dim))
        nn.init.trunc_normal_(self.static, std=0.02)
        self.coord_mlp = nn.Sequential(
            nn.Linear(3, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, predicted_origin: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Returns (Q, D)."""
        pe = self.static
        if predicted_origin is not None:
            pe = pe + self.coord_mlp(predicted_origin)
        return pe


# ---------------------------------------------------------------------------
# Single decoder layer: masked-cross-attn → self-attn → FFN
# ---------------------------------------------------------------------------

class _MaskedDecoderLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int,
                 mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.cross_norm_q = nn.LayerNorm(dim)
        self.cross_norm_k = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.self_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(
        self,
        queries: torch.Tensor,    # (1, Q, D)
        query_pe: torch.Tensor,   # (Q, D)
        keys: torch.Tensor,       # (1, K, D)
        attn_mask: Optional[torch.Tensor] = None,  # (Q, K) bool, True=mask
    ) -> torch.Tensor:
        # Masked cross-attention: queries (with PE added) cross-attend to keys
        q = self.cross_norm_q(queries + query_pe.unsqueeze(0))
        k = self.cross_norm_k(keys)
        cross_out, _ = self.cross_attn(
            q, k, k,
            attn_mask=attn_mask,
            need_weights=False,
        )
        queries = queries + cross_out

        # Query self-attention (also gets PE)
        s = self.self_norm(queries + query_pe.unsqueeze(0))
        self_out, _ = self.self_attn(s, s, s, need_weights=False)
        queries = queries + self_out

        # FFN
        queries = queries + self.ffn(self.ffn_norm(queries))
        return queries


# ---------------------------------------------------------------------------
# Per-layer prediction heads (one set, applied at every layer)
# ---------------------------------------------------------------------------

class _PerLayerHeads(nn.Module):
    """Class logits, origin coord, and (shared) mask embed.

    Per-scale mask logits are computed externally as mask_embed(q) @ key.T;
    this module returns the embedding only.
    """

    def __init__(self, dim: int, num_classes: int):
        super().__init__()
        self.class_head = nn.Linear(dim, num_classes)
        self.origin_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 3),
        )
        self.mask_embed = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, queries: torch.Tensor) -> dict:
        """queries: (Q, D)."""
        return {
            "class_logits": self.class_head(queries),    # (Q, C)
            "origin": self.origin_head(queries),         # (Q, 3)
            "mask_embed": self.mask_embed(queries),      # (Q, D)
        }


# ---------------------------------------------------------------------------
# Mask2Former decoder
# ---------------------------------------------------------------------------

DEFAULT_SCALE_PATTERN = (
    "voxel", "fragment", "voxel", "fragment", "spacepoint", "spacepoint",
)


class Mask2FormerDecoder(nn.Module):
    """
    Multi-scale Mask2Former-style decoder.

    Args:
        dim: token dimension (must match the tokenizer's token_dim).
        num_queries: number of object queries (default 64).
        num_classes: number of class slots, including "no_object" (so for
                     5 origin types pass num_classes=6).
        num_layers: defaults to len(scale_pattern). If both are given they
                    must agree.
        scale_pattern: tuple of "voxel" / "fragment" / "spacepoint" giving
                       the cross-attention scale per layer.
        num_heads: MHA heads.
        mlp_ratio: FFN hidden expansion.
        dropout: passed to MHA + (optionally) FFN.
        mask_threshold: pre-sigmoid threshold for binary attention mask.
                        Mask2Former default: 0.0 (i.e. `sigmoid > 0.5`).
    """

    SCALES = ("voxel", "fragment", "spacepoint")

    def __init__(
        self,
        dim: int,
        num_queries: int = 64,
        num_classes: int = 6,
        num_layers: Optional[int] = None,
        scale_pattern: Sequence[str] = DEFAULT_SCALE_PATTERN,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        mask_threshold: float = 0.0,
    ):
        super().__init__()
        scale_pattern = tuple(scale_pattern)
        for s in scale_pattern:
            if s not in self.SCALES:
                raise ValueError(f"unknown scale '{s}' in scale_pattern; "
                                 f"valid: {self.SCALES}")
        if num_layers is None:
            num_layers = len(scale_pattern)
        elif num_layers != len(scale_pattern):
            raise ValueError(
                f"num_layers ({num_layers}) != len(scale_pattern) "
                f"({len(scale_pattern)})"
            )

        self.dim = dim
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.scale_pattern = scale_pattern
        self.mask_threshold = float(mask_threshold)

        # Initial query content (refined through the decoder layers).
        self.query_content = nn.Parameter(torch.empty(num_queries, dim))
        nn.init.trunc_normal_(self.query_content, std=0.02)

        self.query_pe = QueryPositionalEncoder(num_queries, dim)

        # Initial-layer prediction heads (compute mask logits BEFORE the
        # first masked cross-attention — Mask2Former does this so layer 0
        # has a sensible mask to gate by).
        self.init_heads = _PerLayerHeads(dim, num_classes)

        self.layers = nn.ModuleList([
            _MaskedDecoderLayer(dim, num_heads, mlp_ratio=mlp_ratio,
                                dropout=dropout)
            for _ in range(num_layers)
        ])
        self.layer_heads = nn.ModuleList([
            _PerLayerHeads(dim, num_classes) for _ in range(num_layers)
        ])

    # -- Helpers --------------------------------------------------------

    def _compute_predictions(
        self,
        heads: _PerLayerHeads,
        queries: torch.Tensor,         # (Q, D)
        voxel_tokens: torch.Tensor,    # (V, D)
        fragment_tokens: torch.Tensor, # (F, D)
        spacepoint_tokens: torch.Tensor,  # (N, D)
    ) -> dict:
        out = heads(queries)
        mask_embed = out["mask_embed"]
        mask_logits = {
            "voxel": mask_embed @ voxel_tokens.transpose(0, 1)
                     if voxel_tokens.shape[0] > 0
                     else mask_embed.new_zeros(queries.shape[0], 0),
            "fragment": mask_embed @ fragment_tokens.transpose(0, 1)
                        if fragment_tokens.shape[0] > 0
                        else mask_embed.new_zeros(queries.shape[0], 0),
            "spacepoint": mask_embed @ spacepoint_tokens.transpose(0, 1)
                          if spacepoint_tokens.shape[0] > 0
                          else mask_embed.new_zeros(queries.shape[0], 0),
        }
        return {
            "class_logits": out["class_logits"],
            "origin": out["origin"],
            "mask_logits": mask_logits,
        }

    def _build_attn_mask(
        self,
        prev_mask_logits: Optional[torch.Tensor],  # (Q, K) or None
    ) -> Optional[torch.Tensor]:
        """Mask out (q, k) where prev_mask_logits[q, k] <= mask_threshold.
        Mask2Former safety net: any query that ends up masking ALL keys gets
        its mask cleared so MHA softmax doesn't NaN.
        """
        if prev_mask_logits is None or prev_mask_logits.shape[1] == 0:
            return None
        attn_mask = (prev_mask_logits.detach() <= self.mask_threshold)  # (Q, K)
        # Clear queries that mask everything
        all_masked = attn_mask.all(dim=-1)
        if all_masked.any():
            attn_mask = attn_mask.clone()
            attn_mask[all_masked] = False
        return attn_mask

    # -- Forward --------------------------------------------------------

    def forward(
        self,
        voxel_tokens: torch.Tensor,        # (V, D)
        fragment_tokens: torch.Tensor,     # (F, D)
        spacepoint_tokens: torch.Tensor,   # (N, D)
    ) -> dict:
        """
        Returns dict with:
            init: per-layer-style dict from before any decoder layer
            layers: list of L per-layer dicts (deep supervision)
            final: alias for layers[-1]

        Each dict has: class_logits (Q, C), origin (Q, 3),
        mask_logits {voxel, fragment, spacepoint}.
        """
        Q = self.num_queries
        D = self.dim

        # (1, Q, D) for MHA batch_first convention
        queries = self.query_content.unsqueeze(0).clone()

        # Init predictions: before any decoder layer. Used both as the gate
        # for the first cross-attention (mask logits → attn mask) and as
        # auxiliary deep-supervision target (Mask2Former does this).
        init_predictions = self._compute_predictions(
            self.init_heads, queries.squeeze(0),
            voxel_tokens, fragment_tokens, spacepoint_tokens,
        )
        last_predictions = init_predictions

        per_layer = []
        for li, scale in enumerate(self.scale_pattern):
            keys = {
                "voxel": voxel_tokens,
                "fragment": fragment_tokens,
                "spacepoint": spacepoint_tokens,
            }[scale]
            if keys.shape[0] == 0:
                # Nothing to attend to at this scale — skip cross-attention
                # but still run self-attn + FFN by giving the layer a 1-key
                # zero key set is awkward; simplest: pass a 1-element zero
                # key as a no-op. Cleanest: skip the layer's cross-attn
                # entirely. We synthesize a dummy 1-key set.
                keys = queries.new_zeros(1, D)
                attn_mask = None
            else:
                attn_mask = self._build_attn_mask(
                    last_predictions["mask_logits"][scale]
                )

            keys_b = keys.unsqueeze(0)  # (1, K, D)
            query_pe = self.query_pe(last_predictions["origin"])
            queries = self.layers[li](
                queries, query_pe, keys_b, attn_mask=attn_mask,
            )

            # Compute predictions for this layer
            preds = self._compute_predictions(
                self.layer_heads[li], queries.squeeze(0),
                voxel_tokens, fragment_tokens, spacepoint_tokens,
            )
            preds["scale"] = scale
            per_layer.append(preds)
            last_predictions = preds

        return {
            "init": init_predictions,
            "layers": per_layer,
            "final": per_layer[-1] if per_layer else init_predictions,
        }
