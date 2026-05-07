"""
Mask2Former-style decoder for the shower-clustering model. Phase 4 of the
design (see pointcept/docs/shower_clustering_design.md §3, Phase 4).

Inputs (per event, no batch dim — model wraps batching) — for each scale we
get tokens AND their normalized 3D coords. The decoder owns position
embedding via a shared (3)→(D) MLP applied to every scale and to queries:

    voxel_tokens       (V, D)    voxel_coords       (V, 3)
    fragment_tokens    (F, D)    fragment_coords    (F, 3)
    spacepoint_tokens  (N, D)    spacepoint_coords  (N, 3)

Outputs (list of L layer dicts):
    class_logits       (Q, num_classes) — per-query class score
                                          (last entry is "no object")
    origin             (Q, 3)          — per-query predicted origin
                                          (normalized coord)
    mask_logits["voxel"]      (Q, V)
    mask_logits["fragment"]   (Q, F)
    mask_logits["spacepoint"] (Q, N)

Architecture per layer:
    1. masked cross-attention to ONE scale (rotating pattern), with PE
       added on Q and K but NOT V (DETR canonical pattern)
    2. query self-attention (PE on Q+K, not V)
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

Position embedding (refactor 2026-05-06):
    self.pos_emb : (3) → (D)  shared MLP, applied to:
        - per-spacepoint coords  → key positional embedding for sp scale
        - per-voxel center coords → key positional embedding for voxel scale
        - per-fragment centroids  → key positional embedding for fragment scale
        - per-query predicted origin → dynamic component of query pos
    self.query_pos : (Q, D)  learnable per-query "slot identity" embedding,
                              added to the dynamic origin embedding.

    Mask head is also position-aware:
        mask_logits = mask_embed(q) @ (tokens + pos_emb(coords)).T
"""

from typing import Optional, Sequence

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Single decoder layer: masked-cross-attn → self-attn → FFN
#                        (DETR-style: PE added on Q and K only, NOT V)
# ---------------------------------------------------------------------------

def _with_pos(x: torch.Tensor, pos: Optional[torch.Tensor]) -> torch.Tensor:
    """DETR's `with_pos_embed`: add positional embedding if present."""
    return x if pos is None else x + pos


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
        query_pos: torch.Tensor,  # (Q, D)
        keys: torch.Tensor,       # (1, K, D)
        key_pos: Optional[torch.Tensor],  # (K, D) or None
        attn_mask: Optional[torch.Tensor] = None,  # (Q, K) bool, True=mask
    ) -> torch.Tensor:
        # Pre-norm both sides separately, then add PE for Q/K but NOT V.
        q_normed = self.cross_norm_q(queries)
        k_normed = self.cross_norm_k(keys)
        q = _with_pos(q_normed, query_pos.unsqueeze(0))
        if key_pos is not None:
            k = _with_pos(k_normed, key_pos.unsqueeze(0))
        else:
            k = k_normed
        v = k_normed
        cross_out, _ = self.cross_attn(
            q, k, v,
            attn_mask=attn_mask,
            need_weights=False,
        )
        queries = queries + cross_out

        # Query self-attention (also gets PE on Q+K, not V)
        s_normed = self.self_norm(queries)
        s_q = _with_pos(s_normed, query_pos.unsqueeze(0))
        s_k = _with_pos(s_normed, query_pos.unsqueeze(0))
        s_v = s_normed
        self_out, _ = self.self_attn(s_q, s_k, s_v, need_weights=False)
        queries = queries + self_out

        # FFN
        queries = queries + self.ffn(self.ffn_norm(queries))
        return queries


# ---------------------------------------------------------------------------
# Per-layer prediction heads (one set, applied at every layer)
# ---------------------------------------------------------------------------

class _PerLayerHeads(nn.Module):
    """Class logits, origin coord, and (shared) mask embed.

    Per-scale mask logits are computed externally as
    mask_embed(q) @ (key + key_pos).T; this module returns the embedding only.
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
        pos_emb_hidden_dim: hidden width of the shared (3)→(dim) position-
                            embedding MLP. Defaults to `dim` if None.
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
        pos_emb_hidden_dim: Optional[int] = None,
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

        # Per-query learnable "slot identity" embedding. Combined with
        # pos_emb(predicted_origin) to form the dynamic query_pos at each layer.
        self.query_pos = nn.Parameter(torch.empty(num_queries, dim))
        nn.init.trunc_normal_(self.query_pos, std=0.02)

        # Shared (3) → (D) positional-embedding MLP. Applied to:
        #   - per-spacepoint / voxel-center / fragment-centroid coords
        #     (key-side PE for cross-attention and for the mask head)
        #   - per-query predicted origin (dynamic component of query_pos)
        if pos_emb_hidden_dim is None:
            pos_emb_hidden_dim = dim
        self.pos_emb = nn.Sequential(
            nn.Linear(3, pos_emb_hidden_dim),
            nn.GELU(),
            nn.Linear(pos_emb_hidden_dim, dim),
        )

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
        voxel_keys_pe: torch.Tensor,    # (V, D)  = voxel_tokens + voxel_pos
        fragment_keys_pe: torch.Tensor, # (F, D)  = fragment_tokens + fragment_pos
        spacepoint_keys_pe: torch.Tensor,  # (N, D) = sp_tokens + sp_pos
    ) -> dict:
        """Mask head is position-aware: dot mask_embed(q) against (tokens+pos)."""
        out = heads(queries)
        mask_embed = out["mask_embed"]
        Q = queries.shape[0]
        if voxel_keys_pe.shape[0] > 0:
            mask_voxel = mask_embed @ voxel_keys_pe.transpose(0, 1)
        else:
            mask_voxel = mask_embed.new_zeros(Q, 0)
        if fragment_keys_pe.shape[0] > 0:
            mask_frag = mask_embed @ fragment_keys_pe.transpose(0, 1)
        else:
            mask_frag = mask_embed.new_zeros(Q, 0)
        if spacepoint_keys_pe.shape[0] > 0:
            mask_sp = mask_embed @ spacepoint_keys_pe.transpose(0, 1)
        else:
            mask_sp = mask_embed.new_zeros(Q, 0)
        return {
            "class_logits": out["class_logits"],
            "origin": out["origin"],
            "mask_logits": {
                "voxel": mask_voxel,
                "fragment": mask_frag,
                "spacepoint": mask_sp,
            },
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
        voxel_coords: torch.Tensor,        # (V, 3)
        fragment_tokens: torch.Tensor,     # (F, D)
        fragment_coords: torch.Tensor,     # (F, 3)
        spacepoint_tokens: torch.Tensor,   # (N, D)
        spacepoint_coords: torch.Tensor,   # (N, 3)
    ) -> dict:
        """
        Returns dict with:
            init: per-layer-style dict from before any decoder layer
            layers: list of L per-layer dicts (deep supervision)
            final: alias for layers[-1]

        Each dict has: class_logits (Q, C), origin (Q, 3),
        mask_logits {voxel, fragment, spacepoint}.
        """
        D = self.dim

        # (1, Q, D) for MHA batch_first convention
        queries = self.query_content.unsqueeze(0).clone()

        # Per-scale key positional embedding. Computed once and reused at
        # every layer (cross-attn key_pos + position-aware mask head).
        voxel_pos = (self.pos_emb(voxel_coords)
                     if voxel_tokens.shape[0] > 0
                     else voxel_tokens.new_zeros(0, D))
        fragment_pos = (self.pos_emb(fragment_coords)
                        if fragment_tokens.shape[0] > 0
                        else fragment_tokens.new_zeros(0, D))
        spacepoint_pos = (self.pos_emb(spacepoint_coords)
                          if spacepoint_tokens.shape[0] > 0
                          else spacepoint_tokens.new_zeros(0, D))

        voxel_keys_pe = voxel_tokens + voxel_pos
        fragment_keys_pe = fragment_tokens + fragment_pos
        spacepoint_keys_pe = spacepoint_tokens + spacepoint_pos

        # Init predictions: before any decoder layer. Used both as the gate
        # for the first cross-attention (mask logits → attn mask) and as
        # auxiliary deep-supervision target (Mask2Former does this).
        init_predictions = self._compute_predictions(
            self.init_heads, queries.squeeze(0),
            voxel_keys_pe, fragment_keys_pe, spacepoint_keys_pe,
        )
        last_predictions = init_predictions

        per_layer = []
        for li, scale in enumerate(self.scale_pattern):
            if scale == "voxel":
                keys = voxel_tokens
                key_pos = voxel_pos
            elif scale == "fragment":
                keys = fragment_tokens
                key_pos = fragment_pos
            else:  # "spacepoint"
                keys = spacepoint_tokens
                key_pos = spacepoint_pos

            if keys.shape[0] == 0:
                # Nothing to attend to at this scale — synthesize a 1-key
                # dummy zero set so the layer still runs (no-op contribution).
                keys = queries.new_zeros(1, D)
                key_pos = queries.new_zeros(1, D)
                attn_mask = None
            else:
                attn_mask = self._build_attn_mask(
                    last_predictions["mask_logits"][scale]
                )

            keys_b = keys.unsqueeze(0)  # (1, K, D)

            # Dynamic query_pos = static slot identity + pos_emb(prev origin).
            # This feeds the previous layer's predicted origin coord back as
            # a positional cue for the next round of cross-attention.
            query_pos_dyn = self.query_pos + self.pos_emb(
                last_predictions["origin"]
            )

            queries = self.layers[li](
                queries, query_pos_dyn, keys_b, key_pos, attn_mask=attn_mask,
            )

            preds = self._compute_predictions(
                self.layer_heads[li], queries.squeeze(0),
                voxel_keys_pe, fragment_keys_pe, spacepoint_keys_pe,
            )
            preds["scale"] = scale
            per_layer.append(preds)
            last_predictions = preds

        return {
            "init": init_predictions,
            "layers": per_layer,
            "final": per_layer[-1] if per_layer else init_predictions,
        }
