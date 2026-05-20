"""
Mask2Former-style decoder generalized to an arbitrary, config-driven set of
named levels (see `Pointcept/docs/LArFormer.md` §3).

Inputs per forward (one event):
    levels: ordered dict[level_name → LevelOutput]
            (LevelOutput.tokens (M, D), LevelOutput.coords (M, 3))
    scale_pattern: list of level names — one entry per decoder layer; the
            named level supplies that layer's cross-attention keys/values.

Outputs:
    {
      "init":   <per-layer-style dict, before any decoder layer>,
      "layers": [<per-layer dict>, ...],
      "final":  alias for layers[-1] (or init if no layers).
    }
where each per-layer dict has:
    class_logits   (Q, C)
    origin         (Q, 3)            — zero-filled when origin head disabled
    mask_logits    {level_name: (Q, M_level), ...}  for every level in `levels`

Position embedding: shared (3)→(D) MLP applied to every level's coords and
to the per-query predicted origin (when the origin head is enabled). DETR-
style: PE is added to Q and K but NOT V. Mask head is position-aware via
`mask_embed(q) @ (tokens + pos_emb(coords)).T`.
"""

from collections import OrderedDict
from typing import Optional, Sequence

import torch
import torch.nn as nn

from .builders import LevelOutput


def _with_pos(x: torch.Tensor, pos: Optional[torch.Tensor]) -> torch.Tensor:
    return x if pos is None else x + pos


class _MaskedDecoderLayer(nn.Module):
    """Pre-norm masked-cross-attn → self-attn → FFN. DETR-style PE on Q/K only."""

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
        queries: torch.Tensor,
        query_pos: torch.Tensor,
        keys: torch.Tensor,
        key_pos: Optional[torch.Tensor],
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q_n = self.cross_norm_q(queries)
        k_n = self.cross_norm_k(keys)
        q = _with_pos(q_n, query_pos.unsqueeze(0))
        k = _with_pos(k_n, key_pos.unsqueeze(0)) if key_pos is not None else k_n
        v = k_n
        co, _ = self.cross_attn(q, k, v, attn_mask=attn_mask, need_weights=False)
        queries = queries + co

        s_n = self.self_norm(queries)
        s_q = _with_pos(s_n, query_pos.unsqueeze(0))
        s_k = _with_pos(s_n, query_pos.unsqueeze(0))
        so, _ = self.self_attn(s_q, s_k, s_n, need_weights=False)
        queries = queries + so

        queries = queries + self.ffn(self.ffn_norm(queries))
        return queries


class _PerLayerHeads(nn.Module):
    """Class logits + optional origin head + shared mask embed."""

    def __init__(self, dim: int, num_classes: int,
                 enable_origin_head: bool = True):
        super().__init__()
        self.class_head = nn.Linear(dim, num_classes)
        self.enable_origin_head = bool(enable_origin_head)
        if self.enable_origin_head:
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
        origin = (self.origin_head(queries) if self.enable_origin_head
                  else queries.new_zeros(queries.shape[0], 3))
        return {
            "class_logits": self.class_head(queries),
            "origin":       origin,
            "mask_embed":   self.mask_embed(queries),
        }


class Mask2FormerDecoder(nn.Module):
    """LArFormer's generalized Mask2Former decoder.

    Args:
        dim:                 token dim (must match the composite tokenizer's
                              token_dim and every builder's out dim)
        num_queries:         Q learnable queries
        num_classes:         total class slots (incl. no_object as the last)
        scale_pattern:       list[str] of level names — one per layer
        num_heads:           MHA heads
        mlp_ratio:           FFN expansion
        dropout:             applied to MHA + optional FFN
        mask_threshold:      pre-sigmoid threshold for mask-gating attn mask
        pos_emb_hidden_dim:  hidden dim of the shared (3)→(D) pos-emb MLP
        enable_origin_head:  if False, origin is identically 0 and skipped
                              from the dynamic query_pos
    """

    def __init__(
        self,
        dim: int,
        scale_pattern: Sequence[str],
        num_queries: int = 64,
        num_classes: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        mask_threshold: float = 0.0,
        pos_emb_hidden_dim: Optional[int] = None,
        enable_origin_head: bool = True,
    ):
        super().__init__()
        if len(scale_pattern) == 0:
            raise ValueError("scale_pattern must list at least one level")
        self.scale_pattern = tuple(str(s) for s in scale_pattern)
        self.dim = int(dim)
        self.num_queries = int(num_queries)
        self.num_classes = int(num_classes)
        self.mask_threshold = float(mask_threshold)
        self.enable_origin_head = bool(enable_origin_head)

        self.query_content = nn.Parameter(torch.empty(num_queries, dim))
        nn.init.trunc_normal_(self.query_content, std=0.02)
        self.query_pos = nn.Parameter(torch.empty(num_queries, dim))
        nn.init.trunc_normal_(self.query_pos, std=0.02)

        if pos_emb_hidden_dim is None:
            pos_emb_hidden_dim = dim
        self.pos_emb = nn.Sequential(
            nn.Linear(3, pos_emb_hidden_dim),
            nn.GELU(),
            nn.Linear(pos_emb_hidden_dim, dim),
        )

        self.init_heads = _PerLayerHeads(
            dim, num_classes, enable_origin_head=self.enable_origin_head,
        )
        self.layers = nn.ModuleList([
            _MaskedDecoderLayer(dim, num_heads, mlp_ratio=mlp_ratio,
                                dropout=dropout)
            for _ in self.scale_pattern
        ])
        self.layer_heads = nn.ModuleList([
            _PerLayerHeads(dim, num_classes,
                           enable_origin_head=self.enable_origin_head)
            for _ in self.scale_pattern
        ])

    # ------------------------------------------------------------------

    def _validate_levels(self, levels: "OrderedDict[str, LevelOutput]") -> None:
        missing = [s for s in self.scale_pattern if s not in levels]
        if missing:
            raise KeyError(
                f"scale_pattern references levels not present in the "
                f"tokenizer output: {missing!r}. Tokenizer levels: "
                f"{list(levels.keys())!r}"
            )

    def _compute_predictions(
        self,
        heads: _PerLayerHeads,
        queries: torch.Tensor,
        keys_pe_by_level: "OrderedDict[str, torch.Tensor]",
    ) -> dict:
        out = heads(queries)
        mask_embed = out["mask_embed"]
        Q = queries.shape[0]
        mask_logits = OrderedDict()
        for name, keys_pe in keys_pe_by_level.items():
            if keys_pe.shape[0] == 0:
                mask_logits[name] = mask_embed.new_zeros(Q, 0)
            else:
                mask_logits[name] = mask_embed @ keys_pe.transpose(0, 1)
        return {
            "class_logits": out["class_logits"],
            "origin":       out["origin"],
            "mask_logits":  mask_logits,
        }

    def _build_attn_mask(self, prev_mask_logits: torch.Tensor
                         ) -> Optional[torch.Tensor]:
        if prev_mask_logits.shape[1] == 0:
            return None
        am = (prev_mask_logits.detach() <= self.mask_threshold)
        all_masked = am.all(dim=-1)
        if all_masked.any():
            am = am.clone()
            am[all_masked] = False
        return am

    # ------------------------------------------------------------------

    def forward(
        self,
        levels: "OrderedDict[str, LevelOutput]",
    ) -> dict:
        self._validate_levels(levels)

        D = self.dim
        queries = self.query_content.unsqueeze(0).clone()

        # Compute the shared per-level positional embedding once.
        any_tokens = next(iter(levels.values())).tokens
        pos_by_level: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        keys_pe_by_level: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        for name, lvl in levels.items():
            if lvl.tokens.shape[0] == 0:
                pos_by_level[name] = lvl.tokens.new_zeros(0, D)
                keys_pe_by_level[name] = lvl.tokens.new_zeros(0, D)
            else:
                pe = self.pos_emb(lvl.coords)
                pos_by_level[name] = pe
                keys_pe_by_level[name] = lvl.tokens + pe

        init_predictions = self._compute_predictions(
            self.init_heads, queries.squeeze(0), keys_pe_by_level,
        )
        last_predictions = init_predictions

        per_layer = []
        for li, scale in enumerate(self.scale_pattern):
            lvl = levels[scale]
            if lvl.tokens.shape[0] == 0:
                # Synthesize a 1-key zero set so the layer still runs (a
                # no-op contribution to the queries; matches the legacy
                # ShowerClusteringMask2Former behavior).
                keys = queries.new_zeros(1, D)
                key_pos = queries.new_zeros(1, D)
                attn_mask = None
            else:
                keys = lvl.tokens
                key_pos = pos_by_level[scale]
                attn_mask = self._build_attn_mask(
                    last_predictions["mask_logits"][scale]
                )

            keys_b = keys.unsqueeze(0)

            if self.enable_origin_head:
                query_pos_dyn = self.query_pos + self.pos_emb(
                    last_predictions["origin"]
                )
            else:
                query_pos_dyn = self.query_pos

            queries = self.layers[li](
                queries, query_pos_dyn, keys_b, key_pos, attn_mask=attn_mask,
            )

            preds = self._compute_predictions(
                self.layer_heads[li], queries.squeeze(0), keys_pe_by_level,
            )
            preds["scale"] = scale
            per_layer.append(preds)
            last_predictions = preds

        return {
            "init":   init_predictions,
            "layers": per_layer,
            "final":  per_layer[-1] if per_layer else init_predictions,
        }
