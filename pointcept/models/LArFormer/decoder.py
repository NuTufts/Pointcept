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

import math
from collections import OrderedDict
from typing import Optional, Sequence

import torch
import torch.nn as nn

from .builders import LevelOutput


def _with_pos(x: torch.Tensor, pos: Optional[torch.Tensor]) -> torch.Tensor:
    return x if pos is None else x + pos


class SinusoidalPosEmb3D(nn.Module):
    """NeRF-style fixed sinusoidal position embedding for 3D coords.

    For each of the 3 input axes, emits `num_freq` (sin, cos) pairs at log-
    spaced frequencies in [1, max_freq], so the raw embedding width is
    `3 * 2 * num_freq`. A final Linear projects to `dim` (no nonlinearity:
    the embedding itself is the nonlinear part). Inputs are assumed to be
    pre-normalized to roughly [-1, 1] (LArFormerDataset's coord_norm).

    Unlike the MLP pos_emb, this gives every coord a unique, spatially
    structured signature out of the box, which lets queries learn position-
    specific behavior (e.g. distinguishing two content-similar tracks on
    opposite sides of the detector) without depending on a learned MLP that
    can collapse to mirror-symmetric features early in training.
    """

    def __init__(self, dim: int,
                 num_freq: Optional[int] = None,
                 max_freq: float = 256.0):
        super().__init__()
        if num_freq is None:
            num_freq = max(1, dim // 12)  # 3 axes * 2 (sin,cos) * num_freq ≈ dim/2
        freqs = 2.0 ** torch.linspace(
            0.0, math.log2(float(max_freq)), int(num_freq)
        )
        self.register_buffer("freqs", freqs, persistent=False)
        self.num_freq = int(num_freq)
        self.raw_dim = 3 * 2 * self.num_freq
        self.proj = nn.Linear(self.raw_dim, int(dim))

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: (..., 3)  →  (..., dim)
        if coords.shape[-1] != 3:
            raise ValueError(
                f"SinusoidalPosEmb3D expects last dim == 3, got "
                f"shape {tuple(coords.shape)}"
            )
        # Broadcast: (..., 3, 1) * (F,) → (..., 3, F)
        scaled = coords.unsqueeze(-1) * self.freqs  # (..., 3, F)
        sin = torch.sin(scaled)
        cos = torch.cos(scaled)
        # Interleave to (..., 3*2*F) without losing axis identity
        emb = torch.stack([sin, cos], dim=-1)        # (..., 3, F, 2)
        emb = emb.flatten(-3)                        # (..., 3*F*2)
        return self.proj(emb)


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

        # DETR/Mask2Former zero-init on the OUTPUT projections of each
        # residual block. At init each block is `x + 0 = x` — the layer
        # passes queries through unchanged, so the deep decoder stack
        # produces stable activations and the random-init "amplification
        # cascade" through N layers is avoided. Gradient still flows into
        # these zeroed weights via the residual connection.
        nn.init.zeros_(self.cross_attn.out_proj.weight)
        nn.init.zeros_(self.cross_attn.out_proj.bias)
        nn.init.zeros_(self.self_attn.out_proj.weight)
        nn.init.zeros_(self.self_attn.out_proj.bias)
        nn.init.zeros_(self.ffn[-1].weight)
        nn.init.zeros_(self.ffn[-1].bias)

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

        # DETR/Mask2Former-style zero-init on the OUTPUT projections.
        # Makes class_logits / mask_embed(q) / origin start at 0 at init,
        # so mask_logits = mask_embed(q) @ keys.T starts at 0 (sigmoid 0.5)
        # and softmax over class_logits is uniform. CE / BCE / Dice are
        # all finite and small. Gradient still flows (the matmuls are
        # nontrivial via the OTHER operand), so the model learns; it
        # just doesn't explode in the first iters.
        nn.init.zeros_(self.class_head.weight)
        nn.init.zeros_(self.class_head.bias)
        nn.init.zeros_(self.mask_embed[-1].weight)
        nn.init.zeros_(self.mask_embed[-1].bias)
        if self.enable_origin_head:
            nn.init.zeros_(self.origin_head[-1].weight)
            nn.init.zeros_(self.origin_head[-1].bias)

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
        pos_emb_kind:        "mlp" (default) — learnable (3)→(D) MLP, original
                              behavior; or "sinusoidal" — fixed log-spaced
                              NeRF-style frequencies + a single Linear proj.
                              Sinusoidal gives every coord a unique, spatially
                              structured signature, breaking the mirror-
                              symmetry pathology where two content-similar
                              tracks far apart get bound to the same query.
        pos_emb_hidden_dim:  hidden dim of the shared (3)→(D) pos-emb MLP
                              (only used when pos_emb_kind == "mlp")
        pos_emb_num_freq:    number of log-spaced frequencies per axis (only
                              used when pos_emb_kind == "sinusoidal"). Default
                              picks dim // 12 so the raw embedding ≈ dim/2.
        pos_emb_max_freq:    highest frequency (only used when sinusoidal).
                              Default 256 covers a [-1, 1]-normalized coord.
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
        pos_emb_kind: str = "mlp",
        pos_emb_hidden_dim: Optional[int] = None,
        pos_emb_num_freq: Optional[int] = None,
        pos_emb_max_freq: float = 256.0,
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

        pos_emb_kind = str(pos_emb_kind).lower()
        if pos_emb_kind not in ("mlp", "sinusoidal"):
            raise ValueError(
                f"pos_emb_kind must be 'mlp' or 'sinusoidal'; "
                f"got {pos_emb_kind!r}"
            )
        self.pos_emb_kind = pos_emb_kind
        if pos_emb_kind == "mlp":
            if pos_emb_hidden_dim is None:
                pos_emb_hidden_dim = dim
            self.pos_emb = nn.Sequential(
                nn.Linear(3, pos_emb_hidden_dim),
                nn.GELU(),
                nn.Linear(pos_emb_hidden_dim, dim),
            )
        else:
            self.pos_emb = SinusoidalPosEmb3D(
                dim, num_freq=pos_emb_num_freq, max_freq=pos_emb_max_freq,
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

    # Two-tier sanitization:
    #   _LOGIT_CAP        — tight cap on FINAL mask/class logits. sigmoid
    #                       / softmax saturate well before ±50, so this is
    #                       a no-op for normal training but bounds the
    #                       loss's view of pathological outputs.
    #   _INTERMEDIATE_CAP — generous cap on INTERMEDIATE tensors that go
    #                       into matmuls (mask_embed, keys_pe). The matmul
    #                       backward is `∂(A@B.T)/∂A = B`, so if B has Inf
    #                       entries, A's gradient gets Inf even if the
    #                       output is sanitized. Sanitizing INPUTS to the
    #                       matmul keeps the backward gradient finite.
    #                       1000 is large enough that normal training
    #                       (post-warmup, ~unit-scale activations) is
    #                       unaffected, but caps anything that's exploding
    #                       in the random-init regime.
    _LOGIT_CAP = 50.0
    _INTERMEDIATE_CAP = 1000.0

    @staticmethod
    def _nan_to_num_then_clamp(x: torch.Tensor, cap: float) -> torch.Tensor:
        # Order matters: nan_to_num FIRST, then clamp. If we clamp first,
        # clamp(NaN) = NaN, and then in the backward pass autograd computes
        # 0 (nan_to_num grad at NaN) × NaN (clamp grad at NaN) = NaN —
        # which propagates back into the optimizer state. Doing nan_to_num
        # first replaces NaN/Inf with finite values BEFORE the clamp ever
        # sees them.
        return torch.nan_to_num(
            x, nan=0.0, posinf=cap, neginf=-cap,
        ).clamp(-cap, cap)

    def _sanitize_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Sanitize a FINAL logit tensor (mask / class) — tight cap."""
        return self._nan_to_num_then_clamp(x, self._LOGIT_CAP)

    def _sanitize_intermediate(self, x: torch.Tensor) -> torch.Tensor:
        """Sanitize an INTERMEDIATE tensor that will go into a matmul —
        generous cap. Bounds the gradient produced by the matmul's
        backward (∂(A@B.T)/∂A = B) so a single Inf entry in one operand
        doesn't NaN the other operand's gradient."""
        return self._nan_to_num_then_clamp(x, self._INTERMEDIATE_CAP)

    def _compute_predictions(
        self,
        heads: _PerLayerHeads,
        queries: torch.Tensor,
        keys_pe_by_level: "OrderedDict[str, torch.Tensor]",
    ) -> dict:
        out = heads(queries)
        # Sanitize the matmul INPUTS so backward gradients don't propagate
        # Inf/NaN. See _sanitize_intermediate docstring.
        mask_embed = self._sanitize_intermediate(out["mask_embed"])
        Q = queries.shape[0]
        mask_logits = OrderedDict()
        for name, keys_pe in keys_pe_by_level.items():
            if keys_pe.shape[0] == 0:
                mask_logits[name] = mask_embed.new_zeros(Q, 0)
            else:
                keys_pe_clean = self._sanitize_intermediate(keys_pe)
                mask_logits[name] = self._sanitize_logits(
                    mask_embed @ keys_pe_clean.transpose(0, 1)
                )
        return {
            "class_logits": self._sanitize_logits(out["class_logits"]),
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
        init_query_content: Optional[torch.Tensor] = None,
        init_anchor_coords: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args (mixed query selection, both required to enable):
            init_query_content: (Q, D) token features selected by an
                external MixedQuerySelector. When provided, the decoder
                treats `self.query_content` as a per-slot zero-init delta
                ON TOP of these anchor features.
            init_anchor_coords: (Q, 3) coord_norm-space anchor positions
                from the same selector. The decoder adds
                `pos_emb(anchor_coords)` to `self.query_pos` so each
                query's positional prior starts at its anchor.

        When both are None (default), behavior matches the original
        pure learnable-query path: queries = self.query_content,
        query_pos = self.query_pos.
        """
        self._validate_levels(levels)

        D = self.dim
        if init_query_content is not None:
            # Mixed query selection: queries = anchor features +
            # (zero-init when selector is active) self.query_content delta.
            queries = (init_query_content + self.query_content).unsqueeze(0).clone()
        else:
            queries = self.query_content.unsqueeze(0).clone()

        # Sanitize input tokens ONCE at the decoder entry point. Every
        # downstream consumer (cross-attention `keys`, `keys_pe` for
        # mask-logit matmul, the LayerNorms inside each decoder layer)
        # then sees clean values. Without this, NaN/Inf in `lvl.tokens`
        # poisons `LayerNorm(keys)` inside `_MaskedDecoderLayer` →
        # NaN attention → NaN backward gradient at any param upstream
        # of those tokens.
        levels = OrderedDict(
            (name, LevelOutput(
                tokens=(self._sanitize_intermediate(lvl.tokens)
                        if lvl.tokens.shape[0] > 0 else lvl.tokens),
                coords=lvl.coords,
                sp_to_level_id=lvl.sp_to_level_id,
                name=lvl.name,
            ))
            for name, lvl in levels.items()
        )

        # Compute the shared per-level positional embedding once.
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

        # Mixed-query anchor positional contribution: when an external
        # selector supplied per-query anchor coords, embed them once and
        # add into every layer's `query_pos_dyn`. With self.query_pos
        # zero-init'd by LArFormer (when mixed query selection is active),
        # this gives each query a positional prior centered on its anchor;
        # the origin head's per-layer refinement (and any nonzero delta in
        # self.query_pos) is then a learnable correction on top.
        anchor_pe = (self.pos_emb(init_anchor_coords)
                     if init_anchor_coords is not None else None)

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

            query_pos_dyn = self.query_pos
            if anchor_pe is not None:
                query_pos_dyn = query_pos_dyn + anchor_pe
            if self.enable_origin_head:
                query_pos_dyn = query_pos_dyn + self.pos_emb(
                    last_predictions["origin"]
                )

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
