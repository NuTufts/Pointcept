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


def build_coord_pos_emb(kind: Optional[str], out_dim: int,
                        num_freq: Optional[int] = None,
                        max_freq: float = 256.0,
                        hidden_dim: Optional[int] = None,
                        zero_init: bool = False) -> Optional[nn.Module]:
    """Build a 3D-coordinate position-embedding module (or None).

    kind:
        None / "" / "none" — no pos-emb (returns None).
        "sinusoidal"       — fixed NeRF-style sin/cos (SinusoidalPosEmb3D); a
                             unique, spatially-structured signature per coord,
                             no symmetry-collapse risk. Recommended.
        "mlp"              — learned Linear(3->hidden)->GELU->Linear(hidden->out).
    out_dim: embedding width emitted (dense heads concatenate it onto the
        per-point feature; the query heads add it to the query, so out_dim
        there must equal the token dim).
    zero_init: zero the OUTPUT projection so the module emits 0 at init
        (identity-at-init). Use when the embedding is ADDED to an existing
        signal (the query heads) so adding the pos-emb doesn't shock a loaded
        head; the module learns away from zero. Leave False when the embedding
        is CONCATENATED (dense heads) and a live signal is wanted from iter 1.
    """
    if not kind or str(kind).lower() == "none":
        return None
    if kind == "sinusoidal":
        m = SinusoidalPosEmb3D(out_dim, num_freq=num_freq, max_freq=max_freq)
        out_lin = m.proj
    elif kind == "mlp":
        h = int(hidden_dim) if hidden_dim else int(out_dim)
        m = nn.Sequential(nn.Linear(3, h), nn.GELU(), nn.Linear(h, int(out_dim)))
        out_lin = m[-1]
    else:
        raise ValueError(f"unknown coord pos-emb kind {kind!r}")
    if zero_init:
        nn.init.zeros_(out_lin.weight)
        nn.init.zeros_(out_lin.bias)
    return m


class _MaskedDecoderLayer(nn.Module):
    """Pre-norm masked-cross-attn → self-attn → FFN. DETR-style PE on Q/K only."""

    def __init__(self, dim: int, num_heads: int,
                 mlp_ratio: float = 4.0, dropout: float = 0.0,
                 zero_init_output_proj: bool = False):
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

        # Opt-in zero-init on the OUTPUT projections of each residual
        # block. When True: each block is `x + 0 = x` at init — the
        # layer passes queries through unchanged, so the deep decoder
        # stack produces stable activations and avoids the random-init
        # "amplification cascade" through N layers. Gradient still
        # flows into these zeroed weights via the residual connection,
        # but symmetry-breaking is slow — empirically costs convergence
        # rate on small training sets / refiner-only configs where
        # standard random init is already stable. Recommended ON for
        # configs that train the PTv3 decoder from scratch (where the
        # cascade was actually producing NaNs); OFF (default) restores
        # the original random init.
        if zero_init_output_proj:
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
        self_attn_mask: Optional[torch.Tensor] = None,
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
        so, _ = self.self_attn(
            s_q, s_k, s_n, attn_mask=self_attn_mask, need_weights=False,
        )
        queries = queries + so

        queries = queries + self.ffn(self.ffn_norm(queries))
        return queries


class _PerLayerHeads(nn.Module):
    """Class logits + optional origin head + shared mask embed.

    When `enable_keypoint_head` is set (Phase-2 keypoint module, requires
    `enable_origin_head`), also emits, per query: `end` (3D end point),
    `start_logvar`/`end_logvar` (per-axis log-variance for β-NLL), and
    `end_logit` (track-end existence gate). The `origin` head IS the start
    point under this mode.
    """

    def __init__(self, dim: int, num_classes: int,
                 enable_origin_head: bool = True,
                 zero_init_output_proj: bool = False,
                 enable_keypoint_head: bool = False,
                 keypoint_pos_emb_kind: Optional[str] = None,
                 keypoint_pos_emb_num_freq: Optional[int] = None,
                 keypoint_pos_emb_max_freq: float = 256.0,
                 keypoint_pos_emb_hidden_dim: Optional[int] = None):
        super().__init__()
        self.class_head = nn.Linear(dim, num_classes)
        self.enable_origin_head = bool(enable_origin_head)
        if self.enable_origin_head:
            self.origin_head = nn.Sequential(
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Linear(dim, 3),
            )
        self.enable_keypoint_head = bool(enable_keypoint_head)
        if self.enable_keypoint_head:
            if not self.enable_origin_head:
                raise ValueError(
                    "enable_keypoint_head requires enable_origin_head "
                    "(the origin head supplies the keypoint 'start')."
                )

            def _mlp(out_dim):
                return nn.Sequential(
                    nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, out_dim))
            self.end_head = _mlp(3)
            self.start_logvar_head = _mlp(3)
            self.end_logvar_head = _mlp(3)
            self.end_exist_head = nn.Linear(dim, 1)
            # Start uncertainty at logvar≈0 (var≈1).
            nn.init.zeros_(self.start_logvar_head[-1].weight)
            nn.init.zeros_(self.start_logvar_head[-1].bias)
            nn.init.zeros_(self.end_logvar_head[-1].weight)
            nn.init.zeros_(self.end_logvar_head[-1].bias)
            # Opt-in: add a pos-emb of the query's anchor coord (the prior
            # layer's predicted origin, or the mixed-query anchor at init) to
            # the query content BEFORE the keypoint heads (DAB/DN-DETR style),
            # so the start/end regression conditions on "where am I now". Only
            # the keypoint heads (origin/end/logvar/exist) use it — class /
            # mask_embed stay on the raw query so the (frozen) mask path is
            # untouched. Off by default (kp_pos_emb=None). Named `kp_pos_emb`
            # so it's caught by the keypoint-only freeze marker list.
            self.kp_pos_emb = build_coord_pos_emb(
                keypoint_pos_emb_kind, dim,
                num_freq=keypoint_pos_emb_num_freq,
                max_freq=keypoint_pos_emb_max_freq,
                hidden_dim=keypoint_pos_emb_hidden_dim,
                zero_init=True)  # ADDED to the query → identity at init
        else:
            self.kp_pos_emb = None
        self.mask_embed = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

        # Opt-in zero-init on the OUTPUT projections. When True:
        # class_logits = 0 (softmax uniform), mask_embed(q) = 0 so
        # mask_logits = 0 (sigmoid 0.5), origin = 0. CE / BCE / Dice
        # are finite and uniform across queries — useful for stabilizing
        # from-scratch decoders that would otherwise produce wildly out-
        # of-distribution logits in the first iters. But it makes all
        # queries indistinguishable through the heads at init, so the
        # Hungarian matcher has only random tie-breaking and per-query
        # specialization is slow. OFF (default) keeps the original
        # random init — recommended unless a from-scratch decoder is
        # producing NaN/Inf.
        if zero_init_output_proj:
            nn.init.zeros_(self.class_head.weight)
            nn.init.zeros_(self.class_head.bias)
            nn.init.zeros_(self.mask_embed[-1].weight)
            nn.init.zeros_(self.mask_embed[-1].bias)
            if self.enable_origin_head:
                nn.init.zeros_(self.origin_head[-1].weight)
                nn.init.zeros_(self.origin_head[-1].bias)

    def forward(self, queries: torch.Tensor,
                anchor_coords: Optional[torch.Tensor] = None) -> dict:
        # `origin` is the ONLY keypoint output that feeds back into the (frozen)
        # decoder: the next layer adds pos_emb(origin) to query_pos, so routing
        # origin through the keypoint pos-emb would let it perturb the frozen
        # masks (observed: val IoU dip). So origin + class + mask_embed always
        # use the RAW query; only the non-feedback keypoint heads (end /
        # logvars / end-exist — the ones that actually lack positional info)
        # get the anchor pos-emb. This keeps the query pos-emb mask-safe.
        q_kp = queries
        if (self.enable_keypoint_head and self.kp_pos_emb is not None
                and anchor_coords is not None):
            q_kp = queries + self.kp_pos_emb(anchor_coords)
        origin = (self.origin_head(queries) if self.enable_origin_head
                  else queries.new_zeros(queries.shape[0], 3))
        out = {
            "class_logits": self.class_head(queries),
            "origin":       origin,
            "mask_embed":   self.mask_embed(queries),
        }
        if self.enable_keypoint_head:
            out["end"] = self.end_head(q_kp)
            out["start_logvar"] = self.start_logvar_head(q_kp)
            out["end_logvar"] = self.end_logvar_head(q_kp)
            out["end_logit"] = self.end_exist_head(q_kp).squeeze(-1)
        return out


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
        zero_init_output_proj: bool = False,
        enable_keypoint_head: bool = False,
        # Opt-in anchor pos-emb on the per-query keypoint heads (off by
        # default). See _PerLayerHeads. Distinct from pos_emb_* above, which
        # is the query_pos / mask-key embedding shared with the frozen path.
        keypoint_pos_emb_kind: Optional[str] = None,
        keypoint_pos_emb_num_freq: Optional[int] = None,
        keypoint_pos_emb_max_freq: float = 256.0,
        keypoint_pos_emb_hidden_dim: Optional[int] = None,
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
        self.enable_keypoint_head = bool(enable_keypoint_head)
        self._kp_head_kw = dict(
            keypoint_pos_emb_kind=keypoint_pos_emb_kind,
            keypoint_pos_emb_num_freq=keypoint_pos_emb_num_freq,
            keypoint_pos_emb_max_freq=keypoint_pos_emb_max_freq,
            keypoint_pos_emb_hidden_dim=keypoint_pos_emb_hidden_dim,
        )

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

        self.zero_init_output_proj = bool(zero_init_output_proj)
        self.init_heads = _PerLayerHeads(
            dim, num_classes, enable_origin_head=self.enable_origin_head,
            zero_init_output_proj=self.zero_init_output_proj,
            enable_keypoint_head=self.enable_keypoint_head,
            **self._kp_head_kw,
        )
        self.layers = nn.ModuleList([
            _MaskedDecoderLayer(
                dim, num_heads, mlp_ratio=mlp_ratio, dropout=dropout,
                zero_init_output_proj=self.zero_init_output_proj,
            )
            for _ in self.scale_pattern
        ])
        self.layer_heads = nn.ModuleList([
            _PerLayerHeads(
                dim, num_classes,
                enable_origin_head=self.enable_origin_head,
                zero_init_output_proj=self.zero_init_output_proj,
                enable_keypoint_head=self.enable_keypoint_head,
                **self._kp_head_kw,
            )
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
        anchor_coords: Optional[torch.Tensor] = None,
    ) -> dict:
        out = heads(queries, anchor_coords)
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
        result = {
            "class_logits": self._sanitize_logits(out["class_logits"]),
            "origin":       out["origin"],
            "mask_logits":  mask_logits,
            # Raw per-query embedding (Q, D) — consumed by the Phase-2
            # KeypointQueryHead. Carried on every layer dict so the DN
            # slice (model._slice_decoder_output) splits it correctly.
            "query_embed":  queries,
        }
        # Phase-2 keypoint head outputs (start = `origin` above). Carried on
        # every layer dict; sliced DN-safe in model._slice_decoder_output.
        for k in ("end", "start_logvar", "end_logvar", "end_logit"):
            if k in out:
                result[k] = out[k]
        return result

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
        dn_self_attn_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args (mixed query selection, both required to enable):
            init_query_content: (Q_total, D) token features selected by
                an external MixedQuerySelector. When provided, the
                decoder treats `self.query_content` as a per-slot
                zero-init delta ON TOP of these anchor features.

                Mask denoising appends DN queries past the regular K:
                `init_query_content[K:]` are DN queries, and the
                learnable `self.query_content` is only applied to the
                first K slots (DN queries pass through unchanged from
                their class-embedding init).
            init_anchor_coords: (Q_total, 3) coord_norm-space anchor
                positions from the same selector. The decoder adds
                `pos_emb(anchor_coords)` to `self.query_pos` (zero-
                padded past K) so each query's positional prior starts
                at its anchor.
            dn_self_attn_mask: optional (Q_total, Q_total) bool. When
                provided, applied as `self_attn`'s attn_mask in every
                layer to block DN ↔ regular and DN-cross-group edges.
                See `MaskDenoiser.build_self_attn_mask`.

        When init_query_content is None (default), behavior matches the
        original pure learnable-query path: queries = self.query_content,
        query_pos = self.query_pos, no self-attn mask.
        """
        self._validate_levels(levels)

        D = self.dim
        K_reg = self.query_content.shape[0]
        if init_query_content is not None:
            Q_total = int(init_query_content.shape[0])
            # Compose queries: init content + learnable delta on the first
            # K_reg slots only. DN slots (the trailing Q_total - K_reg)
            # pass through their class-embedding init unchanged.
            queries = init_query_content.clone()
            n_delta = min(Q_total, K_reg)
            if n_delta > 0:
                queries = queries.clone()
                queries[:n_delta] = queries[:n_delta] + self.query_content[:n_delta]
            queries = queries.unsqueeze(0)
        else:
            Q_total = K_reg
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

        # Per-layer base query_pos. self.query_pos has shape (K_reg, D);
        # when DN queries pad init_query_content past K_reg, we zero-pad
        # the trailing slots — DN queries get all their positional prior
        # from anchor_pe (and, if enabled, the origin head's per-layer
        # refinement), not from self.query_pos.
        if Q_total != K_reg:
            query_pos_base = self.query_pos.new_zeros(Q_total, D)
            n_apply = min(Q_total, K_reg)
            if n_apply > 0:
                query_pos_base[:n_apply] = self.query_pos[:n_apply]
        else:
            query_pos_base = self.query_pos

        init_predictions = self._compute_predictions(
            self.init_heads, queries.squeeze(0), keys_pe_by_level,
            anchor_coords=init_anchor_coords,
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

            query_pos_dyn = query_pos_base
            if anchor_pe is not None:
                query_pos_dyn = query_pos_dyn + anchor_pe
            if self.enable_origin_head:
                query_pos_dyn = query_pos_dyn + self.pos_emb(
                    last_predictions["origin"]
                )

            queries = self.layers[li](
                queries, query_pos_dyn, keys_b, key_pos,
                attn_mask=attn_mask,
                self_attn_mask=dn_self_attn_mask,
            )

            preds = self._compute_predictions(
                self.layer_heads[li], queries.squeeze(0), keys_pe_by_level,
                anchor_coords=last_predictions["origin"],
            )
            preds["scale"] = scale
            per_layer.append(preds)
            last_predictions = preds

        return {
            "init":   init_predictions,
            "layers": per_layer,
            "final":  per_layer[-1] if per_layer else init_predictions,
        }
