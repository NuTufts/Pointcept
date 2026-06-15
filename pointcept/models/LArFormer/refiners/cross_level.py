"""CrossLevelAttn — Option 2 token refiner (full cross-attention).

For each target level, the refiner cross-attends its tokens against the
concatenated token soup of all source levels. This is the level-agnostic
analog of Mask2Former's pixel decoder, generalized over LArFormer's
flexible-levels abstraction (no hierarchical pool indices required —
levels are bridged purely through position embeddings on Q and K).

Per layer per target:
    Q  = target.tokens               (M_tgt, D)
    K  = concat(source_tokens)       (sum_M_src, D)
    V  = same as K
    Q += pos_emb(target.coords)
    K += pos_emb(concat source coords)
    out = Q + CrossAttn(Q, K, V)
    out = out + FFN(LayerNorm(out))
    target.tokens = out

A single `pos_emb` module is shared across all levels (coords are in the
same normalized frame). Target tokens UPDATED across layers feed back into
the source soup in subsequent layers when the target is also a source —
this is the same dynamic Mask2Former's pixel decoder uses to let scales
co-evolve.

Source-token cap:
    For very large source levels (typically `spacepoint` with ~50K tokens),
    the K/V tensor grows. `max_source_tokens_per_level` randomly subsamples
    a level's tokens when it's used as a source. This is a per-forward
    random subsample (acts as light dropout). Set to None to disable.
"""

from collections import OrderedDict
from typing import Optional, Sequence

import torch
import torch.nn as nn

from ..builders import LevelOutput
from .base import REFINERS, TokenRefiner
from .pos_emb import build_pos_emb


# ---------------------------------------------------------------------------
# Cross-attention + FFN block
# ---------------------------------------------------------------------------

class _CrossAttnFFNBlock(nn.Module):
    """Pre-norm cross-attn (pos_emb added to Q+K, identity on V) + FFN."""

    def __init__(self, dim: int, num_heads: int,
                 mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.q_norm = nn.LayerNorm(dim)
        self.k_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
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

        # DETR/Mask2Former zero-init on the OUTPUT projections. Each
        # residual block is identity at init → cross-level information
        # flow only ramps up as the model learns; the random-init
        # amplification cascade through N stacked blocks is avoided.
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)
        nn.init.zeros_(self.ffn[-1].weight)
        nn.init.zeros_(self.ffn[-1].bias)

    def forward(
        self,
        q_tokens: torch.Tensor,    # (M_q, D)
        q_pos:    torch.Tensor,    # (M_q, D)
        kv_tokens: torch.Tensor,   # (M_kv, D)
        k_pos:    torch.Tensor,    # (M_kv, D)
    ) -> torch.Tensor:
        # Add batch dim for MHA (batch_first=True takes (B, M, D)).
        q_n = self.q_norm(q_tokens).unsqueeze(0)
        k_n = self.k_norm(kv_tokens).unsqueeze(0)
        q = q_n + q_pos.unsqueeze(0)
        k = k_n + k_pos.unsqueeze(0)
        v = k_n                            # DETR-style: PE on Q+K, not V
        attn_out, _ = self.attn(q, k, v, need_weights=False)
        out = q_tokens + attn_out.squeeze(0)
        out_n = self.ffn_norm(out.unsqueeze(0)).squeeze(0)
        out = out + self.ffn(out_n)
        return out


# ---------------------------------------------------------------------------
# Refiner
# ---------------------------------------------------------------------------

@REFINERS.register_module()
class CrossLevelAttn(TokenRefiner):
    """Inter-level cross-attention refiner.

    Args:
        dim:              token dim — MUST match LArFormer's token_dim.
        num_layers:       cross-attn + FFN blocks per target level.
        num_heads:        MHA heads.
        mlp_ratio:        FFN expansion.
        dropout:          dropout on attn output + FFN.
        target_levels:    iterable of level names to refine (these get
                           updated tokens). Default None = "every
                           voxel_*" level (same heuristic as PerLevelSelfAttn).
                           Spacepoint level intentionally excluded by
                           default (would require updating ~50K tokens
                           per forward).
        source_levels:    iterable of level names whose tokens serve as
                           K/V for the cross-attention. Default None =
                           all levels (including spacepoint, so target
                           voxels can READ from per-SP features).
        levels_cfg:       auto-injected by LArFormer. Used to eagerly
                           build per-target blocks at __init__ (required
                           for DDP + post-construction .to(device)).
        max_source_tokens_per_level:
                           per-forward random subsample cap on each
                           source level's contribution to K/V. Default
                           None = no cap. Common value: 8192 (caps the
                           spacepoint level's contribution to a
                           manageable size).
        pos_emb_kind / pos_emb_hidden_dim / pos_emb_num_freq /
                pos_emb_max_freq:
                           shared (3)→(D) pos_emb forwarded to
                           `build_pos_emb`. ONE module is shared across
                           all levels (coords are in the same frame).
    """

    def __init__(
        self,
        dim: int,
        num_layers: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        target_levels: Optional[Sequence[str]] = None,
        source_levels: Optional[Sequence[str]] = None,
        levels_cfg: Optional[Sequence[dict]] = None,
        max_source_tokens_per_level: Optional[int] = None,
        pos_emb_kind: str = "mlp",
        pos_emb_hidden_dim: Optional[int] = None,
        pos_emb_num_freq: Optional[int] = None,
        pos_emb_max_freq: float = 256.0,
    ):
        super().__init__()
        self.dim = int(dim)
        self.num_layers = int(num_layers)
        self.target_levels = (None if target_levels is None
                              else tuple(str(s) for s in target_levels))
        self.source_levels = (None if source_levels is None
                              else tuple(str(s) for s in source_levels))
        self.max_source_tokens_per_level = (
            int(max_source_tokens_per_level)
            if max_source_tokens_per_level is not None else None
        )

        self._block_kwargs = dict(
            dim=self.dim, num_heads=int(num_heads),
            mlp_ratio=float(mlp_ratio), dropout=float(dropout),
        )

        # ONE shared pos_emb (positions are in the same coord_norm frame
        # for every level, so one module suffices).
        self.pos_emb = build_pos_emb(
            kind=pos_emb_kind, dim=self.dim,
            hidden_dim=pos_emb_hidden_dim,
            num_freq=pos_emb_num_freq, max_freq=pos_emb_max_freq,
        )

        # One stack of cross-attn blocks per target. Params NOT shared
        # across targets — different levels have very different token
        # statistics, and per-target weights let the model specialize the
        # voxel_20cm refinement from the voxel_5cm refinement.
        self.blocks_by_target = nn.ModuleDict()

        # Eager-build path: same rationale as PerLevelSelfAttn — DDP,
        # .to(device), and iter-0 checkpoint saving all need the full
        # parameter set to exist at __init__ time.
        self._eager_targets: Optional[tuple] = None
        self._eager_sources: Optional[tuple] = None
        if levels_cfg is not None:
            all_names = [lc["name"] for lc in levels_cfg]
            if self.target_levels is not None:
                missing = [t for t in self.target_levels if t not in all_names]
                if missing:
                    raise KeyError(
                        f"CrossLevelAttn target_levels references levels "
                        f"not present in levels_cfg: {missing!r}. "
                        f"Available: {all_names!r}"
                    )
                resolved_t = self.target_levels
            else:
                resolved_t = tuple(n for n in all_names if n.startswith("voxel"))
            if self.source_levels is not None:
                missing = [s for s in self.source_levels if s not in all_names]
                if missing:
                    raise KeyError(
                        f"CrossLevelAttn source_levels references levels "
                        f"not present in levels_cfg: {missing!r}. "
                        f"Available: {all_names!r}"
                    )
                resolved_s = self.source_levels
            else:
                # Default: every level can be a source (including SP).
                resolved_s = tuple(all_names)
            self._eager_targets = resolved_t
            self._eager_sources = resolved_s
            self._build_for(resolved_t)

    # ------------------------------------------------------------------

    def _build_for(self, target_names) -> None:
        """Build the per-target cross-attn stacks. Idempotent."""
        for name in target_names:
            if name not in self.blocks_by_target:
                self.blocks_by_target[name] = nn.ModuleList([
                    _CrossAttnFFNBlock(**self._block_kwargs)
                    for _ in range(self.num_layers)
                ])

    def _select_target_source(
        self, levels: "OrderedDict[str, LevelOutput]",
    ) -> "tuple[tuple, tuple]":
        """Lazy fallback resolution of (targets, sources).

        Only used when levels_cfg was NOT passed at __init__ — i.e. in
        standalone smoke-test settings. NOT DDP-safe.
        """
        if self.target_levels is not None:
            missing = [t for t in self.target_levels if t not in levels]
            if missing:
                raise KeyError(
                    f"CrossLevelAttn target_levels references levels not "
                    f"present in the tokenizer output: {missing!r}. "
                    f"Levels: {list(levels.keys())!r}"
                )
            targets = self.target_levels
        else:
            targets = tuple(n for n in levels.keys() if n.startswith("voxel"))
        if self.source_levels is not None:
            missing = [s for s in self.source_levels if s not in levels]
            if missing:
                raise KeyError(
                    f"CrossLevelAttn source_levels references levels not "
                    f"present in the tokenizer output: {missing!r}. "
                    f"Levels: {list(levels.keys())!r}"
                )
            sources = self.source_levels
        else:
            sources = tuple(levels.keys())
        return targets, sources

    def _maybe_subsample(
        self, tokens: torch.Tensor, pos: torch.Tensor, cap: Optional[int],
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        # The random subsample is a TRAIN-TIME regularization ("light dropout").
        # It must NOT run at inference: torch.randperm consumes the global RNG
        # per forward, so a given event's K/V (hence output) would depend on
        # which OTHER events preceded it (membership/list/batch dependence) — the
        # same bug class as shuffle_orders. In eval, use ALL source tokens.
        if cap is None or tokens.shape[0] <= cap or not self.training:
            return tokens, pos
        idx = torch.randperm(tokens.shape[0], device=tokens.device)[:cap]
        return tokens[idx], pos[idx]

    # ------------------------------------------------------------------

    def forward(
        self,
        levels: "OrderedDict[str, LevelOutput]",
    ) -> "OrderedDict[str, LevelOutput]":
        if self._eager_targets is not None:
            targets, sources = self._eager_targets, self._eager_sources
            missing = [t for t in targets if t not in levels]
            if missing:
                raise KeyError(
                    f"CrossLevelAttn eager-built for targets {targets!r}, "
                    f"but input levels {list(levels.keys())!r} lack: "
                    f"{missing!r}"
                )
        else:
            targets, sources = self._select_target_source(levels)
            self._build_for(targets)

        # Current per-level token state (mutated across layers).
        state: "dict[str, torch.Tensor]" = {
            name: lvl.tokens for name, lvl in levels.items()
        }
        # Per-level pos_emb (computed once, reused across layers + as Q-pos
        # for target lookups + as K-pos for source contributions).
        pos_by_level: "dict[str, torch.Tensor]" = {}
        for name, lvl in levels.items():
            if lvl.n_tokens > 0:
                pos_by_level[name] = self.pos_emb(lvl.coords)
            else:
                pos_by_level[name] = lvl.tokens.new_zeros(0, self.dim)

        # Layer loop. Each layer:
        #   • Rebuilds K/V from the CURRENT state of all sources (so the
        #     previous layer's target updates are visible to the next layer).
        #   • Updates every target by cross-attending against that K/V.
        for layer_i in range(self.num_layers):
            # Build the shared source soup for this layer.
            src_tokens_parts = []
            src_pos_parts = []
            for src_name in sources:
                tok = state[src_name]
                if tok.shape[0] == 0:
                    continue
                pos = pos_by_level[src_name]
                tok_sub, pos_sub = self._maybe_subsample(
                    tok, pos, self.max_source_tokens_per_level,
                )
                src_tokens_parts.append(tok_sub)
                src_pos_parts.append(pos_sub)
            if not src_tokens_parts:
                # No sources have tokens — skip this layer's update.
                continue
            kv_tokens = torch.cat(src_tokens_parts, dim=0)
            k_pos = torch.cat(src_pos_parts, dim=0)

            # Update each target.
            for tgt in targets:
                q_tokens = state[tgt]
                if q_tokens.shape[0] == 0:
                    continue
                q_pos = pos_by_level[tgt]
                blk = self.blocks_by_target[tgt][layer_i]
                state[tgt] = blk(q_tokens, q_pos, kv_tokens, k_pos)

        # Emit refined levels (untouched levels pass through unchanged).
        out: "OrderedDict[str, LevelOutput]" = OrderedDict()
        for name, lvl in levels.items():
            if name in targets and lvl.n_tokens > 0:
                out[name] = LevelOutput(
                    tokens=state[name],
                    coords=lvl.coords,
                    sp_to_level_id=lvl.sp_to_level_id,
                    name=lvl.name,
                )
            else:
                out[name] = lvl
        return out
