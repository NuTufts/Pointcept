"""PerLevelSelfAttn — Option 1 token refiner.

Applies N pre-norm self-attention + FFN blocks INDEPENDENTLY to each
target level's tokens. No cross-level interaction. Spacepoint-level
refinement is intentionally not supported by default (cost is O(N²) for
~50K SPs; PTv3's windowed attention already mixes SP context inside the
backbone). Restrict targets to the voxel levels — that's where the merger-
rate diagnostic localized the confident-FP problem.

Each block is the DETR-style "pos_emb added to Q+K but not V":

    q = q_norm + pos_emb(coords)
    k = q_norm + pos_emb(coords)
    v = q_norm
    x = x + SelfAttn(q, k, v)
    x = x + FFN(LayerNorm(x))

The pos_emb is local to the refiner (not shared with the decoder's), so
ablations can vary the refiner's pos_emb_kind independently.
"""

from collections import OrderedDict
from typing import Optional, Sequence

import torch
import torch.nn as nn

from ..builders import LevelOutput
from .base import REFINERS, TokenRefiner
from .pos_emb import build_pos_emb


# Alias for backward compatibility with the inlined helper name.
_build_pos_emb = build_pos_emb


# ---------------------------------------------------------------------------
# Self-attention + FFN block
# ---------------------------------------------------------------------------

class _SelfAttnFFNBlock(nn.Module):
    """Pre-norm self-attn (with pos_emb on Q/K, identity on V) + FFN."""

    def __init__(self, dim: int, num_heads: int,
                 mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim)
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

        # DETR/Mask2Former zero-init on the OUTPUT projections so each
        # residual block is the identity at init. Refiner tokens pass
        # through unchanged → the FFN/attention "amplification cascade"
        # through N stacked random-init blocks is avoided. Gradient still
        # flows back into these zeroed weights via the input side.
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)
        nn.init.zeros_(self.ffn[-1].weight)
        nn.init.zeros_(self.ffn[-1].bias)

    def forward(self, tokens: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        # tokens: (M, D); pos: (M, D). Add batch dim for MHA.
        x = tokens.unsqueeze(0)
        x_n = self.attn_norm(x)
        q = x_n + pos.unsqueeze(0)
        k = x_n + pos.unsqueeze(0)
        v = x_n
        attn_out, _ = self.attn(q, k, v, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x.squeeze(0)


# ---------------------------------------------------------------------------
# Refiner
# ---------------------------------------------------------------------------

@REFINERS.register_module()
class PerLevelSelfAttn(TokenRefiner):
    """Intra-level self-attention refiner.

    Args:
        dim:              token dim — MUST match LArFormer's token_dim.
        num_layers:       number of SA+FFN blocks per target level.
        num_heads:        MHA heads.
        mlp_ratio:        FFN expansion.
        dropout:          dropout on attn output + FFN.
        target_levels:    iterable of level names to refine. Other levels
                           pass through unchanged. Default None = "every
                           level whose name starts with 'voxel'" (a
                           reasonable heuristic; explicit list overrides).
        pos_emb_kind:     "mlp" (default) or "sinusoidal", matching the
                           decoder's flag of the same name.
        pos_emb_hidden_dim / pos_emb_num_freq / pos_emb_max_freq:
                           forwarded to the pos_emb module.
    """

    def __init__(
        self,
        dim: int,
        num_layers: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        target_levels: Optional[Sequence[str]] = None,
        levels_cfg: Optional[Sequence[dict]] = None,
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

        # One stack of blocks per target level — params NOT shared across
        # levels. Different levels have very different token statistics
        # (token count, pooling factor, intrinsic resolution), so per-level
        # weights let them specialize. If memory is a concern, sharing is
        # a future option (a `share_across_levels` flag).
        #
        # If `levels_cfg` is provided (LArFormer auto-injects it), modules
        # are EAGERLY built here so:
        #   • DDP can wrap a complete parameter set,
        #   • model.to(device) places everything on the right device,
        #   • checkpoint save/load at iter=0 is consistent.
        # If not provided (e.g. standalone smoke test), falls back to a
        # lazy build on the first forward — convenient but DOES NOT WORK
        # with DDP or post-construction .to(device).
        self._block_kwargs = dict(
            dim=self.dim, num_heads=int(num_heads),
            mlp_ratio=float(mlp_ratio), dropout=float(dropout),
        )
        self._pos_emb_kwargs = dict(
            kind=str(pos_emb_kind), dim=self.dim,
            hidden_dim=pos_emb_hidden_dim,
            num_freq=pos_emb_num_freq, max_freq=float(pos_emb_max_freq),
        )
        self.blocks_by_level = nn.ModuleDict()
        self.pos_emb_by_level = nn.ModuleDict()

        # Eager-build path.
        self._eager_targets: Optional[tuple] = None
        if levels_cfg is not None:
            all_names = [lc["name"] for lc in levels_cfg]
            if self.target_levels is not None:
                missing = [t for t in self.target_levels if t not in all_names]
                if missing:
                    raise KeyError(
                        f"PerLevelSelfAttn target_levels references levels "
                        f"not present in levels_cfg: {missing!r}. "
                        f"Available: {all_names!r}"
                    )
                resolved = self.target_levels
            else:
                resolved = tuple(n for n in all_names if n.startswith("voxel"))
            self._eager_targets = resolved
            self._build_for(resolved)

    # ------------------------------------------------------------------

    def _select_targets(
        self, levels: "OrderedDict[str, LevelOutput]",
    ) -> tuple:
        if self.target_levels is not None:
            missing = [t for t in self.target_levels if t not in levels]
            if missing:
                raise KeyError(
                    f"PerLevelSelfAttn target_levels references levels not "
                    f"present in the tokenizer output: {missing!r}. Levels: "
                    f"{list(levels.keys())!r}"
                )
            return self.target_levels
        # Heuristic default: every voxel_* level. Skip spacepoint (cost)
        # and skip fragment (small, irregular; less benefit from intra-
        # level SA, can revisit if needed).
        return tuple(name for name in levels.keys()
                     if name.startswith("voxel"))

    def _build_for(self, level_names) -> None:
        """Build SA-block stacks + pos_emb for each named level.

        Idempotent — re-building an existing level is a no-op. Used both
        in __init__ (eager path) and in forward (lazy fallback).
        """
        for name in level_names:
            if name not in self.blocks_by_level:
                # nn.ModuleDict keys must not contain dots — voxel_* / etc.
                # don't, so we use the name verbatim.
                stack = nn.ModuleList([
                    _SelfAttnFFNBlock(**self._block_kwargs)
                    for _ in range(self.num_layers)
                ])
                self.blocks_by_level[name] = stack
                self.pos_emb_by_level[name] = _build_pos_emb(
                    **self._pos_emb_kwargs,
                )

    # ------------------------------------------------------------------

    def forward(
        self,
        levels: "OrderedDict[str, LevelOutput]",
    ) -> "OrderedDict[str, LevelOutput]":
        if self._eager_targets is not None:
            targets = self._eager_targets
            # Validate that the eager-built target set matches what showed
            # up in the input — catches the (rare) case where a config
            # change broke the levels_cfg ↔ tokenizer alignment.
            missing = [t for t in targets if t not in levels]
            if missing:
                raise KeyError(
                    f"PerLevelSelfAttn eager-built for {targets!r}, but "
                    f"input levels {list(levels.keys())!r} lack: {missing!r}"
                )
        else:
            # Lazy path — convenient for standalone smoke tests but does
            # NOT work with DDP or post-construction .to(device).
            targets = self._select_targets(levels)
            self._build_for(targets)

        out: "OrderedDict[str, LevelOutput]" = OrderedDict()
        for name, lvl in levels.items():
            if name not in targets or lvl.n_tokens == 0:
                out[name] = lvl
                continue
            pos = self.pos_emb_by_level[name](lvl.coords)        # (M, D)
            tokens = lvl.tokens
            for blk in self.blocks_by_level[name]:
                tokens = blk(tokens, pos)
            out[name] = LevelOutput(
                tokens=tokens,
                coords=lvl.coords,
                sp_to_level_id=lvl.sp_to_level_id,
                name=lvl.name,
            )
        return out
