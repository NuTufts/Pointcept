"""Mixed query selection for LArFormer (DINO/Mask-DINO-style adaptation).

Selects K query anchors from one of the levels' tokens by:
  1. Scoring each token (default: 1 - p(no_object) from that level's cls
     head — reuses existing supervision, no new training signal needed)
  2. Filtering to the top-M by score (M = K * score_filter_multiplier)
  3. Running FPS (farthest point sampling) on the M survivors to pick K
     spatially-diverse anchors

Why the FPS step (vs DINO's pure top-K): LArTPC slices vary enormously
in size — one long cosmic produces hundreds of high-score voxel_8cm
tokens. Pure top-K would happily put 100 queries on the longest cosmic
and miss several smaller tracks. FPS guarantees spatial coverage, so
each query "owns" a Voronoi cell of the event volume.

The selected anchors become the initial query content + positional anchor
for the Mask2Former decoder. Combined with zero-init `query_content`/
`query_pos` (handled in LArFormer when this selector is active), the
decoder's queries start at sensible specialized positions rather than
random — addresses matching instability (queries know what they're "for"
at init) and over-claim (queries don't all converge on the same easy
SPs).

See `docs/LArFormer.md` §17 for context.
"""

from typing import Optional

import torch
import torch.nn as nn


def _farthest_point_sampling(coords: torch.Tensor, k: int) -> torch.Tensor:
    """Pure-PyTorch FPS. Returns indices into `coords` of K spatially-
    diverse points. Seeds with index 0 — callers should put the most
    informative point first (typically the highest-score token).
    """
    N = coords.shape[0]
    k = int(min(k, N))
    if k <= 0:
        return coords.new_zeros(0, dtype=torch.long)
    if N <= k:
        return torch.arange(N, dtype=torch.long, device=coords.device)
    distances = torch.full(
        (N,), float("inf"), dtype=torch.float32, device=coords.device,
    )
    selected = torch.zeros(k, dtype=torch.long, device=coords.device)
    selected[0] = 0
    for i in range(1, k):
        prev = selected[i - 1]
        d = ((coords - coords[prev]) ** 2).sum(-1).sqrt()
        distances = torch.minimum(distances, d)
        selected[i] = distances.argmax()
    return selected


class MixedQuerySelector(nn.Module):
    """Picks K initial query anchors from a chosen level's tokens.

    Args:
        token_dim:                  decoder token dim (= LArFormer.token_dim)
        num_queries:                K, number of query anchors to produce
        source_level:               which level's tokens to score+pick from
                                     (default: "voxel_8cm"; must have a
                                     `supervision.cls` head declared when
                                     `score_source="cls_head"`)
        score_source:               "cls_head" — use `1 - p(no_object)`
                                     from the source level's cls head as
                                     the per-token objectness score. The
                                     cls head is already trained; nothing
                                     new to supervise. v1 only supports
                                     this; future "dedicated" mode would
                                     add a separate proposer head + aux
                                     BCE loss.
        selection_mode:             "top_m_then_fps" (default, recommended)
                                     or "top_k" (pure top-K by score —
                                     may cluster queries on long high-
                                     confidence tracks).
        score_filter_multiplier:    M = K × this. Larger M = more spatial
                                     diversity (FPS picks K from a wider
                                     pool); smaller M = stronger score
                                     prior. Default 4.
        no_object_class_id:         which column of the cls head's output
                                     is "no_object". Default None → last
                                     class index (matches LArFormer
                                     convention: `num_classes - 1`).
    """

    def __init__(
        self,
        token_dim: int,
        num_queries: int,
        source_level: str = "voxel_8cm",
        score_source: str = "cls_head",
        selection_mode: str = "top_m_then_fps",
        score_filter_multiplier: int = 4,
        no_object_class_id: Optional[int] = None,
    ):
        super().__init__()
        if score_source not in ("cls_head",):
            raise ValueError(
                f"MixedQuerySelector score_source must be 'cls_head'; "
                f"got {score_source!r}"
            )
        if selection_mode not in ("top_m_then_fps", "top_k"):
            raise ValueError(
                f"MixedQuerySelector selection_mode must be "
                f"'top_m_then_fps' or 'top_k'; got {selection_mode!r}"
            )
        self.token_dim = int(token_dim)
        self.num_queries = int(num_queries)
        self.source_level = str(source_level)
        self.score_source = score_source
        self.selection_mode = selection_mode
        self.score_filter_multiplier = max(1, int(score_filter_multiplier))
        self.no_object_class_id = no_object_class_id

    @torch.no_grad()
    def forward(
        self,
        levels,                    # OrderedDict[str, LevelOutput]
        per_level_cls_logits,       # OrderedDict[str, torch.Tensor]
    ):
        """Returns (init_query_content (Q, D), init_anchor_coords (Q, 3)).

        - `init_query_content[i]` = source level's token features at the
          i-th selected anchor (Q, D).
        - `init_anchor_coords[i]` = source level's coords at the i-th
          selected anchor (Q, 3).

        Both are gradient-free (the selection is non-differentiable; the
        decoder's per-slot zero-init delta carries the learnable part).

        Padding: if the source level has fewer than K tokens after the
        score filter, the leftover query slots get zero tokens + zero
        coords. The decoder's `query_content` / `query_pos` will then
        carry those slots (acting as a fallback for rare under-counted
        events).

        Raises KeyError if the source level isn't present or doesn't
        have a cls head (when score_source='cls_head').
        """
        if self.source_level not in levels:
            raise KeyError(
                f"MixedQuerySelector source_level={self.source_level!r} "
                f"not in tokenizer levels {list(levels.keys())!r}"
            )
        lvl = levels[self.source_level]
        K = self.num_queries
        if lvl.n_tokens == 0:
            return (
                lvl.tokens.new_zeros(K, self.token_dim),
                lvl.coords.new_zeros(K, 3),
            )

        if self.score_source == "cls_head":
            if self.source_level not in per_level_cls_logits:
                raise KeyError(
                    f"MixedQuerySelector score_source='cls_head' requires "
                    f"the source_level {self.source_level!r} to have cls "
                    f"supervision declared. per_level_cls_logits has: "
                    f"{list(per_level_cls_logits.keys())!r}"
                )
            cls_logits = per_level_cls_logits[self.source_level]  # (M, C)
            if cls_logits.shape[0] != lvl.n_tokens:
                raise RuntimeError(
                    f"MixedQuerySelector: cls_logits[{self.source_level}] "
                    f"has {cls_logits.shape[0]} rows but level has "
                    f"{lvl.n_tokens} tokens"
                )
            probs = cls_logits.softmax(dim=-1)
            no_obj_idx = (self.no_object_class_id
                          if self.no_object_class_id is not None
                          else cls_logits.shape[-1] - 1)
            scores = 1.0 - probs[:, no_obj_idx]                   # (M,)
        else:  # pragma: no cover — guarded by __init__
            raise RuntimeError(f"Unhandled score_source: {self.score_source!r}")

        N = lvl.tokens.shape[0]

        if self.selection_mode == "top_k":
            picks = scores.topk(min(K, N)).indices
        else:  # "top_m_then_fps"
            M = min(K * self.score_filter_multiplier, N)
            # topk returns indices sorted DESC by score → index 0 is
            # highest-score token, which seeds FPS.
            topm_idx = scores.topk(M).indices
            topm_coords = lvl.coords[topm_idx]
            fps_local = _farthest_point_sampling(topm_coords, K)
            picks = topm_idx[fps_local]

        # Pad to K with -1 (rare: only when N < K, very small events).
        n_picked = int(picks.shape[0])
        if n_picked < K:
            pad = picks.new_full((K - n_picked,), -1)
            picks = torch.cat([picks, pad])

        valid = picks >= 0
        init_q = lvl.tokens.new_zeros(K, self.token_dim)
        init_anchor = lvl.coords.new_zeros(K, 3)
        if valid.any():
            v_picks = picks[valid]
            init_q[valid] = lvl.tokens[v_picks]
            init_anchor[valid] = lvl.coords[v_picks]
        return init_q, init_anchor
