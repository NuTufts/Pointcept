"""
HungarianMatcher for LArFormer (lifted from shower_clustering, parameterized
by the level whose mask logits drive the per-pair mask cost).

The matcher operates on a single event's predictions. Its cost matrix
combines:

    cost[q, k] = λ_cls    · (-p[q, gt_class[k]])
               + λ_mask   · BCE_sampled(pred_primary[q, :], gt_mask[k, :])
               + λ_dice   · (1 - Dice_sampled(...))
               + λ_origin · |pred_origin[q] - gt_origin[k]|_1     (optional)

The mask + dice components are evaluated on a shared `sampled_indices`
set at the primary level (same set the loss reuses). `origin` is optional:
if `cost_origin == 0` or `origin_pred is None`, the term is skipped.
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Clamp range for raw mask logits before BCE / Dice cost computation.
# sigmoid saturates well before ±50, so this is a no-op in normal training
# but prevents (-inf) - logsigmoid(-inf) = (-inf) - (-inf) = NaN when a
# freshly-initialized model produces pathologically extreme logits early on.
_LOGIT_CLAMP = 50.0


def _pairwise_bce_cost(pred_logits: torch.Tensor,
                       gt_mask: torch.Tensor) -> torch.Tensor:
    """Per-pair BCE mean over points. pred_logits (Q, S), gt_mask (K, S)."""
    pred_logits = pred_logits.clamp(-_LOGIT_CLAMP, _LOGIT_CLAMP)
    pos = -F.logsigmoid(pred_logits)
    neg = pred_logits - F.logsigmoid(pred_logits)
    cost = pos @ gt_mask.transpose(0, 1) + neg @ (1.0 - gt_mask.transpose(0, 1))
    return cost / float(pred_logits.shape[1])


def _pairwise_dice_cost(pred_logits: torch.Tensor,
                        gt_mask: torch.Tensor,
                        eps: float = 1.0) -> torch.Tensor:
    """Per-pair (1 - Dice). pred_logits (Q, S), gt_mask (K, S) → (Q, K)."""
    pred_logits = pred_logits.clamp(-_LOGIT_CLAMP, _LOGIT_CLAMP)
    p = pred_logits.sigmoid()
    num = 2.0 * (p @ gt_mask.transpose(0, 1))
    denom = p.sum(dim=-1, keepdim=True) + gt_mask.sum(dim=-1).unsqueeze(0)
    return 1.0 - (num + eps) / (denom + eps)


def _sanitize_cost(cost: np.ndarray) -> "tuple[np.ndarray, int]":
    """Replace NaN / +Inf / -Inf with a large finite sentinel so
    linear_sum_assignment can solve. Returns (clean_cost, n_bad)."""
    bad = ~np.isfinite(cost)
    n_bad = int(bad.sum())
    if n_bad == 0:
        return cost, 0
    cost = cost.copy()
    # Use the largest finite cost in the matrix (×10) so the sanitized
    # cells are guaranteed to be the WORST possible assignment, but in
    # the same ballpark as legitimate large costs (Hungarian behaves
    # poorly when costs span 100+ orders of magnitude).
    finite_max = float(np.nan_to_num(
        cost[np.isfinite(cost)], nan=0.0, posinf=0.0, neginf=0.0,
    ).max()) if np.isfinite(cost).any() else 1.0
    sentinel = max(abs(finite_max) * 10.0, 1e6)
    cost[bad] = sentinel
    return cost, n_bad


class HungarianMatcher(nn.Module):
    """Per-event Hungarian matcher.

    Args:
        cost_class:  weight on -p[gt_class] term
        cost_mask:   weight on per-pair BCE
        cost_dice:   weight on per-pair Dice
        cost_origin: weight on per-pair L1 origin (skipped if 0)
    """

    def __init__(
        self,
        cost_class: float = 2.0,
        cost_mask: float = 5.0,
        cost_dice: float = 5.0,
        cost_origin: float = 1.0,
    ):
        super().__init__()
        self.cost_class = float(cost_class)
        self.cost_mask = float(cost_mask)
        self.cost_dice = float(cost_dice)
        self.cost_origin = float(cost_origin)
        # Count of forward calls that had to sanitize NaN/Inf entries in
        # the cost matrix. Should rapidly drop to 0 once the model warms
        # up; sustained nonzero values mean something is producing
        # non-finite logits and the model is masking a real bug.
        self._n_sanitized_forwards = 0
        self._n_total_forwards = 0

    @torch.no_grad()
    def forward(
        self,
        class_logits: torch.Tensor,            # (Q, C)
        primary_mask_logits: torch.Tensor,     # (Q, M_primary)
        gt_classes: torch.Tensor,              # (K,) long
        gt_masks_sampled: torch.Tensor,        # (K, S) float in {0, 1}
        sampled_indices: torch.Tensor,         # (S,) long, into [0, M_primary)
        origin_pred: Optional[torch.Tensor] = None,   # (Q, 3) or None
        gt_origin: Optional[torch.Tensor] = None,     # (K, 3) or None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """One Hungarian solve. Returns (q_idx, k_idx)."""
        from scipy.optimize import linear_sum_assignment

        Q = class_logits.shape[0]
        K = gt_classes.shape[0]
        if Q == 0 or K == 0:
            return (np.array([], dtype=np.int64),
                    np.array([], dtype=np.int64))

        probs = class_logits.softmax(dim=-1)
        cls_cost = -probs[:, gt_classes]                           # (Q, K)

        pred_sampled = primary_mask_logits[:, sampled_indices]     # (Q, S)
        mask_cost = _pairwise_bce_cost(pred_sampled, gt_masks_sampled)
        dice_cost = _pairwise_dice_cost(pred_sampled, gt_masks_sampled)

        cost = (self.cost_class * cls_cost
                + self.cost_mask * mask_cost
                + self.cost_dice * dice_cost)

        if (self.cost_origin > 0 and origin_pred is not None
                and gt_origin is not None):
            origin_cost = (origin_pred.unsqueeze(1)
                           - gt_origin.unsqueeze(0)).abs().mean(dim=-1)
            cost = cost + self.cost_origin * origin_cost

        cost = cost.detach().cpu().numpy()
        cost, n_bad = _sanitize_cost(cost)
        self._n_total_forwards += 1
        if n_bad > 0:
            self._n_sanitized_forwards += 1
            # Print only on the first few occurrences so we don't spam
            # the log every iter once a real bug surfaces. Forwards
            # 1, 2, 3, 10, 30, 100, 300, 1000, ... get logged.
            n = self._n_sanitized_forwards
            if n <= 3 or n in (10, 30, 100, 300) or n % 1000 == 0:
                import warnings
                warnings.warn(
                    f"HungarianMatcher: cost matrix had {n_bad} non-finite "
                    f"entries on forward {self._n_total_forwards} (sanitized "
                    f"events: {self._n_sanitized_forwards}). Replaced with a "
                    f"large finite sentinel. This is expected for the first "
                    f"few iters with freshly-initialized weights; sustained "
                    f"occurrence means something downstream is producing "
                    f"non-finite logits. Consider adding clip_grad to the "
                    f"optimizer or lowering base_lr.",
                    RuntimeWarning,
                )
        q_idx, k_idx = linear_sum_assignment(cost)
        return (q_idx.astype(np.int64), k_idx.astype(np.int64))
