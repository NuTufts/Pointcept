"""
Hungarian matcher for the shower-clustering Mask2Former model.

Matches the N model queries to K ground-truth shower instances so each GT
instance is claimed by at most one query. Unmatched queries are trained to
predict the "no_object" class.

The cost of pairing query q with GT instance k is:

    cost[q, k] = λ_cls * cost_cls[q, k]
               + λ_mask * cost_mask[q, k]   (BCE on sampled spacepoints)
               + λ_dice * cost_dice[q, k]   (Dice on the same set)
               + λ_origin * cost_origin[q, k]  (L1 in normalized coords)

Mask2Former-style point sampling: rather than computing the (Q, K, N) cost
tensor at full N (up to 250k spacepoints per event), we sample S points and
evaluate the mask cost only at those. This gives an unbiased estimate of
the full-spacepoint BCE/Dice and is cheap. Same sampled set is reused for
the loss computation, so matching and loss are consistent.

Hungarian solve is `scipy.optimize.linear_sum_assignment` (available in the
pointcept container).

Per-event API: matcher operates on one event at a time. The model class
(Phase 7) is responsible for batch iteration.
"""

from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Cost helpers
# ---------------------------------------------------------------------------

def _pairwise_bce_cost(pred_logits: torch.Tensor,
                       gt_mask: torch.Tensor) -> torch.Tensor:
    """Per-pair binary cross-entropy summed over points.

    Args:
        pred_logits: (Q, S) raw mask logits (no sigmoid)
        gt_mask:     (K, S) float in {0, 1}
    Returns:
        (Q, K) cost matrix
    """
    # BCE_with_logits: -[t * log σ(x) + (1-t) * log(1-σ(x))]
    # Pairwise: BCE[q, k] = sum_s -[gt_mask[k, s] * log σ(pred[q, s])
    #                              + (1 - gt_mask[k, s]) * log(1-σ(pred[q, s]))]
    # Stable variant via log-sigmoid identity:
    #    -log(σ(x))     = -log_sigmoid(x)
    #    -log(1-σ(x))   = -log_sigmoid(-x) = x - log_sigmoid(x)
    pos = -F.logsigmoid(pred_logits)             # (Q, S)
    neg = pred_logits - F.logsigmoid(pred_logits)  # (Q, S)
    # cost[q, k] = sum_s gt[k, s] * pos[q, s] + (1 - gt[k, s]) * neg[q, s]
    cost = pos @ gt_mask.transpose(0, 1) + neg @ (1.0 - gt_mask.transpose(0, 1))
    s = pred_logits.shape[1]
    return cost / s   # mean per point keeps this comparable across event sizes


def _pairwise_dice_cost(pred_logits: torch.Tensor,
                        gt_mask: torch.Tensor,
                        eps: float = 1.0) -> torch.Tensor:
    """Per-pair (1 - Dice). Smaller = better overlap.

    Args:
        pred_logits: (Q, S)
        gt_mask:     (K, S) float
    Returns:
        (Q, K)
    """
    p = pred_logits.sigmoid()                     # (Q, S)
    # numerator: 2 * sum p * t  -> (Q, K)
    num = 2.0 * (p @ gt_mask.transpose(0, 1))
    # denom: sum p (per q) + sum t (per k)
    denom = p.sum(dim=-1, keepdim=True) + gt_mask.sum(dim=-1).unsqueeze(0)
    return 1.0 - (num + eps) / (denom + eps)


# ---------------------------------------------------------------------------
# Matcher
# ---------------------------------------------------------------------------

class HungarianMatcher(nn.Module):
    """Hungarian matcher for set-prediction loss.

    Args:
        cost_class:  weight on -p[gt_class] (per-query class probability).
        cost_mask:   weight on the BCE component (per-pair, sampled points).
        cost_dice:   weight on the Dice component.
        cost_origin: weight on L1 between predicted and GT origin coord.
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

    @torch.no_grad()
    def forward(
        self,
        class_logits: torch.Tensor,           # (Q, C)
        spacepoint_mask_logits: torch.Tensor,  # (Q, N)
        origin: torch.Tensor,                  # (Q, 3)
        gt_classes: torch.Tensor,              # (K,) long, in [0, C-1)
        gt_masks_sampled: torch.Tensor,        # (K, S) float in {0, 1}
        gt_origin: torch.Tensor,               # (K, 3) float
        sampled_indices: torch.Tensor,         # (S,) long, in [0, N)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run one Hungarian solve.

        Returns: (q_idx, k_idx) — np arrays of equal length, the matched
        pairs of (query index, GT instance index). The model treats
        unmatched queries (not in q_idx) as no_object.
        """
        from scipy.optimize import linear_sum_assignment

        Q, C = class_logits.shape
        K = gt_classes.shape[0]
        if K == 0 or Q == 0:
            return (np.array([], dtype=np.int64),
                    np.array([], dtype=np.int64))

        # Class cost: -p[q, gt_class[k]]
        # We exclude the no_object slot from being a target — gt_classes are
        # always real classes, never the C-1 'no_object' index.
        probs = class_logits.softmax(dim=-1)          # (Q, C)
        cls_cost = -probs[:, gt_classes]              # (Q, K)

        # Mask + Dice cost on sampled spacepoints
        pred_sampled = spacepoint_mask_logits[:, sampled_indices]  # (Q, S)
        mask_cost = _pairwise_bce_cost(pred_sampled, gt_masks_sampled)
        dice_cost = _pairwise_dice_cost(pred_sampled, gt_masks_sampled)

        # Origin L1 cost (per-axis mean, so the units are normalized-coord
        # L1 and don't grow with the number of axes).
        # pred origin (Q, 3), gt origin (K, 3) -> (Q, K)
        origin_cost = (origin.unsqueeze(1) - gt_origin.unsqueeze(0)).abs().mean(dim=-1)

        cost = (self.cost_class * cls_cost
                + self.cost_mask * mask_cost
                + self.cost_dice * dice_cost
                + self.cost_origin * origin_cost)
        cost = cost.detach().cpu().numpy()
        q_idx, k_idx = linear_sum_assignment(cost)
        return (q_idx.astype(np.int64), k_idx.astype(np.int64))
