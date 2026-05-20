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


def _pairwise_bce_cost(pred_logits: torch.Tensor,
                       gt_mask: torch.Tensor) -> torch.Tensor:
    """Per-pair BCE mean over points. pred_logits (Q, S), gt_mask (K, S)."""
    pos = -F.logsigmoid(pred_logits)
    neg = pred_logits - F.logsigmoid(pred_logits)
    cost = pos @ gt_mask.transpose(0, 1) + neg @ (1.0 - gt_mask.transpose(0, 1))
    return cost / float(pred_logits.shape[1])


def _pairwise_dice_cost(pred_logits: torch.Tensor,
                        gt_mask: torch.Tensor,
                        eps: float = 1.0) -> torch.Tensor:
    """Per-pair (1 - Dice). pred_logits (Q, S), gt_mask (K, S) → (Q, K)."""
    p = pred_logits.sigmoid()
    num = 2.0 * (p @ gt_mask.transpose(0, 1))
    denom = p.sum(dim=-1, keepdim=True) + gt_mask.sum(dim=-1).unsqueeze(0)
    return 1.0 - (num + eps) / (denom + eps)


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
        q_idx, k_idx = linear_sum_assignment(cost)
        return (q_idx.astype(np.int64), k_idx.astype(np.int64))
