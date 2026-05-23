"""
LArFormerLoss — set-prediction loss generalized over the user's level list.

Three families of supervision (any combination, per the config):

    1. Query class CE + origin L1 (Hungarian-matched query → GT instance)
    2. Mask loss at each level that declares `supervision.mask`:
       - one level is `primary` → per-pair PointRend-sampled BCE + Dice
         (the strong signal; expensive)
       - all others → full-mask BCE (cheap) if M_level ≤ aux_max_tokens,
         else fall back to per-pair sampled BCE
    3. Per-level token-classification CE / BCE at each level that
       declares `supervision.cls`. Independent of queries / matching.

Deep supervision: families 1 and 2 are summed over the init prediction +
every decoder layer. Family 3 is computed once per event (level tokens
don't change through the decoder).

Public helpers (also used by `tools/visualize_larformer_gt.py`):

    build_per_level_instance_mask(level_out, gt_instances)
        → (K, M_level) float in {0, 1}

    build_per_level_cls_target(level_out, per_sp_label, reduce, n_classes=None)
        → (M_level,) long

    build_per_level_gt(levels, gt_instances, per_sp_labels, levels_cfg)
        → dict[level_name → {"instance_mask": ..., "cls_target": ...}]
"""

from collections import OrderedDict
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .builders import LevelOutput
from .matcher import HungarianMatcher


# ---------------------------------------------------------------------------
# Public GT-construction helpers (importable; visualizer reuses these)
# ---------------------------------------------------------------------------

def build_per_level_instance_mask(
    level_out: LevelOutput,
    gt_truth_indices: Sequence[torch.Tensor],
) -> torch.Tensor:
    """For one event + one level, build (K, M) GT instance mask.

    A level token m belongs to GT instance k iff any spacepoint in
    `gt_truth_indices[k]` is mapped to that token by `level_out.sp_to_level_id`.

    Args:
        level_out:         LevelOutput for this level
        gt_truth_indices:  length-K list of (M_k,) LongTensor of spacepoint
                            indices per GT instance
    Returns:
        (K, M) float in {0, 1}; M = level_out.n_tokens
    """
    M = level_out.n_tokens
    K = len(gt_truth_indices)
    device = level_out.sp_to_level_id.device
    if K == 0 or M == 0:
        return torch.zeros(K, M, dtype=torch.float32, device=device)
    out = torch.zeros(K, M, dtype=torch.float32, device=device)
    for k, sp_idx in enumerate(gt_truth_indices):
        # Accept either np.ndarray or torch.Tensor — the legacy
        # ShowerClusteringDataset stores per-instance indices as numpy.
        if not isinstance(sp_idx, torch.Tensor):
            sp_idx = torch.as_tensor(sp_idx, dtype=torch.long, device=device)
        else:
            sp_idx = sp_idx.to(device=device, dtype=torch.long)
        if sp_idx.numel() == 0:
            continue
        tok = level_out.sp_to_level_id[sp_idx]
        # Drop unmapped spacepoints (sp_to_level_id == -1, e.g. fragment level)
        tok = tok[tok >= 0]
        if tok.numel() == 0:
            continue
        out[k].index_fill_(0, tok.unique(), 1.0)
    return out


def build_per_level_cls_target(
    level_out: LevelOutput,
    per_sp_label: torch.Tensor,
    reduce: str = "plurality",
    ignore_index: int = -1,
    label_remap: Optional[Dict[int, int]] = None,
) -> torch.Tensor:
    """Reduce a per-spacepoint integer label down to one label per level token.

    Args:
        level_out:        LevelOutput for this level
        per_sp_label:     (N,) long — per-spacepoint label, with `ignore_index`
                          marking points to be excluded from the vote
        reduce:           "plurality" — most common non-ignore label per token
                          "amax"      — max non-ignore label per token (cheap,
                                         right for binary; combine with
                                         `label_remap` to do priority pooling)
        ignore_index:     label value to skip when voting (default −1)
        label_remap:      optional dict {old → new} applied to each per-SP
                          label before the reduce. Use this together with
                          `reduce="amax"` to encode priority pooling: pick
                          integer codes so the highest-priority class has
                          the largest value (e.g. {1: 2, 2: 1} flips the
                          {1=nu, 2=cosmic} convention so amax favors nu).
                          Inputs not present in the dict are passed through
                          unchanged. ignore_index entries are never remapped.
    Returns:
        (M,) long. Tokens with no valid spacepoints get `ignore_index`.
    """
    M = level_out.n_tokens
    device = per_sp_label.device
    if M == 0:
        return torch.zeros(0, dtype=torch.long, device=device)
    sp_to_level = level_out.sp_to_level_id
    valid = (sp_to_level >= 0) & (per_sp_label != ignore_index)

    out = torch.full((M,), ignore_index, dtype=torch.long, device=device)
    if not valid.any():
        return out

    sp_tok = sp_to_level[valid]
    sp_lab = per_sp_label[valid].to(torch.long)

    if label_remap:
        # Build a small LUT covering [0, max_input]. Any index outside the
        # remap dict's keys is identity-mapped, so callers can specify
        # only the keys that change. Negative labels were already filtered
        # by the `valid` mask above.
        max_in = max(
            int(sp_lab.max().item()) if sp_lab.numel() > 0 else 0,
            max(label_remap.keys()),
        )
        lut = torch.arange(max_in + 1, dtype=torch.long, device=device)
        for k, v in label_remap.items():
            if k < 0 or k > max_in:
                continue
            lut[int(k)] = int(v)
        sp_lab = lut[sp_lab.clamp(0, max_in)]

    if reduce == "amax":
        # scatter_reduce with "amax" — gives the max label per token. Right
        # for binary 0/1; meaningful for ordinal labels too.
        tmp = torch.full((M,), -1, dtype=torch.long, device=device)
        tmp.scatter_reduce_(0, sp_tok, sp_lab, reduce="amax", include_self=False)
        out = torch.where(tmp >= 0, tmp, out)
        return out

    if reduce == "plurality":
        # Most-common label per token. Implemented as
        # argmax over a (M, C) count matrix; C is the label range.
        n_classes = int(sp_lab.max().item()) + 1
        if n_classes <= 0:
            return out
        counts = torch.zeros(M, n_classes, dtype=torch.long, device=device)
        ones = torch.ones_like(sp_tok)
        flat_idx = sp_tok * n_classes + sp_lab
        counts.view(-1).scatter_add_(0, flat_idx, ones)
        any_vote = counts.sum(dim=1) > 0
        winners = counts.argmax(dim=1)
        out = torch.where(any_vote, winners, out)
        return out

    raise ValueError(f"unknown reduce rule {reduce!r}; "
                     f"use 'plurality' or 'amax'")


def build_per_level_gt(
    levels: "OrderedDict[str, LevelOutput]",
    gt_instances: List[dict],
    per_sp_labels: Dict[str, torch.Tensor],
    levels_cfg: Sequence[dict],
) -> "OrderedDict[str, dict]":
    """Build all per-level GT (instance masks + cls targets) for one event.

    Args:
        levels:         OrderedDict from the composite tokenizer (one event)
        gt_instances:   list of K dicts from the dataset; each must have
                         "truth_indices" (LongTensor of per-SP indices)
        per_sp_labels:  dict mapping label-source name → (N,) tensor. Keys
                         referenced by any level's `supervision.cls.label_src`
                         must be present.
        levels_cfg:     the model's `levels` config block (a list of dicts
                         with keys: name, supervision)
    Returns:
        OrderedDict {level_name: {"instance_mask": Tensor or None,
                                  "cls_target":    Tensor or None}}
    """
    gt_truth = [g["truth_indices"] for g in gt_instances]

    cfg_by_name = {lc["name"]: lc for lc in levels_cfg}
    out: "OrderedDict[str, dict]" = OrderedDict()
    for name, level_out in levels.items():
        sup = cfg_by_name.get(name, {}).get("supervision", {}) or {}
        entry: dict = {"instance_mask": None, "cls_target": None}

        if "mask" in sup:
            entry["instance_mask"] = build_per_level_instance_mask(
                level_out, gt_truth,
            )

        if "cls" in sup:
            cls_cfg = sup["cls"]
            src = cls_cfg["label_src"]
            if src not in per_sp_labels:
                raise KeyError(
                    f"level {name!r} declares cls supervision with "
                    f"label_src={src!r}, but per_sp_labels has only "
                    f"{list(per_sp_labels.keys())!r}"
                )
            entry["cls_target"] = build_per_level_cls_target(
                level_out, per_sp_labels[src],
                reduce=cls_cfg.get("reduce", "plurality"),
                ignore_index=int(cls_cfg.get("ignore_index", -1)),
                label_remap=cls_cfg.get("label_remap"),
            )

        out[name] = entry
    return out


# ---------------------------------------------------------------------------
# Sampling helpers (lifted from shower_clustering; generalized)
# ---------------------------------------------------------------------------

def _importance_sample_negatives(
    bg_indices: torch.Tensor,            # (n_bg,) long
    pred_logits_q: torch.Tensor,          # (M,) — model's mask logits for this query
    n_neg_target: int,
    oversample_ratio: float = 3.0,
    importance_ratio: float = 0.75,
    hard_neg_ratio: float = 0.0,
) -> torch.Tensor:
    """PointRend / Mask2Former-style importance sampling of negatives, with
    an optional complementary hard-negative branch.

    Three independent budget fractions on `n_neg_target`:

      • `importance_ratio` — picks the smallest `|sigmoid(logit) - 0.5|`,
        i.e. the most-uncertain (boundary halo) points. Default 0.75. Right
        intervention when the failure mode is ambiguous-boundary halos.

      • `hard_neg_ratio` — picks the largest `sigmoid(logit)`, i.e. the
        model's most-confident false-positives among the bg. Right
        intervention when the failure mode is over-clustering / merging
        (confidently wrong predictions sitting at `sigm≈1`, which the
        halo-uncertainty criterion explicitly DEPRIORITIZES). Default 0.0.

      • remainder (`1 - importance_ratio - hard_neg_ratio`) — uniform
        random over all bg, for coverage diversity.

    Both importance and hard-neg picks are drawn from the same oversampled
    candidate pool (`oversample_ratio * n_neg_target` random bg points).
    The two branches can overlap; that's tolerated. If
    `importance_ratio + hard_neg_ratio > 1`, the random budget clamps to 0
    and the two branches' actual sizes are scaled down to fit.

    All scoring uses `torch.no_grad()` — selection doesn't backprop.

    Originally ported from `pointcept.models.shower_clustering.losses.
    _importance_sample_negatives`, where the halo-only variant uplifted a
    `mask_iou=0.03` setting. See `docs/LArFormer.md` §15 for the failure-
    mode analysis that motivates the hard-neg extension.
    """
    device = bg_indices.device
    n_bg = int(bg_indices.numel())
    if n_neg_target <= 0 or n_bg == 0:
        return torch.zeros(0, dtype=torch.long, device=device)

    # (1) oversample candidates
    n_over = min(int(round(oversample_ratio * n_neg_target)), n_bg)
    perm = torch.randperm(n_bg, device=device)[:n_over]
    candidates = bg_indices[perm]

    # Resolve per-branch budgets, clamped to the candidate pool size.
    if importance_ratio + hard_neg_ratio > 1.0:
        # Proportionally rescale the two structured branches to fit.
        scale = 1.0 / (importance_ratio + hard_neg_ratio)
        importance_ratio = importance_ratio * scale
        hard_neg_ratio = hard_neg_ratio * scale

    n_imp = min(int(round(importance_ratio * n_neg_target)), n_over)
    n_hard = min(int(round(hard_neg_ratio * n_neg_target)),
                 max(0, n_over - n_imp))
    n_rand = max(0, n_neg_target - n_imp - n_hard)

    # (2a) halo branch — smallest |sigm - 0.5|
    if n_imp > 0 or n_hard > 0:
        with torch.no_grad():
            sigm = pred_logits_q[candidates].sigmoid()

    if n_imp > 0:
        with torch.no_grad():
            unc = (sigm - 0.5).abs()
            _, imp_idx = torch.topk(unc, n_imp, largest=False)
        sampled_imp = candidates[imp_idx]
    else:
        sampled_imp = torch.zeros(0, dtype=torch.long, device=device)

    # (2b) hard-negative branch — largest sigm (confident false-positives)
    if n_hard > 0:
        with torch.no_grad():
            _, hard_idx = torch.topk(sigm, n_hard, largest=True)
        sampled_hard = candidates[hard_idx]
    else:
        sampled_hard = torch.zeros(0, dtype=torch.long, device=device)

    # (3) random top-up (collisions with the structured picks tolerated)
    if n_rand > 0:
        perm2 = torch.randperm(n_bg, device=device)[:n_rand]
        sampled_rand = bg_indices[perm2]
    else:
        sampled_rand = torch.zeros(0, dtype=torch.long, device=device)

    return torch.cat([sampled_imp, sampled_hard, sampled_rand], dim=0)


def _balanced_point_sample(
    positive_indices: torch.Tensor,
    n_tokens: int,
    n_sample: int,
    device: torch.device,
    pos_fraction: float = 0.5,
) -> torch.Tensor:
    """Balanced sample of up to n_sample indices into [0, n_tokens).

    Falls back to uniform random if there are no positives.
    """
    if n_sample <= 0 or n_tokens == 0:
        return torch.zeros(0, dtype=torch.long, device=device)
    n_pos_avail = int(positive_indices.numel())
    if n_pos_avail == 0:
        s = min(n_sample, n_tokens)
        return torch.randperm(n_tokens, device=device)[:s]
    n_pos_target = min(int(round(pos_fraction * n_sample)), n_pos_avail)
    is_pos = torch.zeros(n_tokens, dtype=torch.bool, device=device)
    is_pos[positive_indices] = True
    bg = torch.where(~is_pos)[0]
    n_bg_avail = int(bg.numel())
    n_neg_target = n_sample - n_pos_target
    if n_neg_target > n_bg_avail:
        n_neg_target = n_bg_avail
        n_pos_target = min(n_sample - n_neg_target, n_pos_avail)
    pos = (positive_indices[torch.randperm(n_pos_avail, device=device)[:n_pos_target]]
           if n_pos_target > 0 else positive_indices.new_zeros(0))
    neg = (bg[torch.randperm(n_bg_avail, device=device)[:n_neg_target]]
           if n_neg_target > 0 else bg.new_zeros(0))
    return torch.cat([pos, neg], dim=0)


def _gt_mask_at_indices(gt_mask: torch.Tensor,
                        idx: torch.Tensor) -> torch.Tensor:
    """gt_mask (K, M) → (K, len(idx))."""
    if gt_mask.shape[0] == 0 or idx.numel() == 0:
        return gt_mask.new_zeros(gt_mask.shape[0], idx.numel())
    return gt_mask.index_select(1, idx)


# ---------------------------------------------------------------------------
# Per-pair sampled BCE + Dice (PointRend-style)
# ---------------------------------------------------------------------------

def _per_pair_sampled_mask_loss(
    pred_logits: torch.Tensor,       # (Q, M)
    gt_mask: torch.Tensor,           # (K, M)
    q_idx: np.ndarray,
    k_idx: np.ndarray,
    n_sample: int,
    pos_fraction: float = 0.5,
    use_importance_sampling: bool = False,
    importance_oversample_ratio: float = 3.0,
    importance_ratio: float = 0.75,
    importance_hard_neg_ratio: float = 0.0,
):
    """Per-pair sampled BCE + Dice. Returns (bce_loss, dice_loss) — scalar
    tensors averaged over the matched pairs.

    With `use_importance_sampling=True`, the negative half of each per-pair
    sample is drawn via `_importance_sample_negatives` — biases negatives
    toward the model's currently-uncertain region (the over-prediction halo
    around the predicted mask boundary). `importance_hard_neg_ratio > 0`
    additionally diverts part of the negative budget to confident false-
    positives (largest sigm), targeting the over-clustering failure mode.
    The positive half is unchanged (random subset of GT positives up to
    n_pos = pos_fraction * n_sample).
    """
    device = pred_logits.device
    if len(q_idx) == 0:
        z = pred_logits.new_zeros(())
        return z, z
    n_target_pos = max(int(round(pos_fraction * n_sample)), 1)
    bce_terms, dice_terms = [], []
    M = pred_logits.shape[1]
    for p in range(len(q_idx)):
        q_p = int(q_idx[p]); k_p = int(k_idx[p])
        pos_idx = torch.where(gt_mask[k_p] > 0)[0]
        n_pos_actual = min(n_target_pos, int(pos_idx.numel()))
        if n_pos_actual == int(pos_idx.numel()):
            sampled_pos = pos_idx
        else:
            sampled_pos = pos_idx[
                torch.randperm(int(pos_idx.numel()), device=device)[:n_pos_actual]
            ]
        n_neg_target = n_sample - n_pos_actual
        is_pos = torch.zeros(M, dtype=torch.bool, device=device)
        is_pos[pos_idx] = True
        bg = torch.where(~is_pos)[0]

        if n_neg_target > 0 and bg.numel() > 0:
            if use_importance_sampling:
                sampled_neg = _importance_sample_negatives(
                    bg_indices=bg,
                    pred_logits_q=pred_logits[q_p],
                    n_neg_target=n_neg_target,
                    oversample_ratio=importance_oversample_ratio,
                    importance_ratio=importance_ratio,
                    hard_neg_ratio=importance_hard_neg_ratio,
                )
            else:
                n_neg = min(n_neg_target, int(bg.numel()))
                sampled_neg = bg[
                    torch.randperm(int(bg.numel()), device=device)[:n_neg]
                ]
        else:
            sampled_neg = bg.new_zeros(0)
        sampled = torch.cat([sampled_pos, sampled_neg], dim=0)
        if sampled.numel() == 0:
            continue
        logit_p = pred_logits[q_p, sampled]
        gt_p = logit_p.new_zeros(sampled.shape[0])
        gt_p[:sampled_pos.shape[0]] = 1.0
        bce_terms.append(F.binary_cross_entropy_with_logits(
            logit_p, gt_p, reduction="mean"
        ))
        sigm = logit_p.sigmoid()
        num = 2.0 * (sigm * gt_p).sum()
        denom = sigm.sum() + gt_p.sum()
        dice_terms.append(1.0 - (num + 1.0) / (denom + 1.0))
    if not bce_terms:
        z = pred_logits.new_zeros(())
        return z, z
    return torch.stack(bce_terms).mean(), torch.stack(dice_terms).mean()


# ---------------------------------------------------------------------------
# Main loss
# ---------------------------------------------------------------------------

class LArFormerLoss(nn.Module):
    """Set-prediction loss with deep supervision, generalized over levels.

    Args:
        num_classes:          query class count, incl. no_object as the last
        no_object_class_id:   defaults to num_classes - 1
        no_object_weight:     CE downweight on no_object (default 0.1)
        weight_class:         scalar weight on query CE
        weight_mask_primary:  weight on per-pair sampled BCE at primary level
        weight_dice_primary:  weight on per-pair sampled Dice at primary level
        weight_aux_mask:      base weight on aux mask BCE at non-primary levels
                              (can be overridden per-level via the level's
                              supervision.mask.weight)
        weight_per_level_cls: base weight on per-level token CE (per-level
                              override available)
        weight_origin:        weight on Hungarian-matched origin L1
        num_sample_points:    S — sampled tokens per pair for primary mask loss
        aux_max_tokens:       per-level threshold: if M_level > this AND the
                              level is not primary, fall back to per-pair
                              sampling for the aux mask loss
        deep_supervision:     apply mask + class + origin at init + every layer
        match_layer:          "final" (default) or "init" — which layer's
                              predictions to run the Hungarian matcher on
        cost_*:               matcher cost weights (matched to weight_*)
    """

    def __init__(
        self,
        levels_cfg: Sequence[dict],
        num_classes: int,
        no_object_class_id: Optional[int] = None,
        no_object_weight: float = 0.1,
        weight_class: float = 2.0,
        weight_mask_primary: float = 5.0,
        weight_dice_primary: float = 5.0,
        weight_aux_mask: float = 1.0,
        weight_per_level_cls: float = 1.0,
        weight_origin: float = 1.0,
        num_sample_points: int = 4096,
        aux_max_tokens: int = 10_000,
        deep_supervision: bool = True,
        match_layer: str = "final",
        cost_class: float = 2.0,
        cost_mask: float = 5.0,
        cost_dice: float = 5.0,
        cost_origin: float = 1.0,
        # Importance sampling (PointRend-style hard-negative mining) for the
        # per-pair mask loss. See _importance_sample_negatives for details.
        # Off by default to preserve baseline behavior; flip on for slicer-
        # style configs where the over-prediction halo is a known failure mode.
        # `importance_hard_neg_ratio > 0` diverts part of the negative budget
        # to confident false-positives (largest sigm); targets the over-
        # clustering / mirror-merge failure mode described in
        # `docs/LArFormer.md` §15. Default 0 keeps the halo-only behavior.
        use_importance_sampling: bool = False,
        importance_oversample_ratio: float = 3.0,
        importance_ratio: float = 0.75,
        importance_hard_neg_ratio: float = 0.0,
        # Mask DINO-style denoising auxiliary loss. When the model passes
        # a non-empty DN slice into `compute_dn_loss`, the per-pair losses
        # (cls/mask/dice/origin) are computed against direct gt_target_idx
        # matching (no Hungarian) and scaled by `weight_dn_loss`. Each
        # individual component reuses the same per-pair sampler the
        # regular path uses. Default 1.0 = DN losses contribute at the
        # SAME magnitude as the matched losses; lower to 0.5 to halve.
        weight_dn_loss: float = 1.0,
    ):
        super().__init__()
        if match_layer not in ("final", "init"):
            raise ValueError("match_layer must be 'final' or 'init'")
        self.levels_cfg = list(levels_cfg)
        self.num_classes = int(num_classes)
        self.no_object_class_id = (no_object_class_id
                                   if no_object_class_id is not None
                                   else num_classes - 1)
        self.weight_class = float(weight_class)
        self.weight_mask_primary = float(weight_mask_primary)
        self.weight_dice_primary = float(weight_dice_primary)
        self.weight_aux_mask = float(weight_aux_mask)
        self.weight_per_level_cls = float(weight_per_level_cls)
        self.weight_origin = float(weight_origin)
        self.num_sample_points = int(num_sample_points)
        self.aux_max_tokens = int(aux_max_tokens)
        self.deep_supervision = bool(deep_supervision)
        self.match_layer = match_layer
        self.use_importance_sampling = bool(use_importance_sampling)
        self.importance_oversample_ratio = float(importance_oversample_ratio)
        self.importance_ratio = float(importance_ratio)
        self.importance_hard_neg_ratio = float(importance_hard_neg_ratio)
        self.weight_dn_loss = float(weight_dn_loss)

        ce_weights = torch.ones(self.num_classes)
        ce_weights[self.no_object_class_id] = float(no_object_weight)
        self.register_buffer("ce_weights", ce_weights)

        # Identify the primary level (must be exactly one if any mask
        # supervision is declared). If no mask supervision is declared on
        # any level, primary_level stays None and the mask losses skip.
        primaries = [lc["name"] for lc in self.levels_cfg
                     if (lc.get("supervision") or {}).get("mask", {}).get("mode") == "primary"]
        if len(primaries) > 1:
            raise ValueError(
                f"multiple levels marked as supervision.mask.mode=primary: "
                f"{primaries!r}; exactly one is allowed"
            )
        any_mask = any((lc.get("supervision") or {}).get("mask")
                       for lc in self.levels_cfg)
        if any_mask and not primaries:
            raise ValueError(
                "at least one level declares supervision.mask but none is "
                "marked supervision.mask.mode=primary — declare exactly one "
                "primary so the matcher knows which level's mask logits to use"
            )
        self.primary_level: Optional[str] = primaries[0] if primaries else None

        self.matcher = HungarianMatcher(
            cost_class=cost_class, cost_mask=cost_mask,
            cost_dice=cost_dice, cost_origin=cost_origin,
        )

    # ------------------------------------------------------------------

    def _aux_mask_loss(
        self,
        pred_logits: torch.Tensor,     # (Q, M)
        gt_mask: torch.Tensor,         # (K, M)
        q_idx: np.ndarray,
        k_idx: np.ndarray,
    ) -> torch.Tensor:
        """Per-level aux BCE: full-mask if M is small, per-pair sampled if not."""
        if len(q_idx) == 0 or pred_logits.shape[1] == 0:
            return pred_logits.new_zeros(())
        q_t = torch.as_tensor(q_idx, dtype=torch.long, device=pred_logits.device)
        k_t = torch.as_tensor(k_idx, dtype=torch.long, device=pred_logits.device)
        if pred_logits.shape[1] <= self.aux_max_tokens:
            return F.binary_cross_entropy_with_logits(
                pred_logits[q_t], gt_mask[k_t], reduction="mean",
            )
        # Token count exceeds the aux cap — fall back to per-pair sampled BCE.
        bce, _ = _per_pair_sampled_mask_loss(
            pred_logits, gt_mask, q_idx, k_idx,
            n_sample=self.num_sample_points,
            use_importance_sampling=self.use_importance_sampling,
            importance_oversample_ratio=self.importance_oversample_ratio,
            importance_ratio=self.importance_ratio,
            importance_hard_neg_ratio=self.importance_hard_neg_ratio,
        )
        return bce

    # ------------------------------------------------------------------

    def _compute_layer_loss(
        self,
        layer_pred: dict,
        gt_classes: torch.Tensor,
        gt_origin: Optional[torch.Tensor],
        per_level_gt_mask: "OrderedDict[str, torch.Tensor]",
        q_idx: np.ndarray,
        k_idx: np.ndarray,
    ) -> Dict[str, torch.Tensor]:
        cls_logits = layer_pred["class_logits"]
        device = cls_logits.device
        Q = cls_logits.shape[0]

        # Query class CE (matched + no-object)
        cls_target = torch.full((Q,), self.no_object_class_id,
                                dtype=torch.long, device=device)
        if len(q_idx) > 0:
            q_t = torch.as_tensor(q_idx, dtype=torch.long, device=device)
            k_t = torch.as_tensor(k_idx, dtype=torch.long, device=device)
            cls_target[q_t] = gt_classes[k_t]
        cls_loss = F.cross_entropy(cls_logits, cls_target, weight=self.ce_weights)

        zero = cls_logits.new_zeros(())
        out: Dict[str, torch.Tensor] = {
            "cls":          cls_loss,
            "mask_primary": zero,
            "dice_primary": zero,
            "origin":       zero,
        }
        for name in per_level_gt_mask.keys():
            if name != self.primary_level:
                out[f"aux_mask_{name}"] = zero

        if len(q_idx) == 0:
            return out

        # Primary mask: per-pair sampled BCE + Dice
        if self.primary_level is not None:
            primary_pred = layer_pred["mask_logits"][self.primary_level]
            primary_gt = per_level_gt_mask[self.primary_level]
            bce, dice = _per_pair_sampled_mask_loss(
                primary_pred, primary_gt, q_idx, k_idx,
                n_sample=self.num_sample_points,
                use_importance_sampling=self.use_importance_sampling,
                importance_oversample_ratio=self.importance_oversample_ratio,
                importance_ratio=self.importance_ratio,
                importance_hard_neg_ratio=self.importance_hard_neg_ratio,
            )
            out["mask_primary"] = bce
            out["dice_primary"] = dice

        # Aux mask BCE at every non-primary level with mask supervision
        for name, gt in per_level_gt_mask.items():
            if name == self.primary_level:
                continue
            pred = layer_pred["mask_logits"][name]
            out[f"aux_mask_{name}"] = self._aux_mask_loss(pred, gt, q_idx, k_idx)

        # Origin L1 (matched only)
        if (gt_origin is not None and self.weight_origin > 0
                and "origin" in layer_pred):
            q_t = torch.as_tensor(q_idx, dtype=torch.long, device=device)
            k_t = torch.as_tensor(k_idx, dtype=torch.long, device=device)
            out["origin"] = (layer_pred["origin"][q_t]
                             - gt_origin[k_t]).abs().mean()

        return out

    # ------------------------------------------------------------------

    def _compute_per_level_cls_terms(
        self,
        per_level_cls_logits: "OrderedDict[str, torch.Tensor]",
        per_level_gt: "OrderedDict[str, dict]",
    ) -> Dict[str, torch.Tensor]:
        """One CE / BCE per declared cls head. Shared by both forward paths."""
        cls_terms: Dict[str, torch.Tensor] = {}
        cfg_by_name = {lc["name"]: lc for lc in self.levels_cfg}
        for name, logits in per_level_cls_logits.items():
            target = per_level_gt[name]["cls_target"]
            if target is None or logits.shape[0] == 0:
                continue
            cls_cfg = (cfg_by_name[name].get("supervision") or {})["cls"]
            ignore_index = int(cls_cfg.get("ignore_index", -1))
            loss_kind = cls_cfg.get("loss", "ce")
            if loss_kind == "ce":
                cls_terms[name] = F.cross_entropy(
                    logits, target, ignore_index=ignore_index,
                )
            elif loss_kind == "bce_with_logits":
                if logits.dim() == 2 and logits.shape[1] == 1:
                    logits = logits.squeeze(-1)
                t = target.to(logits.dtype)
                mask = (target != ignore_index)
                if mask.any():
                    cls_terms[name] = F.binary_cross_entropy_with_logits(
                        logits[mask], t[mask], reduction="mean",
                    )
            else:
                raise ValueError(f"unknown per-level cls loss {loss_kind!r}")
        return cls_terms

    def compute_dn_loss(
        self,
        decoder_output_dn: dict,
        per_level_gt_mask: "OrderedDict[str, torch.Tensor]",
        gt_classes: torch.Tensor,
        gt_origin: Optional[torch.Tensor],
        gt_target_idx: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Direct (non-Hungarian) per-pair losses for Mask DINO-style
        denoising queries.

        Args:
            decoder_output_dn: the DN slice of `Mask2FormerDecoder` output
                — same {init, layers, final} structure as the regular
                output, but with Q rows == Q_dn (number of DN queries)
                rather than K_reg. The model is responsible for slicing
                this before calling.
            per_level_gt_mask: same per-level GT mask dict the regular
                forward already builds. We re-use it — no recomputation.
            gt_classes, gt_origin: per-instance targets, same shapes the
                Hungarian path uses (K,) and (K, 3) or None.
            gt_target_idx: (Q_dn,) long tensor — query i is supervised
                against GT instance `gt_target_idx[i]`. Built by
                `MaskDenoiser`.

        Returns: dict with "total" (already scaled by `weight_dn_loss`)
        plus the per-component scalars (un-scaled, for telemetry). The
        model is expected to prefix these with "dn_" before logging /
        merging with the regular loss dict.

        Returns an all-zero dict (with finite "total") if Q_dn == 0 —
        eval and zero-GT events fall through cleanly.
        """
        # Pull a device handle from any always-available tensor in the
        # DN slice (every layer has a class_logits).
        cls_logits_any = decoder_output_dn["final"]["class_logits"]
        device = cls_logits_any.device
        Q_dn = int(cls_logits_any.shape[0])

        zero = cls_logits_any.new_zeros(())
        empty: Dict[str, torch.Tensor] = {"total": zero}
        if Q_dn == 0 or gt_target_idx.numel() == 0:
            empty["n_dn"] = torch.zeros((), dtype=torch.long, device=device)
            return empty
        # Direct assignment: q_idx is identity, k_idx is gt_target_idx.
        # _compute_layer_loss takes numpy arrays (matches the Hungarian
        # path's interface from `scipy.optimize.linear_sum_assignment`).
        q_idx = np.arange(Q_dn, dtype=np.int64)
        k_idx = gt_target_idx.detach().cpu().numpy().astype(np.int64)

        sup_layers = []
        if self.deep_supervision:
            sup_layers.append(decoder_output_dn["init"])
            sup_layers.extend(decoder_output_dn["layers"])
        else:
            sup_layers.append(decoder_output_dn["final"])

        per_layer_losses = [
            self._compute_layer_loss(
                layer_pred=lyr,
                gt_classes=gt_classes,
                gt_origin=gt_origin,
                per_level_gt_mask=per_level_gt_mask,
                q_idx=q_idx, k_idx=k_idx,
            )
            for lyr in sup_layers
        ]
        agg: Dict[str, torch.Tensor] = {
            k: torch.stack([pl[k] for pl in per_layer_losses]).sum()
            for k in per_layer_losses[0].keys()
        }

        cfg_by_name = {lc["name"]: lc for lc in self.levels_cfg}

        # Aggregate into a DN total using the SAME per-component weights
        # the regular path uses (matched semantics — DN supervises the
        # same heads with the same per-pair sampler) then scale the
        # final sum by `weight_dn_loss`. Aux mask weights still inherit
        # any per-level override declared in the level cfg.
        total = self.weight_class * agg["cls"]
        if self.primary_level is not None:
            total = (total
                     + self.weight_mask_primary * agg["mask_primary"]
                     + self.weight_dice_primary * agg["dice_primary"])
        if gt_origin is not None and self.weight_origin > 0:
            total = total + self.weight_origin * agg["origin"]
        for name in per_level_gt_mask.keys():
            if name == self.primary_level:
                continue
            sup = (cfg_by_name[name].get("supervision") or {}).get("mask", {})
            w = float(sup.get("weight", self.weight_aux_mask))
            total = total + w * agg[f"aux_mask_{name}"]
        total = self.weight_dn_loss * total

        out: Dict[str, torch.Tensor] = {"total": total}
        for k, v in agg.items():
            out[k] = v
        out["n_dn"] = torch.tensor(Q_dn, dtype=torch.long, device=device)
        return out

    # ------------------------------------------------------------------

    def _cls_only_loss(
        self,
        per_level_cls_logits: "OrderedDict[str, torch.Tensor]",
        per_level_gt: "OrderedDict[str, dict]",
        device: torch.device,
        K: int,
    ) -> Dict[str, torch.Tensor]:
        """Stage-1 deghoster fast path: no decoder, no queries, only per-level
        cls supervision. Returns the same kind of scalar-tensor dict the
        full forward returns so the model's per-event aggregator handles
        both paths uniformly.
        """
        cls_terms = self._compute_per_level_cls_terms(
            per_level_cls_logits, per_level_gt,
        )
        cfg_by_name = {lc["name"]: lc for lc in self.levels_cfg}
        total = torch.zeros((), device=device)
        for name, term in cls_terms.items():
            sup = (cfg_by_name[name].get("supervision") or {}).get("cls", {})
            w = float(sup.get("weight", self.weight_per_level_cls))
            total = total + w * term
        out: Dict[str, torch.Tensor] = {"total": total}
        for name, term in cls_terms.items():
            out[f"cls_{name}"] = term
        out["n_matched"] = torch.zeros((), dtype=torch.long, device=device)
        out["n_gt_instances"] = torch.tensor(K, dtype=torch.long, device=device)
        return out

    # ------------------------------------------------------------------

    def forward(
        self,
        decoder_output: Optional[dict],
        levels: "OrderedDict[str, LevelOutput]",
        gt_instances: List[dict],
        per_sp_labels: Dict[str, torch.Tensor],
        per_level_cls_logits: "OrderedDict[str, torch.Tensor]",
        return_matching: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """One event.

        Args:
            decoder_output:        dict with keys init/layers/final from
                                    `Mask2FormerDecoder.forward`, OR None
                                    when the model has no decoder (Stage-1
                                    deghoster pattern — only per-level cls
                                    supervision fires).
            levels:                 tokenizer output (one event)
            gt_instances:           list[K] of dicts; each requires
                                    "truth_indices", "origin_type", and
                                    "origin_coord_norm" (only used iff the
                                    origin head is enabled)
            per_sp_labels:          dict label_src → (N,) tensor (used by
                                    any level with cls supervision)
            per_level_cls_logits:   dict level_name → (M_level, C_level) —
                                    output of the model's per-level cls heads
                                    (empty dict if no level declares cls
                                    supervision)
            return_matching:        if True, also include the Hungarian
                                    assignment (q_idx, k_idx) and the GT
                                    tensors used for matching in the
                                    output dict. The evaluator uses this
                                    to compute per-pair metrics without
                                    redoing the match.
        Returns:
            dict with "total" plus per-component scalars. When
            return_matching=True, additionally includes (as np arrays /
            tensors, not 0-d scalars — the model's per-event loss
            aggregator must filter these out):
                q_idx, k_idx           np.int64 arrays of matched pairs
                gt_classes             (K,) LongTensor on `device`
                gt_origin              (K, 3) FloatTensor or None
                gt_truth_indices       list[K] of LongTensors on `device`
                per_level_gt_mask      OrderedDict[name → (K, M_level)]
        """
        # Pull device from any always-present tensor; decoder_output may be None.
        device = next(iter(levels.values())).tokens.device
        K = len(gt_instances)

        # GT scalar tensors
        if K > 0:
            gt_classes = torch.tensor(
                [int(g["origin_type"]) for g in gt_instances],
                dtype=torch.long, device=device,
            )
            if any("origin_coord_norm" in g for g in gt_instances):
                gt_origin = torch.stack([
                    torch.as_tensor(g.get("origin_coord_norm",
                                          np.zeros(3, dtype=np.float32)),
                                    dtype=torch.float32, device=device)
                    for g in gt_instances
                ], dim=0)
            else:
                gt_origin = None
        else:
            gt_classes = torch.zeros(0, dtype=torch.long, device=device)
            gt_origin = None

        # Per-level GT (mask + cls targets, computed once per event)
        per_level_gt = build_per_level_gt(
            levels=levels,
            gt_instances=gt_instances,
            per_sp_labels=per_sp_labels,
            levels_cfg=self.levels_cfg,
        )
        per_level_gt_mask: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        for name, entry in per_level_gt.items():
            if entry["instance_mask"] is not None:
                per_level_gt_mask[name] = entry["instance_mask"]

        # Cls-only fast path: model has no decoder (num_queries=0). All the
        # query / mask / origin machinery is skipped and the total reduces
        # to the sum of per-level cls losses. Falls through if no level
        # declared cls supervision (total is then identically zero).
        if decoder_output is None:
            return self._cls_only_loss(
                per_level_cls_logits=per_level_cls_logits,
                per_level_gt=per_level_gt,
                device=device, K=K,
            )

        # Hungarian match on chosen layer's primary mask logits
        if self.primary_level is None or K == 0:
            q_idx = np.zeros(0, dtype=np.int64)
            k_idx = np.zeros(0, dtype=np.int64)
        else:
            match_pred = (decoder_output["final"] if self.match_layer == "final"
                          else decoder_output["init"])
            primary_mask = match_pred["mask_logits"][self.primary_level]
            primary_gt = per_level_gt_mask[self.primary_level]
            # Union positives across all instances at the primary level
            union_pos = (primary_gt.sum(dim=0) > 0).nonzero(as_tuple=False).squeeze(-1)
            sampled = _balanced_point_sample(
                positive_indices=union_pos,
                n_tokens=primary_mask.shape[1],
                n_sample=self.num_sample_points,
                device=device, pos_fraction=0.5,
            )
            gt_sampled = _gt_mask_at_indices(primary_gt, sampled)
            q_idx, k_idx = self.matcher(
                class_logits=match_pred["class_logits"],
                primary_mask_logits=primary_mask,
                gt_classes=gt_classes,
                gt_masks_sampled=gt_sampled,
                sampled_indices=sampled,
                origin_pred=match_pred.get("origin"),
                gt_origin=gt_origin,
            )

        # Apply per-layer losses (deep supervision)
        sup_layers = []
        if self.deep_supervision:
            sup_layers.append(decoder_output["init"])
            sup_layers.extend(decoder_output["layers"])
        else:
            sup_layers.append(decoder_output["final"])

        per_layer_losses = [
            self._compute_layer_loss(
                layer_pred=lyr,
                gt_classes=gt_classes,
                gt_origin=gt_origin,
                per_level_gt_mask=per_level_gt_mask,
                q_idx=q_idx, k_idx=k_idx,
            )
            for lyr in sup_layers
        ]
        agg: Dict[str, torch.Tensor] = {
            k: torch.stack([pl[k] for pl in per_layer_losses]).sum()
            for k in per_layer_losses[0].keys()
        }

        cls_terms = self._compute_per_level_cls_terms(
            per_level_cls_logits, per_level_gt,
        )
        cfg_by_name = {lc["name"]: lc for lc in self.levels_cfg}

        # ------- aggregate into a total -------
        total = self.weight_class * agg["cls"]
        if self.primary_level is not None:
            total = (total
                     + self.weight_mask_primary * agg["mask_primary"]
                     + self.weight_dice_primary * agg["dice_primary"])
        if gt_origin is not None and self.weight_origin > 0:
            total = total + self.weight_origin * agg["origin"]
        for name in per_level_gt_mask.keys():
            if name == self.primary_level:
                continue
            sup = (cfg_by_name[name].get("supervision") or {}).get("mask", {})
            w = float(sup.get("weight", self.weight_aux_mask))
            total = total + w * agg[f"aux_mask_{name}"]
        for name, term in cls_terms.items():
            sup = (cfg_by_name[name].get("supervision") or {}).get("cls", {})
            w = float(sup.get("weight", self.weight_per_level_cls))
            total = total + w * term

        out: Dict[str, torch.Tensor] = {"total": total}
        for k, v in agg.items():
            out[k] = v
        for name, term in cls_terms.items():
            out[f"cls_{name}"] = term
        out["n_matched"] = torch.tensor(len(q_idx), dtype=torch.long, device=device)
        out["n_gt_instances"] = torch.tensor(K, dtype=torch.long, device=device)

        if return_matching:
            # GT truth_indices as device LongTensors (gt_instances may
            # store them as numpy; coerce so callers don't have to).
            gt_truth_indices = []
            for g in gt_instances:
                ti = g.get("truth_indices")
                if ti is None:
                    gt_truth_indices.append(
                        torch.zeros(0, dtype=torch.long, device=device)
                    )
                elif isinstance(ti, torch.Tensor):
                    gt_truth_indices.append(ti.to(device=device, dtype=torch.long))
                else:
                    gt_truth_indices.append(
                        torch.as_tensor(ti, dtype=torch.long, device=device)
                    )
            out["q_idx"] = q_idx                              # np.int64
            out["k_idx"] = k_idx                              # np.int64
            out["gt_classes"] = gt_classes                    # (K,) long
            out["gt_origin"] = gt_origin                      # (K, 3) or None
            out["gt_truth_indices"] = gt_truth_indices        # list[K]
            out["per_level_gt_mask"] = per_level_gt_mask      # OrderedDict
        return out
