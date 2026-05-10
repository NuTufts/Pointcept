"""
Set-prediction loss for the shower-clustering Mask2Former model.

Per Mask2Former, the matcher is run once on the model's final-layer
predictions; the resulting (query, GT) pairing is then reused to compute
losses at every decoder layer (init + every layer). This makes the
auxiliary losses consistent with the final prediction.

Loss components, applied per matched pair (q, k):

    L_cls    : CE(class_logits[q], gt_class[k])           — full, not -p
    L_mask   : BCE(spacepoint_mask_logits[q, sampled],
                    gt_mask[k, sampled])                   — sampled S points
    L_dice   : 1 - 2*sum(p*t)/(sum(p)+sum(t))              — same S points
    L_origin : |pred_origin[q] - gt_origin[k]|_1           — normalized coord

Plus, for unmatched queries:

    L_cls_no_object : CE(class_logits[q], no_object_id)

Aux mask losses on the cheap scales (voxel + fragment) are also computed
per layer — these cost almost nothing and give cleaner gradients to the
mask_embed head.

Per-event API: forward takes one event's predictions + GT and returns a
dict of scalar losses. The model class (Phase 7) loops over batch elements.
"""

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .matcher import HungarianMatcher


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def _matched_bce_loss(pred_logits: torch.Tensor,
                      gt_mask: torch.Tensor) -> torch.Tensor:
    """BCE per matched pair, mean over points. pred_logits/gt_mask both (P, S)."""
    return F.binary_cross_entropy_with_logits(
        pred_logits, gt_mask, reduction="mean",
    )


def _matched_dice_loss(pred_logits: torch.Tensor,
                       gt_mask: torch.Tensor,
                       eps: float = 1.0) -> torch.Tensor:
    """Dice loss per matched pair, mean over P pairs."""
    p = pred_logits.sigmoid()
    num = 2.0 * (p * gt_mask).sum(dim=-1)
    denom = p.sum(dim=-1) + gt_mask.sum(dim=-1)
    return (1.0 - (num + eps) / (denom + eps)).mean()


def _build_gt_masks_sampled(
    gt_truth_indices: List[torch.Tensor],
    sampled_indices: torch.Tensor,
    n_spacepoints: int,
) -> torch.Tensor:
    """
    Build (K, S) GT mask at sampled spacepoints.

    Args:
        gt_truth_indices: list of K tensors (M_k,) long — GT spacepoints per
                          instance
        sampled_indices: (S,) long — sampled spacepoint indices
        n_spacepoints: N (for safety bounds)
    Returns:
        (K, S) float in {0, 1}
    """
    K = len(gt_truth_indices)
    if K == 0:
        return sampled_indices.new_zeros(0, sampled_indices.shape[0],
                                         dtype=torch.float32)
    out = torch.zeros(K, sampled_indices.shape[0], dtype=torch.float32,
                      device=sampled_indices.device)
    sampled_set = sampled_indices  # already long
    # For each GT instance, build a 0/1 vector at sampled positions.
    # Use isin for vectorized membership test.
    for k, idx in enumerate(gt_truth_indices):
        membership = torch.isin(sampled_set, idx)
        out[k] = membership.float()
    return out


def _importance_sample_negatives(
    bg_indices: torch.Tensor,          # (n_bg,) long — non-positive spacepoint ids
    pred_logits_q: torch.Tensor,        # (N,) — model's mask logits for this query
    n_neg_target: int,
    oversample_ratio: float = 3.0,
    importance_ratio: float = 0.75,
) -> torch.Tensor:
    """
    PointRend / Mask2Former-style importance sampling of negatives.

    1. Oversample `oversample_ratio * n_neg_target` random points from the
       background.
    2. Score each sampled point by uncertainty `|sigmoid(logit) - 0.5|`
       (smaller value = closer to 0.5 = more uncertain — the halo around
       the predicted mask).
    3. Take the top `importance_ratio * n_neg_target` MOST UNCERTAIN of
       those.
    4. Top up with `(1 - importance_ratio) * n_neg_target` fresh random
       background points (for coverage diversity).

    Returns indices of length `n_neg_target` (or smaller if the
    background is starved). Detached from autograd — uncertainty is used
    only to PICK points; gradients don't flow through the score.
    """
    device = bg_indices.device
    n_bg = int(bg_indices.numel())
    if n_neg_target <= 0 or n_bg == 0:
        return torch.zeros(0, dtype=torch.long, device=device)

    # (1) oversample
    n_over = min(int(round(oversample_ratio * n_neg_target)), n_bg)
    perm = torch.randperm(n_bg, device=device)[:n_over]
    candidates = bg_indices[perm]

    # (2) uncertainty (no grad)
    with torch.no_grad():
        unc = (pred_logits_q[candidates].sigmoid() - 0.5).abs()
    # Smaller `unc` = more uncertain. Pick smallest n_imp values.

    # (3) pick the importance subset
    n_imp = min(int(round(importance_ratio * n_neg_target)), n_over)
    if n_imp > 0:
        _, imp_idx = torch.topk(unc, n_imp, largest=False)
        sampled_imp = candidates[imp_idx]
    else:
        sampled_imp = torch.zeros(0, dtype=torch.long, device=device)

    # (4) top-up with fresh random (with-replacement is OK for diversity;
    # we tolerate occasional collisions with sampled_imp).
    n_rand = max(0, n_neg_target - n_imp)
    if n_rand > 0 and n_bg > 0:
        perm2 = torch.randperm(n_bg, device=device)[:n_rand]
        sampled_rand = bg_indices[perm2]
    else:
        sampled_rand = torch.zeros(0, dtype=torch.long, device=device)

    return torch.cat([sampled_imp, sampled_rand], dim=0)


def _balanced_point_sample(
    positive_indices: torch.Tensor,   # (M,) long — GT-positive spacepoint ids
    n_spacepoints: int,
    n_sample: int,
    device: torch.device,
    pos_fraction: float = 0.5,
    equalize_pos_neg: bool = False,
) -> torch.Tensor:
    """
    Balanced sample of up to `n_sample` spacepoint indices.

    Two modes:
      - default (`equalize_pos_neg=False`): target `pos_fraction * n_sample`
        positives + the remainder from background. If positives are
        scarce we take all of them and fill the rest with background;
        the realised positive ratio can be lower than `pos_fraction`.
      - `equalize_pos_neg=True`: enforce *exactly* the same number of
        negatives as positives. If only K positives are available, the
        sample size collapses to 2K (smaller than `n_sample`). This is
        what we want for the per-pair mask loss because then the loss is
        always class-balanced — the over-predicting halo we saw at
        `mask_iou=0.03` came from training step having ~30:1 negative-
        positive ratio per pair when we used the default mode.

    Returns a 1-D LongTensor of indices on `device`. The first
    `n_pos_returned` entries are positives; the rest are background.
    Falls back to uniform random when there are no positives.

    Why this matters (cf. design doc §6 / Phase 8 diagnostics): a typical
    GT shower instance has ~100-1000 spacepoints out of ~130k total.
    Uniform random sampling at S=4096 lands only ~3-30 positive points
    per matched pair, which lets the BCE/Dice loss minimize trivially
    by predicting "all zero".
    """
    if n_sample <= 0 or n_spacepoints == 0:
        return torch.zeros(0, dtype=torch.long, device=device)

    n_pos_avail = int(positive_indices.numel())
    if n_pos_avail == 0:
        s = min(n_sample, n_spacepoints)
        return torch.randperm(n_spacepoints, device=device)[:s]

    n_pos_target = min(int(round(pos_fraction * n_sample)), n_pos_avail)

    # Build background index set
    is_pos = torch.zeros(n_spacepoints, dtype=torch.bool, device=device)
    is_pos[positive_indices] = True
    bg_indices = torch.where(~is_pos)[0]
    n_bg_avail = int(bg_indices.numel())

    if equalize_pos_neg:
        # Cap negatives at the number of positives (exact balance).
        n_neg_target = min(n_pos_target, n_bg_avail)
    else:
        n_neg_target = n_sample - n_pos_target
        if n_neg_target > n_bg_avail:
            # Background is scarcer than the target — fill remainder with
            # more positives so the total comes out close to n_sample.
            n_neg_target = n_bg_avail
            n_pos_target = min(n_sample - n_neg_target, n_pos_avail)

    if n_pos_target > 0:
        perm = torch.randperm(n_pos_avail, device=device)[:n_pos_target]
        sampled_pos = positive_indices[perm]
    else:
        sampled_pos = positive_indices.new_zeros(0)

    if n_neg_target > 0:
        perm = torch.randperm(n_bg_avail, device=device)[:n_neg_target]
        sampled_bg = bg_indices[perm]
    else:
        sampled_bg = bg_indices.new_zeros(0)

    return torch.cat([sampled_pos, sampled_bg], dim=0)


def _build_fragment_gt_mask(
    gt_trunk_trackids: List[torch.Tensor],
    fragment_trackid: torch.Tensor,
) -> torch.Tensor:
    """Approximate (K, F) fragment-level GT mask.

    Each GT instance corresponds to one trunk trackid (the dataset
    de-duplicates fragments sharing the same trackid into a single
    instance). A fragment f belongs to GT instance k iff
    fragment_trackid[f] == trunk_trackid[k]. Fragments assigned by
    plurality voting to a *descendant* trackid would normally be their
    own GT instance under the dataset's seen_trunks dedup, so this
    approximation captures the dominant signal.

    Args:
        gt_trunk_trackids: list of K (1,) LongTensors holding the trunk
                           trackid of each GT instance. (Stored as
                           tensors rather than ints so the call-site can
                           keep things on-device.)
        fragment_trackid: (F,) Long
    Returns:
        (K, F) float in {0, 1}.
    """
    K = len(gt_trunk_trackids)
    F_ = fragment_trackid.shape[0]
    if K == 0 or F_ == 0:
        return fragment_trackid.new_zeros(K, F_, dtype=torch.float32)
    out = torch.zeros(K, F_, dtype=torch.float32,
                      device=fragment_trackid.device)
    for k, trunk_t in enumerate(gt_trunk_trackids):
        out[k] = (fragment_trackid == trunk_t.item()).float()
    return out


def _build_voxel_gt_mask(
    gt_truth_indices: List[torch.Tensor],
    voxel_id: torch.Tensor,        # (N,)
    n_voxels: int,
) -> torch.Tensor:
    """(K, V) GT mask at voxel scale: voxel v belongs to instance k if
    any spacepoint with voxel_id == v is in instance k's truth_indices."""
    K = len(gt_truth_indices)
    if K == 0 or n_voxels == 0:
        return voxel_id.new_zeros(K, n_voxels, dtype=torch.float32)
    out = torch.zeros(K, n_voxels, dtype=torch.float32, device=voxel_id.device)
    for k, idx in enumerate(gt_truth_indices):
        if idx.numel() == 0:
            continue
        v_ids = voxel_id[idx].unique()
        out[k, v_ids] = 1.0
    return out


# ---------------------------------------------------------------------------
# Main loss
# ---------------------------------------------------------------------------

class ShowerClusteringLoss(nn.Module):
    """Mask2Former-style set-prediction loss with deep supervision.

    Args:
        num_classes: total number of class slots, including the no_object
                     class (the LAST slot). For our 5 origin types pass 6.
        no_object_class_id: index of the no_object slot (default
                            num_classes - 1).
        weight_class:   weight on the matched + no_object CE loss.
        weight_mask:    weight on the spacepoint BCE loss (sampled S points).
        weight_dice:    weight on the spacepoint Dice loss.
        weight_origin:  weight on the L1 origin loss (matched only).
        weight_aux_mask_voxel:    weight on the voxel-level mask BCE loss.
        weight_aux_mask_fragment: weight on the fragment-level mask BCE loss.
        no_object_weight: down-weight for the no_object class in CE
                          (Mask2Former default 0.1).
        num_sample_points: S — sampled spacepoints per event for the
                          dominant mask losses.
        deep_supervision: apply loss at init + every decoder layer (sum).
        match_layer: which layer's predictions to use for Hungarian
                     matching ("final" = last decoder layer; "init" =
                     pre-decoder predictions). Mask2Former uses final.
    """

    def __init__(
        self,
        num_classes: int,
        no_object_class_id: Optional[int] = None,
        weight_class: float = 2.0,
        weight_mask: float = 5.0,
        weight_dice: float = 5.0,
        weight_origin: float = 1.0,
        weight_aux_mask_voxel: float = 1.0,
        weight_aux_mask_fragment: float = 1.0,
        no_object_weight: float = 0.1,
        num_sample_points: int = 4096,
        deep_supervision: bool = True,
        match_layer: str = "final",
        cost_class: float = 2.0,
        cost_mask: float = 5.0,
        cost_dice: float = 5.0,
        cost_origin: float = 1.0,
        # Phase B — importance sampling for negatives (PointRend / Mask2Former).
        # Default OFF so the existing v4 behavior is preserved. When enabled,
        # the negative half of each per-pair sample is biased toward points
        # where the model's current prediction is most uncertain
        # (sigmoid ≈ 0.5) — i.e., the over-prediction halo around the GT mask.
        use_importance_sampling: bool = False,
        importance_oversample_ratio: float = 3.0,
        importance_ratio: float = 0.75,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.no_object_class_id = (no_object_class_id
                                   if no_object_class_id is not None
                                   else num_classes - 1)
        self.weight_class = float(weight_class)
        self.weight_mask = float(weight_mask)
        self.weight_dice = float(weight_dice)
        self.weight_origin = float(weight_origin)
        self.weight_aux_mask_voxel = float(weight_aux_mask_voxel)
        self.weight_aux_mask_fragment = float(weight_aux_mask_fragment)
        self.num_sample_points = int(num_sample_points)
        self.deep_supervision = bool(deep_supervision)
        if match_layer not in ("final", "init"):
            raise ValueError("match_layer must be 'final' or 'init'")
        self.match_layer = match_layer

        self.use_importance_sampling = bool(use_importance_sampling)
        self.importance_oversample_ratio = float(importance_oversample_ratio)
        self.importance_ratio = float(importance_ratio)

        # CE class weights: down-weight no_object so unmatched queries
        # don't dominate the gradient
        ce_weights = torch.ones(self.num_classes)
        ce_weights[self.no_object_class_id] = float(no_object_weight)
        self.register_buffer("ce_weights", ce_weights)

        self.matcher = HungarianMatcher(
            cost_class=cost_class,
            cost_mask=cost_mask,
            cost_dice=cost_dice,
            cost_origin=cost_origin,
        )

    # -- per-layer loss ----------------------------------------------------

    def _compute_layer_loss(
        self,
        layer_pred: dict,
        gt_classes: torch.Tensor,
        gt_origin: torch.Tensor,
        gt_truth_indices: list,            # list[K] of (M_k,) LongTensors
        gt_masks_voxel: torch.Tensor,
        gt_masks_fragment: torch.Tensor,
        n_spacepoints: int,
        q_idx: np.ndarray,
        k_idx: np.ndarray,
    ) -> Dict[str, torch.Tensor]:
        cls_logits = layer_pred["class_logits"]    # (Q, C)
        origin_pred = layer_pred["origin"]          # (Q, 3)
        sp_mask_pred = layer_pred["mask_logits"]["spacepoint"]  # (Q, N)
        v_mask_pred = layer_pred["mask_logits"]["voxel"]        # (Q, V)
        f_mask_pred = layer_pred["mask_logits"]["fragment"]     # (Q, F)
        Q = cls_logits.shape[0]
        device = cls_logits.device

        # -- class loss ----------------------------------------------------
        # Build target: no_object for all, then write matched class.
        cls_target = torch.full((Q,), self.no_object_class_id,
                                dtype=torch.long, device=device)
        if len(q_idx) > 0:
            q_t = torch.as_tensor(q_idx, dtype=torch.long, device=device)
            k_t = torch.as_tensor(k_idx, dtype=torch.long, device=device)
            cls_target[q_t] = gt_classes[k_t]
        cls_loss = F.cross_entropy(cls_logits, cls_target,
                                   weight=self.ce_weights)

        # If no GTs, mask losses + origin loss are zero
        if len(q_idx) == 0:
            zero = cls_logits.new_zeros(())
            return {
                "cls": cls_loss,
                "mask": zero,
                "dice": zero,
                "origin": zero,
                "aux_mask_voxel": zero,
                "aux_mask_fragment": zero,
            }

        q_t = torch.as_tensor(q_idx, dtype=torch.long, device=device)
        k_t = torch.as_tensor(k_idx, dtype=torch.long, device=device)

        # -- per-pair pos-biased mask + dice on sampled spacepoints --------
        # For each matched pair: take all (up to n_sample/2) of this pair's
        # GT positives, then fill the rest of the n_sample budget with
        # uniform random background points. Reverted from the v3
        # `equalize_pos_neg=True` experiment, which capped negatives at
        # n_pos and dropped the model's negative coverage to ~110 points
        # per matched pair — the over-prediction halo got worse and
        # mask_iou regressed (v2: 0.032 → v3: 0.014).
        # The per-pair structure (loop over P pairs) is kept so we can
        # later swap the random-bg sampler for an importance / PointRend
        # sampler that focuses negatives on the model's halo region.
        n_target_pos = self.num_sample_points // 2
        bce_terms = []
        dice_terms = []
        for p in range(len(q_idx)):
            q_p = int(q_idx[p])
            k_p = int(k_idx[p])
            gt_idx_k = gt_truth_indices[k_p]
            if gt_idx_k.numel() == 0:
                continue

            # Per-pair positive sampling: take all positives up to half the
            # budget. The remaining slots become negatives (importance- or
            # uniform-sampled below).
            n_pos_actual = min(n_target_pos, int(gt_idx_k.numel()))
            if n_pos_actual >= int(gt_idx_k.numel()):
                sampled_pos = gt_idx_k
            else:
                perm = torch.randperm(gt_idx_k.numel(), device=device)
                sampled_pos = gt_idx_k[perm[:n_pos_actual]]
            n_neg_target = self.num_sample_points - n_pos_actual

            # Background indices for this pair
            is_pos = torch.zeros(n_spacepoints, dtype=torch.bool, device=device)
            is_pos[gt_idx_k] = True
            bg_indices = torch.where(~is_pos)[0]

            # Negatives: importance-sampled (if enabled) or uniform random.
            if self.use_importance_sampling and bg_indices.numel() > 0:
                sampled_neg = _importance_sample_negatives(
                    bg_indices=bg_indices,
                    pred_logits_q=sp_mask_pred[q_p],
                    n_neg_target=n_neg_target,
                    oversample_ratio=self.importance_oversample_ratio,
                    importance_ratio=self.importance_ratio,
                )
            else:
                if bg_indices.numel() == 0:
                    sampled_neg = bg_indices.new_zeros(0)
                else:
                    n_neg = min(n_neg_target, int(bg_indices.numel()))
                    perm = torch.randperm(bg_indices.numel(), device=device)
                    sampled_neg = bg_indices[perm[:n_neg]]

            sampled_p = torch.cat([sampled_pos, sampled_neg], dim=0)
            if sampled_p.numel() == 0:
                continue
            pred_p = sp_mask_pred[q_p, sampled_p]                # (S_p,)
            gt_p = pred_p.new_zeros(sampled_p.shape[0])
            gt_p[:sampled_pos.shape[0]] = 1.0
            bce_terms.append(F.binary_cross_entropy_with_logits(
                pred_p, gt_p, reduction="mean"
            ))
            # Dice for this pair
            sigm = pred_p.sigmoid()
            num = 2.0 * (sigm * gt_p).sum()
            denom = sigm.sum() + gt_p.sum()
            dice_terms.append(1.0 - (num + 1.0) / (denom + 1.0))
        if bce_terms:
            mask_loss = torch.stack(bce_terms).mean()
            dice_loss = torch.stack(dice_terms).mean()
        else:
            mask_loss = cls_logits.new_zeros(())
            dice_loss = cls_logits.new_zeros(())

        # -- aux mask losses at voxel + fragment scale --------------------
        # No sampling needed at these scales; use full token sets.
        aux_v_loss = _matched_bce_loss(v_mask_pred[q_t], gt_masks_voxel[k_t])
        aux_f_loss = (_matched_bce_loss(f_mask_pred[q_t], gt_masks_fragment[k_t])
                      if f_mask_pred.shape[1] > 0 else cls_logits.new_zeros(()))

        # -- origin L1 (matched only) -------------------------------------
        # mean over (pairs, axes) so the magnitude doesn't scale with
        # batch size or matched-pair count, and the per-axis units stay
        # interpretable as a normalized-coord L1.
        origin_loss = (origin_pred[q_t] - gt_origin[k_t]).abs().mean()

        return {
            "cls": cls_loss,
            "mask": mask_loss,
            "dice": dice_loss,
            "origin": origin_loss,
            "aux_mask_voxel": aux_v_loss,
            "aux_mask_fragment": aux_f_loss,
        }

    # -- forward (per-event) -----------------------------------------------

    def forward(
        self,
        decoder_output: dict,
        gt_instances: list,
        voxel_id: torch.Tensor,
        n_voxels: int,
        n_spacepoints: int,
        fragment_trackid: torch.Tensor,
        return_matching: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            decoder_output: dict returned by Mask2FormerDecoder
                (keys: 'init', 'layers', 'final')
            gt_instances: list of dicts from ShowerClusteringDataset; each
                must have 'truth_indices' (LongTensor), 'origin_type',
                'origin_coord_norm', and 'trunk_trackid'.
            voxel_id: (N,) long
            n_voxels: V
            n_spacepoints: N (= number of surviving spacepoints)
            fragment_trackid: (F,) long — per-fragment plurality trackid
            return_matching: if True, also include the Hungarian assignment
                (q_idx, k_idx) and the GT tensors used for matching in the
                output dict. The evaluator uses this to compute per-pair
                quality metrics without redoing the match.
        Returns:
            dict with key 'total' (scalar tensor) and per-component
            scalars. With return_matching=True, additionally:
                q_idx, k_idx        — np.int64 arrays of matched pairs
                gt_classes          — (K,) LongTensor on `device`
                gt_origin           — (K, 3) FloatTensor on `device`
                gt_truth_indices    — list[K] of LongTensors on `device`
                sampled_indices     — (S,) LongTensor on `device`
        """
        device = decoder_output["final"]["class_logits"].device
        K = len(gt_instances)

        # GT tensors (build once, shared across layers + matcher)
        if K > 0:
            gt_classes = torch.tensor(
                [int(g["origin_type"]) for g in gt_instances],
                dtype=torch.long, device=device,
            )
            gt_origin = torch.stack([
                torch.as_tensor(g["origin_coord_norm"],
                                dtype=torch.float32, device=device)
                for g in gt_instances
            ], dim=0)  # (K, 3)
            gt_truth_indices = [
                torch.as_tensor(g["truth_indices"],
                                dtype=torch.long, device=device)
                for g in gt_instances
            ]
            # For fragment-mask GT we need each instance's "membership set"
            # of trackids. Mask2Former builds this from descendants. We
            # precomputed descendants implicitly (instance built from
            # fragments sharing this trunk's tid AND its tree descendants).
            # For the loss, the simplest valid signal is: fragments whose
            # plurality trackid is the trunk_trackid OR any descendant.
            # We don't carry full descendants per instance through the
            # dataset's gt_instances dict, so use a relaxed rule:
            # fragment_trackid == trunk_trackid OR (fragment_trackid in
            # truth_indices' trackid set — but that's the same set we
            # already enumerated). We simply use fragment_trackid ==
            # trunk_trackid; descendants from secondary brem etc. that
            # got their own fragments are still independent GT instances
            # in our seen_trunks dedup.
            gt_descendants_per_instance = [
                torch.tensor([int(g["trunk_trackid"])],
                             dtype=torch.long, device=device)
                for g in gt_instances
            ]
        else:
            gt_classes = torch.zeros(0, dtype=torch.long, device=device)
            gt_origin = torch.zeros(0, 3, dtype=torch.float32, device=device)
            gt_truth_indices = []
            gt_descendants_per_instance = []

        # ---- shared (matcher) sampled set -------------------------------
        # Union-balanced sampling: ~50% of points come from any GT
        # instance, ~50% from background. Used by the Hungarian matcher's
        # cost matrix so the per-pair mask cost has discriminative signal.
        if K > 0 and gt_truth_indices:
            union_positives = torch.unique(
                torch.cat(gt_truth_indices, dim=0)
            )
        else:
            union_positives = torch.zeros(0, dtype=torch.long, device=device)
        sampled_indices = _balanced_point_sample(
            positive_indices=union_positives,
            n_spacepoints=n_spacepoints,
            n_sample=self.num_sample_points,
            device=device,
            pos_fraction=0.5,
        )
        gt_masks_sampled = _build_gt_masks_sampled(
            gt_truth_indices, sampled_indices, n_spacepoints,
        )                                              # (K, S)
        gt_masks_voxel = _build_voxel_gt_mask(
            gt_truth_indices, voxel_id, n_voxels,
        )                                              # (K, V)
        gt_masks_fragment = _build_fragment_gt_mask(
            gt_descendants_per_instance, fragment_trackid,
        )                                              # (K, F)

        # Run matcher on the chosen layer's predictions
        match_pred = (decoder_output["final"] if self.match_layer == "final"
                      else decoder_output["init"])
        q_idx, k_idx = self.matcher(
            class_logits=match_pred["class_logits"],
            spacepoint_mask_logits=match_pred["mask_logits"]["spacepoint"],
            origin=match_pred["origin"],
            gt_classes=gt_classes,
            gt_masks_sampled=gt_masks_sampled,
            gt_origin=gt_origin,
            sampled_indices=sampled_indices,
        )

        # NOTE: per-pair mask sampling is now done inside
        # `_compute_layer_loss` for each matched pair (variable per-pair
        # sample size to enforce exact 50/50 pos-vs-neg ratio). The
        # union-balanced `sampled_indices` above is shared by the matcher
        # and the (K, S) gt_masks_sampled tensor it consumes.

        # Apply matching to all supervised layers (init + every layer)
        sup_layers = []
        if self.deep_supervision:
            sup_layers.append(decoder_output["init"])
            sup_layers.extend(decoder_output["layers"])
        else:
            sup_layers.append(decoder_output["final"])

        per_layer_losses = []
        for li, lyr in enumerate(sup_layers):
            l = self._compute_layer_loss(
                layer_pred=lyr,
                gt_classes=gt_classes,
                gt_origin=gt_origin,
                gt_truth_indices=gt_truth_indices,
                gt_masks_voxel=gt_masks_voxel,
                gt_masks_fragment=gt_masks_fragment,
                n_spacepoints=n_spacepoints,
                q_idx=q_idx, k_idx=k_idx,
            )
            per_layer_losses.append(l)

        # Sum across supervised layers, weight per component
        agg = {k: torch.stack([pl[k] for pl in per_layer_losses]).sum()
               for k in per_layer_losses[0].keys()}
        total = (self.weight_class * agg["cls"]
                 + self.weight_mask * agg["mask"]
                 + self.weight_dice * agg["dice"]
                 + self.weight_origin * agg["origin"]
                 + self.weight_aux_mask_voxel * agg["aux_mask_voxel"]
                 + self.weight_aux_mask_fragment * agg["aux_mask_fragment"])
        out = {"total": total}
        for k, v in agg.items():
            out[k] = v
        out["n_matched"] = torch.tensor(
            len(q_idx), dtype=torch.long, device=device,
        )
        out["n_gt_instances"] = torch.tensor(
            K, dtype=torch.long, device=device,
        )
        if return_matching:
            out["q_idx"] = q_idx
            out["k_idx"] = k_idx
            out["gt_classes"] = gt_classes
            out["gt_origin"] = gt_origin
            out["gt_truth_indices"] = gt_truth_indices
            out["sampled_indices"] = sampled_indices
        return out
