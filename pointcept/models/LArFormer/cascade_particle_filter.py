"""Stage 2 → Stage 3 boundary — build the nu-candidate SP set the
particle segmenter consumes.

The Stage-3 particle segmenter operates on a subset of the post-deghoster
spacepoints: those whose membership in a model-predicted nu slice (with a
**loosened** mask-prob threshold τ_loose) is high enough to be plausibly
part of the nu interaction. See `docs/LArFormer_particlesegment_stage.md`
§3 for the design rationale (loosened τ recovers under-clustered SPs that
panoptic-argmax dropped).

Public API:

    build_nu_keep_mask(slicer_predictions, n_sp_per_event,
                       nu_class_id, mask_prob_threshold,
                       spacepoint_level)
        → (N_total,) bool tensor — per-SP "is this SP in any model-nu
        query's above-loose-threshold mask?"

    filter_batch_for_particle_segmenter(filtered_batch, keep_mask,
                                         recenter=True)
        → new batch dict with per-SP tensors / gt_instances filtered to
        the nu set, optionally with `coord_norm` recentered to the
        per-event nu-SP centroid (§1f of the plan).

This module is intentionally thin — `filter_batch_by_keep_mask` (the
Stage-1 → 2 filter) already handles the per-SP / per-GT mechanics. We
add the (a) nu-keep-mask construction from slicer predictions and
(b) per-event recentering on top.
"""

from typing import Sequence

import torch

from .cascade_filter import filter_batch_by_keep_mask


def _prob_to_logit(p: float) -> float:
    """Convert mask probability threshold to logit threshold.

    The decoder's mask_logits are pre-sigmoid; comparing logit > 0 is
    equivalent to p > 0.5. For τ_loose < 0.5 we want logit > log(τ/(1-τ))
    (a negative number, so MORE SPs survive).
    """
    eps = 1e-7
    p_clip = float(min(max(p, eps), 1.0 - eps))
    import math
    return math.log(p_clip / (1.0 - p_clip))


def build_nu_keep_mask(
    slicer_predictions: list,
    n_sp_per_event: torch.Tensor,
    nu_class_id: int,
    mask_prob_threshold: float,
    spacepoint_level: str = "spacepoint",
    device=None,
) -> torch.Tensor:
    """Construct the flat per-SP keep mask for Stage 3 input.

    For each event:
      1. Identify queries with `class_argmax == nu_class_id`.
      2. For each such query, build a per-SP bool mask:
             mask_logits[q] > τ_loose_logit
      3. Union across all nu queries → per-event keep mask.

    Concatenated across events the result is a (N_total,) bool tensor
    aligned with the post-deghoster `data_dict["offset"]`.

    Args:
        slicer_predictions: list of per-event dicts as returned by
                              `LArFormer.forward`'s eval path. Each must
                              expose `class_logits` (Q, C) and
                              `mask_logits` (OrderedDict {level_name:
                              (Q, M_level) tensor}).
        n_sp_per_event:    (B,) tensor of per-event post-deghoster SP
                              counts (= filtered `n_spacepoints`).
        nu_class_id:       which column of class_argmax counts as "nu".
        mask_prob_threshold: τ_loose (probability) — recommend 0.3–0.5.
        spacepoint_level:  level name in mask_logits to read. Default
                              "spacepoint" (the primary SP level the
                              cascade slicer uses).
        device:            torch device for the output mask. Defaults to
                              the first prediction tensor's device.

    Returns:
        (N_total,) bool tensor where True = SP belongs to at least one
        model-nu query at τ_loose.
    """
    if device is None and slicer_predictions:
        # Snag a device handle from the first non-empty prediction.
        for p in slicer_predictions:
            if "class_logits" in p:
                device = p["class_logits"].device
                break
        if device is None:
            device = torch.device("cpu")

    tau_logit = _prob_to_logit(float(mask_prob_threshold))

    per_event_masks = []
    n_sp_list = [int(x) for x in n_sp_per_event.detach().cpu().tolist()]
    for ev, n_sp in enumerate(n_sp_list):
        if n_sp == 0:
            per_event_masks.append(torch.zeros(0, dtype=torch.bool, device=device))
            continue
        if ev >= len(slicer_predictions):
            per_event_masks.append(torch.zeros(n_sp, dtype=torch.bool, device=device))
            continue
        pred = slicer_predictions[ev]
        if "class_logits" not in pred or "mask_logits" not in pred:
            per_event_masks.append(torch.zeros(n_sp, dtype=torch.bool, device=device))
            continue
        cls_logits = pred["class_logits"]                 # (Q, C)
        if cls_logits.shape[0] == 0:
            per_event_masks.append(torch.zeros(n_sp, dtype=torch.bool, device=device))
            continue
        ml = pred["mask_logits"].get(spacepoint_level, None)
        if ml is None or ml.shape[1] != n_sp:
            per_event_masks.append(torch.zeros(n_sp, dtype=torch.bool, device=device))
            continue
        cls_argmax = cls_logits.argmax(dim=-1)             # (Q,)
        nu_q = (cls_argmax == int(nu_class_id))            # (Q,) bool
        if not nu_q.any():
            per_event_masks.append(torch.zeros(n_sp, dtype=torch.bool,
                                                device=ml.device))
            continue
        nu_q_idx = nu_q.nonzero(as_tuple=False).squeeze(-1)
        above = ml[nu_q_idx] > tau_logit                   # (Q_nu, n_sp)
        ev_mask = above.any(dim=0)                         # (n_sp,)
        per_event_masks.append(ev_mask.to(device))

    if not per_event_masks:
        return torch.zeros(0, dtype=torch.bool, device=device)
    return torch.cat(per_event_masks, dim=0)


def build_forced_keep_mask(slicer_predictions, n_sp_per_event, spec, nu_class_id,
                           mask_prob_threshold, spacepoint_level="spacepoint",
                           device=None):
    """Keep mask for a CHOSEN slice, built from THIS forward's slicer predictions
    so it always matches the current filtered batch (avoids cross-forward
    nondeterminism). Single-event (B == 1). `spec`:
      "nu"        -> union of nu-class queries (== build_nu_keep_mask), or
      ("q", qi)   -> just query index qi's mask (regardless of its class), i.e. one
                     cosmic slice selected by stable query slot.
    """
    if spec == "nu":
        return build_nu_keep_mask(slicer_predictions, n_sp_per_event, nu_class_id,
                                  mask_prob_threshold, spacepoint_level, device)
    _, qi = spec
    n_sp = [int(x) for x in n_sp_per_event.detach().cpu().tolist()]
    n_filt = n_sp[0] if n_sp else 0
    pred = slicer_predictions[0] if slicer_predictions else {}
    ml = (pred.get("mask_logits") or {}).get(spacepoint_level)
    if ml is None or ml.shape[1] != n_filt or int(qi) >= ml.shape[0]:
        dev = device or (ml.device if ml is not None else torch.device("cpu"))
        return torch.zeros(n_filt, dtype=torch.bool, device=dev)
    keep = ml[int(qi)] > _prob_to_logit(float(mask_prob_threshold))
    return keep.to(device) if device is not None else keep


def recenter_coord_norm_to_per_event_centroid(
    batch: dict, coord_norm_key: str = "coord_norm",
) -> dict:
    """Subtract the per-event SP centroid from `coord_norm`.

    Does NOT mutate the caller's batch — returns a shallow-copied dict
    with a new `coord_norm` tensor. Other keys (`coord`, `grid_coord`,
    `feat`, etc.) pass through unchanged — only the model's pos_emb
    input is recentered, so flash-match / OOB checks against absolute
    detector cm continue to work downstream.

    No-op for events with 0 SPs (their slice contributes nothing).
    Per-event centroid is computed at fp32 even when the input is fp16,
    then the subtraction casts back.
    """
    if coord_norm_key not in batch:
        return dict(batch)
    coord_norm = batch[coord_norm_key]
    offset = batch["offset"]
    out = dict(batch)
    new_cn = coord_norm.clone()
    prev = 0
    offsets_cpu = [int(x) for x in offset.detach().cpu().tolist()]
    for end in offsets_cpu:
        if end > prev:
            sl = slice(prev, end)
            centroid = coord_norm[sl].float().mean(dim=0)   # (3,) fp32
            new_cn[sl] = (coord_norm[sl].float() - centroid).to(coord_norm.dtype)
        prev = end
    out[coord_norm_key] = new_cn
    return out


def filter_batch_for_particle_segmenter(
    filtered_batch: dict,
    keep_mask: torch.Tensor,
    recenter: bool = True,
    drop_empty_instances: bool = True,
) -> dict:
    """Apply the nu-keep mask to the post-deghoster batch, optionally
    recentering coord_norm to the per-event nu-SP centroid.

    Args:
        filtered_batch:   the post-deghoster batch (= what the Stage-2
                            slicer saw). Typically read from CascadedSlicer's
                            eval output under `filtered_batch`.
        keep_mask:        (N_total,) bool — output of `build_nu_keep_mask`.
        recenter:         apply per-event centroid recentering of
                            `coord_norm`. See §1f of the plan.
        drop_empty_instances: forwarded to `filter_batch_by_keep_mask`.

    Returns:
        New batch dict ready for the Stage-3 LArFormer's forward.
    """
    sub = filter_batch_by_keep_mask(
        filtered_batch, keep_mask,
        drop_empty_instances=drop_empty_instances,
    )
    if recenter:
        sub = recenter_coord_norm_to_per_event_centroid(sub)
    return sub


def slicer_predictions_empty(slicer_predictions: Sequence[dict]) -> bool:
    """True if every per-event prediction lacks usable nu-mask info
    (e.g. the slicer's decoder was disabled / cascade dropped the batch).
    """
    if not slicer_predictions:
        return True
    for p in slicer_predictions:
        if ("class_logits" in p and "mask_logits" in p
                and p["class_logits"].shape[0] > 0):
            return False
    return True
