"""Shared inference helpers for LArFormer slicer + Stage-3 particle segmenter.

This module hosts the per-event prediction extraction code that both
inference CLIs use:

  - `slicer_predict_event(...)` — builds the per-event dict that
    `tools/run_slicer_inference.py` writes per event. Output schema is
    the canonical "slicerpred" schema consumed by
    `visualize_larformer_gt.py` (and any analyzer that walks per-event
    HDF5).

  - `stage3_predict_event(...)` — Stage-3 analog. Same panoptic-argmax
    assignment rule for SPs, plus particle-level extras: origin
    prediction (per matched pair), origin-error stats, per-SP particle
    class id GT (from the augmented cache's `particle_class_id`).

  - `write_event_h5(...)` / `load_event_h5(...)` — the on-disk format
    used by both inference paths and consumed by the visualizers.
    Per-event H5 with group structure mirroring the dict's `group/key`
    prefixes (`pre/...`, `post/...`, `queries/...`, `gt/...`,
    `stage3/...`, `stage3_queries/...`, etc.).

The slicer functions were extracted verbatim from
`tools/run_slicer_inference.py` so the slicer CLI's behavior is
unchanged after the refactor. Stage-3 functions mirror the slicer
structure key-for-key under `stage3/` prefixes so a stage3pred file
written in full-cascade mode is also a valid slicerpred file (the
slicer keys are at the top level).
"""
from __future__ import annotations

import os
from typing import Optional, Sequence

import h5py
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Per-SP panoptic-argmax assignment (shared by both stages)
# ---------------------------------------------------------------------------

def per_sp_predicted_slice(
    sp_mask_logits: torch.Tensor,        # (Q, N_post)
    cls_argmax: torch.Tensor,             # (Q,) — argmax class per query
    no_object_class_id: int,
    q_idx_matched: np.ndarray,            # matched queries (np int)
    k_idx_matched: np.ndarray,            # matched GT idx per matched query
    primary_trackid: np.ndarray,          # (K,) GT slice IDs (primary_trackid)
):
    """For each spacepoint, return:
        pred_query (N,)      argmax-mask-logit query among "active" (non-no_object) queries
        pred_class (N,)      class of that query (no_object_class_id if no active query)
        pred_slice_id (N,)   GT primary_trackid of the matched GT for that query,
                             or -1 if the query isn't matched (so we don't have a
                             GT-side slice ID to point at) or if it's no_object

    This is the panoptic Mask2Former-style inference rule. Works equally
    for the slicer (where "slice id" = primary_trackid of the matched
    slice instance) and Stage 3 (where "slice id" = primary_trackid of
    the matched particle instance).

    When no GT is available (K = 0), pred_slice_id is all -1 — the
    pred_query / pred_class outputs are still meaningful (they encode
    the panoptic assignment without a downstream GT lookup).
    """
    Q, N = sp_mask_logits.shape
    active = (cls_argmax != no_object_class_id)
    if not active.any():
        # No active queries → assign everything to "unassigned"
        return (
            np.full(N, -1, dtype=np.int64),
            np.full(N, int(no_object_class_id), dtype=np.int64),
            np.full(N, -1, dtype=np.int64),
        )
    # Among active queries only, argmax over Q. Inactive queries get logit=-inf.
    masked = sp_mask_logits.clone()
    masked[~active] = float("-inf")
    pred_q = masked.argmax(dim=0).cpu().numpy().astype(np.int64)
    pred_c = cls_argmax[torch.as_tensor(pred_q, device=cls_argmax.device)].cpu().numpy().astype(np.int64)

    # Build query → matched_gt_idx lookup (np)
    q_to_k = np.full(Q, -1, dtype=np.int64)
    if len(q_idx_matched) > 0:
        q_to_k[q_idx_matched] = k_idx_matched

    matched_k = q_to_k[pred_q]                       # (N,) GT idx per SP (or -1)
    if primary_trackid.size == 0:
        # No GT instances → no matched_k can be >= 0 anyway, but
        # `primary_trackid[matched_k.clip(min=0)]` crashes on the
        # empty index. Short-circuit to all-unassigned.
        pred_slice = np.full(N, -1, dtype=np.int64)
    else:
        pred_slice = np.where(
            matched_k >= 0,
            primary_trackid[matched_k.clip(min=0)],   # safe lookup
            -1,
        ).astype(np.int64)
    return pred_q, pred_c, pred_slice


# ---------------------------------------------------------------------------
# Per-pair mask IoU (binarize at logit > 0)
# ---------------------------------------------------------------------------

def per_pair_iou(
    sp_mask_logits: torch.Tensor,     # (Q, N_post)
    gt_truth_indices: list,            # list[K] of LongTensors
    q_idx: np.ndarray,
    k_idx: np.ndarray,
):
    device = sp_mask_logits.device
    P = len(q_idx)
    if P == 0:
        return np.zeros(0, dtype=np.float32)
    N = sp_mask_logits.shape[1]
    out = np.zeros(P, dtype=np.float32)
    pred_bool = sp_mask_logits > 0.0
    for p in range(P):
        q = int(q_idx[p]); k = int(k_idx[p])
        gt_idx = gt_truth_indices[k].to(device)
        gt_mask = torch.zeros(N, dtype=torch.bool, device=device)
        if gt_idx.numel() > 0:
            gt_mask[gt_idx] = True
        pm = pred_bool[q]
        inter = (pm & gt_mask).sum().item()
        union = (pm | gt_mask).sum().item()
        out[p] = (inter / union) if union > 0 else 0.0
    return out


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def _coerce_gt_list(raw_gt_list, n_expected: int) -> list:
    """Return the list whose length is `n_expected` (the cascade's
    post-filter GT instance count). Falls back to `raw_gt_list` if it
    already matches that length.
    """
    if raw_gt_list is None:
        return [{} for _ in range(n_expected)]
    if len(raw_gt_list) == n_expected:
        return list(raw_gt_list)
    # Length mismatch — defensive fallback (shouldn't happen since the
    # cascade now exposes `filtered_gt_instances_per_event`).
    return list(raw_gt_list)[:n_expected] + [
        {} for _ in range(max(n_expected - len(raw_gt_list), 0))
    ]


def _recover_affine(pre_coord: np.ndarray, pre_coord_norm: np.ndarray):
    """Recover (scale, center) so that `coord = coord_norm * scale + center`
    by linear fit to a per-event pair of (coord, coord_norm) arrays.

    Returns (scale, center) as float32 (3,) arrays, or (None, None) if
    there aren't enough points.
    """
    if pre_coord.shape[0] <= 1:
        return None, None
    scale = (pre_coord.max(0) - pre_coord.min(0)) / np.maximum(
        pre_coord_norm.max(0) - pre_coord_norm.min(0), 1e-9)
    center = pre_coord.mean(0) - pre_coord_norm.mean(0) * scale
    return scale.astype(np.float32), center.astype(np.float32)


# ---------------------------------------------------------------------------
# Slicer event extractor
# ---------------------------------------------------------------------------

def slicer_predict_event(
    model,
    sample: dict,
    batched: dict,
    no_object_class_id: int,
) -> dict:
    """Run the cascade on one event; return a dict of numpy arrays / scalars
    ready to write to HDF5. All per-SP fields are aligned so a downstream
    viz tool can join pre-filter and post-filter views by index easily.

    NOTE: the slicer's primary mask is at the spacepoint level of the
    POST-filter batch. The pre-filter SP indexing is in `sample`'s order
    (which is what the dataset emitted); the post-filter SP indexing is
    its subset (in original-order) of survivors.
    """
    with torch.no_grad():
        out = model(batched)
    return slicer_predict_event_from_out(out, sample, no_object_class_id)


def slicer_predict_event_from_out(
    out: dict,
    sample: dict,
    no_object_class_id: int,
) -> dict:
    """Same as `slicer_predict_event` but accepts an already-computed
    cascade output dict. Used by the full-cascade Stage-3 inference path
    to avoid running the slicer twice."""
    n_pre = int(sample["n_spacepoints"])
    preds = out["predictions"]
    assert len(preds) == 1, f"expected single-event batch; got {len(preds)} preds"
    ev_pred = preds[0]

    # Pre-filter deghoster outputs
    p_real_pre = out.get("deghost_p_real")
    if p_real_pre is not None:
        p_real_pre = p_real_pre.detach().cpu().numpy().astype(np.float32)
    tau = float(out.get("deghost_tau", float("nan")))
    keep_frac = float(out.get("deghost_keep_frac", float("nan")))

    # Did the cascade drop this event? (drop_empty_events emptied predictions)
    if ev_pred is None or "class_logits" not in ev_pred:
        return {
            "event_dropped": True,
            "pre/coord": sample["coord"].astype(np.float32),
            "pre/hasmatch": sample.get("hasmatch", np.zeros(n_pre, dtype=np.int64)).astype(np.int64),
            "pre/slice_id_gt": sample.get("slice_id", -np.ones(n_pre, dtype=np.int64)).astype(np.int64),
            "pre/lm_score": sample.get("lm_score", np.zeros(n_pre, dtype=np.float32)).astype(np.float32),
            "pre/p_real": p_real_pre if p_real_pre is not None else np.zeros(n_pre, dtype=np.float32),
            "meta/tau": tau, "meta/keep_frac": keep_frac,
            "meta/n_sp_pre": n_pre, "meta/n_sp_post": 0,
            "meta/run": int(sample.get("run", -1)),
            "meta/subrun": int(sample.get("subrun", -1)),
            "meta/event": int(sample.get("event", -1)),
            "meta/name": str(sample.get("name", "")),
        }

    cls_logits = ev_pred["class_logits"]              # (Q, C)
    sp_mask = ev_pred["mask_logits"]["spacepoint"]    # (Q, N_post)
    eval_loss = ev_pred.get("eval_loss")
    has_gt = eval_loss is not None

    if has_gt:
        q_idx = eval_loss["q_idx"]                         # np.int64 (P,)
        k_idx = eval_loss["k_idx"]                         # np.int64 (P,)
        gt_classes = eval_loss["gt_classes"]               # (K,) long
        gt_truth_indices = eval_loss["gt_truth_indices"]   # list[K] long
    else:
        q_idx = np.zeros(0, dtype=np.int64)
        k_idx = np.zeros(0, dtype=np.int64)
        gt_classes = cls_logits.new_zeros(0, dtype=torch.long)
        gt_truth_indices = []

    Q, _ = cls_logits.shape
    K = int(gt_classes.shape[0])
    n_post = sp_mask.shape[1]
    cls_argmax = cls_logits.argmax(dim=-1)             # (Q,) long
    cls_probs = cls_logits.softmax(dim=-1).cpu().numpy().astype(np.float32)

    # Post-filter SP fields. The cascade returns `levels[name].coords`
    # (normalized) and `sp_to_level_id` per level. The spacepoint level's
    # coords ARE the post-filter coord_norm (identity sp_to_level_id).
    sp_level = ev_pred["levels"]["spacepoint"]
    coord_norm_post = sp_level["coords"].detach().cpu().numpy().astype(np.float32)
    pre_coord = sample["coord"].astype(np.float32)
    pre_coord_norm = sample["coord_norm"].astype(np.float32)
    scale, center = _recover_affine(pre_coord, pre_coord_norm)
    if scale is not None:
        coord_post = (coord_norm_post * scale + center).astype(np.float32)
    else:
        coord_post = coord_norm_post

    # Per-GT-instance fields. Use the POST-filter gt_instances exposed by
    # CascadedSlicer (`filtered_gt_instances_per_event`) so the indexing
    # matches `K = gt_classes.shape[0]` from the slicer's eval_loss.
    filtered_gt_per_event = out.get("filtered_gt_instances_per_event", None)
    filtered_gt = (filtered_gt_per_event[0]
                   if filtered_gt_per_event and len(filtered_gt_per_event) > 0
                   else sample.get("gt_instances", []))
    filtered_gt = _coerce_gt_list(filtered_gt, K)
    primary_trackid_gt = np.array(
        [int(g.get("primary_trackid", -1)) for g in filtered_gt],
        dtype=np.int64,
    )
    origin_type_gt = np.array(
        [int(g.get("origin_type", -1)) for g in filtered_gt],
        dtype=np.int64,
    )
    primary_origin_gt = np.array(
        [int(g.get("primary_origin", -1)) for g in filtered_gt],
        dtype=np.int64,
    )
    n_truth_points_gt = np.array(
        [int(g.get("n_truth_points", 0)) for g in filtered_gt],
        dtype=np.int64,
    )
    origin_coord_norm_gt = np.zeros((K, 3), dtype=np.float32)
    for i, g in enumerate(filtered_gt):
        if "origin_coord_norm" in g:
            origin_coord_norm_gt[i] = np.asarray(
                g["origin_coord_norm"], dtype=np.float32,
            )

    # Per-pair IoU + matching bookkeeping
    pair_iou = per_pair_iou(sp_mask, gt_truth_indices, q_idx, k_idx)
    pair_gt_class = gt_classes[torch.as_tensor(k_idx, device=gt_classes.device)] \
        if len(k_idx) > 0 else gt_classes.new_zeros(0)
    pair_pred_class = cls_argmax[torch.as_tensor(q_idx, device=cls_argmax.device)] \
        if len(q_idx) > 0 else cls_argmax.new_zeros(0)
    pair_cls_correct = (pair_pred_class == pair_gt_class).cpu().numpy().astype(bool) \
        if len(q_idx) > 0 else np.zeros(0, dtype=bool)

    pred_mask_bool_per_gt = np.zeros((K, n_post), dtype=bool)
    pred_n_pts_per_gt = np.zeros(K, dtype=np.int64) - 1
    for p in range(len(q_idx)):
        q_p = int(q_idx[p]); k_p = int(k_idx[p])
        m = (sp_mask[q_p] > 0).cpu().numpy()
        pred_mask_bool_per_gt[k_p] = m
        pred_n_pts_per_gt[k_p] = int(m.sum())

    query_matched_gt = np.full(Q, -1, dtype=np.int64)
    if len(q_idx) > 0:
        query_matched_gt[q_idx] = k_idx
    gt_matched_query = np.full(K, -1, dtype=np.int64)
    if len(k_idx) > 0:
        gt_matched_query[k_idx] = q_idx

    pred_query, pred_class, pred_slice_id = per_sp_predicted_slice(
        sp_mask_logits=sp_mask,
        cls_argmax=cls_argmax,
        no_object_class_id=no_object_class_id,
        q_idx_matched=q_idx, k_idx_matched=k_idx,
        primary_trackid=primary_trackid_gt,
    )

    pred_q_t = torch.as_tensor(pred_query, dtype=torch.long, device=sp_mask.device)
    if n_post > 0:
        pred_mask_prob = torch.sigmoid(
            sp_mask.gather(0, pred_q_t.unsqueeze(0)).squeeze(0)
        ).cpu().numpy().astype(np.float32)
    else:
        pred_mask_prob = np.zeros(0, dtype=np.float32)

    pre_slice_id_gt = sample.get("slice_id", -np.ones(n_pre, dtype=np.int64)).astype(np.int64)
    if p_real_pre is not None:
        keep_mask_pre = p_real_pre > tau
    else:
        keep_mask_pre = np.ones(n_pre, dtype=bool)
    post_slice_id_gt = pre_slice_id_gt[keep_mask_pre]
    if post_slice_id_gt.shape[0] != n_post:
        post_slice_id_gt = np.resize(post_slice_id_gt, n_post)

    # Per-voxel-level predicted slice assignment.
    levels_payload: dict = {}
    levels_dict = ev_pred.get("levels", {}) or {}
    mask_logits_by_level = ev_pred.get("mask_logits", {}) or {}
    for lvl_name, lvl in levels_dict.items():
        if lvl_name == "spacepoint":
            continue
        if lvl_name not in mask_logits_by_level:
            continue
        coords_norm_lvl = lvl["coords"].detach().cpu().numpy().astype(np.float32)
        if coords_norm_lvl.shape[0] == 0:
            continue
        if scale is not None:
            coords_cm_lvl = (coords_norm_lvl * scale + center).astype(np.float32)
        else:
            coords_cm_lvl = coords_norm_lvl
        mask_logits_lvl = mask_logits_by_level[lvl_name]
        pq_lvl, pc_lvl, ps_lvl = per_sp_predicted_slice(
            sp_mask_logits=mask_logits_lvl,
            cls_argmax=cls_argmax,
            no_object_class_id=no_object_class_id,
            q_idx_matched=q_idx, k_idx_matched=k_idx,
            primary_trackid=primary_trackid_gt,
        )
        # `pq_lvl` is -1 wherever no active query won the panoptic
        # argmax (untrained / weakly-trained models, or pure no_object
        # events). gather(0, idx) needs non-negative indices, so we
        # substitute 0 for the read and zero the result for those SPs.
        pq_safe = pq_lvl.copy()
        pq_safe[pq_safe < 0] = 0
        pq_lvl_t = torch.as_tensor(pq_safe, dtype=torch.long,
                                   device=mask_logits_lvl.device)
        pmp_lvl = torch.sigmoid(
            mask_logits_lvl.gather(0, pq_lvl_t.unsqueeze(0)).squeeze(0)
        ).cpu().numpy().astype(np.float32)
        pmp_lvl[pq_lvl < 0] = 0.0
        sp_to_lvl = lvl.get("sp_to_level_id")
        if sp_to_lvl is not None:
            sp_to_lvl_np = sp_to_lvl.detach().cpu().numpy().astype(np.int64)
        else:
            sp_to_lvl_np = np.full(post_slice_id_gt.shape[0], -1, dtype=np.int64)
        gt_per_voxel = np.full(coords_norm_lvl.shape[0], -1, dtype=np.int64)
        for v in range(coords_norm_lvl.shape[0]):
            members = post_slice_id_gt[sp_to_lvl_np == v]
            members = members[members >= 0]
            if members.size == 0:
                continue
            vals, counts = np.unique(members, return_counts=True)
            gt_per_voxel[v] = int(vals[counts.argmax()])
        levels_payload.update({
            f"levels/{lvl_name}/coord": coords_cm_lvl,
            f"levels/{lvl_name}/coord_norm": coords_norm_lvl,
            f"levels/{lvl_name}/pred_query": pq_lvl,
            f"levels/{lvl_name}/pred_class": pc_lvl,
            f"levels/{lvl_name}/pred_slice_id": ps_lvl,
            f"levels/{lvl_name}/pred_mask_prob": pmp_lvl,
            f"levels/{lvl_name}/slice_id_gt": gt_per_voxel,
            f"levels/{lvl_name}/pred_correct":
                (ps_lvl == gt_per_voxel).astype(bool),
        })

    pred_correct = (pred_slice_id == post_slice_id_gt).astype(bool)

    argmax_iou_per_pair = np.zeros(len(q_idx), dtype=np.float32)
    for p in range(len(q_idx)):
        q_p = int(q_idx[p]); k_p = int(k_idx[p])
        tid_p = int(primary_trackid_gt[k_p])
        pred_argmax = (pred_query == q_p) & (pred_mask_prob >= 0.5)
        gt_mask_np = (post_slice_id_gt == tid_p)
        inter = int((pred_argmax & gt_mask_np).sum())
        union = int((pred_argmax | gt_mask_np).sum())
        argmax_iou_per_pair[p] = (inter / union) if union > 0 else 0.0

    out_dict = {
        "event_dropped": False,

        # Pre-filter: raw input that the deghoster saw
        "pre/coord": pre_coord,
        "pre/coord_norm": pre_coord_norm,
        "pre/hasmatch": sample.get("hasmatch", np.zeros(n_pre, dtype=np.int64)).astype(np.int64),
        "pre/slice_id_gt": pre_slice_id_gt,
        "pre/lm_score": sample.get("lm_score", np.zeros(n_pre, dtype=np.float32)).astype(np.float32),
        "pre/pixval": sample.get("feat", np.zeros((n_pre, 6), dtype=np.float32))[:, 3:6].astype(np.float32),
        "pre/ssnet_label": sample.get("ssnet_label", -np.ones(n_pre, dtype=np.int64)).astype(np.int64),
        "pre/p_real": p_real_pre if p_real_pre is not None else np.zeros(n_pre, dtype=np.float32),
        "pre/keep": keep_mask_pre.astype(bool),

        # Post-filter: slicer's input
        "post/coord": coord_post,
        "post/coord_norm": coord_norm_post,
        "post/slice_id_gt": post_slice_id_gt.astype(np.int64),
        "post/pred_query": pred_query,
        "post/pred_class": pred_class,
        "post/pred_slice_id": pred_slice_id,
        "post/pred_mask_prob": pred_mask_prob,
        "post/pred_correct": pred_correct,

        # Per-query
        "queries/class_logits": cls_logits.detach().cpu().numpy().astype(np.float32),
        "queries/class_probs": cls_probs,
        "queries/class_argmax": cls_argmax.cpu().numpy().astype(np.int64),
        "queries/matched_gt_idx": query_matched_gt,

        # Per-GT-instance
        "gt/primary_trackid": primary_trackid_gt,
        "gt/origin_type": origin_type_gt,
        "gt/primary_origin": primary_origin_gt,
        "gt/n_truth_points": n_truth_points_gt,
        "gt/origin_coord_norm": origin_coord_norm_gt,
        "gt/matched_query": gt_matched_query,
        "gt/pair_iou": np.concatenate([
            pair_iou,
            np.full(max(K - len(pair_iou), 0), -1.0, dtype=np.float32),
        ])[:K],
        "gt/pred_n_pts": pred_n_pts_per_gt,
        "gt/pred_mask_bool": pred_mask_bool_per_gt,
        "gt/argmax_iou": np.concatenate([
            argmax_iou_per_pair,
            np.full(max(K - len(argmax_iou_per_pair), 0), -1.0, dtype=np.float32),
        ])[:K],
        "gt/pair_cls_correct": np.concatenate([
            pair_cls_correct.astype(np.int8),
            np.full(max(K - len(pair_cls_correct), 0), -1, dtype=np.int8),
        ])[:K].astype(np.int8),

        # Identity + summary
        "meta/run": int(sample.get("run", -1)),
        "meta/subrun": int(sample.get("subrun", -1)),
        "meta/event": int(sample.get("event", -1)),
        "meta/name": str(sample.get("name", "")),
        "meta/tau": tau,
        "meta/keep_frac": keep_frac,
        "meta/n_sp_pre": int(n_pre),
        "meta/n_sp_post": int(n_post),
        "meta/n_queries": int(Q),
        "meta/n_gt_instances": int(K),
        "meta/n_matched": int(len(q_idx)),
        "meta/no_object_class_id": int(no_object_class_id),
        "meta/has_gt": int(has_gt),
    }
    out_dict.update(levels_payload)
    if levels_payload:
        emitted = sorted({k.split("/")[1] for k in levels_payload})
        out_dict["meta/voxel_levels"] = ",".join(emitted)
    return out_dict


# ---------------------------------------------------------------------------
# Stage-3 event extractor
# ---------------------------------------------------------------------------

def stage3_predict_event(
    model,
    sample: dict,
    batched: dict,
    no_object_class_id: int,
    *,
    class_prob_threshold: float = 0.0,
    coord_scale: float = 179.55,
) -> dict:
    """Run the Stage-3 particle segmenter on one event; return a dict
    of numpy arrays / scalars ready to write to HDF5.

    All keys are namespaced under `stage3/`, `stage3_queries/`, `stage3_gt/`,
    `stage3_levels/`, `stage3_meta/` so a single H5 file can carry BOTH
    slicer outputs (top-level `pre/`, `post/`, …) and Stage-3 outputs
    (under `stage3*/`).

    Args:
        model:               LArFormer (the particle segmenter — NOT the
                              cascade wrapper).
        sample:              the dict the dataset's __getitem__ returns
                              (cache-mode: from LArFormerStage12CacheDataset;
                              full-cascade: from filter_batch_for_particle_segmenter).
        batched:             larformer_collate output for a 1-event batch.
        no_object_class_id:  the no_object slot in the cls head.
        class_prob_threshold: queries whose max softmax cls probability is
                              below this floor are demoted to no_object
                              for the per-SP panoptic assignment. Default
                              0.0 = strict argmax (no floor).
        coord_scale:         used to convert coord_norm origin predictions
                              into cm for the diagnostics.
    """
    with torch.no_grad():
        out = model(batched)
    return stage3_predict_event_from_out(
        out, sample, no_object_class_id,
        class_prob_threshold=class_prob_threshold,
        coord_scale=coord_scale,
    )


def stage3_predict_event_from_out(
    out: dict,
    sample: dict,
    no_object_class_id: int,
    *,
    class_prob_threshold: float = 0.0,
    coord_scale: float = 179.55,
    pre_coord_for_affine: Optional[np.ndarray] = None,
    pre_coord_norm_for_affine: Optional[np.ndarray] = None,
) -> dict:
    """Same as `stage3_predict_event` but accepts an already-computed
    particle-segmenter output dict. Used by the full-cascade path
    where the model forward has already happened.

    `pre_coord_for_affine` / `pre_coord_norm_for_affine` let the caller
    pass detector-frame coords for affine recovery. In cached mode the
    sample's own coord / coord_norm are post-recenter, so they don't
    recover the affine to detector cm cleanly — we use `coord_scale`
    + the sample's `coord_center` as a fallback.
    """
    n_stage3 = int(sample["n_spacepoints"])
    preds = out.get("predictions", [])
    assert len(preds) == 1, f"expected single-event batch; got {len(preds)} preds"
    ev_pred = preds[0]

    # Empty / dropped event
    if ev_pred is None or "class_logits" not in ev_pred:
        return {
            "stage3_meta/event_dropped": int(True),
            "stage3_meta/n_stage3_sp": n_stage3,
            "stage3_meta/run": int(sample.get("run", -1)),
            "stage3_meta/subrun": int(sample.get("subrun", -1)),
            "stage3_meta/event": int(sample.get("event", -1)),
            "stage3_meta/name": str(sample.get("name", "")),
            "stage3_meta/no_object_class_id": int(no_object_class_id),
        }

    cls_logits = ev_pred["class_logits"]              # (Q, C)
    sp_mask = ev_pred["mask_logits"]["spacepoint"]    # (Q, N_stage3)
    origin_q = ev_pred.get("origin")                  # (Q, 3) coord_norm

    Q, C = cls_logits.shape
    cls_argmax = cls_logits.argmax(dim=-1)            # (Q,)
    cls_probs = cls_logits.softmax(dim=-1)            # (Q, C)
    cls_max_prob = cls_probs.max(dim=-1).values.cpu().numpy().astype(np.float32)
    cls_argmax_np = cls_argmax.cpu().numpy().astype(np.int64)

    # Apply Q3's confidence floor: demote weak queries to no_object for
    # the panoptic assignment. cls_argmax stays unchanged for the
    # `stage3_queries/class_argmax` diagnostic; effective_argmax is
    # what the per-SP rule uses.
    if class_prob_threshold > 0.0:
        weak = cls_max_prob < float(class_prob_threshold)
        effective_argmax = cls_argmax.clone()
        effective_argmax[torch.as_tensor(weak, device=cls_argmax.device)] = (
            int(no_object_class_id))
    else:
        effective_argmax = cls_argmax

    # GT bookkeeping (when present)
    eval_loss = ev_pred.get("eval_loss")
    has_gt = eval_loss is not None
    if has_gt:
        q_idx = eval_loss["q_idx"]                    # np.int64 (P,)
        k_idx = eval_loss["k_idx"]                    # np.int64 (P,)
        gt_classes = eval_loss["gt_classes"]          # (K,) long
        gt_truth_indices = eval_loss["gt_truth_indices"]
        gt_origin = eval_loss.get("gt_origin")        # (K, 3) coord_norm or None
    else:
        q_idx = np.zeros(0, dtype=np.int64)
        k_idx = np.zeros(0, dtype=np.int64)
        gt_classes = cls_logits.new_zeros(0, dtype=torch.long)
        gt_truth_indices = []
        gt_origin = None

    K = int(gt_classes.shape[0])
    n_post = sp_mask.shape[1]

    # Per-instance metadata from the post-cascade-filter GT instances.
    sample_gt_instances = sample.get("gt_instances", [])
    filtered_gt = _coerce_gt_list(sample_gt_instances, K)
    primary_trackid_gt = np.array(
        [int(g.get("primary_trackid", -1)) for g in filtered_gt],
        dtype=np.int64,
    )
    class_id_gt = np.array(
        [int(g.get("class_id", -1)) for g in filtered_gt],
        dtype=np.int64,
    )
    pid_gt = np.array(
        [int(g.get("pid", 0)) for g in filtered_gt],
        dtype=np.int64,
    )
    ke_mev_gt = np.array(
        [float(g.get("ke_mev", 0.0)) for g in filtered_gt],
        dtype=np.float32,
    )
    n_truth_points_gt = np.array(
        [int(g.get("n_truth_points", 0)) for g in filtered_gt],
        dtype=np.int64,
    )
    origin_cm_gt = np.zeros((K, 3), dtype=np.float32)
    origin_coord_norm_gt = np.zeros((K, 3), dtype=np.float32)
    for i, g in enumerate(filtered_gt):
        if "origin_cm" in g:
            origin_cm_gt[i] = np.asarray(g["origin_cm"], dtype=np.float32)
        if "origin_coord_norm" in g:
            origin_coord_norm_gt[i] = np.asarray(
                g["origin_coord_norm"], dtype=np.float32)

    # Per-SP panoptic assignment (same rule as slicer)
    pred_query, pred_class, pred_particle_tid = per_sp_predicted_slice(
        sp_mask_logits=sp_mask,
        cls_argmax=effective_argmax,
        no_object_class_id=no_object_class_id,
        q_idx_matched=q_idx, k_idx_matched=k_idx,
        primary_trackid=primary_trackid_gt,
    )
    # Map matched-GT-index for each SP (lets the visualizer color by
    # GT instance index rather than by primary_trackid).
    q_to_k = np.full(Q, -1, dtype=np.int64)
    if len(q_idx) > 0:
        q_to_k[q_idx] = k_idx
    pred_particle_idx = q_to_k[pred_query.clip(min=0)]
    pred_particle_idx[pred_query < 0] = -1

    # Per-SP mask probability of assigned query
    if n_stage3 > 0:
        pred_q_safe = pred_query.copy()
        pred_q_safe[pred_q_safe < 0] = 0
        pred_q_t = torch.as_tensor(pred_q_safe, dtype=torch.long,
                                   device=sp_mask.device)
        pred_mask_prob = torch.sigmoid(
            sp_mask.gather(0, pred_q_t.unsqueeze(0)).squeeze(0)
        ).cpu().numpy().astype(np.float32)
        pred_mask_prob[pred_query < 0] = 0.0
    else:
        pred_mask_prob = np.zeros(0, dtype=np.float32)

    # Per-SP GT class id (from particle_class_id), if available
    particle_class_id_gt = sample.get("particle_class_id",
                                      -np.ones(n_stage3, dtype=np.int64))
    particle_class_id_gt = np.asarray(particle_class_id_gt, dtype=np.int64)

    # Per-pair stats: IoU + class correctness + origin error
    pair_iou = per_pair_iou(sp_mask, gt_truth_indices, q_idx, k_idx)
    if len(q_idx) > 0:
        pair_pred_class = cls_argmax[torch.as_tensor(q_idx, device=cls_argmax.device)]
        pair_gt_class = gt_classes[torch.as_tensor(k_idx, device=gt_classes.device)]
        pair_cls_correct = (pair_pred_class == pair_gt_class).cpu().numpy().astype(bool)
    else:
        pair_cls_correct = np.zeros(0, dtype=bool)

    # Origin pred / GT diagnostics. The model predicts origin in
    # coord_norm space; we report both coord_norm and cm for
    # visualization convenience.
    if origin_q is not None:
        origin_pred_norm = origin_q.detach().cpu().numpy().astype(np.float32)
    else:
        origin_pred_norm = np.zeros((Q, 3), dtype=np.float32)
    # Origin error per matched pair (cm). When recentering is active
    # in the dataset, both pred and gt live in the SAME recentered
    # frame; the L2 difference is translation-invariant, so the cm
    # error is honest regardless.
    pair_origin_l2_cm = np.full(len(q_idx), -1.0, dtype=np.float32)
    if origin_q is not None and gt_origin is not None and len(q_idx) > 0:
        diff = (origin_q[torch.as_tensor(q_idx, device=origin_q.device)]
                - gt_origin[torch.as_tensor(k_idx, device=gt_origin.device)])
        pair_origin_l2_cm = (diff.norm(dim=-1) * float(coord_scale)).cpu().numpy().astype(np.float32)

    # Recover affine for cm coordinates of the Stage-3 SPs. In cached
    # mode the sample's coord_norm is post-recenter, so the affine
    # recovered from (coord, coord_norm) will NOT map back to detector
    # cm — it'll be in the recentered frame. That's fine for
    # visualization (everything is in the same frame); the visualizer
    # has its own translation handling.
    pre_coord = sample["coord"].astype(np.float32)
    pre_coord_norm = sample["coord_norm"].astype(np.float32)
    # `coord_post` here = the same SP coords that come from the cache
    # reader (already detector cm).
    coord_post = pre_coord.copy()
    coord_norm_post = pre_coord_norm.copy()

    # Voxel-level per-token panoptic assignment (mirrors the slicer's
    # `levels/...` block but under `stage3_levels/`).
    scale, center = _recover_affine(pre_coord, pre_coord_norm)
    levels_payload: dict = {}
    levels_dict = ev_pred.get("levels", {}) or {}
    mask_logits_by_level = ev_pred.get("mask_logits", {}) or {}
    for lvl_name, lvl in levels_dict.items():
        if lvl_name == "spacepoint":
            continue
        if lvl_name not in mask_logits_by_level:
            continue
        coords_norm_lvl = lvl["coords"].detach().cpu().numpy().astype(np.float32)
        if coords_norm_lvl.shape[0] == 0:
            continue
        if scale is not None:
            coords_cm_lvl = (coords_norm_lvl * scale + center).astype(np.float32)
        else:
            coords_cm_lvl = coords_norm_lvl
        mask_logits_lvl = mask_logits_by_level[lvl_name]
        pq_lvl, pc_lvl, ps_lvl = per_sp_predicted_slice(
            sp_mask_logits=mask_logits_lvl,
            cls_argmax=effective_argmax,
            no_object_class_id=no_object_class_id,
            q_idx_matched=q_idx, k_idx_matched=k_idx,
            primary_trackid=primary_trackid_gt,
        )
        # `pq_lvl` is -1 wherever no active query won the panoptic argmax
        # (happens with untrained / weakly-trained models, or under the
        # confidence-floor demotion). gather(0, idx) requires non-negative
        # indices, so we substitute 0 for the read and zero out the
        # result for those SPs.
        pq_safe = pq_lvl.copy()
        pq_safe[pq_safe < 0] = 0
        pq_lvl_t = torch.as_tensor(pq_safe, dtype=torch.long,
                                   device=mask_logits_lvl.device)
        pmp_lvl = torch.sigmoid(
            mask_logits_lvl.gather(0, pq_lvl_t.unsqueeze(0)).squeeze(0)
        ).cpu().numpy().astype(np.float32)
        pmp_lvl[pq_lvl < 0] = 0.0
        levels_payload.update({
            f"stage3_levels/{lvl_name}/coord": coords_cm_lvl,
            f"stage3_levels/{lvl_name}/coord_norm": coords_norm_lvl,
            f"stage3_levels/{lvl_name}/pred_query": pq_lvl,
            f"stage3_levels/{lvl_name}/pred_class": pc_lvl,
            f"stage3_levels/{lvl_name}/pred_particle_trackid": ps_lvl,
            f"stage3_levels/{lvl_name}/pred_mask_prob": pmp_lvl,
        })

    # Per-query class probs + matched_gt
    query_matched_gt = np.full(Q, -1, dtype=np.int64)
    if len(q_idx) > 0:
        query_matched_gt[q_idx] = k_idx
    gt_matched_query = np.full(K, -1, dtype=np.int64)
    if len(k_idx) > 0:
        gt_matched_query[k_idx] = q_idx
    is_active_query = (cls_argmax_np != no_object_class_id) & (cls_max_prob >= class_prob_threshold)

    out_dict = {
        # Per-SP Stage-3 fields
        "stage3/coord": coord_post,
        "stage3/coord_norm": coord_norm_post,
        "stage3/particle_class_id_gt": particle_class_id_gt,
        "stage3/source_mask": np.asarray(
            sample.get("source_mask_kept",
                       np.zeros(n_stage3, dtype=np.uint8)),
            dtype=np.uint8,
        ),
        "stage3/stage2_nu_mask_prob": np.asarray(
            sample.get("stage2_nu_mask_prob",
                       np.zeros(n_stage3, dtype=np.float32)),
            dtype=np.float32,
        ),
        "stage3/pred_query": pred_query,
        "stage3/pred_class": pred_class,
        "stage3/pred_particle_idx": pred_particle_idx,
        "stage3/pred_particle_trackid": pred_particle_tid,
        "stage3/pred_mask_prob": pred_mask_prob,

        # Per-query
        "stage3_queries/class_logits": cls_logits.detach().cpu().numpy().astype(np.float32),
        "stage3_queries/class_probs": cls_probs.cpu().numpy().astype(np.float32),
        "stage3_queries/class_argmax": cls_argmax_np,
        "stage3_queries/class_max_prob": cls_max_prob,
        "stage3_queries/is_active": is_active_query.astype(bool),
        "stage3_queries/origin_coord_norm": origin_pred_norm,
        "stage3_queries/matched_gt_idx": query_matched_gt,

        # Per-GT
        "stage3_gt/primary_trackid": primary_trackid_gt,
        "stage3_gt/class_id": class_id_gt,
        "stage3_gt/pid": pid_gt,
        "stage3_gt/ke_mev": ke_mev_gt,
        "stage3_gt/n_truth_points": n_truth_points_gt,
        "stage3_gt/origin_cm": origin_cm_gt,
        "stage3_gt/origin_coord_norm": origin_coord_norm_gt,
        "stage3_gt/matched_query": gt_matched_query,
        "stage3_gt/pair_iou": np.concatenate([
            pair_iou,
            np.full(max(K - len(pair_iou), 0), -1.0, dtype=np.float32),
        ])[:K],
        "stage3_gt/pair_cls_correct": np.concatenate([
            pair_cls_correct.astype(np.int8),
            np.full(max(K - len(pair_cls_correct), 0), -1, dtype=np.int8),
        ])[:K].astype(np.int8),
        "stage3_gt/pair_origin_l2_cm": np.concatenate([
            pair_origin_l2_cm,
            np.full(max(K - len(pair_origin_l2_cm), 0), -1.0, dtype=np.float32),
        ])[:K],

        # Identity + summary
        "stage3_meta/event_dropped": int(False),
        "stage3_meta/run": int(sample.get("run", -1)),
        "stage3_meta/subrun": int(sample.get("subrun", -1)),
        "stage3_meta/event": int(sample.get("event", -1)),
        "stage3_meta/name": str(sample.get("name", "")),
        "stage3_meta/n_stage3_sp": int(n_stage3),
        "stage3_meta/n_queries": int(Q),
        "stage3_meta/n_classes": int(C),
        "stage3_meta/n_gt_instances": int(K),
        "stage3_meta/n_matched": int(len(q_idx)),
        "stage3_meta/no_object_class_id": int(no_object_class_id),
        "stage3_meta/class_prob_threshold": float(class_prob_threshold),
        "stage3_meta/coord_scale": float(coord_scale),
        "stage3_meta/has_gt": int(has_gt),
    }
    out_dict.update(levels_payload)
    if levels_payload:
        emitted = sorted({k.split("/")[1] for k in levels_payload})
        out_dict["stage3_meta/voxel_levels"] = ",".join(emitted)
    return out_dict


# ---------------------------------------------------------------------------
# HDF5 I/O
# ---------------------------------------------------------------------------

def write_event_h5(path: str, event_data: dict) -> None:
    """Write one event's predictions to HDF5. Group structure mirrors the
    dict's key prefixes (pre/, post/, queries/, gt/, meta/, stage3/, ...).

    Written atomically (tmp + os.replace) so a killed job never leaves a
    half-written file behind — important for skip-if-exists logic in
    callers that treat the file's presence as "this event is done."
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with h5py.File(tmp, "w") as f:
        for key, val in event_data.items():
            if isinstance(val, np.ndarray):
                if val.size == 0:
                    f.create_dataset(key, data=val)
                else:
                    f.create_dataset(key, data=val, compression="gzip",
                                     compression_opts=4)
            elif isinstance(val, bool):
                f.attrs[key.replace("/", "_")] = int(val)
            elif isinstance(val, (int, float)):
                f.attrs[key.replace("/", "_")] = val
            elif isinstance(val, str):
                f.attrs[key.replace("/", "_")] = val
            else:
                pass
    os.replace(tmp, path)


def load_event_h5(path: str) -> "dict | None":
    """Load a per-event prediction HDF5 written by `write_event_h5`.

    Returns None if the file doesn't exist. Datasets are returned as
    numpy arrays keyed by their `group/.../leaf` path. Attributes are
    returned at the top level with their `group_..._leaf` flat-name.

    Single-source loader: replaces ad-hoc viz-side loaders in
    `visualize_larformer_gt.py` and the Stage-3 viz so the slicer +
    stage3 inference paths and the visualizers agree on the schema.
    """
    if not os.path.exists(path):
        return None
    out: dict = {}
    with h5py.File(path, "r") as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                out[name] = obj[()]
        f.visititems(visit)
        for k, v in f.attrs.items():
            if hasattr(v, "item"):
                out[k] = v.item()
            else:
                out[k] = v
    return out
