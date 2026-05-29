"""
Inference + per-event diagnostics dump for the LArFormer cascaded slicer.

For each input merged_h5 event, runs the trained `CascadedSlicer` in eval
mode and writes a self-contained HDF5 file with everything needed to
visualize and analyze where the model's predictions go wrong:

  - pre-filter (raw input) per-SP arrays: coord, hasmatch, slice_id, lm_score,
    pixval, ssnet_label, deghoster's P(real), kept-by-deghoster mask
  - post-filter (slicer input) per-SP arrays + the slicer's per-SP
    predicted slice assignment (argmax over matched queries)
  - per-query info: class logits, argmax class, matched GT idx
  - per-GT-instance info: which query matched, per-pair mask IoU,
    per-pair class correctness, primary_trackid (= GT slice id)
  - event identity + summary stats (tau, keep_frac, n_matched, etc.)

The per-SP predicted slice assignment uses the standard Mask2Former inference
rule: for each spacepoint, pick the query whose mask logit is highest
*among queries whose argmax class is not no_object*. That query's matched
GT instance (if any) gives the predicted slice ID. SPs whose best query is
unmatched get pred_slice_id = -1 (= "unassigned").

Usage:
    ./run_in_container.sh python tools/run_slicer_inference.py \\
        --config configs/lartpc/larformer-slicer-v1-cascaded-loradeghost.py \\
        --weights exp/larformer_slicer_v1_cascaded_loradeghost/model/model_last.pth \\
        --input-list devdata_mergedh5_pi0filter_10files.txt \\
        --output-dir exp/larformer_slicer_v1_cascaded_loradeghost/inference \\
        --max-events 10

Output file naming: `slicerpred_<input_basename_without_ext>.h5`, one per
input event. Self-contained — no need to keep the source merged_h5 around
to visualize the predictions.
"""

import argparse
import os
import sys

import h5py
import numpy as np
import torch


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)


# ---------------------------------------------------------------------------
# Per-SP predicted slice assignment
# ---------------------------------------------------------------------------

def _per_sp_predicted_slice(
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

def _per_pair_iou(
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
# Main per-event extractor
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


def process_event(
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
    n_pre = int(sample["n_spacepoints"])
    with torch.no_grad():
        out = model(batched)
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
        }

    cls_logits = ev_pred["class_logits"]              # (Q, C)
    sp_mask = ev_pred["mask_logits"]["spacepoint"]    # (Q, N_post)
    eval_loss = ev_pred.get("eval_loss")
    if eval_loss is None:
        raise RuntimeError(
            "eval_loss missing from prediction dict — was the dataset's "
            "`gt_instances_per_event` populated? The cascade's eval-mode "
            "loss + matching only fires when GT is present."
        )
    q_idx = eval_loss["q_idx"]                         # np.int64 (P,)
    k_idx = eval_loss["k_idx"]                         # np.int64 (P,)
    gt_classes = eval_loss["gt_classes"]               # (K,) long
    gt_truth_indices = eval_loss["gt_truth_indices"]   # list[K] long

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
    # Denormalize back to detector cm by inverting the dataset's affine,
    # which we recover from (coord, coord_norm) on the pre-filter sample.
    pre_coord = sample["coord"].astype(np.float32)
    pre_coord_norm = sample["coord_norm"].astype(np.float32)
    # Use median to avoid degenerate single-axis events
    if pre_coord.shape[0] > 1:
        scale = (pre_coord.max(0) - pre_coord.min(0)) / np.maximum(
            pre_coord_norm.max(0) - pre_coord_norm.min(0), 1e-9)
        center = pre_coord.mean(0) - pre_coord_norm.mean(0) * scale
        coord_post = (coord_norm_post * scale + center).astype(np.float32)
    else:
        coord_post = coord_norm_post  # fallback

    # Per-GT-instance fields. Use the POST-filter gt_instances exposed by
    # CascadedSlicer (`filtered_gt_instances_per_event`) so the indexing
    # matches `K = gt_classes.shape[0]` from the slicer's eval_loss.
    # (The pre-filter `sample["gt_instances"]` can be longer — empty
    # instances are pruned by filter_batch_by_keep_mask.)
    filtered_gt_per_event = out.get("filtered_gt_instances_per_event", None)
    filtered_gt = (filtered_gt_per_event[0]
                   if filtered_gt_per_event and len(filtered_gt_per_event) > 0
                   else sample["gt_instances"])
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
    pair_iou = _per_pair_iou(sp_mask, gt_truth_indices, q_idx, k_idx)
    pair_gt_class = gt_classes[torch.as_tensor(k_idx, device=gt_classes.device)] \
        if len(k_idx) > 0 else gt_classes.new_zeros(0)
    pair_pred_class = cls_argmax[torch.as_tensor(q_idx, device=cls_argmax.device)] \
        if len(q_idx) > 0 else cls_argmax.new_zeros(0)
    pair_cls_correct = (pair_pred_class == pair_gt_class).cpu().numpy().astype(bool) \
        if len(q_idx) > 0 else np.zeros(0, dtype=bool)

    # Per-GT MATCHED QUERY MASK (above-threshold), independent of panoptic
    # argmax. This is the "raw" pred-mask view that pair_iou computes
    # against — saving it lets the visualizer show Q's full claim and
    # makes the over-claim pathology directly visible (Q has high prob on
    # SPs but other queries win them in argmax, so the panoptic view
    # hides them while pair_iou penalizes Q for them).
    pred_mask_bool_per_gt = np.zeros((K, n_post), dtype=bool)
    pred_n_pts_per_gt = np.zeros(K, dtype=np.int64) - 1   # -1 = unmatched
    for p in range(len(q_idx)):
        q_p = int(q_idx[p]); k_p = int(k_idx[p])
        m = (sp_mask[q_p] > 0).cpu().numpy()
        pred_mask_bool_per_gt[k_p] = m
        pred_n_pts_per_gt[k_p] = int(m.sum())

    # Query → matched-GT lookup
    query_matched_gt = np.full(Q, -1, dtype=np.int64)
    if len(q_idx) > 0:
        query_matched_gt[q_idx] = k_idx
    # GT → matched-query lookup
    gt_matched_query = np.full(K, -1, dtype=np.int64)
    if len(k_idx) > 0:
        gt_matched_query[k_idx] = q_idx

    # Per-SP predicted slice assignment (Mask2Former inference rule on
    # active queries; see _per_sp_predicted_slice).
    pred_query, pred_class, pred_slice_id = _per_sp_predicted_slice(
        sp_mask_logits=sp_mask,
        cls_argmax=cls_argmax,
        no_object_class_id=no_object_class_id,
        q_idx_matched=q_idx, k_idx_matched=k_idx,
        primary_trackid=primary_trackid_gt,
    )

    # Per-SP sigmoid mask confidence of the assigned query — exactly the
    # value PointRend importance sampling thresholds against. Lets the
    # visualizer's hover text show, for any merged SP, what mask prob the
    # model assigned (sigm≈1 ⇒ a confident-FP that hard-neg mining should
    # pick up; sigm≈0.5 ⇒ a halo point already targeted by halo sampling).
    pred_q_t = torch.as_tensor(pred_query, dtype=torch.long, device=sp_mask.device)
    if n_post > 0:
        pred_mask_prob = torch.sigmoid(
            sp_mask.gather(0, pred_q_t.unsqueeze(0)).squeeze(0)
        ).cpu().numpy().astype(np.float32)
    else:
        pred_mask_prob = np.zeros(0, dtype=np.float32)

    # Compute post-filter GT slice id NOW (the per-voxel block below needs
    # it for plurality voting). Same derivation as the pred_correct block
    # further down — derive from the pre-filter slice_id by applying the
    # same deghoster keep mask the cascade used.
    pre_slice_id_gt = sample.get("slice_id", -np.ones(n_pre, dtype=np.int64)).astype(np.int64)
    if p_real_pre is not None:
        keep_mask_pre = p_real_pre > tau
    else:
        keep_mask_pre = np.ones(n_pre, dtype=bool)
    post_slice_id_gt = pre_slice_id_gt[keep_mask_pre]
    if post_slice_id_gt.shape[0] != n_post:
        post_slice_id_gt = np.resize(post_slice_id_gt, n_post)

    # Per-voxel-level predicted slice assignment — same rule as spacepoint,
    # applied to each non-spacepoint level the decoder produced. Lets the
    # visualizer compare merger behavior across scales (the same mask_embed
    # is dot-producted against each level's tokens + pos_emb(coords), so a
    # merge that appears at the spacepoint level should be diagnosable at
    # the voxel levels too — see docs/LArFormer.md §15).
    levels_payload: dict = {}
    levels_dict = ev_pred.get("levels", {}) or {}
    mask_logits_by_level = ev_pred.get("mask_logits", {}) or {}
    for lvl_name, lvl in levels_dict.items():
        if lvl_name == "spacepoint":
            # Already covered by post/* — don't duplicate (~100K × 12 bytes/event).
            continue
        if lvl_name not in mask_logits_by_level:
            continue
        coords_norm_lvl = lvl["coords"].detach().cpu().numpy().astype(np.float32)
        if coords_norm_lvl.shape[0] == 0:
            continue
        coords_cm_lvl = (coords_norm_lvl * scale + center).astype(np.float32) \
            if pre_coord.shape[0] > 1 else coords_norm_lvl
        mask_logits_lvl = mask_logits_by_level[lvl_name]      # (Q, M_lvl)
        pq_lvl, pc_lvl, ps_lvl = _per_sp_predicted_slice(
            sp_mask_logits=mask_logits_lvl,
            cls_argmax=cls_argmax,
            no_object_class_id=no_object_class_id,
            q_idx_matched=q_idx, k_idx_matched=k_idx,
            primary_trackid=primary_trackid_gt,
        )
        # Per-voxel sigmoid mask confidence of the assigned query — same
        # semantics as post/pred_mask_prob at the SP level.
        pq_lvl_t = torch.as_tensor(pq_lvl, dtype=torch.long,
                                   device=mask_logits_lvl.device)
        pmp_lvl = torch.sigmoid(
            mask_logits_lvl.gather(0, pq_lvl_t.unsqueeze(0)).squeeze(0)
        ).cpu().numpy().astype(np.float32)
        # Per-voxel GT slice id, via the per-SP→level mapping. Take the
        # plurality vote so the level is colorable against the GT panel.
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
            # Plurality vote (np.unique returns sorted, so deterministic ties)
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

    # Per-SP "is the assigned slice correct?" (pre/post slice_id_gt
    # derivation was hoisted above the per-voxel block; reuse it here).
    pred_correct = (pred_slice_id == post_slice_id_gt).astype(bool)

    # Argmax-restricted IoU per matched pair. Uses the panoptic-argmax
    # view (the visualizer's "predicted slice id" mode) — only the SPs
    # the matched query Q WINS in argmax-among-active count as Q's pred
    # set. Compared with pair_iou (which uses Q's full above-threshold
    # mask), the GAP `pair_iou - argmax_iou` measures how much Q is
    # "over-claiming" SPs that other queries actually win.
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
        "pre/coord": pre_coord,                                          # (n_pre, 3)
        "pre/coord_norm": pre_coord_norm,                                # (n_pre, 3)
        "pre/hasmatch": sample.get("hasmatch", np.zeros(n_pre, dtype=np.int64)).astype(np.int64),
        "pre/slice_id_gt": pre_slice_id_gt,                              # (n_pre,)
        "pre/lm_score": sample.get("lm_score", np.zeros(n_pre, dtype=np.float32)).astype(np.float32),
        "pre/pixval": sample.get("feat", np.zeros((n_pre, 6), dtype=np.float32))[:, 3:6].astype(np.float32),
        "pre/ssnet_label": sample.get("ssnet_label", -np.ones(n_pre, dtype=np.int64)).astype(np.int64),
        "pre/p_real": p_real_pre if p_real_pre is not None else np.zeros(n_pre, dtype=np.float32),
        "pre/keep": keep_mask_pre.astype(bool),

        # Post-filter: slicer's input
        "post/coord": coord_post,                                        # (n_post, 3)
        "post/coord_norm": coord_norm_post,                              # (n_post, 3)
        "post/slice_id_gt": post_slice_id_gt.astype(np.int64),           # (n_post,)
        "post/pred_query": pred_query,                                    # (n_post,)
        "post/pred_class": pred_class,                                    # (n_post,)
        "post/pred_slice_id": pred_slice_id,                              # (n_post,)
        "post/pred_mask_prob": pred_mask_prob,                            # (n_post,)
        "post/pred_correct": pred_correct,                                # (n_post,)

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
        ])[:K],                                                            # (K,) -1 if unmatched
        # `gt/pred_n_pts[k]` = number of SPs where the matched query's
        # mask sigm > 0.5 (the "raw" pred mask count used by pair_iou).
        # -1 if unmatched. Compare with `gt/n_truth_points` to spot
        # over-claim: pred_n_pts >> n_truth_points → Q is claiming many
        # more SPs than the GT slice has.
        "gt/pred_n_pts": pred_n_pts_per_gt,                                # (K,) -1 if unmatched
        # Per-SP boolean: True iff this SP is in the matched query's
        # above-threshold mask (pair_iou's view), regardless of panoptic
        # argmax. Used by visualizer's "raw matched-Q mask" color mode.
        "gt/pred_mask_bool": pred_mask_bool_per_gt,                        # (K, n_post)
        # Argmax-restricted IoU (visualizer's panoptic view of IoU).
        # `pair_iou - argmax_iou` measures over-claim magnitude.
        "gt/argmax_iou": np.concatenate([
            argmax_iou_per_pair,
            np.full(max(K - len(argmax_iou_per_pair), 0), -1.0, dtype=np.float32),
        ])[:K],
        "gt/pair_cls_correct": np.concatenate([
            pair_cls_correct.astype(np.int8),
            np.full(max(K - len(pair_cls_correct), 0), -1, dtype=np.int8),
        ])[:K].astype(np.int8),                                            # (K,) -1 if unmatched

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
    }
    out_dict.update(levels_payload)
    # Record which non-spacepoint levels we emitted so the visualizer can
    # offer them as toggle options without having to walk the full HDF5 tree.
    if levels_payload:
        emitted = sorted({k.split("/")[1] for k in levels_payload})
        out_dict["meta/voxel_levels"] = ",".join(emitted)
    return out_dict


# ---------------------------------------------------------------------------
# HDF5 writer
# ---------------------------------------------------------------------------

def write_event_h5(path: str, event_data: dict) -> None:
    """Write one event's predictions to HDF5. Group structure mirrors the
    dict's key prefixes (pre/, post/, queries/, gt/, meta/)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with h5py.File(path, "w") as f:
        for key, val in event_data.items():
            if isinstance(val, np.ndarray):
                f.create_dataset(key, data=val, compression="gzip",
                                 compression_opts=4)
            elif isinstance(val, bool):
                # h5py rejects naked Python bools as attrs/datasets oddly;
                # store as int8 attribute
                f.attrs[key.replace("/", "_")] = int(val)
            elif isinstance(val, (int, float)):
                f.attrs[key.replace("/", "_")] = val
            elif isinstance(val, str):
                f.attrs[key.replace("/", "_")] = val
            else:
                # Skip unhandled types (or could log a warning)
                pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("--config", required=True,
                    help="Path to the cascaded slicer config")
    ap.add_argument("--weights", required=True,
                    help="Path to model_*.pth checkpoint")
    ap.add_argument("--input-list", required=True,
                    help="Text file: one merged_h5 path per line")
    ap.add_argument("--output-dir", required=True,
                    help="Directory to write per-event prediction HDF5 files")
    ap.add_argument("--max-events", type=int, default=None,
                    help="If set, stop after this many events")
    ap.add_argument("--split", default="val",
                    choices=("train", "val", "test"),
                    help="Which dataset split's kwargs to copy from the "
                         "config (only the data_list_file is overridden)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    args = ap.parse_args()

    from pointcept.utils.config import Config
    from pointcept.datasets import build_dataset, larformer_collate
    from pointcept.models.builder import build_model
    import pointcept.models  # noqa: F401 — registers MODELS

    cfg = Config.fromfile(args.config)

    # Dataset: copy the split's kwargs, override data_list_file. The
    # dataset joins relative paths against `data_root`, so absolutize
    # the input-list path here (otherwise a CLI-given relative path
    # resolves to e.g. `/devdata...` and the dataset finds 0 events).
    ds_cfg = dict(cfg.data[args.split])
    ds_cfg["data_list_file"] = os.path.abspath(args.input_list)
    ds_cfg["max_spacepoints"] = None      # no cap during inference — we want
                                          # the full event so the predictions
                                          # are useful for downstream analysis
    dataset = build_dataset(ds_cfg)
    n_events = len(dataset)
    if args.max_events is not None:
        n_events = min(n_events, args.max_events)
    print(f"[infer] Dataset: {len(dataset)} events; will process {n_events}.")

    # Model
    print(f"[infer] Building model from {args.config}")
    model = build_model(cfg.model).to(args.device).eval()

    # Load weights (full checkpoint trained by LArFormerTrainer — the
    # state_dict keys are top-level model keys, no SonataCheckpointLoader-
    # style prefix munging needed). DDP `module.` prefix is stripped.
    print(f"[infer] Loading weights from {args.weights}")
    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  loaded; missing={len(missing)}  unexpected={len(unexpected)}")
    if missing[:5]:
        print(f"  first missing: {missing[:5]}")
    if unexpected[:5]:
        print(f"  first unexpected: {unexpected[:5]}")

    # The cascade's slicer.loss_fn has the no_object_class_id we need for
    # the per-SP slice-assignment rule. Reach through the wrapper.
    inner_model = getattr(model, "module", model)
    if hasattr(inner_model, "slicer"):
        slicer = inner_model.slicer
    else:
        slicer = inner_model     # plain LArFormer (no cascade)
    no_object_class_id = int(slicer.loss_fn.no_object_class_id)
    print(f"[infer] no_object_class_id = {no_object_class_id}")

    # Iterate events
    os.makedirs(args.output_dir, exist_ok=True)
    n_dropped = 0
    for ev_idx in range(n_events):
        sample = dataset[ev_idx]
        batched = larformer_collate([sample])
        for k, v in batched.items():
            if isinstance(v, torch.Tensor):
                batched[k] = v.to(args.device, non_blocking=True)

        event_data = process_event(
            model, sample, batched, no_object_class_id,
        )
        if event_data.get("event_dropped"):
            n_dropped += 1

        # Output file: slicerpred_<input_basename_without_ext>.h5
        in_name = sample.get("name", f"event{ev_idx:06d}.h5")
        stem = os.path.splitext(in_name)[0]
        out_path = os.path.join(args.output_dir, f"slicerpred_{stem}.h5")
        write_event_h5(out_path, event_data)

        keep_frac = event_data.get("meta/keep_frac", float("nan"))
        n_pre = event_data.get("meta/n_sp_pre", 0)
        n_post = event_data.get("meta/n_sp_post", 0)
        n_gt = event_data.get("meta/n_gt_instances", 0)
        n_matched = event_data.get("meta/n_matched", 0)
        pair_iou = event_data.get("gt/pair_iou", np.zeros(0))
        valid_iou = pair_iou[pair_iou >= 0] if pair_iou.size else pair_iou
        mean_iou = float(np.mean(valid_iou)) if valid_iou.size > 0 else float("nan")
        print(
            f"[{ev_idx+1:4d}/{n_events}] {in_name:50s}  "
            f"n_sp {n_pre} → {n_post}  keep_frac={keep_frac:.3f}  "
            f"matched={n_matched}/{n_gt}  mean_pair_IoU={mean_iou:.3f}"
        )

    print(f"\n[infer] Done. Wrote {n_events} files to {args.output_dir} "
          f"({n_dropped} events were dropped by the cascade's empty-event guard).")


if __name__ == "__main__":
    main()
