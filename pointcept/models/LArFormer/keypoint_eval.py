"""Phase 4: decode discrete keypoints from the head outputs + PPN-style metrics.

Two decoders + a matcher, all pure numpy (no torch dependency at call time so
they're trivially unit-testable):

  decode_dense_votes  — Phase-1 dense head: per keypoint type, take the SPs
      whose score exceeds a threshold, move each to its voted location
      (coord + offset), and greedily cluster the votes (weighted by score).
      Yields discrete keypoints that can sit BETWEEN measured points — this is
      where the nu vertex comes from (type 0).

  decode_query_points — Phase-2 query head: per particle query, emit its
      start (and end, gated by the end-existence logit) when the query is a
      real particle (not no_object) above a class-prob floor.

  match_points / per_type_metrics — greedy nearest matching of predicted to GT
      keypoints within distance thresholds → per-type precision/recall (PPN
      "fraction recovered within N cm") + vertex resolution.

All positions are detector cm. Helpers to denormalize from the model's
normalized frame are included.
"""

import numpy as np

from lartpc.data_prep.labels.keypoint_labels import (
    KEYPOINT_TYPE_NAMES, N_KEYPOINT_TYPES, KPTYPE_NU_VERTEX,
    KPTYPE_TRACK_START, KPTYPE_TRACK_END, KPTYPE_SHOWER,
)

# Stage-3 particle taxonomy → which classes are shower-like vs track-like.
# (0=e, 1=γ, 2=μ, 3=π, 4=p, 5=other_track, 6=unused, 7=no_object.)
DEFAULT_SHOWER_CLASS_IDS = (0, 1)
DEFAULT_TRACK_CLASS_IDS = (2, 3, 4, 5)
# Keypoint types best served by the query head (particle-associated) vs the
# dense head (slice-level vertex + michel/delta the query head doesn't emit).
DEFAULT_QUERY_TYPES = (KPTYPE_TRACK_START, KPTYPE_TRACK_END, KPTYPE_SHOWER)


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------

def denorm(x_norm, coord_center, coord_scale):
    """normalized → detector cm. x_norm (..., 3)."""
    return np.asarray(x_norm, dtype=np.float32) * coord_scale + np.asarray(
        coord_center, dtype=np.float32)


# ---------------------------------------------------------------------------
# Dense decode (Phase-1 score + offset/vote head)
# ---------------------------------------------------------------------------

def cluster_votes(votes_cm, weights, radius_cm):
    """Greedy score-weighted clustering. Returns list of dicts:
    {pos_cm (3,), score (max weight in cluster), n_votes}. Highest-weight
    vote seeds each cluster and absorbs all unclaimed votes within radius_cm;
    the cluster position is the weight-weighted mean."""
    from scipy.spatial import cKDTree

    votes_cm = np.ascontiguousarray(votes_cm, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    n = votes_cm.shape[0]
    if n == 0:
        return []
    tree = cKDTree(votes_cm)
    order = np.argsort(-weights)
    claimed = np.zeros(n, dtype=bool)
    clusters = []
    for i in order:
        if claimed[i]:
            continue
        nb = [j for j in tree.query_ball_point(votes_cm[i], radius_cm)
              if not claimed[j]]
        if not nb:
            continue
        claimed[nb] = True
        w = weights[nb]
        pos = (votes_cm[nb] * w[:, None]).sum(0) / max(w.sum(), 1e-9)
        clusters.append({"pos_cm": pos.astype(np.float32),
                         "score": float(w.max()), "n_votes": int(len(nb))})
    return clusters


def decode_dense_votes(coord_norm, scores, offsets_norm, coord_center,
                       coord_scale, score_thresh=0.3, cluster_radius_cm=3.0,
                       n_types=N_KEYPOINT_TYPES):
    """Decode the dense head into discrete keypoints per type.

    Args:
        coord_norm:    (N, 3) normalized SP positions.
        scores:        (N, n_types) predicted kpscores in [0, 1].
        offsets_norm:  (N, n_types, 3) predicted offsets (normalized), or None
                        (then each SP votes for its own location).
        score_thresh:  per-SP score floor to vote.
        cluster_radius_cm: vote-clustering radius.

    Returns: dict {type_idx: [cluster dicts sorted by score desc]}.
    """
    coord_norm = np.asarray(coord_norm, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    out = {}
    for t in range(n_types):
        col = scores[:, t]
        m = col > score_thresh
        if not np.any(m):
            out[t] = []
            continue
        if offsets_norm is not None:
            voted_norm = coord_norm[m] + np.asarray(
                offsets_norm, dtype=np.float32)[m, t, :]
        else:
            voted_norm = coord_norm[m]
        votes_cm = denorm(voted_norm, coord_center, coord_scale)
        clusters = cluster_votes(votes_cm, col[m], cluster_radius_cm)
        clusters.sort(key=lambda c: -c["score"])
        out[t] = clusters
    return out


def decode_nu_vertex(dense_decoded):
    """Best nu-vertex (type 0) cluster, or None. Detector cm."""
    cl = dense_decoded.get(KPTYPE_NU_VERTEX, [])
    return cl[0] if cl else None


# ---------------------------------------------------------------------------
# Query decode (Phase-2 start/end head)
# ---------------------------------------------------------------------------

def _softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def decode_query_points(class_logits, origin_norm, coord_center, coord_scale,
                        end_norm=None, end_logit=None, no_object_class_id=None,
                        class_prob_floor=0.3, end_prob_floor=0.5,
                        track_class_ids=None, effective_argmax=None):
    """Decode per-query keypoints from the Phase-2 head.

    Returns list of dicts: {pos_cm, kind ("start"/"end"), query, class_id,
    class_prob}. A query contributes a start iff its (effective) class is a real
    particle (not no_object) with prob >= class_prob_floor; it contributes an
    end iff additionally sigmoid(end_logit) >= end_prob_floor (and, if
    track_class_ids is given, its class is track-like).

    `effective_argmax` (Q,) optionally overrides the per-query class — pass the
    output of `dedup_query_effective_argmax` (mask-IoU NMS via the SAME
    `inference.dedup_queries` used for the particle masks) so duplicate queries
    demoted to no_object are skipped → no duplicate keypoints. When None, the
    class is `argmax(class_logits)`.
    """
    class_logits = np.asarray(class_logits, dtype=np.float32)
    Q, C = class_logits.shape
    if no_object_class_id is None:
        no_object_class_id = C - 1
    probs = _softmax(class_logits, axis=-1)
    argmax = (np.asarray(effective_argmax).reshape(-1)
              if effective_argmax is not None else probs.argmax(axis=-1))
    maxp = probs.max(axis=-1)
    starts_cm = denorm(origin_norm, coord_center, coord_scale)
    ends_cm = (denorm(end_norm, coord_center, coord_scale)
               if end_norm is not None else None)
    end_prob = (1.0 / (1.0 + np.exp(-np.asarray(end_logit, dtype=np.float32)))
                if end_logit is not None else None)
    out = []
    for q in range(Q):
        cid = int(argmax[q])
        if cid == int(no_object_class_id) or maxp[q] < class_prob_floor:
            continue
        out.append({"pos_cm": starts_cm[q].astype(np.float32), "kind": "start",
                    "query": q, "class_id": cid, "class_prob": float(maxp[q])})
        if ends_cm is not None and end_prob is not None:
            is_track = (track_class_ids is None or cid in track_class_ids)
            if is_track and end_prob[q] >= end_prob_floor:
                out.append({"pos_cm": ends_cm[q].astype(np.float32),
                            "kind": "end", "query": q, "class_id": cid,
                            "class_prob": float(maxp[q])})
    return out


def dedup_query_effective_argmax(class_logits, sp_mask_logits,
                                 no_object_class_id=None, class_prob_floor=0.3,
                                 iou_threshold=0.6):
    """Per-query class after confidence-floor demotion + mask-IoU NMS dedup.

    Uses the SAME `inference.dedup_queries` as the particle-mask decode (single
    source of truth — no drift): co-extensive duplicate queries (mask
    IoU >= iou_threshold) are suppressed (demoted to no_object), so feeding the
    result to `decode_query_points(effective_argmax=...)` drops their duplicate
    keypoints. Inputs are torch tensors (class_logits (Q,C), sp_mask_logits
    (Q,N)); returns a (Q,) int64 numpy array. iou_threshold<=0 = no dedup.
    """
    import torch
    from .inference import dedup_queries

    C = int(class_logits.shape[-1])
    if no_object_class_id is None:
        no_object_class_id = C - 1
    probs = class_logits.softmax(dim=-1)
    argmax = probs.argmax(dim=-1)
    maxp = probs.max(dim=-1).values.detach().cpu().numpy().astype(np.float32)
    eff = argmax.clone()
    eff[torch.as_tensor(maxp < class_prob_floor, device=eff.device)] = \
        int(no_object_class_id)
    _records, eff_dedup = dedup_queries(
        sp_mask_logits=sp_mask_logits, effective_argmax=eff,
        cls_max_prob=maxp, no_object_class_id=int(no_object_class_id),
        iou_threshold=float(iou_threshold))
    return eff_dedup.detach().cpu().numpy().astype(np.int64)


def query_points_to_typed(qpts, shower_class_ids=DEFAULT_SHOWER_CLASS_IDS,
                          track_class_ids=DEFAULT_TRACK_CLASS_IDS,
                          n_types=N_KEYPOINT_TYPES):
    """Map `decode_query_points` output onto keypoint types so it can be scored
    against `mckeypoints` with the same metric as the dense head.

      start of a shower class (e/γ) → shower (3)
      start of a track  class       → track_start (1)
      end   of a track  class       → track_end (2)

    Returns {type_idx: [{pos_cm, score}]} (score = class_prob, used for greedy
    matching order).
    """
    shower = set(int(c) for c in shower_class_ids)
    out = {t: [] for t in range(n_types)}
    for p in qpts:
        if p["kind"] == "start":
            t = (KPTYPE_SHOWER if int(p["class_id"]) in shower
                 else KPTYPE_TRACK_START)
        else:  # "end"
            t = KPTYPE_TRACK_END
        out[t].append({"pos_cm": p["pos_cm"], "score": p["class_prob"],
                       "class_id": int(p["class_id"]), "query": int(p["query"]),
                       "kind": p["kind"]})
    return out


def reconcile_keypoints(dense_decoded, query_typed,
                        query_types=DEFAULT_QUERY_TYPES,
                        n_types=N_KEYPOINT_TYPES):
    """Combine the two decoders into one typed keypoint set.

    For `query_types` (particle-associated start/end/shower) use the query
    head's points, falling back to the dense head when the query head produced
    none. All other types (nu_vertex, michel, delta) come from the dense head.
    Returns {type_idx: [cluster/point dicts]}.
    """
    final = {}
    for t in range(n_types):
        if t in query_types:
            q = query_typed.get(t, [])
            final[t] = q if q else dense_decoded.get(t, [])
        else:
            final[t] = dense_decoded.get(t, [])
    return final


def gt_keypoints_by_type(mckp_pos_norm, mckp_type, coord_center, coord_scale,
                         coord_cm=None, acceptance_cm=None,
                         n_types=N_KEYPOINT_TYPES):
    """GT keypoints per type in detector cm (shared by the dense + query evals).

    If `acceptance_cm` is set, keep only GT keypoints within that distance of
    any SP in `coord_cm` — the "reconstructable" set this slice could vote for.
    """
    pos = denorm(np.asarray(mckp_pos_norm, dtype=np.float32),
                 coord_center, coord_scale)
    typ = np.asarray(mckp_type, dtype=np.int64)
    if acceptance_cm is not None and coord_cm is not None and pos.shape[0] > 0:
        from scipy.spatial import cKDTree
        d, _ = cKDTree(np.asarray(coord_cm, dtype=np.float32)).query(pos, k=1)
        keep = d <= acceptance_cm
        pos, typ = pos[keep], typ[keep]
    return {t: (pos[typ == t] if np.any(typ == t)
                else np.zeros((0, 3), np.float32)) for t in range(n_types)}


def _recover_affine(coord_cm, coord_norm):
    """(scale, center) s.t. coord_cm ≈ coord_norm*scale + center, per-axis,
    from a per-event (coord_cm, coord_norm) pair. Handles the cascade's
    slice-recentering: with recentered coord_norm + true-cm coord, this maps
    recentered_norm → true detector cm. Returns (None, None) if <2 points."""
    coord_cm = np.asarray(coord_cm, dtype=np.float32)
    coord_norm = np.asarray(coord_norm, dtype=np.float32)
    if coord_cm.shape[0] <= 1:
        return None, None
    scale = (coord_cm.max(0) - coord_cm.min(0)) / np.maximum(
        coord_norm.max(0) - coord_norm.min(0), 1e-9)
    center = coord_cm.mean(0) - coord_norm.mean(0) * scale
    return scale.astype(np.float32), center.astype(np.float32)


def decode_event_keypoints(
    class_logits, origin_norm, coord_norm, coord_cm,
    end_norm=None, end_logit=None,
    dense_coord_norm=None, dense_scores=None, dense_offsets=None,
    n_classes=8, coord_center=(125.0, 0.0, 518.0), coord_scale=179.55,
    kp_class_prob_floor=0.3, kp_end_prob_floor=0.5,
    dense_score_thresh=0.3, dense_cluster_radius_cm=3.0,
    dense_in_query_frame=False, effective_argmax=None,
):
    """Full per-event keypoint decode for the cascade: query head (start/end,
    particle-associated) + optional dense head (nu vertex + michel/delta),
    reconciled into one typed set. All outputs in DETECTOR CM.

    Query positions are denormalized via the per-event affine recovered from
    (coord_cm, coord_norm) — correct even when the cascade recentered
    coord_norm to the slice centroid.

    Dense frame: if `dense_in_query_frame` (the INTEGRATED dense head shares the
    cascade's recentered coord_norm), denormalize the dense votes with the SAME
    recovered affine as the queries. Otherwise (a SEPARATE dense model run on a
    non-recentered batch) use the fixed (coord_center, coord_scale).

    Returns (final_by_type, nu_vertex_or_None).
    """
    scale, center = _recover_affine(coord_cm, coord_norm)
    if scale is None:
        scale = np.asarray([coord_scale] * 3, dtype=np.float32)
        center = np.asarray(coord_center, dtype=np.float32)

    qpts = decode_query_points(
        class_logits, origin_norm, center, scale,
        end_norm=end_norm, end_logit=end_logit,
        no_object_class_id=n_classes - 1,
        class_prob_floor=kp_class_prob_floor,
        end_prob_floor=kp_end_prob_floor,
        track_class_ids=set(DEFAULT_TRACK_CLASS_IDS),
        effective_argmax=effective_argmax)
    typed = query_points_to_typed(qpts)

    dense_decoded = {}
    if dense_scores is not None and dense_coord_norm is not None:
        dn_center, dn_scale = ((center, scale) if dense_in_query_frame
                               else (coord_center, coord_scale))
        dense_decoded = decode_dense_votes(
            dense_coord_norm, dense_scores, dense_offsets,
            dn_center, dn_scale,
            score_thresh=dense_score_thresh,
            cluster_radius_cm=dense_cluster_radius_cm)

    final = reconcile_keypoints(dense_decoded, typed)
    nu = decode_nu_vertex(dense_decoded)
    return final, nu


def keypoint_arrays_for_h5(final_by_type, nu_vertex=None, prefix="keypoints"):
    """Flatten the reconciled keypoint set into a write_event_h5-style dict.

    `keypoints/pos_cm` (M,3), `/type` (M,), `/score` (M,), `/source` (M,)
    (0=query, 1=dense), `/class_id` (M,) (query particle class; -1 dense),
    `/query_id` (M,), `/kind` (M,) (0=start, 1=end, -1=n/a), `/n` (attr),
    and `/nu_vertex_cm` (3,) + `/nu_vertex_score` (attr) when available.
    """
    pos, typ, score, source, cls, qid, kind = [], [], [], [], [], [], []
    for t, items in final_by_type.items():
        for it in items:
            pos.append(np.asarray(it["pos_cm"], dtype=np.float32))
            typ.append(int(t))
            score.append(float(it.get("score", 1.0)))
            source.append(1 if "n_votes" in it else 0)   # dense clusters have n_votes
            cls.append(int(it.get("class_id", -1)))
            qid.append(int(it.get("query", -1)))
            k = it.get("kind")
            kind.append(0 if k == "start" else 1 if k == "end" else -1)
    out = {
        f"{prefix}/pos_cm": (np.stack(pos).astype(np.float32) if pos
                             else np.zeros((0, 3), np.float32)),
        f"{prefix}/type": np.asarray(typ, np.int32),
        f"{prefix}/score": np.asarray(score, np.float32),
        f"{prefix}/source": np.asarray(source, np.int32),
        f"{prefix}/class_id": np.asarray(cls, np.int32),
        f"{prefix}/query_id": np.asarray(qid, np.int32),
        f"{prefix}/kind": np.asarray(kind, np.int32),
        f"{prefix}/n": int(len(pos)),
    }
    if nu_vertex is not None:
        out[f"{prefix}/nu_vertex_cm"] = np.asarray(
            nu_vertex["pos_cm"], dtype=np.float32)
        out[f"{prefix}/nu_vertex_score"] = float(nu_vertex["score"])
    return out


# ---------------------------------------------------------------------------
# Matching + metrics (PPN-style)
# ---------------------------------------------------------------------------

def match_points(pred_cm, pred_score, gt_cm, thresholds_cm):
    """Greedy nearest matching (highest-score pred first) of predicted points
    to GT points, per distance threshold.

    Args:
        pred_cm:      (P, 3) predicted positions (cm).
        pred_score:   (P,) score for greedy ordering (high first).
        gt_cm:        (G, 3) GT positions (cm).
        thresholds_cm: iterable of distance thresholds.

    Returns: dict threshold → {tp, fp, fn, dists (list of matched distances)}.
    """
    pred_cm = np.asarray(pred_cm, dtype=np.float32).reshape(-1, 3)
    gt_cm = np.asarray(gt_cm, dtype=np.float32).reshape(-1, 3)
    P, G = pred_cm.shape[0], gt_cm.shape[0]
    order = (np.argsort(-np.asarray(pred_score, dtype=np.float32))
             if P > 0 else np.zeros(0, dtype=int))
    out = {}
    for thr in thresholds_cm:
        claimed = np.zeros(G, dtype=bool)
        tp, dists = 0, []
        for p in order:
            if G == 0:
                break
            d = np.linalg.norm(gt_cm - pred_cm[p], axis=1)
            d[claimed] = np.inf
            j = int(d.argmin())
            if d[j] <= thr:
                claimed[j] = True
                tp += 1
                dists.append(float(d[j]))
        out[thr] = {"tp": tp, "fp": P - tp, "fn": G - tp, "dists": dists}
    return out


def accumulate_metrics(events_pred_by_type, events_gt_by_type, thresholds_cm,
                       n_types=N_KEYPOINT_TYPES):
    """Aggregate match counts over events into per-type precision/recall.

    events_pred_by_type / events_gt_by_type: lists (one per event) of
    {type_idx: list-of-cluster-dicts} (pred) and {type_idx: (G,3) array} (gt).

    Returns: dict type_idx → {threshold → {precision, recall, tp, fp, fn,
    median_dist}}, plus a "vertex_res_cm" list of per-event nu-vertex errors.
    """
    agg = {t: {thr: {"tp": 0, "fp": 0, "fn": 0, "dists": []}
               for thr in thresholds_cm} for t in range(n_types)}
    for pred, gt in zip(events_pred_by_type, events_gt_by_type):
        for t in range(n_types):
            clusters = pred.get(t, [])
            pc = np.array([c["pos_cm"] for c in clusters], dtype=np.float32) \
                if clusters else np.zeros((0, 3), np.float32)
            ps = np.array([c["score"] for c in clusters], dtype=np.float32) \
                if clusters else np.zeros(0, np.float32)
            gc = gt.get(t, np.zeros((0, 3), np.float32))
            m = match_points(pc, ps, gc, thresholds_cm)
            for thr in thresholds_cm:
                for k in ("tp", "fp", "fn"):
                    agg[t][thr][k] += m[thr][k]
                agg[t][thr]["dists"] += m[thr]["dists"]
    result = {}
    for t in range(n_types):
        result[t] = {}
        for thr in thresholds_cm:
            a = agg[t][thr]
            tp, fp, fn = a["tp"], a["fp"], a["fn"]
            result[t][thr] = {
                "precision": tp / max(tp + fp, 1),
                "recall": tp / max(tp + fn, 1),
                "tp": tp, "fp": fp, "fn": fn,
                "median_dist": (float(np.median(a["dists"]))
                                if a["dists"] else float("nan")),
            }
    return result


def format_metrics_table(metrics, thresholds_cm, n_types=N_KEYPOINT_TYPES):
    """Pretty per-type precision/recall table string."""
    lines = []
    hdr = "type          " + "".join(
        f"  P@{t:g}cm  R@{t:g}cm" for t in thresholds_cm)
    lines.append(hdr)
    for t in range(n_types):
        row = f"{KEYPOINT_TYPE_NAMES[t]:13s}"
        for thr in thresholds_cm:
            m = metrics[t][thr]
            row += f"  {m['precision']:6.3f}  {m['recall']:6.3f}"
        lines.append(row)
    return "\n".join(lines)
