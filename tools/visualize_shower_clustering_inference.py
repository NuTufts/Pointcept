"""
Visualize per-event inference outputs from
tools/run_shower_clustering_inference.py.

Inputs are the H5 files produced by the inference script — one per event.
Each event has GT instance assignments, predicted instance assignments,
and the Hungarian-matched pairs precomputed; the visualizer just plots.

Four 3D panels (MicroBooNE detector outline, axes in cm, z-aspect=4 to
match the larmatch visualizer style):

    1. GT instances           — color by trunk_trackid (per-instance)
    2. Predicted instances    — color by predictions/spacepoint_query;
                                 -1 (no query owns) shown gray
    3. Match diff             — for the union of all matched pairs:
                                  green = TP (in matched-pair GT AND pred)
                                  red   = FP (predicted by matched query
                                              but NOT in its matched GT)
                                  blue  = FN (in matched GT but NOT
                                              predicted by matched query)
                                  light gray = neither GT nor pred for any
                                              matched pair
                                Mismatched (unmatched) queries / GT are
                                gray here too — they show up in panels 1
                                and 2 but not in this diff. (See info bar
                                for the unmatched count.)
    4. Per-spacepoint score   — viridis-mapped predictions/spacepoint_score
                                 (sigmoid of the winning query's logit)

Buttons: Prev / Next / Random / Go-to-event-index.

Usage:
    python tools/visualize_shower_clustering_inference.py \\
        --input-dir exp/shower_clustering/run1/inference_epoch4/ \\
        [--port 8050] [--max-points 120000]
"""

import argparse
import glob
import os
import sys
from typing import Optional

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State

from detectoroutline import DetectorOutline


# --------------------------------------------------------------------------
# Constants — must match the dataset/loss class encoding
# --------------------------------------------------------------------------

ORIGIN_CLASS_NAMES = {
    0: "inside", 1: "outside", 2: "on_track", 3: "ghost", 4: "true_track",
    -1: "unknown",
}

# 24 distinct colors — same palette as the dataset visualizer
QUALITATIVE_PALETTE = [
    "#FD3216", "#00FE35", "#6A76FC", "#FED4C4", "#FE00CE", "#0DF9FF",
    "#F6F926", "#FF9616", "#479B55", "#EEA6FB", "#DC587D", "#D626FF",
    "#6E899C", "#00B5F7", "#B68E00", "#C9FBE5", "#FF0092", "#22FFA7",
    "#E3EE9E", "#86CE00", "#BC7196", "#7E7DCD", "#FC6955", "#E48F72",
]

AXIS_TEMPLATE = {
    "showbackground": True,
    "backgroundcolor": "#141414",
    "gridcolor": "rgb(80, 80, 80)",
    "zerolinecolor": "rgb(128, 128, 128)",
    "title_font": {"color": "white"},
    "tickfont": {"color": "white"},
}


def make_layout(title: str, npts: int, height: int = 600) -> dict:
    return {
        "title": {"text": f"{title} ({npts:,} pts)",
                  "font": {"size": 13, "color": "white"}},
        "height": height,
        "margin": {"l": 0, "r": 0, "t": 40, "b": 0},
        "font": {"size": 9, "color": "white"},
        "showlegend": True,
        "legend": {"yanchor": "top", "y": 0.99, "xanchor": "left", "x": 0.01,
                   "font": {"color": "white", "size": 9},
                   "bgcolor": "rgba(20, 20, 20, 0.8)",
                   "itemsizing": "constant"},
        "plot_bgcolor": "#141414",
        "paper_bgcolor": "#141414",
        "scene": {
            "xaxis": {**AXIS_TEMPLATE, "title": "X (cm)"},
            "yaxis": {**AXIS_TEMPLATE, "title": "Y (cm)"},
            "zaxis": {**AXIS_TEMPLATE, "title": "Z (cm)"},
            "aspectratio": {"x": 1, "y": 1, "z": 4},
            "camera": {"eye": {"x": 2, "y": 2, "z": 2},
                       "up": {"x": 0, "y": 1, "z": 0}},
        },
    }


_DET_TRACES = None


def detector_outline_traces():
    global _DET_TRACES
    if _DET_TRACES is None:
        det = DetectorOutline()
        traces = det.getlines(color=(255, 255, 255))
        for t in traces:
            t["showlegend"] = False
            t["hoverinfo"] = "skip"
        _DET_TRACES = traces
    return [dict(t) for t in _DET_TRACES]


# --------------------------------------------------------------------------
# H5 reader
# --------------------------------------------------------------------------

def load_event_h5(path: str) -> dict:
    """Load one inference H5 into a flat dict of numpy arrays + scalars."""
    with h5py.File(path, "r") as f:
        e = f["entry_0"]
        td = e["triplet_data"]
        sf = e["shower_fragments"]
        gi = e["gt_instances"]
        pr = e["predictions"]
        m = pr["matched"]

        coord_center = np.asarray(
            e.attrs.get("coord_center", np.array([125.0, 0.0, 518.0])),
            dtype=np.float32,
        )
        voxel_size_cm = float(e.attrs.get("voxel_size_cm", 5.0))

        # Voxel keys are optional in older inference files
        voxel_keys = (td["voxel_keys"][:].astype(np.int64)
                      if "voxel_keys" in td else None)
        # Fragment membership reconstruction
        f_flat = (sf["pointindices_flat"][:].astype(np.int64)
                  if "pointindices_flat" in sf else np.zeros(0, dtype=np.int64))
        f_counts = (sf["pointindices_counts"][:].astype(np.int64)
                    if "pointindices_counts" in sf
                    else np.zeros(0, dtype=np.int64))

        out = {
            "path": path,
            "name": str(e.attrs.get("name", os.path.basename(path))),
            "run": int(e.attrs.get("run", -1)),
            "subrun": int(e.attrs.get("subrun", -1)),
            "event": int(e.attrs.get("event", -1)),
            "lm_score_threshold": float(e.attrs.get("lm_score_threshold", 0.15)),
            "coord_center": coord_center,
            "voxel_size_cm": voxel_size_cm,

            "coord": td["coord"][:].astype(np.float32),         # (N, 3) cm
            "trackid": td["trackid"][:].astype(np.int64),
            "pid": td["pid"][:].astype(np.int32),
            "origin": td["origin"][:].astype(np.int32),
            "ssnet_label": td["ssnet_label"][:].astype(np.int32),
            "lm_score": td["lm_score"][:].astype(np.float32),
            "voxel_id": td["voxel_id"][:].astype(np.int64),
            "voxel_keys": voxel_keys,
            "gt_instance_id_per_sp": (
                td["gt_instance_id"][:].astype(np.int32)
                if "gt_instance_id" in td
                else np.full(td["coord"].shape[0], -1, dtype=np.int32)
            ),

            "n_fragments": int(sf.attrs.get("num_fragments", 0)),
            "frag_trackid": sf["trackid"][:].astype(np.int64) if "trackid" in sf
                            else np.zeros(0, dtype=np.int64),
            "frag_pointindices_flat": f_flat,
            "frag_pointindices_counts": f_counts,

            "num_gt_instances": int(gi.attrs.get("num_instances", 0)),
            "gt_trunk_trackid": gi["trunk_trackid"][:].astype(np.int64),
            "gt_pid": gi["pid"][:].astype(np.int32),
            "gt_origin_type": gi["origin_type"][:].astype(np.int32),
            "gt_origin_cm": gi["origin_cm"][:].astype(np.float32),
            "gt_n_truth_points": gi["n_truth_points"][:].astype(np.int32),

            "num_queries": int(pr.attrs.get("num_queries", 0)),
            "no_object_class_id": int(pr.attrs.get("no_object_class_id", 5)),
            "pred_class": pr["pred_class"][:].astype(np.int32),
            "pred_class_prob": pr["pred_class_prob"][:].astype(np.float32),
            "is_active": pr["is_active"][:].astype(np.int8),
            "pred_origin_cm": pr["pred_origin_cm"][:].astype(np.float32),
            "spacepoint_query": pr["spacepoint_query"][:].astype(np.int32),
            "spacepoint_score": pr["spacepoint_score"][:].astype(np.float32),
            "voxel_query": pr["voxel_query"][:].astype(np.int32),
            "fragment_query": pr["fragment_query"][:].astype(np.int32),

            "matched_q_idx": m["q_idx"][:].astype(np.int32),
            "matched_k_idx": m["k_idx"][:].astype(np.int32),
            "matched_iou": m["iou"][:].astype(np.float32),
            "matched_cls_match": m["cls_match"][:].astype(np.int8),
            "matched_origin_err_cm": m["origin_err_cm"][:].astype(np.float32),
        }
    out["n_spacepoints"] = out["coord"].shape[0]

    # Reconstruct voxel centers (in cm) and the per-spacepoint fragment-id
    # lookup (-1 if not in any fragment) for fragment-mask coloring.
    if voxel_keys is not None and voxel_keys.shape[0] > 0:
        out["voxel_centers_cm"] = (
            (voxel_keys.astype(np.float32) + 0.5) * voxel_size_cm
            + coord_center
        )
    else:
        out["voxel_centers_cm"] = np.zeros((0, 3), dtype=np.float32)

    sp_frag_id = np.full(out["n_spacepoints"], -1, dtype=np.int32)
    offset = 0
    for fi in range(out["n_fragments"]):
        n = int(f_counts[fi])
        idx = f_flat[offset:offset + n]
        offset += n
        sp_frag_id[idx[(idx >= 0) & (idx < out["n_spacepoints"])]] = fi
    out["sp_fragment_id"] = sp_frag_id
    return out


# --------------------------------------------------------------------------
# Subsampling for browser perf
# --------------------------------------------------------------------------

def build_render_mask(event: dict, max_points: int) -> np.ndarray:
    """Pick which spacepoints to render. Always keep:
      - all spacepoints in any GT instance
      - all spacepoints assigned to any active query (spacepoint scale)
      - all spacepoints belonging to any DBSCAN fragment (so the fragment-
        mask panel has all the relevant points)
    Then random-sample the remainder up to budget.
    """
    n = event["n_spacepoints"]
    if n <= max_points:
        return np.ones(n, dtype=bool)
    keep = np.zeros(n, dtype=bool)
    keep[event["gt_instance_id_per_sp"] >= 0] = True
    keep[event["spacepoint_query"] >= 0] = True
    keep[event["sp_fragment_id"] >= 0] = True
    forced = int(keep.sum())
    budget = max(0, max_points - forced)
    if budget > 0:
        candidates = np.where(~keep)[0]
        if len(candidates) > budget:
            keep[np.random.choice(candidates, budget, replace=False)] = True
        else:
            keep[candidates] = True
    return keep


# --------------------------------------------------------------------------
# Trace helpers
# --------------------------------------------------------------------------

def _scatter_trace(coords, name, color, size=2.0, opacity=0.85,
                   hovertext=None):
    trace = {
        "type": "scatter3d",
        "x": coords[:, 0].tolist(),
        "y": coords[:, 1].tolist(),
        "z": coords[:, 2].tolist(),
        "mode": "markers",
        "name": name,
        "marker": {"color": color, "size": size, "opacity": opacity},
    }
    if hovertext is not None:
        trace["hovertext"] = hovertext
        trace["hoverinfo"] = "text"
    return trace


# --------------------------------------------------------------------------
# Figure builders
# --------------------------------------------------------------------------

def _instance_color_map(instance_ids: np.ndarray) -> tuple:
    """Sort unique non-(-1) instances by occurrence count (largest first),
    map each to a palette color. Returns (sorted_unique, color_lookup)."""
    valid = instance_ids[instance_ids >= 0]
    if valid.size == 0:
        return np.zeros(0, dtype=np.int64), {}
    uniq, counts = np.unique(valid, return_counts=True)
    order = np.argsort(-counts)
    sorted_uniq = uniq[order]
    color_lookup = {
        int(u): QUALITATIVE_PALETTE[i % len(QUALITATIVE_PALETTE)]
        for i, u in enumerate(sorted_uniq)
    }
    return sorted_uniq, color_lookup


def build_gt_instances_figure(event: dict, mask_render: np.ndarray) -> dict:
    coord = event["coord"][mask_render]
    n_render = coord.shape[0]
    sp_inst = event["gt_instance_id_per_sp"][mask_render]
    sorted_uniq, color_lookup = _instance_color_map(sp_inst)

    traces = detector_outline_traces()
    bg = sp_inst < 0
    if bg.any():
        traces.append(_scatter_trace(
            coord[bg], name=f"background ({int(bg.sum()):,})",
            color="rgba(120,120,120,0.25)", size=1.5,
        ))
    for u in sorted_uniq:
        sel = sp_inst == u
        if not sel.any():
            continue
        # Reach into event metadata for class label
        k = int(u)
        pid = int(event["gt_pid"][k])
        otype = int(event["gt_origin_type"][k])
        oname = ORIGIN_CLASS_NAMES.get(otype, "?")
        n = int(event["gt_n_truth_points"][k])
        traces.append(_scatter_trace(
            coord[sel],
            name=f"GT inst {k}: pid={pid} {oname} (n={n})",
            color=color_lookup[k], size=2.2, opacity=0.9,
        ))
    return {"data": traces,
            "layout": make_layout("Ground-Truth Instances", n_render)}


def build_predicted_instances_figure(
    event: dict, mask_render: np.ndarray
) -> dict:
    coord = event["coord"][mask_render]
    n_render = coord.shape[0]
    sp_q = event["spacepoint_query"][mask_render]
    sorted_uniq, color_lookup = _instance_color_map(sp_q)

    traces = detector_outline_traces()
    bg = sp_q < 0
    if bg.any():
        traces.append(_scatter_trace(
            coord[bg], name=f"unassigned ({int(bg.sum()):,})",
            color="rgba(120,120,120,0.25)", size=1.5,
        ))
    no_object_id = event["no_object_class_id"]
    for u in sorted_uniq:
        sel = sp_q == u
        if not sel.any():
            continue
        q = int(u)
        pclass = int(event["pred_class"][q])
        pclass_prob = float(event["pred_class_prob"][q])
        pname = ORIGIN_CLASS_NAMES.get(pclass, "?")
        traces.append(_scatter_trace(
            coord[sel],
            name=(f"pred q{q}: {pname} (p={pclass_prob:.2f}, "
                  f"n={int(sel.sum()):,})"),
            color=color_lookup[q], size=2.2, opacity=0.9,
        ))
    return {"data": traces,
            "layout": make_layout("Predicted Instances (per-spacepoint argmax)",
                                  n_render)}


def build_predicted_voxel_figure(event: dict, mask_render: np.ndarray) -> dict:
    """Color voxel CENTERS by `predictions/voxel_query`. Voxels assigned to
    no active query are gray.

    The voxel scale carries explicit positional info via the voxel PE,
    so this panel is the cleanest spatial-coherence sanity check for
    each query — if a query's voxels are scattered across the detector,
    the model's spatial localization is genuinely confused (not just an
    artifact of the spacepoint-mask head's lack of position).

    Sort order: voxels-per-query descending. Single-voxel queries
    (★ in the legend) trail multi-voxel ones but always show. Active
    queries that win the argmax on no voxel are counted in the title
    but invisible in this panel.
    """
    centers = event["voxel_centers_cm"]
    n_v = centers.shape[0]
    if n_v == 0:
        return {"data": detector_outline_traces(),
                "layout": make_layout(
                    "Predicted Voxel Mask  (no voxels)", 0)}
    voxel_q = event["voxel_query"]

    # Per-query voxel count
    is_active = event["is_active"].astype(bool)
    n_active = int(is_active.sum())
    qs_with_voxel = np.unique(voxel_q[voxel_q >= 0])
    n_v_per_q = {int(q): int((voxel_q == q).sum()) for q in qs_with_voxel}

    active_q_idx = np.where(is_active)[0].astype(np.int32)
    n_active_with_v = int(np.isin(active_q_idx, qs_with_voxel).sum())
    n_active_without_v = n_active - n_active_with_v
    n_singletons = sum(1 for v in n_v_per_q.values() if v == 1)
    n_multivox = sum(1 for v in n_v_per_q.values() if v >= 2)

    # Sort by voxel count desc; tiebreak on query id for stability
    sorted_qs = sorted(
        qs_with_voxel.tolist(),
        key=lambda q: (-n_v_per_q[int(q)], int(q)),
    )
    color_lookup = {
        int(q): QUALITATIVE_PALETTE[i % len(QUALITATIVE_PALETTE)]
        for i, q in enumerate(sorted_qs)
    }

    traces = detector_outline_traces()
    bg = voxel_q < 0
    if bg.any():
        traces.append(_scatter_trace(
            centers[bg], name=f"unassigned ({int(bg.sum()):,})",
            color="rgba(120,120,120,0.35)", size=2.5,
        ))
    for q in sorted_qs:
        sel = voxel_q == q
        if not sel.any():
            continue
        pclass = int(event["pred_class"][q])
        pclass_prob = float(event["pred_class_prob"][q])
        pname = ORIGIN_CLASS_NAMES.get(pclass, "?")
        nv = n_v_per_q[int(q)]
        marker = "★" if nv == 1 else " "
        traces.append(_scatter_trace(
            centers[sel],
            name=(f"{marker} pred q{q}: {pname} (p={pclass_prob:.2f}, "
                  f"V={nv})"),
            color=color_lookup[int(q)], size=4.5, opacity=0.9,
        ))
    title = (
        f"Predicted Voxel Mask  (5 cm grid, V={n_v}, active_q={n_active}, "
        f"multi-vox={n_multivox}, single-vox★={n_singletons}, "
        f"no-vox={n_active_without_v})"
    )
    return {"data": traces, "layout": make_layout(title, n_v)}


def build_predicted_fragment_figure(
    event: dict, mask_render: np.ndarray
) -> dict:
    """Color SPACEPOINTS by their fragment's predicted query assignment.

    For each spacepoint, look up its fragment id (`sp_fragment_id`); if
    that fragment is assigned to active query q, color the spacepoint
    by q. Spacepoints not in any fragment are gray.

    Fragment tokens have explicit positional info (centroid + PCA axis
    + bbox extent) so this panel is the cheapest sanity check on
    "are queries claiming spatially coherent fragments".

    Single-fragment queries DO appear here (with `frags=1` in the
    legend). The sort order is fragment count first, spacepoint count
    second, so multi-fragment groups lead and singletons follow. Active
    queries whose fragment mask never wins the per-fragment argmax own
    zero fragments — they're invisible in this panel but counted in the
    title.
    """
    coord = event["coord"][mask_render]
    n_render = coord.shape[0]
    sp_frag = event["sp_fragment_id"][mask_render]
    frag_q_arr = event["fragment_query"]

    # Per-spacepoint predicted query, via fragment-id → fragment_query lookup.
    sp_pred_q = np.full(n_render, -1, dtype=np.int32)
    valid = sp_frag >= 0
    if valid.any() and frag_q_arr.shape[0] > 0:
        sp_pred_q[valid] = frag_q_arr[sp_frag[valid]]

    # Per-query bookkeeping: fragment count + spacepoint count + class.
    # Sort the legend by (fragments desc, spacepoints desc) so multi-fragment
    # groups lead and singletons trail but stay visible.
    is_active = event["is_active"].astype(bool)
    n_active = int(is_active.sum())
    qs_with_frag = (np.unique(frag_q_arr[frag_q_arr >= 0])
                    if frag_q_arr.size else np.zeros(0, dtype=np.int32))
    n_frag_per_q = {int(q): int((frag_q_arr == q).sum())
                    for q in qs_with_frag}
    n_sp_per_q = {int(q): int((sp_pred_q == q).sum())
                  for q in qs_with_frag}

    # Active queries that own zero fragments — invisible here, counted in title.
    active_q_idx = np.where(is_active)[0].astype(np.int32)
    n_active_with_frag = int(np.isin(active_q_idx, qs_with_frag).sum())
    n_active_without_frag = n_active - n_active_with_frag

    n_singletons = sum(1 for v in n_frag_per_q.values() if v == 1)
    n_multifrag = sum(1 for v in n_frag_per_q.values() if v >= 2)

    # Build sort order: fragments desc, then spacepoints desc.
    sorted_qs = sorted(
        qs_with_frag.tolist(),
        key=lambda q: (-n_frag_per_q[int(q)], -n_sp_per_q[int(q)]),
    )
    color_lookup = {
        int(q): QUALITATIVE_PALETTE[i % len(QUALITATIVE_PALETTE)]
        for i, q in enumerate(sorted_qs)
    }

    traces = detector_outline_traces()
    bg = sp_pred_q < 0
    if bg.any():
        traces.append(_scatter_trace(
            coord[bg],
            name=(f"unassigned/non-fragment ({int(bg.sum()):,})"),
            color="rgba(120,120,120,0.25)", size=1.5,
        ))
    for q in sorted_qs:
        sel = sp_pred_q == q
        if not sel.any():
            continue
        pclass = int(event["pred_class"][q])
        pclass_prob = float(event["pred_class_prob"][q])
        pname = ORIGIN_CLASS_NAMES.get(pclass, "?")
        nf = n_frag_per_q[int(q)]
        marker = "★" if nf == 1 else " "  # flag singletons in the legend
        traces.append(_scatter_trace(
            coord[sel],
            name=(f"{marker} pred q{q}: {pname} (p={pclass_prob:.2f}, "
                  f"frags={nf}, sp={int(sel.sum()):,})"),
            color=color_lookup[int(q)], size=2.2, opacity=0.9,
        ))
    title = (
        f"Predicted Fragment Mask  "
        f"(active_q={n_active}, multi-frag={n_multifrag}, "
        f"single-frag★={n_singletons}, no-frag={n_active_without_frag})"
    )
    return {"data": traces, "layout": make_layout(title, n_render)}


def build_match_diff_figure(event: dict, mask_render: np.ndarray) -> dict:
    """Per-spacepoint TP/FP/FN coloring restricted to matched pairs.

    For each matched pair (q, k):
        positives_pred_n = points where spacepoint_query == q
        positives_gt_n   = points where gt_instance_id_per_sp == k
        TP[n] = pred AND gt
        FP[n] = pred AND NOT gt   (model said "this is k" but GT says no)
        FN[n] = NOT pred AND gt   (GT says "this is k" but model said no)

    We OR these across all matched pairs and color accordingly.
    Spacepoints that aren't part of ANY matched pair stay gray.
    """
    coord = event["coord"][mask_render]
    n = event["n_spacepoints"]
    n_render = coord.shape[0]

    sp_q = event["spacepoint_query"]
    sp_inst = event["gt_instance_id_per_sp"]
    q_idx = event["matched_q_idx"]
    k_idx = event["matched_k_idx"]

    tp = np.zeros(n, dtype=bool)
    fp = np.zeros(n, dtype=bool)
    fn = np.zeros(n, dtype=bool)
    for p in range(len(q_idx)):
        q = int(q_idx[p])
        k = int(k_idx[p])
        in_pred = sp_q == q
        in_gt = sp_inst == k
        tp |= in_pred & in_gt
        fp |= in_pred & (~in_gt)
        fn |= (~in_pred) & in_gt

    tp_r = tp[mask_render]
    fp_r = fp[mask_render]
    fn_r = fn[mask_render]
    bg_r = ~(tp_r | fp_r | fn_r)

    traces = detector_outline_traces()
    if bg_r.any():
        traces.append(_scatter_trace(
            coord[bg_r], name=f"neither ({int(bg_r.sum()):,})",
            color="rgba(120,120,120,0.2)", size=1.5,
        ))
    if tp_r.any():
        traces.append(_scatter_trace(
            coord[tp_r], name=f"TP ({int(tp_r.sum()):,})",
            color="#22FFA7", size=2.2, opacity=0.9,
        ))
    if fp_r.any():
        traces.append(_scatter_trace(
            coord[fp_r], name=f"FP — over-predict ({int(fp_r.sum()):,})",
            color="#FD3216", size=2.0, opacity=0.85,
        ))
    if fn_r.any():
        traces.append(_scatter_trace(
            coord[fn_r], name=f"FN — missed ({int(fn_r.sum()):,})",
            color="#3060FF", size=2.0, opacity=0.85,
        ))
    return {"data": traces,
            "layout": make_layout("Match diff (matched pairs only)",
                                  n_render)}


def build_score_figure(event: dict, mask_render: np.ndarray) -> dict:
    coord = event["coord"][mask_render]
    score = event["spacepoint_score"][mask_render]
    n_render = coord.shape[0]

    traces = detector_outline_traces()
    traces.append({
        "type": "scatter3d",
        "x": coord[:, 0].tolist(),
        "y": coord[:, 1].tolist(),
        "z": coord[:, 2].tolist(),
        "mode": "markers",
        "name": f"all ({n_render:,})",
        "marker": {
            "color": score.tolist(),
            "colorscale": "Viridis",
            "size": 2,
            "opacity": 0.85,
            "cmin": 0.0,
            "cmax": 1.0,
            "colorbar": {"title": "score (sigmoid)", "len": 0.6},
        },
        "customdata": np.stack([score], axis=1),
        "hovertemplate": (
            "<b>x</b>: %{x:.1f} cm<br>"
            "<b>y</b>: %{y:.1f} cm<br>"
            "<b>z</b>: %{z:.1f} cm<br>"
            "<b>score</b>: %{customdata[0]:.3f}<extra></extra>"
        ),
    })
    return {"data": traces,
            "layout": make_layout(
                "Per-spacepoint score (best active query)", n_render)}


# --------------------------------------------------------------------------
# Info bar
# --------------------------------------------------------------------------

def build_info_string(event: dict, idx: int, n_total: int,
                      n_render: int) -> str:
    K = event["num_gt_instances"]
    n_active = int(event["is_active"].sum())
    P = int(len(event["matched_q_idx"]))
    iou = event["matched_iou"]
    iou_mean = float(iou.mean()) if iou.size > 0 else 0.0
    iou_med = float(np.median(iou)) if iou.size > 0 else 0.0
    iou_p25 = float(np.percentile(iou, 25)) if iou.size > 0 else 0.0
    cls_acc = (float(event["matched_cls_match"].mean())
               if iou.size > 0 else 0.0)
    oerr = event["matched_origin_err_cm"]
    oerr_med = float(np.median(oerr)) if oerr.size > 0 else 0.0
    oerr_mean = float(oerr.mean()) if oerr.size > 0 else 0.0

    # Per-class IoU (matched pairs only, by GT class)
    per_class_lines = []
    for cls_id in (0, 1, 2, 3, 4):
        mask = (event["gt_origin_type"][event["matched_k_idx"]] == cls_id)
        if mask.any():
            cls_iou = float(iou[mask].mean())
            per_class_lines.append(
                f"{ORIGIN_CLASS_NAMES[cls_id]} {cls_iou:.3f} (n={int(mask.sum())})"
            )
    per_class_str = " | ".join(per_class_lines) if per_class_lines else "(no matches)"

    return (
        f"Entry {idx + 1}/{n_total}  |  "
        f"{event['name']} (run/subrun/event {event['run']}/"
        f"{event['subrun']}/{event['event']})  |  "
        f"τ={event['lm_score_threshold']:.3f}  |  "
        f"n_sp={event['n_spacepoints']:,} (rendered {n_render:,})  |  "
        f"GT={K} active_q={n_active} matched={P}  |  "
        f"cls_acc(matched)={cls_acc:.3f}  |  "
        f"iou mean/med/p25={iou_mean:.3f}/{iou_med:.3f}/{iou_p25:.3f}  |  "
        f"origin_err_cm mean/med={oerr_mean:.1f}/{oerr_med:.1f}  ||  "
        f"per-class IoU: {per_class_str}"
    )


# --------------------------------------------------------------------------
# Main / Dash
# --------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input-dir", required=True,
                   help="Directory of inference H5s "
                        "(written by run_shower_clustering_inference.py)")
    p.add_argument("--input-glob", default="*.h5",
                   help="Glob pattern within --input-dir (default: '*.h5')")
    p.add_argument("--max-points", type=int, default=120000,
                   help="Cap on rendered points per panel.")
    p.add_argument("--port", type=int, default=8050)
    p.add_argument("--entry", type=int, default=0,
                   help="Initial entry index")
    return p.parse_args()


args = parse_args()
files = sorted(glob.glob(os.path.join(args.input_dir, args.input_glob)))
if not files:
    print(f"[viz] no H5s found in {args.input_dir}/{args.input_glob}",
          file=sys.stderr)
    sys.exit(1)
print(f"[viz] {len(files)} inference H5 files")
for f in files[:5]:
    print(f"   {os.path.basename(f)}")
if len(files) > 5:
    print(f"   ... and {len(files) - 5} more")

# ---------------------------------------------------------------- Dash app

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport",
                "content": "width=device-width, initial-scale=1"}],
)

GRAPH_STYLE = {"width": "33.3%", "display": "inline-block",
               "verticalAlign": "top"}
BUTTON_BASE = {"fontSize": "14px", "padding": "8px 16px", "color": "white",
               "border": "none", "borderRadius": "5px", "cursor": "pointer",
               "marginRight": "8px"}

app.layout = html.Div(
    style={"backgroundColor": "#1e1e1e", "padding": "16px",
           "minHeight": "100vh", "fontFamily": "monospace"},
    children=[
        html.Div([
            html.H2("Shower-Clustering Inference Visualizer",
                    style={"color": "white", "display": "inline-block",
                           "marginRight": "20px"}),
            html.Button("Prev", id="prev-btn", n_clicks=0,
                        style={**BUTTON_BASE, "backgroundColor": "#607D8B"}),
            html.Button("Next", id="next-btn", n_clicks=0,
                        style={**BUTTON_BASE, "backgroundColor": "#4CAF50"}),
            html.Button("Random", id="random-btn", n_clicks=0,
                        style={**BUTTON_BASE, "backgroundColor": "#FF9800"}),
            dcc.Input(id="entry-input", type="number",
                      value=args.entry, min=0, max=max(len(files) - 1, 0),
                      style={"fontSize": "13px", "padding": "6px",
                             "width": "100px", "borderRadius": "5px",
                             "border": "1px solid #555",
                             "backgroundColor": "#333", "color": "white",
                             "marginRight": "8px"}),
            html.Button("Go", id="go-btn", n_clicks=0,
                        style={**BUTTON_BASE, "backgroundColor": "#2196F3"}),
        ], style={"marginBottom": "8px"}),
        html.Div(id="entry-label",
                 style={"color": "#ccc", "fontSize": "12px",
                        "marginBottom": "12px"}),
        # Row 1: GT, predicted (spacepoint argmax), match diff
        html.Div([
            html.Div([dcc.Graph(id="gt-graph",
                                config={"scrollZoom": True})],
                     style=GRAPH_STYLE),
            html.Div([dcc.Graph(id="pred-graph",
                                config={"scrollZoom": True})],
                     style=GRAPH_STYLE),
            html.Div([dcc.Graph(id="diff-graph",
                                config={"scrollZoom": True})],
                     style=GRAPH_STYLE),
        ]),
        # Row 2: predicted voxel mask, predicted fragment mask, score
        html.Div([
            html.Div([dcc.Graph(id="voxel-graph",
                                config={"scrollZoom": True})],
                     style=GRAPH_STYLE),
            html.Div([dcc.Graph(id="frag-graph",
                                config={"scrollZoom": True})],
                     style=GRAPH_STYLE),
            html.Div([dcc.Graph(id="score-graph",
                                config={"scrollZoom": True})],
                     style=GRAPH_STYLE),
        ]),
        dcc.Store(id="current-idx", data=args.entry),
    ],
)


@app.callback(
    [Output("gt-graph", "figure"),
     Output("pred-graph", "figure"),
     Output("diff-graph", "figure"),
     Output("voxel-graph", "figure"),
     Output("frag-graph", "figure"),
     Output("score-graph", "figure"),
     Output("entry-label", "children"),
     Output("current-idx", "data")],
    [Input("prev-btn", "n_clicks"),
     Input("next-btn", "n_clicks"),
     Input("random-btn", "n_clicks"),
     Input("go-btn", "n_clicks")],
    [State("current-idx", "data"),
     State("entry-input", "value")],
)
def update_display(_p, _n, _r, _g, current_idx, entry_input):
    ctx = dash.callback_context
    triggered = ctx.triggered[0]["prop_id"] if ctx.triggered else ""
    n_total = len(files)
    if "prev-btn" in triggered:
        idx = (int(current_idx) - 1) % n_total
    elif "next-btn" in triggered:
        idx = (int(current_idx) + 1) % n_total
    elif "random-btn" in triggered:
        idx = int(np.random.randint(0, n_total))
    elif "go-btn" in triggered and entry_input is not None:
        idx = int(np.clip(entry_input, 0, n_total - 1))
    else:
        idx = int(current_idx) if current_idx is not None else 0

    try:
        event = load_event_h5(files[idx])
        mask = build_render_mask(event, args.max_points)
    except Exception as exc:
        print(f"[viz] error loading {files[idx]}: {exc}")
        import traceback
        traceback.print_exc()
        empty = {"data": [], "layout": make_layout("error", 0)}
        return empty, empty, empty, empty, empty, empty, f"Error: {exc}", idx

    figs = (
        build_gt_instances_figure(event, mask),
        build_predicted_instances_figure(event, mask),
        build_match_diff_figure(event, mask),
        build_predicted_voxel_figure(event, mask),
        build_predicted_fragment_figure(event, mask),
        build_score_figure(event, mask),
    )
    info = build_info_string(event, idx, n_total, int(mask.sum()))
    return *figs, info, idx


if __name__ == "__main__":
    app.run(debug=False, port=args.port)
