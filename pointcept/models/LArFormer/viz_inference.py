"""Shared visualization helpers for LArFormer slicer + Stage-3 predictions.

Renders predictions written by `pointcept/models/LArFormer/inference.py`
(`slicer_predict_event` and `stage3_predict_event`). Used by:

  - `tools/visualize_larformer_gt.py` (slicer prediction overlay, via
    its `--slicerpred-dir` mode)
  - `tools/visualize_stage3_larformer_from_cached.py` (Stage-3
    prediction overlay, via its `--stage3pred-dir` mode)

This module owns the Plotly figure builders for the prediction panels
+ a few small shared utilities (color helpers, confidence thresholding,
hover-text formatting, detector outline). Loaders live in
`pointcept/models/LArFormer/inference.py` (`load_event_h5`).

The two viz CLIs in `tools/` are kept as thin Dash apps; the actual
prediction-rendering code lives here.
"""
from __future__ import annotations

import colorsys
from typing import Optional

import numpy as np
import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# Color helpers (consistent across slicer + Stage-3 panels)
# ---------------------------------------------------------------------------

def instance_color(k: int, origin_type: int = -1, alpha: float = 1.0) -> str:
    """Distinct HSV color per GT instance index. origin_type=0 (nu) gets red."""
    if origin_type == 0:
        return f"rgba(255,60,60,{alpha})"
    h = (k * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.80, 0.95)
    return f"rgba({int(r*255)},{int(g*255)},{int(b*255)},{alpha})"


def class_color(c: int, alpha: float = 1.0) -> str:
    """Stable per-class color used for cls heads (-1 = ignore = grey)."""
    if c < 0:
        return f"rgba(180,180,180,{0.5 * alpha})"
    palette = [
        "rgba(255,60,60,{a})",     # 0 e
        "rgba(60,140,255,{a})",    # 1 γ
        "rgba(80,200,80,{a})",     # 2 μ
        "rgba(255,160,40,{a})",    # 3 π
        "rgba(180,60,220,{a})",    # 4 p
        "rgba(40,200,200,{a})",    # 5 other_track
        "rgba(120,120,120,{a})",   # 6 (unused)
        "rgba(70,70,70,{a})",      # 7 no_object
    ]
    return palette[c % len(palette)].format(a=alpha)


def track_id_color(tid: int, origin_type: int = -1, alpha: float = 1.0) -> str:
    """Per-track stable color from a golden-ratio hash of the trackid.

    Matches `tools/visualize_larformer_gt.py:_track_id_color` byte-for-byte
    when `alpha=1.0` so a given physical track gets the SAME color in both
    the GT and the prediction panel. The pred-side particle-idx color mode
    relies on this for visual cross-referencing of matched queries.

    The `:g` alpha format renders `1.0` as `1` and `0.5` as `0.5`, so
    fully-opaque values match the GT helper's `1` literal exactly.
    """
    if origin_type == 0:
        return f"rgba(255,60,60,{alpha:g})"
    if tid < 0:
        return f"rgba(150,150,150,{0.5 * alpha:g})"
    # Match visualize_larformer_gt._track_id_color: abs(tid), sat=0.80, val=0.95.
    h = (abs(int(tid)) * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.80, 0.95)
    return f"rgba({int(r*255)},{int(g*255)},{int(b*255)},{alpha:g})"


# 7-class particle taxonomy used by Stage-3 (matches
# `configs/lartpc/larformer-particle-v1*.py`'s `names`).
PARTICLE_CLASS_NAMES = (
    "e", "gamma", "mu", "pi", "p", "other", "(unused)", "no_object",
)

# Short symbolic labels for the same 8 slots. Used in hover text + legend
# labels so the prediction panel reads like physics ("μ± mostly here")
# instead of class-id arithmetic ("cls=2 mostly here").
PARTICLE_CLASS_SYMBOLS = (
    "e±",       # 0
    "γ",        # 1
    "μ±",       # 2
    "π±",       # 3
    "p",        # 4
    "other",    # 5
    "—",        # 6 (unused — never assigned)
    "∅",        # 7 no_object
)


def _class_symbol(c: int) -> str:
    """Particle symbol for a class id; '?' for out-of-range."""
    c = int(c)
    if 0 <= c < len(PARTICLE_CLASS_SYMBOLS):
        return PARTICLE_CLASS_SYMBOLS[c]
    return "?"


# ---------------------------------------------------------------------------
# Detector outline (inlined: tools/detectoroutline.py's class but no
# tools/ import dependency from pointcept/).
# ---------------------------------------------------------------------------

_TPC_X = (0.0, 256.0)
_TPC_Y = (-117.0, 117.0)
_TPC_Z = (0.0, 1036.0)


def make_detector_outline_trace(color: str = "rgba(120,120,120,0.4)",
                                width: int = 2) -> list[go.Scatter3d]:
    """Plotly Scatter3d traces drawing the MicroBooNE TPC boundary box."""
    top = [
        [_TPC_X[0], _TPC_Y[1], _TPC_Z[0]],
        [_TPC_X[1], _TPC_Y[1], _TPC_Z[0]],
        [_TPC_X[1], _TPC_Y[1], _TPC_Z[1]],
        [_TPC_X[0], _TPC_Y[1], _TPC_Z[1]],
        [_TPC_X[0], _TPC_Y[1], _TPC_Z[0]],
    ]
    bot = [
        [_TPC_X[0], _TPC_Y[0], _TPC_Z[0]],
        [_TPC_X[1], _TPC_Y[0], _TPC_Z[0]],
        [_TPC_X[1], _TPC_Y[0], _TPC_Z[1]],
        [_TPC_X[0], _TPC_Y[0], _TPC_Z[1]],
        [_TPC_X[0], _TPC_Y[0], _TPC_Z[0]],
    ]
    traces: list[go.Scatter3d] = []
    for boundary in (top, bot):
        traces.append(go.Scatter3d(
            x=[p[2] for p in boundary],
            y=[p[0] for p in boundary],
            z=[p[1] for p in boundary],
            mode="lines",
            line=dict(color=color, width=width),
            showlegend=False, hoverinfo="skip",
        ))
    for i in range(4):
        a, b = top[i], bot[i]
        traces.append(go.Scatter3d(
            x=[a[2], b[2]], y=[a[0], b[0]], z=[a[1], b[1]],
            mode="lines",
            line=dict(color=color, width=width),
            showlegend=False, hoverinfo="skip",
        ))
    return traces


# ---------------------------------------------------------------------------
# Confidence thresholding (shared per-SP demotion rule)
# ---------------------------------------------------------------------------

def apply_pred_threshold(
    pred_mask_prob: Optional[np.ndarray],
    pred_class: np.ndarray,
    pred_secondary_id: np.ndarray,
    no_object_class_id: int,
    min_mask_prob: float,
):
    """Demote per-SP assignments whose mask sigm < `min_mask_prob` to
    no_object. Returns (pred_class', pred_secondary_id', low_conf_mask).

    `pred_secondary_id` is whatever per-SP identifier the caller wants
    demoted to -1 alongside the class (e.g. pred_slice_id, pred_query,
    or pred_particle_idx). Returned unchanged for SPs above the floor.
    If pred_mask_prob is None or min_mask_prob <= 0, no demotion is
    applied (returns inputs unchanged + an all-False mask).
    """
    n = pred_class.shape[0]
    if pred_mask_prob is None or min_mask_prob <= 0.0 or n == 0:
        return pred_class, pred_secondary_id, np.zeros(n, dtype=bool)
    low = pred_mask_prob < float(min_mask_prob)
    pc = pred_class.copy(); pc[low] = int(no_object_class_id)
    ps = pred_secondary_id.copy(); ps[low] = -1
    return pc, ps, low


# ---------------------------------------------------------------------------
# Stage-3 prediction renderer
# ---------------------------------------------------------------------------

# NOTE on axis labels: per-SP traces plot detector coords as
#   plotly_x = coord[:, 2] (= detector Z, beam direction)
#   plotly_y = coord[:, 0] (= detector X, drift)
#   plotly_z = coord[:, 1] (= detector Y, vertical)
# The hover labels "x="/"y="/"z=" below refer to DETECTOR axes (matching
# the scene axis titles and the diamond-hover convention), so the
# placeholders are remapped: detector_x = plotly_y, detector_y = plotly_z,
# detector_z = plotly_x.
_STAGE3_HOVER_TEMPLATE = (
    "x=%{y:.1f} y=%{z:.1f} z=%{x:.1f}<br>"
    "mask_prob=%{customdata[0]:.3f}  "
    "pred=%{customdata[1]}  GT=%{customdata[5]}<br>"
    "query=%{customdata[2]}  "
    "pred_particle_idx=%{customdata[3]}  trackid=%{customdata[4]}<br>"
    "GT origin target (cm): %{customdata[14]}<br>"
    "<i>class probs (assigned query):</i><br>"
    "&nbsp;&nbsp;e±=%{customdata[6]:.2f}  "
    "γ=%{customdata[7]:.2f}  "
    "μ±=%{customdata[8]:.2f}  "
    "π±=%{customdata[9]:.2f}<br>"
    "&nbsp;&nbsp;p=%{customdata[10]:.2f}  "
    "other=%{customdata[11]:.2f}  "
    "∅=%{customdata[13]:.2f}"
    "<extra></extra>"
)


def _stage3_customdata(pred):
    """(N, 15) mixed-dtype customdata array for the Stage-3 SP hover.

    Layout (column → field):
        0   mask_prob (float)
        1   pred_class as particle symbol (str: 'e±', 'γ', …)
        2   pred_query (int — matches the query id on each origin diamond)
        3   pred_particle_idx (int — index into the K GT instances; -1 if none)
        4   pred_particle_trackid (int — physical Geant4 trackid; -1 if none)
        5   particle_class_id_gt as particle symbol (str; '—' if no GT)
        6-13  per-class softmax probability of the assigned query (8 floats)
        14  GT origin target as a pre-formatted string (det cm); reads
            "x=… y=… z=…" for matched queries and "(unmatched)" otherwise.
            Pre-formatted so the hover template stays a single static string.

    Plotly accepts object-dtype customdata; each cell serializes as its
    actual type. Strings render literally via `%{customdata[c]}`; floats
    accept `%{customdata[c]:.2f}` formatting in the template.
    """
    n_classes = len(PARTICLE_CLASS_SYMBOLS)
    n = pred["stage3/coord"].shape[0]
    # 6 base + n_classes per-class + 1 GT-origin string
    n_cols = 6 + n_classes + 1
    out = np.empty((n, n_cols), dtype=object)

    pred_mask_prob = np.asarray(
        pred.get("stage3/pred_mask_prob", np.zeros(n)), dtype=np.float64)
    pred_class = np.asarray(
        pred.get("stage3/pred_class",
                  np.full(n, n_classes - 1)), dtype=np.int64)
    pred_query = np.asarray(
        pred.get("stage3/pred_query", np.full(n, -1)), dtype=np.int64)
    pred_pidx = np.asarray(
        pred.get("stage3/pred_particle_idx",
                  np.full(n, -1)), dtype=np.int64)
    pred_ptid = np.asarray(
        pred.get("stage3/pred_particle_trackid",
                  np.full(n, -1)), dtype=np.int64)
    pcid_gt = np.asarray(
        pred.get("stage3/particle_class_id_gt",
                  np.full(n, -1)), dtype=np.int64)

    # Per-query class probs (Q, C). For each SP, copy the row for its
    # assigned query. SPs whose pred_query is -1 (no active query) get
    # zeros — their class-prob block will read 0.00 across the board.
    cls_probs = pred.get("stage3_queries/class_probs", None)
    if cls_probs is not None and cls_probs.size > 0:
        pq_safe = pred_query.copy()
        pq_safe[pq_safe < 0] = 0
        per_sp_probs = cls_probs[pq_safe].astype(np.float64)
        per_sp_probs[pred_query < 0] = 0.0
    else:
        per_sp_probs = np.zeros((n, n_classes), dtype=np.float64)

    # Per-SP GT origin lookup. Each SP's pred_query maps through
    # stage3_queries/matched_gt_idx → stage3_gt/origin_cm. Pre-format
    # so the hover template only has to drop in one string. Origins are
    # in DETECTOR cm and include the same space-charge + drift-time
    # corrections that the loss saw at training time.
    matched_gt = pred.get("stage3_queries/matched_gt_idx", None)
    gt_origin_cm = pred.get("stage3_gt/origin_cm", None)
    gt_origin_str = ["(unmatched)"] * n
    if matched_gt is not None and gt_origin_cm is not None and gt_origin_cm.size > 0:
        matched_gt = np.asarray(matched_gt).astype(np.int64)
        Q = matched_gt.shape[0]
        for i in range(n):
            pq = int(pred_query[i])
            if pq < 0 or pq >= Q:
                continue
            k = int(matched_gt[pq])
            if k < 0 or k >= gt_origin_cm.shape[0]:
                continue
            gx, gy, gz = (float(gt_origin_cm[k, 0]),
                          float(gt_origin_cm[k, 1]),
                          float(gt_origin_cm[k, 2]))
            gt_origin_str[i] = f"x={gx:.1f} y={gy:.1f} z={gz:.1f}"

    out[:, 0] = pred_mask_prob
    out[:, 1] = [_class_symbol(c) for c in pred_class]
    out[:, 2] = pred_query
    out[:, 3] = pred_pidx
    out[:, 4] = pred_ptid
    out[:, 5] = [
        _class_symbol(c) if c >= 0 else "—" for c in pcid_gt
    ]
    for c in range(n_classes):
        out[:, 6 + c] = per_sp_probs[:, c]
    out[:, 6 + n_classes] = gt_origin_str
    return out


def stage3_color_by_options() -> list[dict]:
    """List of `{label, value}` dropdown options for the Stage-3
    prediction panel."""
    return [
        {"label": "predicted particle (matched GT)",
         "value": "pred_particle_idx"},
        {"label": "predicted class",
         "value": "pred_class"},
        {"label": "predicted query id",
         "value": "pred_query"},
        {"label": "predicted mask probability",
         "value": "pred_mask_prob"},
        {"label": "GT particle class id (per-SP)",
         "value": "particle_class_id_gt"},
        {"label": "correct vs GT (class)",
         "value": "pred_class_correct"},
        {"label": "stage-2 nu mask prob (cache telemetry)",
         "value": "stage2_nu_mask_prob"},
        {"label": "source_mask (cache provenance bitmask)",
         "value": "source_mask"},
    ]


def figure_for_stage3_prediction(
    pred: Optional[dict],
    color_by: str = "pred_particle_idx",
    *,
    min_mask_prob: float = 0.0,
    show_origin_lines: bool = True,
    show_origin_diamonds: bool = True,
    sp_marker_size: float = 3.0,
    title_suffix: str = "",
) -> go.Figure:
    """Build the Stage-3 prediction 3D Plotly figure.

    `pred` is the dict returned by `inference.load_event_h5(...)` on a
    stage3pred_*.h5 file (must contain the `stage3*/...` keys).

    `color_by` picks one of the modes returned by
    `stage3_color_by_options`. Per-SP traces are split into legend
    entries by the color label so the user can toggle each on/off.

    `min_mask_prob` demotes SPs below the floor to no_object (low-
    confidence). Tracking these as a separate trace makes the "model
    unsure" vs "model wrong" distinction visible.

    `show_origin_diamonds` overlays one diamond per active query at
    its predicted origin (colored by predicted class). `show_origin_lines`
    additionally draws a line from each matched pair's pred origin to
    its GT origin, length-encoding the L2 error.
    """
    fig = go.Figure(data=make_detector_outline_trace())

    if pred is None:
        fig.update_layout(
            title=f"No Stage-3 prediction available{title_suffix}",
            **_default_scene_kwargs(),
        )
        return fig

    coord = pred["stage3/coord"].astype(np.float32)
    if coord.shape[0] == 0:
        fig.update_layout(
            title=f"Empty Stage-3 event{title_suffix}",
            **_default_scene_kwargs(),
        )
        return fig

    no_obj = int(pred.get("stage3_meta_no_object_class_id",
                          pred.get("stage3_meta/no_object_class_id", 7)))
    pred_class = pred.get("stage3/pred_class",
                          np.full(coord.shape[0], no_obj, dtype=np.int64)).astype(np.int64)
    pred_mask_prob = pred.get("stage3/pred_mask_prob", None)
    pred_q = pred.get("stage3/pred_query",
                      -np.ones(coord.shape[0], dtype=np.int64)).astype(np.int64)
    pred_pidx = pred.get("stage3/pred_particle_idx",
                         -np.ones(coord.shape[0], dtype=np.int64)).astype(np.int64)
    pred_ptid = pred.get("stage3/pred_particle_trackid",
                         -np.ones(coord.shape[0], dtype=np.int64)).astype(np.int64)
    pcid_gt = pred.get("stage3/particle_class_id_gt",
                       -np.ones(coord.shape[0], dtype=np.int64)).astype(np.int64)

    # Apply confidence threshold
    pred_class, pred_pidx_thr, low_conf = apply_pred_threshold(
        pred_mask_prob, pred_class, pred_pidx,
        no_object_class_id=no_obj, min_mask_prob=min_mask_prob,
    )
    pred_q_thr = pred_q.copy(); pred_q_thr[low_conf] = -1
    pred_ptid_thr = pred_ptid.copy(); pred_ptid_thr[low_conf] = -1

    customdata = _stage3_customdata(pred)

    # Pick value array + color function per mode
    if color_by == "pred_particle_idx":
        # GROUP by pred_particle_idx (one trace per matched GT) but
        # COLOR by pred_particle_trackid so each predicted-particle's
        # color matches the GT panel's color for that same physical
        # track. Within a group, every SP shares the same trackid by
        # construction (a panoptic argmax assignment to a query maps
        # to exactly one GT instance, with one trackid).
        values = pred_pidx_thr

        def _idx_to_trackid(idx: int) -> int:
            if idx < 0:
                return -1
            mask = (pred_pidx_thr == idx)
            if not mask.any():
                return -1
            tids = pred_ptid_thr[mask]
            tids = tids[tids >= 0]
            return int(tids[0]) if tids.size > 0 else -1

        def color_fn(v):
            tid = _idx_to_trackid(int(v))
            return (track_id_color(tid) if tid >= 0
                    else "rgba(150,150,150,0.45)")

        def label_fn(v):
            if int(v) < 0:
                return "no_object/unassigned"
            tid = _idx_to_trackid(int(v))
            if tid < 0:
                return f"particle idx {v}"
            return f"particle idx {v} (trackid {tid})"
    elif color_by == "pred_class":
        values = pred_class
        color_fn = class_color
        label_fn = lambda v: _class_symbol(v) if 0 <= int(v) < len(PARTICLE_CLASS_SYMBOLS) else f"cls={v}"
    elif color_by == "pred_query":
        values = pred_q_thr
        color_fn = lambda v: instance_color(int(v) % 64) if v >= 0 else "rgba(150,150,150,0.45)"
        label_fn = lambda v: f"query {v}" if v >= 0 else "no_object/unassigned"
    elif color_by == "particle_class_id_gt":
        values = pcid_gt
        color_fn = class_color
        label_fn = lambda v: (_class_symbol(v) if 0 <= int(v) < len(PARTICLE_CLASS_SYMBOLS)
                              else "no GT" if int(v) < 0 else f"cls={v}")
    elif color_by == "pred_class_correct":
        # Per-SP "is pred_class correct vs particle_class_id_gt?"
        # Only meaningful where particle_class_id_gt >= 0.
        has_gt = pcid_gt >= 0
        correct = (pred_class == pcid_gt) & has_gt
        wrong = (~correct) & has_gt
        no_gt = ~has_gt
        # Encode as integer bucket
        values = np.zeros(coord.shape[0], dtype=np.int64)
        values[correct] = 1; values[wrong] = 0; values[no_gt] = -1
        color_fn = lambda v: ("rgba(60,200,80,1)" if v == 1
                              else "rgba(255,60,60,1)" if v == 0
                              else "rgba(150,150,150,0.45)")
        label_fn = lambda v: ("correct" if v == 1 else "wrong" if v == 0
                              else "no GT")
    elif color_by == "pred_mask_prob":
        # Continuous colorbar — single trace
        prob = pred_mask_prob if pred_mask_prob is not None else np.zeros(coord.shape[0])
        trace = go.Scatter3d(
            x=coord[:, 2], y=coord[:, 0], z=coord[:, 1],
            mode="markers",
            marker=dict(
                size=sp_marker_size, color=prob,
                colorscale="Plasma", cmin=0.0, cmax=1.0,
                colorbar=dict(title="mask prob", thickness=10,
                              len=0.6, x=1.02, y=0.5),
            ),
            customdata=customdata,
            hovertemplate=_STAGE3_HOVER_TEMPLATE,
            name="SPs (mask prob)",
        )
        fig.add_trace(trace)
        if show_origin_diamonds:
            _add_origin_diamonds(fig, pred, no_obj, show_origin_lines)
        fig.update_layout(
            title=f"Stage-3 prediction — color by mask probability{title_suffix}",
            **_default_scene_kwargs(),
        )
        return fig
    elif color_by == "stage2_nu_mask_prob":
        prob = pred.get("stage3/stage2_nu_mask_prob",
                        np.zeros(coord.shape[0])).astype(np.float32)
        trace = go.Scatter3d(
            x=coord[:, 2], y=coord[:, 0], z=coord[:, 1],
            mode="markers",
            marker=dict(
                size=sp_marker_size, color=prob,
                colorscale="Viridis", cmin=0.0, cmax=1.0,
                colorbar=dict(title="stage2 nu prob", thickness=10,
                              len=0.6, x=1.02, y=0.5),
            ),
            customdata=customdata,
            hovertemplate=_STAGE3_HOVER_TEMPLATE,
            name="SPs (stage2 nu prob)",
        )
        fig.add_trace(trace)
        if show_origin_diamonds:
            _add_origin_diamonds(fig, pred, no_obj, show_origin_lines)
        fig.update_layout(
            title=f"Stage-3 prediction — color by Stage-2 nu mask prob{title_suffix}",
            **_default_scene_kwargs(),
        )
        return fig
    elif color_by == "source_mask":
        smask = pred.get("stage3/source_mask",
                         np.zeros(coord.shape[0], dtype=np.uint8)).astype(np.int64)
        labels_by_mask = {
            0: ("floor only", "rgba(160,160,160,0.6)"),
            1: ("[impossible]", "rgba(0,0,0,1)"),
            2: ("GT only (FN)", "rgba(255,210,40,1)"),
            3: ("[impossible]", "rgba(0,0,0,1)"),
            4: ("δ-only", "rgba(120,80,200,0.6)"),
            5: ("stage2 NOT GT (FP)", "rgba(80,200,200,1)"),
            6: ("δ + GT", "rgba(255,140,40,1)"),
            7: ("stage2 ∩ GT (TP)", "rgba(220,40,200,1)"),
        }
        for val in sorted(np.unique(smask).tolist()):
            mask = (smask == val)
            if not mask.any():
                continue
            label, color = labels_by_mask.get(int(val), (str(val), "rgba(0,0,0,1)"))
            pts = coord[mask]
            fig.add_trace(go.Scatter3d(
                x=pts[:, 2], y=pts[:, 0], z=pts[:, 1],
                mode="markers",
                marker=dict(size=sp_marker_size, color=color),
                customdata=customdata[mask],
                hovertemplate=_STAGE3_HOVER_TEMPLATE,
                name=f"sm={val} {label} ({int(mask.sum())})",
            ))
        if show_origin_diamonds:
            _add_origin_diamonds(fig, pred, no_obj, show_origin_lines)
        fig.update_layout(
            title=f"Stage-3 prediction — color by source_mask{title_suffix}",
            **_default_scene_kwargs(),
        )
        return fig
    else:
        raise ValueError(f"unknown color_by {color_by!r}")

    # Categorical: one trace per unique value
    for val in sorted(np.unique(values).tolist()):
        mask = (values == val)
        if not mask.any():
            continue
        pts = coord[mask]
        fig.add_trace(go.Scatter3d(
            x=pts[:, 2], y=pts[:, 0], z=pts[:, 1],
            mode="markers",
            marker=dict(size=sp_marker_size, color=color_fn(int(val))),
            customdata=customdata[mask],
            hovertemplate=_STAGE3_HOVER_TEMPLATE,
            name=f"{label_fn(int(val))} ({int(mask.sum())})",
        ))

    # Low-confidence "demoted" overlay
    if low_conf.any():
        pts = coord[low_conf]
        fig.add_trace(go.Scatter3d(
            x=pts[:, 2], y=pts[:, 0], z=pts[:, 1],
            mode="markers",
            marker=dict(size=sp_marker_size, color="rgba(200,200,200,0.35)",
                        symbol="x"),
            customdata=customdata[low_conf],
            hovertemplate=_STAGE3_HOVER_TEMPLATE,
            name=f"low-conf demoted ({int(low_conf.sum())})",
        ))

    if show_origin_diamonds:
        _add_origin_diamonds(fig, pred, no_obj, show_origin_lines)

    fig.update_layout(
        title=f"Stage-3 prediction — color by {color_by}{title_suffix}",
        **_default_scene_kwargs(),
    )
    return fig


def _add_origin_diamonds(fig: go.Figure, pred: dict,
                         no_object_class_id: int,
                         show_origin_lines: bool) -> None:
    """Overlay per-query predicted-origin markers + (optionally) lines
    from pred-origin → matched GT origin (cm error visualization)."""
    origin_norm = pred.get("stage3_queries/origin_coord_norm", None)
    cls_argmax = pred.get("stage3_queries/class_argmax", None)
    cls_max_prob = pred.get("stage3_queries/class_max_prob", None)
    is_active = pred.get("stage3_queries/is_active", None)
    matched_gt = pred.get("stage3_queries/matched_gt_idx", None)
    gt_origin_cm = pred.get("stage3_gt/origin_cm", None)
    coord_scale = float(pred.get("stage3_meta_coord_scale",
                                 pred.get("stage3_meta/coord_scale", 179.55)))
    if origin_norm is None or cls_argmax is None:
        return
    if is_active is None:
        is_active = (cls_argmax != int(no_object_class_id))
    is_active = is_active.astype(bool)
    if not is_active.any():
        return

    # Reconstruct origin in detector cm using the SPs' (coord, coord_norm)
    # affine — same trick used in process_event.
    coord = pred["stage3/coord"].astype(np.float32)
    coord_norm = pred["stage3/coord_norm"].astype(np.float32)
    if coord.shape[0] > 1:
        scale = (coord.max(0) - coord.min(0)) / np.maximum(
            coord_norm.max(0) - coord_norm.min(0), 1e-9)
        center = coord.mean(0) - coord_norm.mean(0) * scale
        origin_cm = origin_norm.astype(np.float32) * scale + center
    else:
        # Fallback: assume coord_scale applies, no offset
        origin_cm = origin_norm.astype(np.float32) * coord_scale

    # Pull the full per-query class probability matrix so each diamond's
    # hover can show the distribution that drove its predicted class.
    cls_probs_all = pred.get("stage3_queries/class_probs", None)
    n_classes = len(PARTICLE_CLASS_SYMBOLS)

    Q = origin_cm.shape[0]
    for q in range(Q):
        if not is_active[q]:
            continue
        c = int(cls_argmax[q])
        sym = _class_symbol(c)
        clr = class_color(c, alpha=1.0)
        pt = origin_cm[q]
        # Build the per-class line directly into the hovertemplate
        # (origin diamonds aren't customdata-shared, so inline is cleanest).
        if cls_probs_all is not None and cls_probs_all.shape[0] > q:
            probs = cls_probs_all[q]
            probs_line = "  ".join(
                f"{_class_symbol(ci)}={float(probs[ci]):.2f}"
                for ci in range(min(n_classes, probs.size))
            )
        else:
            probs_line = ""
        max_prob = (float(cls_max_prob[q])
                    if cls_max_prob is not None else float("nan"))
        hover_lines = [
            f"<b>query {q}</b> &nbsp; pred {sym} (prob {max_prob:.2f})",
            f"pred origin (cm): x={float(pt[0]):.1f} "
            f"y={float(pt[1]):.1f} z={float(pt[2]):.1f}",
        ]
        # GT origin target line for matched queries (same coords the
        # loss saw at training time: space-charge + drift-time applied).
        if (matched_gt is not None and gt_origin_cm is not None
                and gt_origin_cm.size > 0):
            k = int(matched_gt[q]) if q < len(matched_gt) else -1
            if 0 <= k < gt_origin_cm.shape[0]:
                gpt = gt_origin_cm[k]
                err = float(np.linalg.norm(pt - gpt))
                hover_lines.append(
                    f"GT origin target (cm): x={float(gpt[0]):.1f} "
                    f"y={float(gpt[1]):.1f} z={float(gpt[2]):.1f}  "
                    f"(err {err:.1f}cm)"
                )
        if probs_line:
            hover_lines.append("class probs:")
            hover_lines.append(f"&nbsp;&nbsp;{probs_line}")
        fig.add_trace(go.Scatter3d(
            x=[float(pt[2])], y=[float(pt[0])], z=[float(pt[1])],
            mode="markers",
            marker=dict(size=8, color=clr, symbol="diamond",
                        line=dict(color="black", width=1)),
            name=f"q={q} pred-origin ({sym})",
            hovertemplate="<br>".join(hover_lines) + "<extra></extra>",
            showlegend=False,
        ))

    if show_origin_lines and matched_gt is not None and gt_origin_cm is not None:
        for q in range(Q):
            if not is_active[q]:
                continue
            k = int(matched_gt[q])
            if k < 0:
                continue
            pred_pt = origin_cm[q]
            gt_pt = gt_origin_cm[k]
            err = float(np.linalg.norm(pred_pt - gt_pt))
            fig.add_trace(go.Scatter3d(
                x=[float(pred_pt[2]), float(gt_pt[2])],
                y=[float(pred_pt[0]), float(gt_pt[0])],
                z=[float(pred_pt[1]), float(gt_pt[1])],
                mode="lines",
                line=dict(color="rgba(0,0,0,0.9)", width=2, dash="dot"),
                name=f"q={q} ↔ gt={k} ({err:.1f}cm)",
                hoverinfo="skip",
                showlegend=False,
            ))


def _default_scene_kwargs() -> dict:
    return dict(
        scene=dict(
            xaxis=dict(title="Z (cm)", range=[-10, 1050]),
            yaxis=dict(title="X (cm)", range=[-10, 270]),
            zaxis=dict(title="Y (cm)", range=[-125, 125]),
            aspectmode="data",
            camera=dict(eye=dict(x=1.7, y=1.3, z=1.0),
                        up=dict(x=0, y=1, z=0),
                        center=dict(x=0, y=0, z=0)),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="v", x=1.05, y=1.0, font=dict(size=10)),
        height=720,
    )


# ---------------------------------------------------------------------------
# Full-cascade combined panels (predicted slices + nu-slice particles, and the
# matching GT). Used by tools/visualize_full_cascade.py.
#
# In GT-less inference (the data-pipeline path) `pred_slice_id` /
# `pred_particle_trackid` are all -1 (they require GT matching), so predicted
# slices are colored by `pred_query` and particles by `stage3/pred_query` or
# `stage3/pred_class`.
# ---------------------------------------------------------------------------

def _scatter_xyz(coord):
    """detector (x,y,z) cm -> plotly (z, x, y) order used everywhere here."""
    return coord[:, 2], coord[:, 0], coord[:, 1]


def cascade_nu_color_options() -> list[dict]:
    return [
        {"label": "nu particles by query/instance", "value": "instance"},
        {"label": "nu particles by predicted class", "value": "class"},
    ]


_SLICER_CLASS_NAMES = {0: "nu", 1: "cosmic", 2: "no_obj"}


def figure_for_cascade_slices(
    pred: Optional[dict],
    *,
    min_mask_prob: float = 0.0,
    nu_class_id: int = 0,
    slicer_no_object_id: int = 2,
    sp_marker_size: float = 1.8,
    title_suffix: str = "",
) -> go.Figure:
    """STAGE-2-ONLY panel: every post-deghost SP colored by the slicer's
    predicted query (= slice). The slice whose dominant predicted class is
    `nu` is drawn red so it's obvious which slice Stage 3 processed. Carries
    no Stage-3 particle layer.

    `pred` is a stage3pred_*.h5 dict from `inference.load_event_h5`.
    """
    fig = go.Figure(data=make_detector_outline_trace())
    if pred is None or pred.get("post/coord") is None or pred["post/coord"].shape[0] == 0:
        fig.update_layout(title=f"No slicer prediction{title_suffix}",
                          **_default_scene_kwargs())
        return fig

    post_coord = pred["post/coord"].astype(np.float32)
    post_q = pred.get("post/pred_query",
                      -np.ones(post_coord.shape[0], np.int64)).astype(np.int64)
    post_cls = pred.get("post/pred_class",
                        np.full(post_coord.shape[0], slicer_no_object_id)).astype(np.int64)
    post_mp = pred.get("post/pred_mask_prob", None)

    keep = np.ones(post_coord.shape[0], dtype=bool)
    if min_mask_prob > 0.0 and post_mp is not None:
        keep = post_mp >= float(min_mask_prob)

    for qv in sorted(np.unique(post_q[keep]).tolist()):
        m = keep & (post_q == qv)
        if not m.any():
            continue
        ch = post_cls[m]
        dom = int(np.bincount(ch[ch >= 0]).argmax()) if (ch >= 0).any() else -1
        is_nu = dom == int(nu_class_id)
        color = ("rgba(255,60,60,1)" if is_nu
                 else instance_color(int(qv)) if qv >= 0
                 else "rgba(150,150,150,0.35)")
        x, y, z = _scatter_xyz(post_coord[m])
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z, mode="markers",
            marker=dict(size=sp_marker_size, color=color),
            name=f"slice q{qv} [{_SLICER_CLASS_NAMES.get(dom, '?')}] ({int(m.sum())})",
            hovertemplate=(f"slice query={qv} class={_SLICER_CLASS_NAMES.get(dom, '?')}"
                           f"<br>x=%{{y:.1f}} y=%{{z:.1f}} z=%{{x:.1f}}<extra></extra>"),
        ))

    fig.update_layout(
        title=f"STAGE 2 (slicer) — SPs by predicted slice; nu slice in red{title_suffix}",
        **_default_scene_kwargs())
    return fig


def figure_for_cascade_prediction(
    pred: Optional[dict],
    *,
    min_mask_prob: float = 0.0,
    nu_class_id: int = 0,
    slicer_no_object_id: int = 2,
    nu_color_by: str = "instance",
    show_origin_diamonds: bool = True,
    sp_marker_size: float = 1.6,
    nu_marker_size: float = 3.0,
    title_suffix: str = "",
) -> go.Figure:
    """Prediction panel: the slicer's predicted slices (post-deghost SPs
    colored by `pred_query`) with the nu-candidate slice rendered as the
    Stage-3 particle segmentation (`stage3` SPs colored by particle
    query/class).

    `pred` is a stage3pred_*.h5 dict from `inference.load_event_h5`.
    """
    fig = go.Figure(data=make_detector_outline_trace())
    if pred is None or pred.get("post/coord") is None or pred["post/coord"].shape[0] == 0:
        fig.update_layout(title=f"No cascade prediction{title_suffix}",
                          **_default_scene_kwargs())
        return fig

    post_coord = pred["post/coord"].astype(np.float32)
    post_q = pred.get("post/pred_query",
                      -np.ones(post_coord.shape[0], np.int64)).astype(np.int64)
    post_cls = pred.get("post/pred_class",
                        np.full(post_coord.shape[0], slicer_no_object_id)).astype(np.int64)
    post_mp = pred.get("post/pred_mask_prob", None)

    # ---- Slice layer: NON-nu post SPs, colored by predicted query (= slice).
    non_nu = post_cls != int(nu_class_id)
    if min_mask_prob > 0.0 and post_mp is not None:
        non_nu = non_nu & (post_mp >= float(min_mask_prob))
    q_vals = post_q[non_nu]
    coord_nn = post_coord[non_nu]
    for qv in sorted(np.unique(q_vals).tolist()):
        m = q_vals == qv
        x, y, z = _scatter_xyz(coord_nn[m])
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z, mode="markers",
            marker=dict(size=sp_marker_size,
                        color=(instance_color(int(qv)) if qv >= 0
                               else "rgba(150,150,150,0.35)")),
            name=f"slice q{qv} ({int(m.sum())})",
            hovertemplate=(f"slice query={qv}<br>x=%{{y:.1f}} y=%{{z:.1f}} "
                           f"z=%{{x:.1f}}<extra></extra>"),
        ))

    # ---- Particle layer: the nu-candidate slice = Stage-3 SPs ----------
    s3_coord = pred.get("stage3/coord", None)
    if s3_coord is not None and s3_coord.shape[0] > 0:
        s3_coord = s3_coord.astype(np.float32)
        no_obj3 = int(pred.get("stage3_meta_no_object_class_id",
                               pred.get("stage3_meta/no_object_class_id", 7)))
        s3_cls = pred.get("stage3/pred_class",
                          np.full(s3_coord.shape[0], no_obj3)).astype(np.int64)
        s3_q = pred.get("stage3/pred_query",
                        -np.ones(s3_coord.shape[0], np.int64)).astype(np.int64)
        s3_mp = pred.get("stage3/pred_mask_prob", None)
        s3_cls_t, s3_q_t, low = apply_pred_threshold(
            s3_mp, s3_cls, s3_q, no_obj3, min_mask_prob)
        customdata = _stage3_customdata(pred)
        if nu_color_by == "class":
            values = s3_cls_t
            color_fn = class_color
            label_fn = lambda v: f"nu {_class_symbol(int(v))}"
        else:  # instance
            values = s3_q_t
            color_fn = lambda v: (instance_color(int(v) % 64) if v >= 0
                                  else "rgba(170,170,170,0.4)")

            def label_fn(v):
                if int(v) < 0:
                    return "nu unassigned"
                # which class dominates this query?
                cm = s3_q_t == int(v)
                cls_here = s3_cls_t[cm]
                dom = int(np.bincount(cls_here[cls_here >= 0]).argmax()) \
                    if (cls_here >= 0).any() else -1
                return f"nu q{int(v)} ({_class_symbol(dom)})"
        for v in sorted(np.unique(values).tolist()):
            m = values == v
            x, y, z = _scatter_xyz(s3_coord[m])
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z, mode="markers",
                marker=dict(size=nu_marker_size, color=color_fn(int(v))),
                customdata=customdata[m],
                hovertemplate=_STAGE3_HOVER_TEMPLATE,
                name=f"{label_fn(int(v))} ({int(m.sum())})",
            ))
        if show_origin_diamonds:
            _add_origin_diamonds(fig, pred, no_obj3, show_origin_lines=False)

    fig.update_layout(
        title=f"PREDICTION — slices (by query) + nu particles{title_suffix}",
        **_default_scene_kwargs())
    return fig


def figure_for_cascade_gt(
    gt: Optional[dict],
    *,
    nu_color_by: str = "instance",
    show_ghosts: bool = False,
    max_ghosts: int = 50000,
    sp_marker_size: float = 1.6,
    nu_marker_size: float = 3.0,
    title_suffix: str = "",
) -> go.Figure:
    """GT panel: truth slices (cosmic primaries + nu) with the nu slice
    rendered as truth particles.

    `gt` is a plain dict built by the tool from `LArFormerDataset`:
        coord            (N,3)  detector cm
        slice_id         (N,)   GT slice id (cosmic primary trackid / nu)
        slice_is_nu      (N,)   bool — SP belongs to the nu slice
        slice_origin     {slice_id: origin_type}  (0 = nu)
        hasmatch         (N,)   1 = real, 0 = ghost (optional)
        nu_coord         (Nn,3) nu-slice SP coords
        nu_class         (Nn,)  per-SP particle class id
        nu_instance      (Nn,)  per-SP particle instance id

    Ghosts (hasmatch==0, or slice_id<0 when hasmatch is absent) are hidden
    by default. When `show_ghosts=True` a random sample of at most
    `max_ghosts` ghost points is drawn (gray).
    """
    fig = go.Figure(data=make_detector_outline_trace())
    if gt is None or gt.get("coord") is None or len(gt["coord"]) == 0:
        fig.update_layout(title=f"No GT (no MC truth){title_suffix}",
                          **_default_scene_kwargs())
        return fig

    coord = np.asarray(gt["coord"], np.float32)
    slice_id = np.asarray(gt["slice_id"], np.int64)
    is_nu = np.asarray(gt.get("slice_is_nu",
                              np.zeros(len(coord), bool))).astype(bool)
    slice_origin = gt.get("slice_origin", {})

    # ---- cosmic slices (non-nu) by slice id ----------------------------
    cosmic = (~is_nu) & (slice_id >= 0)
    for tid in sorted(np.unique(slice_id[cosmic]).tolist()):
        m = (slice_id == tid) & cosmic
        if not m.any():
            continue
        ot = int(slice_origin.get(int(tid), 1))
        x, y, z = _scatter_xyz(coord[m])
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z, mode="markers",
            marker=dict(size=sp_marker_size, color=track_id_color(int(tid), ot)),
            name=f"GT slice tid={tid} ({int(m.sum())})",
            hovertemplate=(f"GT slice tid={tid}<br>x=%{{y:.1f}} y=%{{z:.1f}} "
                           f"z=%{{x:.1f}}<extra></extra>"),
        ))
    # ghost / unassigned — hidden by default; sampled when shown.
    hasmatch = gt.get("hasmatch", None)
    if hasmatch is not None:
        ghost = np.asarray(hasmatch, np.int64) == 0
    else:
        ghost = slice_id < 0
    if show_ghosts and ghost.any():
        ghost_idx = np.flatnonzero(ghost)
        n_ghost = ghost_idx.size
        if max_ghosts and n_ghost > int(max_ghosts):
            # deterministic sample so re-renders are stable.
            rng = np.random.default_rng(0)
            ghost_idx = np.sort(rng.choice(ghost_idx, int(max_ghosts),
                                           replace=False))
        pts = coord[ghost_idx]
        x, y, z = _scatter_xyz(pts)
        shown = ghost_idx.size
        label = (f"ghosts ({shown} of {n_ghost} sampled)"
                 if shown < n_ghost else f"ghosts ({n_ghost})")
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z, mode="markers",
            marker=dict(size=sp_marker_size * 0.8, color="rgba(150,150,150,0.3)"),
            name=label, hoverinfo="skip"))

    # ---- nu slice as truth particles -----------------------------------
    nu_coord = gt.get("nu_coord", None)
    if nu_coord is not None and len(nu_coord) > 0:
        nu_coord = np.asarray(nu_coord, np.float32)
        nu_cls = np.asarray(gt.get("nu_class",
                                   -np.ones(len(nu_coord), np.int64)), np.int64)
        nu_inst = np.asarray(gt.get("nu_instance",
                                    -np.ones(len(nu_coord), np.int64)), np.int64)
        if nu_color_by == "class":
            values, color_fn = nu_cls, class_color
            label_fn = lambda v: f"GT nu {_class_symbol(int(v))}"
        else:
            values = nu_inst
            color_fn = lambda v: (instance_color(int(v) % 64) if v >= 0
                                  else "rgba(170,170,170,0.4)")
            def label_fn(v):
                cm = nu_inst == int(v)
                ch = nu_cls[cm]
                dom = int(np.bincount(ch[ch >= 0]).argmax()) if (ch >= 0).any() else -1
                return f"GT nu inst{int(v)} ({_class_symbol(dom)})"
        for v in sorted(np.unique(values).tolist()):
            m = values == v
            x, y, z = _scatter_xyz(nu_coord[m])
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z, mode="markers",
                marker=dict(size=nu_marker_size, color=color_fn(int(v))),
                name=f"{label_fn(int(v))} ({int(m.sum())})",
                hovertemplate=(f"GT nu class=%{{text}}<br>x=%{{y:.1f}} "
                               f"y=%{{z:.1f}} z=%{{x:.1f}}<extra></extra>"),
                text=[_class_symbol(int(c)) for c in nu_cls[m]],
            ))

    fig.update_layout(
        title=f"GROUND TRUTH — slices + nu particles{title_suffix}",
        **_default_scene_kwargs())
    return fig


# ---------------------------------------------------------------------------
# Stage-3 text-panel metadata
# ---------------------------------------------------------------------------

def stage3_metadata_summary(pred: Optional[dict]) -> list[str]:
    """Plain-text summary rows for the Stage-3 prediction text panel.
    Each entry is one short line — the caller wraps them in Dash html
    elements."""
    if pred is None:
        return ["No Stage-3 prediction file matched this event."]

    def _attr(k, default="?"):
        return pred.get(k.replace("/", "_"), pred.get(k, default))

    n_sp = int(_attr("stage3_meta/n_stage3_sp", 0))
    n_q = int(_attr("stage3_meta/n_queries", 0))
    n_gt = int(_attr("stage3_meta/n_gt_instances", 0))
    n_matched = int(_attr("stage3_meta/n_matched", 0))
    no_obj = int(_attr("stage3_meta/no_object_class_id", 7))
    has_gt = bool(int(_attr("stage3_meta/has_gt", 0)))
    cpt = float(_attr("stage3_meta/class_prob_threshold", 0.0))
    cs = float(_attr("stage3_meta/coord_scale", 179.55))

    lines = [
        f"n_stage3_sp={n_sp}  n_queries={n_q}  no_object={no_obj}  "
        f"class_prob_threshold={cpt}  coord_scale={cs}",
    ]
    if has_gt:
        pair_iou = pred.get("stage3_gt/pair_iou", np.zeros(0))
        valid = pair_iou[pair_iou >= 0] if pair_iou.size else pair_iou
        mean_iou = float(valid.mean()) if valid.size > 0 else float("nan")
        med_iou = float(np.median(valid)) if valid.size > 0 else float("nan")
        l2 = pred.get("stage3_gt/pair_origin_l2_cm", np.zeros(0))
        l2_v = l2[l2 >= 0] if l2.size else l2
        mean_l2 = float(l2_v.mean()) if l2_v.size > 0 else float("nan")
        cls_ok = pred.get("stage3_gt/pair_cls_correct", np.zeros(0, dtype=np.int8))
        cls_ok = cls_ok[cls_ok >= 0] if cls_ok.size else cls_ok
        cls_acc = float(cls_ok.mean()) if cls_ok.size > 0 else float("nan")
        lines.append(
            f"GT: n_gt={n_gt}  matched={n_matched}/{n_gt}  "
            f"pair_IoU mean={mean_iou:.3f} med={med_iou:.3f}  "
            f"origin_L2_cm mean={mean_l2:.1f}  "
            f"cls_acc={cls_acc:.3f}"
        )
        # Per-GT summary table (capped at 16 rows for readability)
        pid_gt = pred.get("stage3_gt/pid", np.zeros(n_gt, dtype=np.int64))
        cls_id_gt = pred.get("stage3_gt/class_id", np.zeros(n_gt, dtype=np.int64))
        ntp_gt = pred.get("stage3_gt/n_truth_points", np.zeros(n_gt, dtype=np.int64))
        matched_q = pred.get("stage3_gt/matched_query", np.full(n_gt, -1, dtype=np.int64))
        for k in range(min(n_gt, 16)):
            iou_k = float(pair_iou[k]) if k < pair_iou.size else float("nan")
            l2_k = float(l2[k]) if k < l2.size else float("nan")
            cls_k = int(cls_id_gt[k]) if k < cls_id_gt.size else -1
            cls_name = PARTICLE_CLASS_NAMES[cls_k] if 0 <= cls_k < len(PARTICLE_CLASS_NAMES) else "?"
            lines.append(
                f"  GT[{k}] pid={int(pid_gt[k])} class={cls_k}({cls_name}) "
                f"n_truth={int(ntp_gt[k])}  matched_q={int(matched_q[k])} "
                f"IoU={iou_k:.3f}  L2={l2_k:.1f}cm"
            )
        if n_gt > 16:
            lines.append(f"  … ({n_gt - 16} more GT instances)")
    else:
        lines.append("No GT — pure-prediction event")

    return lines
