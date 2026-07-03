"""Visualize attempt-2 keypoint-cascade output (CascadedKeypoint).

Reads one per-event H5 from run_larformer_keypoint2_cascade_inference.py and
writes an interactive Plotly HTML:

  LEFT 3D plot  — predicted particles + their predicted keypoints (start/end) +
                  the predicted nu vertex. A dropdown selects "All particles"
                  or an individual particle; when one is selected, the OTHER
                  particles' spacepoints are drawn small + grey for context and
                  the chosen particle is highlighted in colour.
  RIGHT 3D plot — (only if the file carries GT) the GT particle MATCHED to the
                  highlighted prediction (by IoU) + its GT keypoints. If the
                  selected prediction has no match, the right plot is empty.

  python tools/visualize_keypoint2_cascade.py <event.h5> [-o out.html]
  python tools/visualize_keypoint2_cascade.py <dir> --index 0    # pick from a dir
"""
import argparse
import glob
import os

import numpy as np
import h5py
import plotly.graph_objects as go
import plotly.colors as pc
from plotly.subplots import make_subplots

CLASS_NAMES = ["e", "gamma", "mu", "pi", "p", "other", "(unused)", "no_object"]
PALETTE = pc.qualitative.Dark24


def _cls_name(c):
    return CLASS_NAMES[c] if 0 <= c < len(CLASS_NAMES) else str(c)


def load_event(path):
    with h5py.File(path, "r") as f:
        slice_coord = f["slice/coord_cm"][:]
        nu = f["nu_vertex_cm"][:]
        gt_nu = f["gt_nu_vertex_cm"][:]
        parts = []
        if "particle" in f:
            for key in sorted(f["particle"], key=int):
                g = f["particle"][key]
                parts.append(dict(
                    point_idx=g["point_idx"][:],
                    cls=int(g.attrs["cls"]),
                    start=g["start_cm"][:], end=g["end_cm"][:],
                    has_match=bool(g.attrs["has_match"]),
                    iou=float(g.attrs["iou"]),
                    gt_trackid=int(g.attrs["gt_trackid"]),
                    gt_point_idx=g["gt_point_idx"][:],
                    gt_start=g["gt_start_cm"][:], gt_end=g["gt_end_cm"][:]))
        has_gt = bool(f.attrs.get("has_gt", False))
        meta = {k: f.attrs.get(k) for k in ("src_file", "run", "subrun", "event")}
        # Optional dense-head score maps + raw GT keypoints (--save-score-maps).
        # Stashed in `meta` so the 6-tuple signature (and all its unpackers)
        # stays unchanged; absent in older files -> empty dicts.
        smaps = {}
        if "score_maps" in f:
            for name in f["score_maps"]:
                g = f["score_maps"][name]
                smaps[name] = dict(
                    coords_cm=g["coords_cm"][:], score=g["score"][:],
                    level=g.attrs.get("level", ""),
                    kp_types=np.asarray(g.attrs.get("kp_types", []), int))
        meta["score_maps"] = smaps
        gk = {}
        if "gt_keypoints" in f:
            gg = f["gt_keypoints"]
            gk = dict(pos_cm=gg["pos_cm"][:], type=gg["type"][:],
                      trackid=gg["trackid"][:])
        meta["gt_keypoints"] = gk
    return slice_coord, nu, gt_nu, parts, has_gt, meta


def _pts(xyz, **kw):
    return go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
                        mode="markers", **kw)


def _kp(p, name, symbol, color, size=7):
    if p is None or not np.all(np.isfinite(p)):
        # invisible placeholder so the trace index stays stable for the dropdown
        return go.Scatter3d(x=[], y=[], z=[], mode="markers", name=name,
                            marker=dict(size=size, symbol=symbol, color=color),
                            showlegend=False)
    return go.Scatter3d(x=[p[0]], y=[p[1]], z=[p[2]], mode="markers", name=name,
                        marker=dict(size=size, symbol=symbol, color=color,
                                    line=dict(width=1, color="black")),
                        showlegend=False, hovertext=name, hoverinfo="text")


def _assemble_traces(slice_coord, nu, gt_nu, parts, has_gt):
    """Add every trace (left predicted + right matched-GT) once. Returns
    (fig, meta_tr) where meta_tr[k] = (kind, particle_idx|None) parallels
    fig.data for the visibility logic. No layout/menus — callers set those."""
    ncol = 2 if has_gt else 1
    titles = (["Predicted", "Matched GT"] if has_gt else ["Predicted"])
    specs = [[{"type": "scene"}] * ncol]
    fig = make_subplots(rows=1, cols=ncol, specs=specs, subplot_titles=titles,
                        horizontal_spacing=0.02)
    meta_tr = []

    def add(tr, col, kind, pidx):
        fig.add_trace(tr, row=1, col=col)
        meta_tr.append((kind, pidx))

    # ---- LEFT: predicted ----
    # context backdrop = ALL predicted-particle points, grey + small.
    ctx_idx = np.unique(np.concatenate([p["point_idx"] for p in parts])
                        if parts else np.zeros(0, int))
    ctx = slice_coord[ctx_idx] if ctx_idx.size else slice_coord[:0]
    add(_pts(ctx, marker=dict(size=1.4, color="lightgrey"),
             name="context", showlegend=False, hoverinfo="skip"),
        1, "pred_ctx", None)
    for i, p in enumerate(parts):
        col = PALETTE[i % len(PALETTE)]
        xyz = slice_coord[p["point_idx"]]
        add(_pts(xyz, marker=dict(size=2.5, color=col),
                 name=f"P{i} {_cls_name(p['cls'])}",
                 hovertext=f"P{i} {_cls_name(p['cls'])}", hoverinfo="text"),
            1, "pred_pts", i)
        add(_kp(p["start"], f"P{i} start", "diamond", col), 1, "pred_start", i)
        add(_kp(p["end"], f"P{i} end", "x", col), 1, "pred_end", i)
    add(_kp(nu, "nu vertex (pred)", "diamond-open", "magenta", size=10),
        1, "nu_pred", None)

    # ---- RIGHT: matched GT ----
    if has_gt:
        for i, p in enumerate(parts):
            col = PALETTE[i % len(PALETTE)]
            if p["has_match"] and p["gt_point_idx"].size:
                xyz = slice_coord[p["gt_point_idx"]]
                nm = f"GT P{i} trk{p['gt_trackid']} IoU{p['iou']:.2f}"
            else:
                xyz = slice_coord[:0]
                nm = f"GT P{i} (no match)"
            add(_pts(xyz, marker=dict(size=2.5, color=col), name=nm,
                     hovertext=nm, hoverinfo="text"), 2, "gt_pts", i)
            add(_kp(p["gt_start"] if p["has_match"] else None,
                    f"GT P{i} start", "diamond", col), 2, "gt_start", i)
            add(_kp(p["gt_end"] if p["has_match"] else None,
                    f"GT P{i} end", "x", col), 2, "gt_end", i)
        add(_kp(gt_nu, "nu vertex (GT)", "diamond-open", "magenta", size=10),
            2, "gt_nu", None)

    return fig, meta_tr


def _visibility(meta_tr, view):
    """Trace visibility for a view: -1 = all particles; i = highlight particle i
    (context backdrop + nu vertices stay on)."""
    out = []
    for kind, pidx in meta_tr:
        if kind == "pred_ctx":
            v = (view >= 0)                  # grey backdrop only in single view
        elif kind in ("nu_pred", "gt_nu"):
            v = True
        elif view < 0:                       # "all"
            v = True
        else:                                # single particle `view`
            v = (pidx == view)
        out.append(v)
    return out


_CAMERA = dict(eye=dict(x=1.5, y=1.5, z=1.2))


def _ranges(slice_coord, parts, nu, gt_nu):
    """Common x/y/z bounds over the slice points + all finite keypoints, so the
    Predicted and Matched-GT scenes share ONE coordinate box and identical
    points land at identical screen positions (the two scenes otherwise
    auto-range independently → apparent offset)."""
    pts = [np.asarray(slice_coord, np.float32).reshape(-1, 3)]
    for p in parts:
        for k in ("start", "end", "gt_start", "gt_end"):
            v = np.asarray(p[k], np.float32)
            if np.all(np.isfinite(v)):
                pts.append(v.reshape(1, 3))
    for v in (nu, gt_nu):
        v = np.asarray(v, np.float32)
        if np.all(np.isfinite(v)):
            pts.append(v.reshape(1, 3))
    allp = np.concatenate(pts, 0)
    lo, hi = allp.min(0), allp.max(0)
    pad = 0.05 * (hi - lo + 1.0)
    return lo - pad, hi + pad


def _scene(rng=None):
    s = dict(aspectmode="data",
             xaxis_title="x (cm)", yaxis_title="y (cm)", zaxis_title="z (cm)",
             camera=_CAMERA)
    if rng is not None:
        lo, hi = rng
        s["xaxis"] = dict(title="x (cm)", range=[float(lo[0]), float(hi[0])])
        s["yaxis"] = dict(title="y (cm)", range=[float(lo[1]), float(hi[1])])
        s["zaxis"] = dict(title="z (cm)", range=[float(lo[2]), float(hi[2])])
    return s


def _title(meta, n_parts):
    return (f"keypoint2 cascade — run {meta.get('run')} subrun "
            f"{meta.get('subrun')} event {meta.get('event')}  "
            f"[{meta.get('src_file')}]  ({n_parts} predicted particles)")


def build_figure(slice_coord, nu, gt_nu, parts, has_gt, meta):
    """Standalone figure with a built-in Plotly particle dropdown (static HTML)."""
    fig, meta_tr = _assemble_traces(slice_coord, nu, gt_nu, parts, has_gt)
    buttons = [dict(label="All particles", method="update",
                    args=[{"visible": _visibility(meta_tr, -1)}])]
    for i, p in enumerate(parts):
        tag = "" if p["has_match"] else "  (no GT match)"
        buttons.append(dict(label=f"P{i}: {_cls_name(p['cls'])}{tag}",
                            method="update",
                            args=[{"visible": _visibility(meta_tr, i)}]))
    for tr, v in zip(fig.data, _visibility(meta_tr, -1)):
        tr.visible = v
    rng = _ranges(slice_coord, parts, nu, gt_nu)
    fig.update_layout(
        title=_title(meta, len(parts)), height=820,
        scene=_scene(rng), scene2=_scene(rng) if has_gt else None,
        updatemenus=[dict(buttons=buttons, direction="down", showactive=True,
                          x=0.0, y=1.08, xanchor="left", yanchor="top")])
    return fig


def figure_for_view(event_data, view=-1):
    """Figure for a specific view (no built-in dropdown) — for the Dash app,
    which drives the particle selection with its own control."""
    slice_coord, nu, gt_nu, parts, has_gt, meta = event_data
    fig, meta_tr = _assemble_traces(slice_coord, nu, gt_nu, parts, has_gt)
    for tr, v in zip(fig.data, _visibility(meta_tr, view)):
        tr.visible = v
    rng = _ranges(slice_coord, parts, nu, gt_nu)
    fig.update_layout(title=_title(meta, len(parts)), height=820,
                      scene=_scene(rng), scene2=_scene(rng) if has_gt else None,
                      uirevision="keep")  # preserve camera across callbacks
    return fig


# mckeypoint type codes (see configs/lartpc/larformer/stage4_keypoint/larformer-keypoint2-slice-v1.py).
KP_TYPE_NAMES = {0: "nu_vertex", 1: "track_start", 2: "track_end",
                 3: "shower", 4: "michel", 5: "delta"}


def figure_score_map(event_data, head_name, score_thresh=0.0):
    """Single-scene 3D diagnostic: a dense head's per-token sigmoid score
    (colour = score, Hot scale) with the GT keypoints it targets overlaid as
    large markers. Lets you see whether the head fires AT the GT keypoints.

    Score maps live in meta['score_maps'] (written by the inference tool's
    --save-score-maps). `score_thresh` hides low-score tokens to declutter."""
    slice_coord, nu, gt_nu, parts, has_gt, meta = event_data
    sm = (meta.get("score_maps") or {}).get(head_name)
    fig = go.Figure()
    if sm is None:
        fig.update_layout(
            title=f"no score map '{head_name}' in this file "
                  "(rerun inference with --save-score-maps)",
            scene=_scene(), height=820, uirevision="keep")
        return fig

    # faint slice backdrop for spatial context
    fig.add_trace(_pts(np.asarray(slice_coord, np.float32).reshape(-1, 3),
                       marker=dict(size=1.2, color="lightgrey"),
                       name="slice", showlegend=False, hoverinfo="skip"))

    coords = np.asarray(sm["coords_cm"], np.float32).reshape(-1, 3)
    score = np.asarray(sm["score"], np.float32).reshape(-1)
    keep = score >= float(score_thresh)
    coords, score = coords[keep], score[keep]
    fig.add_trace(go.Scatter3d(
        x=coords[:, 0], y=coords[:, 1], z=coords[:, 2], mode="markers",
        name=f"{head_name} score",
        marker=dict(size=3.0, color=score, colorscale="Hot", cmin=0.0,
                    cmax=1.0, showscale=True,
                    colorbar=dict(title="score", x=1.0)),
        hovertext=[f"{head_name}={s:.3f}" for s in score], hoverinfo="text"))

    # GT keypoints this head targets (filtered by its kp_types), if present.
    gk = meta.get("gt_keypoints") or {}
    pos = np.asarray(gk.get("pos_cm", np.zeros((0, 3))), np.float32).reshape(-1, 3)
    typ = np.asarray(gk.get("type", np.zeros(0)), int).reshape(-1)
    want = set(int(t) for t in np.asarray(sm.get("kp_types", []), int).reshape(-1))
    if pos.shape[0] and want:
        sel = np.array([int(t) in want for t in typ], bool)
        gp, gt_ = pos[sel], typ[sel]
        fig.add_trace(go.Scatter3d(
            x=gp[:, 0], y=gp[:, 1], z=gp[:, 2], mode="markers",
            name="GT keypoints",
            marker=dict(size=8, symbol="diamond", color="cyan",
                        line=dict(width=2, color="black")),
            hovertext=[f"GT {KP_TYPE_NAMES.get(int(t), t)}" for t in gt_],
            hoverinfo="text"))

    n_gt = int(sel.sum()) if (pos.shape[0] and want) else 0
    fig.update_layout(
        title=(f"{head_name} score map @ {sm.get('level','')} — "
               f"{coords.shape[0]} tokens (thresh {score_thresh:g}), "
               f"{n_gt} GT keypoints "
               f"[types {sorted(want)}]  |  {_title(meta, len(parts))}"),
        scene=_scene(), height=820, uirevision="keep")
    return fig


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("path", help="event H5, or a directory of them")
    ap.add_argument("--index", type=int, default=0,
                    help="if `path` is a directory, which event (sorted) to show")
    ap.add_argument("-o", "--output", default=None, help="output HTML path")
    args = ap.parse_args()

    if os.path.isdir(args.path):
        files = sorted(glob.glob(os.path.join(args.path, "*.h5")))
        if not files:
            raise SystemExit(f"no .h5 in {args.path}")
        h5 = files[args.index]
    else:
        h5 = args.path

    fig = build_figure(*load_event(h5))
    out = args.output or os.path.splitext(h5)[0] + ".html"
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
