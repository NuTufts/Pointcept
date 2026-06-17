"""Full-cascade visualizer — predicted slices + nu-slice particles (top row)
and the matching ground truth (bottom row), camera-synced.

Reads the `stage3pred_*.h5` files written by
`tools/run_larformer_stage3_inference.py --input-mode full-cascade`, OR the
`keypointpred_*.h5` files from `tools/run_larformer_fullcascade_inference.py`
(a superset that adds a `keypoints/` group). When present, the predicted
keypoints (per-particle start/end + nu vertex) are overlaid on the PREDICTION
scene — toggle with "show keypoints". Sibling of
`tools/visualize_stage3_larformer_from_cached.py` and the slicer overlay in
`tools/visualize_larformer_gt.py`; it shares their color conventions +
camera-sync pattern via `pointcept/models/LArFormer/viz_inference.py`.

Two rows of 3D scatter:

  - PREDICTION (always): the slicer's predicted slices (post-deghost SPs
    colored by predicted query) with the nu-candidate slice rendered as the
    Stage-3 particle segmentation (colored by particle query or class).

  - GROUND TRUTH (only when MC truth is available): truth slices (cosmic
    primaries + nu) with the nu slice rendered as truth particles. Because
    the full-cascade inference runs GT-less, the truth is recomputed from
    the source merged_h5 via `LArFormerDataset` (the exact label code the
    model/loss use). Provide `--merged-dir`; events whose merged_h5 lacks a
    `mc_particle_tree` (real data) just show an empty GT panel.

Usage:

    python tools/visualize_full_cascade.py \\
        --stage3pred-dir <dir of stage3pred_*.h5> \\
        --merged-dir     <dir of merged_*.h5 (for GT; optional)>

Open http://<host>:<port> in a browser.
"""
import argparse
import atexit
import glob
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
from dash import Dash, Input, Output, State, Patch, dcc, html, no_update  # noqa: E402

from pointcept.models.LArFormer.inference import load_event_h5  # noqa: E402
from pointcept.models.LArFormer.viz_inference import (  # noqa: E402
    figure_for_cascade_prediction,
    figure_for_cascade_gt,
    cascade_nu_color_options,
)

# Predicted-keypoint overlay style per keypoint type (det-cm scene).
# (name, plotly-3d marker symbol, size, color). Symbols limited to the
# 3d-supported set: circle/diamond/square/cross/x (+ -open variants).
_KP_STYLE = {
    0: ("nu_vertex",   "diamond",      9, "red"),
    1: ("track_start", "circle",       5, "limegreen"),
    2: ("track_end",   "x",            5, "orange"),
    3: ("shower",      "diamond-open", 7, "deepskyblue"),
    4: ("michel",      "square",       5, "magenta"),
    5: ("delta",       "cross",        5, "gold"),
}


def _add_predicted_keypoints(fig, pred):
    """Overlay the predicted keypoints (from a keypointpred_*.h5's
    `keypoints/` group, written by run_larformer_fullcascade_inference) onto
    the prediction scene. Positions are detector cm — same frame as
    `stage3/coord` — so they drop straight in. One legend trace per type
    (query start/end + shower; dense vertex/michel/delta), plus a big marker
    for the chosen nu vertex. No-op if the file has no keypoints."""
    pos = pred.get("keypoints/pos_cm")
    typ = pred.get("keypoints/type")
    kind = pred.get("keypoints/kind")
    cls = pred.get("keypoints/class_id")
    score = pred.get("keypoints/score")
    source = pred.get("keypoints/source")
    n_added = 0
    if pos is not None and np.asarray(pos).size > 0 and typ is not None:
        pos = np.asarray(pos, np.float32).reshape(-1, 3)
        typ = np.asarray(typ).reshape(-1)
        for t, (name, sym, size, color) in _KP_STYLE.items():
            m = typ == t
            if not np.any(m):
                continue
            p = pos[m]
            hover = []
            for i in np.where(m)[0]:
                parts = [name]
                if kind is not None:
                    k = int(kind[i])
                    parts.append("start" if k == 0 else "end" if k == 1 else "-")
                if cls is not None and int(cls[i]) >= 0:
                    parts.append(f"cls={int(cls[i])}")
                if score is not None:
                    parts.append(f"s={float(score[i]):.2f}")
                if source is not None:
                    parts.append("dense" if int(source[i]) == 1 else "query")
                hover.append("  ".join(parts))
            fig.add_trace(go.Scatter3d(
                x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="markers",
                marker=dict(symbol=sym, size=size, color=color,
                            line=dict(width=1, color="black")),
                name=f"kp:{name}", text=hover, hoverinfo="text"))
            n_added += int(p.shape[0])

    nv = pred.get("keypoints/nu_vertex_cm")
    if nv is not None and np.asarray(nv).size == 3:
        nv = np.asarray(nv, np.float32).reshape(3)
        sc = pred.get("keypoints_nu_vertex_score")
        lbl = ("nu vertex (best)"
               + (f"  s={float(sc):.2f}" if sc is not None else ""))
        fig.add_trace(go.Scatter3d(
            x=[nv[0]], y=[nv[1]], z=[nv[2]], mode="markers",
            marker=dict(symbol="diamond", size=14, color="red",
                        line=dict(width=2, color="black")),
            name="nu vertex (best)", text=[lbl], hoverinfo="text"))
        n_added += 1
    return n_added


def _merged_base_from_pred(pred_path: str) -> str:
    """{stage3pred,keypointpred}_<base>.h5 -> <base>  (merged_h5 basename)."""
    name = os.path.basename(pred_path)
    for pfx in ("stage3pred_", "keypointpred_"):
        if name.startswith(pfx):
            name = name[len(pfx):]
            break
    if name.endswith(".h5"):
        name = name[:-3]
    return name


def _compute_gt(merged_path: str) -> "dict | None":
    """Build the GT dict for `figure_for_cascade_gt` from a merged_h5 by
    reading it twice through LArFormerDataset (slice + particle GT). Returns
    None if the file has no MC truth (no mc_particle_tree)."""
    import h5py
    from pointcept.datasets.larformer import LArFormerDataset

    with h5py.File(merged_path, "r") as f:
        if "entry_0" not in f or "mc_particle_tree" not in f["entry_0"]:
            return None

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                      prefix="viz_cascade_gt_", delete=False)
    tmp.write(os.path.abspath(merged_path) + "\n")
    tmp.close()
    atexit.register(lambda p=tmp.name: os.path.exists(p) and os.unlink(p))

    def _ds(gt_source):
        return LArFormerDataset(split="test", data_root="/",
                                data_list_file=tmp.name, gt_source=gt_source)[0]

    try:
        s_slice = _ds("slice")
        s_part = _ds("particle")
    except Exception as e:  # pragma: no cover
        print(f"[viz] GT compute failed for {merged_path}: {e}")
        return None

    coord = np.asarray(s_slice["coord"], np.float32)
    slice_id = np.asarray(s_slice["slice_id"], np.int64)
    # slice_id value -> origin_type (0 = nu) from the slice gt_instances.
    slice_origin = {}
    for g in s_slice.get("gt_instances", []):
        slice_origin[int(g["primary_trackid"])] = int(g.get("origin_type", 1))
    is_nu = np.array([slice_origin.get(int(t), 1) == 0 for t in slice_id], bool)

    # Per-SP particle class (>=0 only for SPs in a thresholded nu particle).
    # NOTE: s_part["slice_id"] is the SLICE label (cosmic+nu), not the
    # particle instance — the per-SP particle instance is built from the
    # particle gt_instances' truth_indices.
    part_coord = np.asarray(s_part["coord"], np.float32)
    part_cls = np.asarray(s_part["particle_class_id"], np.int64)
    part_inst = np.full(len(part_coord), -1, np.int64)
    for k, g in enumerate(s_part.get("gt_instances", [])):
        ti = np.asarray(g["truth_indices"], np.int64)
        if ti.size:
            part_inst[ti] = k
    nu_mask = part_cls >= 0
    return dict(
        coord=coord, slice_id=slice_id, slice_is_nu=is_nu,
        slice_origin=slice_origin,
        hasmatch=np.asarray(s_slice["hasmatch"], np.int64),
        nu_coord=part_coord[nu_mask],
        nu_class=part_cls[nu_mask], nu_instance=part_inst[nu_mask],
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--stage3pred-dir",
                   help="Directory of stage3pred_*.h5 / keypointpred_*.h5.")
    g.add_argument("--stage3pred",
                   help="A single stage3pred_*.h5 / keypointpred_*.h5 file.")
    ap.add_argument("--merged-dir", default=None,
                    help="Directory of merged_*.h5 for the GT panel (sim "
                         "only). If omitted, the GT panel is hidden.")
    ap.add_argument("--entry", type=int, default=0, help="Initial event idx.")
    ap.add_argument("--min-mask-prob", type=float, default=0.0,
                    help="Demote SPs below this mask sigm to unassigned.")
    ap.add_argument("--nu-color-by", default="instance",
                    choices=("instance", "class"),
                    help="Color the nu-slice particles by query/instance "
                         "(default) or by predicted class.")
    ap.add_argument("--max-ghosts", type=int, default=50000,
                    help="When 'show ghosts' is on in the GT panel, randomly "
                         "sample at most this many ghost points (default 50k).")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8051)
    args = ap.parse_args()

    if args.stage3pred:
        pred_files = [os.path.abspath(args.stage3pred)]
    else:
        d = os.path.abspath(args.stage3pred_dir)
        pred_files = sorted(set(
            f for pat in ("stage3pred_*.h5", "keypointpred_*.h5")
            for f in glob.glob(os.path.join(d, pat))))
    if not pred_files:
        raise SystemExit("No stage3pred_*.h5 / keypointpred_*.h5 files found.")
    n_events = len(pred_files)
    merged_dir = os.path.abspath(args.merged_dir) if args.merged_dir else None
    has_gt_dir = merged_dir is not None
    print(f"[viz] {n_events} prediction file(s); "
          f"GT {'from ' + merged_dir if has_gt_dir else 'disabled'}")

    def _gt_for(idx: int):
        if not has_gt_dir:
            return None
        base = _merged_base_from_pred(pred_files[idx])
        cand = os.path.join(merged_dir, base + ".h5")
        if not os.path.exists(cand):
            # also try a recursive search (3-level hashed output trees)
            hits = glob.glob(os.path.join(merged_dir, "**", base + ".h5"),
                             recursive=True)
            cand = hits[0] if hits else None
        return _compute_gt(cand) if cand and os.path.exists(cand) else None

    app = Dash(__name__, suppress_callback_exceptions=True)
    app.title = "LArFormer full-cascade visualizer"

    app.layout = html.Div([
        html.H3("LArFormer full-cascade visualizer",
                style={"marginBottom": "4px"}),
        html.Div([
            html.Span("Event: "),
            dcc.Input(id="entry", type="number", min=0, max=n_events - 1,
                      step=1, value=args.entry,
                      style={"width": "70px", "marginRight": "16px"}),
            html.Span("nu color by: "),
            dcc.Dropdown(id="nu_color", options=cascade_nu_color_options(),
                         value="instance" if args.nu_color_by == "instance"
                         else "class", clearable=False,
                         style={"width": "260px", "display": "inline-block",
                                "marginRight": "16px"}),
            html.Span("min_mask_prob: "),
            dcc.Input(id="min_mask_prob", type="number", min=0.0, max=1.0,
                      step=0.05, value=args.min_mask_prob,
                      style={"width": "70px", "marginRight": "16px"}),
            dcc.Checklist(id="show_ghosts",
                          options=[{"label": " show ghosts (GT)",
                                    "value": "on"}],
                          value=[], style={"display": "inline-block",
                                           "marginRight": "16px"}),
            dcc.Checklist(id="show_keypoints",
                          options=[{"label": " show keypoints",
                                    "value": "on"}],
                          value=["on"], style={"display": "inline-block",
                                               "marginRight": "16px"}),
            dcc.Checklist(id="sync", options=[{"label": " sync camera",
                                               "value": "on"}],
                          value=["on"], style={"display": "inline-block"}),
        ], style={"marginBottom": "6px"}),
        html.Div("PREDICTION", style={"fontWeight": "bold", "marginLeft": "8px"}),
        dcc.Graph(id="pred_scene", style={"height": "46vh"}),
        html.Div("GROUND TRUTH", style={"fontWeight": "bold",
                                        "marginLeft": "8px"}),
        dcc.Graph(id="gt_scene", style={"height": "46vh"}),
        html.Div(id="meta", style={"fontFamily": "monospace",
                                   "fontSize": "12px", "padding": "8px"}),
    ])

    def _clamp(v):
        return max(0, min(int(v or 0), n_events - 1))

    @app.callback(
        Output("pred_scene", "figure"),
        Output("gt_scene", "figure"),
        Output("meta", "children"),
        Input("entry", "value"),
        Input("nu_color", "value"),
        Input("min_mask_prob", "value"),
        Input("show_ghosts", "value"),
        Input("show_keypoints", "value"),
    )
    def update(entry_val, nu_color, min_prob, show_ghosts_val, show_kp_val):
        idx = _clamp(entry_val)
        try:
            mp = float(min_prob) if min_prob is not None else 0.0
        except (TypeError, ValueError):
            mp = 0.0
        show_ghosts = bool(show_ghosts_val and "on" in show_ghosts_val)
        show_kp = bool(show_kp_val and "on" in show_kp_val)
        pred = load_event_h5(pred_files[idx])
        suffix = f"  —  {os.path.basename(pred_files[idx])}"
        pred_fig = figure_for_cascade_prediction(
            pred, min_mask_prob=mp, nu_color_by=nu_color, title_suffix=suffix)
        n_kp = 0
        if show_kp:
            n_kp = _add_predicted_keypoints(pred_fig, pred)
        gt = _gt_for(idx)
        gt_fig = figure_for_cascade_gt(gt, nu_color_by=nu_color,
                                       show_ghosts=show_ghosts,
                                       max_ghosts=args.max_ghosts,
                                       title_suffix=suffix)
        n_post = int(pred["post/coord"].shape[0]) if pred.get("post/coord") is not None else 0
        n_s3 = int(pred["stage3/coord"].shape[0]) if pred.get("stage3/coord") is not None else 0
        has_kp = pred.get("keypoints/pos_cm") is not None
        kp_str = (f"keypoints shown={n_kp}" if show_kp and has_kp
                  else "keypoints: none in file" if not has_kp
                  else "keypoints: hidden")
        meta = (f"event {idx+1}/{n_events}  |  post(deghost) SPs={n_post}  "
                f"nu-slice(stage3) SPs={n_s3}  |  {kp_str}  |  "
                f"GT={'yes' if gt is not None else 'none (real data / no merged_h5)'}")
        return pred_fig, gt_fig, meta

    # ---- camera sync (both directions) ----
    def _cam(rl):
        if not rl:
            return None
        if "scene.camera" in rl:
            return rl["scene.camera"]
        keys = [k for k in rl if k.startswith("scene.camera.")]
        if not keys:
            return None
        cam = {}
        for k in keys:
            parts = k[len("scene.camera."):].split(".")
            d = cam
            for p in parts[:-1]:
                d = d.setdefault(p, {})
            d[parts[-1]] = rl[k]
        return cam

    @app.callback(Output("gt_scene", "figure", allow_duplicate=True),
                  Input("pred_scene", "relayoutData"),
                  State("sync", "value"), prevent_initial_call=True)
    def sync_to_gt(rl, sync):
        if not sync or "on" not in sync:
            return no_update
        cam = _cam(rl)
        if cam is None:
            return no_update
        p = Patch(); p["layout"]["scene"]["camera"] = cam
        return p

    @app.callback(Output("pred_scene", "figure", allow_duplicate=True),
                  Input("gt_scene", "relayoutData"),
                  State("sync", "value"), prevent_initial_call=True)
    def sync_to_pred(rl, sync):
        if not sync or "on" not in sync:
            return no_update
        cam = _cam(rl)
        if cam is None:
            return no_update
        p = Patch(); p["layout"]["scene"]["camera"] = cam
        return p

    print(f"\nRunning on http://{args.host}:{args.port}\n")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
