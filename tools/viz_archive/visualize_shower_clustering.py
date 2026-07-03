"""
Visualize the ShowerClusteringDataset — what the Mask2Former model will see.

Builds the dataset from a config (default: configs/lartpc/shower_origin/archive/shower-cluster-sonata-v1.py),
loads one event at a time, and shows four 3D plotly panels:

    1. Ground-truth instances    — each Mask2Former GT mask its own color
                                   (the union over a trunk's mc_particle_tree
                                   descendants); spacepoints not in any GT
                                   instance are gray. This is the bipartite
                                   matching target.
    2. DBSCAN fragments          — each fragment its own color (model's
                                   fragment tokens). Non-fragment spacepoints
                                   gray.
    3. SSNet label               — per-spacepoint SSNet class. Highlights
                                   the SSNet-false-negative regions where
                                   true shower spacepoints are mislabeled.
    4. lm_score (post-threshold) — color by lm_score on surviving spacepoints
                                   for the sampled τ (train: U(0.15,0.40);
                                   val: 0.15).

Buttons: Next / Previous / Random / Resample (re-draw same event with a new
sampled τ to visualize the lm_score augmentation).

Usage:
    python tools/viz_archive/visualize_shower_clustering.py \\
        -c configs/lartpc/shower_origin/archive/shower-cluster-sonata-v1.py

    python tools/viz_archive/visualize_shower_clustering.py \\
        -c configs/lartpc/shower_origin/archive/shower-cluster-sonata-v1.py \\
        --split val --data-list /path/to/files.txt --port 8053

The dataset class (pointcept/datasets/shower_clustering.py) already applies
the lm_score threshold + voxelization, so the visualizer renders exactly the
arrays the model receives.
"""

import os
import sys
import argparse
from typing import Optional

import numpy as np

# Project root on path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State

from pointcept.utils.config import Config
from pointcept.datasets.builder import DATASETS

# Tools dir on path so we can import the detector outline directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detectoroutline import DetectorOutline  # noqa: E402


# ── Constants ───────────────────────────────────────────────────────────────

ORIGIN_TYPE_NAMES = {
    0: "inside",
    1: "outside",
    2: "on_track",
    3: "ghost",
    4: "true_track",
    -1: "unknown",
}

# Plotly's Light24 — 24 visually distinct colors. Cycle for higher counts.
QUALITATIVE_PALETTE = [
    "#FD3216", "#00FE35", "#6A76FC", "#FED4C4", "#FE00CE", "#0DF9FF",
    "#F6F926", "#FF9616", "#479B55", "#EEA6FB", "#DC587D", "#D626FF",
    "#6E899C", "#00B5F7", "#B68E00", "#C9FBE5", "#FF0092", "#22FFA7",
    "#E3EE9E", "#86CE00", "#BC7196", "#7E7DCD", "#FC6955", "#E48F72",
]

# Coarse SSNet → human label map (the user's earlier inspection showed
# 0 / 2-4 / 6-8 in this dataset; exact mapping is the upstream PrepFlowMatch
# convention). Color by integer; legend is best-effort.
SSNET_NAMES = {
    0: "bg/empty", 1: "shower-e", 2: "shower-γ", 3: "track-μ",
    4: "track-π", 5: "track-p", 6: "label-6", 7: "label-7", 8: "label-8",
    -1: "unmatched",
}

# Axis template + scene layout matching tools/viz_archive/visualize_larmatch_h5data.py
AXIS_TEMPLATE = {
    "showbackground": True,
    "backgroundcolor": "#141414",
    "gridcolor": "rgb(80, 80, 80)",
    "zerolinecolor": "rgb(128, 128, 128)",
    "title_font": {"color": "white"},
    "tickfont": {"color": "white"},
}


def make_layout(title, npts, height=600):
    return {
        "title": {
            "text": f"{title} ({npts:,} pts)",
            "font": {"size": 13, "color": "white"},
        },
        "height": height,
        "margin": {"l": 0, "r": 0, "t": 40, "b": 0},
        "font": {"size": 9, "color": "white"},
        "showlegend": True,
        "legend": {
            "yanchor": "top", "y": 0.99,
            "xanchor": "left", "x": 0.01,
            "font": {"color": "white", "size": 9},
            "bgcolor": "rgba(20, 20, 20, 0.8)",
            "itemsizing": "constant",
        },
        "plot_bgcolor": "#141414",
        "paper_bgcolor": "#141414",
        "scene": {
            "xaxis": {**AXIS_TEMPLATE, "title": "X (cm)"},
            "yaxis": {**AXIS_TEMPLATE, "title": "Y (cm)"},
            "zaxis": {**AXIS_TEMPLATE, "title": "Z (cm)"},
            "aspectratio": {"x": 1, "y": 1, "z": 4},
            "camera": {
                "eye": {"x": 2, "y": 2, "z": 2},
                "up": {"x": 0, "y": 1, "z": 0},
            },
        },
    }


def empty_figure(title="N/A"):
    return {"data": [], "layout": make_layout(title, 0)}


# Cache the detector outline traces — same for every figure
_DETECTOR_OUTLINE_TRACES = None


def _detector_outline_traces():
    global _DETECTOR_OUTLINE_TRACES
    if _DETECTOR_OUTLINE_TRACES is None:
        det = DetectorOutline()
        traces = det.getlines(color=(255, 255, 255))
        # The dict from getlines is already a valid plotly scatter3d trace,
        # but mark showlegend=False so it doesn't pollute the legend.
        for t in traces:
            t["showlegend"] = False
            t["hoverinfo"] = "skip"
        _DETECTOR_OUTLINE_TRACES = traces
    return [dict(t) for t in _DETECTOR_OUTLINE_TRACES]


# ── Argument parsing ────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize ShowerClusteringDataset (Mask2Former GT)"
    )
    parser.add_argument(
        "-c", "--config", required=True, type=str,
        help="Config file path (e.g. configs/lartpc/shower_origin/archive/shower-cluster-sonata-v1.py)"
    )
    parser.add_argument(
        "--split", default="train", choices=["train", "val", "test"],
        help="Dataset split (default: train; controls lm_score augmentation)"
    )
    parser.add_argument(
        "-e", "--entry", default=0, type=int, help="Initial entry index"
    )
    parser.add_argument(
        "--data-list", type=str, default=None,
        help="Override data_list_file in the config"
    )
    parser.add_argument(
        "--gt-label-source", type=str, default=None,
        choices=["truth", "fragment", "union"],
        help=("Override the dataset's gt_label_source. "
              "'truth' = descendants of trunk trackid (ghost-free, "
              "includes shower points DBSCAN missed). "
              "'fragment' = union of reco fragments whose plurality "
              "trackid descends from trunk (includes ghosts DBSCAN "
              "clustered with shower points; excludes missed points). "
              "'union' = set union of both. Default: use config.")
    )
    parser.add_argument(
        "--max-points", type=int, default=120000,
        help=("Cap rendered point count per panel for browser responsiveness "
              "(default: 120000). Subsampling preserves all GT-instance and "
              "fragment-member spacepoints; only background gets thinned.")
    )
    parser.add_argument(
        "--port", default=8050, type=int, help="Dash server port"
    )
    return parser.parse_args()


# ── Build dataset directly from config ──────────────────────────────────────

def build_dataset(cfg, split, data_list_override=None,
                  gt_label_source_override=None):
    dataset_cfg = getattr(cfg.data, split).copy()
    if data_list_override is not None:
        dataset_cfg["data_list_file"] = os.path.abspath(data_list_override)
    if gt_label_source_override is not None:
        dataset_cfg["gt_label_source"] = gt_label_source_override
    # The visualizer wants augmentation behavior on train (samples τ) but no
    # other transforms. The dataset class is self-contained — no Compose.
    dataset_cfg["transform"] = None
    return DATASETS.build(dataset_cfg)


# ── Subsampling for rendering ────────────────────────────────────────────────

def _build_subsample_mask(data: dict, max_points: int) -> np.ndarray:
    """
    Pick which spacepoints to render. Always keep:
      - all GT-instance member spacepoints
      - all DBSCAN-fragment member spacepoints
    Then random-sample the remainder up to the budget.
    """
    n = data["n_spacepoints"]
    if n <= max_points:
        return np.ones(n, dtype=bool)

    keep = np.zeros(n, dtype=bool)
    for inst in data.get("gt_instances", []):
        keep[inst["truth_indices"]] = True
    for idx in data.get("fragment_indices", []):
        keep[idx] = True
    forced = int(keep.sum())
    remainder_budget = max(0, max_points - forced)
    if remainder_budget > 0:
        candidates = np.where(~keep)[0]
        if len(candidates) > remainder_budget:
            chosen = np.random.choice(
                candidates, remainder_budget, replace=False)
            keep[chosen] = True
        else:
            keep[candidates] = True
    return keep


# ── Figure builders ─────────────────────────────────────────────────────────

def _scatter_trace(coords, name, color, size=2, opacity=0.85, hovertext=None):
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


def build_gt_instances_figure(data, mask_render):
    """Each GT instance (one per Mask2Former target) gets its own color.
    Largest instances first so they get the most-distinct palette colors."""
    coord = data["coord"][mask_render]  # detector cm
    n_render = coord.shape[0]
    instances = data["gt_instances"]

    # Sort by size (descending) for color assignment only — does not affect
    # the underlying dataset ordering.
    order = sorted(range(len(instances)),
                   key=lambda i: -instances[i]["n_truth_points"])

    sp_color_idx = np.full(data["n_spacepoints"], -1, dtype=np.int64)
    for color_idx, orig_idx in enumerate(order):
        sp_color_idx[instances[orig_idx]["truth_indices"]] = color_idx

    sp_color_render = sp_color_idx[mask_render]
    traces = _detector_outline_traces()
    background = sp_color_render == -1
    if background.any():
        traces.append(_scatter_trace(
            coord[background],
            name=f"background ({int(background.sum()):,})",
            color="rgba(120,120,120,0.25)", size=1.5,
        ))
    for color_idx, orig_idx in enumerate(order):
        sel = sp_color_render == color_idx
        if not sel.any():
            continue
        inst = instances[orig_idx]
        color = QUALITATIVE_PALETTE[color_idx % len(QUALITATIVE_PALETTE)]
        traces.append(_scatter_trace(
            coord[sel],
            name=(f"inst#{orig_idx}: pid={inst['pid']} "
                  f"tid={inst['trunk_trackid']} "
                  f"(n={inst['n_truth_points']})"),
            color=color, size=2.2, opacity=0.9,
        ))
    return {"data": traces,
            "layout": make_layout("Ground-Truth Instances (Mask2Former targets)",
                                  n_render)}


def build_fragments_figure(data, mask_render):
    """Each DBSCAN fragment its own color. Largest fragments first."""
    coord = data["coord"][mask_render]  # detector cm
    n_render = coord.shape[0]
    fragments = data["fragment_indices"]
    frag_tids = data["fragment_trackid"]
    frag_pids = data["fragment_pid"]
    frag_types = data["fragment_type"]

    order = sorted(range(len(fragments)), key=lambda i: -len(fragments[i]))

    sp_color_idx = np.full(data["n_spacepoints"], -1, dtype=np.int64)
    for color_idx, orig_idx in enumerate(order):
        sp_color_idx[fragments[orig_idx]] = color_idx

    sp_color_render = sp_color_idx[mask_render]
    traces = _detector_outline_traces()
    background = sp_color_render == -1
    if background.any():
        traces.append(_scatter_trace(
            coord[background],
            name=f"non-fragment ({int(background.sum()):,})",
            color="rgba(120,120,120,0.2)", size=1.5,
        ))
    for color_idx, orig_idx in enumerate(order):
        sel = sp_color_render == color_idx
        if not sel.any():
            continue
        color = QUALITATIVE_PALETTE[color_idx % len(QUALITATIVE_PALETTE)]
        tname = ORIGIN_TYPE_NAMES.get(int(frag_types[orig_idx]), "?")
        traces.append(_scatter_trace(
            coord[sel],
            name=(f"frag#{orig_idx}: pid={int(frag_pids[orig_idx])} "
                  f"tid={int(frag_tids[orig_idx])} type={tname} "
                  f"(n={len(fragments[orig_idx])})"),
            color=color, size=2.2, opacity=0.9,
        ))
    return {"data": traces,
            "layout": make_layout("DBSCAN Fragments (model fragment tokens)",
                                  n_render)}


def build_ssnet_figure(data, mask_render):
    """Color by SSNet label. Highlights SSNet false-negatives on truth showers."""
    coord = data["coord"][mask_render]  # detector cm
    n_render = coord.shape[0]
    ssnet = data["ssnet_label"][mask_render]

    traces = _detector_outline_traces()
    unique = np.unique(ssnet)
    for u in unique:
        sel = ssnet == u
        if not sel.any():
            continue
        u_int = int(u)
        if u_int in (1, 2):  # shower-electron / gamma
            color = "#FF3030"
        elif u_int in (3, 4, 5):  # track classes
            color = "#3060FF"
        elif u_int == 0:
            color = "rgba(120,120,120,0.25)"
        else:
            color = QUALITATIVE_PALETTE[(u_int + 4) % len(QUALITATIVE_PALETTE)]
        name = SSNET_NAMES.get(u_int, f"label-{u_int}")
        traces.append(_scatter_trace(
            coord[sel],
            name=f"{name} ({int(sel.sum()):,})",
            color=color, size=2 if u_int != 0 else 1.5,
            opacity=0.7 if u_int == 0 else 0.85,
        ))
    return {"data": traces,
            "layout": make_layout("SSNet labels (FNs feed voxel scale)",
                                  n_render)}


def build_lm_score_figure(data, mask_render):
    """Color by larmatch score on surviving spacepoints."""
    coord = data["coord"][mask_render]  # detector cm
    n_render = coord.shape[0]
    lm = data["lm_score"][mask_render]

    traces = _detector_outline_traces()
    traces.append({
        "type": "scatter3d",
        "x": coord[:, 0].tolist(),
        "y": coord[:, 1].tolist(),
        "z": coord[:, 2].tolist(),
        "mode": "markers",
        "name": f"all ({n_render:,})",
        "marker": {
            "color": lm.tolist(),
            "colorscale": "Viridis",
            "size": 2,
            "opacity": 0.85,
            "cmin": float(data["lm_score_threshold"]),
            "cmax": 1.0,
            "colorbar": {"title": "lm_score", "len": 0.6},
        },
        "customdata": np.stack([lm], axis=1),
        "hovertemplate": (
            "<b>x</b>: %{x:.1f} cm<br>"
            "<b>y</b>: %{y:.1f} cm<br>"
            "<b>z</b>: %{z:.1f} cm<br>"
            "<b>lm_score</b>: %{customdata[0]:.3f}<extra></extra>"
        ),
    })
    title = (f"lm_score (sampled τ={data['lm_score_threshold']:.3f}, "
             f"surviving)")
    return {"data": traces, "layout": make_layout(title, n_render)}


# ── Main ────────────────────────────────────────────────────────────────────

args = parse_args()
cfg = Config.fromfile(args.config)

dataset = build_dataset(
    cfg, args.split,
    data_list_override=args.data_list,
    gt_label_source_override=args.gt_label_source,
)
n_entries = len(dataset)

print(f"[viz] config: {args.config}")
print(f"[viz] split:  {args.split}  (n_entries={n_entries})")
print(f"[viz] max points rendered per panel: {args.max_points:,}")
if n_entries == 0:
    print("ERROR: dataset is empty. Check data_list_file in the config.",
          file=sys.stderr)
    sys.exit(1)


def process_entry(idx):
    """Load one event and build the four figures."""
    data = dataset[idx]
    n = data["n_spacepoints"]
    mask = _build_subsample_mask(data, args.max_points)
    return data, mask


# ── Dash app ────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ],
)

GRAPH_STYLE = {
    "width": "50%",
    "display": "inline-block",
    "verticalAlign": "top",
}

BUTTON_STYLE_BASE = {
    "fontSize": "14px",
    "padding": "8px 16px",
    "color": "white",
    "border": "none",
    "borderRadius": "5px",
    "cursor": "pointer",
    "marginRight": "8px",
}

app.layout = html.Div(
    style={
        "backgroundColor": "#1e1e1e",
        "padding": "16px",
        "minHeight": "100vh",
        "fontFamily": "monospace",
    },
    children=[
        html.Div(
            [
                html.H2(
                    "Shower-Clustering Dataset Visualizer",
                    style={"color": "white", "display": "inline-block",
                           "marginRight": "20px"},
                ),
                html.Button("Prev", id="prev-btn", n_clicks=0,
                            style={**BUTTON_STYLE_BASE,
                                   "backgroundColor": "#607D8B"}),
                html.Button("Next", id="next-btn", n_clicks=0,
                            style={**BUTTON_STYLE_BASE,
                                   "backgroundColor": "#4CAF50"}),
                html.Button("Random", id="random-btn", n_clicks=0,
                            style={**BUTTON_STYLE_BASE,
                                   "backgroundColor": "#FF9800"}),
                html.Button("Resample τ", id="resample-btn", n_clicks=0,
                            style={**BUTTON_STYLE_BASE,
                                   "backgroundColor": "#9C27B0"}),
                dcc.Input(
                    id="entry-input", type="number",
                    value=args.entry, min=0, max=max(n_entries - 1, 0),
                    placeholder="Entry index",
                    style={"fontSize": "13px", "padding": "6px",
                           "width": "100px", "borderRadius": "5px",
                           "border": "1px solid #555",
                           "backgroundColor": "#333", "color": "white",
                           "marginRight": "8px"},
                ),
                html.Button("Go", id="go-btn", n_clicks=0,
                            style={**BUTTON_STYLE_BASE,
                                   "backgroundColor": "#2196F3"}),
            ],
            style={"marginBottom": "8px"},
        ),
        html.Div(id="entry-label",
                 style={"color": "#ccc", "fontSize": "12px",
                        "marginBottom": "4px"}),
        html.Div(
            f"Config: {os.path.basename(args.config)}  |  "
            f"split: {args.split}  |  "
            f"voxel: {cfg.voxel_size_cm if hasattr(cfg, 'voxel_size_cm') else 5.0} cm  |  "
            f"τ range: [{cfg.lm_score_aug_low if hasattr(cfg, 'lm_score_aug_low') else 0.15}, "
            f"{cfg.lm_score_aug_high if hasattr(cfg, 'lm_score_aug_high') else 0.40}]",
            style={"color": "#888", "fontSize": "11px",
                   "marginBottom": "12px"},
        ),
        html.Div(
            [
                html.Div([dcc.Graph(id="gt-graph",
                                    config={"scrollZoom": True})],
                         style=GRAPH_STYLE),
                html.Div([dcc.Graph(id="frag-graph",
                                    config={"scrollZoom": True})],
                         style=GRAPH_STYLE),
            ]
        ),
        html.Div(
            [
                html.Div([dcc.Graph(id="ssnet-graph",
                                    config={"scrollZoom": True})],
                         style=GRAPH_STYLE),
                html.Div([dcc.Graph(id="lm-graph",
                                    config={"scrollZoom": True})],
                         style=GRAPH_STYLE),
            ]
        ),
        dcc.Store(id="current-idx", data=args.entry),
    ],
)


@app.callback(
    [
        Output("gt-graph", "figure"),
        Output("frag-graph", "figure"),
        Output("ssnet-graph", "figure"),
        Output("lm-graph", "figure"),
        Output("entry-label", "children"),
        Output("current-idx", "data"),
    ],
    [
        Input("prev-btn", "n_clicks"),
        Input("next-btn", "n_clicks"),
        Input("random-btn", "n_clicks"),
        Input("resample-btn", "n_clicks"),
        Input("go-btn", "n_clicks"),
    ],
    [
        State("current-idx", "data"),
        State("entry-input", "value"),
    ],
)
def update_display(_p, _n, _r, _re, _g, current_idx, entry_input):
    ctx = dash.callback_context
    triggered = ctx.triggered[0]["prop_id"] if ctx.triggered else ""

    if "prev-btn" in triggered:
        idx = (int(current_idx) - 1) % n_entries
    elif "next-btn" in triggered:
        idx = (int(current_idx) + 1) % n_entries
    elif "random-btn" in triggered:
        idx = int(np.random.randint(0, n_entries))
    elif "resample-btn" in triggered:
        idx = int(current_idx) if current_idx is not None else 0
        # don't change idx; loading the entry resamples τ since split=train
    elif "go-btn" in triggered and entry_input is not None:
        idx = int(np.clip(entry_input, 0, n_entries - 1))
    else:
        idx = int(current_idx) if current_idx is not None else 0

    for attempt in range(5):
        try:
            data, render_mask = process_entry(idx)
            break
        except Exception as e:
            print(f"[viz] error on entry {idx}: {e}")
            import traceback
            traceback.print_exc()
            idx = (idx + 1) % n_entries
    else:
        return (
            empty_figure("GT instances (error)"),
            empty_figure("Fragments (error)"),
            empty_figure("SSNet (error)"),
            empty_figure("lm_score (error)"),
            f"Error: could not process entries near index {idx}",
            idx,
        )

    figs = (
        build_gt_instances_figure(data, render_mask),
        build_fragments_figure(data, render_mask),
        build_ssnet_figure(data, render_mask),
        build_lm_score_figure(data, render_mask),
    )

    n_render = int(render_mask.sum())
    n_total = data["n_spacepoints"]
    label = (
        f"Entry {idx}/{n_entries}  |  "
        f"{data['name']} (run/subrun/event {data['run']}/{data['subrun']}/{data['event']})  |  "
        f"τ={data['lm_score_threshold']:.3f}  |  "
        f"surviving SPs: {n_total:,} (rendered {n_render:,})  |  "
        f"voxels: {data['n_voxels']:,}  |  "
        f"fragments: {data['n_fragments']}  |  "
        f"GT instances: {data['n_gt_instances']}"
    )
    return *figs, label, idx


if __name__ == "__main__":
    app.run(debug=False, port=args.port)
