"""
Visualize Shower Origin data from the data loader.

Reads a shower-origin config file, builds the dataset (without training
transforms), and displays one sample showing three 3D views:
  1. All points colored by shower_mask (shower vs non-shower)
  2. All points colored by origin_distance (from data loader, cm-space)
  3. All points colored by Gaussian target score (recomputed in normalized
     space, matching _single_fragment_loss)

The data is shown AFTER NormalizeShowerCoords (same coordinate frame the
model sees), but without GridSample/BiasedSphereCrop so all points are
visible.

A "Next Sample" button loads the next entry; "Go" jumps to a specific index.

Usage:
    python tools/visualize_shower_origin.py \
        -c configs/lartpc/shower_origin/archive/shower-origin-sonata-v1m1-v3.py

    python tools/visualize_shower_origin.py \
        -c configs/lartpc/shower_origin/archive/shower-origin-sonata-v1m1-v3.py \
        --data-list /path/to/filelist.txt --port 8052
"""

import os
import sys
import argparse

import numpy as np

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State

from pointcept.utils.config import Config
from pointcept.datasets.builder import DATASETS
from pointcept.datasets.transform import Compose, TRANSFORMS


# ── Origin type labels ──────────────────────────────────────────────────────

ORIGIN_TYPE_NAMES = {
    0: "inside",
    1: "outside",
    2: "on_track",
    -1: "none",
}


# ── Plotly layout templates ─────────────────────────────────────────────────

AXIS_TEMPLATE = {
    "showbackground": True,
    "backgroundcolor": "#141414",
    "gridcolor": "rgb(100, 100, 100)",
    "zerolinecolor": "rgb(200, 200, 200)",
}


def make_layout(title, npts, height=550):
    return {
        "title": {
            "text": f"{title} ({npts:,} pts)",
            "font": {"size": 14, "color": "white"},
        },
        "height": height,
        "margin": {"t": 40, "b": 10, "l": 10, "r": 10},
        "font": {"size": 10, "color": "white"},
        "showlegend": True,
        "legend": {"font": {"color": "white", "size": 10}},
        "plot_bgcolor": "#141414",
        "paper_bgcolor": "#1e1e1e",
        "scene": {
            "xaxis": AXIS_TEMPLATE,
            "yaxis": AXIS_TEMPLATE,
            "zaxis": AXIS_TEMPLATE,
            "aspectmode": "data",
            "camera": {"eye": {"x": 1.5, "y": 1.5, "z": 1.5}},
        },
    }


def empty_figure(title="N/A"):
    return {"data": [], "layout": make_layout(title, 0)}


# ── Argument parsing ────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize shower origin data from data loader"
    )
    parser.add_argument(
        "-c", "--config", required=True, type=str, help="Config file path"
    )
    parser.add_argument(
        "-e", "--entry", default=0, type=int, help="Initial entry index"
    )
    parser.add_argument(
        "--data-list", type=str, default=None,
        help="Override data list file path"
    )
    parser.add_argument(
        "--split", type=str, default="train", choices=["train", "val"],
        help="Dataset split to visualize (default: train)"
    )
    parser.add_argument(
        "--port", default=8050, type=int, help="Dash server port"
    )
    return parser.parse_args()


# ── Build dataset (without transforms) ─────────────────────────────────────

def build_dataset(cfg, split, data_list_override=None):
    """
    Build the dataset WITHOUT transforms.

    We call get_data() directly and apply NormalizeShowerCoords manually
    so we have access to all data fields (name, origin_distance, etc.)
    that would otherwise be dropped by Collect.
    """
    dataset_cfg = getattr(cfg.data, split).copy()

    if data_list_override is not None:
        # Resolve to absolute path so get_data_list() doesn't join with data_root
        dataset_cfg["data_list_file"] = os.path.abspath(data_list_override)

    # Remove transforms — we apply them manually in process_entry
    dataset_cfg["transform"] = []

    dataset = DATASETS.build(dataset_cfg)
    return dataset


def build_normalizer(cfg, split):
    """
    Extract the NormalizeShowerCoords transform from the config so we can
    apply it manually to the raw data.
    """
    dataset_cfg = getattr(cfg.data, split)
    for t in dataset_cfg.get("transform", []):
        if t.get("type") == "NormalizeShowerCoords":
            return TRANSFORMS.build(t)
    return None


# ── Data processing ─────────────────────────────────────────────────────────

def process_entry(dataset, idx, normalizer=None):
    """
    Load one sample via get_data() (no training transforms) and optionally
    apply NormalizeShowerCoords.

    Returns dict with numpy arrays:
        coord, shower_mask, origin_coord, origin_type,
        origin_distance, strength, start_coord, name
    """
    data = dataset.get_data(idx)

    # Apply NormalizeShowerCoords to put coords in model's frame
    if normalizer is not None:
        data = normalizer(data)

    result = {}
    for key in ("coord", "shower_mask", "origin_coord", "origin_type",
                "origin_distance", "strength", "start_coord",
                "fragment_centroid", "color"):
        if key in data:
            val = data[key]
            if hasattr(val, "numpy"):
                result[key] = val.numpy()
            elif isinstance(val, np.ndarray):
                result[key] = val
            else:
                result[key] = val

    result["name"] = data.get("name", f"entry_{idx}")

    # Diagnostic: print distance stats for sanity checking
    if "coord" in result and "origin_coord" in result:
        coord = result["coord"]
        origin = result["origin_coord"]
        dists = np.linalg.norm(coord - origin[np.newaxis, :], axis=1)
        mask = result.get("shower_mask")
        if mask is not None:
            shower_dists = dists[mask.astype(bool)]
            if shower_dists.size > 0:
                print(f"  [entry {idx}] Normalized dist to origin: "
                      f"all min={dists.min():.4f} max={dists.max():.4f} | "
                      f"shower min={shower_dists.min():.4f} max={shower_dists.max():.4f}")
            else:
                print(f"  [entry {idx}] Normalized dist to origin: "
                      f"all min={dists.min():.4f} max={dists.max():.4f} | "
                      f"shower: 0 points")
        else:
            print(f"  [entry {idx}] Normalized dist to origin: "
                  f"min={dists.min():.4f} max={dists.max():.4f}")

    return result


# ── Figure builders ─────────────────────────────────────────────────────────

def build_shower_mask_figure(data, title="Shower Mask"):
    """
    All points, colored by shower_mask.
    Shower points in red, non-shower in gray.
    Origin point as a large green marker.
    """
    coord = data["coord"]
    mask = data.get("shower_mask")
    origin = data.get("origin_coord")
    npts = coord.shape[0]

    traces = []

    if mask is not None:
        shower = mask.astype(bool)
        non_shower = ~shower

        if non_shower.sum() > 0:
            traces.append({
                "type": "scatter3d",
                "x": coord[non_shower, 0].tolist(),
                "y": coord[non_shower, 1].tolist(),
                "z": coord[non_shower, 2].tolist(),
                "mode": "markers",
                "name": f"non-shower ({int(non_shower.sum()):,})",
                "marker": {"color": "rgba(120,120,120,0.3)", "size": 2},
                "hovertemplate": (
                    "<b>x</b>: %{x:.3f}<br>"
                    "<b>y</b>: %{y:.3f}<br>"
                    "<b>z</b>: %{z:.3f}<br>"
                    "shower: no<extra></extra>"
                ),
            })

        if shower.sum() > 0:
            traces.append({
                "type": "scatter3d",
                "x": coord[shower, 0].tolist(),
                "y": coord[shower, 1].tolist(),
                "z": coord[shower, 2].tolist(),
                "mode": "markers",
                "name": f"shower ({int(shower.sum()):,})",
                "marker": {"color": "rgba(255,50,50,0.8)", "size": 2},
                "hovertemplate": (
                    "<b>x</b>: %{x:.3f}<br>"
                    "<b>y</b>: %{y:.3f}<br>"
                    "<b>z</b>: %{z:.3f}<br>"
                    "shower: yes<extra></extra>"
                ),
            })
    else:
        traces.append({
            "type": "scatter3d",
            "x": coord[:, 0].tolist(),
            "y": coord[:, 1].tolist(),
            "z": coord[:, 2].tolist(),
            "mode": "markers",
            "name": f"all ({npts:,})",
            "marker": {"color": coord[:, 2].tolist(), "size": 2,
                       "colorscale": "Viridis"},
        })

    # Origin marker
    if origin is not None:
        traces.append({
            "type": "scatter3d",
            "x": [float(origin[0])],
            "y": [float(origin[1])],
            "z": [float(origin[2])],
            "mode": "markers",
            "name": "origin",
            "marker": {
                "color": "lime",
                "size": 10,
                "symbol": "diamond",
                "line": {"color": "white", "width": 2},
            },
            "hovertemplate": (
                "<b>ORIGIN</b><br>"
                "x: %{x:.3f}<br>y: %{y:.3f}<br>z: %{z:.3f}"
                "<extra></extra>"
            ),
        })

    # Start coord marker
    if "start_coord" in data and data["start_coord"] is not None:
        sc = data["start_coord"]
        traces.append({
            "type": "scatter3d",
            "x": [float(sc[0])],
            "y": [float(sc[1])],
            "z": [float(sc[2])],
            "mode": "markers",
            "name": "start",
            "marker": {
                "color": "cyan",
                "size": 8,
                "symbol": "cross",
                "line": {"color": "white", "width": 1},
            },
            "hovertemplate": (
                "<b>START</b><br>"
                "x: %{x:.3f}<br>y: %{y:.3f}<br>z: %{z:.3f}"
                "<extra></extra>"
            ),
        })

    return {"data": traces, "layout": make_layout(title, npts)}


def build_origin_distance_figure(data, title="origin_distance (from data loader)"):
    """
    All points colored by the raw origin_distance field from the data loader.
    This field is computed in get_data() BEFORE normalization (cm-space).
    Origin point as a large green marker.
    """
    coord = data["coord"]
    origin = data.get("origin_coord")
    npts = coord.shape[0]

    dist = data.get("origin_distance")
    if dist is not None and hasattr(dist, "numpy"):
        dist = dist.numpy()

    traces = []

    if dist is not None:
        traces.append({
            "type": "scatter3d",
            "x": coord[:, 0].tolist(),
            "y": coord[:, 1].tolist(),
            "z": coord[:, 2].tolist(),
            "mode": "markers",
            "name": f"all ({npts:,})",
            "marker": {
                "color": dist.tolist(),
                "colorscale": "Turbo",
                "reversescale": True,
                "size": 2,
                "opacity": 0.8,
                "colorbar": {"title": "dist (cm)", "len": 0.6},
                "cmin": 0,
                "cmax": float(np.percentile(dist, 95)),
            },
            "customdata": np.stack([dist], axis=1),
            "hovertemplate": (
                "<b>x</b>: %{x:.3f}<br>"
                "<b>y</b>: %{y:.3f}<br>"
                "<b>z</b>: %{z:.3f}<br>"
                "<b>dist</b>: %{customdata[0]:.1f} cm"
                "<extra></extra>"
            ),
        })
    else:
        traces.append({
            "type": "scatter3d",
            "x": coord[:, 0].tolist(),
            "y": coord[:, 1].tolist(),
            "z": coord[:, 2].tolist(),
            "mode": "markers",
            "name": f"all ({npts:,})",
            "marker": {"color": coord[:, 2].tolist(), "size": 2,
                       "colorscale": "Viridis"},
        })

    if origin is not None:
        traces.append({
            "type": "scatter3d",
            "x": [float(origin[0])],
            "y": [float(origin[1])],
            "z": [float(origin[2])],
            "mode": "markers",
            "name": "origin",
            "marker": {
                "color": "lime",
                "size": 10,
                "symbol": "diamond",
                "line": {"color": "white", "width": 2},
            },
        })

    return {"data": traces, "layout": make_layout(title, npts)}


def build_gaussian_target_figure(data, gaussian_sigma, title="Gaussian Target (loss computation)"):
    """
    All points colored by Gaussian target score.
    Recomputed from origin_coord + coord in normalized space,
    matching _single_fragment_loss.
    Origin as green marker.
    """
    coord = data["coord"]
    origin = data.get("origin_coord")
    npts = coord.shape[0]

    traces = []

    # Compute Gaussian target from origin_coord (matches model's loss computation)
    if origin is not None:
        gt_dist = np.linalg.norm(coord - origin[np.newaxis, :], axis=1)
        gt_score = np.exp(-0.5 * (gt_dist / gaussian_sigma) ** 2)
    else:
        gt_score = None
        gt_dist = None

    if gt_score is not None:
        traces.append({
            "type": "scatter3d",
            "x": coord[:, 0].tolist(),
            "y": coord[:, 1].tolist(),
            "z": coord[:, 2].tolist(),
            "mode": "markers",
            "name": f"all ({npts:,})",
            "marker": {
                "color": gt_score.tolist(),
                "colorscale": "viridis",
                "size": 2,
                "opacity": 0.9,
                "cmin": 0,
                "cmax": 1,
                "colorbar": {"title": "score", "len": 0.6},
            },
            "customdata": np.stack([
                gt_score,
                gt_dist,
            ], axis=1),
            "hovertemplate": (
                "<b>x</b>: %{x:.3f}<br>"
                "<b>y</b>: %{y:.3f}<br>"
                "<b>z</b>: %{z:.3f}<br>"
                "<b>target score</b>: %{customdata[0]:.4f}<br>"
                "<b>dist to origin</b>: %{customdata[1]:.4f}"
                "<extra></extra>"
            ),
        })
    else:
        traces.append({
            "type": "scatter3d",
            "x": coord[:, 0].tolist(),
            "y": coord[:, 1].tolist(),
            "z": coord[:, 2].tolist(),
            "mode": "markers",
            "name": f"all ({npts:,})",
            "marker": {"color": "gray", "size": 2},
        })

    if origin is not None:
        traces.append({
            "type": "scatter3d",
            "x": [float(origin[0])],
            "y": [float(origin[1])],
            "z": [float(origin[2])],
            "mode": "markers",
            "name": "origin",
            "marker": {
                "color": "lime",
                "size": 10,
                "symbol": "diamond",
                "line": {"color": "white", "width": 2},
            },
        })

    return {"data": traces, "layout": make_layout(title, npts)}


# ── Main ────────────────────────────────────────────────────────────────────

args = parse_args()
cfg = Config.fromfile(args.config)

dataset = build_dataset(cfg, args.split, data_list_override=args.data_list)
normalizer = build_normalizer(cfg, args.split)

# Extract gaussian_sigma from model config
gaussian_sigma = 4.0
if hasattr(cfg, "model") and "gaussian_sigma" in cfg.model:
    gaussian_sigma = cfg.model.gaussian_sigma

n_entries = len(dataset)
print(f"Dataset: {n_entries} entries (split={args.split})")
print(f"  data_list: {len(dataset.data_list)} files")
print(f"  event_index: {len(dataset.event_index)} events with shower_fragments")
print(f"gaussian_sigma: {gaussian_sigma}")
print(f"Normalizer: {normalizer}")
print(f"Config: {args.config}")

if n_entries == 0:
    print("\nERROR: No events with shower_fragments found!")
    print("  The data_list files may not contain shower_fragments data.")
    if len(dataset.data_list) > 0:
        print(f"  First file: {dataset.data_list[0]}")
    print("\n  Try using --data-list to point to a file list with shower fragment HDF5 files.")
    print("  Example: python tools/visualize_shower_origin.py "
          "-c configs/lartpc/shower_origin/archive/shower-origin-sonata-v1m1-v3.py "
          "--data-list /path/to/shower_origin_filelist.txt")
    sys.exit(1)


# ── Dash app ────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ],
)

HEADER_STYLE = {"color": "white", "marginBottom": "5px"}
GRAPH_STYLE = {
    "width": "33.3%",
    "display": "inline-block",
    "verticalAlign": "top",
}

app.layout = html.Div(
    style={
        "backgroundColor": "#1e1e1e",
        "padding": "20px",
        "minHeight": "100vh",
        "fontFamily": "monospace",
    },
    children=[
        # ── Header ──────────────────────────────────────────
        html.Div(
            [
                html.H2(
                    "Shower Origin Data Visualizer",
                    style={
                        "color": "white",
                        "display": "inline-block",
                        "marginRight": "20px",
                    },
                ),
                html.Button(
                    "Next Sample",
                    id="next-btn",
                    n_clicks=0,
                    style={
                        "fontSize": "16px",
                        "padding": "10px 20px",
                        "backgroundColor": "#4CAF50",
                        "color": "white",
                        "border": "none",
                        "borderRadius": "5px",
                        "cursor": "pointer",
                        "marginRight": "10px",
                    },
                ),
                html.Button(
                    "Random",
                    id="random-btn",
                    n_clicks=0,
                    style={
                        "fontSize": "16px",
                        "padding": "10px 20px",
                        "backgroundColor": "#FF9800",
                        "color": "white",
                        "border": "none",
                        "borderRadius": "5px",
                        "cursor": "pointer",
                        "marginRight": "10px",
                    },
                ),
                dcc.Input(
                    id="entry-input",
                    type="number",
                    value=args.entry,
                    min=0,
                    max=max(n_entries - 1, 0),
                    placeholder="Entry index",
                    style={
                        "fontSize": "14px",
                        "padding": "8px",
                        "width": "120px",
                        "borderRadius": "5px",
                        "border": "1px solid #555",
                        "backgroundColor": "#333",
                        "color": "white",
                        "marginRight": "10px",
                    },
                ),
                html.Button(
                    "Go",
                    id="go-btn",
                    n_clicks=0,
                    style={
                        "fontSize": "16px",
                        "padding": "10px 20px",
                        "backgroundColor": "#2196F3",
                        "color": "white",
                        "border": "none",
                        "borderRadius": "5px",
                        "cursor": "pointer",
                    },
                ),
            ],
            style={"marginBottom": "10px"},
        ),
        html.Div(
            id="entry-label",
            style={"color": "#ccc", "fontSize": "13px", "marginBottom": "5px"},
        ),
        html.Div(
            f"Config: {os.path.basename(args.config)}  |  "
            f"gaussian_sigma={gaussian_sigma}  |  split={args.split}",
            style={"color": "#888", "fontSize": "12px", "marginBottom": "15px"},
        ),
        # ── Three-panel view ────────────────────────────────
        html.Div(
            [
                html.Div(
                    [dcc.Graph(id="mask-graph", config={"scrollZoom": True})],
                    style=GRAPH_STYLE,
                ),
                html.Div(
                    [dcc.Graph(id="distance-graph", config={"scrollZoom": True})],
                    style=GRAPH_STYLE,
                ),
                html.Div(
                    [dcc.Graph(id="gaussian-graph", config={"scrollZoom": True})],
                    style=GRAPH_STYLE,
                ),
            ]
        ),
        # ── Hidden store for current entry index ────────────
        dcc.Store(id="current-idx", data=args.entry),
    ],
)


@app.callback(
    [
        Output("mask-graph", "figure"),
        Output("distance-graph", "figure"),
        Output("gaussian-graph", "figure"),
        Output("entry-label", "children"),
        Output("current-idx", "data"),
    ],
    [
        Input("next-btn", "n_clicks"),
        Input("random-btn", "n_clicks"),
        Input("go-btn", "n_clicks"),
    ],
    [
        State("current-idx", "data"),
        State("entry-input", "value"),
    ],
)
def update_display(next_clicks, random_clicks, go_clicks,
                   current_idx, entry_input):
    ctx = dash.callback_context
    triggered = ctx.triggered[0]["prop_id"] if ctx.triggered else ""

    if "next-btn" in triggered:
        idx = (int(current_idx) + 1) % n_entries
    elif "random-btn" in triggered:
        idx = int(np.random.randint(0, n_entries))
    elif "go-btn" in triggered and entry_input is not None:
        idx = int(np.clip(entry_input, 0, n_entries - 1))
    else:
        # Initial load
        idx = int(current_idx) if current_idx is not None else 0

    # Process entry (retry on failure)
    for attempt in range(5):
        try:
            data = process_entry(dataset, idx, normalizer=normalizer)
            break
        except Exception as e:
            print(f"Error processing entry {idx}: {e}")
            import traceback
            traceback.print_exc()
            idx = (idx + 1) % n_entries
    else:
        return (
            empty_figure("Shower Mask (error)"),
            empty_figure("Origin Distance (error)"),
            empty_figure("Gaussian Target (error)"),
            f"Error: could not process entries around index {idx}",
            idx,
        )

    # Build figures
    mask_fig = build_shower_mask_figure(data, title="Shower Mask")
    dist_fig = build_origin_distance_figure(data, title="origin_distance (data loader, cm)")
    gauss_fig = build_gaussian_target_figure(
        data, gaussian_sigma, title="Gaussian Target (loss computation)"
    )

    # Info label
    origin_type_val = data.get("origin_type", -1)
    if hasattr(origin_type_val, "item"):
        origin_type_val = origin_type_val.item()
    elif isinstance(origin_type_val, np.ndarray):
        origin_type_val = int(origin_type_val.flat[0])
    origin_type_name = ORIGIN_TYPE_NAMES.get(int(origin_type_val), "unknown")

    mask = data.get("shower_mask")
    n_shower = int(mask.astype(bool).sum()) if mask is not None else 0
    n_total = data["coord"].shape[0]

    origin = data.get("origin_coord")
    origin_str = (
        f"({origin[0]:.3f}, {origin[1]:.3f}, {origin[2]:.3f})"
        if origin is not None else "N/A"
    )

    entry_label = (
        f"Entry: {idx} / {n_entries}  |  "
        f"Name: {data.get('name', '?')}  |  "
        f"Points: {n_total:,} (shower: {n_shower:,})  |  "
        f"Origin: {origin_str}  |  "
        f"Type: {origin_type_name}"
    )

    return mask_fig, dist_fig, gauss_fig, entry_label, idx


if __name__ == "__main__":
    app.run(debug=True, port=args.port)
