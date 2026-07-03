"""
Visualize shower origin data for RECO (inference) HDF5 files.

Unlike the training visualizer (visualize_shower_origin.py), this script
is designed for reco data where truth labels (originpt, type) are
placeholders. It shows:

Row 1 (two panels, side by side):
  1. Selected Fragment — highlighted in red with start point marker
  2. Shower Score — all points colored by renormed_shower_score

Row 2 (full width):
  3. All Fragments — each DBSCAN fragment in a distinct color

Supports cycling through individual fragments within an event,
and cycling through events.

Reads HDF5 files directly (no config/dataset class needed), so it works
with the output of convert_larlite_to_showerorigin_h5.py without needing
the full Pointcept training environment.

Usage:
    python tools/viz_archive/visualize_shower_origin_reco.py \
        --data-list /path/to/filelist.txt

    python tools/viz_archive/visualize_shower_origin_reco.py \
        --input /path/to/single_event.h5

    python tools/viz_archive/visualize_shower_origin_reco.py \
        --input-dir /path/to/showerorigin_h5/
"""

import os
import sys
import argparse
import glob

import numpy as np
import h5py

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State

from detectoroutline import DetectorOutline


# ── Constants ────────────────────────────────────────────────────────────────

AXIS_TEMPLATE = {
    "showbackground": True,
    "backgroundcolor": "#141414",
    "gridcolor": "rgb(100, 100, 100)",
    "zerolinecolor": "rgb(200, 200, 200)",
}

# Distinct colors for fragment display
TYPE_NAMES = {0: "inside", 1: "outside", 2: "on_track", 3: "ghost", 4: "true_track", -1: "unknown"}

FRAGMENT_COLORS = [
    "rgba(255, 50, 50, 0.8)",    # red
    "rgba(50, 150, 255, 0.8)",   # blue
    "rgba(50, 255, 50, 0.8)",    # green
    "rgba(255, 200, 50, 0.8)",   # yellow
    "rgba(200, 50, 255, 0.8)",   # purple
    "rgba(255, 128, 0, 0.8)",    # orange
    "rgba(0, 255, 200, 0.8)",    # teal
    "rgba(255, 50, 200, 0.8)",   # pink
    "rgba(128, 255, 128, 0.8)",  # light green
    "rgba(128, 128, 255, 0.8)",  # light blue
    "rgba(255, 255, 128, 0.8)",  # light yellow
    "rgba(255, 128, 128, 0.8)",  # light red
]


# ── Plotly helpers ───────────────────────────────────────────────────────────

def make_layout(title, npts, height=550, camera_up=None, aspect_ratio=None,
                axis_labels=None):
    scene = {
        "xaxis": {**AXIS_TEMPLATE},
        "yaxis": {**AXIS_TEMPLATE},
        "zaxis": {**AXIS_TEMPLATE},
        "aspectmode": "data" if aspect_ratio is None else "manual",
        "camera": {
            "eye": {"x": 2, "y": 2, "z": 2},
            "up": camera_up or {"x": 0, "y": 1, "z": 0},
        },
    }
    if aspect_ratio is not None:
        scene["aspectratio"] = aspect_ratio
    if axis_labels is not None:
        for ax, label in zip(["xaxis", "yaxis", "zaxis"], axis_labels):
            scene[ax]["title"] = label
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
        "scene": scene,
    }


def empty_figure(title="N/A"):
    return {"data": [], "layout": make_layout(title, 0)}


def add_detector_outline(traces):
    """Add MicroBooNE detector outline to a trace list."""
    detdata = DetectorOutline()
    det_traces = detdata.getlines(color=(255, 255, 255))
    for det_trace in det_traces:
        traces.append({
            "type": "scatter3d",
            "x": det_trace["x"],
            "y": det_trace["y"],
            "z": det_trace["z"],
            "mode": "lines",
            "name": "Detector",
            "line": det_trace["line"],
            "showlegend": False,
        })


# ── Argument parsing ────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize shower origin reco data (HDF5 files)"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--data-list", type=str,
        help="Text file listing HDF5 file paths (one per line)."
    )
    group.add_argument(
        "--input", type=str,
        help="Single HDF5 file to visualize."
    )
    group.add_argument(
        "--input-dir", type=str,
        help="Directory of HDF5 files to visualize."
    )
    parser.add_argument(
        "-e", "--entry", default=0, type=int,
        help="Initial file index (default: 0)."
    )
    parser.add_argument(
        "--port", default=8050, type=int,
        help="Dash server port."
    )
    parser.add_argument(
        "--min-fragment-points", default=20, type=int,
        help="Minimum points per fragment to display (default: 20)."
    )
    return parser.parse_args()


# ── Data loading ────────────────────────────────────────────────────────────

def get_file_list(args):
    """Build sorted list of HDF5 file paths from CLI args."""
    if args.input:
        return [args.input]
    elif args.input_dir:
        files = sorted(glob.glob(os.path.join(args.input_dir, "*.h5")))
        files += sorted(glob.glob(os.path.join(args.input_dir, "*.hdf5")))
        return files
    elif args.data_list:
        files = []
        with open(args.data_list, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    files.append(line)
        return sorted(files)
    return []


def load_event(filepath, min_fragment_points=20):
    """
    Load a single event HDF5 file produced by
    convert_larlite_to_showerorigin_h5.py.

    Returns dict with:
        pos: (N, 3) coordinates
        shower_score: (N,) or None
        lm_score: (N,) or None
        fragments: list of dicts with 'mask', 'startpt', 'npts'
        filename: basename
    """
    with h5py.File(filepath, 'r') as f:
        entry = f['entry_0']
        triplet = entry['triplet_data']

        pos = triplet['pos'][:].astype(np.float32)
        n_points = pos.shape[0]

        # Load optional fields
        shower_score = None
        if 'shower_score' in triplet:
            shower_score = triplet['shower_score'][:].astype(np.float32)

        lm_score = None
        if 'lm_score' in triplet:
            lm_score = triplet['lm_score'][:].astype(np.float32)

        pixval = None
        if 'pixval' in triplet:
            pixval = triplet['pixval'][:].astype(np.float32)

        # Load fragments
        fragments = []
        if 'shower_fragments' in entry:
            sf = entry['shower_fragments']
            num_frags = int(sf.attrs.get('num_fragments', 0))

            if num_frags > 0 and 'pointindices_flat' in sf:
                flat_indices = sf['pointindices_flat'][:]
                index_counts = sf['pointindices_counts'][:]
                startpts = sf['startpt'][:] if 'startpt' in sf else None
                originpts = sf['originpt'][:] if 'originpt' in sf else None
                frag_types = sf['type'][:] if 'type' in sf else None

                offset = 0
                for i in range(num_frags):
                    count = int(index_counts[i])
                    indices = flat_indices[offset:offset + count]
                    offset += count

                    if count < min_fragment_points:
                        continue

                    mask = np.zeros(n_points, dtype=bool)
                    valid = indices[indices < n_points]
                    mask[valid] = True
                    actual_npts = int(mask.sum())

                    if actual_npts < min_fragment_points:
                        continue

                    frag = {
                        "mask": mask,
                        "npts": actual_npts,
                        "index": i,
                    }
                    if startpts is not None:
                        frag["startpt"] = startpts[i].astype(np.float32)
                    if originpts is not None:
                        frag["originpt"] = originpts[i].astype(np.float32)
                    if frag_types is not None:
                        frag["type"] = int(frag_types[i])

                    fragments.append(frag)

    return {
        "pos": pos,
        "shower_score": shower_score,
        "lm_score": lm_score,
        "pixval": pixval,
        "fragments": fragments,
        "filename": os.path.basename(filepath),
    }


# ── Figure builders ─────────────────────────────────────────────────────────

# Common layout kwargs for the two top-row panels (raw cm coords)
_TOP_LAYOUT = dict(
    camera_up={"x": 0, "y": 1, "z": 0},
    aspect_ratio={"x": 1, "y": 1, "z": 4},
    axis_labels=["X (cm)", "Y (cm)", "Z (cm)"],
)


def build_fragment_mask_figure(data, frag_idx=0,
                               title="Selected Fragment"):
    """
    Row 1, Panel 1: Show selected fragment highlighted, rest in gray.
    Start point marker shown as cyan cross. Detector outline included.
    """
    pos = data["pos"]
    frags = data["fragments"]
    npts = pos.shape[0]
    traces = []

    selected_mask = None
    selected_frag = None
    if frags and 0 <= frag_idx < len(frags):
        selected_frag = frags[frag_idx]
        selected_mask = selected_frag["mask"]

    if selected_mask is not None:
        non_shower = ~selected_mask

        if non_shower.sum() > 0:
            traces.append({
                "type": "scatter3d",
                "x": pos[non_shower, 0].tolist(),
                "y": pos[non_shower, 1].tolist(),
                "z": pos[non_shower, 2].tolist(),
                "mode": "markers",
                "name": f"other ({int(non_shower.sum()):,})",
                "marker": {"color": "rgba(120,120,120,0.2)", "size": 1},
                "hoverinfo": "skip",
            })

        if selected_mask.sum() > 0:
            traces.append({
                "type": "scatter3d",
                "x": pos[selected_mask, 0].tolist(),
                "y": pos[selected_mask, 1].tolist(),
                "z": pos[selected_mask, 2].tolist(),
                "mode": "markers",
                "name": f"fragment {frag_idx} ({int(selected_mask.sum()):,})",
                "marker": {"color": "rgba(255,50,50,0.8)", "size": 3},
                "hovertemplate": (
                    "<b>x</b>: %{x:.1f}<br>"
                    "<b>y</b>: %{y:.1f}<br>"
                    "<b>z</b>: %{z:.1f}<extra></extra>"
                ),
            })

        # Start point marker (magenta cross)
        if selected_frag and "startpt" in selected_frag:
            sp = selected_frag["startpt"]
            traces.append({
                "type": "scatter3d",
                "x": [float(sp[0])],
                "y": [float(sp[1])],
                "z": [float(sp[2])],
                "mode": "markers",
                "name": "start pt",
                "marker": {
                    "color": "magenta", "size": 10,
                    "symbol": "cross",
                    "line": {"color": "white", "width": 2},
                },
                "hovertemplate": (
                    "<b>START</b><br>"
                    "x: %{x:.1f}<br>y: %{y:.1f}<br>z: %{z:.1f}"
                    "<extra></extra>"
                ),
            })

        # Origin point marker (cyan diamond)
        if selected_frag and "originpt" in selected_frag:
            op = selected_frag["originpt"]
            # Only show if origin is non-zero (not a placeholder)
            if np.any(op != 0):
                traces.append({
                    "type": "scatter3d",
                    "x": [float(op[0])],
                    "y": [float(op[1])],
                    "z": [float(op[2])],
                    "mode": "markers",
                    "name": "origin pt",
                    "marker": {
                        "color": "cyan", "size": 12,
                        "symbol": "diamond",
                        "line": {"color": "white", "width": 2},
                    },
                    "hovertemplate": (
                        "<b>ORIGIN</b><br>"
                        "x: %{x:.1f}<br>y: %{y:.1f}<br>z: %{z:.1f}"
                        "<extra></extra>"
                    ),
                })
    else:
        traces.append({
            "type": "scatter3d",
            "x": pos[:, 0].tolist(),
            "y": pos[:, 1].tolist(),
            "z": pos[:, 2].tolist(),
            "mode": "markers",
            "name": f"all ({npts:,})",
            "marker": {"color": "rgba(120,120,120,0.4)", "size": 2},
        })

    add_detector_outline(traces)

    frag_label = (f" (frag {frag_idx}/{len(frags)}, "
                  f"{selected_frag['npts']} pts)"
                  if selected_frag else " (no fragments)")

    return {"data": traces,
            "layout": make_layout(title + frag_label, npts, **_TOP_LAYOUT)}


def build_shower_score_figure(data, title="Shower Score"):
    """
    Row 1, Panel 2: All points colored by renormed_shower_score.
    Detector outline included.
    """
    pos = data["pos"]
    score = data.get("shower_score")
    npts = pos.shape[0]
    traces = []

    if score is not None:
        traces.append({
            "type": "scatter3d",
            "x": pos[:, 0].tolist(),
            "y": pos[:, 1].tolist(),
            "z": pos[:, 2].tolist(),
            "mode": "markers",
            "name": f"all ({npts:,})",
            "marker": {
                "color": score.tolist(),
                "colorscale": "Turbo",
                "size": 2,
                "opacity": 0.7,
                "cmin": 0.0,
                "cmax": 1.0,
                "colorbar": {"title": "shower score", "len": 0.6},
            },
            "customdata": np.stack([score], axis=1),
            "hovertemplate": (
                "<b>x</b>: %{x:.1f}<br>"
                "<b>y</b>: %{y:.1f}<br>"
                "<b>z</b>: %{z:.1f}<br>"
                "<b>shower score</b>: %{customdata[0]:.3f}"
                "<extra></extra>"
            ),
        })
    else:
        # Fallback: color by z coordinate
        traces.append({
            "type": "scatter3d",
            "x": pos[:, 0].tolist(),
            "y": pos[:, 1].tolist(),
            "z": pos[:, 2].tolist(),
            "mode": "markers",
            "name": f"all ({npts:,})",
            "marker": {"color": pos[:, 2].tolist(), "size": 2,
                        "colorscale": "Viridis"},
        })

    add_detector_outline(traces)

    return {"data": traces, "layout": make_layout(title, npts, **_TOP_LAYOUT)}


def build_all_fragments_figure(data, title="All Fragments"):
    """
    Row 2 (full width): Each fragment in a distinct color.
    Non-fragment points in gray. Start points as labeled markers.
    Detector outline included.
    """
    pos = data["pos"]
    frags = data["fragments"]
    npts = pos.shape[0]
    traces = []

    # Build union mask of all fragment points
    any_frag = np.zeros(npts, dtype=bool)
    for frag in frags:
        any_frag |= frag["mask"]

    non_frag = ~any_frag
    if non_frag.sum() > 0:
        traces.append({
            "type": "scatter3d",
            "x": pos[non_frag, 0].tolist(),
            "y": pos[non_frag, 1].tolist(),
            "z": pos[non_frag, 2].tolist(),
            "mode": "markers",
            "name": f"non-shower ({int(non_frag.sum()):,})",
            "marker": {"color": "rgba(120,120,120,0.15)", "size": 1},
            "hoverinfo": "skip",
        })

    # Each fragment in a distinct color
    for fi, frag in enumerate(frags):
        mask = frag["mask"]
        if mask.sum() == 0:
            continue
        color = FRAGMENT_COLORS[fi % len(FRAGMENT_COLORS)]
        traces.append({
            "type": "scatter3d",
            "x": pos[mask, 0].tolist(),
            "y": pos[mask, 1].tolist(),
            "z": pos[mask, 2].tolist(),
            "mode": "markers",
            "name": f"frag {fi} ({frag['npts']})",
            "marker": {"color": color, "size": 2},
            "hovertemplate": (
                f"<b>frag {fi}</b><br>"
                "<b>x</b>: %{x:.1f}<br>"
                "<b>y</b>: %{y:.1f}<br>"
                "<b>z</b>: %{z:.1f}<extra></extra>"
            ),
        })

    # Label color by fragment type
    TYPE_LABEL_COLORS = {0: "red", 1: "lime", 2: "dodgerblue", 4: "yellow"}

    # Start point markers for all fragments
    for fi, frag in enumerate(frags):
        if "startpt" not in frag:
            continue
        sp = frag["startpt"]
        ftype = frag.get("type", -1)
        label_color = TYPE_LABEL_COLORS.get(ftype, "white")
        traces.append({
            "type": "scatter3d",
            "x": [float(sp[0])],
            "y": [float(sp[1])],
            "z": [float(sp[2])],
            "mode": "markers+text",
            "name": f"start {fi}",
            "text": [f"S{fi}"],
            "textposition": "top center",
            "textfont": {"color": label_color, "size": 10},
            "marker": {
                "color": label_color, "size": 8,
                "symbol": "cross",
                "line": {"color": "black", "width": 1},
            },
            "hovertemplate": (
                f"<b>Start {fi}</b><br>"
                "x: %{x:.1f}<br>y: %{y:.1f}<br>z: %{z:.1f}"
                "<extra></extra>"
            ),
            "showlegend": False,
        })

    add_detector_outline(traces)

    frag_label = f" ({len(frags)} fragments, "
    frag_label += f"{int(any_frag.sum()):,} shower pts)"

    return {"data": traces,
            "layout": make_layout(
                title + frag_label, npts, height=650,
                camera_up={"x": 0, "y": 1, "z": 0},
                aspect_ratio={"x": 1, "y": 1, "z": 4},
                axis_labels=["X (cm)", "Y (cm)", "Z (cm)"],
            )}


# ── Main ────────────────────────────────────────────────────────────────────

args = parse_args()
file_list = get_file_list(args)

if not file_list:
    print("ERROR: No HDF5 files found.")
    sys.exit(1)

n_files = len(file_list)
print(f"Found {n_files} HDF5 files.")

# Preload first event to verify
test_data = load_event(file_list[0], args.min_fragment_points)
print(f"  File 0: {test_data['filename']}, "
      f"{test_data['pos'].shape[0]} hits, "
      f"{len(test_data['fragments'])} fragments")


# ── Dash app ────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ],
)

BTN_STYLE = {
    "fontSize": "16px",
    "padding": "10px 20px",
    "color": "white",
    "border": "none",
    "borderRadius": "5px",
    "cursor": "pointer",
    "marginRight": "10px",
}

GRAPH_STYLE_HALF = {
    "width": "50%",
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
        # ── Header: event navigation ────────────────────────
        html.Div([
            html.H2(
                "Shower Origin Reco Visualizer",
                style={"color": "white", "display": "inline-block",
                       "marginRight": "20px"},
            ),
            html.Button("Next Event", id="next-event-btn", n_clicks=0,
                         style={**BTN_STYLE, "backgroundColor": "#4CAF50"}),
            html.Button("Random", id="random-event-btn", n_clicks=0,
                         style={**BTN_STYLE, "backgroundColor": "#FF9800"}),
            dcc.Input(
                id="event-input", type="number", value=args.entry,
                min=0, max=max(n_files - 1, 0), placeholder="Event index",
                style={
                    "fontSize": "14px", "padding": "8px", "width": "120px",
                    "borderRadius": "5px", "border": "1px solid #555",
                    "backgroundColor": "#333", "color": "white",
                    "marginRight": "10px",
                },
            ),
            html.Button("Go", id="go-event-btn", n_clicks=0,
                         style={**BTN_STYLE, "backgroundColor": "#2196F3"}),
        ], style={"marginBottom": "10px"}),

        # ── Fragment dropdown ───────────────────────────────
        html.Div([
            html.Label("Fragment: ",
                        style={"color": "white", "marginRight": "10px",
                               "fontSize": "14px"}),
            dcc.Dropdown(
                id="fragment-dropdown",
                options=[],
                value=None,
                style={
                    "width": "500px", "display": "inline-block",
                    "verticalAlign": "middle",
                },
            ),
        ], style={"marginBottom": "10px", "display": "flex",
                  "alignItems": "center"}),

        # ── Info bar ────────────────────────────────────────
        html.Div(id="info-label",
                  style={"color": "#ccc", "fontSize": "13px",
                         "marginBottom": "5px"}),
        html.Div(
            f"min_fragment_points={args.min_fragment_points}",
            style={"color": "#888", "fontSize": "12px",
                   "marginBottom": "15px"},
        ),

        # ── Row 1: two panels side by side ──────────────────
        html.Div([
            html.Div(
                [dcc.Graph(id="mask-graph", config={"scrollZoom": True})],
                style=GRAPH_STYLE_HALF,
            ),
            html.Div(
                [dcc.Graph(id="score-graph", config={"scrollZoom": True})],
                style=GRAPH_STYLE_HALF,
            ),
        ]),

        # ── Row 2: all fragments (full width) ──────────────
        html.Div([
            dcc.Graph(id="frags-graph", config={"scrollZoom": True}),
        ]),

        # ── Hidden stores ──────────────────────────────────
        dcc.Store(id="current-event-idx", data=args.entry),
        dcc.Store(id="event-cache", data=None),
    ],
)


# ── Callback 1: Event change — load data, cache, populate dropdown ──────────

@app.callback(
    [
        Output("event-cache", "data"),
        Output("fragment-dropdown", "options"),
        Output("fragment-dropdown", "value"),
        Output("current-event-idx", "data"),
        Output("info-label", "children"),
    ],
    [
        Input("next-event-btn", "n_clicks"),
        Input("random-event-btn", "n_clicks"),
        Input("go-event-btn", "n_clicks"),
    ],
    [
        State("current-event-idx", "data"),
        State("event-input", "value"),
    ],
)
def on_event_change(next_clicks, random_clicks, go_clicks,
                    current_idx, event_input):
    ctx = dash.callback_context
    triggered = ctx.triggered[0]["prop_id"] if ctx.triggered else ""

    if "next-event" in triggered:
        idx = (int(current_idx) + 1) % n_files
    elif "random-event" in triggered:
        idx = int(np.random.randint(0, n_files))
    elif "go-event" in triggered and event_input is not None:
        idx = int(np.clip(event_input, 0, n_files - 1))
    else:
        idx = int(current_idx) if current_idx is not None else 0

    # Load event data
    try:
        data = load_event(file_list[idx], args.min_fragment_points)
    except Exception as e:
        print(f"Error loading {file_list[idx]}: {e}")
        return None, [], None, idx, f"Error loading event {idx}: {e}"

    n_frags = len(data["fragments"])

    # Build dropdown options
    options = []
    for i, frag in enumerate(data["fragments"]):
        ftype = frag.get("type", -1)
        type_name = TYPE_NAMES.get(ftype, f"type{ftype}")
        label = f"Fragment {i}: {frag['npts']} pts, {type_name}"
        options.append({"label": label, "value": i})

    # Serialize for dcc.Store (numpy arrays -> lists)
    cache = {
        "pos": data["pos"].tolist(),
        "shower_score": data["shower_score"].tolist()
                        if data["shower_score"] is not None else None,
        "fragments": [
            {
                "mask": f["mask"].tolist(),
                "npts": f["npts"],
                "index": f["index"],
                "startpt": f["startpt"].tolist()
                           if "startpt" in f else None,
                "originpt": f["originpt"].tolist()
                            if "originpt" in f else None,
                "type": f.get("type", -1),
            }
            for f in data["fragments"]
        ],
        "filename": data["filename"],
    }

    # Info
    n_total = data["pos"].shape[0]
    n_shower = 0
    if data["shower_score"] is not None:
        n_shower = int((data["shower_score"] >= 0.5).sum())
    frag_sizes = [f["npts"] for f in data["fragments"]]

    info = (
        f"Event: {idx}/{n_files}  |  "
        f"File: {data['filename']}  |  "
        f"Hits: {n_total:,}  |  "
        f"Shower hits (score>=0.5): {n_shower:,}  |  "
        f"Fragments: {n_frags} {frag_sizes}"
    )

    default_value = 0 if n_frags > 0 else None
    return cache, options, default_value, idx, info


# ── Callback 2: Fragment selection — build figures from cache ────────────────

@app.callback(
    [
        Output("mask-graph", "figure"),
        Output("score-graph", "figure"),
        Output("frags-graph", "figure"),
    ],
    [Input("fragment-dropdown", "value")],
    [State("event-cache", "data")],
)
def on_fragment_select(frag_idx, cache):
    if cache is None:
        return (
            empty_figure("Selected Fragment"),
            empty_figure("Shower Score"),
            empty_figure("All Fragments"),
        )

    # Deserialize
    data = {
        "pos": np.array(cache["pos"], dtype=np.float32),
        "shower_score": np.array(cache["shower_score"], dtype=np.float32)
                        if cache["shower_score"] is not None else None,
        "fragments": [
            {
                "mask": np.array(f["mask"], dtype=bool),
                "npts": f["npts"],
                "index": f["index"],
                "startpt": np.array(f["startpt"], dtype=np.float32)
                           if f["startpt"] is not None else None,
                "originpt": np.array(f["originpt"], dtype=np.float32)
                            if f.get("originpt") is not None else None,
                "type": f.get("type", -1),
            }
            for f in cache["fragments"]
        ],
        "filename": cache["filename"],
    }

    fr_idx = int(frag_idx) if frag_idx is not None else 0

    mask_fig = build_fragment_mask_figure(data, fr_idx,
                                          title="Selected Fragment")
    score_fig = build_shower_score_figure(data, title="Shower Score")
    frags_fig = build_all_fragments_figure(data, title="All Fragments")

    return mask_fig, score_fig, frags_fig


if __name__ == "__main__":
    app.run(debug=True, port=args.port)
