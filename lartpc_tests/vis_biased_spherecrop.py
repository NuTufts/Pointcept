"""
Visualization script for BiasedSphereCrop testing.

Displays cropped points colored by class, with all original points shown
in grey with low opacity for context. Provides a resample button to
generate new crops.

Usage:
    python vis_biased_spherecrop.py -i <h5_file> [-p <prob_random>] [-r <radius>]
"""

import os
import sys
import argparse
import h5py
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

# Add pointcept to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pointcept
from pointcept.datasets.transform import BiasedSphereCrop, GridSample

parser = argparse.ArgumentParser("Visualize BiasedSphereCrop sampling")
parser.add_argument("-i", "--input-h5", required=True, type=str, help="Input HDF5 file")
parser.add_argument("-p", "--prob-random", default=0.0, type=float,
                    help="Probability of random (vs biased) sampling (default: 0.0)")
parser.add_argument("-r", "--radius", default=50.0, type=float,
                    help="Gaussian offset radius in cm (default: 50.0)")
parser.add_argument("--point-max", default=50000, type=int,
                    help="Maximum points per crop (default: 50000)")
parser.add_argument("--point-min", default=10000, type=int,
                    help="Minimum points per crop (default: 10000)")
parser.add_argument("--grid-size", default=0.25, type=float,
                    help="Grid size for voxelization (default: 0.25)")
parser.add_argument("--pos-mode", default="reco", type=str, choices=["reco", "true"],
                    help="Position mode: reco or true (default: reco)")
parser.add_argument("--true-pts-only", default=False, action='store_true',
                    help="If given, remove ghosts and show true points only.")
args = parser.parse_args()

# Class definitions for LArTPC
# Class order: electron=0, muon=1, pion=2, proton=3, gamma=4, ghost=5
CLASS_NAMES = ["electron", "muon", "pion", "proton", "gamma", "ghost"]
CLASS_COLORS = {
    0: 'rgba(255,0,0,1.0)',      # electron - red
    1: 'rgba(0,0,255,1.0)',      # muon - blue
    2: 'rgba(255,165,0,1.0)',    # pion - orange
    3: 'rgba(0,255,0,1.0)',      # proton - green
    4: 'rgba(255,0,255,1.0)',    # gamma - magenta
    5: 'rgba(128,128,128,0.5)',  # ghost - grey, semi-transparent
    -1: 'rgba(200,200,200,0.3)', # unknown/other - light grey
}

# Keypoint type definitions
KPTYPE_COLOR = {
    0: "rgba(255,153,51,1.0)",  # nu - orange
    1: "rgba(255,0,0,1.0)",     # track start - red
    2: "rgba(0,0,255,1.0)",     # track end - blue
    3: "rgba(255,0,125,1.0)",   # shower start - pink
    4: "rgba(125,0,255,1.0)",   # michel start - purple
    5: "rgba(0,125,255,1.0)",   # delta start - cyan
}
KPTYPE_NAME = {
    0: 'Nu',
    1: "TrackStart",
    2: "TrackEnd",
    3: "Shower",
    4: "Michel",
    5: "Delta"
}

# PID to class mapping (from LArTPCDataset)
PID_TO_CLASS = {
    11: 0,   # electron
    -11: 0,  # positron
    13: 1,   # muon-
    -13: 1,  # muon+
    211: 2,  # pion+
    -211: 2, # pion-
    2212: 3, # proton
    22: 4,   # gamma
    0: 5,    # ghost (PID=0 means no particle)
}


def load_data_from_h5(filepath, pos_mode="reco", true_pts_only=False):
    """Load data from HDF5 file."""
    with h5py.File(filepath, 'r') as f:
        # Load coordinates
        coord = np.array(f['/entry_0/triplet_data/pos'], dtype=np.float32)

        # Load particle IDs and map to class indices
        pid = np.array(f['/entry_0/triplet_data/pid'], dtype=np.int64).flatten()
        segment = np.full(len(pid), fill_value=-1, dtype=np.int32)
        for pdg_code, class_idx in PID_TO_CLASS.items():
            segment[pid == pdg_code] = class_idx

        # Load pixel values as strength
        pixval = np.array(f['/entry_0/triplet_data/pixval'], dtype=np.float32)
        strength = (pixval / 500.0).astype(np.float32)

        # Load wire coordinates
        wire_feat = np.stack([
            f['/entry_0/triplet_data/uwire'][:],
            f['/entry_0/triplet_data/vwire'][:],
            f['/entry_0/triplet_data/ywire'][:]
        ], axis=1).astype(np.float32)
        wire_feat *= (1.0 / 3456.0)  # Normalize

        # Load keypoints for neutrino vertices
        keypoint_pos = np.array(f['/entry_0/mckeypoints/pos'], dtype=np.float32)
        keypoint_type = np.array(f['/entry_0/mckeypoints/kptype'], dtype=np.int64)
        nu_mask = keypoint_type == 0  # Nu keypoints
        nu_vertices = keypoint_pos[nu_mask[:], :]

        # Load all keypoints for visualization
        all_keypoints = {
            'pos': keypoint_pos,
            'kptype': keypoint_type
        }

        if true_pts_only:
            truthlabels = np.array(f['/entry_0/triplet_data/hasmatch'],dtype=np.int64)
            truthmask = truthlabels==1
            coord = coord[truthmask[:]]
            segment = segment[truthmask[:]]
            strength = strength[truthmask[:]]
            color = wire_feat[truthmask[:]]


    return {
        "coord": coord,
        "segment": segment,
        "strength": strength,
        "color": wire_feat,
        "nu_vertices": nu_vertices,
        "keypoints": all_keypoints,
    }


def create_biased_crop_transform(prob_random, radius, point_max, point_min, grid_size):
    """Create the transform pipeline with BiasedSphereCrop."""
    return [
        GridSample(
            grid_size=grid_size,
            hash_type="fnv",
            mode="train",
            return_grid_coord=True,
        ),
        BiasedSphereCrop(
            anchor_points_key="nu_vertices",
            anchor_pdf_key=None,
            radius=radius,
            point_max=point_max,
            point_min=point_min,
            prob_random=prob_random,
            max_retries=100,
            fallback_to_random=True,
        ),
    ]


def apply_transforms(data_dict, transforms):
    """Apply a list of transforms to data_dict."""
    import copy
    result = copy.deepcopy(data_dict)
    for t in transforms:
        result = t(result)
    return result


def sample_cropped_data(raw_data, transforms):
    """Apply transforms to get cropped data."""
    # Keep keypoints separate (not transformed)
    keypoints = raw_data.pop("keypoints", None)
    cropped = apply_transforms(raw_data, transforms)
    raw_data["keypoints"] = keypoints  # Restore
    return cropped


def downsample_for_context(coord, segment, max_points=100000):
    """Randomly downsample points for context display."""
    n_points = coord.shape[0]
    if n_points <= max_points:
        return coord, segment
    indices = np.random.choice(n_points, max_points, replace=False)
    return coord[indices], segment[indices]


# Load the data once at startup
print(f"Loading data from {args.input_h5}...")
raw_data = load_data_from_h5(args.input_h5, args.pos_mode, true_pts_only=args.true_pts_only)
print(f"Loaded {raw_data['coord'].shape[0]} points")
print(f"Nu vertices: {raw_data['nu_vertices'].shape[0]} found")

# Create transforms
transforms = create_biased_crop_transform(
    args.prob_random, args.radius, args.point_max, args.point_min, args.grid_size
)

# Downsample context points for display
context_coord, context_segment = downsample_for_context(
    raw_data["coord"], raw_data["segment"], max_points=100000
)

# Store keypoints for display
keypoints = raw_data["keypoints"]


def create_figure_data(cropped_data, context_coord, context_segment, keypoints, nu_vertices):
    """Create plotly figure data from cropped and context data."""
    traces = []

    # 1. Add context points (all points, grey, low opacity)
    bg_context_color='rgba(255,255,0,0.1)'
    traces.append({
        "type": "scatter3d",
        "x": context_coord[:, 0],
        "y": context_coord[:, 1],
        "z": context_coord[:, 2],
        "mode": "markers",
        "name": "All points (context)",
        "marker": {"color": bg_context_color, "size": 0.2},
        "hoverinfo": "skip",
    })

    # 2. Add cropped points colored by class
    cropped_coord = cropped_data["coord"]
    cropped_segment = cropped_data["segment"]

    for class_idx in range(len(CLASS_NAMES)):
        mask = cropped_segment == class_idx
        if not np.any(mask):
            continue

        n_pts = mask.sum()
        frac = n_pts / len(cropped_segment) * 100

        traces.append({
            "type": "scatter3d",
            "x": cropped_coord[mask, 0],
            "y": cropped_coord[mask, 1],
            "z": cropped_coord[mask, 2],
            "mode": "markers",
            "name": f"{CLASS_NAMES[class_idx]} ({n_pts:,}, {frac:.1f}%)",
            "marker": {"color": CLASS_COLORS[class_idx], "size": 2, "opacity": 0.8},
        })

    # Also show unknown/other class if present
    mask = cropped_segment == -1
    if np.any(mask):
        n_pts = mask.sum()
        frac = n_pts / len(cropped_segment) * 100
        traces.append({
            "type": "scatter3d",
            "x": cropped_coord[mask, 0],
            "y": cropped_coord[mask, 1],
            "z": cropped_coord[mask, 2],
            "mode": "markers",
            "name": f"other ({n_pts:,}, {frac:.1f}%)",
            "marker": {"color": CLASS_COLORS[-1], "size": 2, "opacity": 0.5},
        })

    # 3. Add keypoints
    for kptype in np.unique(keypoints['kptype']):
        mask = keypoints['kptype'] == kptype
        kp_pos = keypoints['pos'][mask, :]

        marker_size = 10 if kptype == 0 else 6  # Larger for nu vertices

        traces.append({
            "type": "scatter3d",
            "x": kp_pos[:, 0],
            "y": kp_pos[:, 1],
            "z": kp_pos[:, 2],
            "mode": "markers",
            "name": f"KP: {KPTYPE_NAME.get(kptype, f'type{kptype}')}",
            "marker": {
                "color": KPTYPE_COLOR.get(kptype, "rgba(255,255,255,1)"),
                "size": marker_size,
                "opacity": 1.0,
                "symbol": "diamond",
            },
        })

    return traces


# Initial crop
print("Generating initial crop...")
initial_cropped = sample_cropped_data(raw_data.copy(), transforms)
print(f"Cropped to {initial_cropped['coord'].shape[0]} points")

# Create Dash app
app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

axis_template = {
    "showbackground": True,
    "backgroundcolor": "#141414",
    "gridcolor": "rgb(255, 255, 255)",
    "zerolinecolor": "rgb(255, 255, 255)",
}

plot_layout = {
    "title": f"BiasedSphereCrop Visualization (prob_random={args.prob_random}, radius={args.radius})",
    "height": 800,
    "margin": {"t": 50, "b": 0, "l": 0, "r": 0},
    "font": {"size": 12, "color": "white"},
    "showlegend": True,
    "legend": {"x": 0.02, "y": 0.98, "bgcolor": "rgba(0,0,0,0.5)"},
    "plot_bgcolor": "#141414",
    "paper_bgcolor": "#141414",
    "scene": {
        "xaxis": axis_template,
        "yaxis": axis_template,
        "zaxis": axis_template,
        "aspectratio": {"x": 1, "y": 1, "z": 4},
        "camera": {"eye": {"x": 2, "y": 2, "z": 2}, "up": {"x": 0, "y": 1, "z": 0}},
    },
}

app.layout = html.Div([
    html.Div([
        html.Button("Resample", id="resample-btn", n_clicks=0,
                    style={"fontSize": "16px", "padding": "10px 20px", "margin": "10px"}),
        html.Span(id="sample-info", style={"color": "white", "marginLeft": "20px", "fontSize": "14px"}),
    ], style={"backgroundColor": "#141414", "padding": "10px"}),
    html.Div([
        dcc.Graph(
            id="det3d",
            figure={
                "data": create_figure_data(initial_cropped, context_coord, context_segment,
                                          keypoints, raw_data["nu_vertices"]),
                "layout": plot_layout,
            },
            config={"editable": True, "scrollZoom": False},
        )
    ], className="graph__container"),
], style={"backgroundColor": "#141414"})


@app.callback(
    [Output("det3d", "figure"), Output("sample-info", "children")],
    [Input("resample-btn", "n_clicks")],
    [State("det3d", "figure")]
)
def resample_callback(n_clicks, current_figure):
    """Resample when button is clicked."""
    # Generate new crop
    cropped = sample_cropped_data(raw_data.copy(), transforms)

    # Create new traces
    traces = create_figure_data(cropped, context_coord, context_segment,
                               keypoints, raw_data["nu_vertices"])

    # Preserve camera position if available
    layout = plot_layout.copy()
    if current_figure and "layout" in current_figure:
        if "scene" in current_figure["layout"] and "camera" in current_figure["layout"]["scene"]:
            layout["scene"]["camera"] = current_figure["layout"]["scene"]["camera"]

    # Compute class statistics
    segment = cropped["segment"]
    total = len(segment)
    stats = []
    for class_idx, name in enumerate(CLASS_NAMES):
        count = (segment == class_idx).sum()
        if count > 0:
            stats.append(f"{name}: {count:,} ({count/total*100:.1f}%)")

    info_text = f"Sample #{n_clicks+1} | Total: {total:,} pts | " + " | ".join(stats)

    return {"data": traces, "layout": layout}, info_text


if __name__ == "__main__":
    print(f"\nStarting visualization server...")
    print(f"Settings: prob_random={args.prob_random}, radius={args.radius}, "
          f"point_max={args.point_max}, point_min={args.point_min}")
    app.run(debug=True)
