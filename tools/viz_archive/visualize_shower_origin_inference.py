"""
Visualize Shower Origin model inference results.

Runs inference on all shower fragments in an event, visualizes the predicted
origin scores per fragment, and compares with ground truth.

Three panels:
  1. Predicted Scores — all points colored by model origin_scores (0-1 heatmap)
  2. Shower Mask + GT — shower points red, non-shower gray, GT markers
  3. Classification — GT class vs predicted class with probabilities

Uses two callbacks to avoid re-running inference on fragment switch:
  - Event change: runs inference, caches all fragment results in dcc.Store
  - Fragment selection: reads cache, builds figures for selected fragment

Usage:
    python tools/viz_archive/visualize_shower_origin_inference.py \
        -c configs/lartpc/shower_origin/archive/shower-origin-sonata-v1m1-v3.py \
        --checkpoint path/to/model_epoch_90.pth \
        --data-list shower_origin_single_event_test.txt
"""

import os
import sys
import argparse

import numpy as np
import torch

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State

from pointcept.utils.config import Config
from pointcept.datasets.builder import DATASETS
from pointcept.datasets.transform import Compose, TRANSFORMS
from pointcept.datasets.utils import point_collate_fn
from pointcept.models import build_model
from detectoroutline import DetectorOutline


# ── Constants ────────────────────────────────────────────────────────────────

ORIGIN_TYPE_NAMES = {0: "inside", 1: "outside", 2: "on_track", -1: "none"}

AXIS_TEMPLATE = {
    "showbackground": True,
    "backgroundcolor": "#141414",
    "gridcolor": "rgb(100, 100, 100)",
    "zerolinecolor": "rgb(200, 200, 200)",
}


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


# ── Argument parsing ─────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize shower origin model inference"
    )
    parser.add_argument(
        "-c", "--config", required=True, type=str, help="Config file path"
    )
    parser.add_argument(
        "--checkpoint", required=True, type=str, help="Model checkpoint path"
    )
    parser.add_argument(
        "-e", "--entry", default=0, type=int, help="Initial entry index"
    )
    parser.add_argument(
        "--data-list", type=str, default=None,
        help="Override data list file path"
    )
    parser.add_argument(
        "--split", type=str, default="val", choices=["train", "val"],
        help="Dataset split (default: val)"
    )
    parser.add_argument(
        "--port", default=8051, type=int, help="Dash server port"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device for inference (default: cuda if available, else cpu)"
    )
    return parser.parse_args()


# ── Build dataset (no transforms) ───────────────────────────────────────────

def build_dataset(cfg, split, data_list_override=None):
    dataset_cfg = getattr(cfg.data, split).copy()
    if data_list_override is not None:
        dataset_cfg["data_list_file"] = os.path.abspath(data_list_override)
    dataset_cfg["transform"] = []
    dataset = DATASETS.build(dataset_cfg)
    return dataset


# ── Build inference transforms ───────────────────────────────────────────────

def build_inference_transforms(cfg, split):
    """Build the full val transform pipeline from config."""
    dataset_cfg = getattr(cfg.data, split)
    transform_list = dataset_cfg.get("transform", [])
    if not transform_list:
        return None
    # Build transforms manually and assign to Compose
    # (Compose.__init__ expects config dicts, not built objects)
    comp = Compose(cfg=None)
    comp.transforms = [TRANSFORMS.build(t) for t in transform_list]
    return comp


# ── Load model ───────────────────────────────────────────────────────────────

def load_model(cfg, checkpoint_path, device):
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # Remove "module." prefix (from DDP training)
    state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}

    model = build_model(cfg.model)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        print(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
    print("Model loaded successfully")

    model.to(device).eval()
    return model


# ── Inference ────────────────────────────────────────────────────────────────

def run_event_inference(dataset, event_idx, transform, model, device):
    """
    Run inference on all fragments in an event.

    Returns list of dicts, one per fragment, each containing:
        - coord: (N, 3) raw coords (for display)
        - shower_mask: (N,) float
        - origin_coord: (3,) GT origin
        - origin_type: int GT type
        - start_coord: (3,) GT start
        - name: str
        - pred_scores: (M,) model origin scores (M = post-transform points)
        - pred_coords: (M, 3) model coords
        - pred_class: int predicted class
        - pred_class_logits: (num_classes,) raw logits
        - n_points_raw: int pre-transform point count
        - n_points_model: int post-transform point count
    """
    raw_fragments = dataset.get_all_fragments(event_idx)
    if not raw_fragments:
        return []

    # Save GT metadata and apply transforms
    gt_metadata = []
    transformed = []
    for frag in raw_fragments:
        gt_metadata.append({
            "coord": frag["coord"].copy(),
            "shower_mask": frag["shower_mask"].copy(),
            "origin_coord": frag["origin_coord"].copy(),
            "origin_type": int(frag["origin_type"]),
            "start_coord": frag["start_coord"].copy(),
            "name": frag["name"],
            "n_points_raw": frag["coord"].shape[0],
        })
        if transform is not None:
            t_frag = transform(frag)
        else:
            t_frag = frag
        transformed.append(t_frag)

    # Batch all fragments
    batched = point_collate_fn(transformed)

    # Move tensors to device
    for k, v in batched.items():
        if isinstance(v, torch.Tensor):
            batched[k] = v.to(device)

    # Forward pass
    with torch.no_grad():
        output = model(batched)

    # Unpack results per fragment using offset
    offset = batched["offset"].cpu()
    n_frags = len(offset)

    # Handle single vs multi-sample outputs
    if n_frags == 1:
        all_scores = [output["origin_scores"].cpu()]
        all_coords = [output["all_coords"].cpu()]
        all_classes = [output["origin_class"].cpu().item()]
        all_logits = [output["origin_class_logits"].cpu()]
    else:
        # Multi-sample: origin_scores and all_coords are lists
        if isinstance(output["origin_scores"], list):
            all_scores = [s.cpu() for s in output["origin_scores"]]
            all_coords = [c.cpu() for c in output["all_coords"]]
        else:
            # Split using offset
            scores_cat = output["origin_scores"].cpu()
            coords_cat = output["all_coords"].cpu()
            all_scores = []
            all_coords = []
            start = 0
            for i in range(n_frags):
                end = int(offset[i])
                all_scores.append(scores_cat[start:end])
                all_coords.append(coords_cat[start:end])
                start = end

        cls_tensor = output["origin_class"].cpu()
        logits_tensor = output["origin_class_logits"].cpu()
        all_classes = [cls_tensor[i].item() for i in range(n_frags)]
        all_logits = [logits_tensor[i] for i in range(n_frags)]

    # Combine GT + predictions
    results = []
    for i in range(n_frags):
        gt = gt_metadata[i]
        scores_np = all_scores[i].float().numpy()
        coords_np = all_coords[i].float().numpy()
        logits_np = all_logits[i].float().numpy()

        results.append({
            **gt,
            "pred_scores": scores_np,
            "pred_coords": coords_np,
            "pred_class": all_classes[i],
            "pred_class_logits": logits_np,
            "n_points_model": coords_np.shape[0],
        })

    return results


# ── Serialization helpers for dcc.Store ──────────────────────────────────────

def results_to_json(results):
    """Convert results list to JSON-serializable format."""
    out = []
    for r in results:
        d = {}
        for k, v in r.items():
            if isinstance(v, np.ndarray):
                d[k] = v.tolist()
            else:
                d[k] = v
        out.append(d)
    return out


def results_from_json(data):
    """Convert JSON data back to numpy arrays."""
    out = []
    for d in data:
        r = {}
        for k, v in d.items():
            if k in ("coord", "shower_mask", "origin_coord", "start_coord",
                      "pred_scores", "pred_coords", "pred_class_logits"):
                r[k] = np.array(v, dtype=np.float32)
            else:
                r[k] = v
        out.append(r)
    return out


# ── Figure builders ──────────────────────────────────────────────────────────

def build_predicted_scores_figure(result, gaussian_sigma=None,
                                  coord_center=None, coord_scale=None):
    """Points colored by model predicted origin_scores (0-1 heatmap).

    GT origin/start are in raw cm coords; pred_coords are in normalized space.
    Apply the same normalization to GT points before plotting.
    """
    coord = result["pred_coords"]
    scores = result["pred_scores"]
    origin = np.array(result["origin_coord"], dtype=np.float32)
    npts = coord.shape[0]

    # Normalize GT coords to match model's coordinate frame
    if coord_center is not None and coord_scale is not None:
        center = np.array(coord_center, dtype=np.float32)
        origin = (origin - center) / coord_scale

    # Find predicted peak
    peak_idx = int(np.argmax(scores))
    peak_coord = coord[peak_idx]

    traces = []

    traces.append({
        "type": "scatter3d",
        "x": coord[:, 0].tolist(),
        "y": coord[:, 1].tolist(),
        "z": coord[:, 2].tolist(),
        "mode": "markers",
        "name": f"scores ({npts:,})",
        "marker": {
            "color": scores.tolist(),
            "colorscale": "Turbo",
            "size": 2,
            "opacity": 0.9,
            "cmin": 0,
            "cmax": max(float(scores.max()), 0.01),
            "colorbar": {"title": "score", "len": 0.6},
        },
        "customdata": np.stack([scores], axis=1),
        "hovertemplate": (
            "<b>x</b>: %{x:.3f}<br>"
            "<b>y</b>: %{y:.3f}<br>"
            "<b>z</b>: %{z:.3f}<br>"
            "<b>score</b>: %{customdata[0]:.4f}"
            "<extra></extra>"
        ),
    })

    # GT origin marker (green diamond)
    traces.append({
        "type": "scatter3d",
        "x": [float(origin[0])],
        "y": [float(origin[1])],
        "z": [float(origin[2])],
        "mode": "markers",
        "name": "GT origin",
        "marker": {
            "color": "rgba(0,0,0,0)",
            "size": 12,
            "symbol": "circle-open",
            "line": {"color": "lime", "width": 3},
        },
        "hovertemplate": (
            "<b>GT ORIGIN</b><br>"
            "x: %{x:.3f}<br>y: %{y:.3f}<br>z: %{z:.3f}"
            "<extra></extra>"
        ),
    })

    # Predicted peak marker (yellow star)
    traces.append({
        "type": "scatter3d",
        "x": [float(peak_coord[0])],
        "y": [float(peak_coord[1])],
        "z": [float(peak_coord[2])],
        "mode": "markers",
        "name": f"pred peak (score={scores[peak_idx]:.3f})",
        "marker": {
            "color": "yellow",
            "size": 10,
            "symbol": "cross",
            "line": {"color": "black", "width": 2},
        },
        "hovertemplate": (
            "<b>PREDICTED PEAK</b><br>"
            f"score: {scores[peak_idx]:.4f}<br>"
            "x: %{x:.3f}<br>y: %{y:.3f}<br>z: %{z:.3f}"
            "<extra></extra>"
        ),
    })

    # If we have gaussian_sigma, also show gaussian target for comparison
    if gaussian_sigma is not None:
        gt_dist = np.linalg.norm(coord - origin[np.newaxis, :], axis=1)
        gt_score = np.exp(-0.5 * (gt_dist / gaussian_sigma) ** 2)
        # Show as a faint overlay — just the GT origin ring
        # (We don't add another full scatter, just note in the title)

    title = "Predicted Scores"
    dist_to_gt = np.linalg.norm(peak_coord - origin)
    title += f" | peak-GT dist={dist_to_gt:.4f}"

    return {"data": traces, "layout": make_layout(
        title, npts, camera_up={"x": 0, "y": 1, "z": 0})}


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


def build_shower_mask_figure(all_results, selected_idx=0):
    """All fragments shown with distinct colors + labeled text markers.

    Uses raw coords from the first result (all fragments share the same
    event point cloud). Each fragment's shower mask is shown in a distinct
    color, with a text label at its centroid showing the fragment number.
    The selected fragment is highlighted with larger markers.
    """
    if not all_results:
        return empty_figure("All Fragments (no data)")

    # All fragments share the same raw coord array
    coord = all_results[0]["coord"]
    npts = coord.shape[0]
    n_frags = len(all_results)

    traces = []

    # Build union mask across all fragments
    any_shower = np.zeros(npts, dtype=bool)
    for r in all_results:
        mask = r["shower_mask"]
        if mask is not None:
            any_shower |= mask.astype(bool)

    non_shower = ~any_shower
    if non_shower.sum() > 0:
        traces.append({
            "type": "scatter3d",
            "x": coord[non_shower, 0].tolist(),
            "y": coord[non_shower, 1].tolist(),
            "z": coord[non_shower, 2].tolist(),
            "mode": "markers",
            "name": f"non-shower ({int(non_shower.sum()):,})",
            "marker": {"color": "rgba(120,120,120,0.2)", "size": 1},
            "hoverinfo": "skip",
        })

    # Each fragment in a distinct color
    for fi, r in enumerate(all_results):
        mask = r["shower_mask"]
        if mask is None:
            continue
        shower = mask.astype(bool)
        if shower.sum() == 0:
            continue

        is_selected = (fi == selected_idx)
        color = FRAGMENT_COLORS[fi % len(FRAGMENT_COLORS)]
        size = 4 if is_selected else 2
        opacity_str = "1.0" if is_selected else "0.6"

        # Prediction info for hover
        pred_class = r.get("pred_class", -1)
        pred_name = ORIGIN_TYPE_NAMES.get(pred_class, "?")

        traces.append({
            "type": "scatter3d",
            "x": coord[shower, 0].tolist(),
            "y": coord[shower, 1].tolist(),
            "z": coord[shower, 2].tolist(),
            "mode": "markers",
            "name": f"frag {fi} ({int(shower.sum())}, pred={pred_name})"
                    + (" *" if is_selected else ""),
            "marker": {"color": color, "size": size},
            "hovertemplate": (
                f"<b>frag {fi}</b> (pred={pred_name})<br>"
                "<b>x</b>: %{x:.1f}<br>"
                "<b>y</b>: %{y:.1f}<br>"
                "<b>z</b>: %{z:.1f}<extra></extra>"
            ),
        })

    # Text labels at each fragment centroid
    for fi, r in enumerate(all_results):
        mask = r["shower_mask"]
        if mask is None:
            continue
        shower = mask.astype(bool)
        if shower.sum() == 0:
            continue

        centroid = coord[shower].mean(axis=0)
        pred_class = r.get("pred_class", -1)
        pred_name = ORIGIN_TYPE_NAMES.get(pred_class, "?")

        traces.append({
            "type": "scatter3d",
            "x": [float(centroid[0])],
            "y": [float(centroid[1])],
            "z": [float(centroid[2])],
            "mode": "text+markers",
            "text": [f"F{fi}"],
            "textposition": "top center",
            "textfont": {"color": "white", "size": 11},
            "marker": {
                "color": FRAGMENT_COLORS[fi % len(FRAGMENT_COLORS)],
                "size": 6,
                "symbol": "circle",
                "line": {"color": "white", "width": 1},
            },
            "hovertemplate": (
                f"<b>Fragment {fi}</b><br>"
                f"pred: {pred_name}<br>"
                f"pts: {int(shower.sum())}<br>"
                "centroid: (%{x:.1f}, %{y:.1f}, %{z:.1f})"
                "<extra></extra>"
            ),
            "showlegend": False,
        })

    # Add MicroBooNE detector outline
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

    title = f"All Fragments ({n_frags} frags, selected=F{selected_idx})"

    return {"data": traces, "layout": make_layout(
        title, npts, height=650,
        camera_up={"x": 0, "y": 1, "z": 0},
        aspect_ratio={"x": 1, "y": 1, "z": 4},
        axis_labels=["X (cm)", "Y (cm)", "Z (cm)"],
    )}


def build_classification_figure(result, class_names):
    """Classification bar chart: GT vs predicted with probabilities."""
    import plotly.graph_objects as go

    logits = result["pred_class_logits"]
    probs = np.exp(logits - logits.max())
    probs = probs / probs.sum()

    gt_type = result["origin_type"]
    pred_class = result["pred_class"]
    match = (gt_type == pred_class)

    n_classes = len(logits)
    labels = [class_names.get(i, f"class_{i}") for i in range(n_classes)]

    # Bar colors: highlight GT and predicted
    colors = []
    for i in range(n_classes):
        if i == gt_type and i == pred_class:
            colors.append("rgba(0,255,0,0.8)")   # green = match
        elif i == gt_type:
            colors.append("rgba(0,150,255,0.8)")  # blue = GT
        elif i == pred_class:
            colors.append("rgba(255,165,0,0.8)")  # orange = predicted
        else:
            colors.append("rgba(100,100,100,0.5)")

    bar_trace = {
        "type": "bar",
        "x": labels,
        "y": probs.tolist(),
        "marker": {"color": colors},
        "text": [f"{p:.1%}" for p in probs],
        "textposition": "auto",
        "textfont": {"color": "white"},
    }

    match_str = "MATCH" if match else "MISMATCH"
    match_color = "lime" if match else "red"
    gt_name = class_names.get(gt_type, f"class_{gt_type}")
    pred_name = class_names.get(pred_class, f"class_{pred_class}")

    layout = {
        "title": {
            "text": (
                f"Classification: GT={gt_name} | Pred={pred_name} | "
                f"<span style='color:{match_color}'>{match_str}</span>"
            ),
            "font": {"size": 14, "color": "white"},
        },
        "height": 550,
        "margin": {"t": 50, "b": 40, "l": 40, "r": 20},
        "font": {"size": 12, "color": "white"},
        "plot_bgcolor": "#1e1e1e",
        "paper_bgcolor": "#1e1e1e",
        "xaxis": {"color": "white", "gridcolor": "#333"},
        "yaxis": {
            "color": "white", "gridcolor": "#333",
            "title": "probability", "range": [0, 1],
        },
        "showlegend": False,
    }

    return {"data": [bar_trace], "layout": layout}


# ── Main ─────────────────────────────────────────────────────────────────────

args = parse_args()
cfg = Config.fromfile(args.config)

# Device
if args.device:
    device = torch.device(args.device)
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Device: {device}")

# Build dataset (no transforms — we apply them manually)
dataset = build_dataset(cfg, args.split, data_list_override=args.data_list)
n_entries = len(dataset)
print(f"Dataset: {n_entries} entries (split={args.split})")
print(f"  data_list: {len(dataset.data_list)} files")
print(f"  event_index: {len(dataset.event_index)} events with shower_fragments")

if n_entries == 0:
    print("\nERROR: No events with shower_fragments found!")
    sys.exit(1)

# Build inference transforms (full val pipeline)
transform = build_inference_transforms(cfg, args.split)
print(f"Inference transforms: {transform}")

# Load model
model = load_model(cfg, args.checkpoint, device)

# Extract config values
gaussian_sigma = cfg.model.get("gaussian_sigma", 0.022)
num_origin_classes = cfg.model.get("num_origin_classes", 3)

# Extract coordinate normalization params from the NormalizeShowerCoords transform
coord_center = None
coord_scale = None
val_transforms = getattr(cfg.data, args.split).get("transform", [])
for t in val_transforms:
    if t.get("type") == "NormalizeShowerCoords":
        coord_center = list(t.get("center", [125.0, 0.0, 518.0]))
        coord_scale = float(t.get("scale", 179.55))
        break

print(f"gaussian_sigma={gaussian_sigma}, num_origin_classes={num_origin_classes}")
print(f"coord_center={coord_center}, coord_scale={coord_scale}")


# ── Dash app ─────────────────────────────────────────────────────────────────

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
        # Header
        html.Div([
            html.H2(
                "Shower Origin Inference Visualizer",
                style={"color": "white", "display": "inline-block", "marginRight": "20px"},
            ),
            html.Button("Next Event", id="next-btn", n_clicks=0,
                         style={**BTN_STYLE, "backgroundColor": "#4CAF50"}),
            html.Button("Random", id="random-btn", n_clicks=0,
                         style={**BTN_STYLE, "backgroundColor": "#FF9800"}),
            dcc.Input(
                id="entry-input", type="number", value=args.entry,
                min=0, max=max(n_entries - 1, 0), placeholder="Event index",
                style={
                    "fontSize": "14px", "padding": "8px", "width": "120px",
                    "borderRadius": "5px", "border": "1px solid #555",
                    "backgroundColor": "#333", "color": "white", "marginRight": "10px",
                },
            ),
            html.Button("Go", id="go-btn", n_clicks=0,
                         style={**BTN_STYLE, "backgroundColor": "#2196F3"}),
        ], style={"marginBottom": "10px"}),

        # Fragment dropdown
        html.Div([
            html.Label("Fragment: ", style={"color": "white", "marginRight": "10px",
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
        ], style={"marginBottom": "10px", "display": "flex", "alignItems": "center"}),

        # Info bar
        html.Div(id="info-label",
                  style={"color": "#ccc", "fontSize": "13px", "marginBottom": "5px"}),
        html.Div(
            f"Config: {os.path.basename(args.config)}  |  "
            f"Checkpoint: {os.path.basename(args.checkpoint)}  |  "
            f"split={args.split}  |  device={device}",
            style={"color": "#888", "fontSize": "12px", "marginBottom": "15px"},
        ),

        # Row 1: Predicted Scores + Classification (side by side)
        html.Div([
            html.Div([dcc.Graph(id="scores-graph", config={"scrollZoom": True})],
                      style=GRAPH_STYLE_HALF),
            html.Div([dcc.Graph(id="class-graph", config={"scrollZoom": True})],
                      style=GRAPH_STYLE_HALF),
        ]),

        # Row 2: All Fragments (full width)
        html.Div([
            dcc.Graph(id="mask-graph", config={"scrollZoom": True}),
        ]),

        # Hidden stores
        dcc.Store(id="current-idx", data=args.entry),
        dcc.Store(id="fragment-cache", data=[]),
    ],
)


# ── Callback 1: Event change — run inference, cache results ──────────────────

@app.callback(
    [
        Output("fragment-cache", "data"),
        Output("fragment-dropdown", "options"),
        Output("fragment-dropdown", "value"),
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
def on_event_change(next_clicks, random_clicks, go_clicks,
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
        idx = int(current_idx) if current_idx is not None else 0

    # Run inference
    try:
        results = run_event_inference(dataset, idx, transform, model, device)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error on event {idx}: {e}")
        results = []

    # Build dropdown options
    options = []
    for i, r in enumerate(results):
        type_name = ORIGIN_TYPE_NAMES.get(r["origin_type"], "?")
        n_shower = int(np.asarray(r["shower_mask"]).astype(bool).sum())
        label = f"Fragment {i}: type={type_name}, {n_shower} shower pts, {r['n_points_model']} model pts"
        options.append({"label": label, "value": i})

    cache = results_to_json(results)
    default_value = 0 if results else None

    return cache, options, default_value, idx


# ── Callback 2: Fragment selection — build figures from cache ────────────────

@app.callback(
    [
        Output("scores-graph", "figure"),
        Output("mask-graph", "figure"),
        Output("class-graph", "figure"),
        Output("info-label", "children"),
    ],
    [Input("fragment-dropdown", "value")],
    [
        State("fragment-cache", "data"),
        State("current-idx", "data"),
    ],
)
def on_fragment_select(frag_idx, cache_data, current_idx):
    if frag_idx is None or not cache_data:
        return (
            empty_figure("Predicted Scores"),
            empty_figure("Shower Mask"),
            empty_figure("Classification"),
            "No fragments available",
        )

    results = results_from_json(cache_data)
    frag_idx = int(frag_idx)
    if frag_idx >= len(results):
        frag_idx = 0
    r = results[frag_idx]

    # Build figures
    scores_fig = build_predicted_scores_figure(
        r, gaussian_sigma=gaussian_sigma,
        coord_center=coord_center, coord_scale=coord_scale,
    )
    mask_fig = build_shower_mask_figure(results, selected_idx=frag_idx)
    class_fig = build_classification_figure(r, ORIGIN_TYPE_NAMES)

    # Info
    gt_type = r["origin_type"]
    gt_name = ORIGIN_TYPE_NAMES.get(gt_type, "?")
    pred_class = r["pred_class"]
    pred_name = ORIGIN_TYPE_NAMES.get(pred_class, "?")
    match = "YES" if gt_type == pred_class else "NO"

    info = (
        f"Event: {current_idx} / {n_entries}  |  "
        f"Fragment: {frag_idx} / {len(results)}  |  "
        f"Name: {r.get('name', '?')}  |  "
        f"Raw pts: {r.get('n_points_raw', '?'):,}  |  "
        f"Model pts: {r.get('n_points_model', '?'):,}  |  "
        f"GT: {gt_name}  |  Pred: {pred_name}  |  "
        f"Match: {match}"
    )

    return scores_fig, mask_fig, class_fig, info


if __name__ == "__main__":
    app.run(debug=True, port=args.port)
