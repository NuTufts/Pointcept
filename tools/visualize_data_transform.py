"""
Visualize LArTPC data after the transform pipeline.

Reads a config file, builds the train dataset and transforms,
and displays one entry showing:
  - Original batch (after pre-view transforms, before MultiViewGenerator)
  - Two global views
  - Two masked global views (Sonata-style patch masking with jitter)
  - Four local views

Each point shows hover text with associated features.
A button allows sampling a new random entry.

Usage:
    python tools/visualize_data_transform.py \
        -c configs/lartpc/pretrain-sonata-v1m1-lartpc-v6.py

    python tools/visualize_data_transform.py \
        -c configs/lartpc/pretrain-sonata-v1m1-lartpc-v5-p100.py \
        --data-list /path/to/my_filelist.txt

    python tools/visualize_data_transform.py \
        -c configs/lartpc/pretrain-sonata-v1m1-lartpc-v6.py \
        -e 42 --port 8051 --mask-size 5.0 --mask-ratio 0.7
"""

import os
import sys
import copy
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


# ── Segment class colors (distinct, for up to 11 classes + ignore) ──────────

SEGMENT_COLORS = [
    "rgba(255,0,0,0.8)",       # 0: electron - red
    "rgba(0,0,255,0.8)",       # 1: muon - blue
    "rgba(160,32,240,0.8)",    # 2: pion - purple
    "rgba(0,200,255,0.8)",     # 3: proton - cyan
    "rgba(255,165,0,0.8)",     # 4: gamma - orange
    "rgba(255,0,200,0.8)",     # 5: michel - magenta
    "rgba(0,200,150,0.8)",     # 6: delta - teal
    "rgba(100,200,255,0.8)",   # 7: led - light blue
    "rgba(120,120,120,0.8)",   # 8: ghost - gray
    "rgba(80,80,80,0.8)",      # 9: other - dark gray
    "rgba(200,200,0,0.8)",     # 10: extra
]


# ── Plotly layout templates ─────────────────────────────────────────────────

AXIS_TEMPLATE = {
    "showbackground": True,
    "backgroundcolor": "#141414",
    "gridcolor": "rgb(100, 100, 100)",
    "zerolinecolor": "rgb(200, 200, 200)",
}


def make_layout(title, npts, height=500):
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


# ── Argument parsing ────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize data after transform pipeline"
    )
    parser.add_argument(
        "-c", "--config", required=True, type=str, help="Config file path"
    )
    parser.add_argument(
        "-e", "--entry", default=0, type=int, help="Initial entry index"
    )
    parser.add_argument(
        "--data-list", type=str, default=None,
        help="Override data list file path (if cluster paths are not available)"
    )
    parser.add_argument(
        "--port", default=8050, type=int, help="Dash server port"
    )
    parser.add_argument(
        "--mask-size", type=float, default=None,
        help="Mask patch grid size (default: from cfg.model.mask_size_base)"
    )
    parser.add_argument(
        "--mask-ratio", type=float, default=None,
        help="Fraction of patches to mask (default: from cfg.model.mask_ratio_base)"
    )
    parser.add_argument(
        "--mask-jitter", type=float, default=None,
        help="Jitter sigma for masked coords (default: from cfg.model.mask_jitter)"
    )
    return parser.parse_args()


# ── Build dataset and transforms ────────────────────────────────────────────

def build_dataset_and_transforms(cfg, data_list_override=None):
    """
    Build the dataset (no transforms) and split the transform
    pipeline into pre-view, MultiViewGenerator, and post-view stages.
    """
    dataset_cfg = cfg.data.train.copy()
    transform_cfg_list = dataset_cfg.pop("transform", []) or []

    # Override data list file if provided
    if data_list_override is not None:
        dataset_cfg["data_list_file"] = data_list_override

    # Build dataset with no transforms
    dataset_cfg["transform"] = None
    dataset = DATASETS.build(dataset_cfg)

    # Split transforms into: pre-view | MultiViewGenerator | post-view
    pre_view_cfgs = []
    multi_view_cfg = None
    found_multiview = False

    for t_cfg in transform_cfg_list:
        t_type = t_cfg.get("type", "")
        if t_type == "MultiViewGenerator":
            multi_view_cfg = t_cfg
            found_multiview = True
        elif not found_multiview:
            pre_view_cfgs.append(t_cfg)
        # Skip post-view transforms (ToTensor, Collect, Update) — not needed

    pre_transform = Compose(pre_view_cfgs) if pre_view_cfgs else Compose([])
    multi_view_gen = (
        TRANSFORMS.build(multi_view_cfg) if multi_view_cfg else None
    )

    return dataset, pre_transform, multi_view_gen


def get_mask_params(cfg, args):
    """Extract mask parameters from model config, with CLI overrides."""
    mask_size = 2.0
    mask_ratio = 0.5
    mask_jitter = 0.125

    if hasattr(cfg, "model"):
        m = cfg.model
        if hasattr(m, "mask_size_base"):
            mask_size = m.mask_size_base
        if hasattr(m, "mask_ratio_base"):
            mask_ratio = m.mask_ratio_base
        if hasattr(m, "mask_jitter") and m.mask_jitter is not None:
            mask_jitter = m.mask_jitter

    # CLI overrides take priority
    if args.mask_size is not None:
        mask_size = args.mask_size
    if args.mask_ratio is not None:
        mask_ratio = args.mask_ratio
    if args.mask_jitter is not None:
        mask_jitter = args.mask_jitter

    return mask_size, mask_ratio, mask_jitter


# ── Data processing ─────────────────────────────────────────────────────────

def split_by_offset(arr, offset):
    """Split a concatenated array into individual views using cumulative offsets."""
    views = []
    prev = 0
    for o in offset:
        views.append(arr[prev:int(o)])
        prev = int(o)
    return views


def process_entry(
    dataset, pre_transform, multi_view_gen,
    mask_gen, mask_jitter_gen, idx,
):
    """
    Process a single entry through the transform pipeline.

    Returns:
        original_data: dict with coord, strength, segment after pre-transforms
        global_views: list of dicts with coord, origin_coord, strength
        masked_views: list of dicts with coord, jittered_coord, mask, cluster, strength
        local_views: list of dicts with coord, origin_coord, strength
    """
    raw_data = dataset.get_data(idx)
    pre_data = pre_transform(copy.deepcopy(raw_data))

    original_data = {
        "coord": pre_data["coord"].copy(),
        "name": pre_data.get("name", f"entry_{idx}"),
    }
    if "strength" in pre_data:
        original_data["strength"] = pre_data["strength"].copy()
    if "segment" in pre_data:
        original_data["segment"] = pre_data["segment"].copy()

    global_views = []
    masked_views = []
    local_views = []

    if multi_view_gen is not None:
        view_data = multi_view_gen(copy.deepcopy(pre_data))

        # Apply mask generation and jitter
        if mask_gen is not None:
            view_data = mask_gen(view_data)
        if mask_jitter_gen is not None and "global_mask" in view_data:
            view_data = mask_jitter_gen(view_data)

        # Split global views using cumulative offsets
        g_off = view_data["global_offset"]
        g_coords = split_by_offset(view_data["global_coord"], g_off)
        g_origin = (
            split_by_offset(view_data["global_origin_coord"], g_off)
            if "global_origin_coord" in view_data
            else [None] * len(g_coords)
        )
        g_strength = (
            split_by_offset(view_data["global_strength"], g_off)
            if "global_strength" in view_data
            else [None] * len(g_coords)
        )
        for i in range(len(g_coords)):
            global_views.append({
                "coord": g_coords[i],
                "origin_coord": g_origin[i],
                "strength": g_strength[i],
            })

        # Split masked view arrays
        if "global_mask" in view_data:
            g_mask = split_by_offset(
                view_data["global_mask"], g_off
            )
            g_cluster = split_by_offset(
                view_data["global_cluster"], g_off
            )
            g_jittered = (
                split_by_offset(view_data["global_jittered_coord"], g_off)
                if "global_jittered_coord" in view_data
                else g_coords
            )
            for i in range(len(g_coords)):
                masked_views.append({
                    "coord": g_coords[i],
                    "jittered_coord": g_jittered[i],
                    "mask": g_mask[i],
                    "cluster": g_cluster[i],
                    "strength": g_strength[i],
                    "origin_coord": g_origin[i],
                })

        # Split local views
        l_off = view_data["local_offset"]
        l_coords = split_by_offset(view_data["local_coord"], l_off)
        l_origin = (
            split_by_offset(view_data["local_origin_coord"], l_off)
            if "local_origin_coord" in view_data
            else [None] * len(l_coords)
        )
        l_strength = (
            split_by_offset(view_data["local_strength"], l_off)
            if "local_strength" in view_data
            else [None] * len(l_coords)
        )
        for i in range(len(l_coords)):
            local_views.append({
                "coord": l_coords[i],
                "origin_coord": l_origin[i],
                "strength": l_strength[i],
            })

    return original_data, global_views, masked_views, local_views


# ── Figure builders ─────────────────────────────────────────────────────────

def build_original_figure(data, class_names, title="Original Batch"):
    """Build figure for the original batch, colored by segment class."""
    coord = data["coord"]
    segment = data.get("segment")
    strength = data.get("strength")
    npts = coord.shape[0]

    # Build per-point hover text columns
    hover_parts = [
        "<b>x</b>: %{x:.2f}<br>",
        "<b>y</b>: %{y:.2f}<br>",
        "<b>z</b>: %{z:.2f}<br>",
    ]
    custom_cols = []
    ci = 0

    if strength is not None:
        nchan = strength.shape[1] if strength.ndim > 1 else 1
        for ch in range(min(nchan, 3)):
            hover_parts.append(
                f"<b>feat[{ch}]</b>: %{{customdata[{ci}]:.3f}}<br>"
            )
            ci += 1
        if strength.ndim == 1:
            custom_cols.append(strength.reshape(-1, 1))
        else:
            custom_cols.append(strength[:, :min(nchan, 3)])

    if segment is not None:
        hover_parts.append(f"<b>segment</b>: %{{customdata[{ci}]:d}}<br>")
        custom_cols.append(segment.reshape(-1, 1).astype(np.float64))
        ci += 1

    hovertemplate = "".join(hover_parts)
    customdata = (
        np.concatenate(custom_cols, axis=1) if custom_cols else None
    )

    traces = []
    if segment is not None:
        # One trace per segment class for a proper legend
        unique_segs = np.unique(segment)
        for seg_val in sorted(unique_segs):
            if seg_val < 0:
                continue  # skip ignore index
            mask = segment == seg_val
            if mask.sum() == 0:
                continue
            seg_int = int(seg_val)
            color = SEGMENT_COLORS[seg_int % len(SEGMENT_COLORS)]
            label = (
                class_names[seg_int]
                if class_names and seg_int < len(class_names)
                else f"class {seg_int}"
            )
            traces.append({
                "type": "scatter3d",
                "x": coord[mask, 0],
                "y": coord[mask, 1],
                "z": coord[mask, 2],
                "mode": "markers",
                "name": label,
                "hovertemplate": hovertemplate,
                "customdata": customdata[mask] if customdata is not None else None,
                "marker": {
                    "color": color,
                    "opacity": 0.8,
                    "size": 2,
                },
            })
    else:
        # No segment info — color by z
        traces.append({
            "type": "scatter3d",
            "x": coord[:, 0],
            "y": coord[:, 1],
            "z": coord[:, 2],
            "mode": "markers",
            "name": "points",
            "hovertemplate": hovertemplate,
            "customdata": customdata,
            "marker": {
                "color": coord[:, 2],
                "opacity": 0.8,
                "size": 2,
                "colorscale": "Viridis",
            },
        })

    return {"data": traces, "layout": make_layout(title, npts, height=600)}


def build_view_figure(view_data, title="View"):
    """Build figure for a global or local view, colored by strength sum."""
    coord = view_data["coord"]
    strength = view_data.get("strength")
    origin_coord = view_data.get("origin_coord")
    npts = coord.shape[0]

    hover_parts = [
        "<b>x</b>: %{x:.2f}<br>",
        "<b>y</b>: %{y:.2f}<br>",
        "<b>z</b>: %{z:.2f}<br>",
    ]
    custom_cols = []
    ci = 0

    if strength is not None:
        nchan = strength.shape[1] if strength.ndim > 1 else 1
        for ch in range(min(nchan, 3)):
            hover_parts.append(
                f"<b>feat[{ch}]</b>: %{{customdata[{ci}]:.3f}}<br>"
            )
            ci += 1
        if strength.ndim == 1:
            custom_cols.append(strength.reshape(-1, 1))
        else:
            custom_cols.append(strength[:, :min(nchan, 3)])

    if origin_coord is not None:
        hover_parts.append(
            f"<b>origin</b>: (%{{customdata[{ci}]:.2f}}, "
            f"%{{customdata[{ci+1}]:.2f}}, "
            f"%{{customdata[{ci+2}]:.2f}})<br>"
        )
        custom_cols.append(origin_coord[:, :3])
        ci += 3

    hovertemplate = "".join(hover_parts)
    customdata = (
        np.concatenate(custom_cols, axis=1) if custom_cols else None
    )

    if strength is not None and strength.ndim > 1:
        #color_values = np.sum(strength, axis=1)
        # use the Y-plane, index-2, in order to try and see relative ionization
        color_values = strength[:,2]
    else:
        color_values = coord[:, 2]

    trace = {
        "type": "scatter3d",
        "x": coord[:, 0],
        "y": coord[:, 1],
        "z": coord[:, 2],
        "mode": "markers",
        "name": title,
        "hovertemplate": hovertemplate,
        "customdata": customdata,
        "marker": {
            "color": color_values,
            "opacity": 0.8,
            "size": 2,
            "colorscale": "Viridis",
            "colorbar": {"title": "feat sum", "len": 0.5},
        },
    }

    return {"data": [trace], "layout": make_layout(title, npts, height=500)}


def build_masked_view_figure(view_data, title="Masked View"):
    """
    Build figure for a masked global view.

    Shows unmasked points colored by strength (Viridis) at original coords,
    and masked points in red at jittered coords.
    """
    mask = view_data["mask"]
    jittered_coord = view_data.get("jittered_coord", view_data["coord"])
    orig_coord = view_data["coord"]
    strength = view_data.get("strength")
    cluster = view_data.get("cluster")
    npts = mask.shape[0]
    n_masked = int(mask.sum())
    n_visible = npts - n_masked
    n_patches = int(cluster.max() + 1) if cluster is not None else 0
    pct = 100.0 * n_masked / max(npts, 1)

    traces = []

    # ── Visible (unmasked) points ───────────────────────────
    vis = ~mask
    if vis.sum() > 0:
        vis_hover = [
            "<b>x</b>: %{x:.2f}<br>",
            "<b>y</b>: %{y:.2f}<br>",
            "<b>z</b>: %{z:.2f}<br>",
            "<b>masked</b>: no<br>",
        ]
        vis_cols = []
        ci = 0

        if strength is not None:
            nchan = strength.shape[1] if strength.ndim > 1 else 1
            for ch in range(min(nchan, 3)):
                vis_hover.append(
                    f"<b>feat[{ch}]</b>: %{{customdata[{ci}]:.3f}}<br>"
                )
                ci += 1
            if strength.ndim == 1:
                vis_cols.append(strength[vis].reshape(-1, 1))
            else:
                vis_cols.append(strength[vis, :min(nchan, 3)])

        if cluster is not None:
            vis_hover.append(
                f"<b>patch</b>: %{{customdata[{ci}]:d}}<br>"
            )
            vis_cols.append(cluster[vis].reshape(-1, 1).astype(np.float64))
            ci += 1

        vis_hovertemplate = "".join(vis_hover)
        vis_customdata = (
            np.concatenate(vis_cols, axis=1) if vis_cols else None
        )

        if strength is not None and strength.ndim > 1:
            vis_color = np.sum(strength[vis], axis=1)
        else:
            vis_color = orig_coord[vis, 2]

        traces.append({
            "type": "scatter3d",
            "x": orig_coord[vis, 0],
            "y": orig_coord[vis, 1],
            "z": orig_coord[vis, 2],
            "mode": "markers",
            "name": f"visible ({n_visible:,})",
            "hovertemplate": vis_hovertemplate,
            "customdata": vis_customdata,
            "marker": {
                "color": vis_color,
                "opacity": 0.8,
                "size": 2,
                "colorscale": "Viridis",
            },
        })

    # ── Masked points (shown at jittered coordinates) ───────
    if n_masked > 0:
        m_hover = [
            "<b>jittered x</b>: %{x:.2f}<br>",
            "<b>jittered y</b>: %{y:.2f}<br>",
            "<b>jittered z</b>: %{z:.2f}<br>",
            "<b>masked</b>: yes<br>",
        ]
        m_cols = []
        ci = 0

        # Original (pre-jitter) coordinates
        m_hover.append(
            f"<b>orig</b>: (%{{customdata[{ci}]:.2f}}, "
            f"%{{customdata[{ci+1}]:.2f}}, "
            f"%{{customdata[{ci+2}]:.2f}})<br>"
        )
        m_cols.append(orig_coord[mask, :3])
        ci += 3

        if strength is not None:
            nchan = strength.shape[1] if strength.ndim > 1 else 1
            for ch in range(min(nchan, 3)):
                m_hover.append(
                    f"<b>feat[{ch}]</b>: %{{customdata[{ci}]:.3f}}<br>"
                )
                ci += 1
            if strength.ndim == 1:
                m_cols.append(strength[mask].reshape(-1, 1))
            else:
                m_cols.append(strength[mask, :min(nchan, 3)])

        if cluster is not None:
            m_hover.append(
                f"<b>patch</b>: %{{customdata[{ci}]:d}}<br>"
            )
            m_cols.append(cluster[mask].reshape(-1, 1).astype(np.float64))
            ci += 1

        m_hovertemplate = "".join(m_hover)
        m_customdata = (
            np.concatenate(m_cols, axis=1) if m_cols else None
        )

        traces.append({
            "type": "scatter3d",
            "x": jittered_coord[mask, 0],
            "y": jittered_coord[mask, 1],
            "z": jittered_coord[mask, 2],
            "mode": "markers",
            "name": f"masked ({n_masked:,})",
            "hovertemplate": m_hovertemplate,
            "customdata": m_customdata,
            "marker": {
                "color": "rgba(255,50,50,0.5)",
                "size": 2,
            },
        })

    full_title = (
        f"{title}  —  {pct:.0f}% masked, "
        f"{n_patches} patches"
    )
    return {"data": traces, "layout": make_layout(full_title, npts, height=500)}


def empty_figure(title="N/A"):
    return {
        "data": [],
        "layout": make_layout(title, 0, height=500),
    }


# ── Main ────────────────────────────────────────────────────────────────────

args = parse_args()
cfg = Config.fromfile(args.config)

dataset, pre_transform, multi_view_gen = build_dataset_and_transforms(
    cfg, data_list_override=args.data_list
)

# Get mask parameters from model config + CLI overrides
mask_size, mask_ratio, mask_jitter = get_mask_params(cfg, args)

# Build mask transforms
mask_gen = TRANSFORMS.build(dict(
    type="GeneratePatchMask",
    mask_size=mask_size,
    mask_ratio=mask_ratio,
    prefix="global",
))
mask_jitter_gen = TRANSFORMS.build(dict(
    type="ApplyMaskJitter",
    jitter_sigma=mask_jitter,
    prefix="global",
))

# Get class names from config
class_names = None
if hasattr(cfg.data, "names"):
    class_names = list(cfg.data.names)
elif hasattr(dataset, "class_names") and dataset.class_names is not None:
    class_names = list(dataset.class_names)

n_entries = len(dataset.data_list)
print(f"Dataset: {n_entries} entries")
print(f"MultiViewGenerator: {'present' if multi_view_gen else 'absent'}")
print(f"Mask params: size={mask_size}, ratio={mask_ratio}, jitter={mask_jitter}")
print(f"Class names: {class_names}")
print(f"Config: {args.config}")


# ── Dash app ────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ],
)

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
                    "Data Transform Visualizer",
                    style={
                        "color": "white",
                        "display": "inline-block",
                        "marginRight": "20px",
                    },
                ),
                html.Button(
                    "Sample New Entry",
                    id="sample-btn",
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
                dcc.Input(
                    id="entry-input",
                    type="number",
                    value=args.entry,
                    min=0,
                    max=n_entries - 1,
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
            f"mask_size={mask_size}  mask_ratio={mask_ratio}  "
            f"mask_jitter={mask_jitter}",
            style={"color": "#888", "fontSize": "12px", "marginBottom": "15px"},
        ),
        # ── Original batch ──────────────────────────────────
        html.Div(
            [
                html.H3(
                    "Original Batch (after pre-view transforms)",
                    style={"color": "white", "marginBottom": "5px"},
                ),
                dcc.Graph(
                    id="original-graph",
                    config={"scrollZoom": True},
                ),
            ]
        ),
        # ── Global views ────────────────────────────────────
        html.Div(
            [
                html.H3(
                    "Global Views (teacher input)",
                    style={"color": "white", "marginBottom": "5px"},
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                dcc.Graph(
                                    id=f"global-{i}-graph",
                                    config={"scrollZoom": True},
                                )
                            ],
                            style={
                                "width": "50%",
                                "display": "inline-block",
                                "verticalAlign": "top",
                            },
                        )
                        for i in range(2)
                    ]
                ),
            ]
        ),
        # ── Masked global views ─────────────────────────────
        html.Div(
            [
                html.H3(
                    "Masked Global Views (student input — visible + masked w/ jitter)",
                    style={"color": "white", "marginBottom": "5px"},
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                dcc.Graph(
                                    id=f"masked-{i}-graph",
                                    config={"scrollZoom": True},
                                )
                            ],
                            style={
                                "width": "50%",
                                "display": "inline-block",
                                "verticalAlign": "top",
                            },
                        )
                        for i in range(2)
                    ]
                ),
            ]
        ),
        # ── Local views ─────────────────────────────────────
        html.Div(
            [
                html.H3(
                    "Local Views",
                    style={"color": "white", "marginBottom": "5px"},
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                dcc.Graph(
                                    id=f"local-{i}-graph",
                                    config={"scrollZoom": True},
                                )
                            ],
                            style={
                                "width": "25%",
                                "display": "inline-block",
                                "verticalAlign": "top",
                            },
                        )
                        for i in range(4)
                    ]
                ),
            ]
        ),
        # ── Hidden store for current entry index ────────────
        dcc.Store(id="current-idx", data=args.entry),
    ],
)


@app.callback(
    [
        Output("original-graph", "figure"),
        Output("global-0-graph", "figure"),
        Output("global-1-graph", "figure"),
        Output("masked-0-graph", "figure"),
        Output("masked-1-graph", "figure"),
        Output("local-0-graph", "figure"),
        Output("local-1-graph", "figure"),
        Output("local-2-graph", "figure"),
        Output("local-3-graph", "figure"),
        Output("entry-label", "children"),
        Output("current-idx", "data"),
    ],
    [
        Input("sample-btn", "n_clicks"),
        Input("go-btn", "n_clicks"),
    ],
    [
        State("current-idx", "data"),
        State("entry-input", "value"),
    ],
)
def update_display(sample_clicks, go_clicks, current_idx, entry_input):
    ctx = dash.callback_context
    triggered = ctx.triggered[0]["prop_id"] if ctx.triggered else ""

    if "sample-btn" in triggered:
        idx = int(np.random.randint(0, n_entries))
    elif "go-btn" in triggered and entry_input is not None:
        idx = int(np.clip(entry_input, 0, n_entries - 1))
    else:
        # Initial load
        idx = int(current_idx) if current_idx is not None else 0

    # Process entry (retry on failure)
    for attempt in range(5):
        try:
            original_data, global_views, masked_views, local_views = (
                process_entry(
                    dataset, pre_transform, multi_view_gen,
                    mask_gen, mask_jitter_gen, idx,
                )
            )
            break
        except Exception as e:
            print(f"Error processing entry {idx}: {e}")
            idx = int(np.random.randint(0, n_entries))
    else:
        # All retries failed — return empty figures
        return (
            empty_figure("Original Batch (error)"),
            empty_figure("Global View 0"),
            empty_figure("Global View 1"),
            empty_figure("Masked Global View 0"),
            empty_figure("Masked Global View 1"),
            empty_figure("Local View 0"),
            empty_figure("Local View 1"),
            empty_figure("Local View 2"),
            empty_figure("Local View 3"),
            "Error: could not process any entry",
            idx,
        )

    # Build figures
    orig_fig = build_original_figure(original_data, class_names)

    global_figs = []
    for i in range(2):
        if i < len(global_views):
            global_figs.append(
                build_view_figure(global_views[i], title=f"Global View {i}")
            )
        else:
            global_figs.append(empty_figure(f"Global View {i}"))

    masked_figs = []
    for i in range(2):
        if i < len(masked_views):
            masked_figs.append(
                build_masked_view_figure(
                    masked_views[i], title=f"Masked Global View {i}"
                )
            )
        else:
            masked_figs.append(empty_figure(f"Masked Global View {i}"))

    local_figs = []
    for i in range(4):
        if i < len(local_views):
            local_figs.append(
                build_view_figure(local_views[i], title=f"Local View {i}")
            )
        else:
            local_figs.append(empty_figure(f"Local View {i}"))

    name = original_data.get("name", "?")
    entry_label = (
        f"Entry: {idx} / {n_entries}  |  "
        f"Name: {name}  |  "
        f"Original pts: {original_data['coord'].shape[0]:,}"
    )

    return (
        orig_fig,
        global_figs[0],
        global_figs[1],
        masked_figs[0],
        masked_figs[1],
        local_figs[0],
        local_figs[1],
        local_figs[2],
        local_figs[3],
        entry_label,
        idx,
    )


if __name__ == "__main__":
    app.run(debug=True, port=args.port)
