"""
Interactive Similarity Visualization of Sonata Encoder Features

Extracts encoder features from a trained Sonata model and creates an
interactive 3D visualization where clicking on a point colors all other
points by their cosine similarity to the selected point.

Uses Plotly/Dash for interactive visualization.

Usage:
    python tools/visualize_sonata_similarity.py \
        --config configs/lartpc/pretrain-sonata-v1m1-lartpc.py \
        --checkpoint exp/lartpc/sonata-v1m1/model_best.pth \
        --data-list /path/to/val_split.txt \
        --entry 0

Author: Generated for LArTPC analysis
"""

import os
import sys
import argparse
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add lardly to path
lardly_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..", "ubdl", "lardly"
)
if os.path.exists(lardly_path):
    sys.path.insert(0, lardly_path)

from pointcept.utils.config import Config
from pointcept.models.builder import build_model
from pointcept.datasets.transform import Compose, TRANSFORMS

from sonata_vis_utils import (
    assign_labels_to_output_points,
    CLASS_NAMES,
    NUM_CLASSES,
    GHOST_LABEL,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive similarity visualization of Sonata encoder features"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file (e.g., configs/lartpc/pretrain-sonata-v1m1-lartpc.py)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (.pth file). Not required if --random-init is used.",
    )
    parser.add_argument(
        "--random-init",
        action="store_true",
        default=False,
        help="Use randomly initialized model weights (baseline comparison)",
    )
    parser.add_argument(
        "--data-list",
        type=str,
        default=None,
        help="Path to file list (overrides config).",
    )
    parser.add_argument(
        "--entry",
        type=int,
        default=0,
        help="Event index to visualize",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for model inference",
    )
    parser.add_argument(
        "--grid-size",
        type=float,
        default=None,
        help="Grid size for voxelization (overrides config)",
    )
    parser.add_argument(
        "--marker-size",
        type=float,
        default=2.0,
        help="Size of scatter plot markers",
    )
    parser.add_argument(
        "--opacity",
        type=float,
        default=0.8,
        help="Opacity of scatter plot markers",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="Port for Dash server",
    )
    parser.add_argument(
        "--colormap",
        type=str,
        default="RdYlBu_r",
        help="Colormap for similarity visualization (e.g., 'RdYlBu_r', 'viridis', 'plasma')",
    )
    parser.add_argument(
        "--load-features",
        type=str,
        default=None,
        help="Optional: load pre-extracted features from .npz file",
    )
    parser.add_argument(
        "--save-features",
        type=str,
        default=None,
        help="Optional: save extracted features to .npz file",
    )
    parser.add_argument(
        "--label-threshold-factor",
        type=float,
        default=4.0,
        help="Factor multiplied by grid_size to get label assignment threshold (default: 4.0)",
    )
    return parser.parse_args()


def build_inference_transform(cfg, grid_size=None):
    """
    Build a simplified transform pipeline for inference.
    """
    if grid_size is None:
        grid_size = cfg.get("grid_size", 0.25)

    transform_list = [
        dict(
            type="GridSample",
            grid_size=grid_size,
            hash_type="fnv",
            mode="train",
            return_grid_coord=True,
        ),
        dict(type="ToTensor"),
        dict(type="Update", keys_dict={"grid_size": grid_size}),
        dict(
            type="Collect",
            keys=("coord", "grid_coord", "segment", "name", "grid_size"),
            offset_keys_dict=dict(offset="coord"),
            feat_keys=("strength", "color"),
        ),
    ]

    return transform_list


def load_model(cfg, checkpoint_path, device, random_init=False):
    """
    Load Sonata model from checkpoint or with random initialization.
    """
    print(f"Building model: {cfg.model.type}")
    model = build_model(cfg.model)

    if random_init:
        print("Using RANDOMLY INITIALIZED model weights (no checkpoint loaded)")
    else:
        if checkpoint_path is None:
            raise ValueError("checkpoint_path is required when random_init=False")
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v

        model.load_state_dict(new_state_dict, strict=False)
        print(f"Model loaded successfully")

    model = model.to(device)
    model.eval()

    return model


@torch.no_grad()
def extract_features(model, data_dict, device):
    """
    Extract encoder features from Sonata model.
    """
    input_dict = {
        "coord": data_dict["coord"].to(device),
        "feat": data_dict["feat"].to(device),
        "offset": data_dict["offset"].to(device),
    }

    if "grid_size" in data_dict:
        grid_size = data_dict["grid_size"]
        if isinstance(grid_size, torch.Tensor):
            input_dict["grid_size"] = grid_size[0].item() if grid_size.numel() > 1 else grid_size.item()
        else:
            input_dict["grid_size"] = grid_size
    else:
        input_dict["grid_size"] = 0.25

    if "grid_coord" in data_dict:
        input_dict["grid_coord"] = data_dict["grid_coord"].to(device)

    result = model.forward(input_dict, return_point=True)

    return result["point"]


def collate_fn(batch):
    """
    Custom collate function for inference.
    """
    coords = []
    feats = []
    segments = []
    names = []
    offset = 0
    offsets = []

    for data in batch:
        n = data["coord"].shape[0]
        coords.append(data["coord"])
        feats.append(data["feat"])
        if "segment" in data:
            segments.append(data["segment"])
        names.append(data.get("name", "unknown"))
        offset += n
        offsets.append(offset)

    result = {
        "coord": torch.cat(coords, dim=0),
        "feat": torch.cat(feats, dim=0),
        "offset": torch.tensor(offsets, dtype=torch.long),
        "name": names,
    }

    if segments:
        result["segment"] = torch.cat(segments, dim=0)

    if "grid_coord" in batch[0]:
        grid_coords = [data["grid_coord"] for data in batch]
        result["grid_coord"] = torch.cat(grid_coords, dim=0)

    if "grid_size" in batch[0]:
        result["grid_size"] = batch[0]["grid_size"]

    return result


def compute_cosine_similarity(features, reference_idx):
    """
    Compute cosine similarity between a reference point and all other points.

    Args:
        features: (N, D) array of feature vectors
        reference_idx: Index of the reference point

    Returns:
        (N,) array of cosine similarities in range [-1, 1]
    """
    # Normalize features
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)  # Avoid division by zero
    normalized = features / norms

    # Get reference vector
    ref_vector = normalized[reference_idx]

    # Compute cosine similarity (dot product of normalized vectors)
    similarities = np.dot(normalized, ref_vector)

    return similarities


def similarity_to_colors(similarities, colormap_name='RdYlBu_r'):
    """
    Convert similarity values to RGB colors using a colormap.

    Args:
        similarities: (N,) array of similarity values in [-1, 1]
        colormap_name: Name of matplotlib colormap

    Returns:
        List of color strings for plotly
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    # Normalize similarities from [-1, 1] to [0, 1]
    normalized = (similarities + 1) / 2
    normalized = np.clip(normalized, 0, 1)

    # Get colormap
    cmap = plt.get_cmap(colormap_name)

    # Map to colors
    colors = cmap(normalized)

    # Convert to plotly color strings
    color_strings = [
        f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})'
        for c in colors
    ]

    return color_strings, normalized


def create_dash_app(coords, features, labels, class_names, args):
    """
    Create interactive Dash app for similarity visualization.
    """
    import dash
    from dash import dcc, html
    from dash.dependencies import Input, Output, State
    import plotly.graph_objects as go
    import json

    # Import detector outline
    try:
        from lardly.detectoroutline import DetectorOutline
        detdata = DetectorOutline()
        detector_traces = detdata.getlines()
    except ImportError:
        print("Warning: lardly not found, skipping detector outline")
        detector_traces = []

    # Ghost class index
    ghost_idx = GHOST_LABEL

    # Precompute masks
    ghost_mask = labels == ghost_idx
    non_ghost_mask = labels != ghost_idx

    # Initial colors (gray, no selection)
    initial_colors = ['rgb(150,150,150)'] * len(coords)

    # Normalize features once for faster similarity computation
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    normalized_features = features / norms

    # Create app
    app = dash.Dash(
        __name__,
        meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    )

    # Display filter options
    filter_options = [
        {"label": "All Points", "value": "all"},
        {"label": "Non-Ghost Only", "value": "non_ghost"},
        {"label": "Ghost Only", "value": "ghost"},
    ]

    init_label = "RANDOM INIT" if args.random_init else "trained"

    # Layout
    app.layout = html.Div([
        # Header
        html.Div([
            html.H3(f"Sonata Feature Similarity Visualization [{init_label}]",
                    style={"color": "white", "textAlign": "center", "marginBottom": "10px"}),
            html.P("Click on any point to see similarity of all other points to it",
                   style={"color": "#aaa", "textAlign": "center", "marginBottom": "10px", "fontSize": "14px"}),
            html.Div([
                html.Label("Display Filter:", style={"color": "white", "marginRight": "10px"}),
                dcc.Dropdown(
                    id="display-filter",
                    options=filter_options,
                    value="all",
                    style={"width": "200px", "display": "inline-block"},
                    clearable=False,
                ),
                html.Span(id="point-count", style={"color": "white", "marginLeft": "20px"}),
                html.Span(id="selected-info", style={"color": "#4CAF50", "marginLeft": "20px", "fontWeight": "bold"}),
            ], style={"display": "flex", "alignItems": "center", "justifyContent": "center", "marginBottom": "10px"}),
        ], style={"backgroundColor": "#141414", "padding": "10px"}),

        # Store for selected point
        dcc.Store(id="selected-point-idx", data=None),

        # 3D Plot
        html.Div([
            dcc.Graph(
                id="det3d",
                style={"height": "80vh"},
                config={"scrollZoom": True},
            )
        ]),

        # Colorbar legend
        html.Div([
            html.Div([
                html.Span("Low Similarity", style={"color": "white", "fontSize": "12px"}),
                html.Div(style={
                    "width": "200px",
                    "height": "20px",
                    "background": "linear-gradient(to right, #313695, #4575b4, #74add1, #abd9e9, #e0f3f8, #ffffbf, #fee090, #fdae61, #f46d43, #d73027, #a50026)",
                    "margin": "0 10px",
                    "display": "inline-block",
                    "verticalAlign": "middle",
                }),
                html.Span("High Similarity", style={"color": "white", "fontSize": "12px"}),
            ], style={"textAlign": "center", "padding": "10px"}),
        ], style={"backgroundColor": "#141414"}),

    ], style={"backgroundColor": "#141414", "height": "100vh"})

    @app.callback(
        Output("selected-point-idx", "data"),
        Input("det3d", "clickData"),
        State("selected-point-idx", "data"),
        State("display-filter", "value"),
    )
    def update_selected_point(click_data, current_idx, filter_value):
        if click_data is None:
            return current_idx

        # Get clicked point info
        point_data = click_data.get("points", [{}])[0]

        # Check if it's a scatter point (not detector outline)
        curve_number = point_data.get("curveNumber", 0)

        # Detector traces come first, so scatter is after them
        if curve_number < len(detector_traces):
            return current_idx  # Clicked on detector, ignore

        # Get the point index within the displayed points
        point_number = point_data.get("pointNumber", None)
        if point_number is None:
            return current_idx

        # Map back to original index based on filter
        if filter_value == "all":
            return int(point_number)
        elif filter_value == "non_ghost":
            indices = np.where(non_ghost_mask)[0]
            if point_number < len(indices):
                return int(indices[point_number])
        elif filter_value == "ghost":
            indices = np.where(ghost_mask)[0]
            if point_number < len(indices):
                return int(indices[point_number])

        return current_idx

    @app.callback(
        [Output("det3d", "figure"),
         Output("point-count", "children"),
         Output("selected-info", "children")],
        [Input("display-filter", "value"),
         Input("selected-point-idx", "data")]
    )
    def update_figure(filter_value, selected_idx):
        # Determine which points to show
        if filter_value == "all":
            mask = np.ones(len(coords), dtype=bool)
        elif filter_value == "non_ghost":
            mask = non_ghost_mask
        elif filter_value == "ghost":
            mask = ghost_mask
        else:
            mask = np.ones(len(coords), dtype=bool)

        n_points = mask.sum()

        # Compute colors based on selection
        if selected_idx is not None and 0 <= selected_idx < len(coords):
            # Compute cosine similarity to selected point
            ref_vector = normalized_features[selected_idx]
            similarities = np.dot(normalized_features, ref_vector)

            # Convert to colors
            color_values, _ = similarity_to_colors(similarities, args.colormap)
            colors_to_use = [color_values[i] for i in range(len(coords)) if mask[i]]

            # Get selected point info
            selected_class = class_names[labels[selected_idx]] if labels[selected_idx] < len(class_names) else "unknown"
            selected_info = f"Selected: Point #{selected_idx} ({selected_class}) at ({coords[selected_idx, 0]:.1f}, {coords[selected_idx, 1]:.1f}, {coords[selected_idx, 2]:.1f})"
        else:
            # No selection - gray colors
            colors_to_use = ['rgb(150,150,150)'] * n_points
            selected_info = "No point selected - click on a point"

        # Create scatter trace
        masked_coords = coords[mask]

        # Custom data for hover
        masked_labels = labels[mask]
        masked_indices = np.where(mask)[0]

        hover_text = [
            f"Point #{idx}<br>Class: {class_names[lbl] if lbl < len(class_names) else 'unknown'}<br>"
            f"Pos: ({masked_coords[i, 0]:.1f}, {masked_coords[i, 1]:.1f}, {masked_coords[i, 2]:.1f})"
            for i, (idx, lbl) in enumerate(zip(masked_indices, masked_labels))
        ]

        scatter_trace = go.Scatter3d(
            x=masked_coords[:, 0],
            y=masked_coords[:, 1],
            z=masked_coords[:, 2],
            mode="markers",
            marker=dict(
                color=colors_to_use,
                size=args.marker_size,
                opacity=args.opacity,
            ),
            hovertext=hover_text,
            hoverinfo="text",
            name="Points",
        )

        # Add marker for selected point if visible
        traces = []

        # Add detector traces
        for trace in detector_traces:
            traces.append(go.Scatter3d(
                x=trace["x"],
                y=trace["y"],
                z=trace["z"],
                mode="lines",
                line=dict(color=trace["line"]["color"], width=2),
                name="Detector",
                showlegend=False,
                hoverinfo="skip",
            ))

        # Add main scatter
        traces.append(scatter_trace)

        # Add selected point marker (larger, with border effect)
        if selected_idx is not None and 0 <= selected_idx < len(coords):
            # Check if selected point is in current filter
            if mask[selected_idx]:
                traces.append(go.Scatter3d(
                    x=[coords[selected_idx, 0]],
                    y=[coords[selected_idx, 1]],
                    z=[coords[selected_idx, 2]],
                    mode="markers",
                    marker=dict(
                        color='rgb(0, 255, 0)',
                        size=args.marker_size * 4,
                        opacity=1.0,
                        symbol='diamond',
                        line=dict(color='white', width=2),
                    ),
                    name="Selected",
                    hoverinfo="skip",
                ))

        # Layout
        axis_template = {
            "showbackground": True,
            "backgroundcolor": "#141414",
            "gridcolor": "rgb(100, 100, 100)",
            "zerolinecolor": "rgb(150, 150, 150)",
        }

        layout = go.Layout(
            height=800,
            margin=dict(t=0, b=0, l=0, r=0),
            font=dict(size=12, color="white"),
            showlegend=False,
            plot_bgcolor="#141414",
            paper_bgcolor="#141414",
            scene=dict(
                xaxis=dict(**axis_template, title="X (drift)"),
                yaxis=dict(**axis_template, title="Y (vertical)"),
                zaxis=dict(**axis_template, title="Z (beam)"),
                aspectratio=dict(x=1, y=1, z=4),
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5), up=dict(x=0, y=1, z=0)),
            ),
        )

        fig = go.Figure(data=traces, layout=layout)

        point_count_text = f"Showing {n_points:,} points"

        return fig, point_count_text, selected_info

    return app


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Class names for LArTPC (from shared module)
    class_names = CLASS_NAMES

    # Check if loading pre-extracted features
    if args.load_features is not None:
        print(f"Loading pre-extracted features from {args.load_features}")
        data = np.load(args.load_features)
        features = data["features"]
        labels = data["labels"]
        coords = data["coords"]
        print(f"Loaded {len(features)} points with {features.shape[1]} feature dimensions")
    else:
        # Load config
        print(f"Loading config: {args.config}")
        cfg = Config.fromfile(args.config)

        # Override grid size if specified
        grid_size = args.grid_size if args.grid_size is not None else cfg.get("grid_size", 0.25)

        # Build inference transform
        transform = build_inference_transform(cfg, grid_size)

        # Determine data list file
        data_list = args.data_list

        print(f"Building dataset with data list: {data_list}")

        # Build dataset
        from pointcept.datasets.lartpc import LArTPCDataset

        dataset = LArTPCDataset(
            split="val",
            data_list_file=data_list,
            transform=transform,
            use_reco_coords=True,
            use_edep_as_strength=True,
            label_mode="pid",
            coord_scale=1.0,
            log_transform_edep=True,
            include_ghosts=True,
            exclude_other=True,
            true_points_only=False,
            test_mode=False,
            loop=1,
        )

        print(f"Dataset size: {len(dataset)} events")
        print(f"Loading event {args.entry}...")

        # Load model
        model = load_model(cfg, args.checkpoint, args.device, random_init=args.random_init)

        # Get single event
        data = dataset[args.entry]
        batch_data = collate_fn([data])

        n_points_input = batch_data["coord"].shape[0]
        print(f"Input points: {n_points_input}")

        # Extract features
        point = extract_features(model, batch_data, args.device)

        features = point.feat.cpu().numpy()
        coords = point.coord.cpu().numpy()
        n_points_output = features.shape[0]

        print(f"Output points: {n_points_output}, feature dim: {features.shape[1]}")

        # Assign labels to output points using shared function
        if "segment" in batch_data:
            input_coords = batch_data["coord"].cpu().numpy()
            input_labels = batch_data["segment"].cpu().numpy()

            labels = assign_labels_to_output_points(
                output_coords=coords,
                input_coords=input_coords,
                input_labels=input_labels,
                grid_size=grid_size,
                threshold_factor=args.label_threshold_factor,
                ghost_label=GHOST_LABEL,
                use_gpu=True,
                verbose=True,
            )
        else:
            labels = np.zeros(n_points_output, dtype=np.int64)

        # Save features if requested
        if args.save_features is not None:
            print(f"Saving features to {args.save_features}")
            np.savez(
                args.save_features,
                features=features,
                labels=labels,
                coords=coords,
            )

    # Print class distribution
    print("\nClass distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for cls_idx, count in zip(unique, counts):
        if 0 <= cls_idx < len(class_names):
            print(f"  {class_names[cls_idx]}: {count} ({100*count/len(labels):.1f}%)")

    print(f"\nTotal points: {len(coords)}")

    # Create visualization
    print(f"\nStarting Dash server on port {args.port}...")
    print(f"Open http://localhost:{args.port} in your browser")
    print("Click on any point to visualize similarity to other points")
    app = create_dash_app(coords, features, labels, class_names, args)
    app.run(debug=True, port=args.port)


if __name__ == "__main__":
    main()
