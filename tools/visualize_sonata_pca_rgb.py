"""
PCA-RGB 3D Visualization of Sonata Encoder Features

Extracts encoder features from a trained Sonata model and creates an
interactive 3D visualization where point colors are derived from the
top 3 principal components of the feature embeddings mapped to RGB.

Uses Plotly/Dash for interactive visualization with class filtering.

Usage:
    python tools/visualize_sonata_pca_rgb.py \
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
        description="PCA-RGB 3D visualization of Sonata encoder features"
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
        "--color-mode",
        type=str,
        choices=["pca", "kmeans"],
        default="pca",
        help="Color mode: 'pca' (top 3 PCs → RGB) or 'kmeans' (cluster colors)",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=10,
        help="Number of clusters for K-means color mode",
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
        "--save-html",
        type=str,
        default=None,
        help="Optional: save static HTML file instead of running server",
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


def features_to_pca_rgb(features, n_components=3):
    """
    Apply PCA to features and map top 3 components to RGB colors.

    Uses cuML's GPU-accelerated PCA implementation.
    Each component is normalized to [0, 255] range.
    """
    print(f"Applying PCA to {features.shape[0]} points with {features.shape[1]} features...")

    # Use cuML's GPU-accelerated PCA
    from cuml.decomposition import PCA as cumlPCA
    print("Using cuML (GPU) PCA")

    # cuML expects float32
    features_f32 = features.astype(np.float32)

    # Fit PCA
    pca = cumlPCA(n_components=n_components, svd_solver='full')
    pca_features = pca.fit_transform(features_f32)

    # Convert cuML output to numpy if needed
    if hasattr(pca_features, 'to_numpy'):
        pca_features = pca_features.to_numpy()
    elif hasattr(pca_features, 'get'):
        pca_features = pca_features.get()

    # Get explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_
    if hasattr(explained_variance_ratio, 'to_numpy'):
        explained_variance_ratio = explained_variance_ratio.to_numpy()
    elif hasattr(explained_variance_ratio, 'get'):
        explained_variance_ratio = explained_variance_ratio.get()

    print(f"Explained variance ratio: {explained_variance_ratio}")
    print(f"Total explained variance: {sum(explained_variance_ratio):.3f}")

    # Normalize each component to [0, 1]
    rgb = np.zeros((len(pca_features), 3), dtype=np.float32)
    for i in range(3):
        component = pca_features[:, i]
        # Use percentile-based normalization for robustness to outliers
        p_low, p_high = np.percentile(component, [2, 98])
        normalized = (component - p_low) / (p_high - p_low + 1e-8)
        normalized = np.clip(normalized, 0, 1)
        rgb[:, i] = normalized

    # Convert to 0-255 range
    rgb_255 = (rgb * 255).astype(np.uint8)

    return rgb_255, pca


def features_to_kmeans_colors(features, n_clusters=10, seed=42):
    """
    Apply K-means clustering to features and assign distinct colors to each cluster.

    Uses cuML's GPU-accelerated K-means implementation.
    Returns RGB colors based on cluster assignments.
    """
    print(f"Applying K-means clustering ({n_clusters} clusters) to {features.shape[0]} points...")

    # Use cuML's GPU-accelerated K-means
    from cuml.cluster import KMeans as cumlKMeans
    print("Using cuML (GPU) K-means")

    # cuML expects float32
    features_f32 = features.astype(np.float32)

    # Fit K-means
    kmeans = cumlKMeans(n_clusters=n_clusters, random_state=seed, max_iter=300)
    cluster_labels = kmeans.fit_predict(features_f32)

    # Convert cuML output to numpy if needed
    if hasattr(cluster_labels, 'to_numpy'):
        cluster_labels = cluster_labels.to_numpy()
    elif hasattr(cluster_labels, 'get'):
        cluster_labels = cluster_labels.get()

    # Get inertia (sum of squared distances to cluster centers)
    inertia = kmeans.inertia_
    if hasattr(inertia, 'item'):
        inertia = inertia.item()
    print(f"K-means inertia: {inertia:.2f}")

    # Count points per cluster
    unique, counts = np.unique(cluster_labels, return_counts=True)
    print(f"Cluster sizes: min={counts.min()}, max={counts.max()}, median={np.median(counts):.0f}")

    # Generate distinct colors for each cluster using a qualitative colormap
    # Use a combination of hue rotation and saturation/value variation for distinctiveness
    rgb_colors = np.zeros((len(cluster_labels), 3), dtype=np.uint8)

    # Define a set of distinct colors (extended qualitative palette)
    # These are designed to be maximally distinguishable
    base_colors = [
        [228, 26, 28],    # red
        [55, 126, 184],   # blue
        [77, 175, 74],    # green
        [152, 78, 163],   # purple
        [255, 127, 0],    # orange
        [255, 255, 51],   # yellow
        [166, 86, 40],    # brown
        [247, 129, 191],  # pink
        [153, 153, 153],  # gray
        [0, 206, 209],    # dark turquoise
        [138, 43, 226],   # blue violet
        [0, 100, 0],      # dark green
        [255, 20, 147],   # deep pink
        [0, 0, 139],      # dark blue
        [184, 134, 11],   # dark goldenrod
        [139, 0, 0],      # dark red
        [0, 139, 139],    # dark cyan
        [85, 107, 47],    # dark olive green
        [255, 140, 0],    # dark orange
        [148, 0, 211],    # dark violet
    ]

    # If we need more colors than base_colors, generate them using HSV
    if n_clusters > len(base_colors):
        import colorsys
        for i in range(len(base_colors), n_clusters):
            hue = (i * 0.618033988749895) % 1.0  # Golden ratio for good distribution
            saturation = 0.7 + (i % 3) * 0.1
            value = 0.9 - (i % 4) * 0.1
            r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
            base_colors.append([int(r * 255), int(g * 255), int(b * 255)])

    # Assign colors to points based on cluster labels
    for i in range(n_clusters):
        mask = cluster_labels == i
        rgb_colors[mask] = base_colors[i]

    return rgb_colors, kmeans, cluster_labels


def create_dash_app(coords, rgb_colors, labels, class_names, args):
    """
    Create interactive Dash app for 3D visualization.
    """
    import dash
    from dash import dcc, html
    from dash.dependencies import Input, Output
    import plotly.graph_objects as go

    # Import detector outline
    try:
        from lardly.detectoroutline import DetectorOutline
        detdata = DetectorOutline()
        detector_traces = detdata.getlines()
    except ImportError:
        print("Warning: lardly not found, skipping detector outline")
        detector_traces = []

    # Class info
    num_classes = NUM_CLASSES
    ghost_idx = GHOST_LABEL

    # Create color strings for plotly
    color_strings = [f'rgb({r},{g},{b})' for r, g, b in rgb_colors]

    # Precompute masks for each class
    class_masks = {}
    for i, name in enumerate(class_names):
        class_masks[name] = labels == i
    class_masks["all"] = np.ones(len(labels), dtype=bool)
    class_masks["no_ghost"] = labels != ghost_idx

    # Create app
    app = dash.Dash(
        __name__,
        meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    )

    # Class filter options
    filter_options = [
        {"label": "All Classes", "value": "all"},
        {"label": "No Ghost", "value": "no_ghost"},
    ] + [{"label": name.capitalize(), "value": name} for name in class_names]

    # Layout
    # Get color mode label
    color_mode_label = getattr(args, 'color_mode_label', args.color_mode.upper())

    app.layout = html.Div([
        html.Div([
            html.H3(f"Sonata Feature Visualization ({color_mode_label})",
                    style={"color": "white", "textAlign": "center", "marginBottom": "10px"}),
            html.Div([
                html.Label("Class Filter:", style={"color": "white", "marginRight": "10px"}),
                dcc.Dropdown(
                    id="class-filter",
                    options=filter_options,
                    value="all",
                    style={"width": "200px", "display": "inline-block"},
                    clearable=False,
                ),
                html.Span(id="point-count", style={"color": "white", "marginLeft": "20px"}),
            ], style={"display": "flex", "alignItems": "center", "justifyContent": "center", "marginBottom": "10px"}),
        ], style={"backgroundColor": "#141414", "padding": "10px"}),

        html.Div([
            dcc.Graph(
                id="det3d",
                style={"height": "85vh"},
                config={"scrollZoom": True},
            )
        ]),
    ], style={"backgroundColor": "#141414", "height": "100vh"})

    @app.callback(
        [Output("det3d", "figure"),
         Output("point-count", "children")],
        [Input("class-filter", "value")]
    )
    def update_figure(filter_value):
        mask = class_masks[filter_value]
        n_points = mask.sum()

        # Create scatter trace for filtered points
        scatter_trace = {
            "type": "scatter3d",
            "x": coords[mask, 0],
            "y": coords[mask, 1],
            "z": coords[mask, 2],
            "mode": "markers",
            "name": "PCA-RGB Features",
            "marker": {
                "color": [color_strings[i] for i in np.where(mask)[0]],
                "size": args.marker_size,
                "opacity": args.opacity,
            },
            "hovertemplate": (
                "<b>x</b>: %{x:.1f}<br>"
                "<b>y</b>: %{y:.1f}<br>"
                "<b>z</b>: %{z:.1f}<br>"
            ),
        }

        traces = detector_traces + [scatter_trace]

        # Layout
        axis_template = {
            "showbackground": True,
            "backgroundcolor": "#141414",
            "gridcolor": "rgb(100, 100, 100)",
            "zerolinecolor": "rgb(150, 150, 150)",
        }

        layout = {
            "height": 800,
            "margin": {"t": 0, "b": 0, "l": 0, "r": 0},
            "font": {"size": 12, "color": "white"},
            "showlegend": False,
            "plot_bgcolor": "#141414",
            "paper_bgcolor": "#141414",
            "scene": {
                "xaxis": {**axis_template, "title": "X (drift)"},
                "yaxis": {**axis_template, "title": "Y (vertical)"},
                "zaxis": {**axis_template, "title": "Z (beam)"},
                "aspectratio": {"x": 1, "y": 1, "z": 4},
                "camera": {"eye": {"x": 1.5, "y": 1.5, "z": 1.5},
                           "up": {"x": 0, "y": 1, "z": 0}},
            },
        }

        init_label = "RANDOM INIT" if args.random_init else "trained"
        point_count_text = f"Showing {n_points:,} points [{init_label}]"

        return {"data": traces, "layout": layout}, point_count_text

    return app


def create_static_figure(coords, rgb_colors, labels, class_names, args):
    """
    Create a static plotly figure for HTML export.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Import detector outline
    try:
        from lardly.detectoroutline import DetectorOutline
        detdata = DetectorOutline()
        detector_traces = detdata.getlines()
    except ImportError:
        print("Warning: lardly not found, skipping detector outline")
        detector_traces = []

    ghost_idx = GHOST_LABEL
    color_strings = [f'rgb({r},{g},{b})' for r, g, b in rgb_colors]

    # Create figure with dropdown for class filtering
    fig = go.Figure()

    # Add detector outline
    for trace in detector_traces:
        fig.add_trace(go.Scatter3d(
            x=trace["x"],
            y=trace["y"],
            z=trace["z"],
            mode="lines",
            line=dict(color=trace["line"]["color"], width=2),
            name="Detector",
            showlegend=False,
        ))

    # Add all points trace
    fig.add_trace(go.Scatter3d(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        mode="markers",
        marker=dict(
            color=color_strings,
            size=args.marker_size,
            opacity=args.opacity,
        ),
        name="PCA-RGB Features",
        hovertemplate=(
            "<b>x</b>: %{x:.1f}<br>"
            "<b>y</b>: %{y:.1f}<br>"
            "<b>z</b>: %{z:.1f}<br>"
        ),
    ))

    # Create visibility arrays for dropdown
    n_detector_traces = len(detector_traces)

    # Create buttons for filtering
    buttons = []

    # All points
    buttons.append(dict(
        label="All Classes",
        method="update",
        args=[{"visible": [True] * (n_detector_traces + 1)}]
    ))

    # No ghost
    no_ghost_mask = labels != ghost_idx
    no_ghost_coords = coords[no_ghost_mask]
    no_ghost_colors = [color_strings[i] for i in np.where(no_ghost_mask)[0]]

    fig.add_trace(go.Scatter3d(
        x=no_ghost_coords[:, 0],
        y=no_ghost_coords[:, 1],
        z=no_ghost_coords[:, 2],
        mode="markers",
        marker=dict(
            color=no_ghost_colors,
            size=args.marker_size,
            opacity=args.opacity,
        ),
        name="No Ghost",
        visible=False,
        hovertemplate=(
            "<b>x</b>: %{x:.1f}<br>"
            "<b>y</b>: %{y:.1f}<br>"
            "<b>z</b>: %{z:.1f}<br>"
        ),
    ))

    buttons.append(dict(
        label="No Ghost",
        method="update",
        args=[{"visible": [True] * n_detector_traces + [False, True] + [False] * len(class_names)}]
    ))

    # Individual classes
    for i, name in enumerate(class_names):
        class_mask = labels == i
        class_coords = coords[class_mask]
        class_colors = [color_strings[j] for j in np.where(class_mask)[0]]

        fig.add_trace(go.Scatter3d(
            x=class_coords[:, 0],
            y=class_coords[:, 1],
            z=class_coords[:, 2],
            mode="markers",
            marker=dict(
                color=class_colors,
                size=args.marker_size,
                opacity=args.opacity,
            ),
            name=name.capitalize(),
            visible=False,
            hovertemplate=(
                "<b>x</b>: %{x:.1f}<br>"
                "<b>y</b>: %{y:.1f}<br>"
                "<b>z</b>: %{z:.1f}<br>"
            ),
        ))

        visibility = [True] * n_detector_traces + [False] * (2 + len(class_names))
        visibility[n_detector_traces + 2 + i] = True
        buttons.append(dict(
            label=name.capitalize(),
            method="update",
            args=[{"visible": visibility}]
        ))

    # Update layout
    init_label = "RANDOM INIT" if args.random_init else "trained"

    color_mode_label = getattr(args, 'color_mode_label', args.color_mode.upper())

    fig.update_layout(
        title=f"Sonata Features ({color_mode_label}) [{init_label}]",
        updatemenus=[
            dict(
                active=0,
                buttons=buttons,
                direction="down",
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.1,
                yanchor="top",
            )
        ],
        scene=dict(
            xaxis=dict(title="X (drift)", backgroundcolor="#141414"),
            yaxis=dict(title="Y (vertical)", backgroundcolor="#141414"),
            zaxis=dict(title="Z (beam)", backgroundcolor="#141414"),
            aspectratio=dict(x=1, y=1, z=4),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5), up=dict(x=0, y=1, z=0)),
        ),
        paper_bgcolor="#141414",
        font=dict(color="white"),
        height=800,
    )

    return fig


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

    # Apply color mapping based on selected mode
    if args.color_mode == "pca":
        rgb_colors, _ = features_to_pca_rgb(features)
        color_mode_label = "PCA"
    elif args.color_mode == "kmeans":
        rgb_colors, _, cluster_labels = features_to_kmeans_colors(
            features, n_clusters=args.n_clusters, seed=args.seed
        )
        color_mode_label = f"K-means (k={args.n_clusters})"
    else:
        raise ValueError(f"Unknown color mode: {args.color_mode}")

    # Store color mode label for visualization titles
    args.color_mode_label = color_mode_label

    # Print class distribution
    print("\nClass distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for cls_idx, count in zip(unique, counts):
        if 0 <= cls_idx < len(class_names):
            print(f"  {class_names[cls_idx]}: {count} ({100*count/len(labels):.1f}%)")

    print(f"\nTotal points: {len(coords)}")

    # Create visualization
    if args.save_html is not None:
        print(f"Creating static figure and saving to {args.save_html}...")
        fig = create_static_figure(coords, rgb_colors, labels, class_names, args)
        fig.write_html(args.save_html)
        print(f"Saved to {args.save_html}")
    else:
        print(f"\nStarting Dash server on port {args.port}...")
        print(f"Open http://localhost:{args.port} in your browser")
        app = create_dash_app(coords, rgb_colors, labels, class_names, args)
        app.run(debug=True, port=args.port)


if __name__ == "__main__":
    main()
