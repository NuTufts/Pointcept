"""
Interactive Semantic Segmentation Visualization

Visualizes semantic segmentation predictions from a trained model with an
interactive 3D display. Shows predicted class labels and allows viewing
per-class scores through a dropdown menu.

Uses Plotly/Dash for interactive visualization with event navigation.

Usage:
    python tools/visualize_semseg.py \
        --config configs/lartpc/semseg-sonata-v1m1-lartpc-v3-finetune.py \
        --checkpoint exp/lartpc/semseg/model_best.pth \
        --data-list /path/to/val_split.txt \
        --entry 0

Author: Generated for LArTPC analysis
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add lardly to path for detector outline
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
    CLASS_NAMES,
    CLASS_COLORS,
    CLASS_COLORS_RGBA,
    NUM_CLASSES,
    GHOST_LABEL,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive semantic segmentation visualization"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file (e.g., configs/lartpc/semseg-sonata-v1m1-lartpc-v3-finetune.py)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pth file)",
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
        help="Initial event index to visualize",
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
        "--include-ghosts",
        action="store_true",
        default=False,
        help="Include ghost points in visualization",
    )
    parser.add_argument(
        "--true-points-only",
        action="store_true",
        default=False,
        help="Only show true energy deposition points (if MC)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        default=False,
        help="Force CPU-only inference (ignores --device). NOTE: May not work with spconv-based models.",
    )
    return parser.parse_args()


def build_inference_transform(cfg, grid_size=None):
    """
    Build a simplified transform pipeline for inference.
    """
    if grid_size is None:
        grid_size = cfg.get("grid_size", 0.25)

    # Get feat_keys from config if available
    feat_keys = ("strength",)
    if hasattr(cfg, 'data') and 'val' in cfg.data:
        for t in cfg.data.val.get('transform', []):
            if t.get('type') == 'Collect' and 'feat_keys' in t:
                feat_keys = t['feat_keys']
                break

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
            feat_keys=feat_keys,
        ),
    ]

    return transform_list


class LArTPCDatasetWithSSNet:
    """
    Thin wrapper around LArTPCDataset.

    Note: ssnet_label is now loaded directly by LArTPCDataset.get_data(),
    so this wrapper just delegates to the base dataset.
    """
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        # Copy attributes for compatibility
        self.data_list = base_dataset.data_list

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # ssnet_label is now loaded by LArTPCDataset.get_data() and goes through transforms
        return self.base_dataset[idx]


def check_model_uses_spconv(model):
    """
    Check if the model uses spconv layers.

    Returns:
        bool: True if model contains spconv layers
        int: Number of spconv layers found
    """
    spconv_count = 0
    try:
        import spconv.pytorch as spconv
        spconv_types = (
            spconv.SparseConv3d,
            spconv.SubMConv3d,
            spconv.SparseConvTranspose3d,
            spconv.SparseInverseConv3d,
        )
        for module in model.modules():
            if isinstance(module, spconv_types):
                spconv_count += 1
    except ImportError:
        # If spconv not installed, check by module name
        for name, module in model.named_modules():
            module_type = type(module).__name__
            if 'SparseConv' in module_type or 'SubMConv' in module_type:
                spconv_count += 1

    return spconv_count > 0, spconv_count


def switch_spconv_to_native_algo(model):
    """
    Switch all spconv layers to use ConvAlgo.Native algorithm.

    This is required for CPU inference since MaskImplicitGemm is CUDA-only.
    Native algorithm produces mathematically equivalent results.

    Returns:
        int: Number of layers switched
    """
    try:
        from spconv.core import ConvAlgo
    except ImportError:
        print("WARNING: Could not import spconv.core.ConvAlgo")
        return 0

    count = 0
    for name, module in model.named_modules():
        # Check if this is a spconv layer with an algo attribute
        if hasattr(module, 'algo'):
            old_algo = module.algo
            module.algo = ConvAlgo.Native
            count += 1
            # Only print first few to avoid spam
            if count <= 3:
                print(f"  Switched {name}: {old_algo} -> ConvAlgo.Native")
            elif count == 4:
                print(f"  ... (switching remaining layers)")

    return count


def load_model(cfg, checkpoint_path, device):
    """
    Load semantic segmentation model from checkpoint.

    Returns:
        model: Loaded model
        num_classes: Actual number of classes from checkpoint
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # Remove "module." prefix if present (from DDP training)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    # Detect number of classes from checkpoint
    checkpoint_num_classes = None
    for key in ["seg_head.weight", "seg_head.bias"]:
        if key in new_state_dict:
            checkpoint_num_classes = new_state_dict[key].shape[0]
            break

    config_num_classes = cfg.model.num_classes
    if checkpoint_num_classes is not None and checkpoint_num_classes != config_num_classes:
        print(f"WARNING: Config specifies {config_num_classes} classes but checkpoint has {checkpoint_num_classes} classes")
        print(f"         Overriding config to use {checkpoint_num_classes} classes from checkpoint")
        cfg.model.num_classes = checkpoint_num_classes
        # Also clear class_priors since it won't match
        if hasattr(cfg.model, 'class_priors') and cfg.model.class_priors is not None:
            print("         Clearing class_priors (length mismatch)")
            cfg.model.class_priors = None

    print(f"Building model: {cfg.model.type} with {cfg.model.num_classes} classes")
    model = build_model(cfg.model)

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        print(f"Missing keys: {missing}")
    if unexpected:
        print(f"Unexpected keys: {unexpected}")
    print("Model loaded successfully")

    # Switch spconv algorithm to Native for CPU inference
    is_cpu = device == "cpu" or (isinstance(device, torch.device) and device.type == "cpu")
    if is_cpu:
        print("Switching spconv layers to ConvAlgo.Native for CPU inference...")
        n_switched = switch_spconv_to_native_algo(model)
        print(f"Switched {n_switched} spconv layers to Native algorithm")

    model = model.to(device)
    model.eval()

    return model, cfg.model.num_classes


def collate_fn(batch, in_channels=None):
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
        if in_channels is not None:
            feats.append(data["feat"][:, :in_channels])
        else:
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


@torch.no_grad()
def run_inference(model, data_dict, device):
    """
    Run semantic segmentation inference with timing and memory measurement.

    Returns:
        seg_logits: (N, num_classes) tensor of class logits
        coords: (N, 3) tensor of point coordinates
        inference_stats: dict with timing and memory info
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

    if "grid_coord" in data_dict:
        input_dict["grid_coord"] = data_dict["grid_coord"].to(device)

    # Prepare for memory measurement
    inference_stats = {}
    is_cuda = device.type == "cuda" if isinstance(device, torch.device) else str(device).startswith("cuda")

    if is_cuda:
        # Reset GPU memory stats before inference
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    # Measure forward time
    if is_cuda:
        # Use CUDA events for accurate GPU timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        result = model(input_dict)
        end_event.record()
        torch.cuda.synchronize(device)
        forward_time_ms = start_event.elapsed_time(end_event)
    else:
        # Use time.perf_counter for CPU timing
        start_time = time.perf_counter()
        result = model(input_dict)
        end_time = time.perf_counter()
        forward_time_ms = (end_time - start_time) * 1000.0

    inference_stats["forward_time_ms"] = forward_time_ms
    inference_stats["device"] = str(device)

    # Measure memory usage
    if is_cuda:
        # Get peak GPU memory allocated during forward pass
        peak_memory_bytes = torch.cuda.max_memory_allocated(device)
        inference_stats["peak_memory_mb"] = peak_memory_bytes / (1024 * 1024)
        inference_stats["memory_type"] = "GPU"
    else:
        # For CPU, try to get process memory (less precise but useful)
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            inference_stats["peak_memory_mb"] = memory_info.rss / (1024 * 1024)
            inference_stats["memory_type"] = "CPU (RSS)"
        except ImportError:
            inference_stats["peak_memory_mb"] = None
            inference_stats["memory_type"] = "CPU (unavailable - install psutil)"

    seg_logits = result["seg_logits"]
    coords = data_dict["coord"]

    return seg_logits.cpu(), coords.cpu(), inference_stats


def get_class_colors_for_predictions(predictions, class_names):
    """
    Get color array based on predicted class labels.
    """
    colors = []
    for pred in predictions:
        if pred < len(CLASS_COLORS):
            colors.append(CLASS_COLORS[pred])
        else:
            colors.append("#000000")
    return colors


def score_to_colors(scores, colormap_name='viridis'):
    """
    Convert score values to RGB colors using a colormap.

    Args:
        scores: (N,) array of score values (will be normalized to [0, 1])
        colormap_name: Name of matplotlib colormap

    Returns:
        List of color strings for plotly
    """
    import matplotlib.pyplot as plt

    # Normalize scores to [0, 1]
    min_val = scores.min()
    max_val = scores.max()
    if max_val - min_val > 1e-8:
        normalized = (scores - min_val) / (max_val - min_val)
    else:
        normalized = np.zeros_like(scores)

    # Get colormap
    cmap = plt.get_cmap(colormap_name)

    # Map to colors
    colors_rgba = cmap(normalized)

    # Convert to plotly color strings
    color_strings = [
        f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})'
        for c in colors_rgba
    ]

    return color_strings


def create_dash_app(dataset, model, device, cfg, args):
    """
    Create interactive Dash app for semantic segmentation visualization.
    """
    import dash
    from dash import dcc, html
    from dash.dependencies import Input, Output, State
    import plotly.graph_objects as go

    # Import detector outline
    try:
        from lardly.detectoroutline import DetectorOutline
        detdata = DetectorOutline()
        detector_traces = detdata.getlines()
    except ImportError:
        print("Warning: lardly not found, skipping detector outline")
        detector_traces = []

    # Get class names - use CLASS_NAMES from sonata_vis_utils as the authoritative source
    num_classes = cfg.model.num_classes

    # Always use CLASS_NAMES as the source of truth, sliced to num_classes
    # This ensures we have names for all classes even if config only had 5
    if num_classes <= len(CLASS_NAMES):
        class_names = CLASS_NAMES[:num_classes]
    else:
        # Fallback for more classes than we have names for
        class_names = CLASS_NAMES + [f"class_{i}" for i in range(len(CLASS_NAMES), num_classes)]

    print(f"Using {num_classes} classes: {class_names}")

    # Get input channels from config
    in_channels = cfg.model.backbone.in_channels

    # Create app
    app = dash.Dash(
        __name__,
        meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    )

    # Dropdown options for visualization mode
    viz_options = [
        {"label": "Predicted Class", "value": "predicted"},
        {"label": "True Class (if available)", "value": "true"},
    ] + [
        {"label": f"Score: {name}", "value": f"score_{i}"}
        for i, name in enumerate(class_names)
    ]

    # Layout
    app.layout = html.Div([
        # Header
        html.Div([
            html.H3("Semantic Segmentation Visualization",
                    style={"color": "white", "textAlign": "center", "marginBottom": "10px"}),

            # Event navigation
            html.Div([
                html.Button("< Previous", id="prev-btn", n_clicks=0,
                           style={"marginRight": "10px", "padding": "10px 20px"}),
                html.Span(id="event-info",
                         style={"color": "white", "fontSize": "16px", "marginRight": "10px"}),
                html.Button("Next >", id="next-btn", n_clicks=0,
                           style={"marginLeft": "10px", "padding": "10px 20px"}),
            ], style={"display": "flex", "alignItems": "center", "justifyContent": "center", "marginBottom": "15px"}),

            # Visualization controls
            html.Div([
                html.Label("Visualization Mode:", style={"color": "white", "marginRight": "10px"}),
                dcc.Dropdown(
                    id="viz-mode",
                    options=viz_options,
                    value="predicted",
                    style={"width": "250px", "display": "inline-block"},
                    clearable=False,
                ),
                html.Span(id="point-count", style={"color": "white", "marginLeft": "20px"}),
            ], style={"display": "flex", "alignItems": "center", "justifyContent": "center", "marginBottom": "10px"}),

            # Metrics display
            html.Div(id="metrics-info",
                    style={"color": "#4CAF50", "textAlign": "center", "marginBottom": "10px"}),

        ], style={"backgroundColor": "#141414", "padding": "15px"}),

        # Store for current event index
        dcc.Store(id="current-event-idx", data=args.entry),

        # Store for cached inference results
        dcc.Store(id="cached-results", data=None),

        # 3D Plot
        html.Div([
            dcc.Graph(
                id="det3d",
                style={"height": "75vh"},
                config={"scrollZoom": True},
            )
        ]),

        # Legend for predicted class mode
        html.Div(id="class-legend", style={"backgroundColor": "#141414", "padding": "10px"}),

    ], style={"backgroundColor": "#141414", "height": "100vh"})

    @app.callback(
        Output("current-event-idx", "data"),
        [Input("prev-btn", "n_clicks"),
         Input("next-btn", "n_clicks")],
        State("current-event-idx", "data"),
    )
    def update_event_idx(prev_clicks, next_clicks, current_idx):
        ctx = dash.callback_context
        if not ctx.triggered:
            return current_idx

        button_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if button_id == "prev-btn" and current_idx > 0:
            return current_idx - 1
        elif button_id == "next-btn" and current_idx < len(dataset) - 1:
            return current_idx + 1

        return current_idx

    @app.callback(
        [Output("det3d", "figure"),
         Output("event-info", "children"),
         Output("point-count", "children"),
         Output("metrics-info", "children"),
         Output("class-legend", "children"),
         Output("cached-results", "data")],
        [Input("current-event-idx", "data"),
         Input("viz-mode", "value")],
        State("cached-results", "data"),
    )
    def update_figure(event_idx, viz_mode, cached_results):
        # Check if we need to run inference (new event)
        ctx = dash.callback_context
        need_inference = True

        if cached_results is not None:
            if cached_results.get("event_idx") == event_idx:
                need_inference = False

        if need_inference:
            # Load and process event
            print(f"Loading event {event_idx}...")
            data = dataset[event_idx]
            batch_data = collate_fn([data], in_channels=in_channels)

            # Run inference with error handling for spconv CPU issues
            try:
                seg_logits, coords, inference_stats = run_inference(model, batch_data, device)
            except Exception as e:
                error_msg = str(e)
                # Check if this is a spconv CPU error
                if "implicit_gemm" in error_msg or "spconv" in error_msg.lower() or "is_cpu" in error_msg:
                    print(f"ERROR: spconv does not support CPU inference.")
                    print(f"       Error: {error_msg}")
                    print("")
                    print("       spconv's C++ code has a hard check: '!features.is_cpu() assert failed'")
                    print("       This is a fundamental limitation - spconv is compiled as GPU-only.")
                    print("")
                    print("       Options:")
                    print("         1. Use GPU inference (remove --cpu flag)")
                    print("         2. Build spconv from source with CPU support")
                    print("         3. Use MinkowskiEngine (requires model changes)")
                    # Create an empty figure with error message
                    fig = go.Figure()
                    fig.add_annotation(
                        text=(
                            "spconv does not support CPU inference.<br><br>"
                            "The spconv library is compiled as GPU-only.<br>"
                            "Its C++ code explicitly rejects CPU tensors.<br><br>"
                            "Please use GPU inference (remove --cpu flag)."
                        ),
                        xref="paper", yref="paper",
                        x=0.5, y=0.5, showarrow=False,
                        font=dict(size=14, color="red"),
                    )
                    fig.update_layout(
                        paper_bgcolor="#141414",
                        plot_bgcolor="#141414",
                    )
                    return fig, f"Event {event_idx + 1} / {len(dataset)}", "Error", "spconv: CPU not supported", [], cached_results
                else:
                    raise  # Re-raise if it's a different error

            # Compute probabilities
            probs = F.softmax(seg_logits, dim=-1).numpy()
            predictions = seg_logits.argmax(dim=-1).numpy()

            # Get true labels if available (when using label_mode='ssnet', this is the translated SSNet label)
            if "segment" in batch_data:
                true_labels = batch_data["segment"].numpy()
            else:
                true_labels = None

            coords_np = coords.numpy()

            # Cache results
            cached_results = {
                "event_idx": event_idx,
                "coords": coords_np.tolist(),
                "predictions": predictions.tolist(),
                "probs": probs.tolist(),
                "true_labels": true_labels.tolist() if true_labels is not None else None,
                "name": data.get("name", f"event_{event_idx}"),
                "inference_stats": inference_stats,
            }
        else:
            # Use cached results
            coords_np = np.array(cached_results["coords"])
            predictions = np.array(cached_results["predictions"])
            probs = np.array(cached_results["probs"])
            true_labels = np.array(cached_results["true_labels"]) if cached_results["true_labels"] is not None else None

        n_points = len(coords_np)

        # Determine colors based on visualization mode
        if viz_mode == "predicted":
            colors = get_class_colors_for_predictions(predictions, class_names)
            hover_text = [
                f"Point #{i}<br>Predicted: {class_names[predictions[i]]}<br>"
                f"Confidence: {probs[i][predictions[i]]:.3f}<br>"
                f"Pos: ({coords_np[i, 0]:.1f}, {coords_np[i, 1]:.1f}, {coords_np[i, 2]:.1f})"
                + (f"<br>True: {class_names[true_labels[i]]}" if true_labels is not None and true_labels[i] < len(class_names) else "")
                for i in range(n_points)
            ]
            colorbar_title = None
        elif viz_mode == "true" and true_labels is not None:
            colors = get_class_colors_for_predictions(true_labels, class_names)
            hover_text = [
                f"Point #{i}<br>True: {class_names[true_labels[i]] if true_labels[i] < len(class_names) else 'unknown'}<br>"
                f"Predicted: {class_names[predictions[i]]}<br>"
                f"Pos: ({coords_np[i, 0]:.1f}, {coords_np[i, 1]:.1f}, {coords_np[i, 2]:.1f})"
                for i in range(n_points)
            ]
            colorbar_title = None
        elif viz_mode.startswith("score_"):
            class_idx = int(viz_mode.split("_")[1])
            scores = probs[:, class_idx]
            colors = score_to_colors(scores, 'viridis')
            hover_text = [
                f"Point #{i}<br>{class_names[class_idx]} Score: {scores[i]:.3f}<br>"
                f"Predicted: {class_names[predictions[i]]}<br>"
                f"Pos: ({coords_np[i, 0]:.1f}, {coords_np[i, 1]:.1f}, {coords_np[i, 2]:.1f})"
                + (f"<br>True: {class_names[true_labels[i]]}" if true_labels is not None and true_labels[i] < len(class_names) else "")
                for i in range(n_points)
            ]
            colorbar_title = f"{class_names[class_idx]} Score"
        else:
            # Fallback to predicted
            colors = get_class_colors_for_predictions(predictions, class_names)
            hover_text = [
                f"Point #{i}<br>Predicted: {class_names[predictions[i]]}"
                + (f"<br>True: {class_names[true_labels[i]]}" if true_labels is not None and true_labels[i] < len(class_names) else "")
                for i in range(n_points)
            ]
            colorbar_title = None

        # Create scatter trace
        scatter_trace = go.Scatter3d(
            x=coords_np[:, 0],
            y=coords_np[:, 1],
            z=coords_np[:, 2],
            mode="markers",
            marker=dict(
                color=colors,
                size=args.marker_size,
                opacity=args.opacity,
            ),
            hovertext=hover_text,
            hoverinfo="text",
            name="Points",
        )

        # Build traces list
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

        # Layout
        axis_template = {
            "showbackground": True,
            "backgroundcolor": "#141414",
            "gridcolor": "rgb(100, 100, 100)",
            "zerolinecolor": "rgb(150, 150, 150)",
        }

        layout = go.Layout(
            height=700,
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

        # Event info
        event_info = f"Event {event_idx + 1} / {len(dataset)}"
        if "name" in cached_results:
            event_info += f" ({cached_results['name']})"

        # Point count and inference stats
        point_count_text = f"Points: {n_points:,}"
        if "inference_stats" in cached_results:
            stats = cached_results["inference_stats"]
            point_count_text += f" | Forward: {stats['forward_time_ms']:.1f} ms ({stats['device']})"
            if stats.get("peak_memory_mb") is not None:
                point_count_text += f" | {stats['memory_type']}: {stats['peak_memory_mb']:.1f} MB"

        # Metrics
        metrics_text = ""
        if true_labels is not None:
            # Compute accuracy
            valid_mask = true_labels >= 0  # Ignore -1 labels
            if valid_mask.sum() > 0:
                correct = (predictions[valid_mask] == true_labels[valid_mask]).sum()
                accuracy = correct / valid_mask.sum() * 100
                metrics_text = f"Accuracy: {accuracy:.1f}%"

                # Per-class accuracy - show all classes, even those with 0 points
                class_accs = []
                for i, name in enumerate(class_names):
                    class_mask = (true_labels == i) & valid_mask
                    n_class = class_mask.sum()
                    #print(f"CLASS_INDEX[{i}]: nclasses={n_class}")
                    if n_class > 0:
                        class_correct = (predictions[class_mask] == i).sum()
                        class_acc = class_correct / n_class * 100
                        class_accs.append(f"{name}: {class_acc:.0f}%")
                    else:
                        class_accs.append(f"{name}: -")
                if class_accs:
                    metrics_text += " | " + ", ".join(class_accs)

        # Class legend for predicted/true modes
        legend_children = []
        if viz_mode in ["predicted", "true"]:
            legend_items = []
            for i, name in enumerate(class_names):
                color = CLASS_COLORS.get(i, "#000000")
                count = (predictions == i).sum() if viz_mode == "predicted" else (true_labels == i).sum() if true_labels is not None else 0
                legend_items.append(
                    html.Span([
                        html.Span(style={
                            "display": "inline-block",
                            "width": "15px",
                            "height": "15px",
                            "backgroundColor": color,
                            "marginRight": "5px",
                            "verticalAlign": "middle",
                        }),
                        html.Span(f"{name}: {count}", style={"marginRight": "20px"}),
                    ])
                )
            legend_children = html.Div(
                legend_items,
                style={"display": "flex", "justifyContent": "center", "flexWrap": "wrap", "color": "white"}
            )
        elif viz_mode.startswith("score_"):
            # Colorbar legend for score mode
            class_idx = int(viz_mode.split("_")[1])
            legend_children = html.Div([
                html.Span("Low", style={"color": "white", "fontSize": "12px"}),
                html.Div(style={
                    "width": "200px",
                    "height": "20px",
                    "background": "linear-gradient(to right, #440154, #414487, #2a788e, #22a884, #7ad151, #fde725)",
                    "margin": "0 10px",
                    "display": "inline-block",
                    "verticalAlign": "middle",
                }),
                html.Span("High", style={"color": "white", "fontSize": "12px"}),
                html.Span(f" ({class_names[class_idx]} Score)", style={"color": "#aaa", "marginLeft": "10px"}),
            ], style={"textAlign": "center"})

        return fig, event_info, point_count_text, metrics_text, legend_children, cached_results

    return app


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Handle --cpu flag (overrides --device)
    if args.cpu:
        args.device = "cpu"
        print("Forcing CPU-only inference (--cpu flag set)")
        print("Note: spconv layers will be switched to ConvAlgo.Native for CPU compatibility.")
        print("      CPU inference will be slower than GPU but should produce equivalent results.")

    # Load config
    print(f"Loading config: {args.config}")
    cfg = Config.fromfile(args.config)

    # Override grid size if specified
    grid_size = args.grid_size if args.grid_size is not None else cfg.get("grid_size", 0.25)

    # Build inference transform
    transform = build_inference_transform(cfg, grid_size)

    # Determine data list file
    data_list = args.data_list
    if data_list is None:
        # Try to get from config
        if hasattr(cfg, 'data') and 'val' in cfg.data:
            data_list = cfg.data.val.get('data_list_file', None)

    print(f"Building dataset with data list: {data_list}")

    # Build dataset
    from pointcept.datasets.lartpc import LArTPCDataset

    # Get dataset params from config if available
    dataset_params = {}
    if hasattr(cfg, 'data') and 'val' in cfg.data:
        val_cfg = cfg.data.val
        dataset_params = {
            'use_reco_coords': val_cfg.get('use_reco_coords', True),
            'use_edep_as_strength': val_cfg.get('use_edep_as_strength', True),
            'label_mode': val_cfg.get('label_mode', 'ssnet'),
            'coord_scale': val_cfg.get('coord_scale', 1.0),
            'wire_scale': val_cfg.get('wire_scale', 1.0/3456.0),
            'log_transform_edep': val_cfg.get('log_transform_edep', True),
            'include_ghosts': args.include_ghosts or val_cfg.get('include_ghosts', False),
            'exclude_other': val_cfg.get('exclude_other', True),
            'true_points_only': args.true_points_only or val_cfg.get('true_points_only', False),
        }
    print(dataset_params)

    base_dataset = LArTPCDataset(
        split="val",
        data_list_file=data_list,
        transform=transform,
        test_mode=False,
        loop=1,
        **dataset_params,
    )

    # Wrap dataset to also load ssnet_label
    dataset = LArTPCDatasetWithSSNet(base_dataset)

    print(f"Dataset size: {len(dataset)} events")

    # Load model (this may update cfg.model.num_classes based on checkpoint)
    model, actual_num_classes = load_model(cfg, args.checkpoint, args.device)

    # Print model info
    print(f"Model type: {cfg.model.type}")
    print(f"Number of classes: {actual_num_classes}")
    print(f"Input channels: {cfg.model.backbone.in_channels}")
    print(f"Inference device: {args.device}")

    # Check for spconv + CPU incompatibility
    uses_spconv, spconv_count = check_model_uses_spconv(model)
    if uses_spconv:
        print(f"Model uses {spconv_count} spconv layers")
        if args.device == "cpu":
            print("")
            print("=" * 70)
            print("WARNING: spconv does NOT support CPU inference!")
            print("=" * 70)
            print("")
            print("The spconv library is compiled as GPU-only. Its C++ code contains")
            print("hard assertions that reject CPU tensors:")
            print("  '!features.is_cpu() assert failed. bias and act don't support cpu.'")
            print("")
            print("Options:")
            print("  1. Use GPU inference: remove the --cpu flag")
            print("  2. Build spconv from source with CPU support (not trivial)")
            print("  3. Replace spconv with MinkowskiEngine (requires code changes)")
            print("")
            print("Continuing anyway - inference will fail when attempted...")
            print("=" * 70)
            print("")

    # Create visualization
    print(f"\nStarting Dash server on port {args.port}...")
    print(f"Open http://localhost:{args.port} in your browser")
    print("Use the dropdown to switch between visualization modes")
    print("Use Previous/Next buttons to navigate between events")

    app = create_dash_app(dataset, model, args.device, cfg, args)
    app.run(debug=True, port=args.port)


if __name__ == "__main__":
    main()
