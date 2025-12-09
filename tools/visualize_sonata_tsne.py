"""
t-SNE Visualization of Sonata Encoder Features

Extracts encoder features from a trained Sonata model and creates
t-SNE embeddings colored by particle class for LArTPC data.

Uses cuML's GPU-accelerated t-SNE implementation.

Usage:
    python tools/visualize_sonata_tsne.py \
        --config configs/lartpc/pretrain-sonata-v1m1-lartpc.py \
        --checkpoint exp/lartpc/sonata-v1m1/model_best.pth \
        --data-list /path/to/val_split.txt \
        --output tsne_features.png \
        --max-points 50000 \
        --num-events 5

Author: Generated for LArTPC analysis
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pointcept.utils.config import Config
from pointcept.models.builder import build_model
from pointcept.datasets.builder import build_dataset
from pointcept.datasets.transform import Compose, TRANSFORMS


def parse_args():
    parser = argparse.ArgumentParser(
        description="t-SNE visualization of Sonata encoder features (cuML GPU)"
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
        help="Path to file list (overrides config). If not provided, uses val split from config.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tsne_sonata_features.png",
        help="Output image path",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=50000,
        help="Maximum total points for t-SNE (subsampled if exceeded)",
    )
    parser.add_argument(
        "--points-per-event",
        type=int,
        default=10000,
        help="Maximum points to sample per event",
    )
    parser.add_argument(
        "--num-events",
        type=int,
        default=10,
        help="Number of events to process",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity parameter (typically 5-50, higher = more global structure)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=200.0,
        help="t-SNE learning rate (typically 10-1000)",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=1000,
        help="Number of t-SNE iterations",
    )
    parser.add_argument(
        "--early-exaggeration",
        type=float,
        default=12.0,
        help="t-SNE early exaggeration factor",
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
        "--save-features",
        type=str,
        default=None,
        help="Optional: save extracted features to .npz file",
    )
    parser.add_argument(
        "--load-features",
        type=str,
        default=None,
        help="Optional: load pre-extracted features from .npz file (skip model inference)",
    )
    parser.add_argument(
        "--true-points-only",
        action='store_true',
        default=False,
        help="Optional: If given, only use true (non-ghost) points",
    )
    parser.add_argument(
        "--label-threshold-factor",
        type=float,
        default=4.0,
        help="Factor multiplied by grid_size to get label assignment threshold (default: 4.0)",
    )
    return parser.parse_args()


def get_tsne_reducer(perplexity=30.0, learning_rate=200.0, n_iter=1000,
                     early_exaggeration=12.0, random_state=42):
    """
    Create cuML t-SNE reducer.

    Args:
        perplexity: Perplexity parameter (balance between local and global structure)
        learning_rate: Learning rate for optimization
        n_iter: Number of iterations
        early_exaggeration: Early exaggeration factor
        random_state: Random seed

    Returns:
        t-SNE reducer object
    """
    from cuml.manifold import TSNE as cumlTSNE

    print("Using cuML (GPU) t-SNE")
    reducer = cumlTSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        n_iter=n_iter,
        early_exaggeration=early_exaggeration,
        random_state=random_state,
        verbose=1,
    )
    return reducer


def build_inference_transform(cfg, grid_size=None):
    """
    Build a simplified transform pipeline for inference (no multi-view).

    Returns a list of transform config dicts (not a Compose object),
    since the dataset wraps them in Compose internally.
    """
    if grid_size is None:
        grid_size = cfg.get("grid_size", 0.25)

    transform_list = [
        dict(
            type="GridSample",
            grid_size=grid_size,
            hash_type="fnv",
            mode="train",  # keeps all points
            return_grid_coord=True,
        ),
        dict(type="ToTensor"),
        # Add grid_size to the data dict for the model
        dict(type="Update", keys_dict={"grid_size": grid_size}),
        dict(
            type="Collect",
            keys=("coord", "grid_coord", "segment", "name", "grid_size"),
            offset_keys_dict=dict(offset="coord"),
            feat_keys=("strength", "color"),
        ),
    ]

    # Return list of dicts - dataset wraps in Compose internally
    return transform_list


def load_model(cfg, checkpoint_path, device, random_init=False):
    """
    Load Sonata model from checkpoint or with random initialization.

    Args:
        cfg: Model configuration
        checkpoint_path: Path to checkpoint file (ignored if random_init=True)
        device: Device to load model on
        random_init: If True, use random weights instead of loading checkpoint
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

        # Handle different checkpoint formats
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        # Remove 'module.' prefix if present (from DDP)
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

    The Sonata model's forward() with return_point=True uses the teacher
    backbone and returns upcast features.
    """
    # Prepare input dict for the backbone
    # The backbone wraps this in a Point object internally
    input_dict = {
        "coord": data_dict["coord"].to(device),
        "feat": data_dict["feat"].to(device),
        "offset": data_dict["offset"].to(device),
    }

    # Add grid_size - required for serialization
    if "grid_size" in data_dict:
        grid_size = data_dict["grid_size"]
        if isinstance(grid_size, torch.Tensor):
            input_dict["grid_size"] = grid_size[0].item() if grid_size.numel() > 1 else grid_size.item()
        else:
            input_dict["grid_size"] = grid_size
    else:
        input_dict["grid_size"] = 0.25  # default

    if "grid_coord" in data_dict:
        input_dict["grid_coord"] = data_dict["grid_coord"].to(device)

    # Run through teacher backbone with up_cast
    # The Sonata forward() with return_point=True handles everything
    result = model.forward(input_dict, return_point=True)

    return result["point"]


def collate_fn(batch):
    """
    Custom collate function for inference.
    """
    # Stack coordinates and features, compute offsets
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

    # Preserve grid_size from first sample
    if "grid_size" in batch[0]:
        result["grid_size"] = batch[0]["grid_size"]

    return result


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Class names for LArTPC
    class_names = ["electron", "muon", "pion", "proton", "gamma", "ghost"]
    num_classes = len(class_names)

    # Check if loading pre-extracted features
    if args.load_features is not None:
        print(f"Loading pre-extracted features from {args.load_features}")
        data = np.load(args.load_features)
        all_features = data["features"]
        all_labels = data["labels"]
        all_coords = data.get("coords", None)
        print(f"Loaded {len(all_features)} points with {all_features.shape[1]} feature dimensions")
    else:
        # Load config
        print(f"Loading config: {args.config}")
        cfg = Config.fromfile(args.config)

        # Override grid size if specified
        grid_size = args.grid_size if args.grid_size is not None else cfg.get("grid_size", 0.25)

        # Build inference transform and dataset
        transform = build_inference_transform(cfg, grid_size)

        # Determine data list file
        data_list = args.data_list
        if data_list is None:
            # Try to use val split from config
            if hasattr(cfg, "VAL_FILE_LIST"):
                data_list = cfg.VAL_FILE_LIST
            elif hasattr(cfg.data, "val") and "data_list_file" in cfg.data.val:
                data_list = cfg.data.val.data_list_file
            else:
                print("Warning: No data list specified, using train split")
                if hasattr(cfg, "TRAIN_FILE_LIST"):
                    data_list = cfg.TRAIN_FILE_LIST

        print(f"Building dataset with data list: {data_list}")

        # Build dataset directly using LArTPCDataset
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
            true_points_only=args.true_points_only,
            test_mode=False,  # Use False to avoid needing test_cfg
            loop=1,
        )

        print(f"Dataset size: {len(dataset)} events")

        # Load model
        model = load_model(cfg, args.checkpoint, args.device, random_init=args.random_init)

        # Extract features from events
        all_features = []
        all_labels = []
        all_coords = []

        num_events = min(args.num_events, len(dataset))
        print(f"\nExtracting features from {num_events} events...")

        for event_idx in range(num_events):
            print(f"  Processing event {event_idx + 1}/{num_events}...", end=" ")

            # Get single event
            data = dataset[event_idx]

            # Create batch of 1
            batch_data = collate_fn([data])

            n_points_input = batch_data["coord"].shape[0]
            print(f"{n_points_input} input points", end=" ")

            # Extract features
            point = extract_features(model, batch_data, args.device)

            features = point.feat.cpu().numpy()
            coords = point.coord.cpu().numpy()
            n_points_output = features.shape[0]

            print(f"-> {n_points_output} output points, {features.shape[1]}D features")

            # Get labels at the output resolution
            # The backbone pools points, so we need to find nearest input labels
            # for each output coordinate.
            #
            # Improved algorithm: Prioritize non-ghost labels within pooling radius.
            # Due to pooling, an output point may have both ghost and non-ghost
            # input points nearby. We prefer non-ghost labels when available.
            #
            # The pooling radius is determined by:
            #   - grid_size (voxelization): 0.25 cm
            #   - stride (2,2,2,2): 4 pooling layers, each 2x
            #   - up_cast brings features back some levels
            # A reasonable threshold is ~4x grid_size = 1.0 cm
            ghost_label = 5
            label_threshold = grid_size * args.label_threshold_factor  # Distance threshold for label assignment

            if "segment" in batch_data:
                input_coords = batch_data["coord"].cpu().numpy()
                input_labels = batch_data["segment"].cpu().numpy()

                # Split input points into ghost and non-ghost
                non_ghost_mask = input_labels != ghost_label
                non_ghost_coords = input_coords[non_ghost_mask]
                non_ghost_labels = input_labels[non_ghost_mask]

                # Initialize all labels as ghost
                labels = np.full(n_points_output, ghost_label, dtype=np.int64)

                if len(non_ghost_coords) > 0:
                    # Use cuML NearestNeighbors for GPU-accelerated KNN
                    from cuml.neighbors import NearestNeighbors

                    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
                    nn.fit(non_ghost_coords.astype(np.float32))
                    distances, indices = nn.kneighbors(coords.astype(np.float32))

                    # Flatten results (cuML returns 2D arrays even for k=1)
                    distances = distances.flatten()
                    indices = indices.flatten()

                    # Assign non-ghost label if within threshold distance
                    within_threshold = distances <= label_threshold
                    labels[within_threshold] = non_ghost_labels[indices[within_threshold]]

                    n_assigned = within_threshold.sum()
                    print(f"(labels: {n_assigned} non-ghost, {n_points_output - n_assigned} ghost)", end=" ")
            else:
                labels = np.zeros(n_points_output, dtype=np.int64)

            # Subsample if too many points
            if n_points_output > args.points_per_event:
                indices = np.random.choice(n_points_output, args.points_per_event, replace=False)
                features = features[indices]
                labels = labels[indices]
                coords = coords[indices]

            all_features.append(features)
            all_labels.append(labels)
            all_coords.append(coords)

        # Concatenate all
        all_features = np.vstack(all_features)
        all_labels = np.concatenate(all_labels)
        all_coords = np.vstack(all_coords)

        print(f"\nTotal: {len(all_features)} points, {all_features.shape[1]}D features")

        # Save features if requested
        if args.save_features is not None:
            print(f"Saving features to {args.save_features}")
            np.savez(
                args.save_features,
                features=all_features,
                labels=all_labels,
                coords=all_coords,
            )

    # Final subsampling for t-SNE
    if len(all_features) > args.max_points:
        print(f"Subsampling from {len(all_features)} to {args.max_points} points for t-SNE")
        indices = np.random.choice(len(all_features), args.max_points, replace=False)
        all_features = all_features[indices]
        all_labels = all_labels[indices]
        if all_coords is not None:
            all_coords = all_coords[indices]

    # Print class distribution
    print("\nClass distribution:")
    unique, counts = np.unique(all_labels, return_counts=True)
    for cls_idx, count in zip(unique, counts):
        if cls_idx >= 0 and cls_idx < len(class_names):
            print(f"  {class_names[cls_idx]}: {count} ({100*count/len(all_labels):.1f}%)")

    # Run t-SNE
    print(f"\nRunning t-SNE with perplexity={args.perplexity}, learning_rate={args.learning_rate}, n_iter={args.n_iter}...")
    reducer = get_tsne_reducer(
        perplexity=args.perplexity,
        learning_rate=args.learning_rate,
        n_iter=args.n_iter,
        early_exaggeration=args.early_exaggeration,
        random_state=args.seed,
    )

    # Convert to float32 for t-SNE
    features_f32 = all_features.astype(np.float32)

    # Run t-SNE
    embedding = reducer.fit_transform(features_f32)
    print(f"t-SNE complete: {embedding.shape}")

    # Create visualization
    print(f"Creating visualization...")
    import matplotlib.pyplot as plt

    # Custom high-contrast color palette for 6 particle classes
    # Designed for maximum distinguishability
    class_colors = {
        0: "#E41A1C",  # electron - red
        1: "#377EB8",  # muon - blue
        2: "#4DAF4A",  # pion - green
        3: "#FF7F00",  # proton - orange
        4: "#984EA3",  # gamma - purple
        5: "#999999",  # ghost - gray
    }

    # Ghost class index
    ghost_idx = 5

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Plot 1: All points colored by class (ghost with low opacity)
    ax1 = axes[0]
    # Plot ghost points first (background) with low opacity
    ghost_mask = all_labels == ghost_idx
    if ghost_mask.sum() > 0:
        ax1.scatter(
            embedding[ghost_mask, 0],
            embedding[ghost_mask, 1],
            c=class_colors[ghost_idx],
            label=f"{class_names[ghost_idx]} ({ghost_mask.sum()})",
            s=2,
            alpha=0.10,
        )
    # Plot non-ghost points on top with normal opacity
    for cls_idx in range(num_classes):
        if cls_idx == ghost_idx:
            continue  # Already plotted
        mask = all_labels == cls_idx
        if mask.sum() > 0:
            ax1.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                c=class_colors.get(cls_idx, "#000000"),
                label=f"{class_names[cls_idx]} ({mask.sum()})",
                s=2,
                alpha=0.6,
            )

    ax1.set_xlabel("t-SNE 1")
    ax1.set_ylabel("t-SNE 2")
    init_label = "RANDOM INIT" if args.random_init else "trained"
    ax1.set_title(f"All Points (ghost alpha=0.10) [{init_label}]\nperplexity={args.perplexity}")
    ax1.legend(markerscale=5, loc="best")

    # Plot 2: Non-ghost points only
    ax2 = axes[1]
    non_ghost_mask = all_labels != ghost_idx
    for cls_idx in range(num_classes):
        if cls_idx == ghost_idx:
            continue
        mask = all_labels == cls_idx
        if mask.sum() > 0:
            ax2.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                c=class_colors.get(cls_idx, "#000000"),
                label=f"{class_names[cls_idx]} ({mask.sum()})",
                s=3,
                alpha=0.7,
            )

    ax2.set_xlabel("t-SNE 1")
    ax2.set_ylabel("t-SNE 2")
    ax2.set_title(f"True Points Only (no ghost)\n{non_ghost_mask.sum()} points")
    ax2.legend(markerscale=5, loc="best")

    # Plot 3: Density plot
    ax3 = axes[2]
    ax3.hexbin(
        embedding[:, 0],
        embedding[:, 1],
        gridsize=50,
        cmap="viridis",
        mincnt=1,
    )
    ax3.set_xlabel("t-SNE 1")
    ax3.set_ylabel("t-SNE 2")
    ax3.set_title("Point Density (all)")
    plt.colorbar(ax3.collections[0], ax=ax3, label="Count")

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved visualization to {args.output}")

    # Also save embedding data
    embedding_path = args.output.rsplit(".", 1)[0] + "_embedding.npz"
    np.savez(
        embedding_path,
        embedding=embedding,
        labels=all_labels,
        class_names=class_names,
        class_colors=[class_colors[i] for i in range(num_classes)],
        perplexity=args.perplexity,
        learning_rate=args.learning_rate,
        n_iter=args.n_iter,
    )
    print(f"Saved embedding data to {embedding_path}")

    plt.show()


if __name__ == "__main__":
    main()
