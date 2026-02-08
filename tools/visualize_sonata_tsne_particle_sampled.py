"""
t-SNE Visualization of Sonata Encoder Features with Particle-Based Sampling

Extracts encoder features from a trained Sonata model and creates
t-SNE embeddings colored by particle class for LArTPC data.

This version implements particle-based sampling:
- Limits points collected from any individual particle (by trackid)
- Collects points until each class reaches a target count
- Splits muon category into "muon_nu" (neutrino origin) and "muon_cosmic" (cosmic origin)

Uses cuML's GPU-accelerated t-SNE implementation.

Usage:
    python tools/visualize_sonata_tsne_particle_sampled.py \
        --config configs/lartpc/pretrain-sonata-v1m1-lartpc.py \
        --checkpoint exp/lartpc/sonata-v1m1/model_best.pth \
        --data-list /path/to/val_split.txt \
        --output tsne_features.png \
        --max-points-per-particle 100 \
        --target-points-per-class 5000

Author: Generated for LArTPC analysis
"""

import os
import sys
import argparse
import numpy as np
import h5py
import torch
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pointcept.utils.config import Config
from pointcept.models.builder import build_model
from pointcept.datasets.transform import Compose, TRANSFORMS


# Extended class definitions with muon split by origin
# Note: We use a new indexing scheme for this visualization
EXTENDED_CLASS_NAMES = [
    "electron",     # 0: e+/e-
    "muon_nu",      # 1: muon from neutrino interaction
    "muon_cosmic",  # 2: muon from cosmic ray
    "pion",         # 3: pi+/pi-
    "proton",       # 4: proton
    "gamma",        # 5: photon
    "michel",       # 6: michel
    "delta",        # 7: delta
    "led",          # 8: low energy deposit
    "ghost",        # 9: ghost
    "other",        # 10: everything else
]

EXTENDED_NUM_CLASSES = len(EXTENDED_CLASS_NAMES)
EXTENDED_GHOST_LABEL = 9

# Color palette for extended classes
EXTENDED_CLASS_COLORS = {
    0: "#E41A1C",   # electron - red
    1: "#377EB8",   # muon_nu - blue
    2: "#17BECF",   # muon_cosmic - cyan
    3: "#4DAF4A",   # pion - green
    4: "#FF7F00",   # proton - orange
    5: "#984EA3",   # gamma - purple
    6: "#F682E9",   # michel - pink
    7: "#673A91",   # delta - maroon
    8: "#178787",   # led - teal
    9: "#999999",   # ghost - gray
    10: "#666666",  # other - dark gray
}

# SSNET label to extended class mapping
# SSNET labels: 0=bg, 1=electron, 2=photon, 3=muon, 4=proton, 5=pion, 6=michel, 7=delta, 8=led, 9=other
# For muon (SSNET=3), we need origin to determine if it's muon_nu or muon_cosmic
SSNET_TO_EXTENDED_CLASS = {
    0: 9,   # bg -> ghost
    1: 0,   # electron
    2: 5,   # photon -> gamma
    3: -1,  # muon - special handling needed based on origin
    4: 4,   # proton
    5: 3,   # pion
    6: 6,   # michel
    7: 7,   # delta
    8: 8,   # led
    9: 10,  # other
}

# Origin values: 0=unknown, 1=neutrino, 2=cosmic


def parse_args():
    parser = argparse.ArgumentParser(
        description="t-SNE visualization with particle-based sampling"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (.pth file)",
    )
    parser.add_argument(
        "--random-init",
        action="store_true",
        default=False,
        help="Use randomly initialized model weights",
    )
    parser.add_argument(
        "--data-list",
        type=str,
        required=True,
        help="Path to file list with HDF5 file paths",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tsne_particle_sampled.png",
        help="Output image path",
    )
    parser.add_argument(
        "--max-points-per-particle",
        type=int,
        default=100,
        help="Maximum points to sample from any individual particle (N)",
    )
    parser.add_argument(
        "--target-points-per-class",
        type=int,
        default=5000,
        help="Target total points per particle class (M)",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity parameter",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=200.0,
        help="t-SNE learning rate",
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
        "--metric",
        type=str,
        default="euclidean",
        help="t-SNE distance metric (default: euclidean)",
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
        help="Save extracted features to .npz file",
    )
    parser.add_argument(
        "--load-features",
        type=str,
        default=None,
        help="Load pre-extracted features from .npz file",
    )
    parser.add_argument(
        "--true-points-only",
        action='store_true',
        default=False,
        help="Only use true (non-ghost) points",
    )
    parser.add_argument(
        "--label-threshold-factor",
        type=float,
        default=4.0,
        help="Factor multiplied by grid_size for label assignment threshold",
    )
    parser.add_argument(
        "--include-ghost",
        action='store_true',
        default=False,
        help="Include ghost points in the visualization",
    )
    parser.add_argument(
        "--include-other",
        action='store_true',
        default=False,
        help="Include 'other' class points in the visualization",
    )
    return parser.parse_args()


def load_file_list(file_path):
    """Load list of HDF5 file paths from a text file."""
    data_list = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            data_list.append(line)
    return data_list


def get_extended_class_label(ssnet_label, origin):
    """
    Map SSNET label and origin to extended class label.

    Args:
        ssnet_label: SSNET semantic label (0-9)
        origin: Origin type (0=unknown, 1=neutrino, 2=cosmic)

    Returns:
        Extended class label
    """
    if ssnet_label == 3:  # muon
        if origin == 1:
            return 1  # muon_nu
        else:
            return 2  # muon_cosmic (includes unknown origin=0)
    return SSNET_TO_EXTENDED_CLASS.get(ssnet_label, 10)


def load_raw_event_data(h5_path):
    """
    Load raw data from HDF5 file including trackid and origin.

    Returns:
        dict with pos, ssnet_label, origin, trackid, pixval, wire coordinates
    """
    with h5py.File(h5_path, 'r') as f:
        data = {
            'pos': np.array(f['/entry_0/triplet_data/pos'], dtype=np.float32),
            'ssnet_label': np.array(f['/entry_0/triplet_data/ssnet_label'], dtype=np.int32),
            'origin': np.array(f['/entry_0/triplet_data/origin'], dtype=np.int32),
            'trackid': np.array(f['/entry_0/triplet_data/trackid'], dtype=np.int32),
            'hasmatch': np.array(f['/entry_0/triplet_data/hasmatch'], dtype=np.int32),
            'pixval': np.array(f['/entry_0/triplet_data/pixval'], dtype=np.float32),
            'uwire': np.array(f['/entry_0/triplet_data/uwire'], dtype=np.int32),
            'vwire': np.array(f['/entry_0/triplet_data/vwire'], dtype=np.int32),
            'ywire': np.array(f['/entry_0/triplet_data/ywire'], dtype=np.int32),
        }

        # Load neutrino vertices for biased sampling
        keypoint_pos = np.array(f['/entry_0/mckeypoints/pos'], dtype=np.float32)
        keypoint_type = np.array(f['/entry_0/mckeypoints/kptype'], dtype=np.int64)
        nu_mask = keypoint_type == 0
        data['nu_vertices'] = keypoint_pos[nu_mask]

    return data


def build_inference_transform(cfg, grid_size=None):
    """Build a simplified transform pipeline for inference."""
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
        dict(type="BiasedSphereCrop",
            anchor_points_key="nu_vertices",
            anchor_pdf_key=None,
            radius=20.0,
            point_max=20480,
            point_min=4096,
            prob_random=0.25,
            max_retries=100,
            fallback_to_random=True),
        dict(type="ToTensor"),
        dict(type="Update", keys_dict={"grid_size": grid_size}),
        dict(
            type="Collect",
            keys=("coord", "grid_coord", "segment", "name", "grid_size"),
            offset_keys_dict=dict(offset="coord"),
            feat_keys=("strength", "color"),
        ),
    ]

    # Pass raw dicts to Compose - it will build them internally
    return Compose(transform_list)


def prepare_data_dict(raw_data, coord_scale=1.0, wire_scale=1.0/3456.0):
    """
    Prepare data dict from raw HDF5 data for transform pipeline.
    """
    n_points = raw_data['pos'].shape[0]

    # Apply coordinate scaling
    coord = raw_data['pos'] * coord_scale

    # Prepare strength (normalized pixval)
    strength = (raw_data['pixval'] / 500.0).astype(np.float32)

    # Prepare wire features as color
    wire_feat = np.stack([
        raw_data['uwire'],
        raw_data['vwire'],
        raw_data['ywire'],
    ], axis=1).astype(np.float32) * wire_scale

    # Compute extended class labels
    extended_labels = np.array([
        get_extended_class_label(ssnet_label, origin)
        for ssnet_label, origin in zip(raw_data['ssnet_label'], raw_data['origin'])
    ], dtype=np.int32)

    data_dict = {
        'coord': coord,
        'strength': strength,
        'color': wire_feat,
        'segment': extended_labels,
        'instance': raw_data['trackid'].astype(np.int32),
        'origin': raw_data['origin'],
        'trackid': raw_data['trackid'],
        'hasmatch': raw_data['hasmatch'],
        'nu_vertices': raw_data['nu_vertices'],
        'name': 'event',
    }

    return data_dict


def load_model(cfg, checkpoint_path, device, random_init=False):
    """Load Sonata model from checkpoint or with random initialization."""
    print(f"Building model: {cfg.model.type}")

    cfg.model.up_cast_level = 4
    model = build_model(cfg.model)

    if random_init:
        print("Using RANDOMLY INITIALIZED model weights")
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
    """Extract encoder features from Sonata model."""
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


def collate_fn(batch, in_channels=None):
    """Custom collate function for inference."""
    coords = []
    feats = []
    segments = []
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
        offset += n
        offsets.append(offset)

    result = {
        "coord": torch.cat(coords, dim=0),
        "feat": torch.cat(feats, dim=0),
        "offset": torch.tensor(offsets, dtype=torch.long),
    }

    if segments:
        result["segment"] = torch.cat(segments, dim=0)

    if "grid_coord" in batch[0]:
        grid_coords = [data["grid_coord"] for data in batch]
        result["grid_coord"] = torch.cat(grid_coords, dim=0)

    if "grid_size" in batch[0]:
        result["grid_size"] = batch[0]["grid_size"]

    return result


def assign_metadata_to_output_points(
    output_coords,
    input_coords,
    input_labels,
    input_trackids,
    input_origins,
    grid_size=0.25,
    threshold_factor=4.0,
    ghost_label=EXTENDED_GHOST_LABEL,
    use_gpu=True,
):
    """
    Assign labels, trackids, and origins to output points using nearest neighbor.

    Returns:
        labels, trackids, origins arrays for output points
    """
    n_output = len(output_coords)
    label_threshold = grid_size * threshold_factor

    # Split input points into ghost and non-ghost
    non_ghost_mask = input_labels != ghost_label
    non_ghost_coords = input_coords[non_ghost_mask]
    non_ghost_labels = input_labels[non_ghost_mask]
    non_ghost_trackids = input_trackids[non_ghost_mask]
    non_ghost_origins = input_origins[non_ghost_mask]

    # Initialize defaults
    labels = np.full(n_output, ghost_label, dtype=np.int64)
    trackids = np.full(n_output, -1, dtype=np.int32)
    origins = np.zeros(n_output, dtype=np.int32)

    if len(non_ghost_coords) == 0:
        return labels, trackids, origins

    # Find nearest non-ghost point for each output point
    if use_gpu:
        try:
            from cuml.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
            nn.fit(non_ghost_coords.astype(np.float32))
            distances, indices = nn.kneighbors(output_coords.astype(np.float32))
            distances = distances.flatten()
            indices = indices.flatten()
        except ImportError:
            use_gpu = False

    if not use_gpu:
        from scipy.spatial import cKDTree
        tree = cKDTree(non_ghost_coords)
        distances, indices = tree.query(output_coords, k=1)

    # Assign metadata for points within threshold
    within_threshold = distances <= label_threshold
    labels[within_threshold] = non_ghost_labels[indices[within_threshold]]
    trackids[within_threshold] = non_ghost_trackids[indices[within_threshold]]
    origins[within_threshold] = non_ghost_origins[indices[within_threshold]]

    return labels, trackids, origins


def get_tsne_reducer(perplexity=30.0, learning_rate=200.0, n_iter=1000,
                     early_exaggeration=12.0, metric="euclidean",
                     random_state=42):
    """Create cuML t-SNE reducer."""
    from cuml.manifold import TSNE as cumlTSNE

    print("Using cuML (GPU) t-SNE")
    reducer = cumlTSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        n_iter=n_iter,
        early_exaggeration=early_exaggeration,
        metric=metric,
        random_state=random_state,
        verbose=1,
    )
    return reducer


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Determine which classes to include
    classes_to_collect = list(range(EXTENDED_NUM_CLASSES))
    if not args.include_ghost:
        classes_to_collect.remove(EXTENDED_GHOST_LABEL)
    if not args.include_other:
        classes_to_collect.remove(10)  # 'other' class

    print(f"Classes to collect: {[EXTENDED_CLASS_NAMES[i] for i in classes_to_collect]}")
    print(f"Max points per particle: {args.max_points_per_particle}")
    print(f"Target points per class: {args.target_points_per_class}")

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

        grid_size = args.grid_size if args.grid_size is not None else cfg.get("grid_size", 0.25)

        # Build transform
        transform = build_inference_transform(cfg, grid_size)

        # Load file list
        file_list = load_file_list(args.data_list)
        print(f"Found {len(file_list)} files in data list")

        # Load model
        model = load_model(cfg, args.checkpoint, args.device, random_init=args.random_init)
        in_channels = cfg.model.backbone.in_channels

        # Initialize collection containers
        # class_points[class_idx] = list of (features, coord, trackid) tuples
        class_points = {cls: [] for cls in classes_to_collect}
        class_counts = {cls: 0 for cls in classes_to_collect}

        # Track particles we've seen to avoid resampling
        # seen_particles[(file_idx, trackid)] = set of point indices already used
        seen_particles = defaultdict(set)

        # Check if all classes have reached target
        def all_classes_satisfied():
            return all(class_counts[cls] >= args.target_points_per_class
                      for cls in classes_to_collect)

        file_idx = 0
        events_processed = 0

        print("\nCollecting features by particle...")
        while not all_classes_satisfied() and file_idx < len(file_list):
            h5_path = file_list[file_idx]
            file_idx += 1
            events_processed += 1

            if not os.path.exists(h5_path):
                print(f"  Warning: File not found: {h5_path}")
                continue

            print(f"\n  Event {events_processed}: {os.path.basename(h5_path)}")

            # Load raw data
            try:
                raw_data = load_raw_event_data(h5_path)
            except Exception as e:
                print(f"    Error loading file: {e}")
                continue

            # Prepare data dict for transforms
            data_dict = prepare_data_dict(raw_data)

            # Filter to true points only if requested
            if args.true_points_only:
                true_mask = raw_data['hasmatch'] == 1
                for k in ['coord', 'strength', 'color', 'segment', 'instance',
                         'origin', 'trackid', 'hasmatch']:
                    if k in data_dict:
                        data_dict[k] = data_dict[k][true_mask]

            if data_dict['coord'].shape[0] < 100:
                print(f"    Skipping: too few points ({data_dict['coord'].shape[0]})")
                continue

            # Save raw data before transforms (transforms may modify in place)
            raw_coords = data_dict['coord'].copy()
            raw_labels = data_dict['segment'].copy()
            raw_trackids = data_dict['trackid'].copy()
            raw_origins = data_dict['origin'].copy()

            # Apply transforms
            try:
                transformed_data = transform(data_dict)
            except Exception as e:
                print(f"    Transform error: {e}")
                continue

            # Collate for model
            batch_data = collate_fn([transformed_data], in_channels=in_channels)
            n_input = batch_data["coord"].shape[0]
            print(f"    {n_input} input points after transform")

            # Extract features
            try:
                point = extract_features(model, batch_data, args.device)
            except Exception as e:
                print(f"    Feature extraction error: {e}")
                continue

            features = point.feat.cpu().numpy()
            output_coords = point.coord.cpu().numpy()
            n_output = features.shape[0]
            print(f"    {n_output} output points, {features.shape[1]}D features")

            # Get input data after transforms
            input_coords = batch_data["coord"].cpu().numpy()
            input_labels = batch_data["segment"].cpu().numpy()

            # We need to get trackid and origin for the input points
            # These need to be propagated through the transform...
            # Since transforms may change point count, we use the raw data
            # (saved before transforms) and assign via nearest neighbor to input_coords

            # Assign trackids and origins to input points (post-transform)
            _, input_trackids, input_origins = assign_metadata_to_output_points(
                input_coords, raw_coords, raw_labels, raw_trackids, raw_origins,
                grid_size=grid_size, threshold_factor=args.label_threshold_factor,
                ghost_label=EXTENDED_GHOST_LABEL, use_gpu=True,
            )

            # Assign metadata to output points
            output_labels, output_trackids, output_origins = assign_metadata_to_output_points(
                output_coords, input_coords, input_labels, input_trackids, input_origins,
                grid_size=grid_size, threshold_factor=args.label_threshold_factor,
                ghost_label=EXTENDED_GHOST_LABEL, use_gpu=True,
            )

            # Group output points by particle (class, trackid)
            # particle_points[(class, trackid)] = list of indices
            particle_points = defaultdict(list)
            for pt_idx in range(n_output):
                cls = output_labels[pt_idx]
                tid = output_trackids[pt_idx]
                if cls in classes_to_collect:
                    particle_points[(cls, tid)].append(pt_idx)

            # Sample from each particle
            for (cls, tid), pt_indices in particle_points.items():
                if class_counts[cls] >= args.target_points_per_class:
                    continue

                # How many points can we take from this particle?
                particle_key = (file_idx - 1, tid)
                already_sampled = len(seen_particles[particle_key])
                remaining_budget = args.max_points_per_particle - already_sampled

                if remaining_budget <= 0:
                    continue

                # How many points do we still need for this class?
                class_remaining = args.target_points_per_class - class_counts[cls]

                # Take the minimum
                n_to_sample = min(remaining_budget, class_remaining, len(pt_indices))

                if n_to_sample > 0:
                    # Random sample
                    sampled_indices = np.random.choice(
                        pt_indices, n_to_sample, replace=False
                    )

                    for idx in sampled_indices:
                        class_points[cls].append((
                            features[idx],
                            output_coords[idx],
                            tid,
                        ))
                        seen_particles[particle_key].add(idx)

                    class_counts[cls] += n_to_sample

            # Print progress
            status = ", ".join([
                f"{EXTENDED_CLASS_NAMES[cls]}: {class_counts[cls]}"
                for cls in classes_to_collect
            ])
            print(f"    Current counts: {status}")

        print(f"\n\nCollection complete after {events_processed} events")
        print("Final class counts:")
        for cls in classes_to_collect:
            print(f"  {EXTENDED_CLASS_NAMES[cls]}: {class_counts[cls]}")

        # Assemble final arrays
        all_features = []
        all_labels = []
        all_coords = []

        for cls in classes_to_collect:
            for feat, coord, tid in class_points[cls]:
                all_features.append(feat)
                all_labels.append(cls)
                all_coords.append(coord)

        all_features = np.array(all_features, dtype=np.float32)
        all_labels = np.array(all_labels, dtype=np.int64)
        all_coords = np.array(all_coords, dtype=np.float32)

        print(f"\nTotal: {len(all_features)} points, {all_features.shape[1]}D features")

        # Save features if requested
        if args.save_features is not None:
            print(f"Saving features to {args.save_features}")
            np.savez(
                args.save_features,
                features=all_features,
                labels=all_labels,
                coords=all_coords,
                class_names=EXTENDED_CLASS_NAMES,
            )

    # Print class distribution
    print("\nClass distribution:")
    unique, counts = np.unique(all_labels, return_counts=True)
    for cls_idx, count in zip(unique, counts):
        if cls_idx >= 0 and cls_idx < len(EXTENDED_CLASS_NAMES):
            print(f"  {EXTENDED_CLASS_NAMES[cls_idx]}: {count} ({100*count/len(all_labels):.1f}%)")

    # Run t-SNE
    print(f"\nRunning t-SNE with perplexity={args.perplexity}, lr={args.learning_rate}, n_iter={args.n_iter}...")
    reducer = get_tsne_reducer(
        perplexity=args.perplexity,
        learning_rate=args.learning_rate,
        n_iter=args.n_iter,
        early_exaggeration=args.early_exaggeration,
        metric=args.metric,
        random_state=args.seed,
    )

    features_f32 = all_features.astype(np.float32)
    embedding = reducer.fit_transform(features_f32)
    print(f"t-SNE complete: {embedding.shape}")

    # Create visualization
    print(f"Creating visualization...")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Plot 1: All points colored by class
    ax1 = axes[0]
    for cls_idx in classes_to_collect:
        mask = all_labels == cls_idx
        if mask.sum() > 0:
            ax1.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                c=EXTENDED_CLASS_COLORS.get(cls_idx, "#000000"),
                label=f"{EXTENDED_CLASS_NAMES[cls_idx]} ({mask.sum()})",
                s=2,
                alpha=0.6,
            )

    ax1.set_xlabel("t-SNE 1")
    ax1.set_ylabel("t-SNE 2")
    init_label = "RANDOM INIT" if args.random_init else "trained"
    ax1.set_title(f"All Points [{init_label}]\nperplexity={args.perplexity}")
    ax1.legend(markerscale=5, loc="best", fontsize=8)

    # Plot 2: Muon comparison (nu vs cosmic)
    ax2 = axes[1]
    muon_nu_mask = all_labels == 1
    muon_cosmic_mask = all_labels == 2

    if muon_nu_mask.sum() > 0:
        ax2.scatter(
            embedding[muon_nu_mask, 0],
            embedding[muon_nu_mask, 1],
            c=EXTENDED_CLASS_COLORS[1],
            label=f"muon_nu ({muon_nu_mask.sum()})",
            s=3,
            alpha=0.7,
        )
    if muon_cosmic_mask.sum() > 0:
        ax2.scatter(
            embedding[muon_cosmic_mask, 0],
            embedding[muon_cosmic_mask, 1],
            c=EXTENDED_CLASS_COLORS[2],
            label=f"muon_cosmic ({muon_cosmic_mask.sum()})",
            s=3,
            alpha=0.7,
        )

    ax2.set_xlabel("t-SNE 1")
    ax2.set_ylabel("t-SNE 2")
    ax2.set_title("Muon Comparison: Neutrino vs Cosmic Origin")
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
    ax3.set_title("Point Density")
    plt.colorbar(ax3.collections[0], ax=ax3, label="Count")

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved visualization to {args.output}")

    # Save embedding data
    embedding_path = args.output.rsplit(".", 1)[0] + "_embedding.npz"
    np.savez(
        embedding_path,
        embedding=embedding,
        labels=all_labels,
        class_names=EXTENDED_CLASS_NAMES,
        class_colors=[EXTENDED_CLASS_COLORS[i] for i in range(EXTENDED_NUM_CLASSES)],
        perplexity=args.perplexity,
        learning_rate=args.learning_rate,
        n_iter=args.n_iter,
    )
    print(f"Saved embedding data to {embedding_path}")

    plt.show()


if __name__ == "__main__":
    main()
