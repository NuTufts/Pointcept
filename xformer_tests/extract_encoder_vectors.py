"""
Extract encoder vectors from PointTransformerV3 backbone.

This script loads a pretrained model and extracts encoder features for a single
data file. It can be run with either the flash_attn backend (A100) or the
xformers backend (P100) to compare outputs.

Usage:
    python extract_encoder_vectors.py \
        --config configs/lartpc/pretrain-sonata-v1m1-lartpc.py \
        --checkpoint /path/to/checkpoint.pth \
        --data-file /path/to/data.h5 \
        --output encoder_vectors.pt \
        --flash-backend xformers  # or flash_attn

Author: Generated for xformers backend testing
"""

import argparse
import os
import sys
import torch
import numpy as np

# Add pointcept to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pointcept.utils.config import Config
from pointcept.models.builder import build_model
from pointcept.datasets.builder import build_dataset
from pointcept.datasets.transform import Compose, TRANSFORMS
from pointcept.models.utils.structure import Point


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract encoder vectors from PointTransformerV3"
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
        required=True,
        help="Path to model checkpoint (.pth file)",
    )
    parser.add_argument(
        "--data-file",
        type=str,
        required=True,
        help="Path to single HDF5 data file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file path for encoder vectors (.pt file)",
    )
    parser.add_argument(
        "--flash-backend",
        type=str,
        default="flash_attn",
        choices=["flash_attn", "xformers", "none"],
        help="Flash attention backend: flash_attn (default), xformers, or none",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (cuda or cpu)",
    )
    return parser.parse_args()


def load_single_file_data(data_file, cfg):
    """Load a single HDF5 file and apply transforms."""
    import h5py

    # Get dataset config
    data_cfg = cfg.data.train.copy()

    # Load raw data from HDF5
    with h5py.File(data_file, "r") as f:
        # Get coordinates
        coord = np.array( f['/entry_0/triplet_data/pos'], dtype=np.float32 )        
#         if data_cfg.get("use_reco_coords", True):
#             #coord = np.stack([
#             #    f["pos_x_reco"][:],
#             #    f["pos_y_reco"][:],
#             #    f["pos_z_reco"][:]
#             #], axis=1).astype(np.float32)
#         else:
#             coord = np.stack([
#                 f["pos_x"][:],
#                 f["pos_y"][:],
#                 f["pos_z"][:]
#             ], axis=1).astype(np.float32)

        # Get strength (energy deposition)
        if data_cfg.get("use_edep_as_strength", True):
            strength = np.array( f["/entry_0/triplet_data/pixval"], dtype=np.float32 )/500.0
#             edep_u = f["edep_u"][:].astype(np.float32)
#             edep_v = f["edep_v"][:].astype(np.float32)
#             edep_y = f["edep_y"][:].astype(np.float32)
#             if data_cfg.get("log_transform_edep", True):
#                 edep_u = np.log1p(edep_u)
#                 edep_v = np.log1p(edep_v)
#                 edep_y = np.log1p(edep_y)
#             strength = np.stack([edep_u, edep_v, edep_y], axis=1)
        else:
            strength = np.ones((coord.shape[0], 3), dtype=np.float32)

        # Get wire coordinates for color
        wire_scale = data_cfg.get("wire_scale", 1.0/3456.0)
        uwire = f["/entry_0/triplet_data/uwire"][:].astype(np.float32) * wire_scale
        vwire = f["/entry_0/triplet_data/vwire"][:].astype(np.float32) * wire_scale
        ywire = f["/entry_0/triplet_data/ywire"][:].astype(np.float32) * wire_scale
        color = np.stack([uwire, vwire, ywire], axis=1)

        # Get labels
        pid = f["/entry_0/triplet_data/pid"][:].astype(np.int64)

        # Apply coordinate scale
        coord_scale = data_cfg.get("coord_scale", 1.0)
        coord = coord * coord_scale

    # Create data dict
    data_dict = {
        "coord": coord,
        "strength": strength,
        "color": color,
        "segment": pid,
        "name": os.path.basename(data_file),
    }

    return data_dict


def create_simple_transform(cfg):
    """Create a simplified transform for single-file inference."""
    grid_size = cfg.get("grid_size", 0.25)

    # Simple transform: GridSample -> ToTensor -> Collect
    # Compose expects a list of config dicts, not built transforms
    # Note: 'offset' is created by Collect via offset_keys_dict, not from input
    transform_list = [
        dict(
            type="GridSample",
            grid_size=grid_size,
            hash_type="fnv",
            mode="train",
            return_grid_coord=True,
        ),
        dict(type="ToTensor"),
        dict(
            type="Collect",
            keys=("coord", "grid_coord"),  # offset is auto-created from coord
            offset_keys_dict=dict(offset="coord"),  # creates offset from coord.shape[0]
            feat_keys=("strength", "color"),
        ),
    ]

    # Pass the list of dicts directly to Compose
    return Compose(transform_list)


def extract_encoder_features(model, data_dict, device):
    """Extract encoder features from the backbone."""
    # Move data to device
    for key in data_dict:
        if isinstance(data_dict[key], torch.Tensor):
            data_dict[key] = data_dict[key].to(device)

    # Create Point object and run through backbone
    with torch.no_grad():
        # For Sonata model, we need to access the student backbone
        if hasattr(model, 'student'):
            backbone = model.student['backbone']
        else:
            backbone = model.backbone if hasattr(model, 'backbone') else model

        # Create Point and run serialization
        point = Point(data_dict)

        # Get order from backbone
        order = backbone.order if hasattr(backbone, 'order') else ("z", "z-trans")
        shuffle_orders = backbone.shuffle_orders if hasattr(backbone, 'shuffle_orders') else True

        point.serialization(order=order, shuffle_orders=shuffle_orders)
        point.sparsify()

        # Run through embedding
        if hasattr(backbone, 'embedding'):
            point = backbone.embedding(point)

        # Run through encoder
        if hasattr(backbone, 'enc'):
            point = backbone.enc(point)

        # Extract features
        encoder_features = point.feat.cpu()
        encoder_coords = point.coord.cpu()

        # Also save grid coordinates if available
        grid_coord = point.grid_coord.cpu() if hasattr(point, 'grid_coord') else None

    return {
        'features': encoder_features,
        'coords': encoder_coords,
        'grid_coord': grid_coord,
        'offset': data_dict.get('offset', torch.tensor([encoder_features.shape[0]])).cpu(),
    }


def main():
    args = parse_args()

    # Check files exist
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not os.path.exists(args.data_file):
        raise FileNotFoundError(f"Data file not found: {args.data_file}")

    # Load config
    print(f"Loading config from: {args.config}")
    cfg = Config.fromfile(args.config)

    # Modify backbone config to use specified flash backend
    flash_backend = None if args.flash_backend == "none" else args.flash_backend

    if "backbone" in cfg.model:
        cfg.model.backbone.flash_backend = flash_backend
        print(f"Set backbone flash_backend to: {flash_backend}")

    # Build model
    print("Building model...")
    model = build_model(cfg.model)

    # Load checkpoint
    print(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")

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

    # Load weights (allow missing keys for flexibility)
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        print(f"Missing keys: {len(missing)}")
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")

    # Move to device and eval mode
    device = torch.device(args.device)
    model = model.to(device)
    model.eval()
    print(f"Model loaded on {device}")

    # Load data
    print(f"Loading data from: {args.data_file}")
    data_dict = load_single_file_data(args.data_file, cfg)
    print(f"Loaded {data_dict['coord'].shape[0]} points")

    # Apply transforms
    print("Applying transforms...")
    transform = create_simple_transform(cfg)
    data_dict = transform(data_dict)

    # Add batch dimension info
    n_points = data_dict['coord'].shape[0]
    data_dict['offset'] = torch.tensor([n_points], dtype=torch.long)
    data_dict['batch'] = torch.zeros(n_points, dtype=torch.long)

    print(f"After transforms: {n_points} points")

    # Extract features
    print("Extracting encoder features...")
    result = extract_encoder_features(model, data_dict, device)

    print(f"Encoder output shape: {result['features'].shape}")

    # Save results
    print(f"Saving to: {args.output}")
    save_dict = {
        'features': result['features'],
        'coords': result['coords'],
        'grid_coord': result['grid_coord'],
        'offset': result['offset'],
        'data_file': args.data_file,
        'flash_backend': args.flash_backend,
        'config': args.config,
        'checkpoint': args.checkpoint,
    }
    torch.save(save_dict, args.output)

    print("Done!")

    # Print some statistics
    features = result['features']
    print(f"\nFeature statistics:")
    print(f"  Shape: {features.shape}")
    print(f"  Mean: {features.mean().item():.6f}")
    print(f"  Std: {features.std().item():.6f}")
    print(f"  Min: {features.min().item():.6f}")
    print(f"  Max: {features.max().item():.6f}")


if __name__ == "__main__":
    main()
