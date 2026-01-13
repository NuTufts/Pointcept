"""
Feature Standard Deviation Statistics Calculator for Sonata Pretrained Models

This tool calculates feature standard deviation statistics to detect feature collapse
in Sonata self-supervised pretrained models. Low standard deviation values indicate
potential feature collapse where the model outputs constant/uninformative features.

Statistics calculated:
- global_std: Standard deviation across all feature elements
- global_mean: Mean of all feature elements
- batch_std: Mean of per-sample standard deviations
- channel_mean_std: Mean of per-channel standard deviations
- channel_min_std: Minimum channel standard deviation
- channel_max_std: Maximum channel standard deviation
- dead_channels: Number of channels with near-zero std (potential dead features)

Optional outputs:
- Per-channel std distribution histogram
- Channel-wise statistics CSV

Usage:
    python tools/feature_std_stats.py --config-file CONFIG --weight WEIGHT --file-list FILES.txt

Author: Generated for LArTPC analysis
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pointcept.utils.config import Config
from pointcept.models.builder import build_model
from pointcept.datasets.builder import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate feature standard deviation statistics for Sonata models"
    )
    parser.add_argument(
        "--config-file",
        required=True,
        help="Path to the Sonata pretraining config file"
    )
    parser.add_argument(
        "--weight",
        required=True,
        help="Path to the model checkpoint (.pth file)"
    )
    parser.add_argument(
        "--file-list",
        required=True,
        help="Path to text file containing list of HDF5 files to process"
    )
    parser.add_argument(
        "--output-dir",
        default="feature_std_output",
        help="Directory to save output files (default: feature_std_output)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for processing (default: 1)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (default: all)"
    )
    parser.add_argument(
        "--network",
        type=str,
        default="both",
        choices=["student", "teacher", "both"],
        help="Which network to analyze (default: both)"
    )
    parser.add_argument(
        "--histogram",
        action="store_true",
        help="Generate histogram of per-channel std"
    )
    parser.add_argument(
        "--dead-threshold",
        type=float,
        default=1e-6,
        help="Threshold below which a channel is considered 'dead' (default: 1e-6)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (default: cuda if available)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loader workers (default: 4)"
    )
    return parser.parse_args()


def load_model(config_path, weight_path, device):
    """Load Sonata model from config and checkpoint."""
    print(f"Loading config from: {config_path}")
    cfg = Config.fromfile(config_path)

    print(f"Building model: {cfg.model['type']}")
    model = build_model(cfg.model)
    model = model.to(device)
    model.eval()

    print(f"Loading weights from: {weight_path}")
    checkpoint = torch.load(weight_path, map_location=device, weights_only=False)

    # Handle different checkpoint formats
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Remove module. prefix if present (from DDP training)
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]
        new_state_dict[key] = value

    model.load_state_dict(new_state_dict, strict=False)
    print(f"Model loaded successfully (epoch: {checkpoint.get('epoch', 'unknown')})")

    return model, cfg


def build_inference_dataset(cfg, file_list_path):
    """Build dataset for inference with minimal transforms."""
    # Read file list
    with open(file_list_path, 'r') as f:
        files = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    print(f"Found {len(files)} files in {file_list_path}")

    # Get dataset config from training config
    train_cfg = cfg.data.train.copy()

    # Override with inference settings
    train_cfg['data_list_file'] = file_list_path
    train_cfg['test_mode'] = False
    train_cfg['loop'] = 1

    # Simplify transforms for inference - we need features but not multi-view
    grid_size = cfg.get('grid_size', 0.25)

    # Determine input channels from model config
    in_channels = cfg.model.get('backbone', {}).get('in_channels', 6)
    if in_channels == 3:
        feat_keys = ("strength",)
        print("Using 3 input channels (strength only)")
    else:
        feat_keys = ("strength", "color")
        print(f"Using {in_channels} input channels (strength + color)")

    # Build minimal transform pipeline for inference
    transform = [
        dict(type="SphereCrop", point_max=98304, mode="random"),
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

    train_cfg['transform'] = transform

    dataset = build_dataset(train_cfg)
    return dataset, grid_size


def collate_fn(batch):
    """Custom collate function for point cloud data."""
    result = {}
    for key in batch[0].keys():
        if key == "offset":
            offsets = [0]
            for item in batch:
                offsets.append(offsets[-1] + item["offset"][-1].item())
            result[key] = torch.tensor(offsets[1:], dtype=torch.long)
        elif key == "name":
            result[key] = [item[key] for item in batch]
        elif key == "grid_size":
            result[key] = batch[0][key]
        elif isinstance(batch[0][key], torch.Tensor):
            result[key] = torch.cat([item[key] for item in batch], dim=0)
        else:
            result[key] = [item[key] for item in batch]
    return result


def get_backbone_features(model, data_dict, use_teacher, device):
    """
    Get backbone features for a batch of data.

    Returns:
        features: (N, C) tensor of features
    """
    # Move data to device
    for key in list(data_dict.keys()):
        if isinstance(data_dict[key], torch.Tensor):
            data_dict[key] = data_dict[key].to(device)
        elif key == 'grid_size' and not isinstance(data_dict[key], torch.Tensor):
            data_dict[key] = torch.tensor(data_dict[key], device=device)

    with torch.no_grad():
        # Get the appropriate backbone
        network = model.teacher if use_teacher else model.student
        backbone = network['backbone']

        # Run backbone forward
        point = backbone(data_dict)

        # Get features (after encoder, before heads)
        features = point.feat

    return features.cpu()


class OnlineStatistics:
    """
    Welford's online algorithm for computing mean and variance in a single pass.
    This is numerically stable and memory efficient for large datasets.
    """

    def __init__(self, num_channels):
        self.num_channels = num_channels
        self.n = 0  # Total number of samples (tokens)

        # For global statistics
        self.global_mean = 0.0
        self.global_m2 = 0.0  # Sum of squared differences from mean

        # For per-channel statistics
        self.channel_mean = np.zeros(num_channels, dtype=np.float64)
        self.channel_m2 = np.zeros(num_channels, dtype=np.float64)

        # For batch-wise std accumulation
        self.batch_std_sum = 0.0
        self.batch_count = 0

    def update(self, features):
        """
        Update statistics with a new batch of features.

        Args:
            features: (N, C) numpy array or torch tensor
        """
        if isinstance(features, torch.Tensor):
            features = features.numpy()

        features = features.astype(np.float64)
        n_samples = features.shape[0]

        # Update batch-wise std
        if n_samples > 1:
            batch_std = np.std(features, axis=1).mean()
            self.batch_std_sum += batch_std
            self.batch_count += 1

        # Update global and channel statistics using Welford's algorithm
        for i in range(n_samples):
            self.n += 1
            sample = features[i]

            # Global statistics (flatten to single value per sample)
            for val in sample:
                delta = val - self.global_mean
                self.global_mean += delta / self.n
                delta2 = val - self.global_mean
                self.global_m2 += delta * delta2

        # Per-channel statistics (batch update for efficiency)
        for i in range(n_samples):
            sample = features[i]
            delta = sample - self.channel_mean
            self.channel_mean += delta / (self.n - n_samples + i + 1)
            delta2 = sample - self.channel_mean
            self.channel_m2 += delta * delta2

    def get_statistics(self, dead_threshold=1e-6):
        """
        Compute final statistics.

        Returns:
            dict with all computed statistics
        """
        if self.n < 2:
            return None

        # Global variance and std
        global_var = self.global_m2 / (self.n * self.num_channels - 1)
        global_std = np.sqrt(max(global_var, 0))

        # Per-channel variance and std
        channel_var = self.channel_m2 / (self.n - 1)
        channel_std = np.sqrt(np.maximum(channel_var, 0))

        # Batch-wise std (average)
        batch_std = self.batch_std_sum / max(self.batch_count, 1)

        # Dead channels
        dead_channels = np.sum(channel_std < dead_threshold)

        stats = {
            "total_tokens": self.n,
            "num_channels": self.num_channels,
            "global_mean": float(self.global_mean),
            "global_std": float(global_std),
            "batch_std": float(batch_std),
            "channel_mean_std": float(np.mean(channel_std)),
            "channel_median_std": float(np.median(channel_std)),
            "channel_min_std": float(np.min(channel_std)),
            "channel_max_std": float(np.max(channel_std)),
            "channel_std_of_std": float(np.std(channel_std)),
            "dead_channels": int(dead_channels),
            "dead_channel_percent": float(dead_channels / self.num_channels * 100),
        }

        return stats, channel_std


def create_histogram_plot(channel_std, output_path, network_name, dead_threshold):
    """Create and save histogram of per-channel std."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available, skipping histogram plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Histogram of channel std values
    ax1 = axes[0, 0]
    ax1.hist(channel_std, bins=50, edgecolor='black', alpha=0.7)
    ax1.axvline(x=dead_threshold, color='r', linestyle='--', label=f'Dead threshold ({dead_threshold})')
    ax1.set_xlabel('Standard Deviation')
    ax1.set_ylabel('Number of Channels')
    ax1.set_title(f'Distribution of Per-Channel Std - {network_name}')
    ax1.legend()

    # 2. Log-scale histogram
    ax2 = axes[0, 1]
    ax2.hist(channel_std, bins=50, edgecolor='black', alpha=0.7)
    ax2.axvline(x=dead_threshold, color='r', linestyle='--', label=f'Dead threshold ({dead_threshold})')
    ax2.set_xlabel('Standard Deviation')
    ax2.set_ylabel('Number of Channels (log)')
    ax2.set_yscale('log')
    ax2.set_title(f'Distribution of Per-Channel Std (log scale) - {network_name}')
    ax2.legend()

    # 3. Channel std as bar plot (sorted)
    ax3 = axes[1, 0]
    sorted_std = np.sort(channel_std)
    ax3.bar(range(len(sorted_std)), sorted_std, width=1.0, edgecolor='none')
    ax3.axhline(y=dead_threshold, color='r', linestyle='--', label=f'Dead threshold')
    ax3.set_xlabel('Channel Index (sorted by std)')
    ax3.set_ylabel('Standard Deviation')
    ax3.set_title(f'Per-Channel Std (sorted) - {network_name}')
    ax3.legend()

    # 4. Channel std by original index
    ax4 = axes[1, 1]
    ax4.bar(range(len(channel_std)), channel_std, width=1.0, edgecolor='none')
    ax4.axhline(y=dead_threshold, color='r', linestyle='--', label=f'Dead threshold')
    ax4.set_xlabel('Channel Index')
    ax4.set_ylabel('Standard Deviation')
    ax4.set_title(f'Per-Channel Std (original order) - {network_name}')
    ax4.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved histogram to: {output_path}")


def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model, cfg = load_model(args.config_file, args.weight, args.device)

    # Determine which networks to analyze
    networks_to_analyze = []
    if args.network == "both":
        networks_to_analyze = [("student", False), ("teacher", True)]
    elif args.network == "student":
        networks_to_analyze = [("student", False)]
    else:
        networks_to_analyze = [("teacher", True)]

    print(f"Analyzing networks: {[n[0] for n in networks_to_analyze]}")

    # Build dataset
    dataset, grid_size = build_inference_dataset(cfg, args.file_list)
    print(f"Grid size: {grid_size}")

    # Limit samples if requested
    num_samples = len(dataset)
    if args.max_samples is not None:
        num_samples = min(num_samples, args.max_samples)

    print(f"Processing {num_samples} samples...")

    # Create data loader
    from torch.utils.data import DataLoader, Subset

    if args.max_samples is not None:
        indices = list(range(min(len(dataset), args.max_samples)))
        dataset = Subset(dataset, indices)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if args.device == "cuda" else False,
    )

    # Initialize statistics accumulators
    stats_accumulators = {}
    num_channels = None

    # Process data
    with torch.no_grad():
        for batch_idx, data_dict in enumerate(tqdm(loader, desc="Processing")):
            for network_name, use_teacher in networks_to_analyze:
                try:
                    features = get_backbone_features(
                        model, data_dict.copy(), use_teacher, args.device
                    )

                    # Initialize accumulator on first batch
                    if network_name not in stats_accumulators:
                        num_channels = features.shape[1]
                        stats_accumulators[network_name] = OnlineStatistics(num_channels)

                    # Update statistics
                    stats_accumulators[network_name].update(features)

                except Exception as e:
                    print(f"Warning: Error processing batch {batch_idx} for {network_name}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

    # Calculate and print statistics
    print("\n" + "=" * 80)
    print("FEATURE STANDARD DEVIATION STATISTICS")
    print("=" * 80)

    all_stats = {}
    for network_name, use_teacher in networks_to_analyze:
        if network_name not in stats_accumulators:
            print(f"\n{network_name}: No data processed")
            continue

        stats, channel_std = stats_accumulators[network_name].get_statistics(args.dead_threshold)
        if stats is None:
            print(f"\n{network_name}: Insufficient data")
            continue

        all_stats[network_name] = stats

        print(f"\n{network_name} backbone:")
        print("-" * 40)
        print(f"  Total tokens:          {stats['total_tokens']:,}")
        print(f"  Number of channels:    {stats['num_channels']}")
        print(f"  Global mean:           {stats['global_mean']:.6f}")
        print(f"  Global std:            {stats['global_std']:.6f}")
        print(f"  Batch std (avg):       {stats['batch_std']:.6f}")
        print(f"  Channel mean std:      {stats['channel_mean_std']:.6f}")
        print(f"  Channel median std:    {stats['channel_median_std']:.6f}")
        print(f"  Channel min std:       {stats['channel_min_std']:.6f}")
        print(f"  Channel max std:       {stats['channel_max_std']:.6f}")
        print(f"  Channel std of std:    {stats['channel_std_of_std']:.6f}")
        print(f"  Dead channels:         {stats['dead_channels']} ({stats['dead_channel_percent']:.2f}%)")

        # Interpretation
        if stats['global_std'] < 0.01:
            print(f"\n  WARNING: Very low global std ({stats['global_std']:.6f}) suggests feature collapse!")
        elif stats['global_std'] < 0.1:
            print(f"\n  CAUTION: Low global std ({stats['global_std']:.6f})")
        else:
            print(f"\n  Good: Healthy global std ({stats['global_std']:.6f})")

        if stats['dead_channel_percent'] > 10:
            print(f"  WARNING: High percentage of dead channels ({stats['dead_channel_percent']:.1f}%)")
        elif stats['dead_channel_percent'] > 1:
            print(f"  CAUTION: Some dead channels ({stats['dead_channel_percent']:.1f}%)")

        # Generate histogram if requested
        if args.histogram:
            histogram_path = os.path.join(args.output_dir, f"channel_std_histogram_{network_name}.png")
            create_histogram_plot(channel_std, histogram_path, network_name, args.dead_threshold)

            # Save per-channel statistics as CSV
            csv_path = os.path.join(args.output_dir, f"channel_std_{network_name}.csv")
            with open(csv_path, 'w') as f:
                f.write("channel,std,is_dead\n")
                for i, std_val in enumerate(channel_std):
                    is_dead = 1 if std_val < args.dead_threshold else 0
                    f.write(f"{i},{std_val:.10f},{is_dead}\n")
            print(f"  Saved channel std to: {csv_path}")

            # Save raw channel std as numpy
            npy_path = os.path.join(args.output_dir, f"channel_std_{network_name}.npy")
            np.save(npy_path, channel_std)
            print(f"  Saved channel std array to: {npy_path}")

    # Save summary statistics to JSON
    import json
    summary_path = os.path.join(args.output_dir, "feature_std_summary.json")
    with open(summary_path, 'w') as f:
        json.dump({
            'config_file': args.config_file,
            'weight_file': args.weight,
            'file_list': args.file_list,
            'num_samples_processed': num_samples,
            'dead_threshold': args.dead_threshold,
            'statistics': all_stats,
        }, f, indent=2)
    print(f"\nSaved summary to: {summary_path}")

    print("\n" + "=" * 80)
    print("Done!")


if __name__ == "__main__":
    main()
