"""
Prototype Usage Statistics Calculator for Sonata Pretrained Models

This tool calculates prototype usage statistics to detect representation collapse
in Sonata self-supervised pretrained models. A large number of unused prototypes
indicates potential representation collapse.

Statistics calculated:
- used_count: Number of prototypes with at least one assigned token
- unused_count: Number of prototypes with no assigned tokens
- unused_percent: Percentage of unused prototypes
- tokens_per_prototype: Average tokens per used prototype
- assignment_entropy: Entropy of the prototype assignment distribution

Optional outputs:
- 2D histogram of prototype vs class labels
- Per-prototype class distribution

Usage:
    python tools/prototype_stats.py --config-file CONFIG --weight WEIGHT --file-list FILES.txt

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
        description="Calculate prototype usage statistics for Sonata models"
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
        default="prototype_stats_output",
        help="Directory to save output files (default: prototype_stats_output)"
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
        "--head",
        type=str,
        default="both",
        choices=["mask_head", "unmask_head", "both"],
        help="Which head to analyze (default: both)"
    )
    parser.add_argument(
        "--teacher",
        action="store_true",
        help="Use teacher model instead of student (default: student)"
    )
    parser.add_argument(
        "--histogram",
        action="store_true",
        help="Generate 2D histogram of prototype vs class"
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
    # strength=3, color=3 (wire coords)
    in_channels = cfg.model.get('backbone', {}).get('in_channels', 6)
    if in_channels == 3:
        # Model uses only strength (3 channels)
        feat_keys = ("strength",)
        print("Using 3 input channels (strength only)")
    else:
        # Model uses strength + color (6 channels)
        feat_keys = ("strength", "color")
        print(f"Using {in_channels} input channels (strength + color)")

    # Build minimal transform pipeline for inference
    # This matches the expected input format for Sonata backbone
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
            # Accumulate offsets
            offsets = [0]
            for item in batch:
                offsets.append(offsets[-1] + item["offset"][-1].item())
            result[key] = torch.tensor(offsets[1:], dtype=torch.long)
        elif key == "name":
            result[key] = [item[key] for item in batch]
        elif key == "grid_size":
            # Grid size is same for all items, just take the first
            result[key] = batch[0][key]
        elif isinstance(batch[0][key], torch.Tensor):
            result[key] = torch.cat([item[key] for item in batch], dim=0)
        else:
            result[key] = [item[key] for item in batch]
    return result


def get_prototype_assignments(model, data_dict, head_name, use_teacher, device):
    """
    Get prototype assignments for a batch of data.

    Returns:
        assignments: (N,) tensor of prototype indices
        num_prototypes: total number of prototypes
    """
    # Move data to device
    for key in list(data_dict.keys()):
        if isinstance(data_dict[key], torch.Tensor):
            data_dict[key] = data_dict[key].to(device)
        elif key == 'grid_size' and not isinstance(data_dict[key], torch.Tensor):
            # grid_size might be a float, convert to tensor
            data_dict[key] = torch.tensor(data_dict[key], device=device)

    with torch.no_grad():
        # Get features from backbone using return_point=True mode
        # This runs the teacher backbone and returns upcast features
        result = model(data_dict, return_point=True)
        point = result['point']

        # Get the appropriate network (student or teacher) and head
        network = model.teacher if use_teacher else model.student

        # ModuleDict access
        if head_name in network:
            head = network[head_name]
        else:
            raise ValueError(f"Head '{head_name}' not found in {'teacher' if use_teacher else 'student'}. "
                           f"Available heads: {list(network.keys())}")

        # Get prototype logits
        logits = head(point.feat)  # (N, num_prototypes)

        # Get assignments by argmax
        assignments = logits.argmax(dim=-1)  # (N,)
        num_prototypes = logits.shape[-1]

    return assignments.cpu(), num_prototypes


def calculate_statistics(prototype_counts, total_tokens, num_prototypes):
    """Calculate prototype usage statistics."""
    # Basic counts
    used_mask = prototype_counts > 0
    used_count = used_mask.sum().item()
    unused_count = num_prototypes - used_count
    unused_percent = (unused_count / num_prototypes) * 100

    # Tokens per prototype
    tokens_per_prototype = total_tokens / used_count if used_count > 0 else 0

    # Assignment entropy
    probs = prototype_counts / total_tokens if total_tokens > 0 else prototype_counts
    # Avoid log(0)
    probs_nonzero = probs[probs > 0]
    entropy = -torch.sum(probs_nonzero * torch.log(probs_nonzero)).item()

    # Max entropy (uniform distribution)
    max_entropy = np.log(num_prototypes)
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

    stats = {
        "num_prototypes": num_prototypes,
        "used_count": used_count,
        "unused_count": unused_count,
        "unused_percent": unused_percent,
        "total_tokens": total_tokens,
        "tokens_per_prototype_avg": tokens_per_prototype,
        "assignment_entropy": entropy,
        "max_entropy": max_entropy,
        "normalized_entropy": normalized_entropy,
    }

    return stats


def create_histogram_plot(prototype_class_counts, class_names, output_path, head_name):
    """Create and save 2D histogram of prototype vs class."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available, skipping histogram plot")
        return

    num_prototypes, num_classes = prototype_class_counts.shape

    # Normalize by row (per prototype)
    row_sums = prototype_class_counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    normalized = prototype_class_counts / row_sums

    # Find prototypes with significant usage
    total_per_prototype = prototype_class_counts.sum(axis=1)
    threshold = np.percentile(total_per_prototype[total_per_prototype > 0], 5) if (total_per_prototype > 0).any() else 0
    active_prototypes = total_per_prototype > threshold

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # 1. Full 2D histogram (raw counts, log scale)
    ax1 = axes[0, 0]
    im1 = ax1.imshow(
        np.log10(prototype_class_counts + 1).T,
        aspect='auto',
        cmap='viridis',
        interpolation='nearest'
    )
    ax1.set_xlabel('Prototype Index')
    ax1.set_ylabel('Class')
    ax1.set_yticks(range(len(class_names)))
    ax1.set_yticklabels(class_names)
    ax1.set_title(f'Prototype-Class Counts (log10 scale) - {head_name}')
    plt.colorbar(im1, ax=ax1, label='log10(count + 1)')

    # 2. Normalized by prototype (shows class distribution per prototype)
    ax2 = axes[0, 1]
    im2 = ax2.imshow(
        normalized.T,
        aspect='auto',
        cmap='viridis',
        interpolation='nearest'
    )
    ax2.set_xlabel('Prototype Index')
    ax2.set_ylabel('Class')
    ax2.set_yticks(range(len(class_names)))
    ax2.set_yticklabels(class_names)
    ax2.set_title(f'Class Distribution per Prototype - {head_name}')
    plt.colorbar(im2, ax=ax2, label='Proportion')

    # 3. Usage histogram
    ax3 = axes[1, 0]
    usage = prototype_class_counts.sum(axis=1)
    ax3.bar(range(len(usage)), usage, width=1.0, edgecolor='none')
    ax3.set_xlabel('Prototype Index')
    ax3.set_ylabel('Token Count')
    ax3.set_title(f'Prototype Usage - {head_name}')
    ax3.set_yscale('log')

    # 4. Class-wise aggregated statistics
    ax4 = axes[1, 1]
    class_totals = prototype_class_counts.sum(axis=0)
    bars = ax4.bar(range(len(class_names)), class_totals)
    ax4.set_xlabel('Class')
    ax4.set_ylabel('Total Token Count')
    ax4.set_xticks(range(len(class_names)))
    ax4.set_xticklabels(class_names, rotation=45, ha='right')
    ax4.set_title(f'Total Tokens per Class - {head_name}')

    # Add value labels on bars
    for bar, val in zip(bars, class_totals):
        height = bar.get_height()
        ax4.annotate(f'{int(val):,}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved histogram to: {output_path}")


def calculate_dominant_class_per_prototype(prototype_class_counts, class_names):
    """Calculate the dominant class for each prototype."""
    num_prototypes = prototype_class_counts.shape[0]
    dominant_classes = []

    for i in range(num_prototypes):
        counts = prototype_class_counts[i]
        if counts.sum() > 0:
            dominant_idx = counts.argmax()
            dominant_name = class_names[dominant_idx]
            dominant_frac = counts[dominant_idx] / counts.sum()
            dominant_classes.append({
                'prototype': i,
                'dominant_class': dominant_name,
                'dominant_class_idx': int(dominant_idx),
                'dominant_fraction': float(dominant_frac),
                'total_tokens': int(counts.sum()),
            })
        else:
            dominant_classes.append({
                'prototype': i,
                'dominant_class': 'unused',
                'dominant_class_idx': -1,
                'dominant_fraction': 0.0,
                'total_tokens': 0,
            })

    return dominant_classes


def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model, cfg = load_model(args.config_file, args.weight, args.device)

    # Determine which heads to analyze
    heads_to_analyze = []
    if args.head == "both":
        if hasattr(model.student, 'mask_head') or 'mask_head' in model.student:
            heads_to_analyze.append("mask_head")
        if hasattr(model.student, 'unmask_head') or 'unmask_head' in model.student:
            heads_to_analyze.append("unmask_head")
    else:
        heads_to_analyze.append(args.head)

    if not heads_to_analyze:
        print("Error: No valid heads found in model")
        return

    print(f"Analyzing heads: {heads_to_analyze}")
    print(f"Using {'teacher' if args.teacher else 'student'} network")

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

    # Get class names
    class_names = cfg.data.get('names', [f'class_{i}' for i in range(cfg.data.num_classes)])
    num_classes = len(class_names)

    # Initialize accumulators for each head
    results = {}
    for head_name in heads_to_analyze:
        results[head_name] = {
            'prototype_counts': None,
            'prototype_class_counts': None,
            'total_tokens': 0,
            'num_prototypes': None,
        }

    # Process data
    with torch.no_grad():
        for batch_idx, data_dict in enumerate(tqdm(loader, desc="Processing")):
            # Get segment labels
            segments = data_dict.get('segment', None)
            if segments is not None:
                segments = segments.cpu().numpy()

            for head_name in heads_to_analyze:
                try:
                    assignments, num_prototypes = get_prototype_assignments(
                        model, data_dict.copy(), head_name, args.teacher, args.device
                    )
                except Exception as e:
                    print(f"Warning: Error processing batch {batch_idx} for {head_name}: {e}")
                    continue

                # Initialize accumulators if needed
                if results[head_name]['num_prototypes'] is None:
                    results[head_name]['num_prototypes'] = num_prototypes
                    results[head_name]['prototype_counts'] = torch.zeros(num_prototypes, dtype=torch.float64)
                    if args.histogram and segments is not None:
                        results[head_name]['prototype_class_counts'] = np.zeros(
                            (num_prototypes, num_classes), dtype=np.int64
                        )

                # Accumulate prototype counts
                counts = torch.bincount(assignments, minlength=num_prototypes).float()
                results[head_name]['prototype_counts'] += counts
                results[head_name]['total_tokens'] += len(assignments)

                # Accumulate prototype-class counts for histogram
                if args.histogram and segments is not None:
                    assignments_np = assignments.numpy()
                    for proto_idx, class_idx in zip(assignments_np, segments):
                        if 0 <= class_idx < num_classes:  # Skip ignore_index (-1)
                            results[head_name]['prototype_class_counts'][proto_idx, class_idx] += 1

    # Calculate and print statistics for each head
    print("\n" + "=" * 80)
    print("PROTOTYPE USAGE STATISTICS")
    print("=" * 80)

    all_stats = {}
    for head_name in heads_to_analyze:
        r = results[head_name]
        if r['num_prototypes'] is None:
            print(f"\n{head_name}: No data processed")
            continue

        stats = calculate_statistics(
            r['prototype_counts'],
            r['total_tokens'],
            r['num_prototypes']
        )
        all_stats[head_name] = stats

        print(f"\n{head_name} ({'teacher' if args.teacher else 'student'}):")
        print("-" * 40)
        print(f"  Total prototypes:      {stats['num_prototypes']:,}")
        print(f"  Used prototypes:       {stats['used_count']:,}")
        print(f"  Unused prototypes:     {stats['unused_count']:,}")
        print(f"  Unused percentage:     {stats['unused_percent']:.2f}%")
        print(f"  Total tokens:          {stats['total_tokens']:,}")
        print(f"  Tokens per prototype:  {stats['tokens_per_prototype_avg']:.2f}")
        print(f"  Assignment entropy:    {stats['assignment_entropy']:.4f}")
        print(f"  Max entropy:           {stats['max_entropy']:.4f}")
        print(f"  Normalized entropy:    {stats['normalized_entropy']:.4f}")

        # Interpretation
        if stats['unused_percent'] > 50:
            print(f"\n  WARNING: High unused percentage ({stats['unused_percent']:.1f}%) suggests representation collapse!")
        elif stats['unused_percent'] > 20:
            print(f"\n  CAUTION: Moderate unused percentage ({stats['unused_percent']:.1f}%)")
        else:
            print(f"\n  Good: Low unused percentage ({stats['unused_percent']:.1f}%)")

        if stats['normalized_entropy'] < 0.5:
            print(f"  WARNING: Low normalized entropy ({stats['normalized_entropy']:.2f}) suggests unbalanced usage")

        # Generate histogram if requested
        if args.histogram and r['prototype_class_counts'] is not None:
            histogram_path = os.path.join(args.output_dir, f"prototype_class_histogram_{head_name}.png")
            create_histogram_plot(
                r['prototype_class_counts'],
                class_names,
                histogram_path,
                head_name
            )

            # Calculate and save dominant class info
            dominant_info = calculate_dominant_class_per_prototype(
                r['prototype_class_counts'],
                class_names
            )

            # Save as CSV
            csv_path = os.path.join(args.output_dir, f"prototype_dominant_class_{head_name}.csv")
            with open(csv_path, 'w') as f:
                f.write("prototype,dominant_class,dominant_class_idx,dominant_fraction,total_tokens\n")
                for info in dominant_info:
                    f.write(f"{info['prototype']},{info['dominant_class']},{info['dominant_class_idx']},"
                           f"{info['dominant_fraction']:.4f},{info['total_tokens']}\n")
            print(f"  Saved dominant class info to: {csv_path}")

            # Save raw counts
            counts_path = os.path.join(args.output_dir, f"prototype_class_counts_{head_name}.npy")
            np.save(counts_path, r['prototype_class_counts'])
            print(f"  Saved prototype-class counts to: {counts_path}")

    # Save summary statistics to JSON
    import json
    summary_path = os.path.join(args.output_dir, "prototype_stats_summary.json")
    with open(summary_path, 'w') as f:
        json.dump({
            'config_file': args.config_file,
            'weight_file': args.weight,
            'file_list': args.file_list,
            'num_samples_processed': num_samples,
            'use_teacher': args.teacher,
            'statistics': all_stats,
        }, f, indent=2)
    print(f"\nSaved summary to: {summary_path}")

    print("\n" + "=" * 80)
    print("Done!")


if __name__ == "__main__":
    main()
