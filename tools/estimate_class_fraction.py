"""
Estimate Class Fractions from Training Dataset

This script iterates through the training dataset (with all transforms applied,
including random subsampling like SphereCrop or BiasedSphereCrop) and calculates
the fraction of points belonging to each class.

The output can be used to set the 'class_priors' parameter in model configs
for proper initialization of the segmentation head bias.

Usage:
    python tools/estimate_class_fraction.py \
        --config-file configs/lartpc/semseg/archive/semseg-sonata-v1m1-lartpc-v3-decoder-finetune.py \
        --num-samples 1000

    # With specific output format for config
    python tools/estimate_class_fraction.py \
        --config-file configs/lartpc/semseg/archive/semseg-sonata-v1m1-lartpc-v3-decoder-finetune.py \
        --num-samples 500 \
        --output-format config

Author: Generated for LArTPC analysis
"""

import os
import sys
import argparse
import numpy as np
from collections import defaultdict
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pointcept.utils.config import Config
from pointcept.datasets.builder import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate class fractions from training dataset"
    )
    parser.add_argument(
        "--config-file",
        required=True,
        help="Path to training config file (e.g., configs/lartpc/semseg-*.py)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples to iterate through (default: 100)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Which dataset split to use (default: train)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="table",
        choices=["table", "config", "json"],
        help="Output format: table (human-readable), config (copy-paste for config), json"
    )
    parser.add_argument(
        "--ignore-index",
        type=int,
        default=-1,
        help="Label index to ignore (default: -1)"
    )
    return parser.parse_args()


def build_dataset_from_config(cfg, split="train"):
    """Build dataset from config for the specified split."""
    if split == "train":
        data_cfg = cfg.data.train.copy()
    elif split == "val":
        data_cfg = cfg.data.val.copy()
    elif split == "test":
        data_cfg = cfg.data.test.copy()
    else:
        raise ValueError(f"Unknown split: {split}")

    # Set loop=1 to avoid repeating data
    data_cfg['loop'] = 1

    dataset = build_dataset(data_cfg)
    return dataset


def estimate_class_fractions(dataset, num_samples, ignore_index=-1, seed=42):
    """
    Iterate through dataset samples and count class occurrences.

    Args:
        dataset: The dataset to iterate through
        num_samples: Number of samples to process
        ignore_index: Label index to ignore in counting
        seed: Random seed for reproducibility

    Returns:
        dict with class_counts, class_fractions, total_points, num_samples_processed
    """
    np.random.seed(seed)

    # Initialize counters
    class_counts = defaultdict(int)
    total_points = 0
    samples_processed = 0

    # Determine actual number of samples to process
    num_samples = min(num_samples, len(dataset))

    # Generate random indices if dataset is larger than num_samples
    if len(dataset) > num_samples:
        indices = np.random.choice(len(dataset), size=num_samples, replace=False)
    else:
        indices = list(range(len(dataset)))

    print(f"Processing {num_samples} samples from dataset of size {len(dataset)}...")

    for i, idx in enumerate(tqdm(indices, desc="Processing samples")):
        try:
            # Get data from dataset (this applies all transforms)
            data = dataset[idx]

            # Extract segment labels
            if 'segment' in data:
                segment = data['segment']

                # Convert to numpy if tensor
                if hasattr(segment, 'numpy'):
                    segment = segment.numpy()
                elif hasattr(segment, 'cpu'):
                    segment = segment.cpu().numpy()

                segment = np.asarray(segment).flatten()

                # Count each class
                for label in segment:
                    label = int(label)
                    if label != ignore_index:
                        class_counts[label] += 1
                        total_points += 1

                samples_processed += 1
            else:
                print(f"Warning: Sample {idx} has no 'segment' key")

        except Exception as e:
            print(f"Warning: Error processing sample {idx}: {e}")
            continue

    # Calculate fractions
    class_fractions = {}
    if total_points > 0:
        for label, count in class_counts.items():
            class_fractions[label] = count / total_points

    return {
        'class_counts': dict(class_counts),
        'class_fractions': class_fractions,
        'total_points': total_points,
        'num_samples_processed': samples_processed
    }


def format_output(results, class_names, output_format="table", num_classes=None):
    """Format the results for output."""
    class_counts = results['class_counts']
    class_fractions = results['class_fractions']
    total_points = results['total_points']
    num_samples = results['num_samples_processed']

    # Determine the number of classes
    if num_classes is None:
        num_classes = max(class_counts.keys()) + 1 if class_counts else 0

    # Ensure class_names covers all classes
    if class_names is None or len(class_names) < num_classes:
        class_names = [f"class_{i}" for i in range(num_classes)]

    if output_format == "table":
        output = []
        output.append("=" * 70)
        output.append("CLASS FRACTION ESTIMATION RESULTS")
        output.append("=" * 70)
        output.append(f"Samples processed: {num_samples}")
        output.append(f"Total points counted: {total_points:,}")
        output.append("")
        output.append(f"{'Class':<4} {'Name':<12} {'Count':>15} {'Fraction':>12} {'Percent':>10}")
        output.append("-" * 70)

        for i in range(num_classes):
            count = class_counts.get(i, 0)
            fraction = class_fractions.get(i, 0.0)
            name = class_names[i] if i < len(class_names) else f"class_{i}"
            output.append(f"{i:<4} {name:<12} {count:>15,} {fraction:>12.5f} {fraction*100:>9.3f}%")

        output.append("-" * 70)
        output.append(f"{'Total':<17} {total_points:>15,} {1.0:>12.5f} {100.0:>9.3f}%")
        output.append("=" * 70)

        return "\n".join(output)

    elif output_format == "config":
        # Format for direct copy-paste into config file
        fractions = [class_fractions.get(i, 0.0) for i in range(num_classes)]
        fractions_str = ", ".join([f"{f:.5f}" for f in fractions])

        output = []
        output.append("# Class priors for model config")
        output.append(f"# Estimated from {num_samples} samples, {total_points:,} total points")
        output.append(f"# Classes: {', '.join(class_names[:num_classes])}")
        output.append(f"class_priors=[{fractions_str}],")

        return "\n".join(output)

    elif output_format == "json":
        import json
        data = {
            'num_samples_processed': num_samples,
            'total_points': total_points,
            'class_names': class_names[:num_classes],
            'class_counts': {class_names[i] if i < len(class_names) else f"class_{i}": class_counts.get(i, 0)
                           for i in range(num_classes)},
            'class_fractions': {class_names[i] if i < len(class_names) else f"class_{i}": class_fractions.get(i, 0.0)
                              for i in range(num_classes)},
            'class_priors_list': [class_fractions.get(i, 0.0) for i in range(num_classes)]
        }
        return json.dumps(data, indent=2)

    else:
        raise ValueError(f"Unknown output format: {output_format}")


def main():
    args = parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Load config
    print(f"Loading config from: {args.config_file}")
    cfg = Config.fromfile(args.config_file)

    # Get class names and num_classes from config
    class_names = cfg.data.get('names', None)
    num_classes = cfg.data.get('num_classes', None)

    if class_names:
        print(f"Class names from config: {class_names}")
    if num_classes:
        print(f"Number of classes from config: {num_classes}")

    # Build dataset
    print(f"Building {args.split} dataset...")
    dataset = build_dataset_from_config(cfg, split=args.split)
    print(f"Dataset size: {len(dataset)} samples")

    # Estimate class fractions
    results = estimate_class_fractions(
        dataset,
        num_samples=args.num_samples,
        ignore_index=args.ignore_index,
        seed=args.seed
    )

    # Format and print output
    output = format_output(
        results,
        class_names=class_names,
        output_format=args.output_format,
        num_classes=num_classes
    )
    print("\n" + output)

    # If table format, also print config-ready format
    if args.output_format == "table":
        print("\n" + "=" * 70)
        print("CONFIG-READY FORMAT (copy-paste into your config):")
        print("=" * 70)
        config_output = format_output(
            results,
            class_names=class_names,
            output_format="config",
            num_classes=num_classes
        )
        print(config_output)


if __name__ == "__main__":
    main()
