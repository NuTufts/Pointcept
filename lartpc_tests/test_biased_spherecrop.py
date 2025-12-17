"""
Test script for BiasedSphereCrop transform.

This script compares class distributions when using different prob_random values
to demonstrate the effect of biased sampling toward neutrino interaction vertices.

Usage:
    python test_biased_spherecrop.py
"""

import os
import sys
import numpy as np
from collections import defaultdict

import pointcept
from pointcept.datasets import LArTPCDataset
from pointcept.datasets.transform import BiasedSphereCrop, GridSample, Collect

# Class names for LArTPC (matching the dataset config)
CLASS_NAMES = ["electron", "muon", "pion", "proton", "gamma", "ghost"]
NUM_CLASSES = len(CLASS_NAMES)

# Test parameters
NUM_SAMPLES = 100  # Number of samples to draw per prob_random setting
POINT_MAX = 98304  # Max points per crop
POINT_MIN = 20480  # Min points per crop
RADIUS = 5.0      # Gaussian offset std (in cm, assuming coord_scale=1.0)
GRID_SIZE = 0.01   # Grid size for voxelization (must match training)
TEST_FILE="pi0_test_1entry.txt"

# prob_random values to test
PROB_RANDOM_VALUES = [0.0, 0.5, 1.0]


def create_dataset_with_biased_crop(prob_random):
    """
    Create a LArTPCDataset with BiasedSphereCrop transform.

    Args:
        prob_random: Probability of using random mode instead of biased sampling

    Returns:
        LArTPCDataset instance
    """
    transform = [
        # Grid sampling (voxelization) - must happen before sphere crop
        dict(
            type="GridSample",
            grid_size=GRID_SIZE,
            hash_type="fnv",
            mode="train",
            return_grid_coord=True,
        ),
        # Biased sphere crop toward neutrino vertices
        dict(
            type="BiasedSphereCrop",
            anchor_points_key="nu_vertices",
            anchor_pdf_key=None,  # Uniform sampling over vertices
            radius=RADIUS,
            point_max=POINT_MAX,
            point_min=POINT_MIN,
            prob_random=prob_random,
            max_retries=100,
            fallback_to_random=True,
        ),
        # Convert numpy arrays to torch tensors
        dict(type="ToTensor"),
        # Collect the data we need
        dict(
            type="Collect",
            keys=("coord", "grid_coord", "segment"),
            offset_keys_dict=dict(),
            feat_keys=("strength", "color"),
        ),
    ]

    dataset = LArTPCDataset(
        coord_scale=1.0,
        data_list_file=TEST_FILE,  # Use training file list
        transform=transform,
        exclude_other=True,
        include_ghosts=True,
        use_edep_as_strength=True,
        log_transform_edep=True,
        wire_scale=1.0/3456.0,
    )

    return dataset


def count_class_points(dataset, num_samples, verbose=False):
    """
    Sample from dataset and count points per class.

    Args:
        dataset: LArTPCDataset instance
        num_samples: Number of samples to draw
        verbose: If True, print full traceback on errors

    Returns:
        dict mapping class_idx -> total point count
    """
    import traceback

    class_counts = defaultdict(int)
    total_points = 0
    success_count = 0

    # Get dataset size
    dataset_size = len(dataset)

    for i in range(num_samples):
        # Sample a random index from dataset
        idx = np.random.randint(dataset_size)

        try:
            data = dataset[idx]
            success_count += 1
        except Exception as e:
            if verbose or i == 0:  # Print full traceback for first error
                print(f"  Error loading sample {idx}:")
                traceback.print_exc()
            else:
                print(f"  Warning: Failed to load sample {idx}: {e}")
            continue

        # Get segment labels
        segment = data["segment"].numpy() if hasattr(data["segment"], "numpy") else data["segment"]

        # Count points per class
        for class_idx in range(NUM_CLASSES):
            count = (segment == class_idx).sum()
            class_counts[class_idx] += count
            total_points += count

        # Progress indicator
        if (i + 1) % 20 == 0:
            print(f"    Processed {i + 1}/{num_samples} samples ({success_count} successful)...")

    print(f"    Completed: {success_count}/{num_samples} samples loaded successfully")
    return class_counts, total_points


def print_results(results):
    """
    Print comparison table of results.

    Args:
        results: dict mapping prob_random -> (class_counts, total_points)
    """
    print("\n" + "=" * 80)
    print("RESULTS: Class Distribution Comparison")
    print("=" * 80)

    # Header
    header = f"{'Class':<12}"
    for prob_random in PROB_RANDOM_VALUES:
        header += f"  prob_random={prob_random:<6}"
    print(header)
    print("-" * 80)

    # Per-class rows
    for class_idx in range(NUM_CLASSES):
        row = f"{CLASS_NAMES[class_idx]:<12}"
        for prob_random in PROB_RANDOM_VALUES:
            class_counts, total_points = results[prob_random]
            count = class_counts[class_idx]
            fraction = count / total_points * 100 if total_points > 0 else 0
            row += f"  {fraction:>6.2f}%          "
        print(row)

    print("-" * 80)

    # Total row
    row = f"{'TOTAL':<12}"
    for prob_random in PROB_RANDOM_VALUES:
        _, total_points = results[prob_random]
        row += f"  {total_points:>10,} pts    "
    print(row)

    print("=" * 80)

    # Analysis
    print("\nANALYSIS:")
    print("-" * 40)

    # Compare signal vs background
    signal_classes = [0, 2, 3, 4]  # electron, pion, proton, gamma (neutrino products)
    background_classes = [1, 5]     # muon (cosmic), ghost

    print("\nSignal classes (electron, pion, proton, gamma) - from neutrino:")
    for prob_random in PROB_RANDOM_VALUES:
        class_counts, total_points = results[prob_random]
        signal_count = sum(class_counts[c] for c in signal_classes)
        signal_frac = signal_count / total_points * 100 if total_points > 0 else 0
        print(f"  prob_random={prob_random}: {signal_frac:.2f}%")

    print("\nBackground classes (muon, ghost) - mostly cosmic:")
    for prob_random in PROB_RANDOM_VALUES:
        class_counts, total_points = results[prob_random]
        bg_count = sum(class_counts[c] for c in background_classes)
        bg_frac = bg_count / total_points * 100 if total_points > 0 else 0
        print(f"  prob_random={prob_random}: {bg_frac:.2f}%")

    # Improvement factor
    if 0.0 in results and 1.0 in results:
        biased_counts, biased_total = results[0.0]
        random_counts, random_total = results[1.0]

        if biased_total > 0 and random_total > 0:
            biased_signal = sum(biased_counts[c] for c in signal_classes) / biased_total * 100
            random_signal = sum(random_counts[c] for c in signal_classes) / random_total * 100

            improvement = biased_signal / random_signal if random_signal > 0 else float('inf')
            print(f"\nSignal enrichment factor (biased vs random): {improvement:.2f}x")
        else:
            print("\nCould not compute enrichment factor (no data collected)")


def main():
    print("=" * 80)
    print("BiasedSphereCrop Test: Comparing class distributions")
    print("=" * 80)
    print(f"\nTest parameters:")
    print(f"  NUM_SAMPLES per setting: {NUM_SAMPLES}")
    print(f"  POINT_MAX: {POINT_MAX}")
    print(f"  POINT_MIN: {POINT_MIN}")
    print(f"  RADIUS (Gaussian std): {RADIUS} cm")
    print(f"  GRID_SIZE: {GRID_SIZE} cm")
    print(f"  prob_random values: {PROB_RANDOM_VALUES}")
    print()

    results = {}

    for prob_random in PROB_RANDOM_VALUES:
        print(f"\n{'='*40}")
        print(f"Testing prob_random = {prob_random}")
        print(f"{'='*40}")

        # Create dataset with this prob_random setting
        print("  Creating dataset...")
        dataset = create_dataset_with_biased_crop(prob_random)
        print(f"  Dataset size: {len(dataset)} files")

        # Count class points
        print(f"  Sampling {NUM_SAMPLES} crops...")
        class_counts, total_points = count_class_points(dataset, NUM_SAMPLES)

        results[prob_random] = (class_counts, total_points)

        # Print intermediate results
        print(f"\n  Results for prob_random={prob_random}:")
        for class_idx in range(NUM_CLASSES):
            count = class_counts[class_idx]
            frac = count / total_points * 100 if total_points > 0 else 0
            print(f"    {CLASS_NAMES[class_idx]:<10}: {count:>10,} points ({frac:>5.2f}%)")
        print(f"    {'TOTAL':<10}: {total_points:>10,} points")

    # Print comparison
    print_results(results)


if __name__ == "__main__":
    main()
