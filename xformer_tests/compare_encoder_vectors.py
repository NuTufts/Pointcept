"""
Compare encoder vectors from two different runs.

This script compares encoder outputs from two different flash attention backends
(e.g., flash_attn on A100 vs xformers on P100) and quantifies their differences.

Usage:
    python compare_encoder_vectors.py \
        --file1 encoder_vectors_a100.pt \
        --file2 encoder_vectors_p100.pt \
        --output comparison_report.txt

Metrics computed:
    - Mean Absolute Error (MAE)
    - Root Mean Squared Error (RMSE)
    - Relative Error (percentage)
    - Cosine Similarity (per-point and overall)
    - Max Absolute Difference
    - Correlation coefficient
    - Per-channel statistics

Author: Generated for xformers backend testing
"""

import argparse
import os
import sys
import torch
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare encoder vectors from two different runs"
    )
    parser.add_argument(
        "--file1",
        type=str,
        required=True,
        help="First encoder vectors file (.pt)",
    )
    parser.add_argument(
        "--file2",
        type=str,
        required=True,
        help="Second encoder vectors file (.pt)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for comparison report (optional, prints to stdout if not specified)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-3,
        help="Tolerance for considering values as 'close' (default: 1e-3)",
    )
    return parser.parse_args()


def compute_metrics(feat1, feat2, tolerance=1e-3):
    """Compute comparison metrics between two feature tensors."""
    metrics = {}

    # Ensure same shape
    if feat1.shape != feat2.shape:
        metrics['error'] = f"Shape mismatch: {feat1.shape} vs {feat2.shape}"
        return metrics

    # Convert to float64 for precision
    feat1 = feat1.double()
    feat2 = feat2.double()

    # Absolute differences
    abs_diff = torch.abs(feat1 - feat2)

    # Mean Absolute Error
    metrics['mae'] = abs_diff.mean().item()

    # Root Mean Squared Error
    metrics['rmse'] = torch.sqrt(((feat1 - feat2) ** 2).mean()).item()

    # Max Absolute Difference
    metrics['max_abs_diff'] = abs_diff.max().item()

    # Relative Error (avoid division by zero)
    denom = torch.abs(feat1).clamp(min=1e-8)
    rel_error = abs_diff / denom
    metrics['mean_relative_error'] = rel_error.mean().item() * 100  # percentage
    metrics['max_relative_error'] = rel_error.max().item() * 100

    # Fraction of values within tolerance
    within_tolerance = (abs_diff < tolerance).float().mean().item() * 100
    metrics['percent_within_tolerance'] = within_tolerance

    # Cosine similarity (per point)
    # Normalize each row and compute dot product
    norm1 = torch.nn.functional.normalize(feat1, p=2, dim=1)
    norm2 = torch.nn.functional.normalize(feat2, p=2, dim=1)
    per_point_cosine = (norm1 * norm2).sum(dim=1)
    metrics['cosine_sim_mean'] = per_point_cosine.mean().item()
    metrics['cosine_sim_min'] = per_point_cosine.min().item()
    metrics['cosine_sim_std'] = per_point_cosine.std().item()

    # Overall cosine similarity (flatten)
    flat1 = feat1.flatten()
    flat2 = feat2.flatten()
    overall_cosine = torch.nn.functional.cosine_similarity(
        flat1.unsqueeze(0), flat2.unsqueeze(0)
    ).item()
    metrics['cosine_sim_overall'] = overall_cosine

    # Pearson correlation
    mean1 = flat1.mean()
    mean2 = flat2.mean()
    centered1 = flat1 - mean1
    centered2 = flat2 - mean2
    correlation = (centered1 * centered2).sum() / (
        torch.sqrt((centered1 ** 2).sum() * (centered2 ** 2).sum()) + 1e-8
    )
    metrics['pearson_correlation'] = correlation.item()

    # Per-channel statistics
    n_channels = feat1.shape[1]
    channel_mae = abs_diff.mean(dim=0)
    metrics['channel_mae_mean'] = channel_mae.mean().item()
    metrics['channel_mae_max'] = channel_mae.max().item()
    metrics['channel_mae_min'] = channel_mae.min().item()

    # Distribution of differences
    metrics['diff_percentile_50'] = torch.quantile(abs_diff.flatten(), 0.5).item()
    metrics['diff_percentile_90'] = torch.quantile(abs_diff.flatten(), 0.9).item()
    metrics['diff_percentile_99'] = torch.quantile(abs_diff.flatten(), 0.99).item()

    return metrics


def print_report(data1, data2, metrics, output_file=None):
    """Print/save comparison report."""
    lines = []

    lines.append("=" * 70)
    lines.append("ENCODER VECTORS COMPARISON REPORT")
    lines.append("=" * 70)
    lines.append("")

    # File information
    lines.append("FILE INFORMATION:")
    lines.append("-" * 40)
    lines.append(f"File 1:")
    lines.append(f"  Path: {data1.get('data_file', 'N/A')}")
    lines.append(f"  Backend: {data1.get('flash_backend', 'N/A')}")
    lines.append(f"  Config: {data1.get('config', 'N/A')}")
    lines.append(f"  Checkpoint: {data1.get('checkpoint', 'N/A')}")
    lines.append("")
    lines.append(f"File 2:")
    lines.append(f"  Path: {data2.get('data_file', 'N/A')}")
    lines.append(f"  Backend: {data2.get('flash_backend', 'N/A')}")
    lines.append(f"  Config: {data2.get('config', 'N/A')}")
    lines.append(f"  Checkpoint: {data2.get('checkpoint', 'N/A')}")
    lines.append("")

    # Shape information
    lines.append("TENSOR INFORMATION:")
    lines.append("-" * 40)
    feat1 = data1['features']
    feat2 = data2['features']
    lines.append(f"Features 1 shape: {tuple(feat1.shape)}")
    lines.append(f"Features 2 shape: {tuple(feat2.shape)}")
    lines.append(f"Features 1 dtype: {feat1.dtype}")
    lines.append(f"Features 2 dtype: {feat2.dtype}")
    lines.append("")

    # Check for errors
    if 'error' in metrics:
        lines.append(f"ERROR: {metrics['error']}")
        lines.append("")
    else:
        # Main metrics
        lines.append("COMPARISON METRICS:")
        lines.append("-" * 40)
        lines.append("")

        lines.append("Absolute Error Metrics:")
        lines.append(f"  Mean Absolute Error (MAE):     {metrics['mae']:.6e}")
        lines.append(f"  Root Mean Squared Error:       {metrics['rmse']:.6e}")
        lines.append(f"  Max Absolute Difference:       {metrics['max_abs_diff']:.6e}")
        lines.append("")

        lines.append("Relative Error Metrics:")
        lines.append(f"  Mean Relative Error:           {metrics['mean_relative_error']:.4f}%")
        lines.append(f"  Max Relative Error:            {metrics['max_relative_error']:.4f}%")
        lines.append("")

        lines.append("Similarity Metrics:")
        lines.append(f"  Cosine Similarity (overall):   {metrics['cosine_sim_overall']:.8f}")
        lines.append(f"  Cosine Similarity (per-point mean): {metrics['cosine_sim_mean']:.8f}")
        lines.append(f"  Cosine Similarity (per-point min):  {metrics['cosine_sim_min']:.8f}")
        lines.append(f"  Cosine Similarity (per-point std):  {metrics['cosine_sim_std']:.8f}")
        lines.append(f"  Pearson Correlation:           {metrics['pearson_correlation']:.8f}")
        lines.append("")

        lines.append("Tolerance Check:")
        lines.append(f"  Values within tolerance:       {metrics['percent_within_tolerance']:.2f}%")
        lines.append("")

        lines.append("Per-Channel MAE:")
        lines.append(f"  Mean:                          {metrics['channel_mae_mean']:.6e}")
        lines.append(f"  Max:                           {metrics['channel_mae_max']:.6e}")
        lines.append(f"  Min:                           {metrics['channel_mae_min']:.6e}")
        lines.append("")

        lines.append("Difference Distribution (absolute):")
        lines.append(f"  50th percentile (median):      {metrics['diff_percentile_50']:.6e}")
        lines.append(f"  90th percentile:               {metrics['diff_percentile_90']:.6e}")
        lines.append(f"  99th percentile:               {metrics['diff_percentile_99']:.6e}")
        lines.append("")

        # Interpretation
        lines.append("INTERPRETATION:")
        lines.append("-" * 40)

        cosine_sim = metrics['cosine_sim_overall']
        if cosine_sim > 0.9999:
            lines.append("  [EXCELLENT] Outputs are nearly identical (cosine sim > 0.9999)")
        elif cosine_sim > 0.999:
            lines.append("  [VERY GOOD] Outputs are very similar (cosine sim > 0.999)")
        elif cosine_sim > 0.99:
            lines.append("  [GOOD] Outputs are similar (cosine sim > 0.99)")
        elif cosine_sim > 0.95:
            lines.append("  [MODERATE] Some differences present (cosine sim > 0.95)")
        else:
            lines.append("  [CAUTION] Significant differences (cosine sim <= 0.95)")

        rel_err = metrics['mean_relative_error']
        if rel_err < 0.1:
            lines.append(f"  Relative error is negligible ({rel_err:.4f}%)")
        elif rel_err < 1.0:
            lines.append(f"  Relative error is small ({rel_err:.4f}%)")
        elif rel_err < 5.0:
            lines.append(f"  Relative error is moderate ({rel_err:.4f}%)")
        else:
            lines.append(f"  Relative error is significant ({rel_err:.4f}%)")

    lines.append("")
    lines.append("=" * 70)

    # Print or save
    report = "\n".join(lines)
    print(report)

    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {output_file}")


def compare_coordinates(data1, data2, threshold=0.01):
    """Check if coordinates match between the two files."""
    coord1 = data1.get('coords')
    coord2 = data2.get('coords')

    if coord1 is None or coord2 is None:
        return None, "Coordinates not available", None

    if coord1.shape != coord2.shape:
        return False, f"Coordinate shapes differ: {coord1.shape} vs {coord2.shape}", None

    coord_diff = torch.abs(coord1 - coord2).max().item()
    if coord_diff > 1e-6:
        return False, f"Coordinates differ (max diff: {coord_diff:.6e})", (coord1, coord2)

    return True, "Coordinates match", None


def analyze_coordinate_differences(coord1, coord2, threshold=0.01, max_examples=10):
    """Analyze and report coordinate differences in detail."""
    lines = []

    # Per-point L2 distance
    diff = coord1 - coord2
    l2_dist = torch.sqrt((diff ** 2).sum(dim=1))

    # Per-axis absolute difference
    abs_diff = torch.abs(diff)

    lines.append("COORDINATE DIFFERENCE ANALYSIS:")
    lines.append("-" * 40)
    lines.append(f"Total points: {coord1.shape[0]}")
    lines.append("")

    # Overall statistics
    lines.append("Per-axis statistics (absolute difference):")
    axis_names = ['X', 'Y', 'Z']
    for i, name in enumerate(axis_names):
        axis_diff = abs_diff[:, i]
        lines.append(f"  {name}-axis: mean={axis_diff.mean().item():.6e}, "
                    f"max={axis_diff.max().item():.6e}, "
                    f"std={axis_diff.std().item():.6e}")
    lines.append("")

    lines.append("L2 distance statistics:")
    lines.append(f"  Mean: {l2_dist.mean().item():.6e}")
    lines.append(f"  Max:  {l2_dist.max().item():.6e}")
    lines.append(f"  Std:  {l2_dist.std().item():.6e}")
    lines.append("")

    # Find points exceeding threshold
    exceed_mask = l2_dist > threshold
    n_exceed = exceed_mask.sum().item()
    pct_exceed = 100.0 * n_exceed / coord1.shape[0]

    lines.append(f"Points with L2 diff > {threshold}:")
    lines.append(f"  Count: {n_exceed} ({pct_exceed:.2f}%)")
    lines.append("")

    if n_exceed > 0:
        # Get indices of points that exceed threshold, sorted by distance
        exceed_indices = torch.where(exceed_mask)[0]
        exceed_distances = l2_dist[exceed_indices]
        sorted_order = torch.argsort(exceed_distances, descending=True)

        n_show = min(max_examples, n_exceed)
        lines.append(f"Top {n_show} largest coordinate differences:")
        lines.append(f"  {'Idx':>6} | {'Coord1 (x,y,z)':^30} | {'Coord2 (x,y,z)':^30} | {'L2 Dist':>10}")
        lines.append(f"  {'-'*6}-+-{'-'*30}-+-{'-'*30}-+-{'-'*10}")

        for i in range(n_show):
            idx = exceed_indices[sorted_order[i]].item()
            c1 = coord1[idx]
            c2 = coord2[idx]
            dist = l2_dist[idx].item()
            c1_str = f"({c1[0].item():9.3f}, {c1[1].item():9.3f}, {c1[2].item():9.3f})"
            c2_str = f"({c2[0].item():9.3f}, {c2[1].item():9.3f}, {c2[2].item():9.3f})"
            lines.append(f"  {idx:>6} | {c1_str:^30} | {c2_str:^30} | {dist:>10.6f}")
        lines.append("")

        # Check if differences follow a pattern
        lines.append("Pattern analysis:")

        # Check if it's mostly one axis
        axis_contributions = abs_diff[exceed_mask].mean(dim=0)
        total_contribution = axis_contributions.sum().item()
        if total_contribution > 0:
            for i, name in enumerate(axis_names):
                pct = 100.0 * axis_contributions[i].item() / total_contribution
                lines.append(f"  {name}-axis contributes {pct:.1f}% of differences")

        # Check if there are clusters of differing points
        if n_exceed > 1:
            exceed_coords1 = coord1[exceed_mask]
            coord_range = exceed_coords1.max(dim=0).values - exceed_coords1.min(dim=0).values
            lines.append(f"  Differing points span: X={coord_range[0].item():.2f}, "
                        f"Y={coord_range[1].item():.2f}, Z={coord_range[2].item():.2f}")
    else:
        lines.append(f"All coordinates match within threshold {threshold}")

    lines.append("")
    return "\n".join(lines)


def main():
    args = parse_args()

    # Check files exist
    if not os.path.exists(args.file1):
        raise FileNotFoundError(f"File 1 not found: {args.file1}")
    if not os.path.exists(args.file2):
        raise FileNotFoundError(f"File 2 not found: {args.file2}")

    # Load data
    print(f"Loading file 1: {args.file1}")
    data1 = torch.load(args.file1, map_location="cpu")

    print(f"Loading file 2: {args.file2}")
    data2 = torch.load(args.file2, map_location="cpu")

    # Check coordinates
    coord_match, coord_msg, coord_data = compare_coordinates(data1, data2, threshold=args.tolerance)
    if coord_match is False:
        print(f"WARNING: {coord_msg}")
        print("The encoder outputs may not be directly comparable if coordinates differ.")
        # Print detailed coordinate analysis
        if coord_data is not None:
            coord1, coord2 = coord_data
            print("")
            coord_analysis = analyze_coordinate_differences(coord1, coord2, threshold=args.tolerance)
            print(coord_analysis)
    elif coord_match is True:
        print(f"OK: {coord_msg}")

    # Get features
    feat1 = data1['features']
    feat2 = data2['features']

    # Compute metrics
    print("\nComputing comparison metrics...")
    metrics = compute_metrics(feat1, feat2, tolerance=args.tolerance)

    # Print report
    print("")
    print_report(data1, data2, metrics, args.output)


if __name__ == "__main__":
    main()
