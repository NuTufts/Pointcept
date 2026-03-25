#!/usr/bin/env python
"""
Validate shower origin reco HDF5 files for training.

Checks each merged reco+truth HDF5 file for:
  1. File integrity: can be opened, required datasets present with correct shapes
  2. Spacepoint count bounds: rejects events with too few or too many total points
  3. Minimum trainable fragments: at least 1 shower fragment with >= threshold points
  4. Data integrity: no NaN/Inf, valid index ranges, consistent shapes

Produces:
  - Bad file list (for exclusion)
  - Validated good file list (for train/val splitting)
  - Histograms: spacepoint counts, fragments per event, points per fragment,
    fragment type distribution

Usage:
    python validate_showerorigin_reco_h5.py h5list_showeroriginreco_ncpi0filter.txt
    python validate_showerorigin_reco_h5.py h5list.txt --workers 8 --output badfiles.txt
    python validate_showerorigin_reco_h5.py h5list.txt --min-points 4096 --max-points 200000
    python validate_showerorigin_reco_h5.py h5list.txt --min-fragment-points 50
"""

import argparse
import h5py
import numpy as np
import os
import sys
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


# ── Global defaults (overridden via pool initializer) ────────────────────────
MIN_TOTAL_POINTS = 4096
MAX_TOTAL_POINTS = 300000
MIN_FRAGMENT_POINTS = 50

# Required datasets in triplet_data (key -> expected ndim)
REQUIRED_TRIPLET = {
    'pos': 2,        # (N, 3)
    'pixval': 2,     # (N, 3)
    'uwire': 1,      # (N,)
    'vwire': 1,      # (N,)
    'ywire': 1,      # (N,)
    'hasmatch': 1,   # (N,)
}

# Required datasets in shower_fragments (key -> expected ndim)
REQUIRED_FRAGMENTS = {
    'trackid': 1,              # (F,)
    'pid': 1,                  # (F,)
    'istrunk': 1,              # (F,)
    'type': 1,                 # (F,)
    'startpt': 2,              # (F, 3)
    'originpt': 2,             # (F, 3)
    'pret0shiftedoriginpt': 2, # (F, 4)
    'pointindices_flat': 1,    # (T,)
    'pointindices_counts': 1,  # (F,)
}


def init_worker(min_pts, max_pts, min_frag_pts):
    """Initialize worker with thresholds."""
    global MIN_TOTAL_POINTS, MAX_TOTAL_POINTS, MIN_FRAGMENT_POINTS
    MIN_TOTAL_POINTS = min_pts
    MAX_TOTAL_POINTS = max_pts
    MIN_FRAGMENT_POINTS = min_frag_pts


def validate_file(args):
    """
    Validate a single HDF5 file.

    Args:
        args: tuple of (line_number, file_path)

    Returns:
        tuple: (line_num, file_path, error_msg, stats_dict) or
               (line_num, file_path, None, stats_dict) if valid.
        stats_dict contains numeric info for histograms even on failure.
    """
    line_num, file_path = args
    stats = {
        'n_total_points': 0,
        'n_fragments': 0,
        'fragment_point_counts': [],
        'fragment_types': [],
    }

    # ── File existence & size ────────────────────────────────────────────
    if not os.path.exists(file_path):
        return (line_num, file_path, "File does not exist", stats)

    if os.path.getsize(file_path) == 0:
        return (line_num, file_path, "File is empty (0 bytes)", stats)

    try:
        with h5py.File(file_path, 'r') as f:

            # ── entry_0 group ────────────────────────────────────────────
            if 'entry_0' not in f:
                return (line_num, file_path, "Missing group: entry_0", stats)

            entry = f['entry_0']

            # ── triplet_data checks ──────────────────────────────────────
            if 'triplet_data' not in entry:
                return (line_num, file_path,
                        "Missing group: entry_0/triplet_data", stats)

            triplet = entry['triplet_data']

            for key, expected_ndim in REQUIRED_TRIPLET.items():
                dset_path = f"entry_0/triplet_data/{key}"
                if key not in triplet:
                    return (line_num, file_path,
                            f"Missing dataset: {dset_path}", stats)
                shape = triplet[key].shape
                if len(shape) != expected_ndim:
                    return (line_num, file_path,
                            f"Wrong ndim for {dset_path}: "
                            f"expected {expected_ndim}, got {len(shape)}",
                            stats)

            # Total spacepoints (all points, no hasmatch filtering)
            n_points = triplet['pos'].shape[0]
            stats['n_total_points'] = n_points

            # Shape consistency: all 1-D triplet arrays must have length N
            for key, expected_ndim in REQUIRED_TRIPLET.items():
                if triplet[key].shape[0] != n_points:
                    return (line_num, file_path,
                            f"Shape mismatch: triplet_data/{key} has "
                            f"{triplet[key].shape[0]} rows, expected {n_points}",
                            stats)

            # pos must be (N, 3)
            if triplet['pos'].shape[1] != 3:
                return (line_num, file_path,
                        f"pos shape {triplet['pos'].shape}, expected (N, 3)",
                        stats)

            # pixval must be (N, 3)
            if triplet['pixval'].shape[1] != 3:
                return (line_num, file_path,
                        f"pixval shape {triplet['pixval'].shape}, "
                        f"expected (N, 3)", stats)

            # ── Spacepoint count bounds ──────────────────────────────────
            if n_points < MIN_TOTAL_POINTS:
                return (line_num, file_path,
                        f"Too few spacepoints: {n_points} < {MIN_TOTAL_POINTS}",
                        stats)

            if n_points > MAX_TOTAL_POINTS:
                return (line_num, file_path,
                        f"Too many spacepoints: {n_points} > {MAX_TOTAL_POINTS}",
                        stats)

            # ── NaN/Inf in triplet data ──────────────────────────────────
            for key in ('pos', 'pixval'):
                arr = triplet[key][:]
                if not np.all(np.isfinite(arr)):
                    n_nan = np.isnan(arr).sum()
                    n_inf = np.isinf(arr).sum()
                    return (line_num, file_path,
                            f"NaN/Inf in triplet_data/{key}: "
                            f"nan={n_nan}, inf={n_inf}", stats)

            # ── shower_fragments checks ──────────────────────────────────
            if 'shower_fragments' not in entry:
                return (line_num, file_path,
                        "Missing group: entry_0/shower_fragments", stats)

            sf = entry['shower_fragments']

            # num_fragments attribute
            if 'num_fragments' not in sf.attrs:
                return (line_num, file_path,
                        "Missing attribute: shower_fragments/@num_fragments",
                        stats)

            num_frags = int(sf.attrs['num_fragments'])
            stats['n_fragments'] = num_frags

            if num_frags == 0:
                return (line_num, file_path,
                        "No shower fragments (num_fragments=0)", stats)

            # Required fragment datasets
            for key, expected_ndim in REQUIRED_FRAGMENTS.items():
                dset_path = f"entry_0/shower_fragments/{key}"
                if key not in sf:
                    return (line_num, file_path,
                            f"Missing dataset: {dset_path}", stats)
                shape = sf[key].shape
                if len(shape) != expected_ndim:
                    return (line_num, file_path,
                            f"Wrong ndim for {dset_path}: "
                            f"expected {expected_ndim}, got {len(shape)}",
                            stats)

            # Per-fragment array lengths must equal num_fragments
            per_frag_keys = [
                'trackid', 'pid', 'istrunk', 'type',
                'startpt', 'originpt', 'pret0shiftedoriginpt',
                'pointindices_counts',
            ]
            for key in per_frag_keys:
                if sf[key].shape[0] != num_frags:
                    return (line_num, file_path,
                            f"Fragment array length mismatch: {key} has "
                            f"{sf[key].shape[0]} entries, expected {num_frags}",
                            stats)

            # Shape checks for 2-D fragment arrays
            if sf['startpt'].shape[1] != 3:
                return (line_num, file_path,
                        f"startpt shape {sf['startpt'].shape}, "
                        f"expected (F, 3)", stats)
            if sf['originpt'].shape[1] != 3:
                return (line_num, file_path,
                        f"originpt shape {sf['originpt'].shape}, "
                        f"expected (F, 3)", stats)
            if sf['pret0shiftedoriginpt'].shape[1] != 4:
                return (line_num, file_path,
                        f"pret0shiftedoriginpt shape "
                        f"{sf['pret0shiftedoriginpt'].shape}, "
                        f"expected (F, 4)", stats)

            # ── pointindices consistency ──────────────────────────────────
            index_counts = sf['pointindices_counts'][:]
            flat_indices = sf['pointindices_flat'][:]

            if index_counts.sum() != len(flat_indices):
                return (line_num, file_path,
                        f"pointindices mismatch: sum(counts)={index_counts.sum()}"
                        f" != len(flat)={len(flat_indices)}", stats)

            # All indices must be in [0, n_points)
            if len(flat_indices) > 0:
                idx_min = flat_indices.min()
                idx_max = flat_indices.max()
                if idx_min < 0:
                    return (line_num, file_path,
                            f"Negative point index: min={idx_min}", stats)
                if idx_max >= n_points:
                    return (line_num, file_path,
                            f"Point index out of range: max={idx_max} >= "
                            f"n_points={n_points}", stats)

            # ── NaN/Inf in fragment coordinate arrays ────────────────────
            for key in ('startpt', 'originpt'):
                arr = sf[key][:]
                if not np.all(np.isfinite(arr)):
                    n_nan = np.isnan(arr).sum()
                    n_inf = np.isinf(arr).sum()
                    return (line_num, file_path,
                            f"NaN/Inf in shower_fragments/{key}: "
                            f"nan={n_nan}, inf={n_inf}", stats)

            # ── Collect per-fragment stats ────────────────────────────────
            frag_types = sf['type'][:].tolist()
            frag_pt_counts = index_counts.tolist()
            stats['fragment_point_counts'] = frag_pt_counts
            stats['fragment_types'] = frag_types

            # ── Minimum trainable fragment check ─────────────────────────
            # Need at least 1 fragment with >= MIN_FRAGMENT_POINTS points.
            # Fragment types: 0=nu-inside, 1=outside, 2=cosmic-inside,
            #                 -1=no truth match (reco ghost label), 3=ghost, 4=true_track
            # All types are acceptable for reco training.
            has_trainable = False
            for count in frag_pt_counts:
                if count >= MIN_FRAGMENT_POINTS:
                    has_trainable = True
                    break

            if not has_trainable:
                max_frag = max(frag_pt_counts) if frag_pt_counts else 0
                return (line_num, file_path,
                        f"No trainable fragment: largest has {max_frag} points "
                        f"< {MIN_FRAGMENT_POINTS}", stats)

            # ── nu_vertex_is_visible scalar ──────────────────────────────
            if 'nu_vertex_is_visible' not in sf:
                return (line_num, file_path,
                        "Missing dataset: shower_fragments/nu_vertex_is_visible",
                        stats)

    except OSError as e:
        return (line_num, file_path, f"OSError: {str(e)}", stats)
    except Exception as e:
        return (line_num, file_path, f"{type(e).__name__}: {str(e)}", stats)

    # File is valid
    return (line_num, file_path, None, stats)


def make_histograms(all_stats, output_prefix):
    """Generate and save histogram plots and raw data."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        has_matplotlib = True
    except ImportError:
        has_matplotlib = False
        print("Warning: matplotlib not available, saving raw numpy only")

    # ── Gather arrays ────────────────────────────────────────────────────
    total_points = np.array([s['n_total_points'] for s in all_stats
                             if s['n_total_points'] > 0])
    n_frags = np.array([s['n_fragments'] for s in all_stats])

    all_frag_pt_counts = []
    all_frag_types = []
    for s in all_stats:
        all_frag_pt_counts.extend(s['fragment_point_counts'])
        all_frag_types.extend(s['fragment_types'])
    frag_pt_counts = np.array(all_frag_pt_counts)
    frag_types = np.array(all_frag_types)

    # ── Save raw numpy arrays ────────────────────────────────────────────
    np.save(f"{output_prefix}_spacepoint_counts.npy", total_points)
    np.save(f"{output_prefix}_fragments_per_event.npy", n_frags)
    np.save(f"{output_prefix}_fragment_point_counts.npy", frag_pt_counts)
    np.save(f"{output_prefix}_fragment_types.npy", frag_types)
    print(f"  Raw numpy arrays saved with prefix: {output_prefix}_*.npy")

    # ── Print summary statistics ─────────────────────────────────────────
    print()
    print("Dataset statistics:")
    print(f"  Spacepoints per event:")
    if len(total_points) > 0:
        print(f"    min={total_points.min()}, max={total_points.max()}, "
              f"mean={total_points.mean():.0f}, median={np.median(total_points):.0f}, "
              f"std={total_points.std():.0f}")
        for pct in [1, 5, 25, 50, 75, 95, 99]:
            print(f"    {pct}th percentile: {np.percentile(total_points, pct):.0f}")

    print(f"  Fragments per event:")
    if len(n_frags) > 0:
        print(f"    min={n_frags.min()}, max={n_frags.max()}, "
              f"mean={n_frags.mean():.1f}, median={np.median(n_frags):.0f}")

    print(f"  Points per fragment:")
    if len(frag_pt_counts) > 0:
        print(f"    min={frag_pt_counts.min()}, max={frag_pt_counts.max()}, "
              f"mean={frag_pt_counts.mean():.1f}, median={np.median(frag_pt_counts):.0f}")
        for pct in [1, 5, 25, 50, 75, 95, 99]:
            print(f"    {pct}th percentile: {np.percentile(frag_pt_counts, pct):.0f}")

    print(f"  Fragment type distribution:")
    if len(frag_types) > 0:
        type_labels = {
            -1: "no-truth-match",
            0: "nu-inside",
            1: "outside",
            2: "cosmic-inside",
            3: "ghost",
            4: "true_track",
        }
        for t in sorted(np.unique(frag_types)):
            count = (frag_types == t).sum()
            label = type_labels.get(int(t), f"unknown({int(t)})")
            print(f"    type {int(t):2d} ({label}): {count} "
                  f"({100.0 * count / len(frag_types):.1f}%)")

    if not has_matplotlib:
        return

    # ── Plot histograms ──────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Shower Origin Reco Dataset Validation", fontsize=14)

    # (a) Spacepoint counts per event (log y-axis to see tails)
    ax = axes[0, 0]
    if len(total_points) > 0:
        ax.hist(total_points, bins=100, edgecolor='black', linewidth=0.5)
        ax.axvline(MIN_TOTAL_POINTS, color='red', linestyle='--',
                   label=f'min={MIN_TOTAL_POINTS}')
        ax.axvline(MAX_TOTAL_POINTS, color='red', linestyle='--',
                   label=f'max={MAX_TOTAL_POINTS}')
        ax.set_yscale('log')
        ax.set_xlabel('Total spacepoints per event')
        ax.set_ylabel('Count')
        ax.set_title(f'Spacepoints per Event (N={len(total_points)})')
        ax.legend()

    # (b) Fragments per event
    ax = axes[0, 1]
    if len(n_frags) > 0:
        max_bin = min(int(n_frags.max()) + 1, 200)
        ax.hist(n_frags, bins=np.arange(0, max_bin + 1) - 0.5,
                edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Number of fragments per event')
        ax.set_ylabel('Count')
        ax.set_title(f'Fragments per Event (N={len(n_frags)})')

    # (c) Points per fragment — stacked: ghost vs non-ghost, fine binning
    ax = axes[1, 0]
    if len(frag_pt_counts) > 0:
        # Fine linear bins, capped at 500 points to focus on the bimodal region
        lin_bins = np.linspace(0, 500, 500)
        # Split into ghost (type 3) and non-ghost
        ghost_mask_hist = (frag_types == 3)
        ghost_counts = frag_pt_counts[ghost_mask_hist]
        nonghost_counts = frag_pt_counts[~ghost_mask_hist]
        ax.hist([nonghost_counts, ghost_counts], bins=lin_bins,
                stacked=True, edgecolor='none',
                color=['steelblue', 'salmon'],
                label=[f'non-ghost ({len(nonghost_counts)})',
                       f'ghost ({len(ghost_counts)})'])
        ax.axvline(MIN_FRAGMENT_POINTS, color='red', linestyle='--',
                   linewidth=1.5, label=f'min_frag={MIN_FRAGMENT_POINTS}')
        ax.set_xlabel('Points per fragment')
        ax.set_ylabel('Count')
        ax.set_title(f'Points per Fragment (N={len(frag_pt_counts)})')
        ax.legend(fontsize=8)

    # (d) Fragment type distribution
    ax = axes[1, 1]
    if len(frag_types) > 0:
        type_labels = {
            -1: "no-match", 0: "nu-in", 1: "outside",
            2: "cosmic", 3: "ghost", 4: "track",
        }
        unique_types = sorted(np.unique(frag_types).astype(int))
        counts = [(frag_types == t).sum() for t in unique_types]
        labels = [type_labels.get(t, str(t)) for t in unique_types]
        bars = ax.bar(range(len(unique_types)), counts, edgecolor='black',
                      linewidth=0.5)
        ax.set_xticks(range(len(unique_types)))
        ax.set_xticklabels(labels, rotation=30, ha='right')
        ax.set_ylabel('Count')
        ax.set_title(f'Fragment Type Distribution (N={len(frag_types)})')
        # Add count labels on bars
        for bar, c in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                    f'{c}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    png_path = f"{output_prefix}_histograms.png"
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"  Histograms saved to: {png_path}")


def main():
    global MIN_TOTAL_POINTS, MAX_TOTAL_POINTS, MIN_FRAGMENT_POINTS

    parser = argparse.ArgumentParser(
        description='Validate shower origin reco HDF5 files for training')
    parser.add_argument('file_list',
                        help='Text file containing paths to HDF5 files')
    parser.add_argument('--workers', '-w', type=int,
                        default=min(8, cpu_count()),
                        help='Number of parallel workers (default: 8 or '
                             'cpu_count)')
    parser.add_argument('--output', '-o',
                        help='Output file to write bad file paths')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print each bad file as it is found')
    parser.add_argument('--min-points', type=int, default=4096,
                        help='Minimum total spacepoints per event '
                             '(default: 4096)')
    parser.add_argument('--max-points', type=int, default=300000,
                        help='Maximum total spacepoints per event '
                             '(default: 300000)')
    parser.add_argument('--min-fragment-points', type=int, default=50,
                        help='Minimum points in at least one fragment for '
                             'the event to be trainable (default: 50)')
    parser.add_argument('--histogram-prefix',
                        help='Prefix for histogram output files. '
                             'Defaults to <file_list_basename>_validation')
    parser.add_argument('-n', '--max-files', type=int, default=-1,
                        help='Maximum number of files to process '
                             '(-1 = all, default: -1)')
    args = parser.parse_args()

    MIN_TOTAL_POINTS = args.min_points
    MAX_TOTAL_POINTS = args.max_points
    MIN_FRAGMENT_POINTS = args.min_fragment_points

    # ── Read file list ───────────────────────────────────────────────────
    print(f"Reading file list from: {args.file_list}")
    with open(args.file_list, 'r') as f:
        files = []
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if line and not line.startswith('#'):
                files.append((line_num, line))

    if args.max_files > 0:
        files = files[:args.max_files]

    print(f"Found {len(files)} files to validate")
    print(f"Using {args.workers} workers")
    print(f"Spacepoint bounds: [{args.min_points}, {args.max_points}]")
    print(f"Min fragment points: {args.min_fragment_points}")
    print()

    # ── Validate files in parallel ───────────────────────────────────────
    with Pool(args.workers,
              initializer=init_worker,
              initargs=(args.min_points, args.max_points,
                        args.min_fragment_points)) as pool:
        results = list(tqdm(
            pool.imap(validate_file, files),
            total=len(files),
            desc="Validating",
        ))

    # ── Separate valid/bad and collect stats ─────────────────────────────
    bad_files = []
    all_stats = []
    valid_stats = []

    for result in results:
        line_num, path, error, stats = result
        all_stats.append(stats)
        if error is not None:
            bad_files.append((line_num, path, error))
            if args.verbose:
                print(f"  [{line_num}] {path}: {error}")
        else:
            valid_stats.append(stats)

    # ── Categorize errors ────────────────────────────────────────────────
    error_categories = {}
    for line_num, path, error in bad_files:
        if "Too few spacepoints" in error:
            category = "Too few spacepoints"
        elif "Too many spacepoints" in error:
            category = "Too many spacepoints"
        elif "No trainable fragment" in error:
            category = "No trainable fragment"
        elif "No shower fragments" in error:
            category = "No shower fragments"
        elif "Missing dataset" in error or "Missing group" in error:
            category = "Missing dataset/group"
        elif "Missing attribute" in error:
            category = "Missing attribute"
        elif "NaN/Inf" in error:
            category = "NaN/Inf values"
        elif "pointindices" in error.lower():
            category = "Point index error"
        elif "Shape mismatch" in error or "Wrong ndim" in error:
            category = "Shape error"
        elif "File does not exist" in error:
            category = "File does not exist"
        elif "File is empty" in error:
            category = "File is empty"
        elif "OSError" in error:
            category = "OSError (corrupted/unreadable)"
        else:
            category = "Other"
        error_categories[category] = error_categories.get(category, 0) + 1

    # ── Report ───────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("Validation complete!")
    print(f"  Total files:  {len(files)}")
    print(f"  Valid files:  {len(files) - len(bad_files)}")
    print(f"  Bad files:    {len(bad_files)}")
    if error_categories:
        print()
        print("  Error breakdown:")
        for category, count in sorted(error_categories.items(),
                                      key=lambda x: -x[1]):
            print(f"    {category}: {count}")
    print("=" * 70)

    # ── Histograms (from ALL files, including bad ones with stats) ───────
    hist_prefix = args.histogram_prefix
    if hist_prefix is None:
        base = os.path.splitext(os.path.basename(args.file_list))[0]
        hist_prefix = os.path.join(os.path.dirname(args.file_list) or '.',
                                   f"{base}_validation")

    print()
    print("Generating histograms (all files with loadable stats)...")
    make_histograms(all_stats, hist_prefix)

    print()
    print("Generating histograms (valid files only)...")
    make_histograms(valid_stats, hist_prefix + "_valid")

    # ── Write bad files ──────────────────────────────────────────────────
    if bad_files:
        print()
        print(f"Bad files ({len(bad_files)}):")
        for line_num, path, error in sorted(bad_files):
            print(f"  Line {line_num}: {path}")
            print(f"    Error: {error}")

        if args.output:
            with open(args.output, 'w') as f:
                for line_num, path, error in sorted(bad_files):
                    f.write(f"{path}\n")
            print(f"\nBad file paths written to: {args.output}")

    # ── Write validated good file list ───────────────────────────────────
    good_output = args.file_list.replace('.txt', '_validated.txt')
    if good_output == args.file_list:
        good_output = args.file_list + '.validated'

    bad_paths = set(path for _, path, _ in bad_files)
    n_good = 0
    with open(good_output, 'w') as f:
        for line_num, path in files:
            if path not in bad_paths:
                f.write(f"{path}\n")
                n_good += 1

    print(f"\nValidated file list ({n_good} files) written to: {good_output}")

    return 1 if bad_files else 0


if __name__ == '__main__':
    sys.exit(main())
