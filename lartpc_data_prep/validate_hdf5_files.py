#!/usr/bin/env python
"""
Validate HDF5 files for LArTPC training.

Checks that each file can be opened and contains the required datasets.
Also checks for minimum number of true points (hasmatch==1) to ensure
events have enough signal for training.

Reports bad files that should be removed from the training list.

Usage:
    python validate_hdf5_files.py train_split_shuffled.txt
    python validate_hdf5_files.py train_split_shuffled.txt --workers 8
    python validate_hdf5_files.py train_split_shuffled.txt --output badfiles.txt
    python validate_hdf5_files.py train_split_shuffled.txt --min-true-points 100
"""

import argparse
import h5py
import numpy as np
import os
import sys
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# Global variable for minimum true points (set via pool initializer)
MIN_TRUE_POINTS = 100


def init_worker(min_true_points):
    """Initialize worker process with the minimum true points threshold."""
    global MIN_TRUE_POINTS
    MIN_TRUE_POINTS = min_true_points

# Required datasets in each HDF5 file (based on lartpc.py)
REQUIRED_DATASETS = [
    '/entry_0/triplet_data/pos',
    '/entry_0/triplet_data/pid',
    '/entry_0/triplet_data/pixval',
    '/entry_0/triplet_data/uwire',
    '/entry_0/triplet_data/vwire',
    '/entry_0/triplet_data/ywire',
    '/entry_0/triplet_data/trackid',
    '/entry_0/triplet_data/hasmatch',
    '/entry_0/mckeypoints/pos',
    '/entry_0/mckeypoints/kptype'
]


def validate_file(args):
    """
    Validate a single HDF5 file.

    Args:
        args: tuple of (line_number, file_path)

    Returns:
        tuple: (line_number, file_path, error_message) or None if valid
    """
    line_num, file_path = args

    # Check if file exists
    if not os.path.exists(file_path):
        return (line_num, file_path, "File does not exist")

    # Check if file is empty
    if os.path.getsize(file_path) == 0:
        return (line_num, file_path, "File is empty (0 bytes)")

    # Try to open and validate HDF5 structure
    try:
        with h5py.File(file_path, 'r') as f:
            # Check for required datasets
            for dataset in REQUIRED_DATASETS:
                if dataset not in f:
                    return (line_num, file_path, f"Missing dataset: {dataset}")
                # Try to read shape to ensure data is accessible
                _ = f[dataset].shape

            # Check minimum true points (hasmatch == 1)
            # This is critical for true_points_only=True training mode
            if MIN_TRUE_POINTS > 0:
                hasmatch = f['/entry_0/triplet_data/hasmatch'][:]
                true_point_count = np.sum(hasmatch == 1)
                if true_point_count < MIN_TRUE_POINTS:
                    return (line_num, file_path,
                            f"Insufficient true points: {true_point_count} < {MIN_TRUE_POINTS} "
                            f"(would cause empty array error with true_points_only=True)")

    except OSError as e:
        return (line_num, file_path, f"OSError: {str(e)}")
    except Exception as e:
        return (line_num, file_path, f"{type(e).__name__}: {str(e)}")

    return None  # File is valid


def main():
    global MIN_TRUE_POINTS

    parser = argparse.ArgumentParser(description='Validate HDF5 files for LArTPC training')
    parser.add_argument('file_list', help='Text file containing paths to HDF5 files')
    parser.add_argument('--workers', '-w', type=int, default=min(8, cpu_count()),
                        help='Number of parallel workers (default: 8 or cpu_count)')
    parser.add_argument('--output', '-o', help='Output file to write bad file paths')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print each bad file as it is found')
    parser.add_argument('--min-true-points', '-m', type=int, default=100,
                        help='Minimum number of true points (hasmatch==1) required. '
                             'Events with fewer points will fail validation. '
                             'Set to 0 to disable this check. (default: 100)')
    args = parser.parse_args()

    # Set global threshold for worker processes
    MIN_TRUE_POINTS = args.min_true_points

    # Read file list
    print(f"Reading file list from: {args.file_list}")
    with open(args.file_list, 'r') as f:
        files = []
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if line and not line.startswith('#'):
                files.append((line_num, line))

    print(f"Found {len(files)} files to validate")
    print(f"Using {args.workers} workers")
    print(f"Minimum true points required: {args.min_true_points}")
    print()

    # Validate files in parallel
    bad_files = []
    with Pool(args.workers, initializer=init_worker, initargs=(args.min_true_points,)) as pool:
        results = list(tqdm(
            pool.imap(validate_file, files),
            total=len(files),
            desc="Validating"
        ))

    # Collect bad files
    for result in results:
        if result is not None:
            bad_files.append(result)
            if args.verbose:
                line_num, path, error = result
                print(f"  [{line_num}] {path}: {error}")

    # Categorize errors
    error_categories = {}
    for line_num, path, error in bad_files:
        # Extract error category (first part before ":")
        if "Insufficient true points" in error:
            category = "Insufficient true points"
        elif "Missing dataset" in error:
            category = "Missing dataset"
        elif "File does not exist" in error:
            category = "File does not exist"
        elif "File is empty" in error:
            category = "File is empty"
        elif "OSError" in error:
            category = "OSError (corrupted/unreadable)"
        else:
            category = "Other"
        error_categories[category] = error_categories.get(category, 0) + 1

    # Report results
    print()
    print("=" * 60)
    print(f"Validation complete!")
    print(f"  Total files: {len(files)}")
    print(f"  Valid files: {len(files) - len(bad_files)}")
    print(f"  Bad files:   {len(bad_files)}")
    if error_categories:
        print()
        print("Error breakdown:")
        for category, count in sorted(error_categories.items(), key=lambda x: -x[1]):
            print(f"  {category}: {count}")
    print("=" * 60)

    if bad_files:
        print()
        print("Bad files:")
        for line_num, path, error in sorted(bad_files):
            print(f"  Line {line_num}: {path}")
            print(f"    Error: {error}")

        # Write output file if requested
        if args.output:
            with open(args.output, 'w') as f:
                for line_num, path, error in sorted(bad_files):
                    f.write(f"{path}\n")
            print()
            print(f"Bad file paths written to: {args.output}")

        # Also write a filtered good files list
        good_output = args.file_list.replace('.txt', '_validated.txt')
        if good_output == args.file_list:
            good_output = args.file_list + '.validated'

        bad_paths = set(path for _, path, _ in bad_files)
        with open(good_output, 'w') as f:
            for line_num, path in files:
                if path not in bad_paths:
                    f.write(f"{path}\n")
        print(f"Validated file list written to: {good_output}")

        return 1  # Exit with error code

    return 0


if __name__ == '__main__':
    sys.exit(main())
