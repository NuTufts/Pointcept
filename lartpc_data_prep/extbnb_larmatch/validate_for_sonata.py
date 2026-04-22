#!/usr/bin/env python
"""
Validate extbnb larmatch HDF5 files for Sonata pre-training.

Checks each file for the arrays needed by Sonata training with LArTPC data:
  - pos (N, 3): 3D spacepoint coordinates
  - pixval (N, 3): ADC values from u, v, y wire planes
  - larmatch_score (N,): ghost score from LArMatch [0=ghost, 1=true]

Validation checks:
  - File can be opened and has expected HDF5 structure
  - Required arrays exist with correct shapes
  - No NaN or Inf values in any array
  - Array lengths are consistent (all N match)
  - larmatch_score is in valid range [0, 1]
  - Event has a minimum number of points

Uses multiprocessing for parallel validation of large file lists.

Usage:
    python validate_for_sonata.py filelist.txt
    python validate_for_sonata.py filelist.txt --workers 16
    python validate_for_sonata.py filelist.txt --output bad_files.txt --min-points 50
    python validate_for_sonata.py --file /path/to/single_file.h5 --verbose
"""

import argparse
import h5py
import numpy as np
import os
import sys
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# Global config set via pool initializer
MIN_POINTS = 50

REQUIRED_ARRAYS = {
    "pos": {"ndim": 2, "shape_1": 3},
    "pixval": {"ndim": 2, "shape_1": 3},
    "larmatch_score": {"ndim": 1},
}

# Additional arrays to check if present (wire coords used by Sonata augmentations)
OPTIONAL_ARRAYS = ["uwire", "vwire", "ywire", "tick"]


def init_worker(min_points):
    global MIN_POINTS
    MIN_POINTS = min_points


def validate_file(args):
    """
    Validate a single HDF5 file for Sonata training.

    Args:
        args: tuple of (index, file_path)

    Returns:
        tuple: (index, file_path, error_message) or None if valid
    """
    idx, file_path = args

    if not os.path.exists(file_path):
        return (idx, file_path, "File does not exist")

    if os.path.getsize(file_path) == 0:
        return (idx, file_path, "File is empty (0 bytes)")

    try:
        with h5py.File(file_path, "r") as f:
            if "entry_0/triplet_data" not in f:
                return (idx, file_path, "Missing entry_0/triplet_data group")

            grp = f["entry_0/triplet_data"]

            # Check required arrays exist
            for name in REQUIRED_ARRAYS:
                if name not in grp:
                    return (idx, file_path, f"Missing required array: {name}")

            # Load arrays
            pos = np.array(grp["pos"])
            pixval = np.array(grp["pixval"])
            lm_score = np.array(grp["larmatch_score"])

            n = len(pos)

            # Minimum points check
            if n < MIN_POINTS:
                return (idx, file_path,
                        f"Too few points: {n} < {MIN_POINTS}")

            # Shape checks
            if pos.shape != (n, 3):
                return (idx, file_path,
                        f"pos shape {pos.shape}, expected ({n}, 3)")
            if pixval.shape != (n, 3):
                return (idx, file_path,
                        f"pixval shape {pixval.shape}, expected ({n}, 3)")
            if lm_score.shape != (n,):
                return (idx, file_path,
                        f"larmatch_score shape {lm_score.shape}, expected ({n},)")

            # NaN checks
            if np.any(np.isnan(pos)):
                nan_count = np.sum(np.isnan(pos))
                return (idx, file_path,
                        f"NaN in pos: {nan_count} values")
            if np.any(np.isnan(pixval)):
                nan_count = np.sum(np.isnan(pixval))
                return (idx, file_path,
                        f"NaN in pixval: {nan_count} values")
            if np.any(np.isnan(lm_score)):
                nan_count = np.sum(np.isnan(lm_score))
                return (idx, file_path,
                        f"NaN in larmatch_score: {nan_count} values")

            # Inf checks
            if np.any(np.isinf(pos)):
                inf_count = np.sum(np.isinf(pos))
                return (idx, file_path,
                        f"Inf in pos: {inf_count} values")
            if np.any(np.isinf(pixval)):
                inf_count = np.sum(np.isinf(pixval))
                return (idx, file_path,
                        f"Inf in pixval: {inf_count} values")
            if np.any(np.isinf(lm_score)):
                inf_count = np.sum(np.isinf(lm_score))
                return (idx, file_path,
                        f"Inf in larmatch_score: {inf_count} values")

            # larmatch_score range check
            if lm_score.min() < -0.01 or lm_score.max() > 1.01:
                return (idx, file_path,
                        f"larmatch_score out of range: "
                        f"[{lm_score.min():.4f}, {lm_score.max():.4f}]")

            # Check optional arrays for consistency if present
            for name in OPTIONAL_ARRAYS:
                if name in grp:
                    arr = np.array(grp[name])
                    if len(arr) != n:
                        return (idx, file_path,
                                f"{name} length {len(arr)} != pos length {n}")

    except OSError as e:
        return (idx, file_path, f"OSError: {e}")
    except Exception as e:
        return (idx, file_path, f"{type(e).__name__}: {e}")

    return None  # Valid


def main():
    parser = argparse.ArgumentParser(
        description="Validate extbnb larmatch H5 files for Sonata pre-training")
    parser.add_argument("file_list", nargs="?",
                        help="Text file containing paths to HDF5 files")
    parser.add_argument("--file", type=str,
                        help="Validate a single file (verbose output)")
    parser.add_argument("--workers", "-w", type=int,
                        default=min(16, cpu_count()),
                        help="Number of parallel workers (default: min(16, cpu_count))")
    parser.add_argument("--output", "-o", type=str,
                        help="Output file for bad file paths")
    parser.add_argument("--min-points", "-m", type=int, default=50,
                        help="Minimum number of points per event (default: 50)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print details for each file")
    parser.add_argument("--num-files", "-n", type=int, default=-1,
                        help="Validate a random subset of N files (-1 = all)")
    args = parser.parse_args()

    # Single file mode
    if args.file:
        result = validate_file((0, args.file))
        if result is None:
            with h5py.File(args.file, "r") as f:
                grp = f["entry_0/triplet_data"]
                pos = np.array(grp["pos"])
                pixval = np.array(grp["pixval"])
                lm = np.array(grp["larmatch_score"])
                print(f"PASS: {args.file}")
                print(f"  points: {len(pos)}")
                print(f"  pos range: [{pos.min():.1f}, {pos.max():.1f}]")
                print(f"  pixval range: [{pixval.min():.1f}, {pixval.max():.1f}]")
                print(f"  larmatch_score: [{lm.min():.4f}, {lm.max():.4f}], "
                      f"mean={lm.mean():.4f}")
        else:
            _, path, error = result
            print(f"FAIL: {path} -- {error}")
            sys.exit(1)
        sys.exit(0)

    if not args.file_list:
        parser.error("Provide file_list or --file")

    # Read file list
    print(f"Reading file list: {args.file_list}")
    with open(args.file_list, "r") as f:
        files = [(i, line.strip()) for i, line in enumerate(f, start=1)
                 if line.strip() and not line.startswith("#")]

    # Optionally subsample
    if args.num_files > 0 and args.num_files < len(files):
        np.random.seed(42)
        indices = np.random.choice(len(files), args.num_files, replace=False)
        files = [files[i] for i in sorted(indices)]

    print(f"Files to validate: {len(files)}")
    print(f"Workers: {args.workers}")
    print(f"Min points: {args.min_points}")
    print()

    # Parallel validation
    bad_files = []
    with Pool(args.workers, initializer=init_worker,
              initargs=(args.min_points,)) as pool:
        results = list(tqdm(
            pool.imap_unordered(validate_file, files, chunksize=64),
            total=len(files),
            desc="Validating",
        ))

    for result in results:
        if result is not None:
            bad_files.append(result)
            if args.verbose:
                idx, path, error = result
                print(f"  FAIL [{idx}]: {path} -- {error}")

    # Categorize errors
    categories = {}
    for _, _, error in bad_files:
        if "Too few points" in error:
            cat = "Too few points"
        elif "Missing" in error:
            cat = "Missing array/group"
        elif "NaN" in error:
            cat = "NaN values"
        elif "Inf" in error:
            cat = "Inf values"
        elif "shape" in error:
            cat = "Shape mismatch"
        elif "out of range" in error:
            cat = "Score out of range"
        elif "length" in error:
            cat = "Length mismatch"
        elif "does not exist" in error:
            cat = "File not found"
        elif "empty" in error.lower():
            cat = "Empty file"
        elif "OSError" in error:
            cat = "Corrupted/unreadable"
        else:
            cat = "Other"
        categories[cat] = categories.get(cat, 0) + 1

    # Report
    print()
    print("=" * 60)
    print("Validation Results")
    print(f"  Total files:  {len(files)}")
    print(f"  Passed:       {len(files) - len(bad_files)}")
    print(f"  Failed:       {len(bad_files)}")
    if categories:
        print()
        print("  Error breakdown:")
        for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
            print(f"    {cat}: {count}")
    print("=" * 60)

    # Print bad files
    if bad_files:
        print()
        print("Failed files:")
        for idx, path, error in sorted(bad_files):
            print(f"  [{idx}] {path}")
            print(f"       {error}")

    # Write bad files list
    if args.output and bad_files:
        with open(args.output, "w") as f:
            for _, path, _ in sorted(bad_files):
                f.write(f"{path}\n")
        print(f"\nBad file paths written to: {args.output}")

    # Write validated good files list
    if args.file_list:
        good_output = args.file_list.replace(".txt", "_sonata_validated.txt")
        if good_output == args.file_list:
            good_output = args.file_list + ".sonata_validated"

        bad_paths = set(path for _, path, _ in bad_files)
        good_count = 0
        with open(good_output, "w") as f:
            for _, path in files:
                if path not in bad_paths:
                    f.write(f"{path}\n")
                    good_count += 1
        print(f"Validated file list ({good_count} files): {good_output}")

    return 1 if bad_files else 0


if __name__ == "__main__":
    sys.exit(main())
