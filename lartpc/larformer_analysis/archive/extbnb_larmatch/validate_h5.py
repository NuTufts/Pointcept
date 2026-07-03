"""
Validate HDF5 files produced by the extbnb larmatch pipeline.

Checks that each file has the expected fields and reasonable values.

Usage (inside pointcept container):
    python3 validate_h5.py --file-list extbnb_larmatch_filelist.txt --num-files 100
    python3 validate_h5.py --file /path/to/single/file.h5
"""

import argparse
import sys
import numpy as np
import h5py


REQUIRED_FIELDS = ["pos", "pixval", "uwire", "vwire", "ywire", "tick", "larmatch_score"]


def validate_file(filepath, verbose=False):
    """Validate a single HDF5 file. Returns (ok, message)."""
    try:
        with h5py.File(filepath, "r") as f:
            if "entry_0/triplet_data" not in f:
                return False, "missing entry_0/triplet_data group"

            grp = f["entry_0/triplet_data"]

            # Check required fields
            for field in REQUIRED_FIELDS:
                if field not in grp:
                    return False, f"missing field: {field}"

            pos = np.array(grp["pos"])
            pixval = np.array(grp["pixval"])
            lm_score = np.array(grp["larmatch_score"])

            n = len(pos)
            if n == 0:
                return False, "empty event (0 points)"

            # Shape checks
            if pos.shape != (n, 3):
                return False, f"pos shape {pos.shape}, expected ({n}, 3)"
            if pixval.shape != (n, 3):
                return False, f"pixval shape {pixval.shape}, expected ({n}, 3)"
            if lm_score.shape != (n,):
                return False, f"larmatch_score shape {lm_score.shape}, expected ({n},)"

            # Value checks
            if np.any(np.isnan(pos)):
                return False, "NaN in pos"
            if np.any(np.isnan(lm_score)):
                return False, "NaN in larmatch_score"
            if lm_score.min() < -0.01 or lm_score.max() > 1.01:
                return False, f"larmatch_score out of range: [{lm_score.min():.3f}, {lm_score.max():.3f}]"

            if verbose:
                msg = (f"OK: {n} points, lm_score=[{lm_score.min():.3f}, {lm_score.max():.3f}], "
                       f"mean={lm_score.mean():.3f}, "
                       f"pixval=[{pixval.min():.1f}, {pixval.max():.1f}]")
                return True, msg

            return True, f"OK: {n} points"

    except Exception as e:
        return False, f"exception: {e}"


def main():
    parser = argparse.ArgumentParser(description="Validate extbnb larmatch H5 files")
    parser.add_argument("--file-list", type=str, help="File list to validate")
    parser.add_argument("--file", type=str, help="Single file to validate")
    parser.add_argument("--num-files", type=int, default=-1, help="Max files to check")
    parser.add_argument("--verbose", action="store_true", help="Print details for each file")
    args = parser.parse_args()

    if args.file:
        ok, msg = validate_file(args.file, verbose=True)
        print(f"{'PASS' if ok else 'FAIL'}: {args.file} -- {msg}")
        sys.exit(0 if ok else 1)

    if not args.file_list:
        print("Provide --file-list or --file")
        sys.exit(1)

    with open(args.file_list) as f:
        files = [line.strip() for line in f if line.strip()]

    if args.num_files > 0:
        np.random.seed(42)
        indices = np.random.choice(len(files), min(args.num_files, len(files)), replace=False)
        files = [files[i] for i in sorted(indices)]

    n_pass = 0
    n_fail = 0

    for filepath in files:
        ok, msg = validate_file(filepath, verbose=args.verbose)
        if not ok:
            print(f"FAIL: {filepath} -- {msg}")
            n_fail += 1
        else:
            if args.verbose:
                print(f"PASS: {filepath} -- {msg}")
            n_pass += 1

    print(f"\nResults: {n_pass} passed, {n_fail} failed out of {len(files)} checked")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
