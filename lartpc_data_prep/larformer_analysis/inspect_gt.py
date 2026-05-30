"""Inspect ground-truth instance counts in merged H5 files.

Uses `compute_slice_labels` (same routine the LArFormerDataset calls
internally) to derive per-event slice info, then reports:

  - n_sp_total        : total spacepoints (= triplet_data['trackid'].size)
  - n_sp_real         : spacepoints with hasmatch != 0 (non-ghost)
  - n_gt              : number of GT instances (slices) — primaries
  - n_nu_gt           : primaries with origin == 1 (nu)
  - n_cosmic_gt       : primaries with origin == 2 (cosmic)
  - run, subrun, event: from entry_0 attrs (when present)

Pure h5py + numpy + the slice_labels module — no torch, runs anywhere.

Usage:
  # Single file
  python inspect_gt.py /path/to/merged_*.h5

  # Many files (any mix of paths + --list)
  python inspect_gt.py file1.h5 file2.h5 ... \\
      [--list /path/to/_inputlists/<TAG>/task000000.txt] \\
      [--all]        # print every file, not just the 0-GT ones
      [--csv out.csv] # write a CSV of all rows

Default prints only files with `n_gt == 0` (the ones that would crash
the pre-fix inference). `--all` prints every file.

Tip: to find the file that crashed your SLURM task, the inputlist for
that task is at
   ${OUTPUT_DIR}/_inputlists/<TAG>/task<NNNNNN>.txt
Pass it via `--list` and look at the row with n_gt == 0.
"""

import argparse
import csv
import os
import sys

import h5py
import numpy as np


REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.insert(0, REPO_ROOT)
from lartpc_data_prep.slice_labels import compute_slice_labels  # noqa: E402


def inspect_one(merged_h5_path):
    """Return a dict with the GT summary, or {'error': '...'} on failure."""
    if not os.path.exists(merged_h5_path):
        return {"path": merged_h5_path, "error": "file not found"}
    try:
        with h5py.File(merged_h5_path, "r") as f:
            if "entry_0" not in f:
                return {"path": merged_h5_path,
                        "error": "no entry_0 group"}
            e0 = f["entry_0"]
            run    = int(e0.attrs.get("run", -1))
            subrun = int(e0.attrs.get("subrun", -1))
            event  = int(e0.attrs.get("event", -1))
            if "mc_particle_tree" not in e0 or "triplet_data" not in e0:
                return {"path": merged_h5_path,
                        "run": run, "subrun": subrun, "event": event,
                        "error": "missing mc_particle_tree / triplet_data"}
            mpt = e0["mc_particle_tree"]
            td  = e0["triplet_data"]
            sp_trackid  = td["trackid"][:]
            sp_hasmatch = td["hasmatch"][:] if "hasmatch" in td else None
            n_sp_total = int(sp_trackid.shape[0])
            n_sp_real  = (int((sp_hasmatch != 0).sum())
                          if sp_hasmatch is not None else n_sp_total)

            slice_info = compute_slice_labels(mpt, sp_trackid, sp_hasmatch)
            n_gt        = int(len(slice_info["primary_trackid"]))
            n_nu_gt     = int((slice_info["primary_origin"] == 1).sum())
            n_cosmic_gt = int((slice_info["primary_origin"] == 2).sum())
    except Exception as exc:
        return {"path": merged_h5_path,
                "error": f"{type(exc).__name__}: {exc}"}
    return dict(
        path=merged_h5_path,
        run=run, subrun=subrun, event=event,
        n_sp_total=n_sp_total, n_sp_real=n_sp_real,
        n_gt=n_gt, n_nu_gt=n_nu_gt, n_cosmic_gt=n_cosmic_gt,
    )


def _print_row(rec, header=False):
    if header:
        print(f"  {'run':>5s} {'sub':>3s} {'evt':>5s}  "
              f"{'n_sp':>7s} {'n_real':>7s}  "
              f"{'n_gt':>4s} {'n_nu':>4s} {'n_cos':>5s}  "
              f"path")
        return
    if "error" in rec:
        print(f"  ! ERROR  {rec.get('path', '?')}: {rec['error']}")
        return
    print(
        f"  {rec['run']:>5d} {rec['subrun']:>3d} {rec['event']:>5d}  "
        f"{rec['n_sp_total']:>7d} {rec['n_sp_real']:>7d}  "
        f"{rec['n_gt']:>4d} {rec['n_nu_gt']:>4d} {rec['n_cosmic_gt']:>5d}  "
        f"{os.path.basename(rec['path'])}"
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("paths", nargs="*",
                    help="merged_*.h5 paths to inspect")
    ap.add_argument("--list", default=None,
                    help="Text file with one merged_*.h5 path per line "
                         "(in addition to positional paths)")
    ap.add_argument("--all", action="store_true",
                    help="Print every file, not just those with n_gt==0")
    ap.add_argument("--csv", default=None,
                    help="Write a CSV of all inspected rows (always all, "
                         "regardless of --all flag for stdout)")
    args = ap.parse_args()

    paths = list(args.paths)
    if args.list is not None:
        with open(args.list, "r") as f:
            for raw in f:
                p = raw.strip()
                if p and not p.startswith("#"):
                    paths.append(p)
    if not paths:
        sys.exit("no paths given (provide positional args and/or --list)")

    print(f"Inspecting {len(paths)} merged H5 file(s)…")
    rows = [inspect_one(p) for p in paths]

    # Buckets
    zero_gt = [r for r in rows if r.get("n_gt") == 0]
    errors  = [r for r in rows if "error" in r]
    ok      = [r for r in rows if "error" not in r and r["n_gt"] > 0]

    if args.all:
        _print_row(None, header=True)
        for r in rows:
            _print_row(r)
    else:
        if zero_gt:
            print(f"\n=== files with n_gt == 0  ({len(zero_gt)} of {len(rows)}) ===")
            _print_row(None, header=True)
            for r in zero_gt:
                _print_row(r)
        if errors:
            print(f"\n=== files with errors  ({len(errors)} of {len(rows)}) ===")
            for r in errors:
                _print_row(r)

    print(f"\nSummary: total={len(rows)}  "
          f"ok={len(ok)}  zero_gt={len(zero_gt)}  errors={len(errors)}")

    if args.csv is not None:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path", "run", "subrun", "event",
                        "n_sp_total", "n_sp_real",
                        "n_gt", "n_nu_gt", "n_cosmic_gt", "error"])
            for r in rows:
                w.writerow([
                    r.get("path", ""),
                    r.get("run", ""), r.get("subrun", ""), r.get("event", ""),
                    r.get("n_sp_total", ""), r.get("n_sp_real", ""),
                    r.get("n_gt", ""), r.get("n_nu_gt", ""),
                    r.get("n_cosmic_gt", ""),
                    r.get("error", ""),
                ])
        print(f"  wrote {args.csv}")


if __name__ == "__main__":
    main()
