#!/usr/bin/env python3
"""
Validate merged H5 files against the shower-clustering training pipeline.

For each file the script runs two passes:

1. **Static checks** (h5py-only, fast): the file opens, the required groups
   and datasets exist, per-spacepoint and per-fragment arrays have matching
   lengths, fragment point indices are in range, lm_score/hasmatch are in
   their expected ranges, and `mc_particle_tree` (if present) is internally
   consistent.

2. **Functional check** (optional, --functional): build a one-file
   `ShowerClusteringDataset` and call `__getitem__(0)`. This exercises the
   lm_score filter, the 0.25 cm dedup, fragment-index remap, voxel ID
   construction, and `mc_particle_tree` descendant traversal exactly as
   training does — so anything the loader rejects shows up here.

The dataset reference: pointcept/pointcept/datasets/shower_clustering.py.

Usage:
  python3 validate_shower_clustering_data.py PATH [PATH ...] [options]

PATH may be a merged H5 file, a directory (scanned recursively for
merged_*_entry*.h5), or a text file (with --from-list).

Exit code is 0 if every checked file passes static + (optional) functional;
otherwise 1.
"""

import argparse
import os
import sys
import time
import traceback
from contextlib import contextmanager

import h5py
import numpy as np


# --- Required schema -----------------------------------------------------

REQ_TD = ("pos", "lm_score", "pixval", "trackid", "pid")
OPT_TD_PER_POINT = ("uwire", "vwire", "ywire", "origin", "hasmatch",
                    "ssnet_label", "tick", "shower_score")

REQ_SF = ("pointindices_flat", "pointindices_counts")
OPT_SF_PER_FRAG = ("trackid", "pid", "type", "originpt",
                   "pret0shiftedoriginpt", "startpt", "istrunk")

REQ_MPT = ("trackid", "parent_trackid", "pid")  # only if mc_particle_tree present


class FileReport:
    __slots__ = ("path", "errors", "warnings", "static_ok", "functional_ok",
                 "n_spacepoints", "n_fragments", "n_gt_instances",
                 "elapsed_static", "elapsed_functional")

    def __init__(self, path):
        self.path = path
        self.errors = []
        self.warnings = []
        self.static_ok = False
        self.functional_ok = None  # None == not run
        self.n_spacepoints = None
        self.n_fragments = None
        self.n_gt_instances = None
        self.elapsed_static = 0.0
        self.elapsed_functional = 0.0

    def err(self, msg):
        self.errors.append(msg)

    def warn(self, msg):
        self.warnings.append(msg)

    @property
    def ok(self):
        if not self.static_ok:
            return False
        if self.functional_ok is False:
            return False
        return True


# --- Static validation ---------------------------------------------------

def _check_per_point_lengths(td, n_sp, report):
    """All datasets directly under triplet_data that look per-point should
    have shape[0] == n_sp."""
    for k, v in td.items():
        if not isinstance(v, h5py.Dataset):
            continue
        if v.shape and v.shape[0] != n_sp:
            # Some triplet_data datasets aren't per-point; the per-point
            # ones we know about are checked here. Anything else with a
            # different leading dim is just a warning.
            if k in REQ_TD or k in OPT_TD_PER_POINT:
                report.err(
                    f"triplet_data/{k}: shape[0]={v.shape[0]} != "
                    f"n_spacepoints={n_sp}"
                )
            else:
                report.warn(
                    f"triplet_data/{k}: shape[0]={v.shape[0]} doesn't match "
                    f"n_spacepoints={n_sp} (unknown field — may be fine)"
                )


def _check_per_fragment_lengths(sf, n_frags, report):
    for k, v in sf.items():
        if not isinstance(v, h5py.Dataset):
            continue
        if k in ("pointindices_flat", "nu_vertex_is_visible"):
            continue
        if k in OPT_SF_PER_FRAG and v.shape and v.shape[0] != n_frags:
            report.err(
                f"shower_fragments/{k}: shape[0]={v.shape[0]} != "
                f"num_fragments={n_frags}"
            )


def static_validate(path, report):
    """Open the file with h5py and run all the cheap structural checks."""
    t0 = time.perf_counter()
    try:
        with h5py.File(path, "r") as f:
            if "entry_0" not in f:
                report.err("missing top-level group 'entry_0'")
                return
            entry = f["entry_0"]

            # ---- triplet_data ----
            if "triplet_data" not in entry:
                report.err("missing entry_0/triplet_data group")
                return
            td = entry["triplet_data"]
            for k in REQ_TD:
                if k not in td:
                    report.err(f"missing triplet_data/{k}")
            if report.errors:
                return

            pos = td["pos"]
            n_sp = pos.shape[0]
            report.n_spacepoints = int(n_sp)

            if pos.ndim != 2 or pos.shape[1] != 3:
                report.err(f"triplet_data/pos shape {pos.shape} != (N, 3)")
            if "pixval" in td and (td["pixval"].ndim != 2
                                   or td["pixval"].shape[1] != 3):
                report.err(
                    f"triplet_data/pixval shape {td['pixval'].shape} != (N, 3)"
                )
            if n_sp == 0:
                report.err("triplet_data has zero spacepoints")
                return

            _check_per_point_lengths(td, n_sp, report)

            # Range checks
            lm = td["lm_score"][:]
            if lm.size > 0:
                lo, hi = float(lm.min()), float(lm.max())
                if lo < 0.0 or hi > 1.0 + 1e-4:
                    report.warn(
                        f"lm_score outside [0,1]: min={lo:.4f} max={hi:.4f}"
                    )
            if "hasmatch" in td:
                hm = td["hasmatch"][:]
                uniq = np.unique(hm)
                if not set(uniq.tolist()).issubset({0, 1}):
                    report.err(
                        f"hasmatch has unexpected values "
                        f"{uniq.tolist()[:10]}"
                    )

            # ---- shower_fragments ----
            if "shower_fragments" not in entry:
                report.err("missing entry_0/shower_fragments group")
                return
            sf = entry["shower_fragments"]
            for k in REQ_SF:
                if k not in sf:
                    report.err(f"missing shower_fragments/{k}")
            if any("missing shower_fragments/" in e for e in report.errors):
                return

            counts = sf["pointindices_counts"][:].astype(np.int64)
            flat = sf["pointindices_flat"][:].astype(np.int64)
            n_frags_attr = int(sf.attrs.get("num_fragments", -1))
            n_frags = len(counts)
            report.n_fragments = n_frags
            if n_frags_attr >= 0 and n_frags_attr != n_frags:
                report.err(
                    f"num_fragments attr={n_frags_attr} != "
                    f"len(pointindices_counts)={n_frags}"
                )
            if int(counts.sum()) != int(flat.size):
                report.err(
                    f"sum(pointindices_counts)={int(counts.sum())} != "
                    f"len(pointindices_flat)={int(flat.size)}"
                )
            if flat.size > 0:
                fmin, fmax = int(flat.min()), int(flat.max())
                if fmin < 0 or fmax >= n_sp:
                    report.err(
                        f"pointindices_flat out of range [0, {n_sp}): "
                        f"min={fmin} max={fmax}"
                    )

            _check_per_fragment_lengths(sf, n_frags, report)

            # originpt shape
            if "originpt" in sf and sf["originpt"].ndim == 2 \
                    and sf["originpt"].shape[1] != 3:
                report.err(
                    f"shower_fragments/originpt shape "
                    f"{sf['originpt'].shape} != (F, 3)"
                )

            # ---- mc_particle_tree (optional but expected for GT) ----
            if "mc_particle_tree" in entry:
                mpt = entry["mc_particle_tree"]
                lengths = {}
                for k in REQ_MPT:
                    if k not in mpt:
                        report.err(f"missing mc_particle_tree/{k}")
                    else:
                        lengths[k] = int(mpt[k].shape[0])
                if lengths and len(set(lengths.values())) > 1:
                    report.err(
                        f"mc_particle_tree arrays have mismatched lengths: "
                        f"{lengths}"
                    )
                # parent_trackid values that are positive should mostly map
                # to known trackids. Warn if many don't (orphan parents are
                # common at the tree root, so allow a fraction).
                if "trackid" in mpt and "parent_trackid" in mpt:
                    tids = set(int(x) for x in mpt["trackid"][:].tolist())
                    parents = mpt["parent_trackid"][:].astype(np.int64)
                    pos_parents = parents[parents > 0]
                    if pos_parents.size > 0:
                        miss = sum(1 for p in pos_parents.tolist()
                                   if int(p) not in tids)
                        frac = miss / pos_parents.size
                        if frac > 0.5:
                            report.warn(
                                f"mc_particle_tree: {miss}/{pos_parents.size} "
                                f"({frac:.0%}) positive parent_trackids not "
                                f"in trackid set"
                            )
            else:
                report.warn(
                    "no mc_particle_tree group — GT instances will be empty "
                    "(file usable only without GT supervision)"
                )

            # ---- entry attrs (advisory) ----
            for attr in ("run", "subrun", "event"):
                if attr not in entry.attrs:
                    report.warn(f"entry_0 missing attribute '{attr}'")

        if not report.errors:
            report.static_ok = True
    except OSError as e:
        report.err(f"cannot open as HDF5: {e}")
    except Exception as e:
        report.err(f"static validation crashed: {e!r}")
    finally:
        report.elapsed_static = time.perf_counter() - t0


# --- Functional validation (optional) ------------------------------------

_DATASET_CLS = None


def _get_dataset_cls():
    """Lazy-import ShowerClusteringDataset so static-only mode doesn't need
    pointcept on PYTHONPATH."""
    global _DATASET_CLS
    if _DATASET_CLS is not None:
        return _DATASET_CLS

    here = os.path.dirname(os.path.abspath(__file__))
    pointcept_root = os.path.dirname(here)
    if pointcept_root not in sys.path:
        sys.path.insert(0, pointcept_root)

    from pointcept.datasets.shower_clustering import ShowerClusteringDataset
    _DATASET_CLS = ShowerClusteringDataset
    return _DATASET_CLS


def functional_validate(path, report, lm_threshold=0.15):
    """Build a one-file dataset and call __getitem__(0)."""
    t0 = time.perf_counter()
    try:
        cls = _get_dataset_cls()
        ds = cls(
            split="val",
            data_root=os.path.dirname(path) or ".",
            data_list_file=None,
            lm_score_val_threshold=lm_threshold,
        )
        # Override data_list to just our one file (avoid reading split.txt).
        ds.data_list = [path]
        sample = ds[0]
    except Exception as e:
        report.err(
            f"functional load failed: {type(e).__name__}: {e}\n"
            + "".join(traceback.format_exception(type(e), e, e.__traceback__))
        )
        report.functional_ok = False
        return
    finally:
        report.elapsed_functional = time.perf_counter() - t0

    # Validate the returned dict's shape consistency.
    try:
        n_keep = int(sample["n_spacepoints"])
        for k in ("coord", "coord_norm", "grid_coord", "feat", "lm_score",
                  "wire", "trackid", "pid", "origin_label", "hasmatch",
                  "ssnet_label", "voxel_id"):
            if k not in sample:
                report.err(f"sample missing key '{k}'")
                continue
            arr = sample[k]
            if arr.shape[0] != n_keep:
                report.err(
                    f"sample['{k}'].shape[0]={arr.shape[0]} != "
                    f"n_spacepoints={n_keep}"
                )
        if sample["coord"].shape[1] != 3:
            report.err(
                f"sample['coord'].shape[1]={sample['coord'].shape[1]} != 3"
            )
        if sample["feat"].shape[1] != 6:
            report.err(
                f"sample['feat'].shape[1]={sample['feat'].shape[1]} != 6 "
                "(coord_norm + log(pixval))"
            )
        report.n_fragments = int(sample.get("n_fragments", report.n_fragments or 0))
        report.n_gt_instances = int(sample.get("n_gt_instances", 0))

        # Fragment indices must lie in [0, n_keep).
        for fi, idx in enumerate(sample.get("fragment_indices", [])):
            if idx.size == 0:
                report.warn(f"fragment {fi}: zero indices")
                continue
            if int(idx.min()) < 0 or int(idx.max()) >= n_keep:
                report.err(
                    f"fragment {fi}: indices out of range "
                    f"[0, {n_keep}): min={int(idx.min())} "
                    f"max={int(idx.max())}"
                )
        # GT-instance truth_indices must lie in [0, n_keep).
        for gi, gt in enumerate(sample.get("gt_instances", [])):
            ti = gt["truth_indices"]
            if ti.size == 0:
                report.err(f"gt_instance {gi}: zero truth_indices")
                continue
            if int(ti.min()) < 0 or int(ti.max()) >= n_keep:
                report.err(
                    f"gt_instance {gi}: truth_indices out of range "
                    f"[0, {n_keep})"
                )

        if not report.errors:
            report.functional_ok = True
        else:
            report.functional_ok = False
    except Exception as e:
        report.err(f"sample-shape verification crashed: {e!r}")
        report.functional_ok = False


# --- File discovery ------------------------------------------------------

def discover_paths(args):
    paths = []
    if args.from_list:
        with open(args.from_list, "r") as f:
            for line in f:
                ln = line.strip()
                if ln and not ln.startswith("#"):
                    paths.append(ln)
    for p in args.paths or []:
        if os.path.isfile(p):
            paths.append(p)
        elif os.path.isdir(p):
            for root, _, files in os.walk(p):
                for fn in files:
                    if fn.startswith("merged_") and "_entry" in fn \
                            and fn.endswith(".h5"):
                        paths.append(os.path.join(root, fn))
        else:
            print(f"WARNING: path not found: {p}", file=sys.stderr)
    # Dedup, preserve order.
    seen = set()
    uniq = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    if args.max_files and args.max_files > 0:
        uniq = uniq[: args.max_files]
    return uniq


# --- Reporting -----------------------------------------------------------

def print_report(rep, verbose):
    status = "PASS" if rep.ok else "FAIL"
    func_str = ""
    if rep.functional_ok is True:
        func_str = " func=ok"
    elif rep.functional_ok is False:
        func_str = " func=FAIL"
    extras = []
    if rep.n_spacepoints is not None:
        extras.append(f"N={rep.n_spacepoints}")
    if rep.n_fragments is not None:
        extras.append(f"F={rep.n_fragments}")
    if rep.n_gt_instances is not None:
        extras.append(f"GT={rep.n_gt_instances}")
    extras_str = (" " + " ".join(extras)) if extras else ""
    print(f"[{status}]{func_str}{extras_str}  {rep.path}")
    if verbose or not rep.ok:
        for w in rep.warnings:
            print(f"    WARN: {w}")
        for e in rep.errors:
            print(f"    ERROR: {e}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("paths", nargs="*",
                    help="H5 files or directories.")
    ap.add_argument("--from-list", default=None,
                    help="Text file with one H5 path per line.")
    ap.add_argument("--functional", action="store_true",
                    help="Also build ShowerClusteringDataset and load each "
                         "file via __getitem__(0). Slower but catches "
                         "loader-specific failures.")
    ap.add_argument("--lm-threshold", type=float, default=0.15,
                    help="lm_score threshold for the functional check "
                         "(default: 0.15, the dataset's val default).")
    ap.add_argument("--max-files", type=int, default=0,
                    help="Limit to first N files (0 = all).")
    ap.add_argument("--shard", default=None,
                    help="Process only shard K of N (e.g. '0/50' for the "
                         "first of 50 partitions). Use round-robin "
                         "assignment so neighboring files in the discovery "
                         "order go to different shards. Designed for SLURM "
                         "array jobs where SLURM_ARRAY_TASK_ID is K.")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="Print warnings even on PASS.")
    ap.add_argument("--quiet", "-q", action="store_true",
                    help="Only print failures.")
    ap.add_argument("--fail-list", default=None,
                    help="Write failing file paths to this file, one per "
                         "line. Useful to pipe back into a re-merge step.")
    ap.add_argument("--pass-list", default=None,
                    help="Write passing file paths to this file, one per "
                         "line. Use this output as the data list for "
                         "training/validation splits.")
    args = ap.parse_args()

    paths = discover_paths(args)
    if not paths:
        print("ERROR: no files to validate", file=sys.stderr)
        sys.exit(2)

    total_before_shard = len(paths)
    if args.shard:
        try:
            k_str, n_str = args.shard.split("/")
            shard_k, shard_n = int(k_str), int(n_str)
            if shard_n <= 0 or shard_k < 0 or shard_k >= shard_n:
                raise ValueError
        except ValueError:
            print(f"ERROR: --shard expects 'K/N' with 0<=K<N, "
                  f"got {args.shard!r}", file=sys.stderr)
            sys.exit(2)
        # Round-robin: shard k gets paths [k, k+N, k+2N, ...]. Spreads any
        # spatial bias in the input order (e.g. fileno-sorted) across shards.
        paths = paths[shard_k::shard_n]
        print(f"Shard {shard_k}/{shard_n}: {len(paths)} of "
              f"{total_before_shard} total files")

    print(f"Validating {len(paths)} file(s) "
          f"[functional={args.functional}, lm_threshold={args.lm_threshold}]")

    n_pass = n_fail = 0
    failures = []
    passes = []
    t_start = time.perf_counter()
    for i, p in enumerate(paths):
        rep = FileReport(p)
        static_validate(p, rep)
        if args.functional and rep.static_ok:
            functional_validate(p, rep, lm_threshold=args.lm_threshold)
        if rep.ok:
            n_pass += 1
            passes.append(p)
            if not args.quiet:
                print_report(rep, args.verbose)
        else:
            n_fail += 1
            failures.append(p)
            print_report(rep, args.verbose)

    elapsed = time.perf_counter() - t_start
    print()
    print(f"Summary: {n_pass} PASS, {n_fail} FAIL, total {len(paths)} "
          f"({elapsed:.1f}s)")

    if args.fail_list and failures:
        with open(args.fail_list, "w") as f:
            f.write("\n".join(failures) + "\n")
        print(f"Wrote {len(failures)} failing paths to {args.fail_list}")

    if args.pass_list:
        with open(args.pass_list, "w") as f:
            if passes:
                f.write("\n".join(passes) + "\n")
        print(f"Wrote {len(passes)} passing paths to {args.pass_list}")

    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
