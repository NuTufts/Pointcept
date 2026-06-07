"""Augment Stage-1+2 cache files with a per-SP `particle_class_id` array.

CPU-only — no model load, no source merged_h5 access. Each cache file
already contains `entry_0/particle_instances/instance_<k>/` groups with
`truth_indices` (per-instance SP indices into the cache's SP array) and
a `class_id` attribute. This tool derives a flat per-SP integer array
from those:

    entry_0/particle_class_id  (n_in_cache,) int64
        value ∈ {0..5}    — SP belongs to a GT particle of that class
        value = -1        — SP not in any GT particle (false positive
                            from Stage 2; correct cls-loss target is
                            "ignore", matching the loss's
                            `ignore_index=-1` convention)

Mode:
    Default: IN-PLACE. Opens each .h5 in r+ and adds the new dataset
             (or replaces it with --force). Files already augmented
             are skipped (idempotent).

    --output-dir DIR: copy each file to a mirrored path under DIR and
             augment the copy. Original cache untouched. Useful when
             cluster policy disallows in-place modification.

Parallelism: --workers parallel processes (multiprocessing). Each
worker opens one file at a time. Work is I/O-bound — workers ≈ cpu_count
is appropriate for shared filesystems; lower if the filesystem chokes.

Usage on a cluster:

    # in-place, train split
    python tools/augment_stage12_cache_particle_class_id.py \\
        --cache-root /path/to/stage12_cache_v2/train \\
        --workers 32

    # ... and val
    python tools/augment_stage12_cache_particle_class_id.py \\
        --cache-root /path/to/stage12_cache_v2/val \\
        --workers 32

    # alternative: copy + augment to a parallel directory
    python tools/augment_stage12_cache_particle_class_id.py \\
        --cache-root /path/to/stage12_cache_v2/train \\
        --output-dir /path/to/stage12_cache_v3/train \\
        --workers 32

Sanity flags:
    --dry-run     list files but don't open/modify any
    --limit N     process only the first N files (smoke test before
                  unleashing on 400K)
    --verbose     print one line per file

Inverse:
    Remove `entry_0/particle_class_id` from every file:
      for f in $(find $CACHE_ROOT -name '*.h5'); do
          h5py -c "import h5py;f=h5py.File('$f','r+');del f['entry_0/particle_class_id'];f.close()"
      done
    (or write the equivalent ~5-line script).
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import h5py
import numpy as np


DATASET_NAME = "entry_0/particle_class_id"
NO_GT_LABEL = -1                  # = cls loss `ignore_index`


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def list_cache_files(cache_root: str) -> list[str]:
    out: list[str] = []
    for root, _dirs, files in os.walk(cache_root):
        for fn in files:
            if fn.endswith(".h5"):
                out.append(os.path.join(root, fn))
    return sorted(out)


# ---------------------------------------------------------------------------
# Core: compute the per-SP label array from an open cache file
# ---------------------------------------------------------------------------

def compute_particle_class_id(f: h5py.File) -> tuple[int, np.ndarray, int]:
    """Returns (n_cache, particle_class_id, n_instances_used).

    Raises ValueError if the file doesn't look like a v2 cache (missing
    `n_in_cache` attr or `entry_0/particle_instances` group).
    """
    if "n_in_cache" not in f.attrs:
        raise ValueError(
            "missing attrs['n_in_cache'] — not a v2 cache file? "
            "Available attrs: " + repr(list(f.attrs.keys())[:10])
        )
    n_cache = int(f.attrs["n_in_cache"])
    out = np.full(n_cache, NO_GT_LABEL, dtype=np.int64)

    if "entry_0/particle_instances" not in f:
        # Legal: events with zero surviving GT particles. All -1.
        return n_cache, out, 0

    grp = f["entry_0/particle_instances"]
    n_used = 0
    for name in grp.keys():
        if not name.startswith("instance_"):
            continue
        inst = grp[name]
        if "truth_indices" not in inst:
            continue
        cls = int(inst.attrs.get("class_id", -1))
        if cls < 0:
            continue
        ti = np.asarray(inst["truth_indices"][:], dtype=np.int64)
        if ti.size == 0:
            continue
        if ti.max() >= n_cache or ti.min() < 0:
            raise ValueError(
                f"{name}: truth_indices out of range "
                f"[{int(ti.min())}, {int(ti.max())}] vs n_cache={n_cache}"
            )
        out[ti] = cls
        n_used += 1
    return n_cache, out, n_used


# ---------------------------------------------------------------------------
# Per-file workers
# ---------------------------------------------------------------------------

def process_inplace(path: str, force: bool = False) -> dict:
    """In-place augment a single file. Returns a status dict."""
    try:
        with h5py.File(path, "r+") as f:
            if DATASET_NAME in f and not force:
                return {
                    "path": path, "status": "skip-existing",
                    "n_cache": int(f.attrs.get("n_in_cache", 0)),
                    "n_instances": 0, "n_labeled": 0,
                }
            n_cache, pcid, n_inst = compute_particle_class_id(f)
            if DATASET_NAME in f:
                del f[DATASET_NAME]
            f.create_dataset(DATASET_NAME, data=pcid)
            # File-level attr so a downstream filter can quickly tell
            # augmented from un-augmented caches without opening the
            # entry_0 group.
            f.attrs["has_particle_class_id"] = True
            return {
                "path": path, "status": "augmented",
                "n_cache": n_cache, "n_instances": n_inst,
                "n_labeled": int((pcid >= 0).sum()),
            }
    except (OSError, KeyError, ValueError) as e:
        return {"path": path, "status": "error",
                "error": f"{type(e).__name__}: {e}"}


def process_copy(src_path: str, src_root: str, dst_root: str,
                 force: bool = False) -> dict:
    """Copy src to mirrored path under dst_root, then augment the copy."""
    rel = os.path.relpath(src_path, src_root)
    dst_path = os.path.join(dst_root, rel)
    if os.path.exists(dst_path) and not force:
        try:
            with h5py.File(dst_path, "r") as f:
                if DATASET_NAME in f:
                    return {
                        "path": dst_path, "status": "skip-existing-dst",
                        "n_cache": int(f.attrs.get("n_in_cache", 0)),
                        "n_instances": 0, "n_labeled": 0,
                    }
        except OSError:
            pass  # corrupted, fall through and re-copy
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    shutil.copy2(src_path, dst_path)
    return process_inplace(dst_path, force=True)


def _worker(task: tuple) -> dict:
    src_path, src_root, dst_root, force = task
    if dst_root is None:
        return process_inplace(src_path, force=force)
    return process_copy(src_path, src_root, dst_root, force=force)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--cache-root", required=True,
                   help="Path to a Stage-1+2 cache directory tree "
                        "(recursive crawl for *.h5 files).")
    p.add_argument("--output-dir", default=None,
                   help="If set, copy files to this directory tree "
                        "(mirroring the layout) and augment the copies. "
                        "Default: in-place augment in --cache-root.")
    p.add_argument("--workers", type=int, default=8,
                   help="multiprocessing pool size (default 8).")
    p.add_argument("--force", action="store_true",
                   help="Overwrite particle_class_id even if already "
                        "present.")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N files (for smoke "
                        "tests).")
    p.add_argument("--dry-run", action="store_true",
                   help="List files; don't open or modify anything.")
    p.add_argument("--verbose", action="store_true",
                   help="Print one line per file.")
    p.add_argument("--progress-every", type=int, default=0,
                   help="Print summary line every N files (default: "
                        "auto = ~100 reports total).")
    return p.parse_args()


def main():
    args = parse_args()

    cache_root = os.path.abspath(args.cache_root)
    if not os.path.isdir(cache_root):
        print(f"ERROR: --cache-root does not exist: {cache_root}",
              file=sys.stderr)
        sys.exit(2)
    dst_root = os.path.abspath(args.output_dir) if args.output_dir else None

    print("=" * 70)
    print(f"Cache root:    {cache_root}")
    print(f"Mode:          {'COPY → ' + dst_root if dst_root else 'IN-PLACE'}")
    print(f"Workers:       {args.workers}")
    print(f"Force:         {args.force}")
    if args.limit is not None:
        print(f"Limit:         first {args.limit} files only")
    print("=" * 70)
    print()

    print("Discovering cache files ...")
    files = list_cache_files(cache_root)
    print(f"  Found {len(files)} .h5 file(s).")
    if args.limit is not None:
        files = files[:args.limit]
        print(f"  Limiting to {len(files)} (--limit).")

    if not files:
        print("Nothing to do.")
        return

    if args.dry_run:
        print("\nDry run — would process:")
        for fp in files[:30]:
            print(f"  {fp}")
        if len(files) > 30:
            print(f"  ... ({len(files) - 30} more)")
        return

    report_every = args.progress_every or max(1, len(files) // 100)

    print(f"\nProcessing with {args.workers} worker(s) ...")
    t_start = time.perf_counter()
    tasks = [(fp, cache_root, dst_root, args.force) for fp in files]

    n_aug = n_skip = n_err = 0
    n_labeled_total = n_sp_total = n_inst_total = 0
    errors: list[tuple[str, str]] = []

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_worker, t): t[0] for t in tasks}
        for i, fut in enumerate(as_completed(futures)):
            src_path = futures[fut]
            try:
                result = fut.result()
            except Exception as e:
                n_err += 1
                errors.append((src_path, f"worker raised: {e}"))
                continue

            status = result["status"]
            if status == "augmented":
                n_aug += 1
                n_labeled_total += result["n_labeled"]
                n_sp_total += result["n_cache"]
                n_inst_total += result["n_instances"]
                if args.verbose:
                    print(f"  + {result['path']}  "
                          f"n={result['n_cache']}  "
                          f"inst={result['n_instances']}  "
                          f"labeled={result['n_labeled']}")
            elif status.startswith("skip"):
                n_skip += 1
                if args.verbose:
                    print(f"  . {result['path']}  ({status})")
            elif status == "error":
                n_err += 1
                errors.append((result["path"],
                              result.get("error", "(no error msg)")))
                print(f"  ! {result['path']}  ERROR: "
                      f"{result.get('error')}")

            if (i + 1) % report_every == 0 or (i + 1) == len(files):
                elapsed = time.perf_counter() - t_start
                rate = (i + 1) / max(1e-9, elapsed)
                eta = (len(files) - (i + 1)) / max(1e-9, rate)
                print(f"  [{i+1:6d}/{len(files)}] "
                      f"aug={n_aug} skip={n_skip} err={n_err}  "
                      f"({rate:6.1f}/s, eta {eta/60:5.1f}min)")

    elapsed = time.perf_counter() - t_start
    print()
    print("=" * 70)
    print(f"Done in {elapsed/60:.2f} min ({len(files)} files, "
          f"{len(files)/max(1e-9, elapsed):.1f}/s avg)")
    print(f"  augmented:        {n_aug}")
    print(f"  skipped-existing: {n_skip}")
    print(f"  errors:           {n_err}")
    if n_aug > 0:
        frac = n_labeled_total / max(1, n_sp_total)
        avg_inst = n_inst_total / max(1, n_aug)
        print(f"  SPs labeled in augmented files: "
              f"{n_labeled_total:,} / {n_sp_total:,}  "
              f"({100*frac:.1f}% have a GT particle)")
        print(f"  avg GT instances per augmented event: "
              f"{avg_inst:.2f}")
    if errors:
        print()
        print(f"First {min(10, len(errors))} errors:")
        for path, msg in errors[:10]:
            print(f"  {path}")
            print(f"      → {msg}")
    print("=" * 70)

    if n_err > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
