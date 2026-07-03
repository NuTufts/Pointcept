"""Augment Stage-1+2 cache files with the source `mckeypoints` group.

The stage-12 cache (built by `build_stage12_cache_event.py`) carries the
deghosted + sliced spacepoints and per-particle GT, but NOT the keypoint
ground truth. This tool copies `entry_0/mckeypoints` from each cache file's
source merged_h5 into the cache, so the LArFormer keypoint module can train on
the cache. The cache reader (`LArFormerStage12CacheDataset(emit_keypoints=
True)`) then computes the dense `kpscores` target + per-instance endpoints +
nu-vertex on-the-fly from this group — same path as the raw merged_h5 dataset.

Source resolution: each cache file stores its source basename in the
`source_h5` top-level attr. This tool indexes `--merged-root` by basename and
copies the matching file's `mckeypoints` group. (Unlike the
`particle_class_id` augment tool, this one DOES need the source merged_h5.)

Modes (mirror augment_stage12_cache_particle_class_id.py):
    Default IN-PLACE (r+); files already carrying mckeypoints are skipped
    (idempotent) unless --force. `--output-dir DIR` copies then augments.

Usage:
    python tools/augment_stage12_cache_keypoints.py \\
        --cache-root exp/cache_stage12_devdata/train \\
        --merged-root /mnt/ddrive/data/ub_on_tufts/h5/bnb_nu_pi0filter_corsika/merged_h5 \\
        --workers 16

Sanity flags: --dry-run, --limit N, --verbose.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import h5py

# Make `lartpc_data_prep` importable when run from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lartpc.data_prep.labels.keypoint_labels import copy_mckeypoints_group  # noqa: E402


MCKP_PATH = "entry_0/mckeypoints"


# ---------------------------------------------------------------------------
# Discovery + source index
# ---------------------------------------------------------------------------

def list_cache_files(cache_root: str) -> list[str]:
    out: list[str] = []
    for root, _dirs, files in os.walk(cache_root):
        for fn in files:
            if fn.endswith(".h5"):
                out.append(os.path.join(root, fn))
    return sorted(out)


def index_merged_root(merged_root: str) -> dict[str, str]:
    """basename → absolute path for every .h5 under merged_root.

    The cache's `source_h5` attr is a basename; this lets us find the file
    regardless of the merged tree's hash-dir layout.
    """
    idx: dict[str, str] = {}
    for root, _dirs, files in os.walk(merged_root):
        for fn in files:
            if fn.endswith(".h5"):
                idx.setdefault(fn, os.path.join(root, fn))
    return idx


# ---------------------------------------------------------------------------
# Per-file worker
# ---------------------------------------------------------------------------

def _source_basename(f: h5py.File) -> str:
    v = f.attrs.get("source_h5", None)
    if v is None:
        raise ValueError("missing attrs['source_h5'] — not a v2 cache file?")
    if isinstance(v, bytes):
        v = v.decode()
    return os.path.basename(str(v))


def process_inplace(path: str, merged_index: dict[str, str],
                    force: bool = False) -> dict:
    try:
        with h5py.File(path, "r+") as f:
            if MCKP_PATH in f and not force:
                return {"path": path, "status": "skip-existing", "n_kp": 0}
            base = _source_basename(f)
            src = merged_index.get(base)
            if src is None:
                return {"path": path, "status": "error",
                        "error": f"source merged_h5 not found for {base!r}"}
            n_kp = copy_mckeypoints_group(src, f, overwrite=force)
            if n_kp < 0:
                return {"path": path, "status": "error",
                        "error": f"source {base!r} has no mckeypoints group"}
            f.attrs["has_mckeypoints"] = True
            return {"path": path, "status": "augmented", "n_kp": n_kp}
    except (OSError, KeyError, ValueError) as e:
        return {"path": path, "status": "error",
                "error": f"{type(e).__name__}: {e}"}


def process_copy(src_path: str, src_root: str, dst_root: str,
                 merged_index: dict[str, str], force: bool = False) -> dict:
    rel = os.path.relpath(src_path, src_root)
    dst_path = os.path.join(dst_root, rel)
    if os.path.exists(dst_path) and not force:
        try:
            with h5py.File(dst_path, "r") as f:
                if MCKP_PATH in f:
                    return {"path": dst_path, "status": "skip-existing-dst",
                            "n_kp": 0}
        except OSError:
            pass
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    shutil.copy2(src_path, dst_path)
    return process_inplace(dst_path, merged_index, force=True)


def _worker(task: tuple) -> dict:
    src_path, src_root, dst_root, merged_index, force = task
    if dst_root is None:
        return process_inplace(src_path, merged_index, force=force)
    return process_copy(src_path, src_root, dst_root, merged_index, force=force)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--cache-root", required=True,
                   help="Stage-1+2 cache directory tree (recursive *.h5).")
    p.add_argument("--merged-root", required=True,
                   help="Root of the source merged_h5 tree (recursive); "
                        "indexed by basename to resolve `source_h5`.")
    p.add_argument("--output-dir", default=None,
                   help="If set, copy → augment under this tree. Default: "
                        "in-place.")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--force", action="store_true",
                   help="Re-copy mckeypoints even if already present.")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--progress-every", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    cache_root = os.path.abspath(args.cache_root)
    merged_root = os.path.abspath(args.merged_root)
    if not os.path.isdir(cache_root):
        print(f"ERROR: --cache-root missing: {cache_root}", file=sys.stderr)
        sys.exit(2)
    if not os.path.isdir(merged_root):
        print(f"ERROR: --merged-root missing: {merged_root}", file=sys.stderr)
        sys.exit(2)
    dst_root = os.path.abspath(args.output_dir) if args.output_dir else None

    print("=" * 70)
    print(f"Cache root:   {cache_root}")
    print(f"Merged root:  {merged_root}")
    print(f"Mode:         {'COPY → ' + dst_root if dst_root else 'IN-PLACE'}")
    print(f"Workers:      {args.workers}   Force: {args.force}")
    print("=" * 70)

    print("Indexing merged_h5 by basename ...")
    merged_index = index_merged_root(merged_root)
    print(f"  {len(merged_index)} source file(s) indexed.")

    print("Discovering cache files ...")
    files = list_cache_files(cache_root)
    print(f"  Found {len(files)} cache .h5 file(s).")
    if args.limit is not None:
        files = files[:args.limit]
        print(f"  Limiting to {len(files)} (--limit).")
    if not files:
        print("Nothing to do.")
        return

    if args.dry_run:
        print("\nDry run — first 30 files + resolved source:")
        for fp in files[:30]:
            try:
                with h5py.File(fp, "r") as f:
                    base = _source_basename(f)
                found = "FOUND" if base in merged_index else "MISSING"
                print(f"  {fp}\n      → {base}  [{found}]")
            except Exception as e:
                print(f"  {fp}\n      → ERROR {e}")
        return

    report_every = args.progress_every or max(1, len(files) // 100)
    print(f"\nProcessing with {args.workers} worker(s) ...")
    t0 = time.perf_counter()
    tasks = [(fp, cache_root, dst_root, merged_index, args.force)
             for fp in files]

    n_aug = n_skip = n_err = 0
    n_kp_total = 0
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
                n_kp_total += result["n_kp"]
                if args.verbose:
                    print(f"  + {result['path']}  n_kp={result['n_kp']}")
            elif status.startswith("skip"):
                n_skip += 1
                if args.verbose:
                    print(f"  . {result['path']}  ({status})")
            else:
                n_err += 1
                errors.append((result["path"], result.get("error", "?")))
                print(f"  ! {result['path']}  ERROR: {result.get('error')}")
            if (i + 1) % report_every == 0 or (i + 1) == len(files):
                el = time.perf_counter() - t0
                rate = (i + 1) / max(1e-9, el)
                eta = (len(files) - (i + 1)) / max(1e-9, rate)
                print(f"  [{i+1:6d}/{len(files)}] aug={n_aug} skip={n_skip} "
                      f"err={n_err}  ({rate:6.1f}/s, eta {eta/60:5.1f}min)")

    el = time.perf_counter() - t0
    print("\n" + "=" * 70)
    print(f"Done in {el/60:.2f} min ({len(files)} files)")
    print(f"  augmented:        {n_aug}  (mean {n_kp_total/max(1,n_aug):.1f} kp/file)")
    print(f"  skipped-existing: {n_skip}")
    print(f"  errors:           {n_err}")
    if errors:
        print(f"\nFirst {min(10, len(errors))} errors:")
        for path, msg in errors[:10]:
            print(f"  {path}\n      → {msg}")
    print("=" * 70)
    if n_err > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
