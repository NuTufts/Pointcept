#!/usr/bin/env python3
"""
Smoke-test that the combined SquashFS dataset image, bound at /data inside
the pointcept container, can be read end-to-end by the LArTPCDataset /
LArFormerDataset code paths the actual training jobs use.

Designed to exercise the FUSE-mount read path, NOT to validate model code.
Reports per-item / per-batch tensor shapes + dtypes + simple stats, plus
aggregate samples/sec and MiB/sec. The interesting experiment for FUSE
scaling is varying --num-workers and watching where samples/sec plateaus.

Usage (inside the apptainer container, see test_dataset_read.sh wrapper):
    python isambard/test_dataset_read.py \\
        --config-file configs/lartpc/sonata_pretrain/pretrain-sonata-v7-extbnb-larmatch.py \\
        --data-list-file /projects/u6jo/datasets/test_list_isambard.txt \\
        --data-root /data \\
        --split train \\
        --num-batches 8 \\
        --batch-size 2 \\
        --num-workers 0 \\
        [--apply-transforms]

The --data-list-file must contain paths that resolve inside the container.
After binding combined.sqsh:/data, expect lines like
    /data/ub_on_tufts/hdf5/v3_ext_larmatch/extbnb_run3_G1/000/000/foo.h5
"""

import argparse
import os
import sys
import time

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from pointcept.engines.defaults import default_config_parser
from pointcept.datasets.builder import build_dataset
from pointcept.datasets.utils import point_collate_fn


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config-file", required=True)
    p.add_argument("--data-list-file", required=True,
                   help="Text file with one HDF5 path per line. Paths must "
                        "resolve from inside the container (e.g. /data/...).")
    p.add_argument("--data-root", default="/data",
                   help="Path prefix for relative entries in --data-list-file.")
    p.add_argument("--split", choices=["train", "val", "test"], default="train")
    p.add_argument("--num-batches", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--apply-transforms", action="store_true",
                   help="Apply the config's transform pipeline. Default off.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def log(msg):
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def stat_array(name, arr):
    """One-line summary string for a numpy or torch array."""
    if isinstance(arr, torch.Tensor):
        shape = tuple(arr.shape)
        dtype = str(arr.dtype).replace("torch.", "")
        if arr.numel() == 0:
            return f"  {name}: shape={shape} dtype={dtype} (empty)"
        if arr.dtype.is_floating_point:
            stats = (f" min={arr.min().item():.4g}"
                     f" max={arr.max().item():.4g}"
                     f" mean={arr.float().mean().item():.4g}")
        else:
            try:
                stats = (f" min={arr.min().item()}"
                         f" max={arr.max().item()}"
                         f" uniq={len(arr.unique())}")
            except Exception:
                stats = ""
        return f"  {name}: shape={shape} dtype={dtype}{stats}"
    if isinstance(arr, np.ndarray):
        if arr.size == 0:
            return f"  {name}: shape={arr.shape} dtype={arr.dtype} (empty)"
        if np.issubdtype(arr.dtype, np.floating):
            stats = (f" min={arr.min():.4g}"
                     f" max={arr.max():.4g}"
                     f" mean={arr.mean():.4g}")
        elif np.issubdtype(arr.dtype, np.number):
            stats = (f" min={arr.min()}"
                     f" max={arr.max()}"
                     f" uniq={len(np.unique(arr))}")
        else:
            stats = ""
        return f"  {name}: shape={arr.shape} dtype={arr.dtype}{stats}"
    if isinstance(arr, (list, tuple)):
        return f"  {name}: type={type(arr).__name__} len={len(arr)}"
    return f"  {name}: type={type(arr).__name__} value={arr!r}"


def array_nbytes(v):
    if isinstance(v, torch.Tensor):
        return v.element_size() * v.numel()
    if isinstance(v, np.ndarray):
        return v.nbytes
    return 0


def sniff_h5_attrs(path):
    """Verify entry_0 + (run, subrun, event) attrs present. Returns a string."""
    try:
        with h5py.File(path, "r") as f:
            if "entry_0" not in f:
                return f"  WARNING: 'entry_0' not in {path}"
            entry = f["entry_0"]
            attrs = dict(entry.attrs)
            picked = {k: attrs[k] for k in ("run", "subrun", "event") if k in attrs}
            return (f"  entry_0 attrs (run/subrun/event): {picked}\n"
                    f"  entry_0 all attrs: {sorted(attrs.keys())}\n"
                    f"  entry_0 keys: {list(entry.keys())[:10]}"
                    f"{' ...' if len(entry.keys()) > 10 else ''}")
    except Exception as e:
        return f"  ERROR opening {path}: {type(e).__name__}: {e}"


def main():
    args = parse_args()
    log("starting test_dataset_read")
    print(f"  config       : {args.config_file}")
    print(f"  data_list    : {args.data_list_file}")
    print(f"  data_root    : {args.data_root}")
    print(f"  split        : {args.split}")
    print(f"  num_batches  : {args.num_batches}")
    print(f"  batch_size   : {args.batch_size}")
    print(f"  num_workers  : {args.num_workers}")
    print(f"  apply_xforms : {args.apply_transforms}")
    sys.stdout.flush()

    # === Preflight: list file + first-file open ===
    if not os.path.isfile(args.data_list_file):
        sys.exit(f"ERROR: --data-list-file not found: {args.data_list_file}")
    with open(args.data_list_file) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    print(f"  list lines   : {len(lines)}")
    if not lines:
        sys.exit("ERROR: data list is empty.")

    sample = lines[0]
    resolved = sample if os.path.isabs(sample) else os.path.join(args.data_root, sample)
    print(f"  first listed : {sample}")
    print(f"  resolved to  : {resolved}")
    if not os.path.isfile(resolved):
        sys.exit(f"ERROR: first listed file does not exist. Bind/mount problem?\n"
                 f"  expected at: {resolved}")
    print(sniff_h5_attrs(resolved))
    sys.stdout.flush()

    # === Load config ===
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    log("loading config")
    cfg = default_config_parser(args.config_file, options=None)

    if not hasattr(cfg.data, args.split):
        sys.exit(f"ERROR: cfg.data has no '{args.split}' block "
                 f"(available: {list(cfg.data.keys())})")

    ds_cfg = dict(getattr(cfg.data, args.split))
    ds_cfg["data_list_file"] = args.data_list_file
    ds_cfg["data_root"] = args.data_root
    if not args.apply_transforms:
        ds_cfg["transform"] = []
    print(f"  dataset type : {ds_cfg.get('type')}")
    print(f"  transforms   : {'config pipeline' if args.apply_transforms else 'disabled (empty list)'}")
    sys.stdout.flush()

    # === Build dataset ===
    t0 = time.perf_counter()
    ds = build_dataset(ds_cfg)
    log(f"build_dataset done in {time.perf_counter()-t0:.2f}s, len={len(ds)}")
    if len(ds) == 0:
        sys.exit("ERROR: dataset is empty after build. data_list_file resolution problem?")

    # === Pull batches via DataLoader (num_workers=0 = in-process) ===
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=point_collate_fn,
        pin_memory=False,
        persistent_workers=False,
    )
    log(f"reading {args.num_batches} batches via DataLoader "
        f"(num_workers={args.num_workers}, batch_size={args.batch_size})")
    sys.stdout.flush()

    bytes_total = 0
    per_batch_times = []
    t_loop_start = time.perf_counter()
    t_prev = t_loop_start
    for i, batch in enumerate(loader):
        if i >= args.num_batches:
            break
        t_now = time.perf_counter()
        per_batch_times.append(t_now - t_prev)
        t_prev = t_now

        print(f"\n--- batch {i}  ({per_batch_times[-1]*1000:.1f} ms) ---")
        if isinstance(batch, dict):
            for k, v in batch.items():
                print(stat_array(k, v))
                bytes_total += array_nbytes(v)
        else:
            print(f"  batch type: {type(batch).__name__}")
        sys.stdout.flush()

    t_loop_total = time.perf_counter() - t_loop_start
    n = len(per_batch_times)

    print("\n" + "=" * 60)
    log("SUMMARY")
    if n == 0:
        print("  no batches read (dataset exhausted before --num-batches)")
        return
    samples = n * args.batch_size
    mib = bytes_total / (1024 ** 2)
    print(f"  batches read         : {n}")
    print(f"  samples              : {samples}")
    print(f"  wall time (load loop): {t_loop_total:.2f} s")
    print(f"  samples / sec        : {samples/t_loop_total:.2f}")
    print(f"  approx MiB read      : {mib:.1f}")
    print(f"  approx MiB / sec     : {mib/t_loop_total:.2f}")
    print(f"  per-batch times (ms) : {[f'{t*1000:.0f}' for t in per_batch_times]}")
    print(f"  per-batch median (ms): {np.median(per_batch_times)*1000:.1f}")
    print(f"  per-batch p95 (ms)   : {np.percentile(per_batch_times, 95)*1000:.1f}")
    log("done.")


if __name__ == "__main__":
    main()
