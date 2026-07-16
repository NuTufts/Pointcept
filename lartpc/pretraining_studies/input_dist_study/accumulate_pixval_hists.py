#!/usr/bin/env python3
"""
Stage 1 (map) of the input-distribution study (P05F): accumulate per-class
histograms of RAW per-plane pixel values from SimChTripletLabelMaker HDF5
files, for evaluating input-scaling alternatives to the current LogTransform.

Self-contained on purpose: h5py + numpy only (no torch / pointcept import),
so it runs on any CPU node. Reads the raw `pixval` BEFORE the training
pipeline's clip(0,1000) so saturation loss can be quantified.

Ghost points (hasmatch==0) are dropped, matching the Phase 0.5 training
configuration (true_points_only=True).

Usage (one shard of an array job):
  python3 accumulate_pixval_hists.py --filelist <list.txt> \
      --num-shards 200 --shard $SLURM_ARRAY_TASK_ID --outdir <dir>

Output: <outdir>/pixval_hists_shard{shard:04d}.npz
"""
import argparse
import os
import sys
import time

import h5py
import numpy as np

# SSNet label -> class index, copied from
# pointcept/datasets/lartpc.py::LArTPCDataset.SSNETLABEL_TO_CLASS
# (raw ssnet codes: 0 bg/ghost, 1 electron, 2 photon, 3 muon, 4 proton,
#  5 pion, 6 michel, 7 delta, 8 led, 9 other)
SSNET_TO_CLASS = {0: 8, 1: 0, 2: 4, 3: 1, 4: 3, 5: 2, 6: 5, 7: 6, 8: 7, 9: 9}
CLASS_NAMES = ["electron", "muon", "pion", "proton",
               "gamma", "michel", "delta", "led"]
N_CLASSES = len(CLASS_NAMES)  # classes 8 (ghost) / 9 (other) are not histogrammed

CHANNELS = ["u", "v", "y", "sum", "max"]  # per-plane + plane-reduced variants

# Log-spaced bins over the full raw range; underflow bin catches x <= min edge
# (including exact zeros), overflow catches x >= max edge.
BIN_MIN, BIN_MAX, N_BINS = 1e-2, 2e4, 1500
TRAIN_CLIP = 1000.0  # the training pipeline's np.clip upper bound


def make_edges():
    return np.logspace(np.log10(BIN_MIN), np.log10(BIN_MAX), N_BINS + 1)


def build_ssnet_lut():
    lut = np.full(256, -1, dtype=np.int64)
    for raw, cls in SSNET_TO_CLASS.items():
        lut[raw] = cls
    return lut


def process_file(path, edges, lut, hists, clip_counts, class_counts):
    with h5py.File(path, "r") as f:
        td = f["/entry_0/triplet_data"]
        hasmatch = np.asarray(td["hasmatch"])
        real = hasmatch == 1
        if not real.any():
            return 0
        pixval = np.asarray(td["pixval"], dtype=np.float64)[real]
        ssnet = np.asarray(td["ssnet_label"], dtype=np.int64)[real]
    cls = lut[np.clip(ssnet, 0, 255)]
    keep = (cls >= 0) & (cls < N_CLASSES)
    if not keep.any():
        return 0
    pixval, cls = pixval[keep], cls[keep]

    chans = {
        "u": pixval[:, 0], "v": pixval[:, 1], "y": pixval[:, 2],
        "sum": pixval.sum(axis=1), "max": pixval.max(axis=1),
    }
    for c in range(N_CLASSES):
        m = cls == c
        if not m.any():
            continue
        class_counts[c] += int(m.sum())
        for ci, ch in enumerate(CHANNELS):
            x = chans[ch][m]
            hist, _ = np.histogram(x, bins=edges)
            hists[c, ci, 1:-1] += hist
            hists[c, ci, 0] += int((x <= edges[0]).sum())
            hists[c, ci, -1] += int((x >= edges[-1]).sum())
            if ch in ("u", "v", "y"):
                clip_counts[c, ci] += int((x > TRAIN_CLIP).sum())
    return pixval.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filelist", required=True)
    ap.add_argument("--num-shards", type=int, required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--max-files", type=int, default=None,
                    help="cap files processed in this shard (debug)")
    args = ap.parse_args()

    with open(args.filelist) as f:
        paths = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    shard_paths = paths[args.shard::args.num_shards]
    if args.max_files:
        shard_paths = shard_paths[:args.max_files]

    edges = make_edges()
    lut = build_ssnet_lut()
    # +2 bins: [0]=underflow, [-1]=overflow
    hists = np.zeros((N_CLASSES, len(CHANNELS), N_BINS + 2), dtype=np.int64)
    clip_counts = np.zeros((N_CLASSES, 3), dtype=np.int64)  # u,v,y only
    class_counts = np.zeros(N_CLASSES, dtype=np.int64)

    t0, n_points, n_bad = time.time(), 0, 0
    for i, path in enumerate(shard_paths):
        try:
            n_points += process_file(path, edges, lut, hists,
                                     clip_counts, class_counts)
        except Exception as e:
            n_bad += 1
            print(f"WARN skipping {path}: {e}", file=sys.stderr)
        if (i + 1) % 200 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"[shard {args.shard}] {i+1}/{len(shard_paths)} files "
                  f"({rate:.1f} files/s, {n_points:,} points)", flush=True)

    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, f"pixval_hists_shard{args.shard:04d}.npz")
    np.savez_compressed(
        out,
        hists=hists, clip_counts=clip_counts, class_counts=class_counts,
        bin_edges=edges, channels=np.array(CHANNELS),
        class_names=np.array(CLASS_NAMES),
        n_files=len(shard_paths), n_bad_files=n_bad, n_points=n_points,
        filelist=args.filelist, train_clip=TRAIN_CLIP,
    )
    print(f"[shard {args.shard}] done: {len(shard_paths)} files "
          f"({n_bad} bad), {n_points:,} true points -> {out}")


if __name__ == "__main__":
    main()
