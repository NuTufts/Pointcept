"""Sum any number of shard .npz files into one combined .npz.

Usage:
    python merge_shards.py \
        --shards outputs/shard_*.npz \
        --output outputs/combined.npz
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shards", nargs="+", required=True,
                   help="Shard .npz paths or globs.")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    paths = []
    for pat in args.shards:
        expanded = sorted(glob.glob(pat))
        if not expanded and os.path.exists(pat):
            expanded = [pat]
        paths.extend(expanded)
    if not paths:
        print("[merge] no shard files matched", file=sys.stderr)
        sys.exit(1)
    print(f"[merge] {len(paths)} shard(s) to combine")

    hist_real_total = None
    hist_ghost_total = None
    per_event_rows = []
    file_list_total = []
    score_edges = None
    n_bins = None
    predictor_names = None
    stratum_names = None
    grid_size = None
    coord_scale = None

    for i, p_ in enumerate(paths):
        z = np.load(p_, allow_pickle=False)
        if "hist_real" not in z.files:
            # Robust to picking up per-event score sidecars or other .npz in the dir.
            print(f"  [{i+1}/{len(paths)}] {os.path.basename(p_)}  "
                  f"(no hist_real; skipping — likely a sidecar)")
            continue
        if hist_real_total is None:
            hist_real_total = np.zeros_like(z["hist_real"])
            hist_ghost_total = np.zeros_like(z["hist_ghost"])
            score_edges = z["score_edges"]
            n_bins = int(z["n_bins"])
            predictor_names = z["predictor_names"]
            stratum_names = z["stratum_names"]
            grid_size = z["grid_size"] if "grid_size" in z.files else None
            coord_scale = z["coord_scale"] if "coord_scale" in z.files else None
        else:
            assert z["hist_real"].shape == hist_real_total.shape, \
                f"shape mismatch in {p_}: {z['hist_real'].shape} vs {hist_real_total.shape}"
            assert int(z["n_bins"]) == n_bins
            assert list(z["predictor_names"]) == list(predictor_names)
            assert list(z["stratum_names"]) == list(stratum_names)
        hist_real_total += z["hist_real"]
        hist_ghost_total += z["hist_ghost"]
        if z["per_event"].size:
            per_event_rows.append(z["per_event"])
        if z["file_list"].size:
            file_list_total.extend(list(z["file_list"]))
        print(f"  [{i+1}/{len(paths)}] {os.path.basename(p_)}  "
              f"+{int(z['hist_real'].sum() + z['hist_ghost'].sum())} counts")

    per_event = (np.concatenate(per_event_rows, axis=0)
                 if per_event_rows else np.zeros(0, dtype=z["per_event"].dtype))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".",
                exist_ok=True)
    payload = dict(
        hist_real=hist_real_total,
        hist_ghost=hist_ghost_total,
        score_edges=score_edges,
        n_bins=np.int64(n_bins),
        predictor_names=predictor_names,
        stratum_names=stratum_names,
        per_event=per_event,
        file_list=np.array(file_list_total),
    )
    if grid_size is not None:
        payload["grid_size"] = grid_size
    if coord_scale is not None:
        payload["coord_scale"] = coord_scale
    np.savez_compressed(args.output, **payload)
    print(f"[merge] wrote {args.output}")
    print(f"  events: {len(per_event)}  spacepoints: "
          f"real={int(hist_real_total[0, 0].sum()):,}  "
          f"ghost={int(hist_ghost_total[0].sum()):,}")


if __name__ == "__main__":
    main()
