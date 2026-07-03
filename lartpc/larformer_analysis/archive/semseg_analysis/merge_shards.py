"""Sum any number of semseg shard .npz files into one combined .npz.

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

    confusion_total = None
    score_hist_total = None
    per_event_rows = []
    file_list_total = []
    score_edges = None
    n_bins = None
    class_names = None
    origin_names = None
    grid_size = None
    coord_scale = None
    mode = None
    skip_ghosts = None

    for i, p_ in enumerate(paths):
        z = np.load(p_, allow_pickle=False)
        if "confusion" not in z.files:
            print(f"  [{i+1}/{len(paths)}] {os.path.basename(p_)}  "
                  f"(no confusion; skipping — likely a sidecar)")
            continue
        if confusion_total is None:
            confusion_total = np.zeros_like(z["confusion"])
            score_hist_total = np.zeros_like(z["score_hist"])
            score_edges = z["score_edges"]
            n_bins = int(z["n_bins"])
            class_names = z["class_names"]
            origin_names = z["origin_names"]
            grid_size = z["grid_size"] if "grid_size" in z.files else None
            coord_scale = z["coord_scale"] if "coord_scale" in z.files else None
            mode = z["mode"] if "mode" in z.files else None
            skip_ghosts = z["skip_ghosts"] if "skip_ghosts" in z.files else None
        else:
            assert z["confusion"].shape == confusion_total.shape, (
                f"shape mismatch in {p_}: {z['confusion'].shape} "
                f"vs {confusion_total.shape}"
            )
            assert z["score_hist"].shape == score_hist_total.shape
            assert int(z["n_bins"]) == n_bins
            assert list(z["class_names"]) == list(class_names)
            assert list(z["origin_names"]) == list(origin_names)
        confusion_total += z["confusion"]
        score_hist_total += z["score_hist"]
        if z["per_event"].size:
            per_event_rows.append(z["per_event"])
        if z["file_list"].size:
            file_list_total.extend(list(z["file_list"]))
        print(f"  [{i+1}/{len(paths)}] {os.path.basename(p_)}  "
              f"+{int(z['confusion'].sum())} cm-counts  "
              f"+{int(z['score_hist'].sum())} score-counts")

    per_event = (np.concatenate(per_event_rows, axis=0)
                 if per_event_rows else np.zeros(0, dtype=z["per_event"].dtype))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".",
                exist_ok=True)
    payload = dict(
        confusion=confusion_total,
        score_hist=score_hist_total,
        score_edges=score_edges,
        n_bins=np.int64(n_bins),
        class_names=class_names,
        origin_names=origin_names,
        per_event=per_event,
        file_list=np.array(file_list_total),
    )
    if grid_size is not None:
        payload["grid_size"] = grid_size
    if coord_scale is not None:
        payload["coord_scale"] = coord_scale
    if mode is not None:
        payload["mode"] = mode
    if skip_ghosts is not None:
        payload["skip_ghosts"] = skip_ghosts
    np.savez_compressed(args.output, **payload)
    print(f"[merge] wrote {args.output}")
    print(f"  events: {len(per_event)}  cm-counts: {int(confusion_total.sum()):,}")


if __name__ == "__main__":
    main()
