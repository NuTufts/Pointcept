"""Per-level merger-rate aggregator for slicer predictions.

Reads the `slicerpred_*.h5` files produced by `tools/run_slicer_inference.py`
and reports, at each emitted level (spacepoint + every voxel level), the
fraction of predicted slices that over-cluster — i.e. claim a meaningful
fraction of tokens from more than one GT slice.

Definition (per event, per level):
  - Group tokens by `pred_query` (the per-token argmax over active queries
    at that level). Empty groups and groups with `n_tokens < MIN_TOKENS`
    are skipped.
  - For each remaining group, count tokens by their GT `slice_id` (drop
    GT < 0 = ghost / no-slice).
  - A GT slice is a "claimed" component of this prediction if it covers
    >= MIN_FRACTION of the prediction's tokens.
  - n_claimed >= 2  →  this prediction is a merger.

Per-level aggregate:
    merger_rate  =  n_merger_predictions / n_qualifying_predictions
    mean_components = average number of claimed GT slices per prediction
    overcluster_mass = (tokens not in the dominant GT slice) / (total tokens)
                      summed across all merger predictions / (total tokens)

Run:
  python tools/measure_merger_rates.py \
      --slicerpred-dir exp/<run>/inference_outputs/ \
      [--min-fraction 0.20] [--min-tokens 20] [--verbose-worst 10]

The script also prints a per-event leaderboard (the N worst events by
spacepoint-level merger rate) so the visualizer has a starting list of
events to inspect.
"""

import argparse
import glob
import os
import sys
from collections import defaultdict

import h5py
import numpy as np


def _per_event_per_level_merger_stats(
    pred_query: np.ndarray,        # (M,) int
    slice_id_gt: np.ndarray,       # (M,) int (-1 = ghost)
    pred_mask_prob: "np.ndarray | None",  # (M,) float in [0,1], or None
    min_tokens: int,
    min_fraction: float,
    min_mask_prob: float = 0.0,
) -> dict:
    """Compute merger stats for one (event, level).

    If `pred_mask_prob` is provided and `min_mask_prob > 0`, tokens whose
    assigned-query mask probability falls below the threshold are dropped
    from their prediction group before counting (treated as "unassigned").
    This separates "the model is wrong" from "the model is unsure but the
    no-threshold argmax inference rule put it somewhere anyway".
    """
    if pred_query.shape[0] == 0:
        return dict(n_pred=0, n_merger=0, n_components=[],
                    total_tokens=0, overcluster_tokens=0)
    if min_mask_prob > 0.0 and pred_mask_prob is not None:
        keep = pred_mask_prob >= min_mask_prob
        pred_query = pred_query[keep]
        slice_id_gt = slice_id_gt[keep]
        if pred_query.shape[0] == 0:
            return dict(n_pred=0, n_merger=0, n_components=[],
                        total_tokens=0, overcluster_tokens=0)

    n_pred = 0
    n_merger = 0
    components_per_pred = []
    total_tokens = 0
    overcluster_tokens = 0

    # Group tokens by pred_query
    unique_q = np.unique(pred_query)
    for q in unique_q:
        idx = np.where(pred_query == q)[0]
        n = int(idx.size)
        if n < min_tokens:
            continue
        gt_for_pred = slice_id_gt[idx]
        gt_for_pred = gt_for_pred[gt_for_pred >= 0]
        if gt_for_pred.size == 0:
            continue  # prediction covers only ghost SPs / unassigned voxels

        gt_vals, gt_counts = np.unique(gt_for_pred, return_counts=True)
        fractions = gt_counts / n              # vs prediction size, not GT-side
        claimed = fractions >= min_fraction
        n_components = int(claimed.sum())
        if n_components == 0:
            # No single GT slice clears the threshold; treat as "uncertain
            # / many small fragments" and count as a 1-component prediction
            # (consistent with the canonical Mask2Former IoU definition).
            n_components = 1

        n_pred += 1
        components_per_pred.append(n_components)
        total_tokens += n
        if n_components >= 2:
            n_merger += 1
            # Over-cluster mass = tokens not in the dominant claimed GT slice.
            dominant_idx = int(gt_counts.argmax())
            dominant_count = int(gt_counts[dominant_idx])
            overcluster_tokens += (n - dominant_count)

    return dict(
        n_pred=n_pred,
        n_merger=n_merger,
        n_components=components_per_pred,
        total_tokens=total_tokens,
        overcluster_tokens=overcluster_tokens,
    )


def _level_names_in_h5(f: h5py.File) -> list:
    """Return ["spacepoint"] + sorted non-spacepoint level names found in
    the file's `levels/` group (in the order the run_slicer_inference
    script emitted them)."""
    levels = ["spacepoint"]
    if "levels" in f:
        for name in f["levels"].keys():
            levels.append(name)
    return levels


def _load_level_arrays(
    f: h5py.File, level: str,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray | None]":
    """Return (pred_query, slice_id_gt, pred_mask_prob) for a single level.

    `pred_mask_prob` is None when the HDF5 was produced by an older
    inference-script version that didn't emit it.
    """
    if level == "spacepoint":
        pred_query = f["post/pred_query"][()]
        slice_id_gt = f["post/slice_id_gt"][()]
        pmp = (f["post/pred_mask_prob"][()] if "post/pred_mask_prob" in f
               else None)
    else:
        pred_query = f[f"levels/{level}/pred_query"][()]
        slice_id_gt = f[f"levels/{level}/slice_id_gt"][()]
        pmp_key = f"levels/{level}/pred_mask_prob"
        pmp = f[pmp_key][()] if pmp_key in f else None
    return pred_query, slice_id_gt, pmp


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("--slicerpred-dir", required=True,
                    help="Directory containing slicerpred_*.h5 files")
    ap.add_argument("--glob", default="slicerpred_*.h5",
                    help="Glob within --slicerpred-dir (default: %(default)s)")
    ap.add_argument("--min-fraction", type=float, default=0.20,
                    help="Min fraction of a prediction's tokens that a GT "
                         "slice must cover to be considered a claimed "
                         "component (default: %(default)s)")
    ap.add_argument("--min-tokens", type=int, default=20,
                    help="Skip predictions with fewer than this many tokens "
                         "(default: %(default)s)")
    ap.add_argument("--min-mask-prob", type=float, default=0.0,
                    help="Drop tokens whose assigned-query sigmoid mask "
                         "probability is below this threshold before "
                         "counting (default: 0 = use the raw panoptic "
                         "argmax). 0.5 = only confident assignments. Older "
                         "HDF5 files without pred_mask_prob are unaffected.")
    ap.add_argument("--verbose-worst", type=int, default=10,
                    help="Print the N events with the highest spacepoint-"
                         "level merger rate (default: %(default)s; 0 to "
                         "suppress)")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.slicerpred_dir, args.glob)))
    if not paths:
        sys.exit(f"No files matched {args.slicerpred_dir}/{args.glob}")

    print(f"Found {len(paths)} prediction files.")
    print(f"Thresholds: min_fraction={args.min_fraction}  "
          f"min_tokens={args.min_tokens}  "
          f"min_mask_prob={args.min_mask_prob}\n")

    # Aggregates per level
    agg = defaultdict(lambda: dict(
        n_pred=0, n_merger=0, total_tokens=0, overcluster_tokens=0,
        components_per_pred=[],
        n_events=0, n_events_with_any_pred=0,
    ))

    # Per-event spacepoint merger rate (for the worst-N leaderboard)
    per_event_sp = []

    # Discover levels from the first non-empty file
    level_order = None
    for p in paths:
        with h5py.File(p, "r") as f:
            if "event_dropped" in f.attrs and int(f.attrs["event_dropped"]):
                continue
            if "post" in f:
                level_order = _level_names_in_h5(f)
                break
    if level_order is None:
        sys.exit("All input files appear to have event_dropped=1; nothing "
                 "to aggregate.")
    print(f"Levels discovered: {level_order}\n")

    for p in paths:
        with h5py.File(p, "r") as f:
            if int(f.attrs.get("event_dropped", 0)):
                continue
            run = int(f.attrs.get("meta_run", -1))
            sub = int(f.attrs.get("meta_subrun", -1))
            evn = int(f.attrs.get("meta_event", -1))

            event_sp_stats = None
            for level in level_order:
                # Some events might not have all levels (e.g. an event with
                # < min_tokens at the coarsest voxel scale). Skip gracefully.
                try:
                    pq, gt, pmp = _load_level_arrays(f, level)
                except KeyError:
                    continue
                stats = _per_event_per_level_merger_stats(
                    pq, gt, pmp,
                    min_tokens=args.min_tokens,
                    min_fraction=args.min_fraction,
                    min_mask_prob=args.min_mask_prob,
                )
                a = agg[level]
                a["n_pred"] += stats["n_pred"]
                a["n_merger"] += stats["n_merger"]
                a["total_tokens"] += stats["total_tokens"]
                a["overcluster_tokens"] += stats["overcluster_tokens"]
                a["components_per_pred"].extend(stats["n_components"])
                a["n_events"] += 1
                if stats["n_pred"] > 0:
                    a["n_events_with_any_pred"] += 1
                if level == "spacepoint":
                    event_sp_stats = stats
            if event_sp_stats is not None and event_sp_stats["n_pred"] > 0:
                per_event_sp.append(dict(
                    path=os.path.basename(p),
                    run=run, subrun=sub, event=evn,
                    n_pred=event_sp_stats["n_pred"],
                    n_merger=event_sp_stats["n_merger"],
                    merger_rate=(event_sp_stats["n_merger"]
                                 / max(event_sp_stats["n_pred"], 1)),
                ))

    # ----- print summary -----------------------------------------------
    hdr = (f"{'level':<14} {'n_pred':>8} {'n_merger':>9} {'merger_rate':>12} "
           f"{'mean_comp':>10} {'overcl_mass':>12} {'events':>8}")
    print(hdr)
    print("-" * len(hdr))
    for level in level_order:
        a = agg[level]
        if a["n_pred"] == 0:
            print(f"{level:<14} (no qualifying predictions)")
            continue
        merger_rate = a["n_merger"] / a["n_pred"]
        mean_comp = float(np.mean(a["components_per_pred"]))
        overcl_mass = (a["overcluster_tokens"] / a["total_tokens"]
                       if a["total_tokens"] else 0.0)
        print(f"{level:<14} {a['n_pred']:>8d} {a['n_merger']:>9d} "
              f"{merger_rate:>12.3f} {mean_comp:>10.2f} "
              f"{overcl_mass:>12.3f} {a['n_events']:>8d}")

    # ----- worst events leaderboard ------------------------------------
    if args.verbose_worst > 0 and per_event_sp:
        per_event_sp.sort(key=lambda d: (-d["merger_rate"], -d["n_merger"]))
        print(f"\nTop {min(args.verbose_worst, len(per_event_sp))} events by "
              f"spacepoint merger rate:")
        print(f"{'rate':>6} {'merg/pred':>10}  run/sub/event  file")
        for rec in per_event_sp[:args.verbose_worst]:
            print(f"{rec['merger_rate']:>6.3f} "
                  f"{rec['n_merger']:>4d}/{rec['n_pred']:<5d} "
                  f"{rec['run']:>5d}/{rec['subrun']:>3d}/{rec['event']:<6d} "
                  f"{rec['path']}")


if __name__ == "__main__":
    main()
