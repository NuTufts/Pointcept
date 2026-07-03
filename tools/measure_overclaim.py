"""Per-pair over-claim summary across slicer prediction HDF5s.

Reads `slicerpred_*.h5` files produced by `tools/larformer/run_slicer_inference.py`
and reports, for each matched (query, GT) pair:

  - `pair_iou`   = IoU of the matched query's FULL above-threshold mask
                   vs. the GT slice. Penalizes Q for claiming high mask
                   probability on SPs that are not in K (regardless of
                   whether those SPs end up assigned to Q in panoptic
                   argmax).
  - `argmax_iou` = IoU restricted to SPs Q *wins* in panoptic argmax (the
                   prediction-panel view). Hides Q's over-claims that
                   were stolen by other queries.
  - `pred_n / gt_n` ratio = pred-mask size over GT-slice size. Values >>1
                   indicate over-claim.
  - `over_claim_gap = pair_iou - argmax_iou` (typically <= 0 — over-claim
                   pulls pair_iou DOWN relative to argmax_iou's restricted
                   view). Magnitude measures over-claim severity.

Aggregates these across an inference output directory. Prints summary
statistics, a per-class breakdown (nu / cosmic), and a worst-N event
leaderboard.

Requires the inference HDF5 to carry the v2 fields:
  gt/pair_iou, gt/argmax_iou, gt/pred_n_pts, gt/n_truth_points,
  gt/primary_trackid, gt/origin_type, gt/matched_query

(Re-run `tools/larformer/run_slicer_inference.py` with the current script version
if your HDF5s are missing `gt/argmax_iou` or `gt/pred_n_pts`.)

Usage:
  python tools/measure_overclaim.py \\
      --slicerpred-dir exp/<run>/inference_outputs/ \\
      [--top-n 10]

Reports:
  - per-pair count, pair_iou / argmax_iou / pred/gt ratio mean and median
  - per-class (nu / cosmic) breakdown
  - distribution of `gap = pair_iou - argmax_iou` (mean, median, p10/p90)
  - top-N worst events by mean over_claim_gap magnitude
"""

import argparse
import glob
import os
import sys

import h5py
import numpy as np


# Class id → name. Matches the cascaded-loradeghost / ptv3hybrid config's
# `slice_class_map={1: 0, 2: 1}` then `names=["nu", "cosmic", "no_object"]`.
# So origin_type after remap: 0 = nu, 1 = cosmic, 2 = no_object.
_CLASS_NAMES = {0: "nu", 1: "cosmic", 2: "no_object"}


def _summarize(values: np.ndarray, label: str, fmt: str = "{:.3f}") -> str:
    """Return one summary line: 'label  n=N  mean=…  med=…  p10=…  p90=…'."""
    if values.size == 0:
        return f"  {label:<22s} n=0  (no data)"
    return (
        f"  {label:<22s} n={values.size:<5d}  "
        f"mean=" + fmt.format(float(values.mean())) + "  "
        f"med="  + fmt.format(float(np.median(values))) + "  "
        f"p10="  + fmt.format(float(np.percentile(values, 10))) + "  "
        f"p90="  + fmt.format(float(np.percentile(values, 90)))
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("--slicerpred-dir", required=True,
                    help="Directory containing slicerpred_*.h5 files.")
    ap.add_argument("--glob", default="slicerpred_*.h5",
                    help="Glob within --slicerpred-dir (default: %(default)s)")
    ap.add_argument("--top-n", type=int, default=10,
                    help="Show the N worst events by mean over-claim gap "
                         "(default: %(default)s; 0 to suppress)")
    ap.add_argument("--ratio-cap", type=float, default=20.0,
                    help="Cap pred_n/gt_n ratio at this value for the "
                         "distribution stats (extreme outliers — e.g. tiny "
                         "GT slices with huge over-claim — otherwise pull "
                         "the mean way up). Default 20.")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.slicerpred_dir, args.glob)))
    if not paths:
        sys.exit(f"No files matched {args.slicerpred_dir}/{args.glob}")
    print(f"Found {len(paths)} prediction files.")

    # Per-event aggregates for the leaderboard.
    per_event_records = []
    # Per-pair accumulators (flat across all events).
    all_pair_iou = []
    all_argmax_iou = []
    all_gap = []
    all_ratio = []
    all_ot = []
    n_events_with_pairs = 0
    n_skipped_missing_field = 0

    for p in paths:
        with h5py.File(p, "r") as f:
            if int(f.attrs.get("event_dropped", 0)):
                continue
            # All four diagnostic fields must be present.
            need = ("gt/pair_iou", "gt/argmax_iou",
                    "gt/pred_n_pts", "gt/n_truth_points",
                    "gt/origin_type")
            if any(k not in f for k in need):
                n_skipped_missing_field += 1
                continue
            pair_iou = f["gt/pair_iou"][()]
            argmax_iou = f["gt/argmax_iou"][()]
            pred_n = f["gt/pred_n_pts"][()]
            gt_n = f["gt/n_truth_points"][()]
            ot = f["gt/origin_type"][()]
            run = int(f.attrs.get("meta_run", -1))
            sub = int(f.attrs.get("meta_subrun", -1))
            evn = int(f.attrs.get("meta_event", -1))

            # Keep only matched pairs (where pair_iou >= 0 indicates the
            # GT had a Hungarian match).
            matched = (pair_iou >= 0) & (argmax_iou >= 0) & (pred_n >= 0)
            if not matched.any():
                continue
            pi = pair_iou[matched]
            ai = argmax_iou[matched]
            pn = pred_n[matched].astype(np.float64)
            gn = gt_n[matched].astype(np.float64)
            ot_m = ot[matched]
            ratio = pn / np.maximum(gn, 1.0)
            gap = pi - ai

            all_pair_iou.append(pi)
            all_argmax_iou.append(ai)
            all_gap.append(gap)
            all_ratio.append(ratio)
            all_ot.append(ot_m)
            n_events_with_pairs += 1

            per_event_records.append(dict(
                path=os.path.basename(p),
                run=run, subrun=sub, event=evn,
                n_pairs=int(matched.sum()),
                mean_gap=float(gap.mean()),
                mean_pair_iou=float(pi.mean()),
                mean_argmax_iou=float(ai.mean()),
                mean_ratio=float(np.minimum(ratio, args.ratio_cap).mean()),
            ))

    if n_skipped_missing_field:
        print(f"  skipped {n_skipped_missing_field} files (missing v2 "
              f"diagnostic fields — re-run inference to regenerate)")
    if not all_pair_iou:
        sys.exit("No HDF5s had both pair_iou and argmax_iou. Re-run "
                 "tools/larformer/run_slicer_inference.py with the latest script "
                 "version.")

    pair_iou = np.concatenate(all_pair_iou)
    argmax_iou = np.concatenate(all_argmax_iou)
    gap = np.concatenate(all_gap)
    ratio = np.concatenate(all_ratio)
    ot = np.concatenate(all_ot)
    # Cap ratio for distribution stats.
    ratio_capped = np.minimum(ratio, args.ratio_cap)

    print(f"\nAggregated {pair_iou.size} matched pairs across "
          f"{n_events_with_pairs} events.\n")

    # ----- overall summary --------------------------------------------
    print("=== Overall distributions (across all matched pairs) ===")
    print(_summarize(pair_iou,       "pair_iou"))
    print(_summarize(argmax_iou,     "argmax_iou"))
    print(_summarize(gap,            "gap = pair-argmax"))
    print(_summarize(ratio_capped,   f"pred_n / gt_n (cap {args.ratio_cap:.0f})"))
    n_overclaim = int((ratio > 1.5).sum())
    n_undercov  = int((ratio < 0.5).sum())
    print(f"  fraction(pred_n / gt_n > 1.5): {n_overclaim / pair_iou.size:.2%} "
          f"({n_overclaim} pairs over-claim by >50%)")
    print(f"  fraction(pred_n / gt_n < 0.5): {n_undercov / pair_iou.size:.2%} "
          f"({n_undercov} pairs under-cover by >50%)")
    n_large_gap = int((gap < -0.1).sum())
    print(f"  fraction(gap < -0.1):          {n_large_gap / pair_iou.size:.2%} "
          f"({n_large_gap} pairs where over-claim costs >0.10 IoU)")

    # ----- per-class summary ------------------------------------------
    print("\n=== Per-class breakdown ===")
    for c, name in sorted(_CLASS_NAMES.items()):
        mask = (ot == c)
        if not mask.any():
            continue
        print(f"  class={name} (origin_type={c}):  n_pairs={int(mask.sum())}")
        print(_summarize(pair_iou[mask],     "  pair_iou"))
        print(_summarize(argmax_iou[mask],   "  argmax_iou"))
        print(_summarize(gap[mask],          "  gap"))
        print(_summarize(ratio_capped[mask], "  pred_n / gt_n"))

    # ----- worst events leaderboard -----------------------------------
    if args.top_n > 0 and per_event_records:
        print(f"\n=== Top {min(args.top_n, len(per_event_records))} events "
              f"by mean |gap| (most over-claim) ===")
        # Most negative mean_gap = most over-claim (gap = pair - argmax;
        # over-claim makes pair < argmax → gap < 0).
        ranked = sorted(per_event_records, key=lambda r: r["mean_gap"])
        print(f"  {'gap':>7s}  {'pair':>5s}  {'argm':>5s}  {'ratio':>5s}  "
              f"{'pairs':>5s}  run/sub/event  file")
        for rec in ranked[:args.top_n]:
            print(f"  {rec['mean_gap']:>+7.3f}  "
                  f"{rec['mean_pair_iou']:>5.3f}  "
                  f"{rec['mean_argmax_iou']:>5.3f}  "
                  f"{rec['mean_ratio']:>5.2f}  "
                  f"{rec['n_pairs']:>5d}  "
                  f"{rec['run']:>3d}/{rec['subrun']:>3d}/{rec['event']:<6d} "
                  f"{rec['path']}")


if __name__ == "__main__":
    main()
