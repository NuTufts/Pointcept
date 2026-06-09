"""Aggregate per-event LArFormer Stage-3 records into a summary JSON +
parquet bundle reproducing `LArFormerParticleEvaluator`'s scalars.

Walks `<analysis-dir>/perevent_*.h5`, concatenates the per-pair records
into a single table, and reports:

  - val/mask_iou_mean, _median, _p25                 — overall, matched pairs
  - val/mask_iou_<class>                             — per-class mean IoU
  - val/cls_accuracy                                 — argmax-class accuracy on matched pairs
  - val/matched_fraction                             — n_matched / n_gt
  - val/n_active_queries_mean                        — mean per-event active-query count
  - val/origin_l2_cm_mean, _median, _p25, _p75       — overall, matched pairs
  - val/origin_l2_cm_<class>                         — per-class mean

Stress metric (new — not in the in-training evaluator):

  - val/mask_iou_smallest25                          — mean IoU on the
        subset of GT in the bottom 25% by spacepoint count, MATCHED ONLY
        (matches evaluator semantics for mask_iou_mean).
  - val/mask_iou_smallest25_median
  - val/match_fraction_smallest25                    — matched / total
        in the smallest bucket (so unmatched-small failures are visible).
  - val/size_p25_threshold_sp                        — the n-spacepoints
        threshold used to define the smallest bucket.
  - val/n_gt_smallest25                              — bucket size.

Outputs (next to `--analysis-dir`):
  summary_<TAG>_<MODEL_TAG>.json     — scalar metrics
  pairs_<TAG>_<MODEL_TAG>.parquet    — full per-pair table (for ad-hoc slicing)
  events_<TAG>_<MODEL_TAG>.parquet   — per-event metadata
"""

import argparse
import glob
import json
import os
import sys

import h5py
import numpy as np
import pandas as pd


DEFAULT_PARTICLE_CLASS_NAMES = (
    "e", "gamma", "mu", "pi", "p", "other", "(unused)", "no_object",
)


def _read_perevent(path: str) -> tuple[pd.DataFrame, dict]:
    """Load one perevent_*.h5 → (pairs_df, event_record).

    Empty events (n_gt = 0) still produce a row in `event_record` but
    `pairs_df` will be empty for them.
    """
    with h5py.File(path, "r") as f:
        pair = f["pair"]
        pairs = pd.DataFrame({
            "class_id":          pair["class_id"][...].astype(np.int64),
            "n_truth_points":    pair["n_truth_points"][...].astype(np.int64),
            "matched_query":     pair["matched_query"][...].astype(np.int64),
            "pair_iou":          pair["pair_iou"][...].astype(np.float32),
            "pair_cls_correct":  pair["pair_cls_correct"][...].astype(np.int8),
            "pair_origin_l2_cm": pair["pair_origin_l2_cm"][...].astype(np.float32),
        })
        ev_id = (
            int(f.attrs.get("run", -1)),
            int(f.attrs.get("subrun", -1)),
            int(f.attrs.get("event", -1)),
        )
        pairs["run"]    = ev_id[0]
        pairs["subrun"] = ev_id[1]
        pairs["event"]  = ev_id[2]
        event = {
            "run":                ev_id[0],
            "subrun":             ev_id[1],
            "event":              ev_id[2],
            "name":               str(f.attrs.get("name", "")),
            "n_sp_post":          int(f.attrs.get("n_sp_post", 0)),
            "n_active_queries":   int(f.attrs.get("n_active_queries", 0)),
            "n_gt":               int(f.attrs.get("n_gt", 0)),
            "n_matched":          int(f.attrs.get("n_matched", 0)),
            "perevent_path":      path,
        }
    return pairs, event


def _safe_mean(arr) -> float:
    a = np.asarray(arr, dtype=np.float64)
    return float(a.mean()) if a.size else float("nan")


def _safe_quantile(arr, q) -> float:
    a = np.asarray(arr, dtype=np.float64)
    return float(np.quantile(a, q)) if a.size else float("nan")


def aggregate(analysis_dir: str, class_names: list[str]) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    perevent_paths = sorted(glob.glob(
        os.path.join(analysis_dir, "perevent_*.h5")))
    skipped_paths = sorted(glob.glob(
        os.path.join(analysis_dir, "skipped_*.h5")))
    if not perevent_paths:
        sys.exit(f"no perevent_*.h5 found in {analysis_dir}")

    print(f"[agg] {len(perevent_paths)} perevent + "
          f"{len(skipped_paths)} skipped events")

    pair_frames = []
    event_records = []
    for p in perevent_paths:
        pdf, ev = _read_perevent(p)
        if len(pdf):
            pair_frames.append(pdf)
        event_records.append(ev)

    pairs = (pd.concat(pair_frames, ignore_index=True)
             if pair_frames else pd.DataFrame(columns=[
                 "class_id", "n_truth_points", "matched_query",
                 "pair_iou", "pair_cls_correct", "pair_origin_l2_cm",
                 "run", "subrun", "event",
             ]))
    events = pd.DataFrame.from_records(event_records)

    # Matched pairs only — mirrors evaluator's all_ious accounting.
    matched_mask = pairs["matched_query"] >= 0
    m = pairs[matched_mask]
    iou_valid = pairs[(pairs["pair_iou"] >= 0) & matched_mask]["pair_iou"]
    l2_valid  = pairs[(pairs["pair_origin_l2_cm"] >= 0)
                      & matched_mask]["pair_origin_l2_cm"]

    scalars: dict = {
        "n_events":                 int(len(events)),
        "n_pairs":                  int(len(pairs)),
        "n_matched":                int(matched_mask.sum()),
        "val/mask_iou_mean":        _safe_mean(iou_valid),
        "val/mask_iou_median":      _safe_quantile(iou_valid, 0.5),
        "val/mask_iou_p25":         _safe_quantile(iou_valid, 0.25),
        "val/cls_accuracy":         _safe_mean(
            m[m["pair_cls_correct"] >= 0]["pair_cls_correct"].astype(np.float32)),
        "val/matched_fraction":     (float(matched_mask.sum()) / float(len(pairs))
                                     if len(pairs) else float("nan")),
        "val/n_active_queries_mean": _safe_mean(events["n_active_queries"]),
        "val/origin_l2_cm_mean":    _safe_mean(l2_valid),
        "val/origin_l2_cm_median":  _safe_quantile(l2_valid, 0.5),
        "val/origin_l2_cm_p25":     _safe_quantile(l2_valid, 0.25),
        "val/origin_l2_cm_p75":     _safe_quantile(l2_valid, 0.75),
    }

    # Per-class — keyed by GT class id.
    for cid, name in enumerate(class_names):
        sub = m[m["class_id"] == cid]
        iou_sub = sub[sub["pair_iou"] >= 0]["pair_iou"]
        l2_sub  = sub[sub["pair_origin_l2_cm"] >= 0]["pair_origin_l2_cm"]
        if len(sub):
            scalars[f"val/mask_iou_{name}"]     = _safe_mean(iou_sub)
            scalars[f"val/origin_l2_cm_{name}"] = _safe_mean(l2_sub)
            scalars[f"val/n_pairs_{name}"]      = int(len(sub))

    # Stress metric: bottom-25% GTs by spacepoint count.
    # Threshold is computed over ALL GTs (matched and unmatched) so it
    # measures "small GTs," not "small matched GTs." Headline IoU is
    # over matched-only in the bucket (evaluator-style). Match fraction
    # in the bucket is reported separately so unmatched-small failures
    # are visible.
    if len(pairs):
        size_p25 = float(np.quantile(pairs["n_truth_points"], 0.25))
        small_mask = pairs["n_truth_points"] <= size_p25
        small = pairs[small_mask]
        small_matched = small[small["matched_query"] >= 0]
        small_iou_valid = small_matched[
            small_matched["pair_iou"] >= 0]["pair_iou"]
        scalars["val/size_p25_threshold_sp"]        = size_p25
        scalars["val/n_gt_smallest25"]              = int(len(small))
        scalars["val/match_fraction_smallest25"]    = (
            float(len(small_matched)) / float(len(small))
            if len(small) else float("nan"))
        scalars["val/mask_iou_smallest25"]          = _safe_mean(small_iou_valid)
        scalars["val/mask_iou_smallest25_median"]   = _safe_quantile(
            small_iou_valid, 0.5)

    return scalars, pairs, events


def _print_log_line(scalars: dict, class_names: list[str]) -> None:
    """Mirrors `LArFormerParticleEvaluator._log_and_publish` so the
    output is directly comparable to in-training val logs."""
    line = (
        f"Val: cls_acc {scalars['val/cls_accuracy']:.4f} "
        f"| iou_mean {scalars['val/mask_iou_mean']:.4f} "
        f"(med {scalars['val/mask_iou_median']:.4f}, "
        f"p25 {scalars['val/mask_iou_p25']:.4f}) "
        f"| matched_frac {scalars['val/matched_fraction']:.3f} "
        f"| n_active_q_avg {scalars['val/n_active_queries_mean']:.1f}"
    )
    print(line)

    if "val/origin_l2_cm_mean" in scalars and not np.isnan(scalars["val/origin_l2_cm_mean"]):
        print(
            f"Val origin err (cm): mean "
            f"{scalars['val/origin_l2_cm_mean']:.2f}  "
            f"median {scalars['val/origin_l2_cm_median']:.2f}  "
            f"p25 {scalars['val/origin_l2_cm_p25']:.2f}  "
            f"p75 {scalars['val/origin_l2_cm_p75']:.2f}"
        )

    iou_parts = []
    ori_parts = []
    for name in class_names:
        iou_key = f"val/mask_iou_{name}"
        ori_key = f"val/origin_l2_cm_{name}"
        if iou_key in scalars and not np.isnan(scalars[iou_key]):
            iou_parts.append(f"{name}={scalars[iou_key]:.3f}")
        if ori_key in scalars and not np.isnan(scalars[ori_key]):
            ori_parts.append(f"{name}={scalars[ori_key]:.1f}")
    if iou_parts:
        print("Val per-class IoU: " + " ".join(iou_parts))
    if ori_parts:
        print("Val per-class origin err (cm): " + " ".join(ori_parts))

    if "val/mask_iou_smallest25" in scalars:
        print(
            f"Val smallest-25% by SP count "
            f"(threshold={scalars['val/size_p25_threshold_sp']:.0f} SPs, "
            f"n_gt={scalars['val/n_gt_smallest25']}): "
            f"iou_mean {scalars['val/mask_iou_smallest25']:.4f}  "
            f"median {scalars['val/mask_iou_smallest25_median']:.4f}  "
            f"match_frac {scalars['val/match_fraction_smallest25']:.3f}"
        )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("--analysis-dir", required=True,
                    help="Directory of perevent_*.h5 files.")
    ap.add_argument("--output-dir", default=None,
                    help="Where summary + parquet bundle is written. "
                         "Defaults to parent of --analysis-dir.")
    ap.add_argument("--tag", default="UNKNOWN",
                    help="Embedded in output file names.")
    ap.add_argument("--model-tag", default="UNKNOWN",
                    help="Embedded in output file names.")
    ap.add_argument("--class-names", default=",".join(DEFAULT_PARTICLE_CLASS_NAMES),
                    help="Comma-separated class names by class id. "
                         "Default = 7-class taxonomy + no_object.")
    args = ap.parse_args()

    class_names = [c.strip() for c in args.class_names.split(",") if c.strip()]
    output_dir = args.output_dir or os.path.dirname(
        os.path.abspath(args.analysis_dir).rstrip("/"))
    os.makedirs(output_dir, exist_ok=True)

    scalars, pairs, events = aggregate(args.analysis_dir, class_names)

    stem = f"{args.tag}_{args.model_tag}"
    summary_path = os.path.join(output_dir, f"summary_{stem}.json")
    pairs_path   = os.path.join(output_dir, f"pairs_{stem}.parquet")
    events_path  = os.path.join(output_dir, f"events_{stem}.parquet")

    with open(summary_path, "w") as f:
        json.dump(scalars, f, indent=2, sort_keys=True)
    pairs.to_parquet(pairs_path, index=False)
    events.to_parquet(events_path, index=False)

    print()
    _print_log_line(scalars, class_names)
    print()
    print(f"[agg] wrote {summary_path}")
    print(f"[agg] wrote {pairs_path}    ({len(pairs)} rows)")
    print(f"[agg] wrote {events_path}   ({len(events)} rows)")


if __name__ == "__main__":
    main()
