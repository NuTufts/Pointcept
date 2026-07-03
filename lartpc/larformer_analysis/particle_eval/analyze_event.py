"""Per-event analyzer for the LArFormer Stage-3 particle segmenter.

Thin transformer: reads a `stage3pred_<stem>.h5` produced by
`tools/larformer/run_larformer_stage3_inference.py`, distills the few records
needed for validation metrics, writes either
`perevent_<stem>.h5` (event has GT and inference succeeded) or
`skipped_<stem>.h5` (no GT, or event was dropped) into `--output-dir`.

The aggregator (`aggregate_metrics.py`) walks the resulting tree to
produce the same scalars `LArFormerParticleEvaluator` emits at training
time (+ a size-stratified subset). No per-pixel re-computation happens
here — the inference H5 already carries `stage3_gt/pair_iou`,
`stage3_gt/pair_cls_correct`, `stage3_gt/pair_origin_l2_cm`,
`stage3_gt/class_id`, and `stage3_gt/n_truth_points`.

Output H5 schema:
    pair/class_id            int  (Q for each GT instance)
    pair/n_truth_points      int
    pair/matched_query       int  (-1 = unmatched)
    pair/pair_iou            float (-1 where unmatched)
    pair/pair_cls_correct    int8 (-1 where unmatched)
    pair/pair_origin_l2_cm   float (-1 where unmatched or origin not GT)
    (root attrs)
        run, subrun, event, name, model_tag
        n_sp_post, n_active_queries, n_gt, n_matched
        no_object_class_id, class_prob_threshold
        stage3pred_path
"""

import argparse
import os
import sys

import h5py
import numpy as np


REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
sys.path.insert(0, REPO_ROOT)

from pointcept.models.LArFormer.inference import load_event_h5  # noqa: E402


def _stem_from_stage3pred(path: str) -> str:
    """`/.../stage3pred_<base>.h5` → `<base>`. Fallback to the file's
    own basename if the prefix isn't there (shouldn't happen)."""
    base = os.path.splitext(os.path.basename(path))[0]
    if base.startswith("stage3pred_"):
        return base[len("stage3pred_"):]
    return base


def _write_skipped(out_path: str, *, reason: str, stage3pred_path: str,
                   model_tag: str, ev: dict) -> None:
    """Write a tiny marker file so the driver's skip-if-exists logic
    treats this event as 'analyzed' next time."""
    tmp = out_path + ".tmp"
    with h5py.File(tmp, "w") as f:
        f.attrs["skipped_reason"] = reason
        f.attrs["model_tag"] = model_tag
        f.attrs["stage3pred_path"] = stage3pred_path
        for k in ("stage3_meta_run", "stage3_meta_subrun",
                  "stage3_meta_event", "stage3_meta_name"):
            v = ev.get(k)
            if v is not None:
                f.attrs[k.replace("stage3_meta_", "")] = v
    os.replace(tmp, out_path)


def analyze_event(stage3pred_path: str, output_dir: str,
                  model_tag: str) -> None:
    ev = load_event_h5(stage3pred_path)
    if ev is None:
        sys.exit(f"failed to read {stage3pred_path}")
    stem = _stem_from_stage3pred(stage3pred_path)

    has_gt = bool(ev.get("stage3_meta_has_gt", 0))
    dropped = bool(ev.get("stage3_meta_event_dropped", 0))
    if dropped or not has_gt:
        out_path = os.path.join(output_dir, f"skipped_{stem}.h5")
        reason = "event_dropped" if dropped else "no_gt"
        _write_skipped(out_path, reason=reason,
                       stage3pred_path=stage3pred_path,
                       model_tag=model_tag, ev=ev)
        print(f"[analyze] SKIP ({reason}): {os.path.basename(out_path)}")
        return

    # Pair records — one row per GT instance.
    class_id          = np.asarray(ev["stage3_gt/class_id"]).astype(np.int64)
    n_truth_points    = np.asarray(ev["stage3_gt/n_truth_points"]).astype(np.int64)
    matched_query     = np.asarray(ev["stage3_gt/matched_query"]).astype(np.int64)
    pair_iou          = np.asarray(ev["stage3_gt/pair_iou"]).astype(np.float32)
    pair_cls_correct  = np.asarray(ev["stage3_gt/pair_cls_correct"]).astype(np.int8)
    pair_origin_l2_cm = np.asarray(ev["stage3_gt/pair_origin_l2_cm"]).astype(np.float32)

    # Event-level scalars (the active-query count is a derived metric
    # the in-training evaluator publishes — compute it the same way
    # here: "queries whose argmax class is not no_object").
    class_argmax = np.asarray(ev.get("stage3_queries/class_argmax"))
    no_obj = int(ev.get("stage3_meta_no_object_class_id", -1))
    n_active = int((class_argmax != no_obj).sum()) if class_argmax.size else 0

    out_path = os.path.join(output_dir, f"perevent_{stem}.h5")
    tmp = out_path + ".tmp"
    with h5py.File(tmp, "w") as f:
        pair = f.create_group("pair")
        pair.create_dataset("class_id",            data=class_id,
                            compression="gzip", compression_opts=4)
        pair.create_dataset("n_truth_points",      data=n_truth_points,
                            compression="gzip", compression_opts=4)
        pair.create_dataset("matched_query",       data=matched_query,
                            compression="gzip", compression_opts=4)
        pair.create_dataset("pair_iou",            data=pair_iou,
                            compression="gzip", compression_opts=4)
        pair.create_dataset("pair_cls_correct",    data=pair_cls_correct,
                            compression="gzip", compression_opts=4)
        pair.create_dataset("pair_origin_l2_cm",   data=pair_origin_l2_cm,
                            compression="gzip", compression_opts=4)

        f.attrs["model_tag"]            = model_tag
        f.attrs["stage3pred_path"]      = stage3pred_path
        f.attrs["run"]                  = int(ev.get("stage3_meta_run", -1))
        f.attrs["subrun"]               = int(ev.get("stage3_meta_subrun", -1))
        f.attrs["event"]                = int(ev.get("stage3_meta_event", -1))
        f.attrs["name"]                 = str(ev.get("stage3_meta_name", ""))
        f.attrs["n_sp_post"]            = int(ev.get("stage3_meta_n_stage3_sp", 0))
        f.attrs["n_active_queries"]     = n_active
        f.attrs["n_gt"]                 = int(class_id.size)
        f.attrs["n_matched"]            = int((matched_query >= 0).sum())
        f.attrs["no_object_class_id"]   = no_obj
        f.attrs["class_prob_threshold"] = float(
            ev.get("stage3_meta_class_prob_threshold", 0.0))
    os.replace(tmp, out_path)

    matched = int((matched_query >= 0).sum())
    valid_iou = pair_iou[pair_iou >= 0]
    mean_iou = float(valid_iou.mean()) if valid_iou.size else float("nan")
    print(f"[analyze] {os.path.basename(out_path):60s}  "
          f"n_gt={class_id.size}  matched={matched}  "
          f"mean_pair_IoU={mean_iou:.3f}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("--stage3pred-h5", required=True,
                    help="One stage3pred_<stem>.h5 from "
                         "run_larformer_stage3_inference.py")
    ap.add_argument("--output-dir", required=True,
                    help="Where perevent_<stem>.h5 / skipped_<stem>.h5 "
                         "gets written")
    ap.add_argument("--model-tag", required=True,
                    help="Short identifier embedded in the output H5's "
                         "model_tag attr (used by the aggregator).")
    args = ap.parse_args()

    if not os.path.exists(args.stage3pred_h5):
        sys.exit(f"input not found: {args.stage3pred_h5}")
    os.makedirs(args.output_dir, exist_ok=True)
    analyze_event(args.stage3pred_h5, args.output_dir, args.model_tag)


if __name__ == "__main__":
    main()
