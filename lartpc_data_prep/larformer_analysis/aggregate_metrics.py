"""Stage 4: aggregate per-event H5s into event_summary.h5 + category_summary.h5.

Walks a directory of `perevent_*.h5` files (produced by `analyze_event.py`),
extracts the headline scalars / per-threshold arrays, and emits a single
flat H5 ready for plotting.

Outputs:
    <output-dir>/event_summary.h5
        attrs:
            model_tag, oob_thresholds, default_oob_idx,
            category_names (S32 array), n_events,
            nu_class_id, no_object_class_id
        events/
            run, subrun, event                  (N,)   int64
            category_mask                       (N,)   uint8
            has_nu_prediction                   (N,)   bool
            n_pred_slices, n_pred_nu_slices     (N,)   int32
            m1_iou                              (N,)   float32
            m1_slice_id                         (N,)   int32
            m3_chi2_gt, m3_chi2_nu              (N,)   float32  (NaN-safe)
            m3_delta_chi2                       (N, T) float32
            m4_rank_all, m4_rank_nu             (N, T) int32  (-1 = no rank)
            in_time_flash_total_pe              (N,)   float32
            gt_baseline_oob_frac                (N,)   float32
            gt_baseline_n_sp                    (N,)   int32

    <output-dir>/category_summary.h5
        attrs: oob_thresholds, category_names
        For each category, per-threshold aggregates:
            categories/<name>/
                attrs: n_events
                m1_iou_mean, _median, _p10, _p90        (scalar)
                m3_delta_chi2_mean, _median, _p10, _p90 (T,)
                m4_rank1_frac_all                       (T,)  fraction
                                                        with rank==1
                m4_rank1_frac_nu                        (T,)
                has_nu_prediction_frac                  scalar
"""

import argparse
import glob
import os
import sys

import h5py
import numpy as np


def _summarize(vals, axis=None):
    """mean/median/p10/p90 over `vals`. Returns dict of float32s. NaN-aware."""
    vals = np.asarray(vals)
    if vals.size == 0:
        return dict(mean=np.nan, median=np.nan, p10=np.nan, p90=np.nan,
                    n=0)
    finite = np.isfinite(vals)
    if not finite.any():
        return dict(mean=np.nan, median=np.nan, p10=np.nan, p90=np.nan,
                    n=int(vals.size))
    v = vals[finite]
    return dict(
        mean   = float(np.mean(v)),
        median = float(np.median(v)),
        p10    = float(np.percentile(v, 10)),
        p90    = float(np.percentile(v, 90)),
        n      = int(v.size),
    )


def _summarize_axis0(arr_2d):
    """Apply _summarize per column. Returns dict of (T,) arrays."""
    arr_2d = np.asarray(arr_2d)
    if arr_2d.ndim == 1:
        arr_2d = arr_2d.reshape(-1, 1)
    n, t = arr_2d.shape
    means   = np.full(t, np.nan, dtype=np.float32)
    medians = np.full(t, np.nan, dtype=np.float32)
    p10s    = np.full(t, np.nan, dtype=np.float32)
    p90s    = np.full(t, np.nan, dtype=np.float32)
    counts  = np.zeros(t, dtype=np.int32)
    for ti in range(t):
        s = _summarize(arr_2d[:, ti])
        means[ti]   = s["mean"]
        medians[ti] = s["median"]
        p10s[ti]    = s["p10"]
        p90s[ti]    = s["p90"]
        counts[ti]  = s["n"]
    return dict(mean=means, median=medians, p10=p10s, p90=p90s, n=counts)


def _frac_rank1(rank_arr_2d):
    """Per-threshold fraction of events with rank == 1 (excluding -1)."""
    rank_arr_2d = np.asarray(rank_arr_2d)
    if rank_arr_2d.ndim == 1:
        rank_arr_2d = rank_arr_2d.reshape(-1, 1)
    n, t = rank_arr_2d.shape
    out = np.zeros(t, dtype=np.float32)
    for ti in range(t):
        col = rank_arr_2d[:, ti]
        denom = int((col >= 1).sum())     # exclude -1 (no rank)
        if denom == 0:
            out[ti] = np.nan
        else:
            out[ti] = float((col == 1).sum()) / denom
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--perevent-dir", required=True,
                    help="Directory of perevent_*.h5 from analyze_event.py")
    ap.add_argument("--glob", default="perevent_*.h5",
                    help="Glob within --perevent-dir (default: %(default)s)")
    ap.add_argument("--output-dir", required=True,
                    help="Where to write event_summary.h5 + category_summary.h5")
    ap.add_argument("--model-tag", default=None,
                    help="Override model_tag (default: inferred from the "
                         "first per-event file)")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(args.perevent_dir, args.glob)))
    if not paths:
        sys.exit(f"no files matched {args.perevent_dir}/{args.glob}")
    print(f"Found {len(paths)} per-event H5 files.")

    # First pass: pull T (n_thresholds) + tags + names from file 0.
    with h5py.File(paths[0], "r") as f:
        oob_thresholds = np.asarray(f.attrs["oob_thresholds"], dtype=np.float32)
        default_oob_idx = int(f.attrs["default_oob_idx"])
        category_names_bytes = f.attrs["category_names"]
        category_names = [
            n.decode() if isinstance(n, bytes) else str(n)
            for n in category_names_bytes
        ]
        nu_class_id        = int(f.attrs.get("nu_class_id", 0))
        no_object_class_id = int(f.attrs.get("no_object_class_id", 2))
        first_tag = str(f.attrs.get("model_tag", "unknown"))
    if args.model_tag is None:
        args.model_tag = first_tag
    T = len(oob_thresholds)
    N = len(paths)

    # Allocate.
    run               = np.zeros(N, dtype=np.int64)
    subrun            = np.zeros(N, dtype=np.int64)
    event             = np.zeros(N, dtype=np.int64)
    category_mask     = np.zeros(N, dtype=np.uint8)
    has_nu_prediction = np.zeros(N, dtype=bool)
    n_pred_slices     = np.zeros(N, dtype=np.int32)
    n_pred_nu_slices  = np.zeros(N, dtype=np.int32)
    m1_iou            = np.zeros(N, dtype=np.float32)
    m1_slice_id       = np.full(N, -1, dtype=np.int32)
    m1_iou_intent     = np.zeros(N, dtype=np.float32)
    m1_slice_id_intent = np.full(N, -1, dtype=np.int32)
    m3_chi2_gt        = np.full(N, np.nan, dtype=np.float32)
    m3_chi2_nu        = np.full(N, np.nan, dtype=np.float32)
    m3_delta_chi2     = np.full((N, T), np.nan, dtype=np.float32)
    m4_rank_all       = np.full((N, T), -1, dtype=np.int32)
    m4_rank_nu        = np.full((N, T), -1, dtype=np.int32)
    in_time_total_pe  = np.full(N, np.nan, dtype=np.float32)
    gt_oob_frac       = np.full(N, np.nan, dtype=np.float32)
    gt_n_sp           = np.zeros(N, dtype=np.int32)
    # Over-claim metrics (per-pair view + sp-level confusion).
    sp_level_nu_recall    = np.full(N, np.nan, dtype=np.float32)
    sp_level_nu_precision = np.full(N, np.nan, dtype=np.float32)
    n_post_sp_total       = np.zeros(N, dtype=np.int32)
    n_post_sp_gt_nu       = np.zeros(N, dtype=np.int32)
    n_post_sp_pred_nu     = np.zeros(N, dtype=np.int32)
    n_nu_gt_instances     = np.zeros(N, dtype=np.int32)
    n_nu_gt_matched_to_nu = np.zeros(N, dtype=np.int32)
    n_nu_match_survived   = np.zeros(N, dtype=np.int32)
    # Per-nu-GT (variable-length): flatten across all events with an
    # offset[N+1] index — same convention as a CSR row pointer.
    pair_iou_flat   = []
    argmax_iou_flat = []
    overclaim_gap_flat = []  # = pair_iou - argmax_iou per nu-GT
    matched_class_flat = []
    pair_event_idx  = []      # which event each per-GT row came from
    pair_category_mask_flat = []

    for i, p in enumerate(paths):
        with h5py.File(p, "r") as f:
            run[i]    = int(f.attrs["run"])
            subrun[i] = int(f.attrs["subrun"])
            event[i]  = int(f.attrs["event"])
            tr = f["truth"]
            category_mask[i] = int(tr.attrs.get("category_mask", 0))
            ps = f["pred_slices"]
            n_pred_slices[i]    = int(ps.attrs.get("n_pred_slices", 0))
            n_pred_nu_slices[i] = int(ps.attrs.get("n_pred_nu_slices", 0))
            m = f["metrics"]
            has_nu_prediction[i] = bool(m.attrs.get("has_nu_prediction", False))
            m1_iou[i]            = float(m.attrs.get("m1_iou", 0.0))
            m1_slice_id[i]       = int(m.attrs.get("m1_slice_id", -1))
            m1_iou_intent[i]     = float(m.attrs.get("m1_iou_intent", 0.0))
            m1_slice_id_intent[i] = int(m.attrs.get("m1_slice_id_intent", -1))
            m3_chi2_gt[i]  = float(m.attrs.get("m3_chi2_gt", np.nan))
            m3_chi2_nu[i]  = float(m.attrs.get("m3_chi2_nu", np.nan))
            m3_delta_chi2[i, :] = m["m3_delta_chi2"][:]
            m4_rank_all[i, :]   = m["m4_rank_all"][:]
            m4_rank_nu[i, :]    = m["m4_rank_nu"][:]
            it = f["in_time_flash"]
            in_time_total_pe[i] = float(it["pe_obs"][:].sum())
            gb = f["gt_baseline"]
            gt_oob_frac[i] = float(gb.attrs.get("oob_frac", np.nan))
            gt_n_sp[i]     = int(gb.attrs.get("n_sp", 0))
            # Over-claim (optional — legacy perevent files without
            # overclaim/ fall through to NaN/0).
            if "overclaim" in f:
                oc = f["overclaim"]
                sp_level_nu_recall[i]    = float(oc.attrs.get("sp_level_nu_recall", np.nan))
                sp_level_nu_precision[i] = float(oc.attrs.get("sp_level_nu_precision", np.nan))
                n_post_sp_total[i]       = int(oc.attrs.get("n_post_sp_total", 0))
                n_post_sp_gt_nu[i]       = int(oc.attrs.get("n_post_sp_gt_nu", 0))
                n_post_sp_pred_nu[i]     = int(oc.attrs.get("n_post_sp_pred_nu", 0))
                n_nu_gt_instances[i]     = int(oc.attrs.get("n_nu_gt_instances", 0))
                n_nu_gt_matched_to_nu[i] = int(oc.attrs.get("n_nu_gt_matched_to_nu_query", 0))
                n_nu_match_survived[i]   = int(oc.attrs.get("n_nu_match_survived_panoptic", 0))
                p_iou = oc["matched_query_pair_iou"][:]
                a_iou = oc["matched_query_argmax_iou"][:]
                m_cls = oc["matched_query_class"][:]
                for j in range(p_iou.shape[0]):
                    pair_iou_flat.append(float(p_iou[j]))
                    argmax_iou_flat.append(float(a_iou[j]))
                    overclaim_gap_flat.append(float(p_iou[j]) - float(a_iou[j]))
                    matched_class_flat.append(int(m_cls[j]))
                    pair_event_idx.append(i)
                    pair_category_mask_flat.append(int(category_mask[i]))

    # ----- event_summary.h5 -----
    out1 = os.path.join(args.output_dir, "event_summary.h5")
    with h5py.File(out1 + ".tmp", "w") as f:
        f.attrs["model_tag"]          = args.model_tag
        f.attrs["oob_thresholds"]     = oob_thresholds
        f.attrs["default_oob_idx"]    = default_oob_idx
        f.attrs["category_names"]     = np.array(category_names, dtype="S32")
        f.attrs["n_events"]           = int(N)
        f.attrs["nu_class_id"]        = nu_class_id
        f.attrs["no_object_class_id"] = no_object_class_id
        e = f.create_group("events")
        for name, arr in (
            ("run", run), ("subrun", subrun), ("event", event),
            ("category_mask", category_mask),
            ("has_nu_prediction", has_nu_prediction),
            ("n_pred_slices", n_pred_slices),
            ("n_pred_nu_slices", n_pred_nu_slices),
            ("m1_iou", m1_iou),
            ("m1_slice_id", m1_slice_id),
            ("m1_iou_intent", m1_iou_intent),
            ("m1_slice_id_intent", m1_slice_id_intent),
            ("m3_chi2_gt", m3_chi2_gt),
            ("m3_chi2_nu", m3_chi2_nu),
            ("m3_delta_chi2", m3_delta_chi2),
            ("m4_rank_all", m4_rank_all),
            ("m4_rank_nu", m4_rank_nu),
            ("in_time_flash_total_pe", in_time_total_pe),
            ("gt_baseline_oob_frac", gt_oob_frac),
            ("gt_baseline_n_sp", gt_n_sp),
            # Over-claim per-event headlines.
            ("sp_level_nu_recall", sp_level_nu_recall),
            ("sp_level_nu_precision", sp_level_nu_precision),
            ("n_post_sp_total", n_post_sp_total),
            ("n_post_sp_gt_nu", n_post_sp_gt_nu),
            ("n_post_sp_pred_nu", n_post_sp_pred_nu),
            ("n_nu_gt_instances", n_nu_gt_instances),
            ("n_nu_gt_matched_to_nu", n_nu_gt_matched_to_nu),
            ("n_nu_match_survived", n_nu_match_survived),
        ):
            e.create_dataset(name, data=arr,
                             compression="gzip", compression_opts=4)
        # nu_pairs/  — flat per-nu-GT-instance arrays across all events
        # (CSR-style). Lets plotters do per-pair distributions.
        np_g = f.create_group("nu_pairs")
        np_g.attrs["n_pairs"] = int(len(pair_iou_flat))
        np_g.create_dataset("event_idx",
                            data=np.asarray(pair_event_idx, dtype=np.int32))
        np_g.create_dataset("category_mask",
                            data=np.asarray(pair_category_mask_flat, dtype=np.uint8))
        np_g.create_dataset("pair_iou",
                            data=np.asarray(pair_iou_flat, dtype=np.float32))
        np_g.create_dataset("argmax_iou",
                            data=np.asarray(argmax_iou_flat, dtype=np.float32))
        np_g.create_dataset("overclaim_gap",
                            data=np.asarray(overclaim_gap_flat, dtype=np.float32))
        np_g.create_dataset("matched_class",
                            data=np.asarray(matched_class_flat, dtype=np.int32))
    os.replace(out1 + ".tmp", out1)
    print(f"wrote {out1}  (N={N})")

    # ----- category_summary.h5 -----
    out2 = os.path.join(args.output_dir, "category_summary.h5")
    with h5py.File(out2 + ".tmp", "w") as f:
        f.attrs["model_tag"]       = args.model_tag
        f.attrs["oob_thresholds"]  = oob_thresholds
        f.attrs["default_oob_idx"] = default_oob_idx
        f.attrs["category_names"]  = np.array(category_names, dtype="S32")
        # ALL bin (every event).
        cat_groups = [("all", np.ones(N, dtype=bool))]
        for ci, name in enumerate(category_names):
            bit = 1 << ci
            mask = (category_mask & bit) != 0
            cat_groups.append((name, mask))
        for name, mask in cat_groups:
            g = f.create_group(f"categories/{name}")
            g.attrs["n_events"] = int(mask.sum())
            if not mask.any():
                # Empty category — write NaN scalars but keep the group.
                for k in ("m1_iou_mean", "m1_iou_median",
                          "m1_iou_p10",  "m1_iou_p90",
                          "m1_iou_intent_mean", "m1_iou_intent_median",
                          "has_nu_prediction_frac",
                          "sp_level_nu_recall_mean",
                          "sp_level_nu_recall_median",
                          "sp_level_nu_precision_mean",
                          "frac_nu_match_survived_panoptic",
                          "frac_nu_gt_matched_to_nu_query"):
                    g.attrs[k] = float("nan")
                for k in ("m3_delta_chi2_mean", "m3_delta_chi2_median",
                          "m3_delta_chi2_p10",  "m3_delta_chi2_p90",
                          "m4_rank1_frac_all",  "m4_rank1_frac_nu"):
                    g.create_dataset(k, data=np.full(T, np.nan, dtype=np.float32))
                continue
            s1 = _summarize(m1_iou[mask])
            g.attrs["m1_iou_mean"]   = s1["mean"]
            g.attrs["m1_iou_median"] = s1["median"]
            g.attrs["m1_iou_p10"]    = s1["p10"]
            g.attrs["m1_iou_p90"]    = s1["p90"]
            s1i = _summarize(m1_iou_intent[mask])
            g.attrs["m1_iou_intent_mean"]   = s1i["mean"]
            g.attrs["m1_iou_intent_median"] = s1i["median"]
            g.attrs["has_nu_prediction_frac"] = float(
                has_nu_prediction[mask].mean()
            )
            # Over-claim per-category headlines.
            s_rec = _summarize(sp_level_nu_recall[mask])
            s_pre = _summarize(sp_level_nu_precision[mask])
            g.attrs["sp_level_nu_recall_mean"]    = s_rec["mean"]
            g.attrs["sp_level_nu_recall_median"]  = s_rec["median"]
            g.attrs["sp_level_nu_precision_mean"] = s_pre["mean"]
            denom = int(n_nu_gt_matched_to_nu[mask].sum())
            num   = int(n_nu_match_survived[mask].sum())
            g.attrs["frac_nu_match_survived_panoptic"] = (
                float(num) / float(denom) if denom > 0 else float("nan")
            )
            g.attrs["frac_nu_gt_matched_to_nu_query"] = (
                float(int(n_nu_gt_matched_to_nu[mask].sum()))
                / float(int(n_nu_gt_instances[mask].sum()))
                if int(n_nu_gt_instances[mask].sum()) > 0 else float("nan")
            )
            sd = _summarize_axis0(m3_delta_chi2[mask, :])
            g.create_dataset("m3_delta_chi2_mean",   data=sd["mean"])
            g.create_dataset("m3_delta_chi2_median", data=sd["median"])
            g.create_dataset("m3_delta_chi2_p10",    data=sd["p10"])
            g.create_dataset("m3_delta_chi2_p90",    data=sd["p90"])
            g.create_dataset("m4_rank1_frac_all",
                             data=_frac_rank1(m4_rank_all[mask, :]))
            g.create_dataset("m4_rank1_frac_nu",
                             data=_frac_rank1(m4_rank_nu[mask, :]))
    os.replace(out2 + ".tmp", out2)
    print(f"wrote {out2}")

    # Console summary.
    print(f"\n=== {args.model_tag}: {N} events ===")
    print(f"  has_nu_prediction frac: {float(has_nu_prediction.mean()):.2%}")
    print(f"  m1_iou (panoptic): mean={_summarize(m1_iou)['mean']:.3f}, "
          f"med={_summarize(m1_iou)['median']:.3f}")
    print(f"  m1_iou (intent):   mean={_summarize(m1_iou_intent)['mean']:.3f}, "
          f"med={_summarize(m1_iou_intent)['median']:.3f}")
    print(f"  sp_level_nu_recall:    mean={_summarize(sp_level_nu_recall)['mean']:.3f}")
    print(f"  sp_level_nu_precision: mean={_summarize(sp_level_nu_precision)['mean']:.3f}")
    denom = int(n_nu_gt_matched_to_nu.sum())
    num   = int(n_nu_match_survived.sum())
    frac = (float(num) / float(denom)) if denom > 0 else float("nan")
    print(f"  frac_nu_match_survived_panoptic: "
          f"{frac:.2%} ({num}/{denom})")
    for ti, th in enumerate(oob_thresholds):
        s = _summarize(m3_delta_chi2[:, ti])
        rank1_all = _frac_rank1(m4_rank_all[:, ti:ti+1])[0]
        rank1_nu  = _frac_rank1(m4_rank_nu[:,  ti:ti+1])[0]
        print(f"  oob<={float(th):.2f}: "
              f"delta_chi2 mean={s['mean']:+.2f}  "
              f"rank1_all={rank1_all:.2%}  rank1_nu={rank1_nu:.2%}")


if __name__ == "__main__":
    main()
