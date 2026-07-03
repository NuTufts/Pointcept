"""Inspect a per-event analysis H5 + (optionally) the source inference H5.

Dumps everything in a human-readable form so we can debug discrepancies
between the analysis output and what the visualizer / val metrics say.

What it shows:

  1. perevent attrs and the truth + in-time-flash + GT-baseline summary.
  2. The full pred_slices table (one row per predicted slice), sorted
     by IoU vs GT-nu descending. Shows class_argmax, IoU, n_sp, oob,
     chi2 at default threshold.
  3. The metrics dict (M1/M3/M4).
  4. If --inference-h5 is given, ALSO loads the raw inference outputs:
     - queries/class_argmax + queries/class_probs histogram
     - gt/origin_type + gt/matched_query + gt/pair_iou + gt/pair_cls_correct
     - flags any disagreement between what perevent recorded and what
       the inference H5 actually says.

This is the right tool when the headline metrics don't match what the
visualizer shows or what the val-time evaluator reported.

Usage:
  python inspect_perevent.py --perevent-h5 /path/to/perevent_*.h5
  python inspect_perevent.py --perevent-h5 /path/... \\
                             --inference-h5 /path/to/slicerpred_*.h5
"""

import argparse
import os
import sys

import h5py
import numpy as np


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

from lartpc.larformer_analysis.slicer_eval.lib import categorize  # noqa: E402


# Model-side class names (the column labels of `queries/class_probs`).
# These are NOT the analysis-side category names (which include
# ccnumu/ccnue/pi0/etc. and live in the perevent's `category_names`
# attr). Don't confuse the two.
_MODEL_CLASS_NAMES = ["nu", "cosmic", "no_object"]


def _fmt_or(v, fmt="{:.3f}", null="--"):
    try:
        if np.isnan(v) or v is None:
            return null
        return fmt.format(float(v))
    except Exception:
        return null


def dump_perevent(p, inference_p=None):
    print(f"\n=== perevent file: {p}")
    if not os.path.exists(p):
        sys.exit(f"  no such file: {p}")
    with h5py.File(p, "r") as f:
        # ----- top-level attrs --------------------------------------------
        run    = int(f.attrs["run"])
        subrun = int(f.attrs["subrun"])
        event  = int(f.attrs["event"])
        model_tag = str(f.attrs.get("model_tag", "?"))
        oob_thresholds = np.asarray(f.attrs["oob_thresholds"],
                                    dtype=np.float32)
        default_oob_idx = int(f.attrs["default_oob_idx"])
        cat_names_bytes = f.attrs.get(
            "category_names",
            np.array(categorize.CATEGORY_NAMES, "S32"),
        )
        category_names = [n.decode() if isinstance(n, bytes) else str(n)
                          for n in cat_names_bytes]
        nu_class_id        = int(f.attrs.get("nu_class_id", 0))
        no_object_class_id = int(f.attrs.get("no_object_class_id", 2))
        has_nu_pred        = bool(f.attrs.get("has_nu_prediction", False))

        print(f"  (run, subrun, event) = ({run}, {subrun}, {event})")
        print(f"  model_tag            = {model_tag}")
        print(f"  has_nu_prediction    = {has_nu_pred}")
        print(f"  oob_thresholds       = {oob_thresholds}  "
              f"(default idx={default_oob_idx} → {oob_thresholds[default_oob_idx]:.2f})")
        print(f"  nu_class_id={nu_class_id}  no_object_class_id={no_object_class_id}")

        # ----- truth -------------------------------------------------------
        tr = f["truth"].attrs
        cm = int(tr.get("category_mask", 0))
        cat_str = categorize.category_str(cm)
        print(f"\n--- truth ---")
        print(f"  category_mask = 0b{cm:05b} = {cat_str}")
        print(f"  has_neutrino={bool(tr.get('has_neutrino', False))}  "
              f"nu_pdg={int(tr.get('nu_pdg', 0))}  ccnc={int(tr.get('ccnc', -1))}  "
              f"Enu={float(tr.get('nu_energy_MeV', 0.0)):.1f} MeV")
        print(f"  n_visible_nu_gammas={int(tr.get('n_visible_nu_gammas', 0))}  "
              f"n_primary_pi0={int(tr.get('n_primary_pi0', 0))}")

        # ----- in-time flash + GT-baseline --------------------------------
        it = f["in_time_flash"]
        print(f"\n--- in-time flash ---")
        print(f"  flash_idx={int(it.attrs['flash_idx'])}  "
              f"t0={float(it.attrs['t0_us']):.3f}us  "
              f"producer={int(it.attrs['producer_id'])}  "
              f"paired_slice_id={int(it.attrs['paired_slice_id'])}")
        pe_obs = it["pe_obs"][:]
        print(f"  pe_obs total = {float(pe_obs.sum()):.1f}  "
              f"max={float(pe_obs.max()):.1f}  argmax_pmt={int(pe_obs.argmax())}")

        gb = f["gt_baseline"]
        print(f"\n--- GT-baseline ---")
        print(f"  n_sp={int(gb.attrs['n_sp'])}  "
              f"oob_frac={float(gb.attrs['oob_frac']):.3f}")
        pe_pred_gt = gb["pe_pred"][:]
        chi2_gt = gb["chi2"][:]
        print(f"  pe_pred total = {float(pe_pred_gt.sum()):.1f}  "
              f"max={float(pe_pred_gt.max()):.1f}")
        print(f"  chi2 sweep = " + " ".join(
            f"{float(c):.1f}" if np.isfinite(c) else "nan" for c in chi2_gt))

        # ----- pred slices table ------------------------------------------
        ps = f["pred_slices"]
        q_id = ps["query_id"][:]
        cls  = ps["class_argmax"][:]
        n_sp = ps["n_sp"][:]
        iou  = ps["iou_vs_gt_nu"][:]
        oob  = ps["oob_frac"][:]
        chi2 = ps["chi2"][:]
        chi2_default = chi2[:, default_oob_idx] if chi2.size else np.zeros(0)
        n_pred = int(ps.attrs.get("n_pred_slices", 0))
        n_pred_nu = int(ps.attrs.get("n_pred_nu_slices", 0))
        print(f"\n--- pred slices ({n_pred} total, {n_pred_nu} nu) "
              f"— sorted by IoU desc ---")

        def _class_name(c):
            if c == nu_class_id:        return "nu"
            if c == no_object_class_id: return "no_obj"
            return f"c{int(c)}"

        if n_pred == 0:
            print("  (empty)")
        else:
            order = np.argsort(-iou)
            print(f"  {'qid':>4s}  {'class':>7s}  {'n_sp':>5s}  "
                  f"{'iou':>6s}  {'oob':>5s}  "
                  f"{'chi2@'+str(default_oob_idx):>9s}")
            for j in order:
                print(f"  {int(q_id[j]):>4d}  {_class_name(int(cls[j])):>7s}  "
                      f"{int(n_sp[j]):>5d}  {float(iou[j]):>6.3f}  "
                      f"{float(oob[j]):>5.3f}  "
                      f"{_fmt_or(chi2_default[j], '{:>9.2f}', '      nan')}")

            # class histogram for sanity
            print(f"\n  class_argmax distribution:")
            for c in sorted(set(int(x) for x in cls)):
                hits = int((cls == c).sum())
                print(f"    {_class_name(c):>7s} (id={c}): {hits} queries")

        # ----- metrics -----------------------------------------------------
        m = f["metrics"]
        print(f"\n--- metrics ---")
        print(f"  has_nu_prediction = {bool(m.attrs.get('has_nu_prediction', False))}")
        print(f"  m1_iou = {float(m.attrs.get('m1_iou', 0)):.3f}  "
              f"(slice_id={int(m.attrs.get('m1_slice_id', -1))})")
        print(f"  m3_chi2_gt = {float(m.attrs.get('m3_chi2_gt', 0)):.2f}  "
              f"chi2_nu = {_fmt_or(float(m.attrs.get('m3_chi2_nu', np.nan)), '{:.2f}')}")
        m3_dc = m["m3_delta_chi2"][:]
        print(f"  m3_delta_chi2 sweep = " + " ".join(
            _fmt_or(x, "{:+.2f}") for x in m3_dc))
        print(f"  m4_rank_all = {list(int(x) for x in m['m4_rank_all'][:])}")
        print(f"  m4_rank_nu  = {list(int(x) for x in m['m4_rank_nu'][:])}")

        # ----- cross-check vs raw inference -------------------------------
        if inference_p is not None:
            # Pass the MODEL-side class names (column labels of
            # queries/class_probs), NOT the analysis-side category_names
            # — those are different vocabularies and confusing them
            # labels nu predictions as 'ccnumu', etc.
            _cross_check_inference(
                inference_p, q_id, cls, iou,
                nu_class_id, no_object_class_id, _MODEL_CLASS_NAMES,
            )


def _cross_check_inference(inf_p, perevent_qid, perevent_cls, perevent_iou,
                           nu_class_id, no_object_class_id, class_names):
    print(f"\n=== inference file: {inf_p}")
    if not os.path.exists(inf_p):
        print(f"  [warn] no such file; skipping cross-check")
        return
    with h5py.File(inf_p, "r") as f:
        q_cls    = f["queries/class_argmax"][:].astype(np.int64)
        q_probs  = f["queries/class_probs"][:].astype(np.float32)
        q_match  = f["queries/matched_gt_idx"][:].astype(np.int64)
        gt_origin = f["gt/origin_type"][:].astype(np.int64)
        gt_tid    = f["gt/primary_trackid"][:].astype(np.int64)
        gt_n_pts  = f["gt/n_truth_points"][:].astype(np.int64)
        gt_match  = f["gt/matched_query"][:].astype(np.int64)
        gt_pair_iou = f["gt/pair_iou"][:].astype(np.float32)
        gt_pair_cls = f["gt/pair_cls_correct"][:].astype(np.int64)
        # pred_query is the actual query id (0..Q); pred_slice_id is
        # the matched GT's primary_trackid for matched queries (a huge
        # number — NOT useful for "did query mq win SPs" counts).
        post_pred_query = f["post/pred_query"][:].astype(np.int64)
        post_gt    = f["post/slice_id_gt"][:].astype(np.int64)
        meta_no_obj = int(f.attrs.get("meta_no_object_class_id", 2))

    n_queries = q_cls.shape[0]
    print(f"--- queries/class_argmax distribution ({n_queries} queries) ---")
    for c in sorted(set(int(x) for x in q_cls)):
        n = int((q_cls == c).sum())
        cname = (class_names[c]
                 if 0 <= c < len(class_names) else f"c{c}")
        print(f"  class {c} ({cname}): {n} queries")
    nu_queries = int((q_cls == nu_class_id).sum())
    print(f"  >>> queries with class_argmax == nu_class_id ({nu_class_id}): "
          f"{nu_queries}")
    print(f"  inference file's meta_no_object_class_id = {meta_no_obj} "
          f"(should match analyzer's no_object_class_id={no_object_class_id})")

    print(f"\n--- queries/class_probs[nu_class_id] — top 10 ---")
    nu_probs = q_probs[:, nu_class_id]
    top10 = np.argsort(-nu_probs)[:10]
    for q in top10:
        cargm = int(q_cls[q])
        cname = (class_names[cargm]
                 if 0 <= cargm < len(class_names) else f"c{cargm}")
        print(f"  qid={int(q):>3d}  p(nu)={float(nu_probs[q]):.3f}  "
              f"argmax={cargm} ({cname})  matched_gt_idx={int(q_match[q])}")

    print(f"\n--- GT instances ({len(gt_origin)} total) ---")
    print(f"  {'k':>3s}  {'origin':>6s}  {'tid':>8s}  {'n_truth':>8s}  "
          f"{'matched_q':>9s}  {'pair_iou':>9s}  {'cls_corr':>8s}")
    for k in range(len(gt_origin)):
        print(f"  {k:>3d}  {int(gt_origin[k]):>6d}  "
              f"{int(gt_tid[k]):>8d}  {int(gt_n_pts[k]):>8d}  "
              f"{int(gt_match[k]):>9d}  "
              f"{float(gt_pair_iou[k]):>9.3f}  "
              f"{int(gt_pair_cls[k]):>8d}")
    n_nu_gt = int((gt_origin == nu_class_id).sum())
    n_cls_correct_nu = int(((gt_origin == nu_class_id) & (gt_pair_cls == 1)).sum())
    print(f"  -> nu GT instances: {n_nu_gt}  "
          f"class-correct (matched query's argmax == nu): {n_cls_correct_nu}")

    # ----- the key diagnostic --------------------------------------------
    # Is the matched query for the nu-GT predicting nu? If yes, our
    # analyzer should have set class_argmax==nu for SOME query but
    # didn't — which means our class_argmax loading is wrong OR the
    # pred_slice_id panoptic-argmax dropped those queries.
    print(f"\n--- diagnostic: what each nu-GT's matched query is predicting ---")
    for k in np.flatnonzero(gt_origin == nu_class_id):
        mq = int(gt_match[k])
        if mq < 0:
            print(f"  GT k={k}: unmatched")
            continue
        c = int(q_cls[mq])
        cname = (class_names[c] if 0 <= c < len(class_names) else f"c{c}")
        # How many SPs did this query WIN in the panoptic argmax?
        won = int((post_pred_query == mq).sum())
        # Did it appear in perevent?
        in_perevent = int(mq in set(int(x) for x in perevent_qid))
        print(f"  GT k={k}, tid={int(gt_tid[k])}: matched query qid={mq}, "
              f"class_argmax={c} ({cname}), won_sp_panoptic={won}, "
              f"in_perevent_pred_slices={bool(in_perevent)}")
        if in_perevent:
            idx = int(np.flatnonzero(perevent_qid == mq)[0])
            print(f"    perevent: class={int(perevent_cls[idx])}, "
                  f"iou={float(perevent_iou[idx]):.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--perevent-h5", required=True,
                    help="Per-event H5 from analyze_event.py")
    ap.add_argument("--inference-h5", default=None,
                    help="Source slicerpred_*.h5 — enables the raw-fields "
                         "cross-check section")
    args = ap.parse_args()
    dump_perevent(args.perevent_h5, args.inference_h5)


if __name__ == "__main__":
    main()
