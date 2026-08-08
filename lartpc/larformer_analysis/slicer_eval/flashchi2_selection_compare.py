"""Compare flash-chi2 nu-slice SELECTION quality between analysis variants.

For each event and each analysis variant (e.g. (a) no flash-charge cut vs
(b) --flash-charge-preal-min 0.5), select the predicted nu-classed slice
with minimum chi2 (at the default OOB threshold) and score it against the
GT nu interaction:

    sel_recall  = |selected ∩ GT-nu| / |GT-nu|       (post-SP counts)
    sel_purity  = |selected ∩ GT-nu| / |selected|
    sel_iou     = stored iou_vs_gt_nu of the selected slice
    picked_best = selected slice == argmax-IoU nu slice (selection correct)

Also reports the same for selection over ALL slices (not just nu-classed).
Events with no in-time flash / no surviving chi2 are counted separately.

Usage:
    python3 flashchi2_selection_compare.py \
        --inference-dir <slicerpred dir> \
        --variant NAME=<analysis dir> [--variant NAME=<analysis dir> ...] \
        [--oob-idx 3]
"""

import argparse
import csv
import os
import sys
from glob import glob

import h5py
import numpy as np
from scipy.spatial import cKDTree

JOIN_TOL_CM = 0.05  # same exact-position join tolerance as analyze_event

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_event  # noqa: E402  (reuse _load_inference + _gt_nu_sp_mask)


def event_key(path, prefix):
    b = os.path.basename(path)
    return b[len(prefix):] if b.startswith(prefix) else None


def load_manifest(path):
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            out[os.path.basename(row["merged_h5"])] = row["merged_h5"]
    return out


def raw_charge_for_pre(sp_path, pre_coord, merged_path):
    """Raw Y-plane ADC per pre-SP via the exact-position triplet join
    (same convention as particle_slice_completeness.py)."""
    with h5py.File(merged_path, "r") as mh:
        g = mh[list(mh.keys())[0]]
        td_pos = g["triplet_data/pos"][()].astype(np.float64)
        td_px = g["triplet_data/pixval"][()]
    dist, idx = cKDTree(td_pos).query(pre_coord.astype(np.float64), k=1)
    good = dist < JOIN_TOL_CM
    return np.where(good, td_px[idx, 2], 0.0).astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inference-dir", required=True)
    ap.add_argument("--variant", action="append", required=True,
                    help="NAME=analysis_dir (repeatable)")
    ap.add_argument("--manifest-csv", default=None,
                    help="stage-0 manifest for the raw-charge join; enables "
                         "charge-weighted + event-level efficiency metrics")
    ap.add_argument("--oob-idx", type=int, default=None,
                    help="index into the oob-threshold sweep; default = "
                         "the perevent file's default_oob_idx attr")
    ap.add_argument("--max-events", type=int, default=None)
    args = ap.parse_args()

    variants = []
    for v in args.variant:
        name, d = v.split("=", 1)
        files = {os.path.basename(f): f for f in
                 glob(os.path.join(d, "**", "perevent_*.h5"), recursive=True)}
        variants.append((name, files))
        print(f"variant {name}: {len(files)} perevent files under {d}")

    sp_files = sorted(glob(os.path.join(args.inference_dir, "**",
                                        "slicerpred_*.h5"), recursive=True))
    if args.max_events:
        sp_files = sp_files[: args.max_events]

    manifest = load_manifest(args.manifest_csv) if args.manifest_csv else None

    stats = {name: dict(recall=[], purity=[], iou=[], picked_best=[],
                        recall_all=[], purity_all=[], iou_all=[],
                        qrec_e2e=[], qpur=[],
                        no_candidate=0, n=0)
             for name, _ in variants}

    n_ev = 0
    for sp_path in sp_files:
        # perevent filename: perevent_<run>_<subrun>_<event>.h5 — match by
        # reading run/subrun/event from any variant's file set via the
        # slicerpred meta name instead: use the analysis files common to
        # all variants keyed by basename after loading one variant lazily.
        # Simpler: perevent basename can't be derived from slicerpred name
        # directly, so index variants by (run,subrun,event) attrs once.
        break
    # Build (run,subrun,event) -> path index per variant (one-time).
    var_index = []
    for name, files in variants:
        idx = {}
        for b, f in files.items():
            try:
                with h5py.File(f, "r") as h:
                    idx[(int(h.attrs["run"]), int(h.attrs["subrun"]),
                         int(h.attrs["event"]))] = f
            except Exception:
                pass
        var_index.append((name, idx))

    for sp_path in sp_files:
        try:
            inf = analyze_event._load_inference(sp_path)
        except Exception:
            continue
        gt_nu = analyze_event._gt_nu_sp_mask(inf)
        n_gt = int(gt_nu.sum())

        # Charge-weighted machinery: raw Y charge per pre-SP + pre-level
        # GT-nu mask (denominator INCLUDES charge the deghoster removed,
        # so event-level recall/efficiency are END-TO-END).
        pre_q = post_q = None
        q_nu_pre = 0.0
        gt_nu_pre = None
        if manifest is not None:
            with h5py.File(sp_path, "r") as f:
                pre_slice_gt = f["pre/slice_id_gt"][()]
                pre_keep = f["pre/keep"][()].astype(bool)
            nu_gt_idx = np.flatnonzero(inf["gt_origin_type"] == 0)
            gt_nu_pre = np.zeros(pre_slice_gt.shape[0], dtype=bool)
            for tid in inf["gt_primary_trackid"][nu_gt_idx]:
                gt_nu_pre |= (pre_slice_gt == int(tid))
            mkey = event_key(sp_path, "slicerpred_")
            mpath = manifest.get(mkey) if mkey else None
            if mpath is not None and gt_nu_pre.any():
                pre_q = raw_charge_for_pre(sp_path, inf["pre_coord"], mpath)
                post_q = pre_q[pre_keep]
                q_nu_pre = float(pre_q[gt_nu_pre].sum())

        # Events with NO true-nu charge at all (pre level) are out of scope;
        # events whose nu was fully deghosted (n_gt==0 but q_nu_pre>0) stay
        # IN the event-level denominator as recall-0 failures.
        if n_gt == 0 and q_nu_pre <= 0:
            continue
        rse = (int(inf["run"]), int(inf["subrun"]), int(inf["event"])) \
            if "run" in inf else None
        pred_query = inf["post_pred_query"]
        n_ev += 1

        for name, idx in var_index:
            f = None
            if rse is not None:
                f = idx.get(rse)
            if f is None:
                # fall back: match by stripping prefixes (same event set
                # ordering) — skip if ambiguous
                continue
            with h5py.File(f, "r") as h:
                qid = h["pred_slices/query_id"][()]
                chi2 = h["pred_slices/chi2"][()]
                cls = h["pred_slices/class_argmax"][()]
                iou = h["pred_slices/iou_vs_gt_nu"][()]
                oi = (args.oob_idx if args.oob_idx is not None
                      else int(h.attrs.get("default_oob_idx", 3)))
            st = stats[name]
            st["n"] += 1
            if q_nu_pre > 0:
                # default: failure (overwritten below on a valid selection)
                st["qrec_e2e"].append(0.0)
            if qid.size == 0 or n_gt == 0:
                st["no_candidate"] += 1
                continue
            chi2_col = chi2[:, oi]

            def score(cand_mask, rkey, pkey, ikey):
                cand = np.where(cand_mask & np.isfinite(chi2_col))[0]
                if cand.size == 0:
                    return False
                sel = cand[np.argmin(chi2_col[cand])]
                member = pred_query == qid[sel]
                inter = int((member & gt_nu).sum())
                st[rkey].append(inter / n_gt)
                st[pkey].append(inter / max(int(member.sum()), 1))
                st[ikey].append(float(iou[sel]))
                if rkey == "recall" and post_q is not None and q_nu_pre > 0:
                    q_sel = float(post_q[member].sum())
                    q_sel_nu = float(post_q[member & gt_nu].sum())
                    st["qrec_e2e"][-1] = q_sel_nu / q_nu_pre
                    if q_sel > 0:
                        st["qpur"].append(q_sel_nu / q_sel)
                if rkey == "recall":
                    # picked_best: selected == argmax-iou among nu-classed
                    nu_c = np.where(cand_mask)[0]
                    best = nu_c[np.argmax(iou[nu_c])] if nu_c.size else -1
                    st["picked_best"].append(sel == best)
                return True

            ok_nu = score(cls == 0, "recall", "purity", "iou")
            score(np.ones_like(cls, dtype=bool),
                  "recall_all", "purity_all", "iou_all")
            if not ok_nu:
                st["no_candidate"] += 1

    print(f"\nevents with GT nu processed: {n_ev}")
    hdr = (f"{'variant':24s} {'n':>5s} {'noCand':>6s} "
           f"{'recall':>7s} {'purity':>7s} {'IoU':>7s} {'pickBest':>8s} "
           f"| {'rec_all':>7s} {'pur_all':>7s}")
    print(hdr)
    for name, _ in var_index:
        st = stats[name]
        def m(k):
            return float(np.mean(st[k])) if st[k] else float("nan")
        print(f"{name:24s} {st['n']:5d} {st['no_candidate']:6d} "
              f"{m('recall'):7.3f} {m('purity'):7.3f} {m('iou'):7.3f} "
              f"{m('picked_best'):8.3f} "
              f"| {m('recall_all'):7.3f} {m('purity_all'):7.3f}")

    if args.manifest_csv:
        print("\nEVENT-LEVEL (charge-weighted, END-TO-END: denominator = "
              "pre-deghost true-nu raw Y charge;\n  fully-deghosted / "
              "no-candidate events count as recall-0 failures)")
        print(f"{'variant':24s} {'nEv':>5s} {'<qrec>':>7s} {'med':>6s} "
              f"{'eff>0.5':>8s} {'eff>0.7':>8s} {'eff>0.9':>8s} "
              f"{'<qpur>':>7s} {'pur>0.7':>8s}")
        for name, _ in var_index:
            st = stats[name]
            qr = np.asarray(st["qrec_e2e"], dtype=float)
            qp = np.asarray(st["qpur"], dtype=float)
            if qr.size == 0:
                print(f"{name:24s}  (no charge data)")
                continue
            print(f"{name:24s} {qr.size:5d} {qr.mean():7.3f} "
                  f"{np.median(qr):6.3f} "
                  f"{(qr > 0.5).mean():8.3f} {(qr > 0.7).mean():8.3f} "
                  f"{(qr > 0.9).mean():8.3f} "
                  f"{(qp.mean() if qp.size else float('nan')):7.3f} "
                  f"{((qp > 0.7).mean() if qp.size else float('nan')):8.3f}")


if __name__ == "__main__":
    main()
