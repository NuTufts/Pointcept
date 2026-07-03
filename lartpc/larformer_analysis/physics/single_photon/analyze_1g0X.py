"""
Phase-4b: how well the LArFormer cascade SELECTS 1-gamma + 0X events.

RECO 1g+0X (from stage3pred active post-dedup queries):
  exactly 1 active PHOTON query (class==1) AND no other significant particle query
  (active query of any non-photon, non-no_object class with >= --other-npts-min
  stage3 spacepoints). Small spurious queries below that size don't veto.

TRUTH 1g+0X comes from select_visible_photon_events.py (visible_photon_events.csv,
is_1g0X column): 1 visible photon (nSP_trunk>=80) and no visible nu-induced X above
per-type KE thresholds.

Matching reco<->truth by (run,subrun,event) gives the confusion matrix:
  efficiency = P(reco 1g0X | truth 1g0X)   (selection efficiency)
  purity     = P(truth 1g0X | reco 1g0X)   (sample purity)
to compare against the prior analysis's ~10% / ~40%.

Note: events present in stage3pred but NOT in the truth CSV had no visible photon
(truth n_vis_gamma==0); they form part of the purity denominator if reco-selected.

Run inside the pointcept container:
  python3 analyze_1g0X.py --pred-dir <stage3pred> --truth-csv <visible_photon_events.csv> \
      --other-npts-min 20 --out workdir_scale/reco_1g0X.csv
"""
import argparse
import csv
import glob
import os
import h5py
import numpy as np

PHOTON_CLASS = 1


def reco_label(ph5, other_npts_min):
    """(n_photon_q, n_other_q, reco_1g0X) from active post-dedup queries."""
    if "stage3" not in ph5 or "stage3_queries" not in ph5:
        return 0, 0, 0
    q = ph5["stage3_queries"]
    if "class_argmax" not in q:
        return 0, 0, 0
    ca = q["class_argmax"][:]
    act = q["is_active_postdedup"][:].astype(bool)
    pq = ph5["stage3"]["pred_query"][:]
    no_obj = int(ph5.attrs.get("stage3_meta_no_object_class_id", 7))
    n_phot = n_other = 0
    for qi in np.where(act)[0]:
        c = int(ca[qi])
        if c == no_obj:
            continue
        npts = int((pq == qi).sum())
        if c == PHOTON_CLASS:
            n_phot += 1
        elif npts >= other_npts_min:
            n_other += 1
    reco = int(n_phot == 1 and n_other == 0)
    return n_phot, n_other, reco


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--truth-csv", required=True,
                    help="visible_photon_events.csv (has run,subrun,event,is_1g0X)")
    ap.add_argument("--other-npts-min", type=int, default=20)
    ap.add_argument("--out", default="workdir_scale/reco_1g0X.csv")
    args = ap.parse_args()

    # truth: (run,subrun,event) -> is_1g0X  (events absent = no visible photon)
    truth = {}
    with open(args.truth_csv) as fh:
        for r in csv.DictReader(fh):
            truth[(int(r["run"]), int(r["subrun"]), int(r["event"]))] = int(r["is_1g0X"])

    preds = sorted(glob.glob(os.path.join(args.pred_dir, "**", "stage3pred_*.h5"),
                             recursive=True))
    rows = []
    for pf in preds:
        with h5py.File(pf, "r") as ph5:
            a = ph5.attrs
            # slicer-half meta_* is set even for stage3-dropped events (where
            # stage3_meta_* = -1); prefer it so dropped truth-1g0X events are
            # correctly counted as FN, not silently removed from the denominator.
            run = int(a.get("meta_run", a.get("stage3_meta_run", -1)))
            sub = int(a.get("meta_subrun", a.get("stage3_meta_subrun", -1)))
            evt = int(a.get("meta_event", a.get("stage3_meta_event", -1)))
            n_phot, n_other, reco = reco_label(ph5, args.other_npts_min)
        t = truth.get((run, sub, evt), 0)   # absent -> not a truth-1g0X (no vis photon)
        rows.append((run, sub, evt, n_phot, n_other, reco, t))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write("run,subrun,event,n_photon_q,n_other_q,reco_1g0X,truth_1g0X\n")
        for r in rows:
            fh.write("%d,%d,%d,%d,%d,%d,%d\n" % r)

    tp = sum(1 for r in rows if r[5] == 1 and r[6] == 1)
    fp = sum(1 for r in rows if r[5] == 1 and r[6] == 0)
    fn = sum(1 for r in rows if r[5] == 0 and r[6] == 1)
    tn = sum(1 for r in rows if r[5] == 0 and r[6] == 0)
    eff = tp / float(tp + fn) if (tp + fn) else float("nan")
    pur = tp / float(tp + fp) if (tp + fp) else float("nan")
    print("=" * 60)
    print("1gamma+0X cascade selection  (other-npts-min=%d)" % args.other_npts_min)
    print("events analyzed     : %d" % len(rows))
    print("truth 1g0X          : %d" % (tp + fn))
    print("reco  1g0X          : %d" % (tp + fp))
    print("  TP / FP / FN / TN : %d / %d / %d / %d" % (tp, fp, fn, tn))
    print("-" * 60)
    print("EFFICIENCY  P(reco|truth) : %.3f   (prior ~0.10)" % eff)
    print("PURITY      P(truth|reco) : %.3f   (prior ~0.40)" % pur)
    print("wrote: %s" % args.out)


if __name__ == "__main__":
    main()
