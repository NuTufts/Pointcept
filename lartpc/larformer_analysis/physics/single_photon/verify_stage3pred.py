"""
Quick sanity look at the Pass-2 cascade output (stage3pred_*.h5).

Per event it reports the predicted particle content: the per-spacepoint class
histogram and the active post-dedup query classes (each active query = one
predicted particle instance / shower). Flags events with >=1 predicted PHOTON
query — the single-photon candidates from the model's point of view.

Particle class taxonomy (LArFormer 7-class, from pointcept/datasets/larformer.py):
    0 e±   1 gamma   2 mu±   3 pi±   4 proton   5 other_track   (7 = no_object)
So PHOTON = class 1.

Schema used (see an stage3pred file):
    stage3/pred_class                 (Nsp,)   per-SP predicted class
    stage3_queries/class_argmax       (Q,)     per-query predicted class
    stage3_queries/is_active_postdedup(Q,)     active after mask-IoU dedup
    attrs: stage3_meta_run/subrun/event, stage3_meta_n_active_queries_postdedup

Run inside the pointcept container (needs h5py):
  python3 verify_stage3pred.py --dir workdir/larformer_h5 [--photon-class 1]
"""
import argparse
import glob
import os
import h5py
import numpy as np

CLASS_NAMES = {0: "e", 1: "gamma", 2: "mu", 3: "pi", 4: "p", 5: "other", 7: "no_obj"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="workdir/larformer_h5",
                    help="dir holding stage3pred_*.h5 (searched recursively)")
    ap.add_argument("--photon-class", type=int, default=1)
    ap.add_argument("--signal-csv", default="workdir/signal_events.csv",
                    help="optional: cross-check that events are signal events")
    args = ap.parse_args()

    signal = set()
    if args.signal_csv and os.path.exists(args.signal_csv):
        import csv
        with open(args.signal_csv) as fh:
            for row in csv.DictReader(fh):
                signal.add((int(row["run"]), int(row["subrun"]), int(row["event"])))

    files = sorted(glob.glob(os.path.join(args.dir, "**", "stage3pred_*.h5"),
                             recursive=True))
    print("stage3pred files:", len(files))
    print("-" * 78)
    print("%-46s %5s %4s %s" % ("event (run,subrun,event)", "Qact", "phot", "active-query classes"))
    print("-" * 78)

    n_with_photon = 0
    tot_photon_q = 0
    active_class_tally = {}
    n_signal_match = 0

    for p in files:
        with h5py.File(p, "r") as f:
            a = f.attrs
            run = int(a.get("stage3_meta_run", -1))
            sub = int(a.get("stage3_meta_subrun", -1))
            evt = int(a.get("stage3_meta_event", -1))

            if "stage3_queries" not in f or "class_argmax" not in f["stage3_queries"]:
                # event dropped / no queries
                print("%-46s %5s %4s %s"
                      % ("(%d,%d,%d)" % (run, sub, evt), "-", "-", "(no stage3_queries — dropped)"))
                if (run, sub, evt) in signal:
                    n_signal_match += 1
                continue
            qcls = f["stage3_queries"]["class_argmax"][:]
            qact = f["stage3_queries"]["is_active_postdedup"][:].astype(bool)
            active_cls = qcls[qact]
            for c in active_cls:
                active_class_tally[int(c)] = active_class_tally.get(int(c), 0) + 1
            n_phot_q = int((active_cls == args.photon_class).sum())
            tot_photon_q += n_phot_q
            if n_phot_q > 0:
                n_with_photon += 1
            if (run, sub, evt) in signal:
                n_signal_match += 1

        names = ",".join(CLASS_NAMES.get(int(c), str(int(c))) for c in active_cls)
        flag = " <-- PHOTON" if n_phot_q > 0 else ""
        print("%-46s %5d %4d %s%s"
              % ("(%d,%d,%d)" % (run, sub, evt), int(qact.sum()), n_phot_q, names, flag))

    print("-" * 78)
    print("events                        : %d" % len(files))
    if signal:
        print("  matching signal_events.csv  : %d" % n_signal_match)
    print("events with >=1 photon query  : %d" % n_with_photon)
    print("total predicted photon queries: %d" % tot_photon_q)
    print("active-query class tally      : %s"
          % {CLASS_NAMES.get(k, k): v for k, v in sorted(active_class_tally.items())})


if __name__ == "__main__":
    main()
