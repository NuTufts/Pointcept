"""Apply the event-level EXT-rejection BDT to a pi0 selection table.

Replaces the inline python that the cew6 suite scripts carried. Re-derives
the two-photon candidates from the ntuple (datamc_diagnostics.load, same
shower-BDT / muon-finder / chi2 gates as the table build), scores them with
the saved HistGradientBoosting model, and writes a copy of the table in
which events scoring below --te have sel_ge2/sel_eq2 cleared.

--hygiene cew6 : the classifier training conventions of the run-3 cew6
    samples -- MC true-signal EVEN events and EXT rows<100k or EVEN events
    are training events (sel cleared); the held-out odd halves get w x2.
--hygiene none : the sample never fed any classifier (e.g. the run-1
    table-gamma productions): nothing cleared, no re-weighting.
--chi2-open    : score with the flash-chi2 gates open (the *_nochi2bdt
    tables used for chi2 spectra and step tables).

    python3 apply_event_bdt.py --model ext_bdt_model_flashblind_cew6.joblib \
        --sample mc --ntuple ... --table mc_X_table.npz --out mc_X_bdt_table.npz \
        --shower-bdt-min 0.164 --te 0.21 --hygiene none
"""
import argparse
import os
import sys

import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datamc_diagnostics import load  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--sample", required=True, choices=["mc", "data", "ext"])
    ap.add_argument("--ntuple", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--te", type=float, default=0.21)
    ap.add_argument("--shower-bdt-min", type=float, default=0.164)
    ap.add_argument("--muon-finder", default="union")
    ap.add_argument("--mu-ke-min", type=float, default=50.0)
    ap.add_argument("--chi2-cc", type=float, default=1e4)
    ap.add_argument("--chi2-nc", type=float, default=1778.0)
    ap.add_argument("--chi2-open", action="store_true")
    ap.add_argument("--hygiene", default="none", choices=["none", "cew6"])
    ap.add_argument("--recal-gamma-a", type=float, default=0.020100999623537064,
                    help="identity for recal3-baked ntuples")
    ap.add_argument("--recal-gamma-b", type=float, default=-15.489999771118164)
    args = ap.parse_args()

    M = joblib.load(args.model)
    clf, feats = M["clf"], M["feats"]
    cuts = (1e12, 1e12) if args.chi2_open else (args.chi2_cc, args.chi2_nc)
    d = load(args.ntuple, args.table, args.recal_gamma_a, args.recal_gamma_b,
             args.mu_ke_min, cuts[0], cuts[1],
             shower_bdt_min=args.shower_bdt_min, muon_finder=args.muon_finder)
    d["flashPE"] = np.full(len(d["run"]), np.nan)     # flash-blind models
    d["logchi2"] = d.get("logchi2", np.full(len(d["run"]), np.nan))
    sc = clf.predict_proba(np.column_stack(
        [d[f].astype(float) for f in feats]))[:, 1]

    tab = dict(np.load(args.table, allow_pickle=True))
    n = len(tab["w"])
    fail = np.zeros(n, bool); train = np.zeros(n, bool); held = np.zeros(n, bool)
    fail[d["row"][sc < args.te]] = True
    if args.hygiene == "cew6":
        cat = tab.get("cat")
        rows, ev = d["row"], d["event"]
        if args.sample == "mc":
            sig = cat[rows] < 2
            train[rows[sig & (ev % 2 == 0)]] = True
            held[rows[sig & (ev % 2 == 1)]] = True
        elif args.sample == "ext":
            tr = (rows < 100000) | (ev % 2 == 0)
            train[rows[tr]] = True
            held[rows[~tr]] = True
    for k in ("sel_ge2", "sel_eq2"):
        tab[k] = np.asarray(tab[k]).copy()
        tab[k][fail | train] = 0
    tab["w"] = np.asarray(tab["w"], np.float64).copy()
    tab["w"][held] *= 2.0
    tab["event_bdt_score"] = np.full(n, np.nan)
    tab["event_bdt_score"][d["row"]] = sc
    np.savez(args.out, **tab)
    print(f">>> {args.sample}: scored {len(sc)} | fail(<{args.te}) "
          f"{int(fail.sum())} | hygiene={args.hygiene}: train-cleared "
          f"{int(train.sum())}, held x2 {int(held.sum())} | chi2 "
          f"{'open' if args.chi2_open else cuts} -> {args.out}")


if __name__ == "__main__":
    main()
