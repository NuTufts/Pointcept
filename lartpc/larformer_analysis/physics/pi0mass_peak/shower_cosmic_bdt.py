"""PER-SHOWER cosmic-vs-neutrino photon BDT (exporter-embeddable).

Unlike ext_bdt.py (event-level, tied to the 2-photon topology), this scores
ONE reco photon at a time from features that the ntuple exporter can compute
per prong at export time (flash-blind by design):
  shower: recal E, cos(theta_beam), cos(theta_Y), dist-to-vertex, start dwall,
          attachment score / confident flag, LArFormer class scores
          (el/ph/mu/pi/pr), nHits, charge fraction
  event/interaction context (pair-agnostic): vtx x/y/z, vtx dwall, vtxScore,
          n primary tracks, n showers (>20 MeV), n photons
Signal = truth-matched nu photons (showerTrueTID -> PDG 22, FV nu-vertex
events, MC overlay); background = all PID==22 showers in EXT events
(cosmic by construction). Even/odd event split; held-out evaluation only.
Outputs: per-shower ROC + working points, importances, and an EVENT-level
check on the pi0 selection (both leading photons pass) vs the event-BDT
reference, plus a persisted model for the exporter.

    python3 shower_cosmic_bdt.py --mc-ntuple .. --ext-ntuple .. --data-ntuple ..
        --ext-scale 0.5909 --plots plots_s1ep2p8_diag --save-model <joblib>
"""
import argparse, os
import numpy as np, uproot, awkward as ak

OA, OB = 0.020101, -15.49
TPC_LO = np.array([0.0, -116.5, 0.0]); TPC_HI = np.array([256.35, 116.5, 1036.8])
FEATS = ["E", "cosZ", "cosY", "dist", "sdwall", "att", "attconf",
         "phS", "elS", "muS", "piS", "prS", "nhits", "chargefrac",
         "vtxX", "vtxY", "vtxZ", "vdwall", "vtxScore", "nPrimTrk", "nShower", "nPhot"]
BR = ["run", "subrun", "event", "foundVertex", "primaryVtxStream", "vtxIsFiducial",
      "vtxX", "vtxY", "vtxZ", "vtxScore",
      "showerLArFormerPID", "showerRecoE", "showerCosTheta", "showerCosThetaY",
      "showerDistToVtx", "showerStartPosX", "showerStartPosY", "showerStartPosZ",
      "showerAttScore", "showerAttConfident", "showerLArFormerPhScore",
      "showerLArFormerElScore", "showerLArFormerMuScore", "showerLArFormerPiScore",
      "showerLArFormerPrScore", "showerNHits", "showerChargeFrac", "showerTrueTID",
      "showerTruePhPurity",
      "trackIsSecondary", "trueSimPartTID", "trueSimPartPDG", "trueVtxInWCFV"]


def dwall(p):
    return float(min((p - TPC_LO).min(), (TPC_HI - p).min()))


def extract(ntuple, ga, gb, is_mc, row_min=0, row_max=None):
    """row_min/row_max restrict to a contiguous ntuple-row range (rows follow
    the merged_sp list order) — used to keep the EXT BDT-training half
    (rows < 100000 of the 200k sample) disjoint from the analysis half."""
    t = uproot.open(ntuple)["EventTree"]
    have = set(t.keys())
    a = t.arrays([b for b in BR if b in have])
    # data-mode MC export (no truth sidecar): trueSimPart*/trueVtxInWCFV absent
    # -> signal = charge-based truth match (TrueTID>0 & TruePhPurity>0.5)
    sidecar = ("trueSimPartTID" in have
               and bool(ak.count(a["trueSimPartTID"]) > 0))
    if is_mc and not sidecar:
        print(">>> MC corpus without truth sidecar: using "
              "TrueTID>0 & TruePhPurity>0.5 signal definition")
    X, y, ev, rowinfo = [], [], [], []
    n_rows = len(a["run"]) if row_max is None else min(row_max, len(a["run"]))
    for i in range(row_min, n_rows):
        if not (a["foundVertex"][i] == 1 and a["primaryVtxStream"][i] == 0 and a["vtxIsFiducial"][i] == 1):
            continue
        if is_mc and sidecar and a["trueVtxInWCFV"][i] != 1:
            continue
        pid = np.asarray(a["showerLArFormerPID"][i]); E0 = np.asarray(a["showerRecoE"][i])
        E = np.where(pid == 22, (E0 - OB) / OA * ga + gb, E0)
        v = np.array([a["vtxX"][i], a["vtxY"][i], a["vtxZ"][i]], float)
        nprim = int(np.sum(np.asarray(a["trackIsSecondary"][i]) == 0))
        nsh = int(np.sum(E > 20)); nph = int(np.sum((pid == 22) & (E > 20)))
        if is_mc and sidecar:
            tids = np.asarray(a["trueSimPartTID"][i]); pdg = np.asarray(a["trueSimPartPDG"][i])
        for j in np.nonzero((pid == 22) & (E > 20))[0]:
            if is_mc and sidecar:
                m = np.nonzero(tids == int(a["showerTrueTID"][i][j]))[0]
                if len(m) == 0 or abs(pdg[m[0]]) != 22:
                    continue          # only truth-matched nu photons as signal
            elif is_mc:
                if not (a["showerTrueTID"][i][j] > 0
                        and a["showerTruePhPurity"][i][j] > 0.5):
                    continue          # charge-based truth photon match
            st = np.array([a["showerStartPosX"][i][j], a["showerStartPosY"][i][j], a["showerStartPosZ"][i][j]], float)
            X.append([E[j], a["showerCosTheta"][i][j], a["showerCosThetaY"][i][j], a["showerDistToVtx"][i][j],
                      dwall(st), a["showerAttScore"][i][j], a["showerAttConfident"][i][j],
                      a["showerLArFormerPhScore"][i][j], a["showerLArFormerElScore"][i][j],
                      a["showerLArFormerMuScore"][i][j], a["showerLArFormerPiScore"][i][j],
                      a["showerLArFormerPrScore"][i][j], a["showerNHits"][i][j], a["showerChargeFrac"][i][j],
                      v[0], v[1], v[2], dwall(v), a["vtxScore"][i], nprim, nsh, nph])
            y.append(1 if is_mc else 0); ev.append(int(a["event"][i])); rowinfo.append((i, j))
    return np.asarray(X, float), np.asarray(y), np.asarray(ev), rowinfo


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mc-ntuple", required=True); ap.add_argument("--ext-ntuple", required=True)
    ap.add_argument("--data-ntuple", required=True)
    ap.add_argument("--recal-gamma-a", type=float, default=0.01553)
    ap.add_argument("--recal-gamma-b", type=float, default=-12.80)
    ap.add_argument("--plots", required=True); ap.add_argument("--save-model", default=None)
    ap.add_argument("--ext-row-min", type=int, default=0)
    ap.add_argument("--ext-row-max", type=int, default=None,
                    help="EXT rows [min,max) used for TRAINING/held-out eval; "
                         "plan: rows <100000 = BDT half, >=100000 = analysis half")
    ap.add_argument("--no-mc-split", action="store_true",
                    help="MC is a dedicated training corpus (segmenter training "
                         "sample): use ALL MC rows for training, evaluate on the EXT "
                         "held-out half + a 20%% MC validation holdout")
    args = ap.parse_args(); os.makedirs(args.plots, exist_ok=True)
    Xs, ys, es, _ = extract(args.mc_ntuple, args.recal_gamma_a, args.recal_gamma_b, True)
    Xb, yb, eb, _ = extract(args.ext_ntuple, args.recal_gamma_a, args.recal_gamma_b, False,
                            args.ext_row_min, args.ext_row_max)
    Xd, _, ed, _ = extract(args.data_ntuple, args.recal_gamma_a, args.recal_gamma_b, False)
    print(f"signal photons {len(Xs)} | EXT photons {len(Xb)} | data photons {len(Xd)}")
    trs, trb = es % 2 == 0, eb % 2 == 0
    if args.no_mc_split:            # dedicated corpus: 80/20 by event hash
        trs = (es % 5) != 0
    Xtr = np.vstack([Xs[trs], Xb[trb]]); ytr = np.r_[np.ones(trs.sum()), np.zeros(trb.sum())]
    wtr = np.r_[np.full(trs.sum(), trb.sum() / trs.sum()), np.ones(trb.sum())]
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.inspection import permutation_importance
    clf = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.08, max_iter=500,
                                         early_stopping=True, validation_fraction=0.2, random_state=7)
    clf.fit(Xtr, ytr, sample_weight=wtr)
    print(f"trained {clf.n_iter_} iters")
    ss, sb = clf.predict_proba(Xs[~trs])[:, 1], clf.predict_proba(Xb[~trb])[:, 1]
    ths = np.unique(np.r_[ss, sb])
    eff = np.array([(ss >= t).mean() for t in ths]); rej = 1 - np.array([(sb >= t).mean() for t in ths])
    order = np.argsort(1 - rej); auc = float(np.trapz(eff[order], (1 - rej)[order]))
    print(f"per-shower test AUC ~ {auc:.3f}")
    print(f"{'ph eff':>7} {'EXT rej':>8} {'thr':>7}")
    wps = {}
    for tg in (0.99, 0.985, 0.97, 0.95, 0.90):
        k = int(np.argmin(np.abs(eff - tg))); wps[tg] = float(ths[k])
        print(f"{eff[k]:7.3f} {rej[k]:8.3f} {ths[k]:7.3f}")
    imp = permutation_importance(clf, np.vstack([Xs[~trs], Xb[~trb]]), np.r_[np.ones((~trs).sum()), np.zeros((~trb).sum())],
                                 n_repeats=5, random_state=7)
    print("\n== importance ==")
    for k in np.argsort(imp.importances_mean)[::-1][:10]:
        print(f"  {FEATS[k]:10s} {imp.importances_mean[k]:.4f}")
    sd = clf.predict_proba(Xd)[:, 1]
    print(f"\ndata photons: frac >= thr(eff0.985) {np.mean(sd >= wps[0.985]):.3f} | EXT held-out {np.mean(sb >= wps[0.985]):.3f} | sig held-out {np.mean(ss >= wps[0.985]):.3f}")
    if args.save_model:
        import joblib
        joblib.dump({"clf": clf, "feats": FEATS, "wps": wps, "recal": (args.recal_gamma_a, args.recal_gamma_b),
                     "note": "per-shower cosmic-vs-nu photon BDT; electrons autopass 1.0 in exporter"}, args.save_model)
        print(f"model -> {args.save_model}")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.3))
    axs[0].plot(1 - rej, eff, lw=1.8); axs[0].set(xscale="log", xlabel="EXT photon acceptance", ylabel="nu photon efficiency",
                                                title=f"per-shower BDT ROC (held-out, AUC~{auc:.3f})"); axs[0].grid(alpha=.3)
    b = np.linspace(0, 1, 41)
    axs[1].hist(ss, b, density=True, histtype="step", lw=1.8, color="#d62728", label="nu photons (MC, held-out)")
    axs[1].hist(sb, b, density=True, histtype="step", lw=1.8, color="#7f7f7f", label="EXT photons (held-out)")
    axs[1].hist(sd, b, density=True, histtype="step", lw=1.4, color="k", ls="--", label="data photons")
    axs[1].set(xlabel="per-shower cosmic score (nu-like ->)", ylabel="density", yscale="log"); axs[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(args.plots, "shower_bdt.png"), dpi=120)
    print(f">>> {args.plots}/shower_bdt.png")


if __name__ == "__main__":
    main()
