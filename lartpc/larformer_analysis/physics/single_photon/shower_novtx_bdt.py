"""Vertex-FREE per-shower cosmic-vs-nu photon BDT (exporter-embeddable via env
LARFORMER_SHOWER_BDT_NOVTX -> ntuple branch showerNoVtxScore).

Motivation (2026-09-06): the official per-shower BDT (showerCosmicScore) leans
on distance-to-vertex and interaction context, so it cannot score the
vertex-less prongs (showerVtxIdx == -1) that the exporter now writes for
segmenter particles outside any nu_reco interaction -- exactly where entering
single photons (56% of the signal) are lost. This model uses ONLY
vertex-independent, flash-blind features that exist for every shower prong:
  E (showerRecoE, calo-calibrated at export), cos(theta_beam), cos(theta_Y),
  start dwall, segmenter objectness (1-P(no_object)), LArFormer class scores
  (ph/el/mu/pi/pr), nHits, comb charge, hasVtx flag, n particles in the slice.
(The exporter's novtx_features() exposes more -- start x/y/z, logchi2,
chargefrac, stream -- selectable with --feats; keep flash features out for
data/MC robustness.)

Signal = MC overlay photon showers (showerLArFormerPID==22, E>20) truth-matched
to a nu photon (showerTruePID==22, showerTruePurity>=0.5, unlabeled<0.5), from
EVEN events only (POT weights); background = EXT photon showers from the
per-shower-BDT training half (rows < --ext-row-max). Held-out evaluation on
ODD MC events + EXT rows >= --ext-row-max, split by attached/vertex-less and by
in-FV/entering truth. Analyses using the score must therefore use ODD MC
events (w x2) and EXT rows >= 100k.

    python3 shower_novtx_bdt.py --mc-ntuple <novtx MC> --ext-ntuple <novtx EXT> \
        --plots plots_v2_s1ep2p8/novtx_bdt --save-model \
        ../../../larformer_reco/export/data/shower_novtx_bdt.joblib
"""
import argparse
import os
import datetime

import numpy as np
import uproot
import awkward as ak

TPC_LO = np.array([0.0, -116.5, 0.0]); TPC_HI = np.array([256.35, 116.5, 1036.8])
DEFAULT_FEATS = ["E", "cosZ", "cosY", "sdwall", "objectness", "phS", "elS",
                 "muS", "piS", "prS", "nhits", "charge", "hasVtx", "nSlicePart"]
BR = ["run", "subrun", "event", "nuSliceNParticles", "fmSliceNParticles",
      "nuSliceFlashChi2", "fmSliceFlashChi2",
      "showerLArFormerPID", "showerRecoE", "showerCosTheta", "showerCosThetaY",
      "showerStartPosX", "showerStartPosY", "showerStartPosZ",
      "showerObjectness", "showerLArFormerPhScore", "showerLArFormerElScore",
      "showerLArFormerMuScore", "showerLArFormerPiScore",
      "showerLArFormerPrScore", "showerNHits", "showerCharge",
      "showerChargeFrac", "showerVtxIdx", "showerStream", "showerTruePID",
      "showerTruePurity", "showerTrueUnlabeledPurity"]
BR_MC = ["xsecWeight", "trueVtxInWCFV"]


def shower_table(a, e_min=20.0):
    """Flatten photon showers (PID 22, E > e_min) to per-shower feature dict
    (names = exporter novtx_features keys) + bookkeeping columns."""
    is_g = (a["showerLArFormerPID"] == 22) & (a["showerRecoE"] > e_min)
    n = len(a["run"])
    evt = ak.to_numpy(ak.flatten(ak.broadcast_arrays(np.arange(n), is_g)[0][is_g]))
    f = lambda k: ak.to_numpy(ak.flatten(a[k][is_g])).astype(np.float64)
    st = np.stack([f("showerStartPosX"), f("showerStartPosY"), f("showerStartPosZ")], 1)
    stream = f("showerStream").astype(np.int64)
    nsp_nu = ak.to_numpy(a["nuSliceNParticles"])[evt]
    nsp_fm = ak.to_numpy(a["fmSliceNParticles"])[evt]
    chi_nu = ak.to_numpy(a["nuSliceFlashChi2"])[evt]
    chi_fm = ak.to_numpy(a["fmSliceFlashChi2"])[evt]
    chi2 = np.where(stream == 0, chi_nu, chi_fm)
    d = {"evt": evt, "E": f("showerRecoE"), "cosZ": f("showerCosTheta"),
         "cosY": f("showerCosThetaY"),
         "sdwall": np.minimum((st - TPC_LO).min(1), (TPC_HI - st).min(1)),
         "startX": st[:, 0], "startY": st[:, 1], "startZ": st[:, 2],
         "objectness": f("showerObjectness"),
         "phS": f("showerLArFormerPhScore"), "elS": f("showerLArFormerElScore"),
         "muS": f("showerLArFormerMuScore"), "piS": f("showerLArFormerPiScore"),
         "prS": f("showerLArFormerPrScore"), "nhits": f("showerNHits"),
         "charge": f("showerCharge"), "chargefrac": f("showerChargeFrac"),
         "hasVtx": (f("showerVtxIdx") >= 0).astype(np.float64), "stream": stream,
         "nSlicePart": np.where(stream == 0, nsp_nu, nsp_fm).astype(np.float64),
         "logchi2": np.where(chi2 > 0, np.log10(np.clip(chi2, 1e-3, None)), -1.0),
         "truePID": f("showerTruePID"), "truePur": f("showerTruePurity"),
         "unl": f("showerTrueUnlabeledPurity")}
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mc-ntuple", required=True)
    ap.add_argument("--ext-ntuple", required=True)
    ap.add_argument("--ext-row-max", type=int, default=100000,
                    help="EXT rows below this = training half")
    ap.add_argument("--pot", type=float, default=4.4e19)
    ap.add_argument("--feats", default=",".join(DEFAULT_FEATS))
    ap.add_argument("--plots", required=True)
    ap.add_argument("--save-model", default=None)
    args = ap.parse_args()
    os.makedirs(args.plots, exist_ok=True)
    feats = [x for x in args.feats.split(",") if x]
    print(f">>> features: {feats}")

    print(">>> loading MC ...", flush=True)
    fin = uproot.open(args.mc_ntuple)
    p = fin["potTree"].arrays(library="np")
    pot = float(np.sum(p["totGoodPOT"])) or float(np.sum(p["totPOT"]))
    a = fin["EventTree"].arrays(BR + BR_MC)
    M = shower_table(a)
    w0 = ak.to_numpy(a["xsecWeight"]).astype(np.float64)
    wev = np.where(w0 > 0, w0, 0.0) * (args.pot / pot)
    ev_par = ak.to_numpy(a["event"]).astype(np.int64)
    fv_ev = ak.to_numpy(a["trueVtxInWCFV"]) == 1
    del a
    M["w"] = wev[M["evt"]]
    M["even"] = (ev_par[M["evt"]] % 2 == 0)
    M["infv"] = fv_ev[M["evt"]]
    M["is_nu_g"] = (M["truePID"] == 22) & (M["truePur"] >= 0.5) & (M["unl"] < 0.5)
    M["is_cos"] = (M["truePID"] <= 0) | (M["unl"] >= 0.5)
    print(f">>> MC photon showers {len(M['E'])}: nu photons {int(M['is_nu_g'].sum())} "
          f"(vertex-less {int((M['is_nu_g'] & (M['hasVtx']==0)).sum())}, "
          f"entering-event {int((M['is_nu_g'] & ~M['infv']).sum())}), "
          f"overlay-cosmic {int(M['is_cos'].sum())}")

    print(">>> loading EXT ...", flush=True)
    ax = uproot.open(args.ext_ntuple)["EventTree"].arrays(BR)
    X = shower_table(ax)
    del ax
    X["train"] = X["evt"] < args.ext_row_max
    print(f">>> EXT photon showers {len(X['E'])}: training half {int(X['train'].sum())}, "
          f"analysis half {int((~X['train']).sum())}; vertex-less "
          f"{int((X['hasVtx']==0).sum())}")

    fm = lambda d, m: np.column_stack([d[k][m] for k in feats])
    sig_tr = M["is_nu_g"] & M["even"]
    sig_te = M["is_nu_g"] & ~M["even"]
    X_tr = np.vstack([fm(M, sig_tr), fm(X, X["train"])])
    y_tr = np.r_[np.ones(sig_tr.sum()), np.zeros(X["train"].sum())]
    w_tr = np.r_[M["w"][sig_tr], np.ones(X["train"].sum())]
    w_tr[y_tr == 1] *= w_tr[y_tr == 0].sum() / max(w_tr[y_tr == 1].sum(), 1e-9)
    print(f">>> train: sig {int(sig_tr.sum())} | EXT {int(X['train'].sum())}")

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.inspection import permutation_importance
    clf = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.06,
                                         max_iter=500, early_stopping=True,
                                         validation_fraction=0.2, random_state=7)
    clf.fit(X_tr, y_tr, sample_weight=w_tr)
    print(f">>> trained: {clf.n_iter_} iters")
    M["score"] = clf.predict_proba(fm(M, np.ones(len(M["E"]), bool)))[:, 1]
    X["score"] = clf.predict_proba(fm(X, np.ones(len(X["E"]), bool)))[:, 1]

    # held-out ROC / working points
    thrs = np.linspace(0, 0.99, 100)
    def passfrac(w, s):
        return np.array([w[s >= t].sum() / max(w.sum(), 1e-9) for t in thrs])
    ext_te = ~X["train"]
    ext_pass = passfrac(np.ones(ext_te.sum()), X["score"][ext_te])
    groups = [("all nu photons", sig_te, "#d62728", "-"),
              ("attached (hasVtx)", sig_te & (M["hasVtx"] == 1), "#ff9896", "--"),
              ("vertex-less", sig_te & (M["hasVtx"] == 0), "#9467bd", "-"),
              ("entering-event photons", sig_te & ~M["infv"], "#ff7f0e", "-"),
              ("in-FV-event photons", sig_te & M["infv"], "#1f77b4", "--")]
    lines = [f"{'thr':>6}" + "".join(f"{g[0][:18]:>20}" for g in groups)
             + f"{'EXT rej':>9}{'EXTnovtx rej':>13}"]
    ext_nv = ext_te & (X["hasVtx"] == 0)
    ext_nv_pass = passfrac(np.ones(ext_nv.sum()), X["score"][ext_nv])
    curves = {g[0]: passfrac(M["w"][g[1]], M["score"][g[1]]) for g in groups}
    for t in (0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9):
        k = int(np.argmin(np.abs(thrs - t)))
        lines.append(f"{thrs[k]:6.2f}" + "".join(f"{curves[g[0]][k]:20.3f}" for g in groups)
                     + f"{1-ext_pass[k]:9.3f}{1-ext_nv_pass[k]:13.3f}")
    auc = float(np.trapz(np.clip(curves["all nu photons"], 0, 1)[np.argsort(ext_pass)],
                         np.sort(ext_pass)))
    txt = f"held-out (odd MC, EXT rows>={args.ext_row_max}); AUC ~ {auc:.3f}\n" + "\n".join(lines)
    print("\n" + txt)
    with open(os.path.join(args.plots, "novtx_bdt_table.txt"), "w") as fh:
        fh.write(txt + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.6))
    ax = axs[0]
    bins = np.linspace(0, 1, 41)
    for lab, m, col, ls in groups[:3]:
        ax.hist(np.clip(M["score"][m], 0, 1 - 1e-6), bins=bins, weights=M["w"][m],
                histtype="step", lw=1.8, color=col, ls=ls, label=f"MC {lab}")
    ax.hist(np.clip(X["score"][ext_te], 0, 1 - 1e-6), bins=bins, histtype="stepfilled",
            alpha=0.3, color="k", label="EXT photons (analysis half)")
    ax.hist(np.clip(X["score"][ext_nv], 0, 1 - 1e-6), bins=bins, histtype="step",
            color="k", ls=":", label="EXT vertex-less")
    ax.set(xlabel="vertex-free score", yscale="log", title="score distributions (held-out)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    ax = axs[1]
    for lab, m, col, ls in groups:
        ax.plot(thrs, curves[lab], ls=ls, color=col, label=f"{lab} (N={int(m.sum())})")
    ax.plot(thrs, 1 - ext_pass, "k-", lw=2, label="EXT photon rejection")
    ax.plot(thrs, 1 - ext_nv_pass, "k:", lw=1.5, label="EXT vertex-less rejection")
    ax.set(xlabel="score threshold", ylabel="pass fraction / rejection", ylim=(0, 1.02),
           title="per-shower efficiency vs threshold")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    ax = axs[2]
    for lab, m, col, ls in groups:
        ax.plot(1 - ext_pass, curves[lab], ls=ls, color=col, label=lab)
    ax.set(xlabel="EXT photon rejection", ylabel="nu photon pass fraction",
           xlim=(0, 1.02), ylim=(0, 1.02), title=f"ROC (held-out), AUC~{auc:.3f}")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{args.plots}/novtx_bdt_performance.png", dpi=110)
    plt.close(fig)

    # feature importances (permutation, held-out signal vs EXT sample)
    Xh = np.vstack([fm(M, sig_te), fm(X, ext_te)])
    yh = np.r_[np.ones(sig_te.sum()), np.zeros(ext_te.sum())]
    wh = np.r_[M["w"][sig_te], np.ones(ext_te.sum())]
    wh[yh == 1] *= wh[yh == 0].sum() / max(wh[yh == 1].sum(), 1e-9)
    rng = np.random.default_rng(3)
    sub = rng.choice(len(yh), size=min(40000, len(yh)), replace=False)
    pi = permutation_importance(clf, Xh[sub], yh[sub], sample_weight=wh[sub],
                                n_repeats=3, random_state=1, scoring="roc_auc")
    order = np.argsort(pi.importances_mean)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.barh([feats[i] for i in order], pi.importances_mean[order],
            xerr=pi.importances_std[order])
    ax.set(xlabel="permutation importance (AUC drop)", title="vertex-free BDT features")
    fig.tight_layout(); fig.savefig(f"{args.plots}/novtx_bdt_importance.png", dpi=110)
    plt.close(fig)
    print(">>> importances: " + ", ".join(f"{feats[i]} {pi.importances_mean[i]:.3f}"
                                         for i in order[::-1]))

    if args.save_model:
        import joblib
        joblib.dump({"clf": clf, "feats": feats,
                     "trained": datetime.date.today().isoformat(),
                     "signal": "MC overlay nu photons (even events)",
                     "background": f"EXT photon showers rows<{args.ext_row_max}",
                     "note": "vertex-free, flash-blind; apply via exporter env "
                             "LARFORMER_SHOWER_BDT_NOVTX -> showerNoVtxScore"},
                    args.save_model)
        print(f">>> model -> {args.save_model}")
    np.savez(os.path.join(args.plots, "novtx_bdt_scores.npz"),
             mc_evt=M["evt"], mc_score=M["score"], mc_is_nu_g=M["is_nu_g"],
             mc_hasVtx=M["hasVtx"], mc_even=M["even"],
             ext_evt=X["evt"], ext_score=X["score"], ext_hasVtx=X["hasVtx"])
    print(f">>> plots -> {args.plots}")


if __name__ == "__main__":
    main()
