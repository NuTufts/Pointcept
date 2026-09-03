"""Shower energy resolution vs cluster completeness / purity (reco performance).

Truth-matched photons (showerTrueTID -> trueSimPart PDG 22) in the new-chain
MC overlay ntuple, recalibrated energy E = a*Q + b, actual energy = |p_true|.
Measures, from the ntuple alone (showerTrueComp / showerTruePurity):
  - E_reco/E_true distribution overall and per completeness bin (median, IQR)
  - completeness & purity distributions (how often is a photon well-clustered?)
  - resolution headroom: IQR at comp>0.9 vs all
  - pi0 pairs: m_gg width vs min(comp1, comp2)
  - correlations: completeness vs conversion distance / energy
"""
import argparse, os
import numpy as np, uproot, awkward as ak

OA, OB = 0.020101, -15.49


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ntuple", required=True)
    ap.add_argument("--recal-gamma-a", type=float, default=0.01556)
    ap.add_argument("--recal-gamma-b", type=float, default=-11.47)
    ap.add_argument("--plots", required=True)
    args = ap.parse_args()
    os.makedirs(args.plots, exist_ok=True)
    t = uproot.open(args.ntuple)["EventTree"]
    a = t.arrays(["foundVertex", "primaryVtxStream", "vtxIsFiducial",
                  "vtxX", "vtxY", "vtxZ",
                  "showerLArFormerPID", "showerRecoE", "showerTrueTID",
                  "showerTrueComp", "showerTruePurity", "showerDistToVtx",
                  "showerStartPosX", "showerStartPosY", "showerStartPosZ",
                  "trueSimPartTID", "trueSimPartPDG",
                  "trueSimPartPx", "trueSimPartPy", "trueSimPartPz",
                  "truePrimPartPDG", "trueVtxInWCFV"])
    n = len(a["vtxX"])
    R, C, P, D, Et, Er, EV = [], [], [], [], [], [], []
    pairs = []      # (mgg, min comp)
    for i in range(n):
        if not (a["foundVertex"][i] == 1 and a["primaryVtxStream"][i] == 0
                and a["vtxIsFiducial"][i] == 1 and a["trueVtxInWCFV"][i] == 1):
            continue
        pid = np.asarray(a["showerLArFormerPID"][i])
        E0 = np.asarray(a["showerRecoE"][i])
        E = np.where(pid == 22, (E0 - OB) / OA * args.recal_gamma_a
                     + args.recal_gamma_b, E0)
        tids = np.asarray(a["trueSimPartTID"][i]); pdg = np.asarray(a["trueSimPartPDG"][i])
        Pm = np.stack([np.asarray(a["trueSimPartPx"][i]), np.asarray(a["trueSimPartPy"][i]),
                       np.asarray(a["trueSimPartPz"][i])], 1)
        comp = np.asarray(a["showerTrueComp"][i]); pur = np.asarray(a["showerTruePurity"][i])
        dist = np.asarray(a["showerDistToVtx"][i])
        v = np.array([a["vtxX"][i], a["vtxY"][i], a["vtxZ"][i]], float)
        ph = []
        for j in np.nonzero((pid == 22) & (E > 20))[0]:
            m = np.nonzero(tids == int(a["showerTrueTID"][i][j]))[0]
            if len(m) == 0 or abs(pdg[m[0]]) != 22:
                continue
            ea = float(np.linalg.norm(Pm[m[0]]))
            if ea < 20:
                continue
            R.append(E[j] / ea); C.append(comp[j]); P.append(pur[j])
            D.append(dist[j]); Et.append(ea); Er.append(E[j]); EV.append(i)
            st = np.array([a["showerStartPosX"][i][j], a["showerStartPosY"][i][j],
                           a["showerStartPosZ"][i][j]], float)
            dv = st - v
            ph.append((E[j], comp[j], dv / max(np.linalg.norm(dv), 1e-9)))
        if int(np.sum(np.asarray(a["truePrimPartPDG"][i]) == 111)) == 1 and len(ph) >= 2:
            ph = sorted(ph, key=lambda x: -x[0])[:2]
            m = np.sqrt(max(2 * ph[0][0] * ph[1][0] * (1 - np.dot(ph[0][2], ph[1][2])), 0))
            pairs.append((m, min(ph[0][1], ph[1][1])))
    R, C, P, D, Et = map(np.asarray, (R, C, P, D, Et))
    pairs = np.array(pairs)
    print(f"truth-matched photons: {len(R)} | pi0 pairs: {len(pairs)}")

    def iqr(x): return float(np.subtract(*np.percentile(x, [75, 25])))
    def stats(m, lab):
        r = R[m]
        print(f"  {lab:>26}: N {len(r):5d} | median E/Etrue {np.median(r):.3f} | "
              f"IQR {iqr(r):.3f} | IQR/med {iqr(r)/np.median(r):.3f}")
    print("\n== E_reco/E_true vs completeness ==")
    stats(np.ones(len(R), bool), "all")
    edges = [0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.01]
    for lo, hi in zip(edges[:-1], edges[1:]):
        stats((C >= lo) & (C < hi), f"comp [{lo:.2f},{hi:.2f})")
    print("\n== E_reco/E_true vs purity ==")
    for lo, hi in zip(edges[:-1], edges[1:]):
        stats((P >= lo) & (P < hi), f"pur [{lo:.2f},{hi:.2f})")
    print(f"\n  completeness: median {np.median(C):.3f}, frac >0.9 {np.mean(C>0.9):.3f}, "
          f"frac <0.7 {np.mean(C<0.7):.3f}")
    print(f"  purity      : median {np.median(P):.3f}, frac >0.9 {np.mean(P>0.9):.3f}")
    hi_ = (C > 0.9) & (P > 0.9)
    print(f"  HEADROOM: IQR/med all {iqr(R)/np.median(R):.3f} -> comp&pur>0.9 "
          f"{iqr(R[hi_])/np.median(R[hi_]):.3f} (N {int(hi_.sum())})")
    print("\n== pi0 m_gg vs min photon completeness ==")
    for lo, hi in zip([0, 0.7, 0.9], [0.7, 0.9, 1.01]):
        m = (pairs[:, 1] >= lo) & (pairs[:, 1] < hi)
        mg = pairs[m, 0]
        if len(mg) > 20:
            print(f"  min comp [{lo:.1f},{hi:.1f}): N {len(mg):4d} | median {np.median(mg):.1f} | "
                  f"IQR {iqr(mg):.1f} | frac in 100-170 {np.mean((mg>100)&(mg<170)):.3f}")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.6))
    ax = axs[0, 0]; ax.hist2d(C, R, bins=[np.linspace(0, 1, 41), np.linspace(0, 2, 41)], norm=LogNorm(), cmap="viridis")
    ax.set(xlabel="true completeness", ylabel="E_reco / E_true", title="resolution vs completeness")
    ax = axs[0, 1]; ax.hist2d(P, R, bins=[np.linspace(0, 1, 41), np.linspace(0, 2, 41)], norm=LogNorm(), cmap="viridis")
    ax.set(xlabel="true purity", ylabel="E_reco / E_true", title="resolution vs purity")
    ax = axs[0, 2]
    ax.hist(C, np.linspace(0, 1, 41), histtype="step", lw=1.8, label="completeness")
    ax.hist(P, np.linspace(0, 1, 41), histtype="step", lw=1.8, label="purity")
    ax.set(xlabel="fraction", ylabel="photons", title="clustering quality"); ax.legend(fontsize=8)
    ax = axs[1, 0]
    cb = np.linspace(0, 1, 11); cc = 0.5 * (cb[:-1] + cb[1:]); med, q1, q3 = [], [], []
    for lo, hi in zip(cb[:-1], cb[1:]):
        m = (C >= lo) & (C < hi)
        if m.sum() > 20:
            med.append(np.median(R[m])); q1.append(np.percentile(R[m], 25)); q3.append(np.percentile(R[m], 75))
        else:
            med.append(np.nan); q1.append(np.nan); q3.append(np.nan)
    ax.errorbar(cc, med, yerr=[np.array(med) - np.array(q1), np.array(q3) - np.array(med)], fmt="ko", ms=4)
    ax.axhline(1, color="r", lw=1); ax.set(xlabel="true completeness", ylabel="median E/Etrue (IQR bars)", title="profile")
    ax = axs[1, 1]; ax.hist2d(D, C, bins=[np.linspace(0, 100, 41), np.linspace(0, 1, 41)], norm=LogNorm(), cmap="viridis")
    ax.set(xlabel="conv. distance [cm]", ylabel="completeness", title="completeness vs conversion distance")
    ax = axs[1, 2]; ax.hist2d(Et, C, bins=[np.linspace(0, 600, 41), np.linspace(0, 1, 41)], norm=LogNorm(), cmap="viridis")
    ax.set(xlabel="E_true [MeV]", ylabel="completeness", title="completeness vs energy")
    fig.tight_layout(); fig.savefig(os.path.join(args.plots, "shower_completeness.png"), dpi=115)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for lo, hi, col in ((0, 0.7, "#d62728"), (0.7, 0.9, "#ff7f0e"), (0.9, 1.01, "#2ca02c")):
        m = (pairs[:, 1] >= lo) & (pairs[:, 1] < hi)
        ax.hist(pairs[m, 0], np.linspace(0, 400, 41), histtype="step", lw=1.8, density=True,
                color=col, label=f"min comp [{lo:.1f},{hi:.1f}) N={int(m.sum())}")
    ax.axvline(135, color="k", ls=":", lw=1); ax.set(xlabel="m_gg [MeV]", ylabel="density", title="pi0 peak vs pair completeness")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(args.plots, "pi0_vs_completeness.png"), dpi=115)
    np.savez(os.path.join(args.plots, "shower_completeness.npz"), R=R, C=C, P=P, D=D, Et=Et, pairs=pairs)
    print(f">>> {args.plots}/shower_completeness.png, pi0_vs_completeness.png")


if __name__ == "__main__":
    main()
