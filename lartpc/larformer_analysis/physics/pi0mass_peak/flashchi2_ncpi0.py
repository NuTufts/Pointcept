"""Flash-match chi2 for the reco-NC two-photon pi0 selection: data vs MC.

Motivation (user hypothesis): the reco-NC data excess sits partly under the
pi0 mass peak and may be cosmogenic pi0s (e.g. from cosmic-neutron
interactions). Cosmics are out-of-time, so their nu-slice flash prediction
should match the in-time PMT flash poorly -> a high flash-chi2 tail present
in DATA but not in the (in-time-by-construction) neutrino-overlay MC. This
script isolates the reco-NC, exactly-2-photon sample and overlays the
primary-vertex flash-chi2 (log10) for data vs POT-scaled MC, then quantifies
how a high-chi2 cut changes the near-peak (100-170 MeV) data/MC ratio.

The flash-chi2 lives in the jagged recoVtx table (recoVtxFlashChi2); the
primary vertex is matched to its recoVtx row by coordinate (stream-matched).

    PYTHONPATH=./ python3 .../flashchi2_ncpi0.py \
        --mc-ntuple .../dlgen2_..._67k.root \
        --data-ntuple .../dlgen2_larformer_ntuple_bnb5e19_full.root \
        --plots plots_datamc_full/
"""
import argparse
import os

import numpy as np
import uproot
import awkward as ak

from pi0_mass_analysis import (CATS, CAT_COLORS, PI0_MASS, RECO_G_MIN,
                               MU_KE_MIN, truth_category)

PEAK_LO, PEAK_HI = 100.0, 170.0   # near-peak mass window [MeV]


def load(ntuple, is_data, pot):
    """Return per-event arrays for the reco-NC pi0 study."""
    fin = uproot.open(ntuple)
    if is_data:
        scale = 1.0
    else:
        p = fin["potTree"].arrays(library="np")
        psum = float(np.sum(p["totGoodPOT"])) or float(np.sum(p["totPOT"]))
        scale = pot / psum
    t = fin["EventTree"]
    a = t.arrays(["xsecWeight", "trueVtxInWCFV", "trueNuCCNC",
                  "trueSimPartPDG", "trueSimPartTID", "trueSimPartMID",
                  "trueSimPartProcess", "trueSimPartPixelSumQ",
                  "truePrimPartPDG",
                  "foundVertex", "primaryVtxStream", "vtxIsFiducial",
                  "vtxX", "vtxY", "vtxZ",
                  "recoVtxX", "recoVtxY", "recoVtxZ", "recoVtxStream",
                  "recoVtxFlashChi2",
                  "showerLArFormerPID", "showerRecoE",
                  "showerStartPosX", "showerStartPosY", "showerStartPosZ",
                  "trackLArFormerPID", "trackIsSecondary", "trackRecoE"])
    n = len(a["vtxX"])
    if is_data:
        w = np.ones(n)
        cat = np.zeros(n, np.int64)
    else:
        w0 = np.asarray(a["xsecWeight"], np.float64)
        w = np.where(w0 > 0, w0, 0.0) * scale
        cat = np.array([truth_category(a, i) for i in range(n)], np.int64)

    vok = ((np.asarray(a["foundVertex"]) == 1)
           & (np.asarray(a["primaryVtxStream"]) == 0)
           & (np.asarray(a["vtxIsFiducial"]) == 1))
    is_g = (a["showerLArFormerPID"] == 22) & (a["showerRecoE"] > RECO_G_MIN)
    n_g = ak.to_numpy(ak.sum(is_g, axis=1))
    is_mu = ((a["trackLArFormerPID"] == 13) & (a["trackIsSecondary"] == 0)
             & (a["trackRecoE"] > MU_KE_MIN))
    reco_cc = ak.to_numpy(ak.any(is_mu, axis=1))

    mA = np.full(n, np.nan)
    chi2 = np.full(n, np.nan)
    for i in np.nonzero(vok & (n_g >= 2))[0]:
        gi = np.nonzero(ak.to_numpy(is_g[i]))[0]
        E = ak.to_numpy(a["showerRecoE"][i])[gi]
        o = gi[np.argsort(E)[::-1][:2]]
        E1, E2 = ak.to_numpy(a["showerRecoE"][i])[o].tolist()
        v = np.array([a["vtxX"][i], a["vtxY"][i], a["vtxZ"][i]], np.float64)
        sp = np.stack([ak.to_numpy(a[f"showerStartPos{c}"][i])[o]
                       for c in "XYZ"], 1).astype(np.float64)
        dA = sp - v
        nA = np.linalg.norm(dA, axis=1)
        if np.all(nA > 1e-3):
            dA = dA / nA[:, None]
            mA[i] = float(np.sqrt(max(2.0 * E1 * E2 * (1.0 - dA[0] @ dA[1]),
                                      0.0)))
        # primary-vertex flash chi2 via stream-matched coordinate match
        rx = ak.to_numpy(a["recoVtxX"][i]); ry = ak.to_numpy(a["recoVtxY"][i])
        rz = ak.to_numpy(a["recoVtxZ"][i]); st = ak.to_numpy(a["recoVtxStream"][i])
        c2 = ak.to_numpy(a["recoVtxFlashChi2"][i])
        if len(rx):
            d = np.sqrt((rx - v[0])**2 + (ry - v[1])**2 + (rz - v[2])**2)
            d = np.where(st == a["primaryVtxStream"][i], d, 1e9)
            j = int(np.argmin(d))
            if d[j] < 1.0 and c2[j] >= 0:
                chi2[i] = c2[j]

    sel2p = vok & (n_g >= 2) & np.isfinite(mA)
    sel2x = sel2p & (n_g == 2)
    return dict(cat=cat, w=w, n_g=n_g, reco_cc=reco_cc, m=mA,
                chi2=chi2, sel2p=sel2p, sel2x=sel2x, scale=scale)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mc-ntuple", required=True)
    ap.add_argument("--data-ntuple", required=True)
    ap.add_argument("--plots", required=True)
    ap.add_argument("--pot", type=float, default=4.4e19)
    ap.add_argument("--chi2-cut", type=float, default=1000.0,
                    help="high-chi2 (out-of-time) cut for the ratio table")
    args = ap.parse_args()
    os.makedirs(args.plots, exist_ok=True)

    mc = load(args.mc_ntuple, False, args.pot)
    da = load(args.data_ntuple, True, args.pot)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lbins = np.linspace(0, 8, 33)     # log10(chi2)
    lcut = np.log10(args.chi2_cut)

    def sel(d, eq2):
        s = (d["sel2x"] if eq2 else d["sel2p"]) & (~d["reco_cc"]) \
            & np.isfinite(d["chi2"])
        return s

    for eq2, vlab, fname in ((True, "exactly 2", "flashchi2_ncpi0_eq2"),
                             (False, ">=2", "flashchi2_ncpi0_ge2")):
        sm = sel(mc, eq2); sd = sel(da, eq2)
        lchi_mc = np.log10(np.clip(mc["chi2"], 1, None))
        lchi_da = np.log10(np.clip(da["chi2"], 1, None))
        ctr = 0.5 * (lbins[:-1] + lbins[1:])

        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
        for ax, peak_only, ttl in (
                (axes[0], False, "all m$_{\\gamma\\gamma}$"),
                (axes[1], True, f"near-peak {PEAK_LO:.0f}-{PEAK_HI:.0f} MeV")):
            mmask = sm & ((mc["m"] >= PEAK_LO) & (mc["m"] < PEAK_HI)
                          if peak_only else True)
            dmask = sd & ((da["m"] >= PEAK_LO) & (da["m"] < PEAK_HI)
                          if peak_only else True)
            stack = [np.clip(lchi_mc[mmask & (mc["cat"] == c)], 0, 7.999)
                     for c in range(6)]
            ws = [mc["w"][mmask & (mc["cat"] == c)] for c in range(6)]
            ax.hist(stack, bins=lbins, weights=ws, stacked=True,
                    color=CAT_COLORS,
                    label=[f"{CATS[c]} ({ws[c].sum():.0f})" for c in range(6)])
            dh, _ = np.histogram(np.clip(lchi_da[dmask], 0, 7.999), bins=lbins)
            ax.errorbar(ctr, dh, yerr=np.sqrt(np.clip(dh, 1, None)),
                        fmt="ko", ms=3.5, lw=1, capsize=0,
                        label=f"beam data ({int(dh.sum())})")
            ax.axvline(lcut, color="crimson", ls="--", lw=1.4,
                       label=f"cut chi2={args.chi2_cut:.0f}")
            ax.set(xlabel=r"$\log_{10}$ flash $\chi^2$ (primary nu vtx)",
                   ylabel=f"events / {args.pot:.1e} POT",
                   title=ttl)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)
        fig.suptitle(f"reco-NC flash-$\\chi^2$ ({vlab} photons): "
                     "beam data vs prediction", fontsize=11)
        fig.tight_layout()
        fig.savefig(f"{args.plots}/{fname}.png", dpi=110)
        plt.close(fig)

    # quantify: near-peak reco-NC data/MC before and after the chi2 cut
    print(f"== reco-NC near-peak ({PEAK_LO:.0f}-{PEAK_HI:.0f} MeV) "
          f"flash-chi2 cut @ {args.chi2_cut:.0f} ==")
    for eq2, vlab in ((False, ">=2"), (True, "exactly-2")):
        sm = sel(mc, eq2); sd = sel(da, eq2)
        pm = sm & (mc["m"] >= PEAK_LO) & (mc["m"] < PEAK_HI)
        pd = sd & (da["m"] >= PEAK_LO) & (da["m"] < PEAK_HI)
        lo_m = pm & (mc["chi2"] < args.chi2_cut)
        lo_d = pd & (da["chi2"] < args.chi2_cut)
        wm = mc["w"][pm].sum(); wm_lo = mc["w"][lo_m].sum()
        nd = int(pd.sum()); nd_lo = int(lo_d.sum())
        print(f"  [{vlab:9s}] all-chi2: data {nd:4d} / MC {wm:6.1f} = "
              f"{nd/max(wm,1e-9):.2f}  |  chi2<cut: data {nd_lo:4d} / "
              f"MC {wm_lo:6.1f} = {nd_lo/max(wm_lo,1e-9):.2f}  |  "
              f"data high-chi2 removed: {1-nd_lo/max(nd,1):.0%}, "
              f"MC removed: {1-wm_lo/max(wm,1e-9):.0%}")
    print(f">>> plots -> {args.plots}")


if __name__ == "__main__":
    main()
