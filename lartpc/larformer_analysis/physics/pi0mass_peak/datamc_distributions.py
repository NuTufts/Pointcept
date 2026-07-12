"""Data-vs-simulation reco distribution checkup (ntuple-only).

Overlays AREA-NORMALIZED shapes of the key reconstruction quantities from a
beam-data ntuple against the nu-overlay simulation ntuple (xsecWeight-
weighted). Absolute normalization is NOT attempted -- beam POT accounting is
outside this chain, and beam-on data contains cosmic-only triggers the
nu-overlay sim does not model, so shapes are compared per population:

  event level (vertexed, nu stream, in-FV): vtx position/score, nTracks,
      nShowers, n reco photons, recoVtxFlashChi2 (nu-stream rows);
  shower level: RecoE, AttScore, LArFormerPhScore;
  track level:  RecoE, LArFormer PID class fractions.

Meant for eyeballing scale-up pathologies (spikes, empty branches, shifted
scales), not physics agreement -- cosmic contamination in data is expected
to distort most shapes.

    PYTHONPATH=./ python3 .../datamc_distributions.py \
        --data .../dlgen2_larformer_ntuple_bnb5e19_1k.root \
        --sim  .../dlgen2_larformer_ntuple_mcc9_bnbnu_overlay_1500_full_67k.root \
        --plots plots_datamc/
"""
import argparse
import os

import numpy as np
import uproot
import awkward as ak

BR_EV = ["foundVertex", "primaryVtxStream", "vtxIsFiducial", "vtxX", "vtxY",
         "vtxZ", "vtxScore", "nTracks", "nShowers", "xsecWeight",
         "nRecoVtx", "recoVtxStream", "recoVtxFlashChi2"]
BR_SH = ["showerRecoE", "showerAttScore", "showerLArFormerPhScore",
         "showerLArFormerPID"]
BR_TK = ["trackRecoE", "trackLArFormerPID", "trackIsSecondary"]


def load(path, is_data):
    t = uproot.open(path)["EventTree"]
    a = t.arrays(BR_EV + BR_SH + BR_TK)
    sel = ((np.asarray(a["foundVertex"]) == 1)
           & (np.asarray(a["primaryVtxStream"]) == 0)
           & (np.asarray(a["vtxIsFiducial"]) == 1))
    if is_data:
        w = np.ones(len(sel))
    else:
        w0 = np.asarray(a["xsecWeight"], np.float64)
        w = np.where(w0 > 0, w0, 0.0)
    return a, sel, w


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--sim", required=True)
    ap.add_argument("--plots", required=True)
    args = ap.parse_args()
    os.makedirs(args.plots, exist_ok=True)

    da, dsel, dw = load(args.data, True)
    sa, ssel, sw = load(args.sim, False)
    print(f">>> data: {len(dsel)} events, {int(dsel.sum())} vertexed in-FV "
          f"({dsel.mean():.3f})")
    print(f">>> sim  : {len(ssel)} events, {int(ssel.sum())} vertexed in-FV "
          f"({ssel.mean():.3f})")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def overlay(vals_d, w_d, vals_s, w_s, bins, xlabel, fname, logy=False):
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        vs = np.clip(vals_s, bins[0], bins[-1] - 1e-9)
        ax.hist(vs, bins=bins, weights=w_s, density=True,
                histtype="stepfilled", alpha=0.4, color="C0",
                label=f"sim nu-overlay (N={len(vals_s)})")
        vd = np.clip(vals_d, bins[0], bins[-1] - 1e-9)
        hd, edges = np.histogram(vd, bins=bins, weights=w_d)
        err = np.sqrt(np.histogram(vd, bins=bins, weights=w_d**2)[0])
        norm = max(hd.sum() * np.diff(edges)[0], 1e-9)
        ctr = 0.5 * (edges[:-1] + edges[1:])
        ax.errorbar(ctr, hd / norm, yerr=err / norm, fmt="ko", ms=3.5,
                    label=f"beam data (N={len(vals_d)})")
        if logy:
            ax.set_yscale("log")
        ax.set(xlabel=xlabel, ylabel="area-normalized",
               title=f"{fname} (shape comparison)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{args.plots}/{fname}.png", dpi=110)
        plt.close(fig)

    # ---- event level (vertexed in-FV) --------------------------------------
    for br, bins, lab in (("vtxX", np.linspace(0, 260, 40), "vtx X [cm]"),
                          ("vtxY", np.linspace(-120, 120, 40), "vtx Y [cm]"),
                          ("vtxZ", np.linspace(0, 1040, 52), "vtx Z [cm]"),
                          ("vtxScore", np.linspace(0, 1, 41), "vertex score"),
                          ("nTracks", np.arange(-0.5, 10.5), "n tracks"),
                          ("nShowers", np.arange(-0.5, 10.5), "n showers")):
        overlay(np.asarray(da[br])[dsel], dw[dsel],
                np.asarray(sa[br])[ssel], sw[ssel], bins, lab, f"ev_{br}")

    # nu-stream flash chi2 (log10)
    def fchi2(a, sel):
        rows = a["recoVtxFlashChi2"][sel][a["recoVtxStream"][sel] == 0]
        v = np.asarray(ak.flatten(rows), np.float64)
        return np.log10(v[np.isfinite(v) & (v > 0)])
    overlay(fchi2(da, dsel), np.ones_like(fchi2(da, dsel)),
            fchi2(sa, ssel), np.ones_like(fchi2(sa, ssel)),
            np.linspace(0, 5, 41), "log10 nu-slice flash chi2", "ev_flashchi2")

    # ---- prong level (from vertexed in-FV events) ---------------------------
    def flat(a, sel, br, wev):
        v = a[br][sel]
        n = ak.num(v)
        return (np.asarray(ak.flatten(v), np.float64),
                np.repeat(wev, ak.to_numpy(n)))

    for br, bins, lab, logy in (
            ("showerRecoE", np.linspace(0, 600, 40), "shower RecoE [MeV]", True),
            ("showerAttScore", np.linspace(-15, 25, 41), "shower LLR att score", False),
            ("showerLArFormerPhScore", np.linspace(0, 1, 41), "gamma class score", False),
            ("trackRecoE", np.linspace(0, 1500, 50), "track RecoE [MeV]", True)):
        vd, wd = flat(da, dsel, br, dw[dsel])
        vs, ws = flat(sa, ssel, br, sw[ssel])
        overlay(vd, wd, vs, ws, bins, lab, f"prong_{br}", logy=logy)

    # LArFormer PID class fractions (tracks + showers)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    pids = [11, 22, 13, 211, 2212, 0]
    names = ["e", "gamma", "mu", "pi", "p", "other"]
    for a, sel, wev, lab, off in ((da, dsel, dw[dsel], "data", -0.15),
                                  (sa, ssel, sw[ssel], "sim", 0.15)):
        allpid = np.concatenate([
            np.asarray(ak.flatten(a["trackLArFormerPID"][sel])),
            np.asarray(ak.flatten(a["showerLArFormerPID"][sel]))])
        fr = [np.mean(allpid == p) for p in pids]
        ax.bar(np.arange(6) + off, fr, width=0.3, label=lab)
    ax.set(xticks=range(6), xticklabels=names,
           ylabel="fraction of prongs",
           title="LArFormer PID composition (vertexed in-FV events)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{args.plots}/prong_pid_composition.png", dpi=110)
    plt.close(fig)
    print(f">>> plots -> {args.plots}")


if __name__ == "__main__":
    main()
