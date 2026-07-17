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
# canonical active-TPC volume [cm] (same def as eval --true-vtx-in-tpc)
TPC_LO = np.array([0.0, -116.5, 0.0])
TPC_HI = np.array([256.35, 116.5, 1036.8])


def load(ntuple, is_data, pot, cascade_dir=None, dead=(15,),
         saturation=False):
    """Return per-event arrays for the reco-NC pi0 study. If cascade_dir is
    given, the primary-vertex flash chi2 is REPLACED (for the pre-selected
    events) by the dead-PMT-masked nu-slice chi2 recomputed from the cascade
    per-PMT arrays -- the analysis-level flash-fix preview."""
    fin = uproot.open(ntuple)
    if is_data:
        scale = 1.0
    else:
        p = fin["potTree"].arrays(library="np")
        psum = float(np.sum(p["totGoodPOT"])) or float(np.sum(p["totPOT"]))
        scale = pot / psum
    t = fin["EventTree"]
    a = t.arrays(["run", "subrun", "event",
                  "xsecWeight", "trueVtxInWCFV", "trueNuCCNC",
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

    # analysis-level flash-fix: replace primary-vertex chi2 with the
    # dead-PMT-masked nu-slice chi2 recomputed from the cascade per-PMT arrays
    # (matched by run/subrun/event; cached RSE->path map per cascade dir)
    if cascade_dir is not None:
        import os as _os
        from flash_correction import corrected_chi2_by_rse
        run = np.asarray(a["run"]); sub = np.asarray(a["subrun"])
        evt = np.asarray(a["event"])
        ent = np.nonzero(vok & (n_g >= 2))[0]
        cache = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                              "rse_" + _os.path.basename(
                                  _os.path.dirname(cascade_dir.rstrip("/")))
                              + "_" + _os.path.basename(cascade_dir.rstrip("/"))
                              + ".npz")
        cc = corrected_chi2_by_rse(cascade_dir, run, sub, evt, ent, dead, cache,
                                   saturation=saturation)
        for i in ent:
            chi2[i] = cc.get(int(i), np.nan)

    # primary-vertex position + distance to nearest active-TPC wall (dwall)
    vx = np.asarray(a["vtxX"], np.float64)
    vy = np.asarray(a["vtxY"], np.float64)
    vz = np.asarray(a["vtxZ"], np.float64)
    dwall = np.minimum.reduce([vx - TPC_LO[0], TPC_HI[0] - vx,
                               vy - TPC_LO[1], TPC_HI[1] - vy,
                               vz - TPC_LO[2], TPC_HI[2] - vz])
    return dict(cat=cat, w=w, n_g=n_g, reco_cc=reco_cc, m=mA,
                chi2=chi2, sel2p=sel2p, sel2x=sel2x, scale=scale,
                vx=vx, vy=vy, vz=vz, dwall=dwall)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mc-ntuple", required=True)
    ap.add_argument("--data-ntuple", required=True)
    ap.add_argument("--ext-ntuple", default=None,
                    help="EXT-BNB cosmic ntuple; stacked as a cosmic component")
    ap.add_argument("--ext-scale", type=float, default=0.17682554549,
                    help="per-EXT-event weight (0.17682554549 / fraction); "
                         "default = full-sample spill ratio")
    ap.add_argument("--mc-cascade", default=None,
                    help="MC keypoint2_streams dir; enables dead-PMT-masked "
                         "flash-chi2 recompute (analysis-level flash fix)")
    ap.add_argument("--data-cascade", default=None)
    ap.add_argument("--ext-cascade", default=None)
    ap.add_argument("--saturation-mask", action="store_true",
                    help="also mask SATURATED PMTs in the chi2 recompute (see "
                         "lartpc/flashmatch/saturation.py). Applied to ALL "
                         "samples so MC/data/EXT get identical treatment -- the "
                         "MC cascade may already bake it in while the data/EXT "
                         "cascades predate it, and mixing the two would be an "
                         "unfair comparison.")
    ap.add_argument("--dead-channels", default="15",
                    help="comma-sep opdet indices to mask (default 15, the "
                         "run3 dead PMT; masked in ALL samples for fairness)")
    ap.add_argument("--plots", required=True)
    ap.add_argument("--pot", type=float, default=4.4e19)
    ap.add_argument("--chi2-cut-nc", type=float, default=None,
                    help="separate (tighter) cut for the reco-NC cut LINE + tail "
                         "ratio; defaults to --chi2-cut. Matches the per-stream "
                         "cut used in datamc_ext_overlay (NC log10<3.5=3162).")
    ap.add_argument("--chi2-cut", type=float, default=10000.0,
                    help="high-chi2 (out-of-time) cut line + ratio table; "
                         "10^4 accommodates the run1 beam-data light-yield "
                         "shift (see per-run gamma note)")
    args = ap.parse_args()
    os.makedirs(args.plots, exist_ok=True)
    dead = tuple(int(x) for x in args.dead_channels.split(",") if x != "")

    S = args.saturation_mask
    mc = load(args.mc_ntuple, False, args.pot, args.mc_cascade, dead, S)
    da = load(args.data_ntuple, True, args.pot, args.data_cascade, dead, S)
    ex = (load(args.ext_ntuple, True, args.pot, args.ext_cascade, dead, S)
          if args.ext_ntuple else None)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lbins = np.linspace(0, 8, 33)     # log10(chi2)
    lcut = np.log10(args.chi2_cut)

    def sel(d, eq2, want_cc):
        s = (d["sel2x"] if eq2 else d["sel2p"]) \
            & (d["reco_cc"] == want_cc) & np.isfinite(d["chi2"])
        return s

    lchi_mc = np.log10(np.clip(mc["chi2"], 1, None))
    lchi_da = np.log10(np.clip(da["chi2"], 1, None))
    lchi_ex = np.log10(np.clip(ex["chi2"], 1, None)) if ex else None
    ctr = 0.5 * (lbins[:-1] + lbins[1:])

    # reco-CC = likely genuine in-time neutrinos (control); reco-NC = probe
    for want_cc, slab, sabb in ((True, "reco-CC (likely nu, in-time)", "cc"),
                                (False, "reco-NC", "nc")):
        cutv = (args.chi2_cut if want_cc or args.chi2_cut_nc is None
                else args.chi2_cut_nc)
        lcut_s = np.log10(cutv)
        for eq2, vlab, vabb in ((True, "exactly 2", "eq2"),
                                (False, ">=2", "ge2")):
            sm = sel(mc, eq2, want_cc); sd = sel(da, eq2, want_cc)
            fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
            for ax, peak_only, ttl in (
                    (axes[0], False, "all m$_{\\gamma\\gamma}$"),
                    (axes[1], True,
                     f"near-peak {PEAK_LO:.0f}-{PEAK_HI:.0f} MeV")):
                mmask = sm & ((mc["m"] >= PEAK_LO) & (mc["m"] < PEAK_HI)
                              if peak_only else True)
                dmask = sd & ((da["m"] >= PEAK_LO) & (da["m"] < PEAK_HI)
                              if peak_only else True)
                stack = [np.clip(lchi_mc[mmask & (mc["cat"] == c)], 0, 7.999)
                         for c in range(6)]
                ws = [mc["w"][mmask & (mc["cat"] == c)] for c in range(6)]
                colors = list(CAT_COLORS)
                labels = [f"{CATS[c]} ({ws[c].sum():.0f})" for c in range(6)]
                if ex is not None:
                    emask = sel(ex, eq2, want_cc) & (
                        (ex["m"] >= PEAK_LO) & (ex["m"] < PEAK_HI)
                        if peak_only else True)
                    stack.append(np.clip(lchi_ex[emask], 0, 7.999))
                    ws.append(np.full(int(emask.sum()), args.ext_scale))
                    colors.append("#e5e5e5")
                    labels.append(f"EXT cosmic ({ws[-1].sum():.0f})")
                ax.hist(stack, bins=lbins, weights=ws, stacked=True,
                        color=colors, label=labels)
                dh, _ = np.histogram(np.clip(lchi_da[dmask], 0, 7.999),
                                     bins=lbins)
                ax.errorbar(ctr, dh, yerr=np.sqrt(np.clip(dh, 1, None)),
                            fmt="ko", ms=3.5, lw=1, capsize=0,
                            label=f"beam data ({int(dh.sum())})")
                ax.axvline(lcut_s, color="crimson", ls="--", lw=1.4,
                           label=f"cut chi2={cutv:.0f}")
                ax.set(xlabel=r"$\log_{10}$ flash $\chi^2$ (primary nu vtx)",
                       ylabel=f"events / {args.pot:.1e} POT", title=ttl)
                ax.legend(fontsize=7)
                ax.grid(alpha=0.3)
            fig.suptitle(f"{slab} flash-$\\chi^2$ ({vlab} photons): "
                         "beam data vs prediction", fontsize=11)
            fig.tight_layout()
            fig.savefig(f"{args.plots}/flashchi2_{sabb}_{vabb}.png", dpi=110)
            plt.close(fig)

    # quantify: median chi2 per stream (data vs MC vs EXT cosmic) + how the
    # high-chi2 tail (chi2 > cut) is populated -- the cosmic sits there if the
    # EXT nu-slice chi2 is high, so a cut would separate it.
    print("== median flash-chi2 (data vs MC vs EXT) + high-chi2 tail share ==")
    for want_cc, slab in ((True, "reco-CC"), (False, "reco-NC")):
        cutv = (args.chi2_cut if want_cc or args.chi2_cut_nc is None
                else args.chi2_cut_nc)
        for eq2, vlab in ((False, ">=2"), (True, "exactly-2")):
            sm = sel(mc, eq2, want_cc); sd = sel(da, eq2, want_cc)
            med_d = np.median(da["chi2"][sd]) if sd.any() else np.nan
            med_m = np.median(mc["chi2"][sm]) if sm.any() else np.nan
            se = sel(ex, eq2, want_cc) if ex is not None else None
            med_e = (np.median(ex["chi2"][se]) if se is not None and se.any()
                     else np.nan)
            # fraction of each above the cut (the putative cosmic/out-of-time band)
            hi = lambda d, s: (float(np.mean(d["chi2"][s] > cutv))
                               if s.any() else np.nan)
            print(f"  [{slab} {vlab:9s}] median chi2: data {med_d:8.0f} | "
                  f"MC {med_m:8.0f} | EXT {med_e:9.0f}  ||  frac chi2>"
                  f"{cutv:.0f}: data {hi(da,sd):.2f} MC {hi(mc,sm):.2f}"
                  f" EXT {hi(ex,se) if ex is not None else float('nan'):.2f}")
    print(f">>> plots -> {args.plots}  (EXT scale "
          f"{args.ext_scale if ex is not None else float('nan'):.4f})")


if __name__ == "__main__":
    main()
