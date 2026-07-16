"""Compare the pi0 two-photon selection between two ntuples that differ ONLY in
the flash chi2 mask (dead-only vs dead+saturation), reusing the selection, truth
categories and mass definition of pi0_mass_analysis.py.

The saturated-PMT mask changes which slice the flashmatch stream picks, so the
question is what that does to the physics: does the pi0 peak keep its shape while
the cosmic-fake background drops?

    PYTHONPATH=./ python3 .../pi0_compare_masks.py \
        --old <dead_only.root> --new <dead_plus_sat.root> --plots plots_pi0cmp
"""
import argparse
import os
import sys

import numpy as np
import uproot
import awkward as ak
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pi0_mass_analysis import (            # noqa: E402
    truth_category, CATS, CAT_COLORS, RECO_G_MIN, MU_KE_MIN, PI0_MASS)

GOOD_VTX = 5.0     # cm from the true nu vertex to call a reco vertex real
PEAK_LO, PEAK_HI = 100.0, 170.0

KEYS = ["run", "subrun", "event", "xsecWeight",
        "trueVtxInWCFV", "trueNuCCNC", "truePrimPartPDG",
        "trueSimPartPDG", "trueSimPartTID", "trueSimPartMID",
        "trueSimPartProcess", "trueSimPartPixelSumQ",
        "trueVtxX", "trueVtxY", "trueVtxZ",
        "foundVertex", "primaryVtxStream", "vtxIsFiducial",
        "vtxX", "vtxY", "vtxZ",
        "showerLArFormerPID", "showerRecoE",
        "showerStartPosX", "showerStartPosY", "showerStartPosZ",
        "trackLArFormerPID", "trackIsSecondary", "trackRecoE"]


def load(path, pot_target=None):
    f = uproot.open(path)
    a = f["EventTree"].arrays(KEYS)
    n = len(a["run"])
    psum = float(np.sum(f["potTree"].arrays(library="np")["totGoodPOT"]))
    w0 = np.asarray(a["xsecWeight"], np.float64)
    scale = (pot_target / psum) if pot_target else 1.0
    w = np.where(w0 > 0, w0, 0.0) * scale
    cat = np.array([truth_category(a, i) for i in range(n)], np.int64)

    # MUST match pi0_mass_analysis.py / flashchi2_ncpi0.py exactly: the canonical
    # pi0 selection is NU-STREAM ONLY (primaryVtxStream == 0). Dropping that term
    # pulls in fm-stream vertices the analysis never uses -- and since the
    # saturation mask only changes the fm stream, omitting it would attribute a
    # change to the mask that the real selection would never see.
    vtx_ok = ((np.asarray(a["foundVertex"]) == 1)
              & (np.asarray(a["primaryVtxStream"]) == 0)
              & (np.asarray(a["vtxIsFiducial"]) == 1))
    is_g = (a["showerLArFormerPID"] == 22) & (a["showerRecoE"] > RECO_G_MIN)
    n_g = ak.to_numpy(ak.sum(is_g, axis=1))
    is_mu = ((a["trackLArFormerPID"] == 13) & (a["trackIsSecondary"] == 0)
             & (a["trackRecoE"] > MU_KE_MIN))
    reco_cc = ak.to_numpy(ak.any(is_mu, axis=1))

    m = np.full(n, np.nan)
    for i in np.nonzero(vtx_ok & (n_g >= 2))[0]:
        gi = np.nonzero(ak.to_numpy(is_g[i]))[0]
        E = ak.to_numpy(a["showerRecoE"][i])[gi]
        o = gi[np.argsort(E)[::-1][:2]]
        E1, E2 = (ak.to_numpy(a["showerRecoE"][i])[o]).tolist()
        v = np.array([a["vtxX"][i], a["vtxY"][i], a["vtxZ"][i]], np.float64)
        sp = np.stack([ak.to_numpy(a[f"showerStartPos{c}"][i])[o]
                       for c in "XYZ"], 1).astype(np.float64)
        d = sp - v
        nn = np.linalg.norm(d, axis=1)
        if np.all(nn > 1e-3):
            d = d / nn[:, None]
            m[i] = float(np.sqrt(max(2.0 * E1 * E2 * (1.0 - d[0] @ d[1]), 0.0)))

    dv = np.sqrt((np.asarray(a["vtxX"]) - np.asarray(a["trueVtxX"])) ** 2
                 + (np.asarray(a["vtxY"]) - np.asarray(a["trueVtxY"])) ** 2
                 + (np.asarray(a["vtxZ"]) - np.asarray(a["trueVtxZ"])) ** 2)
    key = {(int(r), int(s), int(e)): i for i, (r, s, e)
           in enumerate(zip(np.asarray(a["run"]), np.asarray(a["subrun"]),
                            np.asarray(a["event"])))}
    return dict(a=a, n=n, w=w, cat=cat, m=m, n_g=n_g, reco_cc=reco_cc,
                vtx_ok=vtx_ok, dvtx=dv, key=key, pot=psum,
                stream=np.asarray(a["primaryVtxStream"]))


def report(d, tag, eq2):
    sel = d["vtx_ok"] & (d["n_g"] >= 2) & np.isfinite(d["m"])
    if eq2:
        sel &= (d["n_g"] == 2)
    out = {}
    print("\n===== %s =====" % tag)
    for cc, lab in ((True, "reco-CC"), (False, "reco-NC")):
        s = sel & (d["reco_cc"] == cc)
        sig = s & ((d["cat"] == 0) | (d["cat"] == 1))
        w = d["w"]
        nsig = float(w[sig].sum()); ntot = float(w[s].sum())
        pk = s & (d["m"] > PEAK_LO) & (d["m"] < PEAK_HI)
        pk_sig = pk & ((d["cat"] == 0) | (d["cat"] == 1))
        med = float(np.median(d["m"][sig])) if sig.any() else np.nan
        print("  %s: selected %6.1f (raw %5d) | true-pi0 signal %6.1f (%4.1f%% pure)"
              % (lab, ntot, int(s.sum()), nsig, 100 * nsig / max(ntot, 1e-9)))
        print("       signal mass median %6.1f MeV | near-peak(%.0f-%.0f) %6.1f"
              " tot, %6.1f sig (%4.1f%% pure)"
              % (med, PEAK_LO, PEAK_HI, float(w[pk].sum()),
                 float(w[pk_sig].sum()),
                 100 * float(w[pk_sig].sum()) / max(float(w[pk].sum()), 1e-9)))
        out[lab] = dict(sel=s, sig=sig, med=med, ntot=ntot, nsig=nsig,
                        pk=float(w[pk].sum()), pksig=float(w[pk_sig].sum()))
    return out, sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True, help="dead-mask-only ntuple")
    ap.add_argument("--new", required=True, help="dead+saturation ntuple")
    ap.add_argument("--old-label", default="dead-mask only")
    ap.add_argument("--new-label", default="dead + saturation")
    ap.add_argument("--plots", default="plots_pi0cmp")
    ap.add_argument("--eq2", action="store_true", help="exactly 2 photons")
    args = ap.parse_args()
    os.makedirs(args.plots, exist_ok=True)

    O = load(args.old)
    N = load(args.new)
    print(">>> OLD %d entries (POT %.4g) | NEW %d entries (POT %.4g)"
          % (O["n"], O["pot"], N["n"], N["pot"]))
    ro, selo = report(O, args.old_label, args.eq2)
    rn, seln = report(N, args.new_label, args.eq2)

    # ---- what moved, event by event (matched on RSE) -----------------------
    com = sorted(set(O["key"]) & set(N["key"]))
    io = np.array([O["key"][k] for k in com]); inn = np.array([N["key"][k] for k in com])
    print("\n===== selection churn (matched on run/subrun/event, n=%d) =====" % len(com))
    for cc, lab in ((True, "reco-CC"), (False, "reco-NC")):
        so = ro[lab]["sel"][io]; sn = rn[lab]["sel"][inn]
        real_o = O["dvtx"][io] < GOOD_VTX
        real_n = N["dvtx"][inn] < GOOD_VTX
        add = (~so) & sn
        drop = so & (~sn)
        print("  %s: entered %4d (%4d on a REAL nu vtx) | left %4d (%4d real)"
              % (lab, int(add.sum()), int((add & real_n).sum()),
                 int(drop.sum()), int((drop & real_o).sum())))

    # ---- mass plots ---------------------------------------------------------
    bins = np.linspace(0, 500, 51)
    for cc, lab in ((True, "reco-CC"), (False, "reco-NC")):
        fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
        for k, (D, R, name, c) in enumerate(
                ((O, ro, args.old_label, "0.4"), (N, rn, args.new_label, "tab:red"))):
            s = R[lab]["sel"]
            ax[0].hist(np.clip(D["m"][s], 0, 499), bins=bins, weights=D["w"][s],
                       histtype="step", lw=2, color=c,
                       label="%s (%.0f evt)" % (name, float(D["w"][s].sum())))
            sg = R[lab]["sig"]
            ax[1].hist(np.clip(D["m"][sg], 0, 499), bins=bins, weights=D["w"][sg],
                       histtype="step", lw=2, color=c,
                       label="%s (median %.0f MeV)" % (name, R[lab]["med"]))
        for x in ax:
            x.axvline(PI0_MASS, color="k", ls="--", lw=1, label="pi0 135 MeV")
            x.set_xlabel(r"$m_{\gamma\gamma}$ [MeV]")
            x.set_ylabel("events (xsec-weighted)")
            x.legend(fontsize=8)
        ax[0].set_title("%s: ALL selected (signal + background)" % lab)
        ax[1].set_title("%s: TRUE-pi0 signal only" % lab)
        fig.tight_layout()
        p = os.path.join(args.plots, "pi0mass_cmp_%s%s.png"
                         % (lab.replace("reco-", "").lower(),
                            "_eq2" if args.eq2 else ""))
        fig.savefig(p, dpi=110)
        print(">>> wrote", p)


if __name__ == "__main__":
    main()
