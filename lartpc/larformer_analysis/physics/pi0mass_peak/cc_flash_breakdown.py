"""CC flash-chi2 mismodeling breakdown: direction and containment splits.

Tests three hypotheses for the reco-CC flash-chi2 data/MC disagreement
(user, 2026-08-30): (1) broken muon reco (missing track end), (2) muon
exits the TPC (light modeling outside the TPC is poor), (3) unmodeled
Cherenkov light (directional: a muon heading toward the PMT plane, dirX<0,
would beam extra light at the PMTs).

Sample: reco-CC events (primary mu, KE>--mu-ke-min), >=2 recal'd photons,
FV vertex, official flash-blind BDT applied (score >= --bdt-thr), chi2
UNCUT (the chi2 distribution is the observable). BDT-training hygiene:
even-event signal-MC/EXT excluded, held-out halves w x2.

Muon = highest-KE primary LArFormerPID==13 track. Splits:
  dirX  : trackStartDirX >= 0 (away from PMTs) vs < 0 (toward PMTs)
  cont  : SCE-corrected trackEndPos dwall > --cont-cm  (contained)
          vs dwall < --exit-cm (exiting); intermediate dropped
Per split: log10 chi2 overlay (MC+EXT stacked, data points) and the
data/pred ratio in the nu-peak (log10<3) and tail (3-6) regions.

    PYTHONPATH=./ python3 cc_flash_breakdown.py --mc-ntuple ... (see ext_bdt)
"""
import argparse
import os
import sys

import numpy as np
import awkward as ak
import uproot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))
from datamc_diagnostics import load, add_flash_pe  # noqa: E402
from lartpc.flashmatch.sce_microboone import SCEBackward  # noqa: E402

TPC_LO = np.array([0.0, -116.5, 0.0])
TPC_HI = np.array([256.35, 116.5, 1036.8])


def mu_vars(ntuple, rows, mu_ke):
    """Per selected-event muon (dirX, SCE-corrected end dwall)."""
    t = uproot.open(ntuple)["EventTree"]
    a = t.arrays(["trackLArFormerPID", "trackIsSecondary", "trackRecoE",
                  "trackStartDirX", "trackEndPosX", "trackEndPosY",
                  "trackEndPosZ"])
    sce = SCEBackward()
    dirx = np.full(len(rows), np.nan)
    dwall = np.full(len(rows), np.nan)
    for k, i in enumerate(rows):
        pid = np.asarray(a["trackLArFormerPID"][i])
        sec = np.asarray(a["trackIsSecondary"][i])
        ke = np.asarray(a["trackRecoE"][i])
        m = (pid == 13) & (sec == 0) & (ke > mu_ke)
        if not m.any():
            continue
        j = int(np.nonzero(m)[0][np.argmax(ke[m])])
        dirx[k] = float(a["trackStartDirX"][i][j])
        end = np.array([a["trackEndPosX"][i][j], a["trackEndPosY"][i][j],
                        a["trackEndPosZ"][i][j]], float)
        endc = sce.correct(end)
        dwall[k] = float(min((endc - TPC_LO).min(), (TPC_HI - endc).min()))
    return dirx, dwall


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for s in ("mc", "data", "ext"):
        ap.add_argument(f"--{s}-ntuple", required=True)
        ap.add_argument(f"--{s}-table", required=True)
    ap.add_argument("--ext-scale", type=float, required=True)
    ap.add_argument("--recal-gamma-a", type=float, default=0.01556)
    ap.add_argument("--recal-gamma-b", type=float, default=-11.47)
    ap.add_argument("--mu-ke-min", type=float, default=50.0)
    ap.add_argument("--bdt-model", required=True)
    ap.add_argument("--bdt-thr", type=float, default=0.280)
    ap.add_argument("--cont-cm", type=float, default=15.0)
    ap.add_argument("--exit-cm", type=float, default=5.0)
    ap.add_argument("--plots", required=True)
    args = ap.parse_args()
    os.makedirs(args.plots, exist_ok=True)
    import joblib
    M = joblib.load(args.bdt_model)
    clf, FEATS = M["clf"], M["feats"]
    here = os.path.dirname(os.path.abspath(__file__))

    def cache_for(c):
        return os.path.join(here, "rse_" + os.path.basename(os.path.dirname(
            c.rstrip("/"))) + "_" + os.path.basename(c.rstrip("/")) + ".npz")

    CASC = {"mc": os.path.dirname(args.mc_ntuple) + "/keypoint2_streams",
            "data": os.path.dirname(args.data_ntuple) + "/keypoint2_streams",
            "ext": os.path.dirname(args.ext_ntuple) + "/keypoint2_streams"}
    smp = {}
    for s in ("mc", "data", "ext"):
        print(f">>> loading {s} ...", flush=True)
        d = load(getattr(args, f"{s}_ntuple"), getattr(args, f"{s}_table"),
                 args.recal_gamma_a, args.recal_gamma_b,
                 args.mu_ke_min, 1e12, 1e12)          # chi2 UNCUT
        d = add_flash_pe(d, CASC[s], cache_for(CASC[s]))
        X = np.column_stack([d[f].astype(float) for f in FEATS])
        sc = clf.predict_proba(X)[:, 1]
        keep = d["recoCC"] & (sc >= args.bdt_thr)
        # hygiene: drop BDT-training halves, x2 held-out
        w = d["w"].copy()
        if s == "ext":
            keep &= (d["event"] % 2 == 1)
            w *= 2.0 * args.ext_scale / np.where(w != 0, w, 1)  # uniform
            w = np.full(len(w), 2.0 * args.ext_scale)
        if s == "mc":
            cat = np.load(getattr(args, f"{s}_table"))["cat"][d["row"]]
            train = (cat < 2) & (d["event"] % 2 == 0)
            keep &= ~train
            w[(cat < 2) & (d["event"] % 2 == 1)] *= 2.0
        rows = d["row"][keep]
        dirx, dwall = mu_vars(getattr(args, f"{s}_ntuple"), rows,
                              args.mu_ke_min)
        smp[s] = dict(logchi2=d["logchi2"][keep], w=w[keep],
                      dirx=dirx, dwall=dwall)
        print(f"    {s}: {int(keep.sum())} reco-CC post-BDT "
              f"(mu found {int(np.isfinite(dirx).sum())})")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bins = np.linspace(0, 8, 41)
    ctr = 0.5 * (bins[:-1] + bins[1:])

    def split_mask(d, name):
        if name == "dirX>0 (away from PMTs)":
            return d["dirx"] >= 0
        if name == "dirX<0 (toward PMTs)":
            return d["dirx"] < 0
        if name == "mu contained (SCE dwall>15)":
            return d["dwall"] > args.cont_cm
        if name == "mu exiting (SCE dwall<5)":
            return d["dwall"] < args.exit_cm
        raise KeyError(name)

    names = ["dirX>0 (away from PMTs)", "dirX<0 (toward PMTs)",
             "mu contained (SCE dwall>15)", "mu exiting (SCE dwall<5)"]
    fig, axs = plt.subplots(2, 2, figsize=(12, 8.4))
    print(f"\n{'split':>28} {'peak d/p':>9} {'tail d/p':>9} "
          f"{'data N':>7} {'pred N':>7}")
    for ax, nm in zip(axs.ravel(), names):
        hm, _ = np.histogram(smp["mc"]["logchi2"][split_mask(smp["mc"], nm)],
                             bins, weights=smp["mc"]["w"][split_mask(smp["mc"], nm)])
        he, _ = np.histogram(smp["ext"]["logchi2"][split_mask(smp["ext"], nm)],
                             bins, weights=smp["ext"]["w"][split_mask(smp["ext"], nm)])
        md = split_mask(smp["data"], nm)
        hd, _ = np.histogram(smp["data"]["logchi2"][md], bins)
        ax.bar(ctr, hm, width=np.diff(bins), color="#7bafd4", label="MC")
        ax.bar(ctr, he, width=np.diff(bins), bottom=hm, color="#d9d9d9",
               label="EXT")
        ax.errorbar(ctr, hd, yerr=np.sqrt(np.maximum(hd, 1)), fmt="ko",
                    ms=3, label="data")
        ax.axvline(4.0, color="r", ls="--", lw=1)
        ax.set(xlabel="log10 flash chi2", title=nm)
        ax.legend(fontsize=7)
        pk = (ctr < 3); tl = (ctr >= 3) & (ctr < 6)
        pd_, pp = hd[pk].sum(), (hm + he)[pk].sum()
        td_, tp = hd[tl].sum(), (hm + he)[tl].sum()
        print(f"{nm:>28} {pd_/max(pp,1e-9):9.2f} {td_/max(tp,1e-9):9.2f} "
              f"{int(hd.sum()):7d} {(hm+he).sum():7.0f}")
    fig.tight_layout()
    fig.savefig(os.path.join(args.plots, "cc_chi2_splits.png"), dpi=120)
    print(f">>> {args.plots}/cc_chi2_splits.png")


if __name__ == "__main__":
    main()
