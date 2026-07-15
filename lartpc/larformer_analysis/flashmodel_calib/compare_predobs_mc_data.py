"""Overlay the total pred/obs PE ratio for the reco-NC 2-gamma (pi0) nu slices:
MC vs bnb5e19 data (data with the muon-fit run1 gamma correction, pred*scale).

Total over LIVE PMTs, run-aware dead mask (MC run3 -> opdet 15; data run1 ->
none). MC scale 1.0; data scale = --data-scale (default 0.79).

    python3 compare_predobs_mc_data.py --mc-ntuple ... --mc-cascade ... \
        --data-ntuple ... --data-cascade ... --data-scale 0.79 --plots plots/
"""
import argparse
import os
import sys

import numpy as np
import uproot
import awkward as ak
import h5py

_PI0 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "..", "physics", "pi0mass_peak")
sys.path.insert(0, _PI0)
from flash_correction import rse_map               # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", ".."))
from lartpc.flashmatch.dead_channels import dead_opdets_for_run  # noqa: E402


def _cache(cascade_dir):
    return os.path.join(_PI0, "rse_" + os.path.basename(
        os.path.dirname(cascade_dir.rstrip("/"))) + "_"
        + os.path.basename(cascade_dir.rstrip("/")) + ".npz")


def nu_pe(path):
    try:
        with h5py.File(path, "r") as f:
            if "slices" not in f or "flash" not in f \
                    or "observed_pe" not in f["flash"]:
                return None, None
            labs = [l.decode() if isinstance(l, bytes) else str(l)
                    for l in f["slices/label"][()]]
            if "nu" not in labs:
                return None, None
            j = labs.index("nu")
            return (np.nan_to_num(np.asarray(f["slices/pred_pe"][()][j], float)),
                    np.nan_to_num(np.asarray(f["flash/observed_pe"][()], float)))
    except Exception:
        return None, None


def ratios(ntuple, cascade_dir, eq2, scale, want_cc=False, min_pe=50.0):
    t = uproot.open(ntuple)["EventTree"]
    a = t.arrays(["run", "subrun", "event", "foundVertex", "primaryVtxStream",
                  "vtxIsFiducial", "showerLArFormerPID", "showerRecoE",
                  "trackLArFormerPID", "trackIsSecondary", "trackRecoE"])
    run = np.asarray(a["run"]); sub = np.asarray(a["subrun"])
    evt = np.asarray(a["event"])
    vok = ((np.asarray(a["foundVertex"]) == 1)
           & (np.asarray(a["primaryVtxStream"]) == 0)
           & (np.asarray(a["vtxIsFiducial"]) == 1))
    n_g = ak.to_numpy(ak.sum((a["showerLArFormerPID"] == 22)
                             & (a["showerRecoE"] > 20), axis=1))
    reco_cc = ak.to_numpy(ak.any((a["trackLArFormerPID"] == 13)
                                 & (a["trackIsSecondary"] == 0)
                                 & (a["trackRecoE"] > 100), axis=1))
    sel = vok & (n_g == 2 if eq2 else n_g >= 2) & (reco_cc == want_cc)
    rmap = rse_map(cascade_dir, _cache(cascade_dir))
    r = []
    for i in np.nonzero(sel)[0]:
        p = rmap.get((int(run[i]), int(sub[i]), int(evt[i])))
        if p is None:
            continue
        pred, obs = nu_pe(p)
        if pred is None:
            continue
        live = np.ones(32, bool)
        for d in dead_opdets_for_run(int(run[i])):
            live[d] = False
        pp = (pred * scale)[live].sum(); oo = obs[live].sum()
        if pp < min_pe or oo <= 0:
            continue
        r.append(pp / oo)
    return np.array(r)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mc-ntuple", required=True)
    ap.add_argument("--mc-cascade", required=True)
    ap.add_argument("--data-ntuple", required=True)
    ap.add_argument("--data-cascade", required=True)
    ap.add_argument("--data-scale", type=float, default=0.79)
    ap.add_argument("--eq2", action="store_true", default=True)
    ap.add_argument("--ge2", dest="eq2", action="store_false")
    ap.add_argument("--reco-cc", action="store_true",
                    help="reco-CC stream (default reco-NC)")
    ap.add_argument("--plots", required=True)
    args = ap.parse_args()
    os.makedirs(args.plots, exist_ok=True)
    vlab = "eq2" if args.eq2 else "ge2"
    slab = "recoCC" if args.reco_cc else "recoNC"

    mc = ratios(args.mc_ntuple, args.mc_cascade, args.eq2, 1.0, args.reco_cc)
    d0 = ratios(args.data_ntuple, args.data_cascade, args.eq2, 1.0, args.reco_cc)
    ds = ratios(args.data_ntuple, args.data_cascade, args.eq2, args.data_scale,
                args.reco_cc)
    print(f">>> {slab} {vlab} pred/obs: MC med {np.median(mc):.2f} (N={len(mc)})"
          f" | data med {np.median(d0):.2f} -> x{args.data_scale:g} "
          f"{np.median(ds):.2f} (N={len(ds)})")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    b = np.linspace(0, 3, 46)
    fig, ax = plt.subplots(figsize=(7.2, 4.7))
    for r, col, ls, lab in (
            (mc, "#1f77b4", "-", f"MC (run3, ch15 masked)  med {np.median(mc):.2f}"),
            (d0, "0.45", "--", f"bnb5e19 (gamma_beam)  med {np.median(d0):.2f}"),
            (ds, "#d62728", "-", f"bnb5e19 (gamma x {args.data_scale:g})  "
                                 f"med {np.median(ds):.2f}")):
        ax.hist(np.clip(r, 0, 3), bins=b, density=True, histtype="step",
                lw=2.2, color=col, ls=ls, label=lab)
    ax.axvline(1.0, color="0.3", ls=":", lw=1.2)
    ax.set(xlabel="predicted / observed PE (live PMTs)", ylabel="normalized",
           title=f"{slab} {vlab} pi0: nu-slice pred/obs PE, MC vs bnb5e19\n"
                 "(run-aware dead mask; data with run1 gamma correction)")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{args.plots}/predobs_mc_vs_data_{slab}_{vlab}.png", dpi=110)
    plt.close(fig)
    print(f">>> plots -> {args.plots}/predobs_mc_vs_data_{vlab}.png")


if __name__ == "__main__":
    main()
