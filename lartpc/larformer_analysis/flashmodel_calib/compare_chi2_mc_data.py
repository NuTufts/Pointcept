"""Overlay reco-NC 2-gamma nu-slice log(chi2): MC vs bnb5e19 data, showing the
effect of the per-run gamma correction on the data.

Run-aware dead-PMT mask (MC run3 -> opdet 15; data run1 -> none, ch15 live).
Data is shown at the current gamma_beam and at gamma_beam*--data-scale (the
muon-fit run1 correction, default 0.79). Area-normalized so the shapes are
comparable (MC is POT-weighted, data is raw counts).

    python3 compare_chi2_mc_data.py --mc-ntuple ... --mc-cascade ... \
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
from flash_correction import rse_map, neyman_masked   # noqa: E402
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


def chi2_list(ntuple, cascade_dir, eq2, scales):
    """{scale: np.array of nu-slice chi2} for reco-NC events, run-aware dead."""
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
    sel = vok & (n_g == 2 if eq2 else n_g >= 2) & ~reco_cc
    rmap = rse_map(cascade_dir, _cache(cascade_dir))
    out = {s: [] for s in scales}
    for i in np.nonzero(sel)[0]:
        p = rmap.get((int(run[i]), int(sub[i]), int(evt[i])))
        if p is None:
            continue
        pred, obs = nu_pe(p)
        if pred is None or obs.sum() <= 0 or pred.sum() <= 0:
            continue
        dead = dead_opdets_for_run(int(run[i]))
        for s in scales:
            out[s].append(neyman_masked(pred * s, obs, dead))
    return {s: np.array(v) for s, v in out.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mc-ntuple", required=True)
    ap.add_argument("--mc-cascade", required=True)
    ap.add_argument("--data-ntuple", required=True)
    ap.add_argument("--data-cascade", required=True)
    ap.add_argument("--data-scale", type=float, default=0.79)
    ap.add_argument("--eq2", action="store_true", default=True)
    ap.add_argument("--ge2", dest="eq2", action="store_false")
    ap.add_argument("--plots", required=True)
    args = ap.parse_args()
    os.makedirs(args.plots, exist_ok=True)
    vlab = "eq2" if args.eq2 else "ge2"

    mc = chi2_list(args.mc_ntuple, args.mc_cascade, args.eq2, (1.0,))[1.0]
    dd = chi2_list(args.data_ntuple, args.data_cascade, args.eq2,
                   (1.0, args.data_scale))
    d0, ds = dd[1.0], dd[args.data_scale]
    print(f">>> reco-NC {vlab}: MC N={len(mc)} med={np.median(mc):.0f} | "
          f"data N={len(d0)} med={np.median(d0):.0f} -> "
          f"x{args.data_scale:g} med={np.median(ds):.0f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lb = np.linspace(0, 7, 36)

    def lg(x):
        return np.clip(np.log10(np.clip(x, 1, None)), 0, 7)

    fig, ax = plt.subplots(figsize=(7.2, 4.7))
    ax.hist(lg(mc), bins=lb, density=True, histtype="step", lw=2.2,
            color="#1f77b4",
            label=f"MC (run3, ch15 masked)  med {np.median(mc):.0f}")
    ax.hist(lg(d0), bins=lb, density=True, histtype="step", lw=2.2,
            color="0.45", ls="--",
            label=f"bnb5e19 (gamma_beam)  med {np.median(d0):.0f}")
    ax.hist(lg(ds), bins=lb, density=True, histtype="step", lw=2.2,
            color="#d62728",
            label=f"bnb5e19 (gamma x {args.data_scale:g})  "
                  f"med {np.median(ds):.0f}")
    ax.set(xlabel=r"$\log_{10}$ flash $\chi^2$ (nu slice)",
           ylabel="normalized", title=f"reco-NC {vlab} pi0: nu-slice flash "
           f"chi2, MC vs bnb5e19 data\n(area-normalized; run-aware dead mask)")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{args.plots}/chi2_mc_vs_data_{vlab}.png", dpi=110)
    plt.close(fig)
    print(f">>> plots -> {args.plots}/chi2_mc_vs_data_{vlab}.png")


if __name__ == "__main__":
    main()
