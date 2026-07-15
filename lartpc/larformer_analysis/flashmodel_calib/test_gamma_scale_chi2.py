"""Cross-check the muon-derived per-run gamma on the pi0 (shower) slices.

The gamma(run1)=0.79*gamma_beam was fit on MIP muons. If that offset is a
genuine GLOBAL run1 charge/light scale, scaling the predicted PE down by the
same factor should also reduce the flash chi2 of the reco-NC two-photon
(pi0-candidate) slices -- which contain NO muon. If the chi2 does NOT improve,
the muon result is likely a muon-reco artifact (e.g. broken tracks) rather than
a global scale.

Recomputes the nu-slice Neyman chi2 from the stored cascade pred_pe/observed_pe
(no re-run), at scale 1.0 and at --scale, run-aware dead mask (run1: none).

    python3 test_gamma_scale_chi2.py --ntuple <bnb5e19.root> \
        --cascade-dir <keypoint2_streams> --scale 0.79 --plots plots/
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


def nu_slice_pe(path):
    """(pred_pe32, obs_pe32) for the cascade nu slice, or (None, None)."""
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


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ntuple", required=True)
    ap.add_argument("--cascade-dir", required=True)
    ap.add_argument("--scale", type=float, default=0.79,
                    help="multiply predicted PE by this (gamma(run)/gamma_beam)")
    ap.add_argument("--eq2", action="store_true", default=True,
                    help="reco-NC exactly-2 (default); --ge2 for >=2")
    ap.add_argument("--ge2", dest="eq2", action="store_false")
    ap.add_argument("--sample-tag", default="bnb5e19_run1")
    ap.add_argument("--plots", required=True)
    args = ap.parse_args()
    os.makedirs(args.plots, exist_ok=True)

    t = uproot.open(args.ntuple)["EventTree"]
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
    sel = vok & (n_g == 2 if args.eq2 else n_g >= 2) & ~reco_cc
    idx = np.nonzero(sel)[0]
    print(f">>> {args.sample_tag}: {len(idx)} reco-NC "
          f"{'eq2' if args.eq2 else 'ge2'} pi0 candidates")

    cache = os.path.join(_PI0, "rse_" + os.path.basename(
        os.path.dirname(args.cascade_dir.rstrip("/"))) + "_"
        + os.path.basename(args.cascade_dir.rstrip("/")) + ".npz")
    rmap = rse_map(args.cascade_dir, cache)      # reuse the pi0 study's cache

    c0, cS = [], []
    for i in idx:
        p = rmap.get((int(run[i]), int(sub[i]), int(evt[i])))
        if p is None:
            continue
        pred, obs = nu_slice_pe(p)
        if pred is None or obs.sum() <= 0 or pred.sum() <= 0:
            continue
        dead = dead_opdets_for_run(int(run[i]))     # run1 -> ()
        c0.append(neyman_masked(pred, obs, dead))
        cS.append(neyman_masked(pred * args.scale, obs, dead))
    c0 = np.array(c0); cS = np.array(cS)
    if not len(c0):
        raise SystemExit("no pi0 candidates with a cascade nu slice")
    impr = float(np.mean(cS < c0))
    print(f">>> N={len(c0)} | median chi2: pred {np.median(c0):.0f} -> "
          f"pred*{args.scale:g} {np.median(cS):.0f} | "
          f"{impr:.0%} of events improved (chi2 down)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lb = np.linspace(0, 6, 31)
    fig, ax = plt.subplots(figsize=(6.6, 4.5))
    ax.hist(np.clip(np.log10(np.clip(c0, 1, None)), 0, 6), bins=lb,
            histtype="step", lw=2, color="0.4",
            label=f"pred (gamma_beam)  med {np.median(c0):.0f}")
    ax.hist(np.clip(np.log10(np.clip(cS, 1, None)), 0, 6), bins=lb,
            histtype="step", lw=2, color="#d62728",
            label=f"pred x {args.scale:g}  med {np.median(cS):.0f}")
    ax.set(xlabel=r"$\log_{10}$ flash $\chi^2$ (nu slice)", ylabel="events",
           title=f"{args.sample_tag} reco-NC "
                 f"{'eq2' if args.eq2 else 'ge2'} pi0: chi2 vs pred scale\n"
                 f"N={len(c0)}, {impr:.0%} improved with x{args.scale:g}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{args.plots}/chi2_scale_{args.sample_tag}.png", dpi=110)
    plt.close(fig)
    print(f">>> plots -> {args.plots}/chi2_scale_{args.sample_tag}.png")


if __name__ == "__main__":
    main()
