"""Vertex position + dwall for the CC/NC two-photon pi0 selection: data vs MC.

Cosmic-contamination check: cosmic-induced vertices pile up near the active-TPC
boundaries (entering tracks, edge effects), so a data excess concentrated at
small dwall (or at the y/z edges) relative to the neutrino-overlay MC would be
a spatial signature of cosmic background. Genuine neutrinos are ~uniform in the
active volume. Produces, per reco stream (CC = likely in-time nu control,
NC = the probe), a 2x2 panel of vtx X / Y / Z / dwall, beam data (points) over
POT-scaled truth-category-stacked MC, using the >=2 photon pi0 selection.

    PYTHONPATH=<repo> python3 .../vtx_datamc_pi0.py \
        --mc-ntuple .../dlgen2_..._67k.root \
        --data-ntuple .../dlgen2_larformer_ntuple_bnb5e19_full.root \
        --plots plots_datamc_full/
"""
import argparse
import os

import numpy as np

from pi0_mass_analysis import CATS, CAT_COLORS
from flashchi2_ncpi0 import load, TPC_LO, TPC_HI


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mc-ntuple", required=True)
    ap.add_argument("--data-ntuple", required=True)
    ap.add_argument("--plots", required=True)
    ap.add_argument("--pot", type=float, default=4.4e19)
    ap.add_argument("--dwall-edge", type=float, default=20.0,
                    help="dwall threshold [cm] for the boundary-fraction table")
    args = ap.parse_args()
    os.makedirs(args.plots, exist_ok=True)

    mc = load(args.mc_ntuple, False, args.pot)
    da = load(args.data_ntuple, True, args.pot)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    VARS = [("vx", np.linspace(0, TPC_HI[0], 27), "vtx X (drift) [cm]"),
            ("vy", np.linspace(TPC_LO[1], TPC_HI[1], 27), "vtx Y [cm]"),
            ("vz", np.linspace(0, TPC_HI[2], 27), "vtx Z (beam) [cm]"),
            ("dwall", np.linspace(0, 130, 27), "dwall (dist to TPC wall) [cm]")]

    for want_cc, slab, sabb in ((True, "reco-CC (likely nu, in-time)", "cc"),
                                (False, "reco-NC", "nc")):
        sm = mc["sel2p"] & (mc["reco_cc"] == want_cc)
        sd = da["sel2p"] & (da["reco_cc"] == want_cc)
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        for ax, (key, bins, xlab) in zip(axes.ravel(), VARS):
            ctr = 0.5 * (bins[:-1] + bins[1:])
            stack = [np.clip(mc[key][sm & (mc["cat"] == c)], bins[0],
                             bins[-1] - 1e-6) for c in range(6)]
            ws = [mc["w"][sm & (mc["cat"] == c)] for c in range(6)]
            ax.hist(stack, bins=bins, weights=ws, stacked=True,
                    color=CAT_COLORS,
                    label=[f"{CATS[c]} ({ws[c].sum():.0f})" for c in range(6)])
            dh, _ = np.histogram(np.clip(da[key][sd], bins[0], bins[-1] - 1e-6),
                                 bins=bins)
            ax.errorbar(ctr, dh, yerr=np.sqrt(np.clip(dh, 1, None)),
                        fmt="ko", ms=3, lw=1, capsize=0,
                        label=f"beam data ({int(dh.sum())})")
            ax.set(xlabel=xlab, ylabel=f"events / {args.pot:.1e} POT")
            ax.grid(alpha=0.3)
            if key == "vx":
                ax.legend(fontsize=6.5, ncol=1)
        fig.suptitle(f"{slab}: vertex position + dwall, beam data vs "
                     "prediction (>=2 photon pi0 selection)", fontsize=12)
        fig.tight_layout()
        fig.savefig(f"{args.plots}/vtx_datamc_{sabb}.png", dpi=110)
        plt.close(fig)

    # boundary-fraction table: fraction of selected events with dwall < edge
    e = args.dwall_edge
    print(f"== fraction of pi0 candidates near the wall (dwall < {e:.0f} cm) ==")
    for want_cc, slab in ((True, "reco-CC"), (False, "reco-NC")):
        sm = mc["sel2p"] & (mc["reco_cc"] == want_cc)
        sd = da["sel2p"] & (da["reco_cc"] == want_cc)
        fm = (mc["w"][sm & (mc["dwall"] < e)].sum()
              / max(mc["w"][sm].sum(), 1e-9))
        fd = int((sd & (da["dwall"] < e)).sum()) / max(int(sd.sum()), 1)
        print(f"  [{slab}] data {fd:.1%} (N={int(sd.sum())})  vs  "
              f"MC {fm:.1%}  |  median dwall: data "
              f"{np.median(da['dwall'][sd]):.1f} vs MC "
              f"{np.median(mc['dwall'][sm]):.1f} cm")
    print(f">>> plots -> {args.plots}")


if __name__ == "__main__":
    main()
