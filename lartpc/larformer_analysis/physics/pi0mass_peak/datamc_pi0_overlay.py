"""Data/MC overlay for the pi0 mass peak and reconstructed pi0 momentum.

Reads the selection tables saved by pi0_mass_analysis.py --out:
  --mc-npz    sim table (weights already scaled to the data POT, 4.4e19)
  --data-npz  beam-data table (unit weights)

Draws beam data as points (Poisson errors) over the truth-category-stacked
MC prediction, split reco-CC / reco-NC, for m_gg (vertex->start, >=2 gamma)
and reco p_pi0.

    PYTHONPATH=./ python3 .../datamc_pi0_overlay.py \
        --mc-npz .../pi0_selection_table.npz \
        --data-npz .../pi0_selection_table_bnb5e19_full.npz --plots plots/
"""
import argparse
import os

import numpy as np

from pi0_mass_analysis import CATS, CAT_COLORS, PI0_MASS


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mc-npz", required=True)
    ap.add_argument("--data-npz", required=True)
    ap.add_argument("--plots", required=True)
    ap.add_argument("--pot", type=float, default=4.4e19)
    args = ap.parse_args()
    os.makedirs(args.plots, exist_ok=True)

    mc = np.load(args.mc_npz)
    da = np.load(args.data_npz)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def overlay(var, bins, xlabel, fname, title, vline=None):
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=False)
        ctr = 0.5 * (bins[:-1] + bins[1:])
        for ax, want_cc, cclab in ((axes[0], True, "reco-CC"),
                                   (axes[1], False, "reco-NC")):
            mm = (mc["sel_ge2"] & (mc["reco_cc"] == want_cc)
                  & np.isfinite(mc[var]))
            md = (da["sel_ge2"] & (da["reco_cc"] == want_cc)
                  & np.isfinite(da[var]))
            stack = [np.clip(mc[var][mm & (mc["cat"] == c)], bins[0],
                             bins[-1] - 1e-3) for c in range(6)]
            ws = [mc["w"][mm & (mc["cat"] == c)] for c in range(6)]
            ax.hist(stack, bins=bins, weights=ws, stacked=True,
                    color=CAT_COLORS,
                    label=[f"{CATS[c]} ({ws[c].sum():.0f})"
                           for c in range(6)])
            dh, _ = np.histogram(np.clip(da[var][md], bins[0],
                                         bins[-1] - 1e-3), bins=bins)
            ax.errorbar(ctr, dh, yerr=np.sqrt(np.clip(dh, 1, None)),
                        fmt="ko", ms=3.5, lw=1, capsize=0,
                        label=f"beam data ({int(dh.sum())})")
            if vline is not None:
                ax.axvline(vline, color="k", ls=":", lw=1.2)
            wmc = sum(x.sum() for x in ws)
            ax.set(xlabel=xlabel, ylabel=f"events / {args.pot:.1e} POT",
                   title=f"{cclab}   data/MC = {dh.sum()/max(wmc,1e-9):.2f}")
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)
            print(f"  {fname} {cclab}: data {int(dh.sum())} | "
                  f"MC {wmc:.1f} | ratio {dh.sum()/max(wmc,1e-9):.3f}")
        fig.suptitle(title, fontsize=11)
        fig.tight_layout()
        fig.savefig(f"{args.plots}/{fname}.png", dpi=110)
        plt.close(fig)

    print("== data/MC overlays (MC scaled to "
          f"{args.pot:.1e} POT) ==")
    overlay("m_vtx2start", np.linspace(0, 500, 26),
            r"$m_{\gamma\gamma}$ [MeV]", "datamc_pi0_mass",
            "two-photon invariant mass: beam data vs prediction "
            "(vertex->start dir, >=2 gamma)", vline=PI0_MASS)
    overlay("p_reco", np.linspace(0, 1200, 25),
            r"reco $p_{\pi^0}$ [MeV/c]", "datamc_pi0_momentum",
            "reconstructed pi0 momentum: beam data vs prediction "
            "(>=2 gamma)")
    print(f">>> plots -> {args.plots}")


if __name__ == "__main__":
    main()
