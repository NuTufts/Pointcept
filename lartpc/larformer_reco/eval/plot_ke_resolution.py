"""1D fractional KE resolution from merged eval records.

Per species, histogram (reco_ke - true_ke)/true_ke for particles that were
ATTACHED to a reconstructed nu interaction (found_B), with a dedicated,
visually separated underflow bin counting the particles that did NOT attach
(the eval stores those with reco_ke = 0, which the 2D reco-vs-true plot mixes
into its bottom row).

    PYTHONPATH=./ python3 lartpc/larformer_reco/eval/plot_ke_resolution.py \
        --results lartpc/larformer_reco/results_eval_reco_<TAG>.npz \
        --plots lartpc/larformer_reco/plots/<dir> [--min-true-ke 50]
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SPECIES = ["e", "gamma", "mu", "pi", "p"]
XLO, XHI, NBINS = -1.2, 1.2, 48
UNDERFLOW_X = -1.55          # placement of the "not attached" bin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="merged eval .npz")
    ap.add_argument("--plots", required=True, help="output directory")
    ap.add_argument("--min-true-ke", type=float, default=25.0,
                    help="drop particles below this true KE [MeV] "
                         "(protects the ratio; default 25)")
    args = ap.parse_args()

    d = np.load(args.results)
    sp = d["species"]
    tke = d["true_ke"].astype(np.float64)
    rke = d["reco_ke"].astype(np.float64)
    att = d["found_B"].astype(bool)
    os.makedirs(args.plots, exist_ok=True)

    edges = np.linspace(XLO, XHI, NBINS + 1)
    width = edges[1] - edges[0]
    print(f"species |     N | attached | not-attached | frac@0-row(2D)")
    for i, name in enumerate(SPECIES):
        m = (sp == i) & (tke > args.min_true_ke)
        n = int(m.sum())
        if n == 0:
            continue
        na = int((m & att).sum())
        frac = (rke[m & att] - tke[m & att]) / tke[m & att]
        n_unatt = n - na
        print(f"{name:>7} | {n:5d} |  {na/n:6.2f}  |    {n_unatt/n:6.2f}    |"
              f" (unattached fraction = height of reco=0 band)")

        fig, ax = plt.subplots(figsize=(6, 4))
        counts, _ = np.histogram(np.clip(frac, XLO + 1e-9, XHI - 1e-9), edges)
        ax.bar(0.5 * (edges[:-1] + edges[1:]), counts, width=width,
               color="tab:blue", label=f"attached (N={na})")
        ax.bar([UNDERFLOW_X], [n_unatt], width=2.5 * width, color="tab:red",
               label=f"not attached (N={n_unatt})")
        ax.axvline(0.0, color="k", lw=0.8, ls="--", alpha=0.6)
        ax.set(xlabel="(reco KE - true KE) / true KE",
               ylabel="particles",
               title=f"{name}: KE resolution "
                     f"(true KE > {args.min_true_ke:.0f} MeV)")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out = f"{args.plots}/keres_{name}.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(f"        -> {out}")


if __name__ == "__main__":
    main()
