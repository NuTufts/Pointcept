"""Reproduce the flashchi2_{cc,nc}_{eq2,ge2} plots from the pre-built selection
npz tables (mc/data/ext), instead of re-scanning the cascades.

Same output as flashchi2_ncpi0.py but reads flash_chi2 / reco_cc / cat /
m_vtx2start / w straight from the tables -- so it honours whatever mu-ke-min
was used to build them, and works when the cascades are unavailable (e.g. mid
reprocessing). Per-stream cut line: reco-CC --chi2-cut, reco-NC --chi2-cut-nc.

    python3 flashchi2_from_tables.py --mc-npz .. --data-npz .. --ext-npz .. \
        --ext-scale 4.125814 --chi2-cut 1e4 --chi2-cut-nc 3162.3 --plots <dir>
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pi0_mass_analysis import CATS, CAT_COLORS

PEAK_LO, PEAK_HI = 100.0, 170.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc-npz", required=True)
    ap.add_argument("--data-npz", required=True)
    ap.add_argument("--ext-npz", required=True)
    ap.add_argument("--ext-scale", type=float, default=4.125814)
    ap.add_argument("--chi2-cut", type=float, default=1e4)
    ap.add_argument("--chi2-cut-nc", type=float, default=None)
    ap.add_argument("--pot", type=float, default=4.4e19)
    ap.add_argument("--plots", required=True)
    args = ap.parse_args()
    os.makedirs(args.plots, exist_ok=True)
    mc = np.load(args.mc_npz); da = np.load(args.data_npz)
    ex = np.load(args.ext_npz)

    lbins = np.linspace(0, 8, 33)
    ctr = 0.5 * (lbins[:-1] + lbins[1:])

    def lchi(t):
        return np.log10(np.clip(t["flash_chi2"], 1, None))

    def base(t, eq2, want_cc):
        s = (t["sel_eq2"] if eq2 else t["sel_ge2"]) & (t["reco_cc"] == want_cc)
        return s & np.isfinite(t["flash_chi2"])

    lmc, lda, lex = lchi(mc), lchi(da), lchi(ex)
    for want_cc, slab, sabb in ((True, "reco-CC (likely nu, in-time)", "cc"),
                                (False, "reco-NC", "nc")):
        cutv = (args.chi2_cut if want_cc or args.chi2_cut_nc is None
                else args.chi2_cut_nc)
        lcut = np.log10(cutv)
        for eq2, vlab, vabb in ((True, "exactly 2", "eq2"),
                                (False, ">=2", "ge2")):
            sm, sd, se = (base(mc, eq2, want_cc), base(da, eq2, want_cc),
                          base(ex, eq2, want_cc))
            fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
            for ax, peak_only, ttl in (
                    (axes[0], False, "all m$_{\\gamma\\gamma}$"),
                    (axes[1], True, f"near-peak {PEAK_LO:.0f}-{PEAK_HI:.0f} MeV")):
                def pk(t, s):
                    return (s & (t["m_vtx2start"] >= PEAK_LO)
                            & (t["m_vtx2start"] < PEAK_HI)) if peak_only else s
                mm, dm, em = pk(mc, sm), pk(da, sd), pk(ex, se)
                stack = [np.clip(lmc[mm & (mc["cat"] == c)], 0, 7.999)
                         for c in range(6)]
                ws = [mc["w"][mm & (mc["cat"] == c)] for c in range(6)]
                colors = list(CAT_COLORS)
                labels = [f"{CATS[c]} ({ws[c].sum():.0f})" for c in range(6)]
                stack.append(np.clip(lex[em], 0, 7.999))
                ws.append(np.full(int(em.sum()), args.ext_scale))
                colors.append("#e5e5e5")
                labels.append(f"EXT cosmic ({ws[-1].sum():.0f})")
                ax.hist(stack, bins=lbins, weights=ws, stacked=True,
                        color=colors, label=labels)
                dh, _ = np.histogram(np.clip(lda[dm], 0, 7.999), bins=lbins)
                ax.errorbar(ctr, dh, yerr=np.sqrt(np.clip(dh, 1, None)),
                            fmt="ko", ms=3.5, lw=1, capsize=0,
                            label=f"beam data ({int(dh.sum())})")
                ax.axvline(lcut, color="crimson", ls="--", lw=1.4,
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
            print(f">>> wrote flashchi2_{sabb}_{vabb}.png")


if __name__ == "__main__":
    main()
