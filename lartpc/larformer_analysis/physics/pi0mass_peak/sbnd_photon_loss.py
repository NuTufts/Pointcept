"""Where are true photons LOST? Reco failure-stage breakdown of true photons vs
three variables -- true KE, true VISIBLE energy (a_gamma * de-double-counted
charge), and conversion distance (nu-vertex origin -> pair-conversion point) --
to diagnose the source of the SBND CC 1pi0 photon inefficiency.

Reads the per-photon records written by
lartpc/larformer_reco/eval/eval_reco_performance.py (--out) and reuses its
stage_codes (0 ok, 1 misID, 2 seg!att, 3 noInst, 4 missSlice). For each variable
it makes a 2-panel figure: LEFT stacked COUNTS by stage (the distribution of
photons, with the lost fraction shown by the non-green stack), RIGHT stacked
FRACTION by stage (the reference stage_gamma.png view -- fraction lost at each
value). 'Lost' = any stage != ok.

    python3 sbnd_photon_loss.py --records eval_records.npz --plots plots_sbnd_cc1pi0
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "..", "larformer_reco", "eval"))
from eval_reco_performance import (stage_codes, STAGE_NAMES,  # noqa: E402
                                   STAGE_DESC, SPECIES)

STAGE_COLORS = ["#3ca03c", "#e08000", "#d03030", "#5060d0", "#909090"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--plots", default="plots_sbnd_cc1pi0")
    args = ap.parse_args()
    os.makedirs(args.plots, exist_ok=True)

    z = np.load(args.records, allow_pickle=True)
    rec = {k: z[k] for k in z.files if k not in ("species_names",)}
    a_gamma = float(z["gamma_evis_calib"]) if "gamma_evis_calib" in z.files else 0.0
    g = SPECIES.index("gamma")
    m_g = np.asarray(rec["species"]) == g
    codes = stage_codes(rec)

    print(f">>> {int(m_g.sum())} true photons | a_gamma={a_gamma:.5f} MeV/charge")
    lost = codes[m_g] != 0
    print(f"    overall: ok {100*(~lost).mean():.1f}%  lost {100*lost.mean():.1f}%")
    for k in range(1, len(STAGE_NAMES)):
        f = (codes[m_g] == k).mean()
        print(f"      {STAGE_NAMES[k]:10s} {100*f:5.1f}%  ({STAGE_DESC[STAGE_NAMES[k]]})")

    tke = np.asarray(rec["true_ke"], float)
    vis = a_gamma * np.asarray(rec["q_true"], float)
    cd = np.asarray(rec["conv_dist"], float)

    VARS = [("true_ke", tke, "true KE [MeV]",
             np.array([0, 25, 50, 100, 200, 400, 800, 1500])),
            ("visKE", vis, "true visible energy [MeV]",
             np.array([0, 20, 40, 80, 160, 320, 640, 1200])),
            ("convdist", cd, "true |conversion point - origin| [cm]",
             np.array([0, 2, 5, 10, 20, 35, 55, 80, 120]))]

    for key, x, xlabel, bins in VARS:
        good = m_g & np.isfinite(x)
        xc = x[good]
        cc = codes[good]
        labels = [f"{lo:.0f}-{hi:.0f}" for lo, hi in zip(bins[:-1], bins[1:])]
        ctr = np.arange(len(labels))
        cnt = np.zeros((len(STAGE_NAMES), len(labels)))
        for j, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
            b = (xc >= lo) & (xc < hi)
            for k in range(len(STAGE_NAMES)):
                cnt[k, j] = int(((cc == k) & b).sum())
        tot = cnt.sum(0)
        frac = np.divide(cnt, tot, out=np.zeros_like(cnt), where=tot > 0)

        fig, (axc, axf) = plt.subplots(1, 2, figsize=(12.5, 4.6))
        for ax, data, ylab, norm in ((axc, cnt, "true photons", False),
                                     (axf, frac, "fraction of true photons",
                                      True)):
            bottoms = np.zeros(len(labels))
            for k in range(len(STAGE_NAMES)):
                ax.bar(ctr, data[k], bottom=bottoms, color=STAGE_COLORS[k],
                       label=STAGE_NAMES[k], width=0.85)
                bottoms += data[k]
            ax.set_xticks(ctr)
            ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylab)
            if norm:
                ax.set_ylim(0, 1.12)
                for xx, N in enumerate(tot.astype(int)):
                    ax.text(xx, 1.01, f"N={N}", ha="center", va="bottom",
                            fontsize=6)
        axc.set_title("stage mix (counts) -- where the lost photons are")
        axf.set_title("stage mix (fraction) -- fraction lost per bin")
        axf.legend(fontsize=8, ncol=5, loc="lower center",
                   bbox_to_anchor=(0.5, -0.34))
        fig.suptitle(f"true-photon reco failure-stage vs {xlabel.split(' [')[0]}"
                     f"  (green=ok; N={int(good.sum())})", fontsize=11)
        fig.tight_layout()
        p = os.path.join(args.plots, f"photonloss_stage_vs_{key}.png")
        fig.savefig(p, dpi=110)
        plt.close(fig)
        print(">>> wrote", p)


if __name__ == "__main__":
    main()
