"""Invariant-mass plots following the SBND-ordered cutflow (sbnd_cutflow.py:
FV -> exactly-1 primary mu>143 -> 0 cpi>25 -> exactly-2 photons -> m_gg in
[30,300) -> flash chi2 < 1e4).

One stacked m_gg panel per stage where the pair exists:
  A. after the exactly-2-photon cut (mass window drawn, not applied)
  B. after the flash-chi2 cut (mass window drawn; events outside it shown)
MC stacked by SBND truth category, EXT cosmic on top, beam data as points.

    PYTHONPATH=./ python3 sbnd_mgg_sbndflow.py --muon-finder union \
        --mc-ntuple .. --mc-table .. --data-ntuple .. --data-table .. \
        --ext-ntuple .. --ext-table .. --plots <dir>
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sbnd_cutflow as CF  # noqa: E402

CATS = ["signal CC 1pi0", "true vtx out of FV", "NC", "CC wrong/undet pi0",
        "CC 1pi0 + charged pi", "CC 1pi0, soft muon"]
CAT_COLORS = ["#c62828", "#bdbdbd", "#1565c0", "#ef6c00", "#6a1b9a", "#795548"]
EXT_COLOR = "#e5e5e5"
MGG_WIN = (30.0, 300.0)


def main():
    ap = CF.argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for s in ("mc", "data", "ext"):
        ap.add_argument(f"--{s}-ntuple", required=True)
        ap.add_argument(f"--{s}-table", required=True)
    ap.add_argument("--ext-scale", type=float, default=1.1818)
    ap.add_argument("--ext-row-min", type=int, default=100000)
    ap.add_argument("--recal-gamma-a", type=float, default=0.01553)
    ap.add_argument("--recal-gamma-b", type=float, default=-12.80)
    ap.add_argument("--shower-bdt-min", type=float, default=0.192)
    ap.add_argument("--muon-finder", default="union",
                    choices=["segmenter", "larpid", "union"])
    ap.add_argument("--pot", type=float, default=4.4e19)
    ap.add_argument("--plots", required=True)
    args = ap.parse_args()
    os.makedirs(args.plots, exist_ok=True)

    mc_steps, cat, wmc, m_mc = CF.build(args.mc_ntuple, args.mc_table, False, args)
    da_steps, _, _, m_da = CF.build(args.data_ntuple, args.data_table, True, args)
    ex_steps, _, _, m_ex = CF.build(args.ext_ntuple, args.ext_table, True, args)
    we = np.zeros(len(m_ex))
    we[args.ext_row_min:] = args.ext_scale

    # stage masks from the ordered flow: index 4 = exactly-2 photons,
    # 6 = + mass window + flash chi2. For panel B drop the mass window from
    # the mask (draw it instead) so the sidebands stay visible.
    stage5_mc, stage5_da, stage5_ex = mc_steps[4][1], da_steps[4][1], ex_steps[4][1]
    bins = np.linspace(0, 500, 51)
    ctr = 0.5 * (bins[:-1] + bins[1:])

    def panel(mm, dm, em, ttl, fname):
        stack = [np.clip(m_mc[mm & (cat == c)], 0, 499) for c in range(6)]
        ws = [wmc[mm & (cat == c)] for c in range(6)]
        stack.append(np.clip(m_ex[em], 0, 499))
        ws.append(we[em])
        labels = [f"{CATS[c]} ({ws[c].sum():.0f})" for c in range(6)]
        labels.append(f"EXT cosmic ({ws[6].sum():.0f})")
        fig, ax = plt.subplots(figsize=(7.0, 4.8))
        ax.hist(stack, bins=bins, weights=ws, stacked=True,
                color=CAT_COLORS + [EXT_COLOR], label=labels)
        dh, _ = np.histogram(np.clip(m_da[dm], 0, 499), bins)
        ax.errorbar(ctr, dh, yerr=np.sqrt(np.clip(dh, 1, None)), fmt="ko",
                    ms=3.5, lw=1, label=f"beam data ({int(dh.sum())})")
        for x in MGG_WIN:
            ax.axvline(x, color="crimson", ls="--", lw=1.2)
        ax.axvline(134.9768, color="0.4", ls=":", lw=1.1)
        pred = sum(w.sum() for w in ws)
        ax.set(xlabel=r"$m_{\gamma\gamma}$ [MeV]",
               ylabel=f"events / {args.pot:.1e} POT",
               title=f"{ttl}\ndata {int(dh.sum())} vs pred {pred:.0f} (MC+EXT)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(args.plots, fname), dpi=110)
        plt.close(fig)
        pk = lambda v, m: (v[m] >= MGG_WIN[0]) & (v[m] < MGG_WIN[1])
        sg = wmc[mm & (cat == 0)][pk(m_mc, mm & (cat == 0))].sum()
        ot = (sum(wmc[mm & (cat == c)][pk(m_mc, mm & (cat == c))].sum()
                  for c in range(1, 6)) + we[em][pk(m_ex, em)].sum())
        print(f"{fname}: in-window signal {sg:.1f} | other {ot:.1f} | "
              f"purity {sg/(sg+ot):.3f}")

    fchi_mc = np.load(args.mc_table)["flash_chi2"]
    fchi_da = np.load(args.data_table)["flash_chi2"]
    fchi_ex = np.load(args.ext_table)["flash_chi2"]
    chi_ok = lambda f: np.isfinite(f) & (f < 1e4)
    panel(stage5_mc, stage5_da, stage5_ex,
          "SBND flow, after exactly-2-photon cut (mass window drawn)",
          "mgg_sbndflow_2gamma.png")
    panel(stage5_mc & chi_ok(fchi_mc), stage5_da & chi_ok(fchi_da),
          stage5_ex & chi_ok(fchi_ex),
          "SBND flow, after flash-chi2 cut (mass window drawn)",
          "mgg_sbndflow_final.png")
    print(f">>> plots -> {args.plots}")


if __name__ == "__main__":
    main()
