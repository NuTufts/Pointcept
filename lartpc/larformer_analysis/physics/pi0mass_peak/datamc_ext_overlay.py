"""3-way pi0 overlay: beam data vs (MC + EXT cosmic) prediction.

Completes the beam-data prediction with the beam-off cosmic contribution.
Reads three selection tables written by pi0_mass_analysis.py (--out):
  - MC   : xsecWeight-scaled to 4.4e19 POT (w already scaled), truth cats 0-5
  - data : beam data, unit weight, single cat
  - EXT  : beam-off cosmic, unit weight; scaled here by the spill ratio

EXT scale (see memory extbnb-cosmic-normalization): beam/EXT spills
10375708/58677653 = 0.17682554549 for the FULL EXT sample; for a partial
sample pass --ext-scale = 0.17682554549 / f  (f = fraction of EXT processed).

Total prediction = MC(stacked by truth cat) + EXT(single cosmic component).
Plots m_gg and reco p_pi0, split reco-CC / reco-NC, for the >=2 and exactly-2
photon selections.

    python3 datamc_ext_overlay.py --mc-npz ... --data-npz ... --ext-npz ... \
        --ext-scale 4.457 --plots plots_ext_val/
"""
import argparse
import os

import numpy as np

from pi0_mass_analysis import CATS, CAT_COLORS, PI0_MASS

EXT_COLOR = "#e5e5e5"      # light grey so black data points read on top
                           # (distinct from the darker out-of-FV grey #bdbdbd)
FULL_EXT_SCALE = 0.17682554549   # beam/EXT full-sample spill ratio


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mc-npz", required=True)
    ap.add_argument("--data-npz", required=True)
    ap.add_argument("--ext-npz", required=True)
    ap.add_argument("--ext-scale", type=float, default=FULL_EXT_SCALE,
                    help="per-EXT-event weight (0.17682554549 / fraction)")
    ap.add_argument("--plots", required=True)
    ap.add_argument("--pot", type=float, default=4.4e19)
    ap.add_argument("--flashchi2-cut", type=float, default=None,
                    help="provisional cut: keep events with dead-PMT-masked "
                         "flash_chi2 < this (requires flash_chi2 in the npz)")
    args = ap.parse_args()
    os.makedirs(args.plots, exist_ok=True)

    mc = np.load(args.mc_npz)
    da = np.load(args.data_npz)
    ex = np.load(args.ext_npz)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def stream_mask(t, want_cc, eq2):
        sel = t["sel_eq2"] if eq2 else t["sel_ge2"]
        sel = sel & (t["reco_cc"] == want_cc)
        if args.flashchi2_cut is not None:
            if "flash_chi2" not in t.files:
                raise SystemExit("--flashchi2-cut needs flash_chi2 in the npz "
                                 "(rebuild tables with --cascade-dir)")
            fc = t["flash_chi2"]
            sel = sel & np.isfinite(fc) & (fc < args.flashchi2_cut)
        return sel

    def panel(obs_key, bins, xlabel, fname_base, clip_hi):
        for want_cc, sabb, slab in ((True, "recoCC", "reco-CC"),
                                    (False, "recoNC", "reco-NC")):
            for eq2, vabb, vlab in ((False, "ge2", ">=2"),
                                    (True, "eq2", "exactly 2")):
                mm = stream_mask(mc, want_cc, eq2) & np.isfinite(mc[obs_key])
                dm = stream_mask(da, want_cc, eq2) & np.isfinite(da[obs_key])
                em = stream_mask(ex, want_cc, eq2) & np.isfinite(ex[obs_key])
                ctr = 0.5 * (bins[:-1] + bins[1:])

                # stacked prediction: 6 MC truth cats + EXT cosmic on top
                stack = [np.clip(mc[obs_key][mm & (mc["cat"] == c)],
                                 bins[0], clip_hi) for c in range(6)]
                ws = [mc["w"][mm & (mc["cat"] == c)] for c in range(6)]
                stack.append(np.clip(ex[obs_key][em], bins[0], clip_hi))
                ws.append(np.full(int(em.sum()), args.ext_scale))
                colors = CAT_COLORS + [EXT_COLOR]
                labels = [f"{CATS[c]} ({ws[c].sum():.0f})" for c in range(6)]
                labels.append(f"EXT cosmic ({ws[6].sum():.0f})")

                fig, ax = plt.subplots(figsize=(6.8, 4.6))
                ax.hist(stack, bins=bins, weights=ws, stacked=True,
                        color=colors, label=labels)
                pred_tot = sum(w.sum() for w in ws)
                dh, _ = np.histogram(np.clip(da[obs_key][dm], bins[0], clip_hi),
                                     bins=bins)
                ax.errorbar(ctr, dh, yerr=np.sqrt(np.clip(dh, 1, None)),
                            fmt="ko", ms=3.5, lw=1, capsize=0,
                            label=f"beam data ({int(dh.sum())})")
                if obs_key.startswith("m_"):
                    ax.axvline(PI0_MASS, color="0.4", ls=":", lw=1.1)
                ax.set(xlabel=xlabel, ylabel=f"events / {args.pot:.1e} POT",
                       title=f"{slab}: {fname_base} ({vlab} photons)\n"
                             f"data {int(dh.sum())} vs pred "
                             f"{pred_tot:.0f} (MC+EXT)")
                ax.legend(fontsize=7)
                ax.grid(alpha=0.3)
                fig.tight_layout()
                fig.savefig(f"{args.plots}/{fname_base}_{sabb}_{vabb}.png",
                            dpi=110)
                plt.close(fig)

    panel("m_vtx2start", np.linspace(0, 500, 51),
          r"$m_{\gamma\gamma}$ [MeV]", "mgg_ext", 499)
    panel("p_reco", np.linspace(0, 1200, 49),
          r"reco $p_{\pi^0}$ [MeV/c]", "ppi0_ext", 1199)

    # ---- purity vs reco m_gg: true pi0 signal / (all MC bkg + EXT cosmic) ---
    # with (solid) and without (dashed) the flash-chi2 cut, showing its gain.
    cutv = args.flashchi2_cut if args.flashchi2_cut is not None else 10000.0
    if "flash_chi2" in mc.files and "flash_chi2" in ex.files:
        mbins = np.linspace(0, 500, 26)
        mctr = 0.5 * (mbins[:-1] + mbins[1:])
        for want_cc, sabb, slab in ((True, "recoCC", "reco-CC"),
                                    (False, "recoNC", "reco-NC")):
            base_mc = (mc["sel_ge2"] & (mc["reco_cc"] == want_cc)
                       & np.isfinite(mc["m_vtx2start"]))
            base_ex = (ex["sel_ge2"] & (ex["reco_cc"] == want_cc)
                       & np.isfinite(ex["m_vtx2start"]))
            fig, ax = plt.subplots(figsize=(6.6, 4.5))
            for use_cut, sty, al, lab in (
                    (False, "s--", 0.45, "no cut"),
                    (True, "o-", 1.0, f"flash chi2<{cutv:.0f}")):
                mm = base_mc.copy(); em = base_ex.copy()
                if use_cut:
                    mm = mm & np.isfinite(mc["flash_chi2"]) & (mc["flash_chi2"] < cutv)
                    em = em & np.isfinite(ex["flash_chi2"]) & (ex["flash_chi2"] < cutv)
                pur, err = [], []
                for lo, hi in zip(mbins[:-1], mbins[1:]):
                    mb = mm & (mc["m_vtx2start"] >= lo) & (mc["m_vtx2start"] < hi)
                    eb = em & (ex["m_vtx2start"] >= lo) & (ex["m_vtx2start"] < hi)
                    sig = mc["w"][mb & (mc["cat"] <= 1)].sum()
                    tot = mc["w"][mb].sum() + args.ext_scale * int(eb.sum())
                    p = sig / tot if tot > 0 else np.nan
                    n = int(mb.sum()) + int(eb.sum())
                    pur.append(p)
                    err.append(np.sqrt(p * (1 - p) / n)
                               if n and np.isfinite(p) else 0)
                ax.errorbar(mctr, pur, yerr=err, fmt=sty, ms=4,
                            color="#1f77b4", alpha=al, label=lab)
            ax.axvline(PI0_MASS, color="0.4", ls=":", lw=1.1)
            ax.set(xlabel=r"reco $m_{\gamma\gamma}$ [MeV]", ylabel="purity",
                   ylim=(0, 1.05),
                   title=f"{slab}: pi0 purity vs reco mass (>=2 gamma)\n"
                         "true pi0 signal / (all MC + EXT cosmic)")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(f"{args.plots}/purity_mgg_{sabb}.png", dpi=110)
            plt.close(fig)

    # near-peak reco-NC data/pred ratio, with and without the EXT component
    print("== reco-NC near-peak (100-170 MeV) data vs prediction ==")
    for eq2, vlab in ((False, ">=2"), (True, "exactly-2")):
        mm = stream_mask(mc, False, eq2) & np.isfinite(mc["m_vtx2start"])
        em = stream_mask(ex, False, eq2) & np.isfinite(ex["m_vtx2start"])
        dm = stream_mask(da, False, eq2) & np.isfinite(da["m_vtx2start"])
        pk = lambda t, m: m & (t["m_vtx2start"] >= 100) & (t["m_vtx2start"] < 170)
        mc_pk = mc["w"][pk(mc, mm)].sum()
        ext_pk = args.ext_scale * int(pk(ex, em).sum())
        dat_pk = int(pk(da, dm).sum())
        print(f"  [{vlab:9s}] data {dat_pk:4d} | MC {mc_pk:6.1f} "
              f"(ratio {dat_pk/max(mc_pk,1e-9):.2f}) | MC+EXT "
              f"{mc_pk+ext_pk:6.1f} (ratio {dat_pk/max(mc_pk+ext_pk,1e-9):.2f}) "
              f"| EXT alone {ext_pk:6.1f}")
    print(f">>> plots -> {args.plots}  (EXT scale {args.ext_scale:.4f})")


if __name__ == "__main__":
    main()
