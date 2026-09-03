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
    ap.add_argument("--combined-chi2-cut", type=float, default=None,
                    help="also make combined CC+NC m_gg panels (eq2 and ge2, "
                         "no stream split) with this SINGLE flash-chi2 cut "
                         "(e.g. 3162.3 = log10<3.5); signal = true CC+NC pi0")
    ap.add_argument("--plots", required=True)
    ap.add_argument("--pot", type=float, default=4.4e19)
    ap.add_argument("--flashchi2-cut-nc", type=float, default=None,
                    help="separate (tighter) flash-chi2 cut for the reco-NC "
                         "stream; defaults to --flashchi2-cut. The NC cosmic "
                         "contamination extends to lower chi2 than CC, so NC "
                         "can afford log10(chi2)<3.5 (=3162) where CC uses 1e4.")
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

    def cut_for(want_cc):
        """Per-stream flash-chi2 cut: CC keeps --flashchi2-cut, NC may be
        tighter (--flashchi2-cut-nc)."""
        if want_cc or args.flashchi2_cut_nc is None:
            return args.flashchi2_cut
        return args.flashchi2_cut_nc

    def stream_mask(t, want_cc, eq2):
        sel = t["sel_eq2"] if eq2 else t["sel_ge2"]
        sel = sel & (t["reco_cc"] == want_cc)
        cv = cut_for(want_cc)
        if cv is not None:
            if "flash_chi2" not in t.files:
                raise SystemExit("--flashchi2-cut needs flash_chi2 in the npz "
                                 "(rebuild tables with --cascade-dir)")
            fc = t["flash_chi2"]
            sel = sel & np.isfinite(fc) & (fc < cv)
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

    # ---- combined CC+NC panels: one common flash-chi2 cut, no stream split -
    if args.combined_chi2_cut is not None:
        cv = args.combined_chi2_cut
        bins = np.linspace(0, 500, 51)
        ctr = 0.5 * (bins[:-1] + bins[1:])
        for eq2, vabb, vlab in ((False, "ge2", ">=2"), (True, "eq2", "exactly 2")):
            def cmask(t):
                sel = (t["sel_eq2"] if eq2 else t["sel_ge2"]).astype(bool)
                fc = t["flash_chi2"]
                return sel & np.isfinite(fc) & (fc < cv)
            mm = cmask(mc) & np.isfinite(mc["m_vtx2start"])
            dm = cmask(da) & np.isfinite(da["m_vtx2start"])
            em = cmask(ex) & np.isfinite(ex["m_vtx2start"])
            stack = [np.clip(mc["m_vtx2start"][mm & (mc["cat"] == c)], 0, 499)
                     for c in range(6)]
            ws = [mc["w"][mm & (mc["cat"] == c)] for c in range(6)]
            stack.append(np.clip(ex["m_vtx2start"][em], 0, 499))
            ws.append(np.full(int(em.sum()), args.ext_scale))
            labels = [f"{CATS[c]} ({ws[c].sum():.0f})" for c in range(6)]
            labels.append(f"EXT cosmic ({ws[6].sum():.0f})")
            fig, ax = plt.subplots(figsize=(6.8, 4.6))
            ax.hist(stack, bins=bins, weights=ws, stacked=True,
                    color=CAT_COLORS + [EXT_COLOR], label=labels)
            dh, _ = np.histogram(np.clip(da["m_vtx2start"][dm], 0, 499), bins)
            ax.errorbar(ctr, dh, yerr=np.sqrt(np.clip(dh, 1, None)), fmt="ko",
                        ms=3.5, lw=1, label=f"beam data ({int(dh.sum())})")
            ax.axvline(PI0_MASS, color="0.4", ls=":", lw=1.1)
            pred = sum(w.sum() for w in ws)
            ax.set(xlabel=r"$m_{\gamma\gamma}$ [MeV]",
                   ylabel=f"events / {args.pot:.1e} POT",
                   title=f"CC+NC combined: mgg_ext ({vlab} photons), "
                         f"chi2<{cv:g}\ndata {int(dh.sum())} vs pred {pred:.0f}")
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(f"{args.plots}/mgg_ext_all_{vabb}.png", dpi=110)
            plt.close(fig)
            pk = lambda v: (v >= 100) & (v < 170)
            sg = sum(w[pk(x)].sum() for x, w in zip(stack[:2], ws[:2]))
            ot = sum(w[pk(x)].sum() for x, w in zip(stack[2:], ws[2:]))
            dp = int(pk(np.clip(da["m_vtx2start"][dm], 0, 499)).sum())
            print(f"== combined CC+NC ({vlab}) chi2<{cv:g} near-peak: "
                  f"signal {sg:.1f} | other {ot:.1f} | purity {sg/(sg+ot):.3f} "
                  f"| data {dp} / pred {sg+ot:.1f} = {dp/(sg+ot):.2f}")

    # ---- purity vs reco observable: true pi0 signal / (all MC + EXT cosmic) -
    # with (solid) and without (dashed) the flash-chi2 cut, showing its gain.
    if "flash_chi2" in mc.files and "flash_chi2" in ex.files:
        def purity_plot(obs_key, bins, xlabel, fname, vline=None,
                        stream_matched=False):
            """purity vs a reco observable.

            stream_matched=False: numerator = ANY true-pi0 nu interaction
                (cat 0 or 1) -- "did we select a pi0 event at all".
            stream_matched=True : numerator = the true-pi0 interaction OF THIS
                STREAM'S TYPE only (reco-CC -> true CC pi0 = cat 0; reco-NC ->
                true NC pi0 = cat 1). This is the stricter question -- how well
                each sample ISOLATES its own interaction type -- so a true NC
                pi0 landing in the reco-CC sample now counts against CC purity
                instead of for it.
            """
            ctr = 0.5 * (bins[:-1] + bins[1:])
            for selk, vabb, vlab in (("sel_ge2", "ge2", ">=2 gamma"),
                                     ("sel_eq2", "eq2", "exactly 2 gamma")):
                for want_cc, sabb, slab in ((True, "recoCC", "reco-CC"),
                                            (False, "recoNC", "reco-NC")):
                    cutv = cut_for(want_cc)
                    if cutv is None:
                        cutv = 10000.0
                    base_mc = (mc[selk] & (mc["reco_cc"] == want_cc)
                               & np.isfinite(mc[obs_key]))
                    base_ex = (ex[selk] & (ex["reco_cc"] == want_cc)
                               & np.isfinite(ex[obs_key]))
                    fig, ax = plt.subplots(figsize=(6.6, 4.5))
                    for use_cut, sty, al, lab in (
                            (False, "s--", 0.45, "no cut"),
                            (True, "o-", 1.0, f"flash chi2<{cutv:.0f}")):
                        mm = base_mc.copy(); em = base_ex.copy()
                        if use_cut:
                            mm = mm & np.isfinite(mc["flash_chi2"]) & (
                                mc["flash_chi2"] < cutv)
                            em = em & np.isfinite(ex["flash_chi2"]) & (
                                ex["flash_chi2"] < cutv)
                        pur, err = [], []
                        for lo, hi in zip(bins[:-1], bins[1:]):
                            mb = mm & (mc[obs_key] >= lo) & (mc[obs_key] < hi)
                            eb = em & (ex[obs_key] >= lo) & (ex[obs_key] < hi)
                            if stream_matched:
                                # cat 0 = true signal CC, cat 1 = true signal NC
                                sig_cat = mc["cat"] == (0 if want_cc else 1)
                            else:
                                sig_cat = mc["cat"] <= 1
                            sig = mc["w"][mb & sig_cat].sum()
                            tot = (mc["w"][mb].sum()
                                   + args.ext_scale * int(eb.sum()))
                            p = sig / tot if tot > 0 else np.nan
                            n = int(mb.sum()) + int(eb.sum())
                            pur.append(p)
                            err.append(np.sqrt(p * (1 - p) / n)
                                       if n and np.isfinite(p) else 0)
                        ax.errorbar(ctr, pur, yerr=err, fmt=sty, ms=4,
                                    color="#1f77b4", alpha=al, label=lab)
                    if vline is not None:
                        ax.axvline(vline, color="0.4", ls=":", lw=1.1)
                    numlab = (f"true {'CC' if want_cc else 'NC'} pi0 signal"
                              if stream_matched else "true pi0 signal (CC+NC)")
                    ax.set(xlabel=xlabel, ylabel="purity", ylim=(0, 1.05),
                           title=f"{slab}: pi0 purity vs {xlabel} ({vlab})\n"
                                 f"{numlab} / (all MC + EXT cosmic)")
                    ax.legend(fontsize=8)
                    ax.grid(alpha=0.3)
                    fig.tight_layout()
                    fig.savefig(f"{args.plots}/{fname}_{sabb}_{vabb}.png",
                                dpi=110)
                    plt.close(fig)

        purity_plot("m_vtx2start", np.linspace(0, 500, 26),
                    r"reco $m_{\gamma\gamma}$ [MeV]", "purity_mgg",
                    vline=PI0_MASS)
        purity_plot("p_reco", np.linspace(0, 1200, 25),
                    r"reco $p_{\pi^0}$ [MeV/c]", "purity_ppi0")
        # stream-matched: how well does each sample isolate its OWN type?
        purity_plot("m_vtx2start", np.linspace(0, 500, 26),
                    r"reco $m_{\gamma\gamma}$ [MeV]", "puritySameStream_mgg",
                    vline=PI0_MASS, stream_matched=True)
        purity_plot("p_reco", np.linspace(0, 1200, 25),
                    r"reco $p_{\pi^0}$ [MeV/c]", "puritySameStream_ppi0",
                    stream_matched=True)

    # ---- NC charged-pion veto: split reco-NC into 0-charged-pi and the rest --
    # Goal: enrich true NC-pi0 (+Np, 0 charged-pi) by vetoing reco charged pions,
    # which removes CC feed-down (a CC muon mis-called a pi, or a real charged
    # pion). Applied on top of the per-stream flash-chi2 cut, NC-only.
    have_cpi = all("n_cpi_reco" in t.files for t in (mc, ex, da))
    if have_cpi:
        nc_cut = cut_for(False)

        def nc_sub(t, selk, sub, obs_key=None):
            m = t[selk] & (~t["reco_cc"]) & np.isfinite(t["flash_chi2"])
            if nc_cut is not None:
                m = m & (t["flash_chi2"] < nc_cut)
            if sub == "0cpi":
                m = m & (t["n_cpi_reco"] == 0)
            elif sub == "hascpi":
                m = m & (t["n_cpi_reco"] > 0)
            if obs_key is not None:
                m = m & np.isfinite(t[obs_key])
            return m

        SUBS = [("0cpi", "reco-NC, 0 charged-$\\pi$", "recoNC0cpi"),
                ("hascpi", r"reco-NC, $\geq$1 charged-$\pi$", "recoNChascpi")]

        def nc_panel(obs_key, bins, xlabel, fbase, clip_hi):
            for sub, slab, sabb in SUBS:
                for selk, vabb, vlab in (("sel_ge2", "ge2", ">=2"),
                                         ("sel_eq2", "eq2", "exactly 2")):
                    mm = nc_sub(mc, selk, sub, obs_key)
                    dm = nc_sub(da, selk, sub, obs_key)
                    em = nc_sub(ex, selk, sub, obs_key)
                    ctr = 0.5 * (bins[:-1] + bins[1:])
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
                    dh, _ = np.histogram(np.clip(da[obs_key][dm], bins[0],
                                                 clip_hi), bins=bins)
                    ax.errorbar(ctr, dh, yerr=np.sqrt(np.clip(dh, 1, None)),
                                fmt="ko", ms=3.5, lw=1, capsize=0,
                                label=f"beam data ({int(dh.sum())})")
                    if obs_key.startswith("m_"):
                        ax.axvline(PI0_MASS, color="0.4", ls=":", lw=1.1)
                    ax.set(xlabel=xlabel, ylabel=f"events / {args.pot:.1e} POT",
                           title=f"{slab} (flash $\\chi^2$<{nc_cut:.0f}): "
                                 f"{fbase} ({vlab} photons)\n"
                                 f"data {int(dh.sum())} vs pred "
                                 f"{pred_tot:.0f} (MC+EXT)")
                    ax.legend(fontsize=7)
                    ax.grid(alpha=0.3)
                    fig.tight_layout()
                    fig.savefig(f"{args.plots}/{fbase}_{sabb}_{vabb}.png",
                                dpi=110)
                    plt.close(fig)

        nc_panel("m_vtx2start", np.linspace(0, 500, 51),
                 r"$m_{\gamma\gamma}$ [MeV]", "mgg_ext", 499)
        nc_panel("p_reco", np.linspace(0, 1200, 49),
                 r"reco $p_{\pi^0}$ [MeV/c]", "ppi0_ext", 1199)

        # true-NC-pi0 purity vs reco observable: all NC vs 0cpi vs the rest
        def nc_purity(obs_key, bins, xlabel, fname, vline=None):
            ctr = 0.5 * (bins[:-1] + bins[1:])
            for selk, vabb, vlab in (("sel_ge2", "ge2", ">=2 gamma"),
                                     ("sel_eq2", "eq2", "exactly 2 gamma")):
                fig, ax = plt.subplots(figsize=(6.8, 4.6))
                for sub, slab, col in (("all", "all reco-NC", "0.5"),
                                       ("0cpi", "0 charged-$\\pi$", "tab:green"),
                                       ("hascpi", r"$\geq$1 charged-$\pi$",
                                        "tab:red")):
                    mm = nc_sub(mc, selk, sub, obs_key)
                    em = nc_sub(ex, selk, sub, obs_key)
                    pur, err = [], []
                    for lo, hi in zip(bins[:-1], bins[1:]):
                        mb = mm & (mc[obs_key] >= lo) & (mc[obs_key] < hi)
                        eb = em & (ex[obs_key] >= lo) & (ex[obs_key] < hi)
                        sig = mc["w"][mb & (mc["cat"] == 1)].sum()
                        tot = mc["w"][mb].sum() + args.ext_scale * int(eb.sum())
                        p = sig / tot if tot > 0 else np.nan
                        nn = int(mb.sum()) + int(eb.sum())
                        pur.append(p)
                        err.append(np.sqrt(p * (1 - p) / nn)
                                   if nn and np.isfinite(p) else 0)
                    ax.errorbar(ctr, pur, yerr=err, fmt="o-", ms=4, color=col,
                                label=slab)
                if vline is not None:
                    ax.axvline(vline, color="0.4", ls=":", lw=1.1)
                ax.set(xlabel=xlabel, ylabel="purity", ylim=(0, 1.05),
                       title=f"reco-NC: TRUE NC-pi0 purity vs {xlabel} ({vlab})\n"
                             "charged-pi veto -- true NC pi0 / (all MC + EXT)")
                ax.legend(fontsize=8)
                ax.grid(alpha=0.3)
                fig.tight_layout()
                fig.savefig(f"{args.plots}/{fname}_{vabb}.png", dpi=110)
                plt.close(fig)

        nc_purity("m_vtx2start", np.linspace(0, 500, 26),
                  r"reco $m_{\gamma\gamma}$ [MeV]", "purity_nccpi_mgg",
                  vline=PI0_MASS)
        nc_purity("p_reco", np.linspace(0, 1200, 25),
                  r"reco $p_{\pi^0}$ [MeV/c]", "purity_nccpi_ppi0")

        # composition: reco-NC near-peak (100-170), all vs 0cpi vs the rest
        print("== reco-NC near-peak (100-170 MeV, eq2, flash chi2<%.0f): "
              "charged-pi veto composition ==" % nc_cut)
        hdr = "%-16s %7s %7s %7s %7s %7s %7s %7s %8s"
        print(hdr % ("sample", "sigCC", "sigNC", "outFV", "no-pi0", "multi",
                     "undet", "EXT", "NCpi0pur"))
        for sub, slab in (("all", "all reco-NC"), ("0cpi", "0 charged-pi"),
                          ("hascpi", ">=1 charged-pi")):
            mm = nc_sub(mc, "sel_eq2", sub, "m_vtx2start")
            em = nc_sub(ex, "sel_eq2", sub, "m_vtx2start")
            pk = lambda t, m: (m & (t["m_vtx2start"] >= 100)
                               & (t["m_vtx2start"] < 170))
            mpk = pk(mc, mm)
            cw = [mc["w"][mpk & (mc["cat"] == c)].sum() for c in range(6)]
            extw = args.ext_scale * int(pk(ex, em).sum())
            tot = sum(cw) + extw
            print(hdr % (slab, "%.1f" % cw[0], "%.1f" % cw[1], "%.1f" % cw[2],
                         "%.1f" % cw[3], "%.1f" % cw[4], "%.1f" % cw[5],
                         "%.1f" % extw, "%.3f" % (cw[1] / max(tot, 1e-9))))

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
