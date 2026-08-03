"""nue CC selection: data vs (nue-signal + bnb-nu bkg + EXT cosmic) prediction.

Stacks the per-sample tables from nue_cc_analysis.py:
  - nue overlay  -> the nue CC signal expectation (intrinsic-nue sample; the
                    ONLY source of nue CC events)
  - bnb-nu overlay -> non-nue-CC background. True nue CC is VETOED here
                    (`is_nuecc`) so the intrinsic-nue sample isn't double counted;
                    the remainder is split into numu CC / NC.
  - EXT (beam-off) -> cosmic background, scaled by the beam/EXT spill ratio
                    (0.17682554549 for the full EXT sample; see memory
                    extbnb-cosmic-normalization).
MC weights are already POT-scaled to --pot (4.4e19) in the tables. bnb5e19 beam
data is overlaid with the SAME selection (unit weight).

    PYTHONPATH=./ python3 nue_cc_overlay.py \
        --nue-npz nue.npz --bnb-npz bnb.npz --ext-npz ext.npz --data-npz data.npz \
        --plots plots/ [--flashchi2-cut 3.0]
"""
import argparse
import os

import numpy as np

FULL_EXT_SCALE = 0.17682554549    # beam/EXT full-sample spill ratio
# stack order (bottom->top) + colors
COMPONENTS = ["nu_e CC (signal)", "nu_mu CC", "NC", "EXT cosmic"]
COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#e5e5e5"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--nue-npz", required=True)
    ap.add_argument("--bnb-npz", required=True)
    ap.add_argument("--ext-npz", required=True)
    ap.add_argument("--data-npz", required=True)
    ap.add_argument("--plots", required=True)
    ap.add_argument("--pot", type=float, default=4.4e19)
    ap.add_argument("--ext-scale", type=float, default=FULL_EXT_SCALE,
                    help="per-EXT-event weight = 0.17682554549 / fraction")
    ap.add_argument("--flashchi2-cut", type=float, default=3.0,
                    help="keep events with log10(flash_chi2) < this (provisional,"
                         " README suggests ~3.0). Use -1 to disable.")
    # LArPID electron cuts (leading e-shower log-softmax scores). None = off.
    ap.add_argument("--elconf-cut", type=float, default=None,
                    help="keep log p_e - 0.5(log p_pi + log p_gamma) > this")
    ap.add_argument("--egamma-cut", type=float, default=None,
                    help="LArPID e/gamma: keep log p_e - log p_gamma > this")
    ap.add_argument("--primariness-cut", type=float, default=None,
                    help="keep log p_prim - max(log p_fromN, log p_fromC) > this")
    ap.add_argument("--mu-cut", type=float, default=None,
                    help="keep log p_mu < this (e-shower muon veto)")
    ap.add_argument("--vtxmu-cut", type=float, default=None,
                    help="keep vtx_mu_score < this OR no other particle at the "
                         "e-vertex (veto a muon-like track sharing the vertex)")
    # LArFormer (segmentation-model) analogues -- same formulas on log(prob).
    ap.add_argument("--elconf-lf-cut", type=float, default=None,
                    help="LArFormer e-confidence: keep lf_el-0.5(lf_pi+lf_ph) > this")
    ap.add_argument("--egamma-lf-cut", type=float, default=None,
                    help="LArFormer e/gamma: keep lf_el - lf_ph > this (the key "
                         "LArFormer discriminant; mu/pi probs ~0)")
    ap.add_argument("--mu-lf-cut", type=float, default=None,
                    help="keep LArFormer e-shower log p_mu < this")
    ap.add_argument("--vtxmu-lf-cut", type=float, default=None,
                    help="keep LArFormer vertex muon score < this (or no other "
                         "particle at the e-vertex)")
    ap.add_argument("--nphoton-max", type=int, default=None,
                    help="keep events with <= this many reco photons "
                         "(LArFormerPID==22, >20 MeV) at the nu vertex "
                         "(pi0 / mis-id-gamma veto; e.g. 0)")
    ap.add_argument("--no-var-plots", action="store_true",
                    help="skip the per-cut-variable stacked plots (var_*.png)")
    args = ap.parse_args()
    os.makedirs(args.plots, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nue = dict(np.load(args.nue_npz)); bnb = dict(np.load(args.bnb_npz))
    ext = dict(np.load(args.ext_npz)); dat = dict(np.load(args.data_npz))
    cutv = None if args.flashchi2_cut is not None and args.flashchi2_cut < 0 \
        else args.flashchi2_cut

    def larpid_ok(tab):
        """leading-e-shower LArPID cuts (only where a score exists)."""
        m = np.ones(len(tab["sel"]), bool)
        if args.elconf_cut is not None:
            ec = tab["el_score"] - 0.5 * (tab["pi_score"] + tab["ph_score"])
            m &= np.isfinite(ec) & (ec > args.elconf_cut)
        if args.egamma_cut is not None:
            eg = tab["el_score"] - tab["ph_score"]
            m &= np.isfinite(eg) & (eg > args.egamma_cut)
        if args.primariness_cut is not None:
            pr = tab["prim_score"] - np.maximum(tab["fromneut_score"],
                                                tab["fromchg_score"])
            m &= np.isfinite(pr) & (pr > args.primariness_cut)
        if args.mu_cut is not None:
            m &= np.isfinite(tab["mu_score"]) & (tab["mu_score"] < args.mu_cut)
        if args.vtxmu_cut is not None:
            vm = tab["vtx_mu_score"]
            m &= ~(np.isfinite(vm) & (vm >= args.vtxmu_cut))   # NaN=no muon=keep
        if args.elconf_lf_cut is not None:
            ec = tab["lf_el_score"] - 0.5 * (tab["lf_pi_score"] + tab["lf_ph_score"])
            m &= np.isfinite(ec) & (ec > args.elconf_lf_cut)
        if args.egamma_lf_cut is not None:
            eg = tab["lf_el_score"] - tab["lf_ph_score"]
            m &= np.isfinite(eg) & (eg > args.egamma_lf_cut)
        if args.mu_lf_cut is not None:
            m &= np.isfinite(tab["lf_mu_score"]) & (tab["lf_mu_score"] < args.mu_lf_cut)
        if args.vtxmu_lf_cut is not None:
            vm = tab["vtx_lf_mu_score"]
            m &= ~(np.isfinite(vm) & (vm >= args.vtxmu_lf_cut))
        if args.nphoton_max is not None:
            nph = np.where(tab["n_photons"] < 0, 0, tab["n_photons"])
            m &= nph <= args.nphoton_max
        return m

    def base(tab, use_flash):
        """selected + finite observable (+ optional flash-chi2 + LArPID cuts)."""
        m = tab["sel"].astype(bool) & np.isfinite(tab["reco_ele_E"])
        if use_flash and cutv is not None:
            fc = tab["flash_chi2"]
            m = m & np.isfinite(fc) & (fc > 0) & (np.log10(fc) < cutv)
        if use_flash:
            m = m & larpid_ok(tab)
        return m

    def components(use_flash):
        """(obs-list, weight-list) for the 4 stacked components."""
        mn = base(nue, use_flash)
        mb = base(bnb, use_flash) & ~bnb["is_nuecc"]        # VETO bnb nue CC
        numucc = mb & (np.abs(bnb["nu_pdg"]) == 14) & (bnb["ccnc"] == 0)
        nc = mb & (bnb["ccnc"] == 1)
        me = base(ext, use_flash)
        obs = [nue["reco_ele_E"][mn], bnb["reco_ele_E"][numucc],
               bnb["reco_ele_E"][nc], ext["reco_ele_E"][me]]
        wts = [nue["w"][mn], bnb["w"][numucc], bnb["w"][nc],
               np.full(int(me.sum()), args.ext_scale)]
        return obs, wts

    def stacked_plot(obs_of, bins, xlabel, title, fname, clip=True,
                     data_obs=None):
        lo, hi = bins[0], bins[-1]
        obs, wts = obs_of
        if clip:
            obs = [np.clip(o, lo, hi - 1e-6) for o in obs]
        labels = [f"{c} ({w.sum():.1f})" for c, w in zip(COMPONENTS, wts)]
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        ax.hist(obs, bins=bins, weights=wts, stacked=True, color=COLORS,
                label=labels)
        pred = sum(w.sum() for w in wts)
        if data_obs is not None:
            dd = np.clip(data_obs, lo, hi - 1e-6) if clip else data_obs
            dh, _ = np.histogram(dd, bins=bins)
            ctr = 0.5 * (bins[:-1] + bins[1:])
            ax.errorbar(ctr, dh, yerr=np.sqrt(np.clip(dh, 1, None)),
                        fmt="ko", ms=4, lw=1, capsize=0,
                        label=f"bnb5e19 data ({int(dh.sum())})")
            title += f"\ndata {int(dh.sum())} vs pred {pred:.1f}"
        ax.set(xlabel=xlabel, ylabel=f"events / {args.pot:.1e} POT", title=title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        p = os.path.join(args.plots, fname)
        fig.savefig(p, dpi=120)
        plt.close(fig)
        print(">>> wrote", p)
        return pred

    def var_plot(key, label, get, cutval=None, logx=False, mc_only=False,
                 bins=None):
        """Stacked prediction + data for a candidate cut variable, at the
        CURRENT selection (all set cut flags applied). log-y so the small nueCC
        signal is visible; draws the current cut line if set."""
        mnu = base(nue, True); mbn = base(bnb, True) & ~bnb["is_nuecc"]
        numu = mbn & (np.abs(bnb["nu_pdg"]) == 14) & (bnb["ccnc"] == 0)
        ncc = mbn & (bnb["ccnc"] == 1)
        mex = base(ext, True); mda = base(dat, True)

        def pick(tab, m, w=None):
            v = np.asarray(get(tab), float)[m]; ok = np.isfinite(v)
            return v[ok], (w[m][ok] if w is not None else None)

        sv, sw = pick(nue, mnu, nue["w"])
        uv, uw = pick(bnb, numu, bnb["w"])
        cv, cw = pick(bnb, ncc, bnb["w"])
        ev, _ = pick(ext, mex); ew = np.full(len(ev), args.ext_scale)
        dv, _ = pick(dat, mda)
        obs = [sv, uv, cv, ev]; wts = [sw, uw, cw, ew]
        colors = list(COLORS); comps = list(COMPONENTS); data_obs = dv
        if mc_only:                 # EXT/data carry no truth for this variable
            obs, wts = obs[:3], wts[:3]
            colors, comps, data_obs = colors[:3], comps[:3], None
        allv = np.concatenate([o for o in obs if len(o)]
                              + ([data_obs] if data_obs is not None
                                 and len(data_obs) else []))
        if len(allv) == 0:
            print(f"  (skip var {key}: no events)"); return
        if bins is None:
            if logx:
                pos = allv[allv > 0]
                if not len(pos): return
                bins = np.logspace(np.log10(pos.min()), np.log10(pos.max()), 41)
            else:
                blo, bhi = np.percentile(allv, [0.5, 99.5])
                bins = np.linspace(blo, bhi if bhi > blo else blo + 1, 41)
        lo, hi = bins[0], bins[-1]
        clip = lambda a: np.clip(a, lo, hi - (hi - lo) * 1e-6)
        ctr = 0.5 * (bins[:-1] + bins[1:])
        fig, ax = plt.subplots(figsize=(7.4, 4.8))
        ax.hist([clip(o) for o in obs], bins=bins, weights=wts, stacked=True,
                color=colors,
                label=[f"{c} ({w.sum():.1f})" for c, w in zip(comps, wts)])
        pred = sum(w.sum() for w in wts)
        if data_obs is not None:
            dh, _ = np.histogram(clip(data_obs), bins=bins)
            ax.errorbar(ctr, dh, yerr=np.sqrt(np.clip(dh, 1, None)), fmt="ko",
                        ms=3.5, lw=1, capsize=0, label=f"data ({int(dh.sum())})")
        if cutval is not None:
            ax.axvline(cutval, color="k", ls="--", lw=1.2, label=f"cut {cutval:g}")
        if logx:
            ax.set_xscale("log")
        ax.set_yscale("log"); ax.set_ylim(0.03, None)
        ax.set(xlabel=label, ylabel=f"events / {args.pot:.1e} POT",
               title=f"{key} at current selection  (pred {pred:.1f})")
        ax.legend(fontsize=7.5); ax.grid(alpha=0.3, which="both")
        fig.tight_layout()
        p = os.path.join(args.plots, f"var_{key}.png")
        fig.savefig(p, dpi=115); plt.close(fig); print(">>> wrote", p)

    # ---- flash-chi2 distribution (selection WITHOUT the flash cut) ---------
    def logchi2(tab, m):
        fc = tab["flash_chi2"][m]
        fc = fc[np.isfinite(fc) & (fc > 0)]
        return np.log10(fc)
    mn = base(nue, False); mb = base(bnb, False) & ~bnb["is_nuecc"]
    numucc = mb & (np.abs(bnb["nu_pdg"]) == 14) & (bnb["ccnc"] == 0)
    nc = mb & (bnb["ccnc"] == 1); me = base(ext, False); md = base(dat, False)
    fbins = np.linspace(0, 6, 49)
    obs_f = ([logchi2(nue, mn), logchi2(bnb, numucc), logchi2(bnb, nc),
              logchi2(ext, me)],
             [nue["w"][mn][np.isfinite(nue["flash_chi2"][mn]) & (nue["flash_chi2"][mn] > 0)],
              bnb["w"][numucc][np.isfinite(bnb["flash_chi2"][numucc]) & (bnb["flash_chi2"][numucc] > 0)],
              bnb["w"][nc][np.isfinite(bnb["flash_chi2"][nc]) & (bnb["flash_chi2"][nc] > 0)],
              np.full(int((np.isfinite(ext["flash_chi2"][me]) & (ext["flash_chi2"][me] > 0)).sum()),
                      args.ext_scale)])
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.hist(obs_f[0], bins=fbins, weights=obs_f[1], stacked=True, color=COLORS,
            label=[f"{c}" for c in COMPONENTS])
    dlog = logchi2(dat, md)
    dh, _ = np.histogram(dlog, bins=fbins)
    ctr = 0.5 * (fbins[:-1] + fbins[1:])
    ax.errorbar(ctr, dh, yerr=np.sqrt(np.clip(dh, 1, None)), fmt="ko", ms=4,
                lw=1, capsize=0, label=f"bnb5e19 data ({int(dh.sum())})")
    if cutv is not None:
        ax.axvline(cutv, color="k", ls="--", lw=1.2,
                   label=f"cut log10<{cutv:g}")
    ax.set(xlabel=r"$\log_{10}(\mathrm{flash}\ \chi^2)$",
           ylabel=f"events / {args.pot:.1e} POT",
           title="nu-vtx flash-chi2 (reco nu-vtx in FV + >=1 primary e shower)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(args.plots, "flashchi2.png"), dpi=120)
    plt.close(fig)
    print(">>> wrote", os.path.join(args.plots, "flashchi2.png"))

    # ---- reco electron shower energy (WITH the flash + LArPID cuts) --------
    md_e = base(dat, True)
    ebins = np.arange(0, 2000 + 100, 100)   # 100 MeV bins
    cut_bits = [] if cutv is None else [f"log10(flashchi2)<{cutv:g}"]
    if args.elconf_cut is not None:
        cut_bits.append(f"elconf>{args.elconf_cut:g}")
    if args.egamma_cut is not None:
        cut_bits.append(f"egamma>{args.egamma_cut:g}")
    if args.egamma_lf_cut is not None:
        cut_bits.append(f"egammaLF>{args.egamma_lf_cut:g}")
    if args.primariness_cut is not None:
        cut_bits.append(f"prim>{args.primariness_cut:g}")
    if args.mu_cut is not None:
        cut_bits.append(f"mu<{args.mu_cut:g}")
    if args.vtxmu_cut is not None:
        cut_bits.append(f"vtxmu<{args.vtxmu_cut:g}")
    if args.elconf_lf_cut is not None:
        cut_bits.append(f"elconfLF>{args.elconf_lf_cut:g}")
    if args.mu_lf_cut is not None:
        cut_bits.append(f"muLF<{args.mu_lf_cut:g}")
    if args.vtxmu_lf_cut is not None:
        cut_bits.append(f"vtxmuLF<{args.vtxmu_lf_cut:g}")
    if args.nphoton_max is not None:
        cut_bits.append(f"nphoton<={args.nphoton_max}")
    cut_txt = " & ".join(cut_bits) if cut_bits else "no cut"
    pred = stacked_plot(
        components(True), ebins, "reco electron shower energy [MeV]",
        f"nue CC selection: reco e-shower energy\n({cut_txt})",
        "reco_ele_energy.png", data_obs=dat["reco_ele_E"][md_e])

    # ---- every candidate cut variable at the current selection ------------
    if not args.no_var_plots:
        el = lambda t: t["el_score"] - 0.5 * (t["pi_score"] + t["ph_score"])
        eg = lambda t: t["el_score"] - t["ph_score"]
        pr = lambda t: t["prim_score"] - np.maximum(t["fromneut_score"],
                                                    t["fromchg_score"])
        var_plot("elconf", "e-confidence  log p_e - 0.5(log p_pi+log p_gamma)",
                 el, args.elconf_cut)
        var_plot("egamma", "e/gamma  log p_e - log p_gamma", eg, args.egamma_cut)
        var_plot("primariness", "primariness  log p_prim - max(log p_fN,fC)",
                 pr, args.primariness_cut)
        var_plot("mu_eshower", "e-shower muon score  log p_mu",
                 lambda t: t["mu_score"], args.mu_cut)
        var_plot("vtx_mu_score", "vertex muon score (other particles)  log p_mu",
                 lambda t: t["vtx_mu_score"], args.vtxmu_cut)
        var_plot("vtx_dist_true", "reco-true vtx dist [cm] (MC only)",
                 lambda t: t["vtx_dist_true"], logx=True, mc_only=True)
        if "n_photons" in nue:
            var_plot("n_photons",
                     "# reco photons (>20 MeV) at the nu vertex  (pi0 tag)",
                     lambda t: np.where(t["n_photons"] < 0, 0, t["n_photons"]),
                     args.nphoton_max, bins=np.arange(-0.5, 6.5, 1.0))
        # --- LArFormer (segmentation-model) analogues -------------------
        if "lf_el_score" in nue:
            el_lf = lambda t: t["lf_el_score"] - 0.5 * (t["lf_pi_score"]
                                                        + t["lf_ph_score"])
            eg_lf = lambda t: t["lf_el_score"] - t["lf_ph_score"]
            var_plot("elconf_lf",
                     "LArFormer e-confidence  log p_e - 0.5(log p_pi+log p_gamma)",
                     el_lf, args.elconf_lf_cut)
            var_plot("egamma_lf", "LArFormer e/gamma  log p_e - log p_gamma", eg_lf, args.egamma_lf_cut)
            var_plot("mu_lf_eshower", "LArFormer e-shower muon score  log p_mu",
                     lambda t: t["lf_mu_score"], args.mu_lf_cut)
            var_plot("vtx_mu_score_lf",
                     "LArFormer vertex muon score (other particles)  log p_mu",
                     lambda t: t["vtx_lf_mu_score"], args.vtxmu_lf_cut)

        # ---- MC-truth validation: signal efficiency + bkg truth-PID -------
        def eff_vs(true_key, xlabel, bins, fname):
            if true_key not in nue:
                return
            sig = nue["is_nuecc_fv"].astype(bool)
            tv = nue[true_key]
            sel_sig = base(nue, True) & sig       # true signal passing full sel
            ctr = 0.5 * (bins[:-1] + bins[1:])
            eff, err = [], []
            for lo, hi in zip(bins[:-1], bins[1:]):
                inb = sig & np.isfinite(tv) & (tv >= lo) & (tv < hi)
                den = nue["w"][inb].sum(); nden = int(inb.sum())
                num = nue["w"][inb & sel_sig].sum()
                e = num / den if den > 0 else np.nan
                eff.append(e)
                err.append(np.sqrt(e * (1 - e) / nden)
                           if nden > 0 and np.isfinite(e) else np.nan)
            fig, ax = plt.subplots(figsize=(7, 4.6))
            ax.errorbar(ctr, eff, yerr=err, fmt="o-", color="#d62728",
                        lw=1.5, ms=4)
            ax.set(xlabel=xlabel, ylabel="selection efficiency", ylim=(0, 1.02),
                   title=f"nue CC efficiency vs {xlabel}\n({cut_txt})")
            ax.grid(alpha=0.3)
            fig.tight_layout()
            p = os.path.join(args.plots, fname)
            fig.savefig(p, dpi=120); plt.close(fig); print(">>> wrote", p)

        eff_vs("true_nu_e", "true neutrino energy [GeV]",
               np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0]),
               "eff_vs_true_nu_e.png")
        eff_vs("true_ele_ke", "true electron KE [MeV]",
               np.array([0, 100, 200, 300, 500, 800, 1200, 2000]),
               "eff_vs_true_ele_ke.png")
        eff_vs("true_ele_vise", "true electron visible energy [MeV]",
               np.array([0, 50, 100, 200, 350, 600, 1000, 1600]),
               "eff_vs_true_ele_vise.png")

        # (3) MC background: truth-matched particle of the reco'd electron
        if "shower_true_pid" in bnb:
            mb = base(bnb, True) & ~bnb["is_nuecc"]
            tp = np.abs(bnb["shower_true_pid"][mb]); wb = bnb["w"][mb]
            cats = [(11, "electron"), (22, "photon"), (13, "muon"),
                    (211, "pion"), (2212, "proton")]
            vals = [wb[tp == pdg].sum() for pdg, _ in cats]
            vals.append(wb[~np.isin(tp, [c[0] for c in cats])].sum())
            labels = [nm for _, nm in cats] + ["other/none"]
            fig, ax = plt.subplots(figsize=(7, 4.6))
            ax.bar(range(len(vals)), vals, color="#1f77b4")
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=20)
            ax.set(ylabel=f"bkg events / {args.pot:.1e} POT",
                   title=f"MC bkg: truth of the reco'd electron  (total "
                         f"{sum(vals):.1f})\n({cut_txt})")
            for i, v in enumerate(vals):
                ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
            ax.grid(alpha=0.3, axis="y")
            fig.tight_layout()
            p = os.path.join(args.plots, "bg_truth_pid.png")
            fig.savefig(p, dpi=120); plt.close(fig); print(">>> wrote", p)

    # ---- summary + true-signal efficiency ---------------------------------
    obs, wts = components(True)
    mn = base(nue, True)
    sig_true_sel = nue["w"][mn & nue["is_nuecc_fv"]].sum()   # selected signal
    sig_true_all = nue["w"][nue["is_nuecc_fv"]].sum()        # all true signal FV
    eff = sig_true_sel / sig_true_all if sig_true_all > 0 else np.nan
    print(f"\n== SELECTION SUMMARY ({cut_txt}) ==")
    for c, w in zip(COMPONENTS, wts):
        print(f"  {c:20s} {w.sum():8.2f}")
    print(f"  {'TOTAL pred':20s} {pred:8.2f}")
    print(f"  {'bnb5e19 data':20s} {int(md_e.sum()):8d}   "
          f"(data/pred {md_e.sum() / pred:.2f})" if pred > 0 else "")
    print(f"  purity (nueCC/pred)     = {wts[0].sum() / pred:.3f}"
          if pred > 0 else "")
    print(f"  efficiency (sel/true-FV)= {eff:.3f}  "
          f"[{sig_true_sel:.1f} / {sig_true_all:.1f} true nueCC-FV]")
    print(f"  (LANTERN benchmark: 55% eff / 90% purity)")


if __name__ == "__main__":
    main()
