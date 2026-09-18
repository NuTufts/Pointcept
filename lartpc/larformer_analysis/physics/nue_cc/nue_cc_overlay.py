"""nue CC selection: data vs (nue-signal + bnb-nu bkg + EXT cosmic) prediction.

Stacks the per-sample tables from nue_cc_analysis.py:
  - nue overlay    -> the nue CC signal expectation (intrinsic-nue sample; the
                      ONLY source of nue CC events)
  - bnb-nu overlay -> non-nue-CC background. True nue CC is VETOED here
                      (`is_nuecc`) so the intrinsic-nue sample isn't double
                      counted; the remainder is split into numu CC / NC.
  - EXT (beam-off) -> cosmic background at the spill weight for the cew6 200k
                      subset (nue_cc_common.ext_scale; NOT the full-sample
                      0.17683 the July analysis used).
MC weights are already POT-scaled to --pot (4.4e19) in the tables. bnb5e19 beam
data is overlaid with the SAME selection (unit weight).

This script owns the HEADLINE plots (the observable, the flash-chi2 distribution
the cut is chosen from, the signal-efficiency turn-ons, and the truth breakdown
of the surviving background). Per-variable cut-definition plots live in
`nue_cc_scan.py`, which also ranks the discriminants.

    PYTHONPATH=./ python3 nue_cc_overlay.py \
        --nue-npz nue.npz --bnb-npz bnb.npz --ext-npz ext.npz --data-npz data.npz \
        --plots plots/ [--flashchi2-cut 3.0] [--elconf-cut 9]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nue_cc_common as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--nue-npz", required=True)
    ap.add_argument("--bnb-npz", required=True)
    ap.add_argument("--ext-npz", required=True)
    ap.add_argument("--data-npz", required=True)
    ap.add_argument("--plots", required=True)
    ap.add_argument("--pot", type=float, default=C.DEFAULT_POT)
    C.add_cut_args(ap)
    args = ap.parse_args()
    os.makedirs(args.plots, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cuts = C.cuts_from_args(args)
    label = C.cut_label(cuts)
    sbdt = C.uses_shower_bdt(cuts)
    tabs = C.load_tables(args.nue_npz, args.bnb_npz, args.ext_npz, args.data_npz,
                         ext_scale_override=args.ext_scale,
                         ext_rescale=args.ext_rescale)
    label = f"[{C.active_set()}] {label}"
    dat = tabs["data"]
    nue, bnb, ext = tabs["nue"], tabs["bnb"], tabs["ext"]
    ext_w = C.ext_scale(shower_bdt_used=sbdt)
    print(f">>> selection: {label}")
    print(f">>> EXT per-event weight {ext_w:.4f}"
          f"{' (rows>=100k hygiene half)' if sbdt and C.SAMPLE_SETS[C.active_set()]['hygiene'] else ''}")

    def stacked(obs, wts, bins, xlabel, title, fname, data_obs=None, logy=False):
        lo, hi = bins[0], bins[-1]
        clip = lambda z: np.clip(z, lo, hi - (hi - lo) * 1e-6)  # noqa: E731
        fig, ax = plt.subplots(figsize=(7.4, 4.8))
        ax.hist([clip(o) for o in obs], bins=bins, weights=wts, stacked=True,
                color=C.COLORS,
                label=[f"{c} ({w.sum():.1f})"
                       for c, w in zip(C.COMPONENTS, wts)])
        pred = float(sum(w.sum() for w in wts))
        if data_obs is not None:
            dh, _ = np.histogram(clip(data_obs), bins=bins)
            ctr = 0.5 * (bins[:-1] + bins[1:])
            ax.errorbar(ctr, dh, yerr=np.sqrt(np.clip(dh, 1, None)), fmt="ko",
                        ms=4, lw=1, capsize=0,
                        label=f"bnb5e19 data ({int(dh.sum())})")
            title += f"\ndata {int(dh.sum())} vs pred {pred:.1f}"
        if logy:
            ax.set_yscale("log"); ax.set_ylim(0.03, None)
        ax.set(xlabel=xlabel, ylabel=f"events / {args.pot:.1e} POT", title=title)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.tight_layout()
        p = os.path.join(args.plots, fname)
        fig.savefig(p, dpi=120); plt.close(fig); print(">>> wrote", p)
        return pred

    # ---- flash-chi2 distribution (selection WITHOUT any cut applied) -------
    # This is the plot the flash cut is CHOSEN from, so it must not have the
    # flash cut (or the PID cuts) already applied to it.
    masks0, wts0 = C.components(tabs, cuts, apply_cuts=False)
    d = {k: C.derive(v) for k, v in tabs.items()}
    src = [("nue", masks0[0]), ("bnb", masks0[1]), ("bnb", masks0[2]),
           ("ext", masks0[3])]
    obs_f, wts_f = [], []
    for (name, m), w in zip(src, wts0):
        v = d[name]["logchi2"][m]
        ok = np.isfinite(v)
        obs_f.append(v[ok]); wts_f.append(w[ok])
    md0 = C.data_mask(dat, cuts, apply_cuts=False)
    dv = d["data"]["logchi2"][md0]
    dv = dv[np.isfinite(dv)]
    fbins = np.linspace(0, 6, 49)
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.hist(obs_f, bins=fbins, weights=wts_f, stacked=True, color=C.COLORS,
            label=[f"{c} ({w.sum():.1f})" for c, w in zip(C.COMPONENTS, wts_f)])
    dh, _ = np.histogram(dv, bins=fbins)
    ctr = 0.5 * (fbins[:-1] + fbins[1:])
    ax.errorbar(ctr, dh, yerr=np.sqrt(np.clip(dh, 1, None)), fmt="ko", ms=4,
                lw=1, capsize=0, label=f"bnb5e19 data ({int(dh.sum())})")
    if "flashchi2" in cuts:
        ax.axvline(cuts["flashchi2"], color="k", ls="--", lw=1.2,
                   label=f"cut log10<{cuts['flashchi2']:g}")
    ax.set(xlabel=r"$\log_{10}(\mathrm{flash}\ \chi^2)$",
           ylabel=f"events / {args.pot:.1e} POT",
           title="nu-vtx flash-chi2 (reco nu-vtx in FV + >=1 primary e shower)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); fig.tight_layout()
    p = os.path.join(args.plots, "flashchi2.png")
    fig.savefig(p, dpi=120); plt.close(fig); print(">>> wrote", p)

    # ---- the observable, WITH the full selection --------------------------
    masks, wts = C.components(tabs, cuts)
    md = C.data_mask(dat, cuts)
    ebins = np.arange(0, 2000 + 100, 100)          # 100 MeV bins
    obs = [nue["reco_ele_E"][masks[0]], bnb["reco_ele_E"][masks[1]],
           bnb["reco_ele_E"][masks[2]], ext["reco_ele_E"][masks[3]]]
    stacked(obs, wts, ebins, "reco electron shower energy [MeV]",
            f"nue CC selection: reco e-shower energy\n({label})",
            "reco_ele_energy.png", data_obs=dat["reco_ele_E"][md])

    # ---- signal efficiency turn-ons (MC truth) ----------------------------
    sig = np.asarray(nue["is_nuecc_fv"]).astype(bool)
    wn = np.asarray(nue["w"], float)

    def eff_vs(true_key, xlabel, bins, fname):
        if true_key not in nue:
            return
        tv = np.asarray(nue[true_key], float)
        sel_sig = masks[0] & sig
        ctr = 0.5 * (bins[:-1] + bins[1:])
        eff, err = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            inb = sig & np.isfinite(tv) & (tv >= lo) & (tv < hi)
            den, nden = wn[inb].sum(), int(inb.sum())
            e = wn[inb & sel_sig].sum() / den if den > 0 else np.nan
            eff.append(e)
            err.append(np.sqrt(e * (1 - e) / nden)
                       if nden > 0 and np.isfinite(e) else np.nan)
        fig, ax = plt.subplots(figsize=(7, 4.6))
        ax.errorbar(ctr, eff, yerr=err, fmt="o-", color="#d62728", lw=1.5, ms=4)
        ax.set(xlabel=xlabel, ylabel="selection efficiency", ylim=(0, 1.02),
               title=f"nue CC efficiency vs {xlabel}\n({label})")
        ax.grid(alpha=0.3); fig.tight_layout()
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

    # ---- what the surviving background actually IS ------------------------
    # Replaces the old bg_truth_pid.png (which keyed on the raw PDG of the
    # truth match and so could not tell a nu primary electron from a Michel,
    # nor a cosmic from an unmatched shower).
    cat_nue, cat_bnb = C.truth_category(nue), C.truth_category(bnb)
    if (cat_bnb >= 0).any() or (cat_nue >= 0).any():
        wb = np.asarray(bnb["w"], float)
        mb_bkg = masks[1] | masks[2]
        names, sv, bv = [], [], []
        for ci, name in enumerate(C.TRUTH_CATS):
            s = float(wn[masks[0] & (cat_nue == ci)].sum())
            b = float(wb[mb_bkg & (cat_bnb == ci)].sum())
            if s > 0 or b > 0:
                names.append(name); sv.append(s); bv.append(b)
        ext_tot = float(wts[3].sum())
        fig, ax = plt.subplots(figsize=(9, 4.8))
        x = np.arange(len(names))
        ax.bar(x - 0.2, sv, 0.4, color="#d62728",
               label=f"nue sample ({sum(sv):.1f})")
        ax.bar(x + 0.2, bv, 0.4, color="#1f77b4",
               label=f"bnb-nu background ({sum(bv):.1f})")
        for i, (a_, b_) in enumerate(zip(sv, bv)):
            if a_ > 0:
                ax.text(i - 0.2, a_, f"{a_:.2f}", ha="center", va="bottom",
                        fontsize=6.5)
            if b_ > 0:
                ax.text(i + 0.2, b_, f"{b_:.2f}", ha="center", va="bottom",
                        fontsize=6.5)
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=25, ha="right")
        ax.set_yscale("log")
        ax.set(ylabel=f"events / {args.pot:.1e} POT",
               title=f"What the reco'd electron candidate really is\n({label})"
                     f"   [EXT cosmic, no truth: {ext_tot:.1f}]")
        ax.grid(alpha=0.3, axis="y", which="both"); ax.legend(fontsize=8)
        fig.tight_layout()
        p = os.path.join(args.plots, "bg_truth_category.png")
        fig.savefig(p, dpi=120); plt.close(fig); print(">>> wrote", p)

    # ---- summary ----------------------------------------------------------
    C.summarize(tabs, cuts, dat=dat, pot=args.pot)


if __name__ == "__main__":
    main()
