#!/usr/bin/env python3
"""Data / prediction agreement of every (kinematics-free) BDT input used by the
MB BDT chain, at the S3 pre-selection (EXT-dominated) and after the base flash
cuts (S5, nu-enriched). One stacked plot per variable (linear + log-y) with a
data / pred ratio, and a summary table sorted by chi2/ndf. Same evaluation
samples as the chain (RESERVED overlay half, EXT second half, data).

    python3 mb_input_agreement.py --mc-ntuple B.root --ext-ntuple X2.root --data-ntuple D.root \
        --novtx --ext-scale 1.202 --plots plots_run1_mbchain_inputs
"""
import argparse, os, sys
import numpy as np
import awkward as ak
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import single_photon_selection as SPS                       # noqa: E402
import single_photon_mb_selection as MBS                    # noqa: E402
from single_photon_diagnostics import add_args              # noqa: E402
from mb_variable_study import features, FEATURES            # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(ap); ap.set_defaults(signal_def="mb")
    for k, v in (("veto2g_emin", 10.0), ("contain_dwall", 10.0), ("dfw_cut", SPS.DFW_CUT), ("mb_mu_kemin", 40.0),
                 ("mb_pi_kemin", 50.0), ("mb_p_kemax", 342.0), ("extra_emax", 10.0), ("cone_e2_min", 5.0),
                 ("cone_e2_max", 600.0), ("tight_dwall", 10.0), ("dfw_tight", 100.0), ("conv_dist_min", 0.0),
                 ("cand_bdt_min_mb", 0.7), ("mb_pe_ratio_max", 2.0), ("mb_logchi2_max", 2.25)):
        ap.add_argument("--" + k.replace("_", "-"), type=float, default=v)
    ap.add_argument("--pi0-cone", action="store_true")
    args = ap.parse_args(); SPS.configure(args); os.makedirs(args.plots, exist_ok=True)
    a, pot, M, _ = features(args.mc_ntuple, True, args)
    w0 = ak.to_numpy(a["xsecWeight"]).astype(np.float64); w = np.where(w0 > 0, w0, 0.0) * (args.pot / pot)
    T = SPS.truth_tables(a); cat = T["cat"]; del a
    _, _, MX, _ = features(args.ext_ntuple, False, args); wx = np.zeros(len(MX["E"])); wx[args.ext_row_min:] = args.ext_scale
    _, _, MD, _ = features(args.data_ntuple, False, args)
    SENTINEL = ("bdt_cosmic", "cand_novtx", "cand_score")     # exporter / selection fill -9 = no score
    for d in (M, MX, MD):
        d["pc"] = d["pc"].astype(np.float64); d["veto2g_fail"] = d["veto2g_fail"].astype(np.float64)
        for k in SENTINEL:
            v = np.asarray(d[k], np.float64).copy(); v[v <= -8.0] = np.nan; d[k] = v
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    CATS, NMC, COL = SPS.CATS, SPS.N_MC_CAT, SPS.CAT_COLORS
    rows = []
    for stage, key in (("S3", "base_S3"), ("S5", "base")):
        m, mx, md = M[key], MX[key], MD[key]
        for name, desc in FEATURES:
            v, vx, vd = (np.asarray(M[name], np.float64), np.asarray(MX[name], np.float64), np.asarray(MD[name], np.float64))
            allv = np.concatenate([v[m], vx[mx], vd[md]]); allv = allv[np.isfinite(allv)]
            if len(allv) == 0:
                continue
            lo, hi = np.percentile(allv, [0.5, 99.5])
            if hi <= lo:
                lo, hi = allv.min() - 0.5, allv.max() + 0.5
            nb = min(25, max(5, len(np.unique(allv)))); bins = np.linspace(lo, hi, nb + 1)
            ok, okx, okd = m & np.isfinite(v), mx & np.isfinite(vx), md & np.isfinite(vd)
            hp = np.sum([np.histogram(np.clip(v[ok & (cat == c)], lo, hi - 1e-9), bins=bins, weights=w[ok & (cat == c)])[0] for c in range(NMC)], 0)
            hx = np.histogram(np.clip(vx[okx], lo, hi - 1e-9), bins=bins, weights=wx[okx])[0]; hp = hp + hx
            hd = np.histogram(np.clip(vd[okd], lo, hi - 1e-9), bins=bins)[0]
            chi2 = np.sum((hd - hp) ** 2 / np.maximum(hp, 1.0)); ndf = int((hp > 0).sum())
            nanfrac = (1 - ok.sum() / max(m.sum(), 1), 1 - okd.sum() / max(md.sum(), 1))
            rows.append((stage, name, chi2 / max(ndf, 1), chi2, ndf, hd.sum() / max(hp.sum(), 1e-9), nanfrac))
            fig, axs = plt.subplots(2, 1, figsize=(7.4, 6.2), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
            data = [np.clip(v[ok & (cat == c)], lo, hi - 1e-9) for c in range(NMC)] + [np.clip(vx[okx], lo, hi - 1e-9)]
            ws = [w[ok & (cat == c)] for c in range(NMC)] + [wx[okx]]
            axs[0].hist(data, bins=bins, weights=ws, stacked=True, color=list(COL[:NMC + 1]), label=[f"{CATS[c]} ({ws[c].sum():.1f})" for c in range(NMC + 1)])
            ctr = 0.5 * (bins[:-1] + bins[1:])
            axs[0].errorbar(ctr[hd > 0], hd[hd > 0], yerr=np.sqrt(hd[hd > 0]), fmt="ko", ms=3.5, label=f"beam data ({int(hd.sum())})")
            axs[0].set(ylabel=f"events / {args.pot:.1e} POT", title=f"{name} at {stage}: {desc}  (data/pred {hd.sum() / max(hp.sum(), 1e-9):.2f}, chi2/ndf {chi2:.0f}/{ndf})")
            axs[0].legend(fontsize=5); axs[0].grid(alpha=0.3)
            r = np.where(hp > 0, hd / np.maximum(hp, 1e-9), np.nan); er = np.where(hp > 0, np.sqrt(hd) / np.maximum(hp, 1e-9), np.nan)
            axs[1].errorbar(ctr, r, yerr=er, fmt="ko", ms=3); axs[1].axhline(1, color="r"); axs[1].set(ylim=(0, 2), ylabel="data / pred", xlabel=name)
            fig.tight_layout(); fig.savefig(os.path.join(args.plots, f"{stage}_{name}.png"), dpi=100)
            axs[0].set_yscale("log"); axs[0].set_ylim(0.02, max(hp.max(), 1) * 3); fig.savefig(os.path.join(args.plots, f"{stage}_{name}_logy.png"), dpi=100); plt.close(fig)
    L = []
    for stage in ("S3", "S5"):
        L.append(f"== {stage}: inputs sorted by chi2/ndf (data vs MC+EXT, <=25 bins between the 0.5-99.5 percentiles)")
        L.append(f"{'variable':22s}{'chi2/ndf':>9}{'chi2':>8}{'ndf':>5}{'d/p':>7}{'NaN MC':>8}{'NaN data':>9}")
        for r in sorted([r for r in rows if r[0] == stage], key=lambda r: -r[2]):
            L.append(f"{r[1]:22s}{r[2]:9.2f}{r[3]:8.1f}{r[4]:5d}{r[5]:7.3f}{r[6][0]:8.2f}{r[6][1]:9.2f}")
    txt = "\n".join(L); print(txt)
    open(os.path.join(args.plots, "input_agreement.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
