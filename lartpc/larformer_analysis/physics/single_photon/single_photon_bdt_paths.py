#!/usr/bin/env python3
"""BDT-path tree tables + stacks (2026-10-08). Tree:

  base S1-S6
  ├─ vertex path:   veto FAIL (sideband) | veto PASS
  │     PASS ├─ ccnumu     chain BDT_numu (flattened) <= --numu-cut
  │          └─ nonccnumu  > --numu-cut
  │                ├─ entering  BDT_enter < --enter-cut (Np) / --enter-cut-0p (0p)
  │                └─ infv      BDT_enter >= ...  (MB target) -> 0p / np
  └─ no-vertex path: veto FAIL | veto PASS
with the GEOMETRIC 2nd-photon veto (--veto2g-emin, --veto2g-dcmax) from
single_photon_selection.build_steps; the old any-prong veto is tabulated too.
Reuses single_photon_diagnostics.per_sample (chain scores, flattened).

    python3 single_photon_bdt_paths.py --mc-ntuple ... --ext-ntuple ... --data-ntuple ... --novtx \\
        --mb-chain-model plots_run1_mbchain/mb_bdt_chain_noflash_novtx_lar.joblib --veto2g-dcmax 100 --plots <dir>
"""
import argparse
import os
import sys

import numpy as np
import awkward as ak

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import single_photon_selection as SPS                      # noqa: E402
import single_photon_diagnostics as SPD                    # noqa: E402
from mb_variable_study import features_from                # noqa: E402


def cells_for(M, R, Mf, args):
    fin = M["steps"][-1]; path = M["path"]
    og = R["other_gE_max"]; og_old = R["other_gE_max_old"]
    fail = np.nan_to_num(og, nan=-1.0) > args.veto2g_emin
    fail_old = np.nan_to_num(og_old, nan=-1.0) > args.veto2g_emin
    numu = np.nan_to_num(M["bdtChainNumu"], nan=-1.0)
    ent = np.nan_to_num(M["bdtChainEnter"], nan=-1.0)
    npm = np.asarray(Mf["n_p"]) >= 1
    thr = np.where(npm, args.enter_cut, args.enter_cut_0p)
    v = fin & (path == 0); nv = fin & (path == 1)
    vp = v & ~fail; cc = vp & (numu <= args.numu_cut); ncc = vp & (numu > args.numu_cut)
    infv = ncc & (ent >= thr); entc = ncc & (ent < thr)
    C = [("vertex, veto FAIL (sideband)", v & fail), ("vertex, veto PASS", vp),
         ("  vertex_pass, CC-numu-like (BDT_numu <= %.2f)" % args.numu_cut, cc),
         ("  vertex_pass, non-CC-numu", ncc),
         ("    non-CC-numu, ENTERING (BDT_enter below)", entc),
         ("    non-CC-numu, IN-FV (MB target)", infv),
         ("      in-FV, 1g0p", infv & ~npm), ("      in-FV, 1gNp", infv & npm),
         ("no-vertex, veto FAIL", nv & fail), ("no-vertex, veto PASS (entering sideband)", nv & ~fail)]
    OLD = [("vertex, OLD veto FAIL", v & fail_old), ("vertex, OLD veto PASS", v & ~fail_old),
           ("no-vertex, OLD veto FAIL", nv & fail_old), ("no-vertex, OLD veto PASS", nv & ~fail_old)]
    return C, OLD


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    SPD.add_args(ap)
    ap.set_defaults(signal_def="lar")
    for k, v in (("mb_mu_kemin", 40.0), ("mb_pi_kemin", 50.0), ("mb_p_kemax", 342.0), ("extra_emax", 10.0), ("cone_e2_min", 5.0), ("cone_e2_max", 600.0),
                 ("contain_dwall", 10.0), ("dfw_cut", SPS.DFW_CUT), ("veto2g_emin", 10.0), ("veto2g_dcmax", 100.0),
                 ("numu_cut", 0.25), ("enter_cut", 0.30), ("enter_cut_0p", 0.15)):
        ap.add_argument("--" + k.replace("_", "-"), type=float, default=v)
    ap.add_argument("--mb-chain-model", default=os.path.join(HERE, "plots_run1_mbchain", "mb_bdt_chain_noflash_novtx_lar.joblib"))
    ap.add_argument("--mb-chain-model-extra", default="")
    args = ap.parse_args()
    SPS.configure(args); os.makedirs(args.plots, exist_ok=True)
    M, a, pot, R = SPD.per_sample(args.mc_ntuple, True, args, None, "MC")
    w0 = ak.to_numpy(a["xsecWeight"]).astype(np.float64); w = np.where(w0 > 0, w0, 0.0) * (args.pot / pot)
    T = SPS.truth_tables(a); cat = T["cat"]; sig = T["sig_incl"]; infv_t = sig & T["in_fv"]
    Mf = features_from(a, R, M["names"], M["steps"], M["E"], M["path"], args); del a
    C, OLD = cells_for(M, R, Mf, args)
    X = D = None
    if args.ext_ntuple:
        X, ax, _, RX = SPD.per_sample(args.ext_ntuple, False, args, None, "EXT")
        MfX = features_from(ax, RX, X["names"], X["steps"], X["E"], X["path"], args); del ax
        wx = np.zeros(len(X["E"])); wx[args.ext_row_min:] = args.ext_scale
        CX, OLDX = cells_for(X, RX, MfX, args)
    if args.data_ntuple:
        D, ad, _, RD = SPD.per_sample(args.data_ntuple, False, args, None, "DATA")
        MfD = features_from(ad, RD, D["names"], D["steps"], D["E"], D["path"], args); del ad
        CD, OLDD = cells_for(D, RD, MfD, args)
    CATS, NSIG, NMC = SPS.CATS, SPS.N_SIG_CAT, SPS.N_MC_CAT
    den_fv = w[infv_t].sum()
    L = [f"BDT-PATH TREE  (truth '{SPS.SIGNAL_DEF}', veto: photon prong E > {args.veto2g_emin:.0f} MeV AND axes cross within {args.veto2g_dcmax:.0f} cm of both starts; "
         f"chain model {os.path.basename(args.mb_chain_model)}; BDT_numu > {args.numu_cut:.2f}; BDT_enter >= {args.enter_cut:.2f} Np / {args.enter_cut_0p:.2f} 0p) @ {args.pot:.1e} POT",
         f"in-FV signal denominator {den_fv:.1f} (w)",
         f"{'cell':52s}{'signal':>8}{'in-FV':>7}{'effFV':>7}{'enter':>7}{'MC bkg':>8}{'EXT':>7}{'pred':>8}{'data':>6}{'d/p':>6}{'pur':>7}{'purFV':>7}"]

    def row(lab, m, mx, md):
        st, sf, se, b = w[m & sig].sum(), w[m & infv_t].sum(), w[m & sig & ~T["in_fv"]].sum(), w[m & ~sig].sum()
        e = wx[mx].sum() if mx is not None else 0.0; d = int(md.sum()) if md is not None else 0
        tot = max(st + b + e, 1e-9)
        return f"{lab[:52]:52s}{st:8.1f}{sf:7.1f}{sf / max(den_fv, 1e-9):7.3f}{se:7.1f}{b:8.1f}{e:7.1f}{tot:8.1f}{d:6d}{d / tot:6.2f}{st / tot:7.3f}{sf / tot:7.3f}"
    for k, (lab, m) in enumerate(C):
        L.append(row(lab, m, CX[k][1] if X is not None else None, CD[k][1] if D is not None else None))
        L.append("    bkg: " + " | ".join(f"{CATS[c][5:26]} {w[m & (cat == c)].sum():.1f}" for c in range(NSIG, NMC) if w[m & (cat == c)].sum() > 0))
    L.append("\nOLD any-prong veto for comparison:")
    for k, (lab, m) in enumerate(OLD):
        L.append(row(lab, m, OLDX[k][1] if X is not None else None, OLDD[k][1] if D is not None else None))
    txt = "\n".join(L); print(txt)
    open(os.path.join(args.plots, "bdt_paths.txt"), "w").write(txt + "\n")
    # candidate-energy stacks per cell
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    bins = np.linspace(0, 1000, 41); E = M["E"]
    for k, (lab, m) in enumerate(C):
        mm = m & np.isfinite(E)
        data = [np.clip(E[mm & (cat == c)], 0, 999.9) for c in range(NMC)]; ws = [w[mm & (cat == c)] for c in range(NMC)]; cols = list(SPS.CAT_COLORS[:NMC])
        if X is not None:
            mx = CX[k][1] & np.isfinite(X["E"]); data.append(np.clip(X["E"][mx], 0, 999.9)); ws.append(wx[mx]); cols.append(SPS.CAT_COLORS[NMC])
        fig, ax_ = plt.subplots(figsize=(7.4, 4.8))
        ax_.hist(data, bins=bins, weights=ws, stacked=True, color=cols, label=[f"{CATS[c]} ({ws[c].sum():.1f})" for c in range(len(ws))])
        if D is not None:
            md = CD[k][1] & np.isfinite(D["E"]); hd, _ = np.histogram(np.clip(D["E"][md], 0, 999.9), bins=bins); ctr = 0.5 * (bins[:-1] + bins[1:])
            ax_.errorbar(ctr[hd > 0], hd[hd > 0], yerr=np.sqrt(hd[hd > 0]), fmt="ko", ms=3.5, label=f"beam data ({int(hd.sum())})")
        ax_.set(xlabel="reco photon energy [MeV]", ylabel=f"events / {args.pot:.1e} POT", title=f"candidate photon energy: {lab.strip()}")
        ax_.legend(fontsize=6.5); ax_.grid(alpha=0.3)
        fn = lab.strip().replace(" ", "_").replace(",", "").replace("(", "").replace(")", "").replace("/", "").replace("<=", "le").replace(">", "gt").replace(".", "p")[:60]
        fig.tight_layout(); fig.savefig(os.path.join(args.plots, f"photon_energy_{k:02d}_{fn}.png"), dpi=110); plt.close(fig)
    print(f">>> -> {args.plots}")


if __name__ == "__main__":
    main()
