"""Efficiency / purity tables for the single-photon selection split into FIVE
paths: (VERTEX sample: candidate on the primary in-FV nu vertex, cosmic BDT)
x (passes the SECOND-PHOTON veto) x (FULLY | PARTIALLY contained), plus the
vertex veto-fail cell, plus the NO-VERTEX sample (vertex-less / non-FV-vertex
candidate, vertex-free BDT) x (passes | fails the veto). Second-photon veto:
another nu-stream photon-labelled prong with reco E > --veto2g-emin (any BDT,
any vertex). Partially contained: any nu-stream track prong (any PID / E)
whose start or end is < --contain-dwall from a TPC wall (the exiting-muon
loophole of the CC-numu background). Same selection machinery / options as
single_photon_selection.py (--novtx required; the path is decided by the
leading candidate at the "1 photon" step).

Motivation (2026-09-06): rather than merging the two paths into one sample,
use the no-vertex sample as the entering-photon sideband / constraint (and a
second place to look for an excess) and the vertex sample as the in-FV
1g+X / 1g0X sample where MiniBooNE-invisible p / pi can be seen.

Outputs (--plots): path_tables.txt with, per sample and per step from S2 on:
signal by category (in-FV 1g0X, in-FV 1g+X, entering no-mu, entering CC-mu)
with per-category efficiency (denominator = ALL true events of that category),
MC background by category, EXT, prediction, data, purity (all signal),
in-FV purity, entering purity, and the in-FV : entering signal ratio; plus
stacked candidate-energy spectra for each sample at the final step.

    python3 single_photon_path_tables.py --mc-ntuple ... --ext-ntuple ... \
        --data-ntuple ... --novtx --mc-odd-only --plots plots_v2_s1ep2p8_novtx/paths
"""
import argparse
import os
import sys

import numpy as np
import awkward as ak

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import single_photon_selection as SPS            # noqa: E402
from single_photon_diagnostics import add_args   # noqa: E402


def run(ntuple, is_mc, args):
    a, pot = SPS.load(ntuple, is_mc, args.novtx)
    R = SPS.reco_tables(a, args)
    R["novtx"] = args.novtx
    names, S, E, _, _, path = SPS.build_steps(R, args.shower_bdt_min, args.flashchi2_cut,
                                              args.bdt_after_multiplicity, args.flashpe_ratio_max)
    return a, pot, names, S, E, path, R["other_gE_max"], R["track_min_dwall"], R["cand_dist_from_wall"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_args(ap)
    ap.add_argument("--contain-dwall", type=float, default=10.0,
                    help="PARTIALLY contained = any nu-stream track prong (any PID / E) "
                         "starts or ends closer than this [cm] to a TPC wall; splits "
                         "the vertex + veto-pass cell into FC / PC")
    ap.add_argument("--dfw-cut", type=float, default=SPS.DFW_CUT,
                    help="split the vertex/veto-pass/FC cell by the candidate's BACKWARD wall "
                         "distance: >= cut = in-TPC enriched, < cut (or start outside) = entering-like")
    ap.add_argument("--veto2g-emin", type=float, default=10.0,
                    help="second-photon tag: another nu-stream photon prong with reco E "
                         "above this [MeV] (any BDT score, any vertex) -> 'veto' half")
    args = ap.parse_args()
    SPS.set_truth_scale(getattr(args, 'truth_scale', 'deposited'))
    if not args.novtx:
        sys.exit("--novtx is required (the path split only exists there)")
    os.makedirs(args.plots, exist_ok=True)

    a, pot, names, S, E, path, og, tdw, dfw = run(args.mc_ntuple, True, args)
    v2 = og > args.veto2g_emin                     # True = FAILS the second-photon veto
    pc = tdw < args.contain_dwall                  # True = PARTIALLY contained
    dwin = dfw >= args.dfw_cut                     # True = in-TPC-like (NaN -> False = entering-like)
    w0 = ak.to_numpy(a["xsecWeight"]).astype(np.float64)
    w = np.where(w0 > 0, w0, 0.0) * (args.pot / pot)
    if args.mc_odd_only:
        odd = ak.to_numpy(a["event"]) % 2 == 1
        w = np.where(odd, 2.0 * w, 0.0)
    T = SPS.truth_tables(a); cat = T["cat"]; sig = T["sig_incl"]
    del a
    SX = EX = PX = None; wx = None
    if args.ext_ntuple:
        _, _, _, SX, EX, PX, ogx, tdwx, dfwx = run(args.ext_ntuple, False, args)
        v2x = ogx > args.veto2g_emin; pcx = tdwx < args.contain_dwall; dwinx = dfwx >= args.dfw_cut
        wx = np.zeros(len(PX)); wx[args.ext_row_min:] = args.ext_scale
    SD = ED = PD = None
    if args.data_ntuple:
        _, _, _, SD, ED, PD, ogd, tdwd, dfwd = run(args.data_ntuple, False, args)
        v2d = ogd > args.veto2g_emin; pcd = tdwd < args.contain_dwall; dwind = dfwd >= args.dfw_cut

    CATS, NSIG, NMC = SPS.CATS, SPS.N_SIG_CAT, SPS.N_MC_CAT
    den = {c: w[cat == c].sum() for c in range(NSIG)}
    infv = np.isin(cat, [0, 1]); enter = np.isin(cat, [2, 3])
    lines = [f"denominators (w @ {args.pot:.1e} POT): "
             + " | ".join(f"{CATS[c]} {den[c]:.1f}" for c in range(NSIG))]
    k2 = 2                                     # first step with a candidate
    # six paths: (path, veto outcome, containment or None, dfw-in or None)
    SAMPLES = [(0, False, False, True, f"VERTEX, veto-pass, FC, dist-from-wall >= {args.dfw_cut:.0f} (in-TPC)", "vertex_pass_FC_in"),
               (0, False, False, False, f"VERTEX, veto-pass, FC, dist-from-wall < {args.dfw_cut:.0f} (entering-like)", "vertex_pass_FC_ent"),
               (0, False, True, None, "VERTEX, veto-pass, PARTIALLY contained", "vertex_pass_PC"),
               (0, True, None, None, "VERTEX, FAILS 2nd-photon veto (another photon prong E > e_min)", "vertex_fail"),
               (1, False, None, None, "NO-VERTEX, passes 2nd-photon veto", "novertex_pass"),
               (1, True, None, None, "NO-VERTEX, FAILS 2nd-photon veto", "novertex_fail")]

    def cell(pv, vf, cf, df, S_, P_, V_, C_, W_):
        m = S_ & (P_ == pv) & (V_ == vf)
        if cf is not None:
            m = m & (C_ == cf)
        if df is not None:
            m = m & (W_ == df)
        return m
    lines.append(f"second-photon tag: another nu-stream photon prong with reco E > "
                 f"{args.veto2g_emin:.0f} MeV (any BDT, any vertex); partially contained = any "
                 f"nu-stream track prong start/end < {args.contain_dwall:.0f} cm from a TPC wall")
    for pv, vf, cf, df, plab, _fn in SAMPLES:
        lines.append(f"\n===== {plab} =====")
        hdr = (f"{'step':<14}" + "".join(f"{CATS[c][:12]:>13}{'eff':>6}" for c in range(NSIG))
               + f"{'sigTot':>8}{'MCbkg':>8}{'EXT':>8}{'pred':>8}{'data':>6}{'d/p':>6}"
               f"{'pur':>6}{'purFV':>7}{'purEnt':>7}{'FV:Ent':>7}")
        lines.append(hdr)
        for k in range(k2, len(names)):
            m = cell(pv, vf, cf, df, S[k], path, v2, pc, dwin)
            cells = ""
            for c in range(NSIG):
                s = w[m & (cat == c)].sum()
                cells += f"{s:13.1f}{s / max(den[c], 1e-9):6.3f}"
            st = w[m & sig].sum(); b = w[m & ~sig].sum()
            e = wx[cell(pv, vf, cf, df, SX[k], PX, v2x, pcx, dwinx)].sum() if SX is not None else 0.0
            d = int(cell(pv, vf, cf, df, SD[k], PD, v2d, pcd, dwind).sum()) if SD is not None else 0
            tot = max(st + b + e, 1e-9)
            sfv, sen = w[m & infv].sum(), w[m & enter].sum()
            lines.append(f"{names[k]:<14}" + cells
                         + f"{st:8.1f}{b:8.1f}{e:8.1f}{tot:8.1f}{d:6d}{d / tot:6.2f}"
                         f"{st / tot:6.3f}{sfv / tot:7.3f}{sen / tot:7.3f}"
                         f"{sfv / max(sen, 1e-9):7.2f}")
        m = cell(pv, vf, cf, df, S[-1], path, v2, pc, dwin)
        lines.append("  final MC background composition: " + " | ".join(
            f"{CATS[c][5:]} {w[m & (cat == c)].sum():.1f}" for c in range(NSIG, NMC)
            if w[m & (cat == c)].sum() > 0))
    lines.append("\nSIX-PATH SUMMARY at the final step (w)")
    lines.append(f"{'':44s}{'signal':>9}{'in-FV sig':>10}{'MC bkg':>9}{'EXT':>8}{'pred':>9}{'data':>7}{'d/p':>6}{'purity':>8}{'purFV':>7}")
    for pv, vf, cf, df, plab, _fn in SAMPLES:
        m = cell(pv, vf, cf, df, S[-1], path, v2, pc, dwin)
        st, sf, b = w[m & sig].sum(), w[m & infv].sum(), w[m & ~sig].sum()
        e = wx[cell(pv, vf, cf, df, SX[-1], PX, v2x, pcx, dwinx)].sum() if SX is not None else 0.0
        d = int(cell(pv, vf, cf, df, SD[-1], PD, v2d, pcd, dwind).sum()) if SD is not None else 0
        tot = max(st + b + e, 1e-9)
        lines.append(f"{plab[:44]:44s}{st:9.1f}{sf:10.1f}{b:9.1f}{e:8.1f}{tot:9.1f}{d:7d}{d/tot:6.2f}{st/tot:8.3f}{sf/tot:7.3f}")
    # ---- TRUTH containment: definition match + FC/PC efficiencies -------------
    pct = T["true_track_min_dwall"] < args.contain_dwall     # truth PC (KE >= 0)
    lines.append(f"\nTRUTH CONTAINMENT: PC_true = any charged track-like nu-origin particle "
                 f"(mu/pi/K/p/charged baryons) that deposits charge with true START or END < "
                 f"{args.contain_dwall:.0f} cm from a TPC wall (start/end only; no trajectory)")
    fin_ = S[-1]
    vp = fin_ & (path == 0) & ~v2                                # vertex veto-pass cell
    for kel, key in (("KE>=0", "true_track_min_dwall"), ("KE>=20", "true_track_min_dwall_ke20"),
                     ("KE>=50", "true_track_min_dwall_ke50")):
        pt = T[key] < args.contain_dwall
        m = vp & sig
        agree = w[m & (pt == pc)].sum() / max(w[m].sum(), 1e-9)
        lines.append(f"  truth({kel}) vs reco agreement on selected vertex-pass SIGNAL: {agree:.3f} "
                     f"| reco PC among truth FC {w[m & ~pt & pc].sum()/max(w[m & ~pt].sum(),1e-9):.3f}"
                     f" | reco FC among truth PC {w[m & pt & ~pc].sum()/max(w[m & pt].sum(),1e-9):.3f}")
    lines.append(f"\nEFFICIENCY into each reco cell, denominator = ALL true events of the "
                 f"(category x truth containment) row (w); 'any' = passes the full selection anywhere")
    cells5 = [("FC in-TPC", lambda S_, P_, V_, C_: cell(0, False, False, True, S_, P_, V_, C_, dwin)),
              ("FC entering", lambda S_, P_, V_, C_: cell(0, False, False, False, S_, P_, V_, C_, dwin)),
              ("vtx-pass PC", lambda S_, P_, V_, C_: cell(0, False, True, None, S_, P_, V_, C_, dwin)),
              ("vtx-fail", lambda S_, P_, V_, C_: cell(0, True, None, None, S_, P_, V_, C_, dwin)),
              ("novtx-pass", lambda S_, P_, V_, C_: cell(1, False, None, None, S_, P_, V_, C_, dwin)),
              ("novtx-fail", lambda S_, P_, V_, C_: cell(1, True, None, None, S_, P_, V_, C_, dwin))]
    lines.append(f"{'row':38s}{'N(w)':>8}" + "".join(f"{c_[0]:>13}" for c_ in cells5) + f"{'any':>8}")
    for c in range(NSIG):
        for tl, tm in (("truth FC", ~pct), ("truth PC", pct)):
            den_ = (cat == c) & tm
            D_ = w[den_].sum()
            if D_ <= 0:
                continue
            row = f"{CATS[c][:26] + ', ' + tl:38s}{D_:8.1f}"
            for _, f_ in cells5:
                row += f"{w[den_ & f_(fin_, path, v2, pc)].sum() / D_:13.3f}"
            row += f"{w[den_ & fin_].sum() / D_:8.3f}"
            lines.append(row)
    # composition of the reco cells by truth containment (selected signal)
    lines.append("truth containment of the SELECTED SIGNAL in each reco cell (w): "
                 + " | ".join(f"{nm} FC {w[f_(fin_, path, v2, pc) & sig & ~pct].sum():.1f} / PC "
                              f"{w[f_(fin_, path, v2, pc) & sig & pct].sum():.1f}" for nm, f_ in cells5[:2]))
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(args.plots, "path_tables.txt"), "w") as fh:
        fh.write(txt + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bins = np.linspace(0, 1000, 41)
    for pv, vf, cf, df, plab, fn in SAMPLES:
        m = cell(pv, vf, cf, df, S[-1], path, v2, pc, dwin) & np.isfinite(E)
        data = [np.clip(E[m & (cat == c)], 0, 999.9) for c in range(NMC)]
        ws = [w[m & (cat == c)] for c in range(NMC)]
        cols = list(SPS.CAT_COLORS[:NMC])
        if SX is not None:
            mx = cell(pv, vf, cf, df, SX[-1], PX, v2x, pcx, dwinx) & np.isfinite(EX)
            data.append(np.clip(EX[mx], 0, 999.9)); ws.append(wx[mx]); cols.append(SPS.CAT_COLORS[NMC])
        fig, ax = plt.subplots(figsize=(7.4, 4.8))
        ax.hist(data, bins=bins, weights=ws, stacked=True, color=cols,
                label=[f"{CATS[c]} ({ws[c].sum():.1f})" for c in range(len(ws))])
        if SD is not None:
            md = cell(pv, vf, cf, df, SD[-1], PD, v2d, pcd, dwind) & np.isfinite(ED)
            hd, _ = np.histogram(np.clip(ED[md], 0, 999.9), bins=bins)
            ctr = 0.5 * (bins[:-1] + bins[1:])
            ax.errorbar(ctr[hd > 0], hd[hd > 0], yerr=np.sqrt(hd[hd > 0]), fmt="ko", ms=3.5,
                        label=f"beam data ({int(hd.sum())})")
        ax.set(xlabel="reco photon energy [MeV]", ylabel=f"events / {args.pot:.1e} POT",
               title=f"candidate photon energy, {plab} ({names[-1]})")
        ax.legend(fontsize=6.5); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(f"{args.plots}/photon_energy_{fn}.png", dpi=110)
        plt.close(fig)
    print(f">>> -> {args.plots}")


if __name__ == "__main__":
    main()
