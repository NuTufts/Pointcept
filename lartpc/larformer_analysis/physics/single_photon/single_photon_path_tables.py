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
    import single_photon_mb_selection as MBS
    import single_photon_mb_bdt_chain as MBC
    a, pot = SPS.load(ntuple, is_mc, args.novtx, extra=MBS.BR_EXTRA)
    R = SPS.reco_tables(a, args)
    R["novtx"] = args.novtx
    names, S, E, sc, _, path = SPS.build_steps(R, args.shower_bdt_min, args.flashchi2_cut,
                                               args.bdt_after_multiplicity, args.flashpe_ratio_max)
    need_chain = args.fc_split == "enter" or args.chain_ext_min is not None or args.chain_numu_min is not None
    tight = np.ones(len(E), bool)
    if need_chain:
        ch = MBC.chain_scores(a, R, names, S, E, path, args, args.mb_chain_model)
        if ch is None:
            sys.exit(f"the MB-chain model is needed: {args.mb_chain_model}")
        if args.chain_ext_min is not None:
            tight &= np.nan_to_num(ch["ext"], nan=-1.0) > args.chain_ext_min
        if args.chain_numu_min is not None:
            tight &= np.nan_to_num(ch["numu"], nan=-1.0) > args.chain_numu_min
    if args.fc_split == "enter":
        inflag = np.nan_to_num(ch["enter"], nan=-1.0) > args.enter_cut
    else:
        inflag = np.nan_to_num(R["cand_dist_from_wall"], nan=-1.0) >= args.dfw_cut
    return a, pot, names, S, E, path, R["other_gE_max"], R["track_min_dwall"], R["cand_dist_from_wall"], sc, (inflag, tight)


def dfw_code(inflag, score, args):
    """Per-event in-TPC code for the FC cell split: 1 = in-TPC-like (inflag: backward
    wall distance >= --dfw-cut, or chain BDT_enter >= --enter-cut with --fc-split enter)
    AND candidate score >= --fc-in-bdt-min (if set), 0 = entering-like,
    -1 = in-TPC-like but failing the tighter FC-in BDT cut (lands in no path).
    cell() compares the code to df (True/False), so -1 matches neither cell."""
    inflag, tight = inflag                      # (in-class flag, passes the chain BDT_ext / BDT_numu cuts)
    code = np.asarray(inflag, bool).astype(np.int64)
    if args.fc_in_bdt_min is not None:
        code[(code == 1) & ~(np.nan_to_num(score, nan=-9.0) >= args.fc_in_bdt_min)] = -1
    code[(code == 1) & ~tight] = -1
    return code


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_args(ap)
    ap.add_argument("--contain-dwall", type=float, default=10.0,
                    help="PARTIALLY contained = any nu-stream track prong (any PID / E) "
                         "starts or ends closer than this [cm] to a TPC wall; splits "
                         "the vertex + veto-pass cell into FC / PC")
    ap.add_argument("--fc-split", default="dfw", choices=["dfw", "enter"],
                    help="how the vertex / veto-pass / FC cell is split into in-TPC vs entering-like: "
                         "'dfw' = backward wall distance >= --dfw-cut; 'enter' = MB-chain BDT_enter >= --enter-cut")
    ap.add_argument("--enter-cut", type=float, default=0.05)
    ap.add_argument("--chain-ext-min", type=float, default=None,
                    help="in-FV-like cell only: MB-chain BDT_ext (flattened) > this (fails -> no path)")
    ap.add_argument("--chain-numu-min", type=float, default=None,
                    help="in-FV-like cell only: MB-chain BDT_numu (flattened) > this (fails -> no path)")
    ap.add_argument("--mb-chain-model", default=os.path.join(HERE, "plots_run1_mbchain", "mb_bdt_chain_noflash_novtx.joblib"))
    for k, v in (("mb_mu_kemin", 40.0), ("mb_pi_kemin", 50.0), ("mb_p_kemax", 342.0), ("extra_emax", 10.0),
                 ("cone_e2_min", 5.0), ("cone_e2_max", 600.0)):
        ap.add_argument("--" + k.replace("_", "-"), type=float, default=v, help="MB-chain feature option")
    ap.add_argument("--dfw-cut", type=float, default=SPS.DFW_CUT,
                    help="split the vertex/veto-pass/FC cell by the candidate's BACKWARD wall "
                         "distance: >= cut = in-TPC enriched, < cut (or start outside) = entering-like")
    ap.add_argument("--fc-in-bdt-min", type=float, default=None,
                    help="extra cut on the FC in-TPC cell only: leading-candidate cosmic BDT "
                         "score >= this (candidate set still defined by --shower-bdt-min); "
                         "failing events go to NO path")
    ap.add_argument("--veto2g-emin", type=float, default=10.0,
                    help="second-photon tag: another nu-stream photon prong with reco E "
                         "above this [MeV] (any BDT score, any vertex) -> 'veto' half")
    args = ap.parse_args()
    SPS.configure(args)
    if not args.novtx:
        sys.exit("--novtx is required (the path split only exists there)")
    os.makedirs(args.plots, exist_ok=True)

    a, pot, names, S, E, path, og, tdw, dfw, bsc, infl = run(args.mc_ntuple, True, args)
    v2 = og > args.veto2g_emin                     # True = FAILS the second-photon veto
    pc = tdw < args.contain_dwall                  # True = PARTIALLY contained
    dwin = dfw_code(infl, bsc, args)               # 1 = in-TPC-like, 0 = entering-like, -1 = fails FC-in BDT
    w0 = ak.to_numpy(a["xsecWeight"]).astype(np.float64)
    w = np.where(w0 > 0, w0, 0.0) * (args.pot / pot)
    if args.mc_odd_only:
        odd = ak.to_numpy(a["event"]) % 2 == 1
        w = np.where(odd, 2.0 * w, 0.0)
    T = SPS.truth_tables(a); cat = T["cat"]; sig = T["sig_incl"]
    del a
    SX = EX = PX = None; wx = None
    if args.ext_ntuple:
        _, _, _, SX, EX, PX, ogx, tdwx, dfwx, bscx, inflx = run(args.ext_ntuple, False, args)
        v2x = ogx > args.veto2g_emin; pcx = tdwx < args.contain_dwall; dwinx = dfw_code(inflx, bscx, args)
        wx = np.zeros(len(PX)); wx[args.ext_row_min:] = args.ext_scale
    SD = ED = PD = None
    if args.data_ntuple:
        _, _, _, SD, ED, PD, ogd, tdwd, dfwd, bscd, infld = run(args.data_ntuple, False, args)
        v2d = ogd > args.veto2g_emin; pcd = tdwd < args.contain_dwall; dwind = dfw_code(infld, bscd, args)

    CATS, NSIG, NMC = SPS.CATS, SPS.N_SIG_CAT, SPS.N_MC_CAT
    den = {c: w[cat == c].sum() for c in range(NSIG)}
    infv = np.isin(cat, [0, 1]); enter = np.isin(cat, [2, 3])
    lines = [f"denominators (w @ {args.pot:.1e} POT): "
             + " | ".join(f"{CATS[c]} {den[c]:.1f}" for c in range(NSIG))]
    k2 = 2                                     # first step with a candidate
    # six paths: (path, veto outcome, containment or None, dfw-in or None)
    tb = f", BDT >= {args.fc_in_bdt_min:.2f}" if args.fc_in_bdt_min is not None else ""
    if args.chain_ext_min is not None:
        tb += f", BDT_ext > {args.chain_ext_min:.2f}"
    if args.chain_numu_min is not None:
        tb += f", BDT_numu > {args.chain_numu_min:.2f}"
    if args.fc_split == "enter":
        lin, lent = (f"BDT_enter > {args.enter_cut:.2f} (in-FV-like)", f"BDT_enter <= {args.enter_cut:.2f} (entering-like)")
    else:
        lin, lent = (f"dist-from-wall >= {args.dfw_cut:.0f} (in-TPC)", f"dist-from-wall < {args.dfw_cut:.0f} (entering-like)")
    SAMPLES = [(0, False, False, True, f"VERTEX, veto-pass, FC, {lin}{tb}", "vertex_pass_FC_in"),
               (0, False, False, False, f"VERTEX, veto-pass, FC, {lent}", "vertex_pass_FC_ent"),
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
    # ---- TRUTH backward wall distance: definition match ----------------------
    tdfw = T["g_dfw_true"]
    tin = tdfw >= args.dfw_cut                                    # truth in-TPC-like
    lines.append(f"\nTRUTH DIST-FROM-WALL (truth analogue of the dfw split; with --fc-split enter the reco class below is the BDT_enter class): photon first deposit (trueSimPartEDep*) traced back along "
                 f"-p_true to the TPC box; in-TPC_true = >= {args.dfw_cut:.0f} cm (NaN -> entering-like), "
                 f"same construction as the reco cand_dist_from_wall")
    for c in range(NSIG):
        m = (cat == c)
        lines.append(f"  {CATS[c][:36]:36s} truth in-TPC fraction (all true events) "
                     f"{w[m & tin].sum()/max(w[m].sum(),1e-9):.3f} | median truth dfw "
                     f"{np.nanmedian(tdfw[m]) if m.any() else np.nan:6.1f} cm")
    vpf = vp & ~pc                                                # vertex veto-pass FC cell
    m = vpf & sig & (dwin >= 0)
    agree = w[m & (tin == (dwin == 1))].sum() / max(w[m].sum(), 1e-9)
    lines.append(f"  truth vs reco in-TPC agreement on selected vertex-pass-FC SIGNAL: {agree:.3f} "
                 f"| reco entering-like among truth in-TPC {w[m & tin & (dwin == 0)].sum()/max(w[m & tin].sum(),1e-9):.3f}"
                 f" | reco in-TPC among truth entering-like {w[m & ~tin & (dwin == 1)].sum()/max(w[m & ~tin].sum(),1e-9):.3f}")
    r_ = dfw[m]; t_ = tdfw[m]; ok_ = np.isfinite(r_) & np.isfinite(t_)
    lines.append(f"  |reco - truth| dfw on that sample (w-unweighted): median "
                 f"{np.median(np.abs(r_[ok_]-t_[ok_])) if ok_.any() else np.nan:.1f} cm, "
                 f"68% {np.percentile(np.abs(r_[ok_]-t_[ok_]), 68) if ok_.any() else np.nan:.1f} cm")
    # ---- FC in-TPC cell: in-FV single-photon eff / purity + BDT-cut scan ------
    den_fv = w[infv].sum()
    base_in = fin_ & (path == 0) & ~v2 & ~pc & infl[0] & infl[1]               # before any FC-in cosmic-BDT cut
    base_inx = SX[-1] & (PX == 0) & ~v2x & ~pcx & inflx[0] & inflx[1] if SX is not None else None
    base_ind = SD[-1] & (PD == 0) & ~v2d & ~pcd & infld[0] & infld[1] if SD is not None else None
    lines.append(f"\nFC in-TPC cell vs an extra leading-candidate cosmic-BDT cut (candidate set unchanged; "
                 f"applied cut = {args.fc_in_bdt_min}); in-FV eff denominator = ALL true in-FV 1g events "
                 f"({den_fv:.1f} w: 1g0X {den[0]:.1f} + 1g+X {den[1]:.1f})")
    lines.append(f"{'BDT >=':>7}{'sig':>7}{'inFV':>7}{'1g0X':>7}{'1g+X':>7}{'enter':>7}{'MCbkg':>7}{'EXT':>7}"
                 f"{'pred':>7}{'data':>6}{'d/p':>6}{'effFV':>7}{'e1g0X':>7}{'e1g+X':>7}{'pur':>6}{'purFV':>7}{'eff*purFV':>10}")
    scan = sorted(set([args.shower_bdt_min, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
                      + ([args.fc_in_bdt_min] if args.fc_in_bdt_min is not None else [])))
    for t in scan:
        m = base_in & (np.nan_to_num(bsc, nan=-9.0) >= t)
        st, sf, b = w[m & sig].sum(), w[m & infv].sum(), w[m & ~sig].sum()
        s0, s1, se = w[m & (cat == 0)].sum(), w[m & (cat == 1)].sum(), w[m & enter].sum()
        e = wx[base_inx & (np.nan_to_num(bscx, nan=-9.0) >= t)].sum() if SX is not None else 0.0
        d = int((base_ind & (np.nan_to_num(bscd, nan=-9.0) >= t)).sum()) if SD is not None else 0
        tot = max(st + b + e, 1e-9); effv = sf / max(den_fv, 1e-9)
        lines.append(f"{t:7.3f}{st:7.1f}{sf:7.1f}{s0:7.1f}{s1:7.1f}{se:7.1f}{b:7.1f}{e:7.1f}{tot:7.1f}{d:6d}"
                     f"{d/tot:6.2f}{effv:7.3f}{s0/max(den[0],1e-9):7.3f}{s1/max(den[1],1e-9):7.3f}"
                     f"{st/tot:6.3f}{sf/tot:7.3f}{effv*sf/tot:10.4f}")
    lines.append(f"\nEFFICIENCY into each reco cell, denominator = ALL true events of the "
                 f"(category x truth containment x truth dist-from-wall) row (w); "
                 f"'any' = passes the full selection anywhere")
    cells5 = [("FC in-TPC", lambda S_, P_, V_, C_: cell(0, False, False, True, S_, P_, V_, C_, dwin)),
              ("FC entering", lambda S_, P_, V_, C_: cell(0, False, False, False, S_, P_, V_, C_, dwin)),
              ("vtx-pass PC", lambda S_, P_, V_, C_: cell(0, False, True, None, S_, P_, V_, C_, dwin)),
              ("vtx-fail", lambda S_, P_, V_, C_: cell(0, True, None, None, S_, P_, V_, C_, dwin)),
              ("novtx-pass", lambda S_, P_, V_, C_: cell(1, False, None, None, S_, P_, V_, C_, dwin)),
              ("novtx-fail", lambda S_, P_, V_, C_: cell(1, True, None, None, S_, P_, V_, C_, dwin))]
    lines.append(f"{'row':40s}{'N(w)':>8}" + "".join(f"{c_[0]:>13}" for c_ in cells5) + f"{'any':>8}")
    for c in range(NSIG):
        for tl, tm in (("truth FC", ~pct), ("truth FC, in-TPC", ~pct & tin),
                       ("truth FC, entering", ~pct & ~tin), ("truth PC", pct)):
            den_ = (cat == c) & tm
            D_ = w[den_].sum()
            if D_ <= 0:
                continue
            row = f"{CATS[c][:20] + ', ' + tl:40s}{D_:8.1f}"
            for _, f_ in cells5:
                row += f"{w[den_ & f_(fin_, path, v2, pc)].sum() / D_:13.3f}"
            row += f"{w[den_ & fin_].sum() / D_:8.3f}"
            lines.append(row)
    # composition of the reco cells by truth containment (selected signal)
    lines.append("truth containment of the SELECTED SIGNAL in each reco cell (w): "
                 + " | ".join(f"{nm} FC {w[f_(fin_, path, v2, pc) & sig & ~pct].sum():.1f} / PC "
                              f"{w[f_(fin_, path, v2, pc) & sig & pct].sum():.1f}" for nm, f_ in cells5[:2]))
    # ---- CONFUSION TABLE: truth path -> reco path ------------------------------
    # truth analogues: vertex = true nu vertex in the WC FV; veto-fail = a second
    # true photon with deposited E > veto2g_emin; PC = pc_true; entering-like =
    # truth dfw < dfw_cut. Rows normalised; 'no path' = fails the selection.
    tv2 = np.nan_to_num(T["g2_evis_true"], nan=-1.0) > args.veto2g_emin
    tfv = T["in_fv"]
    TPATHS = [("vtx, pass, FC, in-TPC", tfv & ~tv2 & ~pct & tin),
              ("vtx, pass, FC, entering", tfv & ~tv2 & ~pct & ~tin),
              ("vtx, pass, PC", tfv & ~tv2 & pct),
              ("vtx, fail", tfv & tv2),
              ("no-vtx, pass", ~tfv & ~tv2),
              ("no-vtx, fail", ~tfv & tv2)]
    rcells = [(nm, f_(fin_, path, v2, pc)) for nm, f_ in cells5]
    hdr = f"{'truth path (rows sum to 1)':44s}{'N(w)':>8}" + "".join(f"{nm:>13}" for nm, _ in rcells) + f"{'no path':>9}"
    lines.append(f"\nCONFUSION: fraction of each TRUTH path landing in each RECO path (final step). "
                 f"Truth path = (true vtx in FV) x (2nd true photon dep E > {args.veto2g_emin:.0f} MeV) x "
                 f"(truth PC) x (truth dfw < {args.dfw_cut:.0f})")
    for grp, gm in [("ALL SIGNAL (cats 0-3)", sig)] + [(CATS[c], cat == c) for c in range(NSIG)]:
        lines.append(f"-- {grp}"); lines.append(hdr)
        for tn, tm in TPATHS:
            den_ = gm & tm; D_ = w[den_].sum()
            if D_ <= 0:
                continue
            row = f"{tn:44s}{D_:8.1f}"
            for _, rm in rcells:
                row += f"{w[den_ & rm].sum() / D_:13.3f}"
            row += f"{w[den_ & ~fin_].sum() / D_:9.3f}"
            lines.append(row)
    lines.append("-- by TRUE EVENT TYPE (no truth-path split; MC backgrounds included)")
    lines.append(f"{'true event type':44s}{'N(w)':>8}" + "".join(f"{nm:>13}" for nm, _ in rcells) + f"{'no path':>9}")
    for c in range(NMC):
        den_ = cat == c; D_ = w[den_].sum()
        if D_ <= 0:
            continue
        row = f"{CATS[c][:44]:44s}{D_:8.1f}"
        for _, rm in rcells:
            row += f"{w[den_ & rm].sum() / D_:13.4f}"
        row += f"{w[den_ & ~fin_].sum() / D_:9.4f}"
        lines.append(row)
    lines.append("truth dist-from-wall class of the SELECTED SIGNAL in each reco cell (w): "
                 + " | ".join(f"{nm} in-TPC {w[f_(fin_, path, v2, pc) & sig & tin].sum():.1f} / entering "
                              f"{w[f_(fin_, path, v2, pc) & sig & ~tin].sum():.1f}" for nm, f_ in cells5[:2]))
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(args.plots, "path_tables.txt"), "w") as fh:
        fh.write(txt + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # truth vs reco backward wall distance on the selected vertex-pass-FC signal
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    m = vpf & sig
    r_, t_ = dfw[m], tdfw[m]
    r_ = np.where(np.isfinite(r_), np.clip(r_, 0, 499.9), -5.0)   # NaN (start outside box) -> -5
    t_ = np.where(np.isfinite(t_), np.clip(t_, 0, 499.9), -5.0)
    eb = np.linspace(-10, 500, 52)
    ax[0].hist2d(t_, r_, bins=[eb, eb], weights=w[m], cmap="viridis", cmin=1e-3)
    ax[0].axhline(args.dfw_cut, color="r", ls="--"); ax[0].axvline(args.dfw_cut, color="r", ls="--")
    ax[0].set_xlabel("TRUTH backward wall distance [cm]"); ax[0].set_ylabel("reco cand_dist_from_wall [cm]")
    ax[0].set_title(f"selected vertex/veto-pass/FC signal (w), agreement {agree:.2f}")
    for c in range(NSIG):
        mm = (cat == c)
        ax[1].hist(np.where(np.isfinite(tdfw[mm]), np.clip(tdfw[mm], 0, 499.9), -5.0), bins=eb,
                   weights=w[mm], histtype="step", lw=1.5, color=SPS.CAT_COLORS[c],
                   label=f"{CATS[c]} ({w[mm & tin].sum()/max(w[mm].sum(),1e-9):.2f} in-TPC)")
    ax[1].axvline(args.dfw_cut, color="r", ls="--")
    ax[1].set_xlabel("TRUTH backward wall distance [cm]  (-5 = undefined)")
    ax[1].set_ylabel(f"true events / {args.pot:.1e} POT"); ax[1].legend(fontsize=8)
    ax[1].set_title("all true signal events (denominators)")
    fig.tight_layout(); fig.savefig(os.path.join(args.plots, "dfw_truth_vs_reco.png"), dpi=110)
    plt.close(fig)
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
