#!/usr/bin/env python3
"""Expected single-photon signal under the WireCell photon-like LEE definition
(arXiv:2502.06064), from the run-1 BNB nu overlay ntuples, scaled to --pot.

Signal (paper, sec. II):
  * exactly ONE "reconstructable" photon-initiated EM shower: a photon with
    TRUE energy > 20 MeV (--e-min) whose shower START (first energy deposit,
    trueSimPartEDep* = mcreco MCShower DetProfile) is at least 3 cm (--dwall-min)
    inside the TPC boundary;
  * two reconstructable photons from the SAME vertex (identical start point,
    e.g. a pi0 decay) with opening angle < 20 deg (--open-angle) count as ONE
    shower (MiniBooNE ring-resolution proxy);
  * any number of protons / charged pions (no restriction);
  * no muon with KE >= 100 MeV (--mu-ke-max); softer muons allowed;
  * no PRIMARY electron-initiated shower (any GENIE-primary e+-).
NO requirement on the true neutrino vertex: entering photons from
out-of-TPC vertices are signal as long as the shower starts >= 3 cm inside.

PRIMARY tag (--prim-dist): a photon is "primary" when it comes from the
neutrino vertex either directly (GENIE photon) or through the immediate decay
of a primary pi0/eta, i.e. its true START point is within --prim-dist cm of
the true neutrino vertex (pi0 c*tau = 25 nm). Photons from secondary pi0s
made in hadronic re-interactions (in the dirt / cryostat / argon) start
elsewhere and are NOT primary. A bookkeeping variant (Process == primary, or
Process == Decay with a mother that is itself a primary or is absent from the
trueSimPart table) is printed for cross-check. NOTE: a primary pi0 is absent
from mcreco's tables when the vertex is inside the TPC but IS present
(Process == primary, at the vertex) when the vertex is outside, so "mother
absent" alone is not a primary criterion. Two primary
selections are reported: "nominal & primary" (the single shower is primary)
and "primary-only counting" (only primary photons are counted as showers;
secondary photons are ignored like any other X).

The muon and electron vetoes are evaluated over ALL nu-origin simulated
particles (trueSimPart: primaries + secondaries), matching how the present
analysis treats X; "primary-only" variants are printed as well.

Prints the expected signal at --pot, split by true nu vertex INSIDE / OUTSIDE
the TPC active box (and, for reference, inside / outside the WireCell FV flag
used by the present analysis), the sensitivity to each ingredient, and the
overlap with the present analysis categories (single_photon_selection.CATS).

    python3 single_photon_wc_lee_definition.py --mc-ntuple a.root,b.root \
        --pot 4.4e19 --plots plots_run1_novtx_dep_pe3/wc_lee_definition
"""
import argparse
import os
import sys

import numpy as np
import awkward as ak

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import single_photon_selection as SPS            # noqa: E402

TPC_LO, TPC_HI = SPS.TPC_LO, SPS.TPC_HI


def dwall(xyz):
    """Signed distance of (n,3) points to the TPC box (negative = outside)."""
    return np.minimum((xyz - TPC_LO).min(1), (TPC_HI - xyz).min(1))


def in_box(xyz):
    return ((xyz >= TPC_LO) & (xyz <= TPC_HI)).all(1)


count_showers = SPS.count_showers      # shared with single_photon_selection


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mc-ntuple", required=True)
    ap.add_argument("--pot", type=float, default=4.4e19)
    ap.add_argument("--e-min", type=float, default=20.0, help="photon TRUE energy [MeV]")
    ap.add_argument("--dwall-min", type=float, default=3.0,
                    help="shower start distance inside the TPC boundary [cm]")
    ap.add_argument("--open-angle", type=float, default=20.0,
                    help="same-vertex photon pairs below this angle [deg] count as one shower")
    ap.add_argument("--mu-ke-max", type=float, default=100.0,
                    help="muons with KE >= this [MeV] exclude the event")
    ap.add_argument("--paper-pot", type=float, default=6.34e20,
                    help="POT of the paper's data set (Table III), for the scaled comparison")
    ap.add_argument("--paper-selected", type=float, default=247.0,
                    help="paper's selected 1g signal events (Table III)")
    ap.add_argument("--paper-eff", type=float, default=0.070,
                    help="paper's quoted single-photon signal efficiency (Table I)")
    ap.add_argument("--paper-outfv-frac", type=float, default=0.295,
                    help="paper's 'Out of FV 1g' share of the selected signal")
    ap.add_argument("--prim-dist", type=float, default=1.0,
                    help="photon start within this [cm] of the true nu vertex = primary")
    ap.add_argument("--plots", default=None, help="output folder (text + plots)")
    args = ap.parse_args()

    a, pot = SPS.load(args.mc_ntuple, True, False,
                      extra=("trueVtxX", "trueVtxY", "trueVtxZ"))
    n = len(a["run"])
    w0 = ak.to_numpy(a["xsecWeight"]).astype(np.float64)
    w = np.where(w0 > 0, w0, 0.0) * (args.pot / pot)
    print(f">>> {n} events, POT {pot:.4e} -> scaled to {args.pot:.2e} (x {args.pot / pot:.5f})")

    pdg = a["trueSimPartPDG"]; apdg = np.abs(pdg)
    proc = a["trueSimPartProcess"]
    E = a["trueSimPartE"]
    ke = E - SPS._mass_of(pdg)
    q = a["trueSimPartPixelSumQ"]

    # ---- reconstructable photons ------------------------------------------
    is_g = apdg == 22
    ed = np.stack([ak.to_numpy(ak.flatten(a["trueSimPartEDep" + c])) for c in "XYZ"], 1)
    dw_flat = dwall(ed)
    dw = ak.unflatten(dw_flat, ak.num(pdg))
    reco_g = is_g & (E > args.e_min) & (dw >= args.dwall_min)
    reco_g_q = reco_g & (q > 0)                    # variant: also deposits labelled charge
    n_rg = ak.to_numpy(ak.sum(reco_g, axis=1)).astype(np.int64)
    n_rg_q = ak.to_numpy(ak.sum(reco_g_q, axis=1)).astype(np.int64)

    # ---- PRIMARY tag: photon start at the true nu vertex ----------------------
    vx = np.stack([np.asarray(ak.to_numpy(a["trueVtx" + c]), np.float64) for c in "XYZ"], 1)
    cnt = ak.to_numpy(ak.num(pdg))
    st_flat = np.stack([ak.to_numpy(ak.flatten(a["trueSimPart" + c])) for c in "XYZ"], 1)
    dvtx_flat = np.linalg.norm(st_flat - np.repeat(vx, cnt, axis=0), axis=1)
    prim_g = ak.unflatten(dvtx_flat < args.prim_dist, cnt)
    reco_g_prim = reco_g & prim_g
    n_rg_prim = ak.to_numpy(ak.sum(reco_g_prim, axis=1)).astype(np.int64)
    # bookkeeping variant: Process==primary, or Decay with the mother either a
    # primary itself or absent from the table (a primary pi0 is absent from the
    # mcreco tables for in-TPC vertices but present for out-of-TPC vertices)
    tids, mids = a["trueSimPartTID"], a["trueSimPartMID"]
    mother_ok = np.zeros(len(dvtx_flat), bool)          # mother absent or primary
    off = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    for i in np.nonzero(n_rg >= 1)[0]:
        ti = np.asarray(tids[i]); pr = np.asarray(proc[i])
        look = dict(zip(ti.tolist(), pr.tolist()))
        mi = np.asarray(mids[i])
        mother_ok[off[i]:off[i] + cnt[i]] = [look.get(int(m), 0) == 0 for m in mi]
    mother_ok = ak.unflatten(mother_ok, cnt)
    prim_g_book = (proc == 0) | ((proc == 1) & mother_ok)
    n_rg_prim_book = ak.to_numpy(ak.sum(reco_g & prim_g_book, axis=1)).astype(np.int64)
    # mother species of NON-primary reconstructable photons (for the breakdown)
    mpdg_flat = np.zeros(len(dvtx_flat), np.int64)
    for i in np.nonzero(n_rg >= 1)[0]:
        ti = np.asarray(tids[i]); pi = np.asarray(pdg[i]); mi = np.asarray(mids[i])
        look = dict(zip(ti.tolist(), pi.tolist()))
        mpdg_flat[off[i]:off[i] + cnt[i]] = [look.get(int(m), 0) for m in mi]
    mpdg = ak.unflatten(mpdg_flat, cnt)

    # merged shower count (same start + opening angle < open_angle)
    n_sh = n_rg.copy()
    multi = np.nonzero(n_rg >= 2)[0]
    st_all = [a["trueSimPart" + c] for c in "XYZ"]
    p_all = [a["trueSimPartP" + c] for c in "xyz"]
    n_merged_pairs = 0
    for i in multi:
        m = np.asarray(reco_g[i])
        st = np.stack([np.asarray(x[i])[m] for x in st_all], 1)
        pv = np.stack([np.asarray(x[i])[m] for x in p_all], 1)
        pn = np.linalg.norm(pv, axis=1, keepdims=True)
        dr = pv / np.maximum(pn, 1e-12)
        n_sh[i] = count_showers(st, dr, args.open_angle)
        n_merged_pairs += int(n_rg[i] - n_sh[i])
    n_sh_q = n_rg_q.copy()                         # merging under the charge variant
    for i in np.nonzero(n_rg_q >= 2)[0]:
        m = np.asarray(reco_g_q[i])
        st = np.stack([np.asarray(x[i])[m] for x in st_all], 1)
        pv = np.stack([np.asarray(x[i])[m] for x in p_all], 1)
        dr = pv / np.maximum(np.linalg.norm(pv, axis=1, keepdims=True), 1e-12)
        n_sh_q[i] = count_showers(st, dr, args.open_angle)

    n_sh_prim = n_rg_prim.copy()                   # merging among PRIMARY photons only
    for i in np.nonzero(n_rg_prim >= 2)[0]:
        m = np.asarray(reco_g_prim[i])
        st = np.stack([np.asarray(x[i])[m] for x in st_all], 1)
        pv = np.stack([np.asarray(x[i])[m] for x in p_all], 1)
        dr = pv / np.maximum(np.linalg.norm(pv, axis=1, keepdims=True), 1e-12)
        n_sh_prim[i] = count_showers(st, dr, args.open_angle)

    # ---- vetoes ---------------------------------------------------------------
    mu_hard = (apdg == 13) & (ke >= args.mu_ke_max)
    has_mu = ak.to_numpy(ak.any(mu_hard, axis=1))
    has_mu_prim = ak.to_numpy(ak.any(mu_hard & (proc == 0), axis=1))
    has_prim_e = ak.to_numpy(ak.any((apdg == 11) & (proc == 0), axis=1))
    has_prim_e20 = ak.to_numpy(ak.any((apdg == 11) & (proc == 0) & (E > args.e_min), axis=1))

    sig = (n_sh == 1) & ~has_mu & ~has_prim_e
    sig_nomerge = (n_rg == 1) & ~has_mu & ~has_prim_e
    sig_q = (n_sh_q == 1) & ~has_mu & ~has_prim_e
    sig_muprim = (n_sh == 1) & ~has_mu_prim & ~has_prim_e
    sig_e20 = (n_sh == 1) & ~has_mu & ~has_prim_e20
    # PRIMARY variants: the single shower is primary / count primary photons only
    sig_prim = sig & (n_sh_prim == 1)
    sig_primonly = (n_sh_prim == 1) & ~has_mu & ~has_prim_e
    sig_prim_book = sig & (n_rg_prim_book == 1)
    vin = in_box(vx)
    wcfv = ak.to_numpy(a["trueVtxInWCFV"]) == 1
    ccnc = ak.to_numpy(a["trueNuCCNC"]).astype(np.int64)

    def W(m):
        return w[m].sum()

    L = []
    L.append(f"WireCell photon-like LEE signal definition (arXiv:2502.06064) on {args.mc_ntuple}")
    L.append(f"  photon true E > {args.e_min:.0f} MeV, shower start >= {args.dwall_min:.0f} cm inside the TPC box, "
             f"same-vertex pairs < {args.open_angle:.0f} deg merged, no muon KE >= {args.mu_ke_max:.0f} MeV, "
             f"no primary e+-; NO vertex requirement")
    L.append(f"  weights: xsecWeight x {args.pot:.2e} / {pot:.4e} POT; {n} MC events, "
             f"{int(sig.sum())} raw signal events")
    L.append("")
    L.append(f"EXPECTED SIGNAL at {args.pot:.2e} POT: {W(sig):.1f}")
    L.append(f"  true nu vertex INSIDE the TPC active box : {W(sig & vin):.1f}  ({W(sig & vin) / max(W(sig), 1e-9):.3f})")
    L.append(f"  true nu vertex OUTSIDE the TPC active box: {W(sig & ~vin):.1f}  ({W(sig & ~vin) / max(W(sig), 1e-9):.3f})")
    L.append(f"  (inside the WireCell FV flag / outside: {W(sig & wcfv):.1f} / {W(sig & ~wcfv):.1f})")
    L.append(f"  CC / NC: {W(sig & (ccnc == 0)):.1f} / {W(sig & (ccnc == 1)):.1f}"
             f"   inside-TPC CC / NC: {W(sig & vin & (ccnc == 0)):.1f} / {W(sig & vin & (ccnc == 1)):.1f}")
    L.append("")
    L.append("SENSITIVITY to the ingredients (total | inside TPC | outside TPC):")
    for lab, m in (("nominal", sig),
                   ("no opening-angle merging (n reconstructable photons == 1)", sig_nomerge),
                   ("photon must also deposit labelled charge (PixelSumQ > 0)", sig_q),
                   ("muon veto on PRIMARY muons only", sig_muprim),
                   (f"electron veto only for primary e with E > {args.e_min:.0f} MeV", sig_e20)):
        L.append(f"  {lab:64s} {W(m):7.1f} | {W(m & vin):7.1f} | {W(m & ~vin):7.1f}")
    L.append(f"  events where merging reduced the shower count: {W((n_rg >= 2) & (n_sh < n_rg)):.1f} w "
             f"({n_merged_pairs} raw merged pairs); of these, signal after merging: "
             f"{W(sig & (n_sh < n_rg)):.1f}")
    L.append(f"  shower multiplicity (all events, w): " + ", ".join(
        f"{k}: {W(n_sh == k):.1f}" for k in range(4)) + f", >=4: {W(n_sh >= 4):.1f}")
    L.append(f"  vetoes among n_shower==1 events: mu>= {args.mu_ke_max:.0f}: {W((n_sh == 1) & has_mu):.1f}, "
             f"primary e: {W((n_sh == 1) & has_prim_e):.1f}, both: {W((n_sh == 1) & has_mu & has_prim_e):.1f}")

    # ---- PRIMARY tag ---------------------------------------------------------
    L.append("")
    L.append(f"PRIMARY TAG: photon start within {args.prim_dist:.1f} cm of the true nu vertex "
             f"(direct GENIE photon or immediate primary pi0/eta decay)")
    L.append(f"  {'variant':64s} {'total':>7} | {'inside':>7} | {'outside':>7} | {'CC':>7} | {'NC':>7}")
    for lab, m in (("nominal", sig),
                   ("nominal & the single shower is PRIMARY", sig_prim),
                   ("PRIMARY-only counting (secondary photons ignored as X)", sig_primonly),
                   ("nominal & primary by bookkeeping (Process / mother primary or absent)", sig_prim_book)):
        L.append(f"  {lab:64s} {W(m):7.1f} | {W(m & vin):7.1f} | {W(m & ~vin):7.1f} | "
                 f"{W(m & (ccnc == 0)):7.1f} | {W(m & (ccnc == 1)):7.1f}")
    L.append(f"  primary fraction of the nominal signal: all {W(sig_prim) / max(W(sig), 1e-9):.3f}, "
             f"inside TPC {W(sig_prim & vin) / max(W(sig & vin), 1e-9):.3f}, "
             f"outside TPC {W(sig_prim & ~vin) / max(W(sig & ~vin), 1e-9):.3f}")
    L.append(f"  geometric vs bookkeeping primary tag agreement on nominal signal: "
             f"{W(sig & ((n_sh_prim == 1) == (n_rg_prim_book == 1))) / max(W(sig), 1e-9):.3f}")
    nonp = sig & (n_sh_prim == 0)
    mp1 = np.asarray(ak.to_numpy(ak.fill_none(ak.firsts(mpdg[reco_g]), 0)), np.int64)
    dv1 = np.asarray(ak.to_numpy(ak.fill_none(ak.firsts(ak.unflatten(dvtx_flat, cnt)[reco_g]), np.nan)), np.float64)
    parts = []
    for lab_, sel_ in (("secondary pi0", np.abs(mp1) == 111), ("eta", np.abs(mp1) == 221),
                       ("neutron", np.abs(mp1) == 2112), ("proton", np.abs(mp1) == 2212),
                       ("charged pi", np.abs(mp1) == 211), ("photon/e (brem/compton)", np.isin(np.abs(mp1), [22, 11])),
                       ("mother not in table", mp1 == 0)):
        if W(nonp & sel_) > 0:
            parts.append(f"{lab_} {W(nonp & sel_):.1f}")
    other_ = nonp & ~np.isin(np.abs(mp1), [111, 221, 2112, 2212, 211, 22, 11, 0])
    if W(other_) > 0:
        parts.append(f"other {W(other_):.1f}")
    L.append(f"  NON-primary nominal signal ({W(nonp):.1f}) by mother of the shower photon: " + ", ".join(parts))
    L.append(f"  distance of the shower start from the nu vertex, non-primary nominal signal (w-unweighted): "
             f"median {np.nanmedian(dv1[nonp]) if nonp.any() else np.nan:.0f} cm, "
             f"16-84% {np.nanpercentile(dv1[nonp], 16) if nonp.any() else np.nan:.0f}-"
             f"{np.nanpercentile(dv1[nonp], 84) if nonp.any() else np.nan:.0f} cm")
    L.append(f"  nominal signal events with >=1 reconstructable NON-primary photon anywhere: "
             f"{W(sig & (n_rg_prim < n_rg)):.1f}; primary-only-counting events that are NOT nominal signal "
             f"(extra secondary photon(s)): {W(sig_primonly & ~sig):.1f}")

    # ---- the paper's five signal categories + implied-rate comparison --------
    vdw = dwall(vx)                                   # true vertex distance to the TPC box (neg = outside)
    vfv3 = vdw >= args.dwall_min                      # paper FV: 3 cm inside the TPC boundary
    mp1_all = np.asarray(ak.to_numpy(ak.fill_none(ak.firsts(mpdg[reco_g]), 0)), np.int64)
    pr1 = np.asarray(ak.to_numpy(ak.fill_none(ak.firsts(proc[reco_g]), -1)), np.int64)
    from_pi0 = (pr1 == 1) & ((np.abs(mp1_all) == 111) | (mp1_all == 0))   # Decay, mother pi0 or pi0 dropped from table
    direct = pr1 == 0                                                       # GENIE photon (Delta radiative, ...)
    nc = ccnc == 1
    PCATS = [("out of FV 1g (vertex < 3 cm inside / outside)", sig & ~vfv3),
             ("NC pi0 1g", sig & vfv3 & nc & from_pi0),
             ("NC Delta / direct-photon 1g", sig & vfv3 & nc & direct),
             ("NC other 1g", sig & vfv3 & nc & ~from_pi0 & ~direct),
             ("numu CC 1g (mu KE < 100)", sig & vfv3 & ~nc)]
    sc = args.paper_pot / args.pot
    L.append("")
    L.append(f"PAPER CATEGORIES (arXiv:2502.06064 sec. II) at {args.pot:.2e} POT and scaled to {args.paper_pot:.3e} POT:")
    for lab, m in PCATS:
        L.append(f"  {lab:50s} {W(m):8.1f}   {W(m) * sc:9.1f}")
    infv_tot = W(sig & vfv3); tot = W(sig)
    L.append(f"  {'in-FV total (4 categories)':50s} {infv_tot:8.1f}   {infv_tot * sc:9.1f}")
    L.append(f"  {'all 5 categories':50s} {tot:8.1f}   {tot * sc:9.1f}")
    L.append(f"  IMPLIED by the paper: selected {args.paper_selected:.0f} / eff {args.paper_eff:.3f} = "
             f"{args.paper_selected / args.paper_eff:.0f} signal at {args.paper_pot:.3e} POT = "
             f"{args.paper_selected / args.paper_eff / sc:.1f} at {args.pot:.2e}")
    infv_sel = args.paper_selected * (1 - args.paper_outfv_frac)
    L.append(f"    (a) if the efficiency denominator is ALL 5 categories: ntuple predicts {tot * sc:.0f} "
             f"-> ratio ntuple/paper {tot * sc / (args.paper_selected / args.paper_eff):.2f}")
    L.append(f"    (b) if the denominator is the in-FV signal only (WC convention) and the {args.paper_outfv_frac:.1%} "
             f"out-of-FV selected share ({infv_sel:.0f} in-FV selected) is additive: implied in-FV signal "
             f"{infv_sel / args.paper_eff:.0f} at {args.paper_pot:.3e} = {infv_sel / args.paper_eff / sc:.1f} at {args.pot:.2e}; "
             f"ntuple in-FV {infv_tot:.1f} -> ratio {infv_tot / (infv_sel / args.paper_eff / sc):.2f}; "
             f"implied out-of-FV efficiency {args.paper_selected * args.paper_outfv_frac / max(W(sig & ~vfv3) * sc, 1e-9):.4f}")
    L.append(f"  selected-signal composition in the paper vs ntuple PRE-selection in-FV shares: "
             + ", ".join(f"{lab.split(' (')[0]} {W(m) / max(infv_tot, 1e-9):.3f}" for lab, m in PCATS[1:]))

    # ---- overlap with the present analysis categories ----------------------
    T = SPS.truth_tables(a)
    cat = T["cat"]; ours = T["sig_incl"]
    L.append("")
    L.append("OVERLAP with the present analysis definition (single_photon_selection: exactly one photon with "
             f"DEPOSITED E >= {SPS.EVIS_MIN:.0f} MeV, no e >= 20 / mu KE >= 100 depositing; FV = WC flag):")
    L.append(f"  present-analysis signal (cats 0-3): {W(ours):.1f}   WC-definition signal: {W(sig):.1f}   "
             f"both: {W(ours & sig):.1f}   WC only: {W(sig & ~ours):.1f}   present only: {W(ours & ~sig):.1f}")
    L.append(f"  {'present category':40s}{'N(w)':>8}{'WC sig':>8}{'frac':>7}{'WC&prim':>9}{'frac':>7}")
    for c in range(SPS.N_MC_CAT):
        m = cat == c
        if W(m) > 0:
            L.append(f"  {SPS.CATS[c][:40]:40s}{W(m):8.1f}{W(m & sig):8.1f}{W(m & sig) / W(m):7.3f}"
                     f"{W(m & sig_prim):9.1f}{W(m & sig_prim) / W(m):7.3f}")
    L.append(f"  present-analysis signal that is PRIMARY (same geometric tag on its visible photon): "
             + ", ".join(f"{SPS.CATS[c][4:]} {W((cat == c) & (n_rg_prim >= 1)) / max(W(cat == c), 1e-9):.3f}"
                         for c in range(SPS.N_SIG_CAT)))
    # why WC-only events are not present-analysis signal
    wc_only = sig & ~ours
    L.append(f"  WC-only events ({W(wc_only):.1f}) by present category: " + ", ".join(
        f"{SPS.CATS[c][5:]} {W(wc_only & (cat == c)):.1f}" for c in range(SPS.N_MC_CAT)
        if W(wc_only & (cat == c)) > 0))
    # deposited energy of the WC photon in WC-only events
    evis = SPS.A_GAMMA * np.maximum(q, 0.0)
    g1 = ak.fill_none(ak.firsts(evis[reco_g]), np.nan)
    g1 = np.asarray(ak.to_numpy(g1), np.float64)
    g1E = np.asarray(ak.to_numpy(ak.fill_none(ak.firsts(E[reco_g]), np.nan)), np.float64)
    m = wc_only & (n_rg == 1)
    L.append(f"  WC-only single-photon events: photon deposited E < {SPS.EVIS_MIN:.0f} MeV: "
             f"{W(m & (g1 < SPS.EVIS_MIN)):.1f}, deposited >= {SPS.EVIS_MIN:.0f}: {W(m & (g1 >= SPS.EVIS_MIN)):.1f}")
    po = ours & ~sig
    L.append(f"  present-only events ({W(po):.1f}): shower start < {args.dwall_min:.0f} cm or true E <= {args.e_min:.0f}: "
             f"{W(po & (n_rg == 0)):.1f}, >=2 reconstructable photons: {W(po & (n_rg >= 2)):.1f}, "
             f"mu/e veto: {W(po & (n_sh == 1) & (has_mu | has_prim_e)):.1f}")

    txt = "\n".join(L)
    print(txt)
    if args.plots:
        os.makedirs(args.plots, exist_ok=True)
        with open(os.path.join(args.plots, "wc_lee_definition.txt"), "w") as fh:
            fh.write(txt + "\n")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        bins = np.linspace(0, 1000, 41)
        for key, vals, xlab in (("trueE", g1E, "reconstructable photon true E [MeV]"),
                                ("depE", g1, "reconstructable photon deposited E [MeV] (a_gamma x Q)")):
            fig, ax = plt.subplots(figsize=(7.2, 4.6))
            m = sig & np.isfinite(vals)
            ax.hist([np.clip(vals[m & vin], 0, 999.9), np.clip(vals[m & ~vin], 0, 999.9)], bins=bins,
                    weights=[w[m & vin], w[m & ~vin]], stacked=True, color=["#d62728", "#ff7f0e"],
                    label=[f"vertex inside TPC ({W(m & vin):.1f})", f"vertex outside TPC ({W(m & ~vin):.1f})"])
            ax.set(xlabel=xlab, ylabel=f"events / {args.pot:.1e} POT",
                   title="WC photon-like LEE signal definition (arXiv:2502.06064)")
            ax.legend(); ax.grid(alpha=0.3)
            fig.tight_layout(); fig.savefig(os.path.join(args.plots, f"wc_signal_{key}.png"), dpi=110)
            plt.close(fig)
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        m = sig & np.isfinite(g1E)
        pr = n_sh_prim == 1
        groups = [(m & vin & pr, "vertex inside TPC, primary", "#d62728"),
                  (m & vin & ~pr, "vertex inside TPC, secondary", "#f4a3a3"),
                  (m & ~vin & pr, "vertex outside TPC, primary", "#ff7f0e"),
                  (m & ~vin & ~pr, "vertex outside TPC, secondary", "#ffd0a0")]
        ax.hist([np.clip(g1E[g], 0, 999.9) for g, _, _ in groups], bins=bins,
                weights=[w[g] for g, _, _ in groups], stacked=True, color=[c for _, _, c in groups],
                label=[f"{l} ({W(g):.1f})" for g, l, _ in groups])
        ax.set(xlabel="reconstructable photon true E [MeV]", ylabel=f"events / {args.pot:.1e} POT",
               title="WC photon-like LEE signal, primary tag (start at nu vertex)")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(args.plots, "wc_signal_trueE_primary.png"), dpi=110)
        plt.close(fig)
        print(f">>> written to {args.plots}")


if __name__ == "__main__":
    main()
