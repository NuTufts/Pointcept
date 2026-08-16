"""Pilot-matrix comparison: {old,new chain} x {pred,true vertex} efficiency /
purity for (a) SBND-style CC 1pi0 (bnb-nu pilot) and (b) inclusive nue CC
(intrinsic-nue pilot signal + bnb-nu pilot nu-induced background).

Inputs: the 8 pilot ntuples written by the export pipeline
    .../pilot_ntuples/dlgen2_pilot_{old,new}_{bnbnu,nue}_{pred,true}.root
plus per-ntuple nue tables built beforehand with nue_cc_analysis.py
(--nue-table-dir, files named nuetab_<cell>.npz for BOTH samples' ntuples).

CC 1pi0: truth categories + reco selection are IMPORTED from
pi0mass_peak/sbnd_cc1pi0.py; only w and flash_chi2 come from the ntuple
directly (xsecWeight * POT scale; recoVtxFlashChi2[0] = the primary vertex's
chi2, valid whenever primaryVtxStream==0 because the exporter sorts vertices
nu-first score-desc and row 0 is the primary). Reported before/after the
CC working-point flash cut (chi2 < 1e4).

nue CC: applies the run_nuecc_cutflow.sh step-5 working point to the tables
(log10 flash < 3.0, LArPID primariness > 1.0, LArPID mu < -6.0, LArFormer
elconf > 5.0, LArFormer vtx-mu veto >= -4.0). Efficiency from the nue table
(signal = is_nuecc_fv); purity from nue signal + bnb-nu nu-induced background,
both POT-scaled to --pot. EXT cosmics are NOT included (the EXT sample was not
reprocessed with the new chain), so purity here is MC-only — comparable across
cells, not absolute.

CAVEAT for the writeup: true-vertex cells measure selection performance
conditional on a perfect nu-vertex — they bound, rather than equal, deployable
performance until stage-4 retrains.

    PYTHONPATH=./ python3 compare_pilot_matrix.py \
        --ntuple-dir .../pilot_ntuples --nue-table-dir .../pilot_ntuples/nue_tables \
        --out summary_pilot_matrix.txt
"""
import argparse
import os
import sys

import numpy as np
import uproot
import awkward as ak

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "pi0mass_peak"))
from sbnd_cc1pi0 import (_BR, _BR_MC, _in_fv, sbnd_truth_cat,   # noqa: E402
                         GAMMA_RECO_MIN, MU_KE_MIN, CPI_KE_MIN, MGG_MAX)

CELLS = [f"{c}_{s}_{m}" for c in ("old", "new")
         for s in ("bnbnu", "nue") for m in ("pred", "true")]


def _pot_scale(fin, pot_target):
    pot = fin["potTree"].arrays(library="np")
    pot_sum = float(np.sum(pot["totGoodPOT"])) or float(np.sum(pot["totPOT"]))
    return pot_target / pot_sum, pot_sum


def _wfrac(w, num, den):
    a, b = float(w[num].sum()), float(w[den].sum())
    return (a / b if b > 0 else np.nan), int(num.sum()), int(den.sum())


def pi0_cell(ntuple, pot_target, flash_cut):
    """CC-1pi0 efficiency/purity on one bnb-nu pilot ntuple (MC only)."""
    fin = uproot.open(ntuple)
    scale, pot_sum = _pot_scale(fin, pot_target)
    t = fin["EventTree"]
    have = set(t.keys())
    br = list(_BR) + list(_BR_MC) + ["xsecWeight", "recoVtxFlashChi2"]
    a = t.arrays([b for b in br if b in have])
    n = len(a["run"])
    w0 = np.asarray(a["xsecWeight"], np.float64)
    w = np.where(w0 > 0, w0, 0.0) * scale
    fchi2 = ak.to_numpy(ak.firsts(a["recoVtxFlashChi2"], axis=1))
    fchi2 = np.asarray(fchi2, np.float64)

    # ---- reco selection (verbatim logic from sbnd_cc1pi0.load) -------------
    vtx_ok = ((np.asarray(a["foundVertex"]) == 1)
              & (np.asarray(a["primaryVtxStream"]) == 0)
              & _in_fv(np.asarray(a["vtxX"]), np.asarray(a["vtxY"]),
                       np.asarray(a["vtxZ"])))
    conf = a["showerAttConfident"] != 0
    is_g = ((a["showerLArFormerPID"] == 22)
            & (a["showerRecoE"] > GAMMA_RECO_MIN) & conf)
    n_g = ak.to_numpy(ak.sum(is_g, axis=1))
    is_mu = ((a["trackLArFormerPID"] == 13) & (a["trackIsSecondary"] == 0)
             & (a["trackRecoE"] > MU_KE_MIN))
    has_mu = ak.to_numpy(ak.any(is_mu, axis=1))
    is_cpi = ((a["trackLArFormerPID"] == 211) & (a["trackIsSecondary"] == 0)
              & (a["trackRecoE"] > CPI_KE_MIN))
    n_cpi = ak.to_numpy(ak.sum(is_cpi, axis=1))

    m_gg = np.full(n, np.nan)
    for i in np.nonzero(vtx_ok & (n_g >= 2))[0]:
        gi = np.nonzero(ak.to_numpy(is_g[i]))[0]
        E = ak.to_numpy(a["showerRecoE"][i])[gi]
        o = gi[np.argsort(E)[::-1][:2]]
        E1, E2 = ak.to_numpy(a["showerRecoE"][i])[o].tolist()
        v = np.array([a["vtxX"][i], a["vtxY"][i], a["vtxZ"][i]], np.float64)
        sp = np.stack([ak.to_numpy(a[f"showerStartPos{c}"][i])[o]
                       for c in "XYZ"], 1).astype(np.float64)
        d = sp - v
        nn = np.linalg.norm(d, axis=1)
        if np.all(nn > 1e-3):
            d = d / nn[:, None]
            m_gg[i] = float(np.sqrt(max(2.0 * E1 * E2 * (1.0 - d[0] @ d[1]),
                                        0.0)))
    sel = (vtx_ok & (n_g >= 2) & has_mu & (n_cpi == 0)
           & np.isfinite(m_gg) & (m_gg < MGG_MAX))

    cat = np.zeros(n, np.int64)
    for i in range(n):
        cat[i], _ = sbnd_truth_cat(a, i)
    sig = cat == 0

    out = dict(n_events=n, pot=pot_sum, n_sig_raw=int(sig.sum()),
               w_sig=float(w[sig].sum()))
    for tag, m in (("preflash", sel),
                   ("flash", sel & np.isfinite(fchi2) & (fchi2 < flash_cut))):
        eff, nnum, nden = _wfrac(w, m & sig, sig)
        pur, _, nsel = _wfrac(w, m & sig, m)
        out[tag] = dict(eff=eff, pur=pur, n_sel=nsel, n_sel_sig=nnum,
                        n_sig=nden)
    return out


# ---- nue CC step-5 working point (run_nuecc_cutflow.sh) ---------------------
NUE_FLASH_LOG10 = 3.0
NUE_PRIMARINESS = 1.0
NUE_MU_CUT = -6.0
NUE_ELCONF_LF = 5.0
NUE_VTXMU_LF = -4.0


def nue_wp_mask(tab, flash_only=False):
    m = tab["sel"].astype(bool) & np.isfinite(tab["reco_ele_E"])
    fc = tab["flash_chi2"]
    m &= np.isfinite(fc) & (fc > 0) & (np.log10(fc) < NUE_FLASH_LOG10)
    if flash_only:
        return m
    pr = tab["prim_score"] - np.maximum(tab["fromneut_score"],
                                        tab["fromchg_score"])
    m &= np.isfinite(pr) & (pr > NUE_PRIMARINESS)
    m &= np.isfinite(tab["mu_score"]) & (tab["mu_score"] < NUE_MU_CUT)
    ec = tab["lf_el_score"] - 0.5 * (tab["lf_pi_score"] + tab["lf_ph_score"])
    m &= np.isfinite(ec) & (ec > NUE_ELCONF_LF)
    vm = tab["vtx_lf_mu_score"]
    m &= ~(np.isfinite(vm) & (vm >= NUE_VTXMU_LF))
    return m


def nue_cell(nue_tab_path, bnb_tab_path):
    nue = dict(np.load(nue_tab_path))
    bnb = dict(np.load(bnb_tab_path))
    sig_all = nue["is_nuecc_fv"].astype(bool)
    out = dict(n_sig_raw=int(sig_all.sum()),
               w_sig=float(nue["w"][sig_all].sum()))
    for tag, kw in (("flashonly", dict(flash_only=True)),
                    ("wp", dict(flash_only=False))):
        mn = nue_wp_mask(nue, **kw)
        mb = nue_wp_mask(bnb, **kw) & ~bnb["is_nuecc"].astype(bool)
        eff, nnum, nden = _wfrac(nue["w"], mn & sig_all, sig_all)
        s = float(nue["w"][mn & sig_all].sum())
        b_sig_leak = float(nue["w"][mn & ~sig_all].sum())  # nue-sample non-sig
        b_nu = float(bnb["w"][mb].sum())
        pur = s / (s + b_sig_leak + b_nu) if (s + b_sig_leak + b_nu) > 0 \
            else np.nan
        out[tag] = dict(eff=eff, pur_mc=pur, n_sel_sig=nnum, n_sig=nden,
                        w_sel_sig=s, w_bkg_nue_sample=b_sig_leak,
                        w_bkg_bnb=b_nu, n_sel_bnb=int(mb.sum()))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ntuple-dir", required=True)
    ap.add_argument("--nue-table-dir", required=True,
                    help="dir with nuetab_<cell>.npz for all 8 cells")
    ap.add_argument("--pot", type=float, default=4.4e19)
    ap.add_argument("--pi0-flash-cut", type=float, default=1e4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    lines = []

    def emit(s=""):
        print(s, flush=True)
        lines.append(s)

    emit("=" * 78)
    emit("PILOT MATRIX: {old,new chain} x {pred,true vertex}")
    emit("true-vertex cells = selection performance conditional on a perfect")
    emit("nu vertex (bounds deployable performance until stage-4 retrains)")
    emit("=" * 78)

    emit("\n--- SBND-style CC 1pi0 (bnb-nu pilot, MC-only; no EXT) ---")
    emit(f"{'cell':22s} {'eff(pre)':>9s} {'eff(flash)':>10s} "
         f"{'pur(flash)':>10s} {'selN':>6s} {'selSigN':>8s} {'sigN':>6s}")
    for chain in ("old", "new"):
        for mode in ("pred", "true"):
            cell = f"{chain}_bnbnu_{mode}"
            p = os.path.join(args.ntuple_dir, f"dlgen2_pilot_{cell}.root")
            r = pi0_cell(p, args.pot, args.pi0_flash_cut)
            f = r["flash"]
            emit(f"{cell:22s} {r['preflash']['eff']:9.3f} {f['eff']:10.3f} "
                 f"{f['pur']:10.3f} {f['n_sel']:6d} {f['n_sel_sig']:8d} "
                 f"{f['n_sig']:6d}")

    emit("\n--- nue CC inclusive (nue pilot signal + bnb-nu pilot nu bkg; "
         "MC-only, no EXT) ---")
    emit(f"{'cell':22s} {'eff(flash)':>10s} {'eff(WP)':>8s} {'pur(WP)':>8s} "
         f"{'selSigN':>8s} {'sigN':>6s} {'w_bkg_bnb':>10s}")
    for chain in ("old", "new"):
        for mode in ("pred", "true"):
            cell = f"{chain}_nue_{mode}"
            nt = os.path.join(args.nue_table_dir, f"nuetab_{cell}.npz")
            bt = os.path.join(args.nue_table_dir,
                              f"nuetab_{chain}_bnbnu_{mode}.npz")
            r = nue_cell(nt, bt)
            emit(f"{cell:22s} {r['flashonly']['eff']:10.3f} "
                 f"{r['wp']['eff']:8.3f} {r['wp']['pur_mc']:8.3f} "
                 f"{r['wp']['n_sel_sig']:8d} {r['wp']['n_sig']:6d} "
                 f"{r['wp']['w_bkg_bnb']:10.2f}")

    if args.out:
        with open(args.out, "w") as fo:
            fo.write("\n".join(lines) + "\n")
        print(f"\n>>> wrote {args.out}")


if __name__ == "__main__":
    main()
