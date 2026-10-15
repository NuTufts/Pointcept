#!/usr/bin/env python3
"""MiniBooNE-targeted single-photon selection (MB_SELECTION_PLAN.md).

Starts from the shared selection (single_photon_selection: S1-S5, vertex path,
2nd-photon veto, FC, backward wall distance >= --dfw-cut = the "primary cell")
and adds PURITY-oriented, KINEMATICS-FREE steps for events MiniBooNE would have
seen as one EM ring (truth: --signal-def mb by default):

  M1 MB hadrons   : no mu-PID track with KE >= --mb-mu-kemin, no pi-PID track
                    with KE >= --mb-pi-kemin, no proton track with KE >=
                    --mb-p-kemax (ring-making in mineral oil); sub-threshold
                    mu / pi / p are allowed (MiniBooNE-blind).
  M2 extra energy : summed energy of every OTHER nu-stream prong that is not
                    an accepted proton or an allowed sub-threshold mu/pi (other
                    showers of any PID, unclassified tracks) <= --extra-emax
                    (second-shower completeness + stray activity; vertex-less
                    tracks have no range energy -> charge-based estimate).
  M3 tight FV     : 1gNp (>= 1 accepted proton with KE >= --proton-ke-min):
                    reco vertex dwall >= --tight-dwall; 1g0p: the vertex IS
                    the shower start, so candidate start dwall >= --tight-dwall
                    AND backward wall distance >= --dfw-tight.
  M4 conv. dist.  : 1gNp only, candidate start to vertex >= --conv-dist-min
                    (0 = off; N-1 plot only). An entering photon's "vertex" is
                    its conversion point (distance 0).
  M2b pi0 cone    : (--pi0-cone) veto when a SOFT cluster (any other nu-stream
                    shower prong, E <= --extra-emax, i.e. below the M2 veto)
                    lies in the pi0-decay cone of the candidate: with the
                    candidate energy E1 and the angle theta between the two
                    prong directions, the second photon would need
                    E2 = m_pi0^2 / (2 E1 (1 - cos theta)); tag if E2 is within
                    [--cone-e2-min, --cone-e2-max] and >= the cluster's own
                    (incomplete) energy. Topological veto conditioned on the
                    candidate, not a cut on its phase space.
  M5 beam-like    : candidate cosmic BDT >= --cand-bdt-min, flash PE ratio <
                    --mb-pe-ratio-max, log10 flash chi2 < --mb-logchi2-max.
  final split     : 1g0p / 1gNp.

Defaults = the 2026-10-01 scan optimum (tight FV 10 cm, no extra 0p dfw cut,
cosmic BDT >= 0.7): see MB_SELECTION_PLAN.md.

RULE: no cut uses the candidate photon's energy or direction. Energy, cosBeam
and E_QE (electron-CCQE hypothesis) are REPORTED for the final samples only.

Efficiency denominators (w): mb-truth signal with true vertex in the FV, and
with true vertex >= --tight-dwall inside the TPC (tight-FV truth); purity
counts all mb signal (in-FV + entering). N-1 plots for every M-step variable.

    python3 single_photon_mb_selection.py --mc-ntuple ... --ext-ntuple ... \
        --data-ntuple ... --novtx --plots plots_run1_mbsel
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

TPC_LO, TPC_HI = SPS.TPC_LO, SPS.TPC_HI
BR_EXTRA = ("recoVtxX", "recoVtxY", "recoVtxZ", "trackCharge", "showerCosTheta",
            "showerDistToVtx")
M_P, M_N, M_E, E_B = 938.272, 939.565, 0.511, 34.0


def dwall(xyz):
    return np.minimum((xyz - TPC_LO).min(1), (TPC_HI - xyz).min(1))


def pick(x, idx, n_ok, fill=np.nan):
    v = ak.fill_none(ak.firsts(x[idx]), fill)
    v = np.asarray(ak.to_numpy(v), np.float64)
    v[~n_ok] = np.nan
    return v


def e_qe(E, cos):
    """Electron-CCQE-hypothesis neutrino energy [MeV] from shower E and cos(theta_beam)."""
    Mn = M_N - E_B
    p = np.sqrt(np.maximum(E ** 2 - M_E ** 2, 0.0))
    num = Mn * E - 0.5 * (Mn ** 2 + M_E ** 2 - M_P ** 2)
    den = Mn - E + p * cos
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / den, np.nan)


def mb_tables(a, R, args, names, S, E_cand, path_cand):
    """Per-event MB-selection variables (numpy) for one sample."""
    n = len(E_cand)
    idx = R["_cand_idx"]
    has = np.isfinite(E_cand)
    out = {"E": E_cand, "path": path_cand}
    # ---- primary cell ------------------------------------------------------
    og = R.get("other_gE_max_old", R["other_gE_max"])   # BDT-chain feature: the old (any-prong) quantity
    out["other_gE_max"] = og
    out["veto2g_fail"] = np.nan_to_num(og, nan=-1.0) > args.veto2g_emin
    out["pc"] = np.nan_to_num(R["track_min_dwall"], nan=1e9) < args.contain_dwall
    out["dfw"] = R["cand_dist_from_wall"]
    out["base"] = S[-1]
    out["base_S3"] = S[SPS.step_index(names, "no vis X")]           # pre-selection before the flash cuts
    out["dtw"] = R["cand_dist_to_wall"]
    out["cand_score"] = pick(R["_score"], idx, has, -9.0)            # unified selection score
    out["cand_novtx"] = pick(a["showerNoVtxScore"], idx, has, -9.0) if "showerNoVtxScore" in ak.fields(a) else np.full(n, np.nan)
    out["track_min_dwall"] = R["track_min_dwall"]
    out["vtx_ok"] = R["vtx_ok"].astype(np.float64)
    out["cell"] = (S[-1] & (path_cand == 0) & ~out["veto2g_fail"] & ~out["pc"]
                   & (np.nan_to_num(out["dfw"], nan=-1.0) >= args.dfw_cut))
    # ---- tracks: PID / KE --------------------------------------------------
    tpid = a["trackLArFormerPID"]
    seg_mu = tpid == 13
    lp_mu = ((a["trackClassified"] == 1) & (a["trackMuScore"] > a["trackElScore"])
             & (a["trackMuScore"] > a["trackPhScore"]) & (a["trackMuScore"] > a["trackPiScore"])
             & (a["trackMuScore"] > a["trackPrScore"]))
    mu_pid = {"segmenter": seg_mu, "larpid": lp_mu, "union": seg_mu | lp_mu}[args.muon_finder]
    nu_t = (a["trackStream"] == 0) if args.novtx else ak.ones_like(tpid, dtype=bool)
    ke = a["trackRecoE"]
    ke_q = SPS.NTUP_G_A * a["trackCharge"] + SPS.NTUP_G_B      # vertex-less tracks: charge-based
    ke_any = ak.where(ke >= 0, ke, ak.where(ke_q > 0, ke_q, 0.0))
    pi_pid = (tpid == 211) & ~mu_pid
    p_pid = (tpid == 2212) & ~mu_pid
    bad_mu = nu_t & mu_pid & (ke_any >= args.mb_mu_kemin)
    bad_pi = nu_t & pi_pid & (ke_any >= args.mb_pi_kemin)
    bad_p = nu_t & p_pid & (ke_any >= args.mb_p_kemax)
    acc_p = nu_t & p_pid & (ke_any < args.mb_p_kemax)
    allowed = nu_t & ((mu_pid & (ke_any < args.mb_mu_kemin)) | (pi_pid & (ke_any < args.mb_pi_kemin)))
    other_t = nu_t & ~acc_p & ~allowed & ~bad_mu & ~bad_pi & ~bad_p      # unclassified etc.
    cnt = lambda m: ak.to_numpy(ak.sum(m, axis=1)).astype(np.int64)
    out["n_bad_mu"], out["n_bad_pi"], out["n_bad_p"] = cnt(bad_mu), cnt(bad_pi), cnt(bad_p)
    out["n_ring"] = out["n_bad_mu"] + out["n_bad_pi"] + out["n_bad_p"]
    out["n_p"] = cnt(acc_p & (ke_any >= args.proton_ke_min))        # 1gNp counter
    out["p_kemax"] = np.asarray(ak.to_numpy(ak.fill_none(ak.max(ak.where(nu_t & p_pid, ke_any, -1.0), axis=1), -1.0)), np.float64)
    out["mupi_kemax"] = np.asarray(ak.to_numpy(ak.fill_none(ak.max(ak.where(nu_t & (mu_pid | pi_pid), ke_any, -1.0), axis=1), -1.0)), np.float64)
    e_other_t = np.asarray(ak.to_numpy(ak.sum(ak.where(other_t, ke_any, 0.0), axis=1)), np.float64)
    # MUON-AWARE containment / PID inputs (2026-10-05): the MB restriction is on
    # the muon (and pion) ring, so expose per-PID energies, per-PID wall
    # distances, an exiting-muon flag and the LArPID muon score.
    def amax(m, v, fill=-1.0):
        return np.asarray(ak.to_numpy(ak.fill_none(ak.max(ak.where(m, v, fill), axis=1), fill)), np.float64)
    def amin(m, v, fill=1e9):
        r = np.asarray(ak.to_numpy(ak.fill_none(ak.min(ak.where(m, v, fill), axis=1), fill)), np.float64)
        r[r >= 1e8] = np.nan
        return r
    tdw = None
    for pre in ("trackStartPos", "trackEndPos"):
        for k, c in enumerate("XYZ"):
            v = a[pre + c]
            d = np.minimum(v - TPC_LO[k], TPC_HI[k] - v)
            tdw = d if tdw is None else np.minimum(tdw, d)
    out["mu_kemax"] = amax(nu_t & mu_pid, ke_any)
    out["pi_kemax"] = amax(nu_t & pi_pid, ke_any)
    out["mu_min_dwall"] = amin(nu_t & mu_pid, tdw)
    out["pi_min_dwall"] = amin(nu_t & pi_pid, tdw)
    out["mu_exiting"] = (np.nan_to_num(out["mu_min_dwall"], nan=1e9) < args.contain_dwall).astype(np.float64)
    out["mu_score_max"] = amax(nu_t & (a["trackClassified"] == 1), a["trackMuScore"])
    out["n_mu_tracks"] = cnt(nu_t & mu_pid)
    # ---- showers: everything but the candidate ---------------------------------
    Es = R["_E"]
    nu_s = (a["showerStream"] == 0) if args.novtx else ak.ones_like(Es, dtype=bool)
    li = ak.local_index(Es, axis=1)
    not_cand = li != ak.fill_none(ak.firsts(idx), -1)
    other_s = nu_s & not_cand & (Es > 0)
    e_other_s = np.asarray(ak.to_numpy(ak.sum(ak.where(other_s, Es, 0.0), axis=1)), np.float64)
    out["extra_E"] = e_other_s + e_other_t
    out["extra_E_showers"] = e_other_s
    # ---- pi0-decay cone: soft clusters vs the candidate direction ---------------
    cd = np.stack([pick(R["_sd" + c], idx, has) for c in "XYZ"], 1)
    cdn = cd / np.maximum(np.linalg.norm(cd, axis=1, keepdims=True), 1e-12)
    cx, cy, cz = (ak.Array(np.nan_to_num(cdn[:, k], nan=0.0)) for k in range(3))   # per-event scalars
    cosang = a["showerStartDirX"] * cx + a["showerStartDirY"] * cy + a["showerStartDirZ"] * cz
    soft = nu_s & not_cand & (Es <= args.extra_emax)
    one_m_cos = ak.where(soft, np.maximum(1.0 - cosang, 1e-6), np.nan)
    E1 = np.maximum(np.nan_to_num(E_cand, nan=1.0), 1.0)
    e2_req = (134.977 ** 2) / (2.0 * ak.Array(E1) * one_m_cos)
    cone = soft & (e2_req >= args.cone_e2_min) & (e2_req <= args.cone_e2_max) & (e2_req >= Es)
    out["n_cone"] = cnt(cone)
    ang = ak.where(soft, np.degrees(np.arccos(np.minimum(np.maximum(cosang, -1.0), 1.0))), np.nan)
    out["soft_min_angle"] = np.asarray(ak.to_numpy(ak.fill_none(ak.min(ang, axis=1), np.nan)), np.float64)
    out["n_soft"] = cnt(soft)
    e2min = ak.fill_none(ak.min(ak.where(soft, e2_req, np.nan), axis=1), np.nan)
    out["cone_e2_min"] = np.asarray(ak.to_numpy(e2min), np.float64)
    # ---- geometry ----------------------------------------------------------
    st = np.stack([pick(R["_sp" + c], idx, has) for c in "XYZ"], 1)
    out["start_dwall"] = dwall(st)
    nu_v = a["recoVtxStream"] == 0
    vx = np.stack([np.asarray(ak.to_numpy(ak.fill_none(ak.firsts(a["recoVtx" + c][nu_v]), np.nan)), np.float64)
                   for c in "XYZ"], 1)
    vx[~R["vtx_ok"]] = np.nan
    out["vtx_dwall"] = dwall(vx)
    out["conv_dist"] = pick(a["showerDistToVtx"], idx, has)
    # ---- beam-likeness -------------------------------------------------------
    out["bdt_cosmic"] = pick(a["showerCosmicScore"], idx, has, -9.0)
    out["pe_ratio"] = R["pe_ratio"]
    with np.errstate(divide="ignore", invalid="ignore"):
        out["logchi2"] = np.log10(R["chi2"])
    # ---- reported kinematics (never cut) -------------------------------------
    out["cosBeam"] = pick(a["showerCosTheta"], idx, has)
    out["E_QE"] = e_qe(E_cand, out["cosBeam"])
    return out


def mb_conditions(M, args):
    """Per-step boolean conditions (each independent of the others) + names + Np mask."""
    nz = lambda x, f: np.nan_to_num(x, nan=f)
    is_np = M["n_p"] >= 1
    conds = [M["cell"]]
    names = ["M0 primary cell (vtx, veto-pass, FC, dfw)"]
    conds.append(M["n_ring"] == 0); names.append("M1 MB hadrons (no ring-making mu/pi/p)")
    conds.append(M["extra_E"] <= args.extra_emax); names.append(f"M2 extra energy <= {args.extra_emax:.0f} MeV")
    if args.pi0_cone:
        conds.append(M["n_cone"] == 0); names.append(f"M2b no soft cluster in the pi0 cone (E2 in [{args.cone_e2_min:.0f},{args.cone_e2_max:.0f}])")
    tight = np.where(is_np, nz(M["vtx_dwall"], -1) >= args.tight_dwall,
                     (nz(M["start_dwall"], -1) >= args.tight_dwall) & (nz(M["dfw"], -1) >= args.dfw_tight))
    conds.append(tight); names.append(f"M3 tight FV (dwall >= {args.tight_dwall:.0f}; 0p also dfw >= {args.dfw_tight:.0f})")
    if args.conv_dist_min > 0:
        conds.append(~is_np | (nz(M["conv_dist"], -1) >= args.conv_dist_min))
        names.append(f"M4 Np conversion distance >= {args.conv_dist_min:.0f} cm")
    beam = ((M["bdt_cosmic"] >= args.cand_bdt_min_mb) & (nz(M["pe_ratio"], 1e9) < args.mb_pe_ratio_max)
            & (nz(M["logchi2"], 1e9) < args.mb_logchi2_max))
    conds.append(beam); names.append(f"M5 beam-like (BDT >= {args.cand_bdt_min_mb:.2f}, PE ratio < {args.mb_pe_ratio_max:.1f}, logchi2 < {args.mb_logchi2_max:.2f})")
    return names, conds, is_np


def mb_steps(M, args):
    """Cumulative MB steps on top of the primary cell; returns (names, steps, np_mask, conds)."""
    names, conds, is_np = mb_conditions(M, args)
    s = [conds[0]]
    for c in conds[1:]:
        s.append(s[-1] & c)
    return names, np.stack(s), is_np, conds


def all_but(conds, k):
    """Selection with every step condition except step k (N-1)."""
    m = conds[0].copy()
    for j, c in enumerate(conds[1:], start=1):
        if j != k:
            m &= c
    return m


def run(ntuple, is_mc, args):
    a, pot = SPS.load(ntuple, is_mc, args.novtx, extra=BR_EXTRA)
    R = SPS.reco_tables(a, args)
    R["novtx"] = args.novtx
    names, S, E, _, _, path = SPS.build_steps(R, args.shower_bdt_min, args.flashchi2_cut,
                                              args.bdt_after_multiplicity, args.flashpe_ratio_max)
    M = mb_tables(a, R, args, names, S, E, path)
    return a, pot, names, M


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(ap)
    ap.set_defaults(signal_def="mb")
    ap.add_argument("--veto2g-emin", type=float, default=10.0)
    ap.add_argument("--contain-dwall", type=float, default=10.0)
    ap.add_argument("--dfw-cut", type=float, default=SPS.DFW_CUT)
    ap.add_argument("--mb-mu-kemin", type=float, default=40.0, help="mu-PID track KE veto [MeV]")
    ap.add_argument("--mb-pi-kemin", type=float, default=50.0, help="pi-PID track KE veto [MeV]")
    ap.add_argument("--mb-p-kemax", type=float, default=342.0, help="proton track KE veto [MeV]")
    ap.add_argument("--extra-emax", type=float, default=10.0, help="other-prong energy sum max [MeV]")
    ap.add_argument("--pi0-cone", action="store_true", help="enable the M2b pi0-cone soft-cluster veto")
    ap.add_argument("--cone-e2-min", type=float, default=5.0)
    ap.add_argument("--cone-e2-max", type=float, default=600.0)
    ap.add_argument("--tight-dwall", type=float, default=10.0)
    ap.add_argument("--dfw-tight", type=float, default=100.0, help="0p: backward wall distance min [cm]")
    ap.add_argument("--conv-dist-min", type=float, default=0.0, help="Np: vertex-to-shower distance min [cm]; 0 = off")
    ap.add_argument("--cand-bdt-min-mb", type=float, default=0.7)
    ap.add_argument("--mb-pe-ratio-max", type=float, default=2.0)
    ap.add_argument("--mb-logchi2-max", type=float, default=2.25)
    args = ap.parse_args()
    SPS.configure(args)
    if not args.novtx:
        sys.exit("--novtx is required (the vertex-path cell is defined there)")
    os.makedirs(args.plots, exist_ok=True)

    a, pot, names, M = run(args.mc_ntuple, True, args)
    w0 = ak.to_numpy(a["xsecWeight"]).astype(np.float64)
    w = np.where(w0 > 0, w0, 0.0) * (args.pot / pot)
    if args.mc_odd_only:
        odd = ak.to_numpy(a["event"]) % 2 == 1
        w = np.where(odd, 2.0 * w, 0.0)
    T = SPS.truth_tables(a); cat = T["cat"]; sig = T["sig_incl"]
    tvx = np.stack([np.asarray(ak.to_numpy(a["trueVtx" + c]), np.float64) for c in "XYZ"], 1)
    true_tight = dwall(tvx) >= args.tight_dwall
    M["trueE"] = T["g_trueE"]
    del a
    MX = MD = None; wx = wd = None
    if args.ext_ntuple:
        _, _, _, MX = run(args.ext_ntuple, False, args)
        wx = np.zeros(len(MX["E"])); wx[args.ext_row_min:] = args.ext_scale
    if args.data_ntuple:
        _, _, _, MD = run(args.data_ntuple, False, args)
        wd = np.ones(len(MD["E"]))
    mnames, S, is_np, C = mb_steps(M, args)
    SX, npx, CX = (mb_steps(MX, args)[1:] if MX is not None else (None, None, None))
    SD, npd, CD = (mb_steps(MD, args)[1:] if MD is not None else (None, None, None))

    CATS, NSIG, NMC = SPS.CATS, SPS.N_SIG_CAT, SPS.N_MC_CAT
    infv = sig & T["in_fv"]; tight = infv & true_tight
    den_fv, den_tight, den_0p = w[infv].sum(), w[tight].sum(), w[tight & (T["n_x_strict"] == 0)].sum()
    L = [f"MB-targeted selection; truth '{SPS.SIGNAL_DEF}'; denominators (w @ {args.pot:.1e} POT): "
         f"in-FV signal {den_fv:.1f} | tight-FV (true vtx dwall >= {args.tight_dwall:.0f}) {den_tight:.1f} "
         f"| tight-FV 1g0X {den_0p:.1f}; total mb signal {w[sig].sum():.1f}"]
    hdr = (f"{'step':62s}{'signal':>8}{'in-FV':>7}{'effFV':>7}{'effTight':>9}{'pur':>7}{'purFV':>7}"
           f"{'MC bkg':>8}{'EXT':>7}{'pred':>8}{'data':>6}{'d/p':>6}")
    L.append(hdr)

    def row(lab, m, mx, md):
        st, sf, b = w[m & sig].sum(), w[m & infv].sum(), w[m & ~sig].sum()
        e = wx[mx].sum() if mx is not None else 0.0
        d = int(md.sum()) if md is not None else 0
        tot = max(st + b + e, 1e-9)
        return (f"{lab[:62]:62s}{st:8.1f}{sf:7.1f}{sf / max(den_fv, 1e-9):7.3f}{w[m & tight].sum() / max(den_tight, 1e-9):9.3f}"
                f"{st / tot:7.3f}{sf / tot:7.3f}{b:8.1f}{e:7.1f}{tot:8.1f}{d:6d}{d / tot:6.2f}")

    for k, nm in enumerate(mnames):
        L.append(row(nm, S[k], SX[k] if SX is not None else None, SD[k] if SD is not None else None))
        L.append("    bkg: " + " | ".join(f"{CATS[c][5:28]} {w[S[k] & (cat == c)].sum():.1f}"
                                          for c in range(NSIG, NMC) if w[S[k] & (cat == c)].sum() > 0))
    fin = S[-1]
    L.append("FINAL split:")
    for lab, mm, mmx, mmd in (("1g0p (no accepted proton >= %.0f MeV)" % args.proton_ke_min, ~is_np, None if npx is None else ~npx, None if npd is None else ~npd),
                              ("1gNp", is_np, npx, npd)):
        L.append(row("  " + lab, fin & mm, None if SX is None else SX[-1] & mmx, None if SD is None else SD[-1] & mmd))
        L.append("    bkg: " + " | ".join(f"{CATS[c][5:28]} {w[fin & mm & (cat == c)].sum():.1f}"
                                          for c in range(NSIG, NMC) if w[fin & mm & (cat == c)].sum() > 0))
        L.append("    signal: " + " | ".join(f"{CATS[c][4:]} {w[fin & mm & (cat == c)].sum():.1f}" for c in range(NSIG)))
    # anatomy of the remaining >=2-photon and entering backgrounds (truth-aided)
    g2 = np.nan_to_num(T["g2_evis_true"], nan=-1.0)
    for lab, mm in (("final", fin), ("final 1g0p", fin & ~is_np), ("final 1gNp", fin & is_np)):
        b2 = mm & np.isin(cat, [4, 9])
        W2 = w[b2].sum()
        if W2 > 0:
            hasprong = np.isfinite(M["other_gE_max"])
            L.append(f"  {lab}: >=2-photon bkg {W2:.1f}: 2nd photon deposited < {SPS.EVIS_MIN:.0f} MeV (invisible in LAr) "
                     f"{w[b2 & (g2 < SPS.EVIS_MIN)].sum():.1f} | deposited >= {SPS.EVIS_MIN:.0f} and NO other photon prong "
                     f"{w[b2 & (g2 >= SPS.EVIS_MIN) & ~hasprong].sum():.1f} | with a soft other photon prong "
                     f"{w[b2 & (g2 >= SPS.EVIS_MIN) & hasprong].sum():.1f}")
        be = mm & np.isin(cat, [4, 5, 6])
        if w[be].sum() > 0:
            L.append(f"  {lab}: entering bkg {w[be].sum():.1f} (vertex outside the FV: physically an entering photon; "
                     f"cos-free handles only: dfw / start dwall)")
    # N-1 summary: each M-step removed from the final selection
    L.append("N-1 (final selection without one step): signal | MC bkg | EXT | purity")
    for k in range(1, len(mnames)):
        m = all_but(C, k)
        mx_ = all_but(CX, k) if CX is not None else None
        st, b = w[m & sig].sum(), w[m & ~sig].sum(); e = wx[mx_].sum() if mx_ is not None else 0.0
        L.append(f"  without {mnames[k][:50]:50s} {st:7.1f} | {b:7.1f} | {e:6.1f} | {st / max(st + b + e, 1e-9):.3f}")
    txt = "\n".join(L); print(txt)
    with open(os.path.join(args.plots, "mb_cutflow.txt"), "w") as fh:
        fh.write(txt + "\n")

    # ---- plots -----------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    COL = SPS.CAT_COLORS

    def stack(key, bins, xlab, m, mx, md, fname, title):
        lo, hi = bins[0], bins[-1] - 1e-6
        v = M[key]; ok = m & np.isfinite(v)
        data = [np.clip(v[ok & (cat == c)], lo, hi) for c in range(NMC)]
        ws = [w[ok & (cat == c)] for c in range(NMC)]; cols = list(COL[:NMC])
        if MX is not None and mx is not None:
            okx = mx & np.isfinite(MX[key]); data.append(np.clip(MX[key][okx], lo, hi)); ws.append(wx[okx]); cols.append(COL[NMC])
        fig, ax = plt.subplots(figsize=(7.6, 4.8))
        ax.hist(data, bins=bins, weights=ws, stacked=True, color=cols,
                label=[f"{CATS[c]} ({ws[c].sum():.1f})" for c in range(len(ws))])
        if MD is not None and md is not None:
            okd = md & np.isfinite(MD[key]); hd, _ = np.histogram(np.clip(MD[key][okd], lo, hi), bins=bins)
            ctr = 0.5 * (bins[:-1] + bins[1:])
            ax.errorbar(ctr[hd > 0], hd[hd > 0], yerr=np.sqrt(hd[hd > 0]), fmt="ko", ms=3.5, label=f"beam data ({int(hd.sum())})")
        ax.set(xlabel=xlab, ylabel=f"events / {args.pot:.1e} POT", title=title)
        ax.legend(fontsize=6); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(args.plots, fname), dpi=110); plt.close(fig)

    ebins = np.linspace(0, 1000, 41)
    for k, nm in enumerate(mnames):
        stack("E", ebins, "reco photon energy [MeV] (reported, not cut)", S[k],
              None if SX is None else SX[k], None if SD is None else SD[k],
              f"photon_energy_M{k}.png", f"candidate photon energy after {nm[:40]}")
    for lab, mm, mmx, mmd in (("0p", ~is_np, npx if npx is None else ~npx, npd if npd is None else ~npd), ("Np", is_np, npx, npd)):
        mx_ = None if SX is None else SX[-1] & mmx; md_ = None if SD is None else SD[-1] & mmd
        stack("E", ebins, "reco photon energy [MeV] (reported, not cut)", fin & mm, mx_, md_, f"final_{lab}_E.png", f"final 1g{lab}: photon energy")
        stack("cosBeam", np.linspace(-1, 1, 41), r"shower $\cos\theta_{beam}$ (reported, not cut)", fin & mm, mx_, md_, f"final_{lab}_cosBeam.png", f"final 1g{lab}: cos theta")
        stack("E_QE", np.linspace(0, 1500, 31), "E_QE [MeV] (e-CCQE hypothesis; reported, not cut)", fin & mm, mx_, md_, f"final_{lab}_EQE.png", f"final 1g{lab}: E_QE")
    # N-1 plots of every M-step variable: selection with that step's cut removed
    NM1 = [("n_ring", np.arange(-0.5, 5.5, 1), "n ring-making mu/pi/p tracks", 1),
           ("mupi_kemax", np.linspace(0, 600, 31), "max mu/pi-PID track KE [MeV] (-1 = none)", 1),
           ("p_kemax", np.linspace(0, 800, 33), "max proton-PID track KE [MeV] (-1 = none)", 1),
           ("extra_E", np.linspace(0, 200, 41), "other-prong energy sum [MeV]", 2),
           ("extra_E_showers", np.linspace(0, 200, 41), "other-SHOWER energy sum [MeV]", 2),
           ("n_soft", np.arange(-0.5, 6.5, 1), "n soft other shower prongs (E <= extra-emax)", 2),
           ("soft_min_angle", np.linspace(0, 180, 37), "min angle soft cluster - candidate [deg]", 2),
           ("n_cone", np.arange(-0.5, 4.5, 1), "n soft clusters in the pi0 cone", 2),
           ("vtx_dwall", np.linspace(0, 130, 27), "reco vertex dwall [cm]", 3),
           ("start_dwall", np.linspace(0, 130, 27), "candidate start dwall [cm]", 3),
           ("dfw", np.linspace(0, 1000, 41), "backward wall distance [cm]", 3),
           ("conv_dist", np.linspace(0, 100, 41), "candidate start to vertex [cm]", 3),
           ("bdt_cosmic", np.linspace(0, 1, 41), "candidate cosmic BDT", len(mnames) - 1),
           ("pe_ratio", np.linspace(0, 4, 41), "flash PE observed / predicted", len(mnames) - 1),
           ("logchi2", np.linspace(0, 5, 41), "log10 flash chi2", len(mnames) - 1)]
    for key, bins, xlab, kstep in NM1:
        if key in ("n_soft", "soft_min_angle", "n_cone") and args.pi0_cone:
            kstep = next(i for i, nm_ in enumerate(mnames) if nm_.startswith("M2b"))
        elif kstep >= 3 and args.pi0_cone:
            kstep += 1                       # M3/M4/M5 shift by one when M2b is present
        kstep = min(kstep, len(mnames) - 1)
        m = all_but(C, kstep); mx_ = all_but(CX, kstep) if CX is not None else None
        md_ = all_but(CD, kstep) if CD is not None else None
        stack(key, bins, xlab, m, mx_, md_, f"nminus1_{key}.png", f"N-1: {xlab} (all MB steps except {mnames[kstep][:3]})")
    # efficiency vs true photon energy for the final 0p / Np samples (reported)
    tedges = np.array([20, 40, 60, 80, 100, 150, 200, 300, 400, 600, 1000]); tctr = 0.5 * (tedges[:-1] + tedges[1:])
    fig, ax = plt.subplots(figsize=(7, 4.6))
    x = np.nan_to_num(M["trueE"], nan=-1)
    for lab, den in (("in-FV signal", infv), (f"tight-FV signal (dwall >= {args.tight_dwall:.0f})", tight), ("tight-FV 1g0X", tight & (T["n_x_strict"] == 0))):
        for sl, mm, sty in (("0p", ~is_np, "o-"), ("Np", is_np, "s--")):
            num = fin & mm & den
            e, er = SPS.binned_ratio(w * num, w * den, x, tedges, n_raw=den)
            ax.errorbar(tctr, e, yerr=er, fmt=sty, ms=4, label=f"{lab} -> 1g{sl} ({w[num].sum() / max(w[den].sum(), 1e-9):.3f})")
    ax.set(xlabel="true photon energy [MeV]", ylabel="efficiency", xscale="log", xlim=(tedges[0], tedges[-1]), ylim=(0, 0.6),
           title="MB selection efficiency vs true photon energy (final)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(os.path.join(args.plots, "eff_vs_true_E_final.png"), dpi=110); plt.close(fig)
    np.savez(os.path.join(args.plots, "mb_mc.npz"), w=w, cat=cat, steps=S, is_np=is_np,
             **{k: v for k, v in M.items() if isinstance(v, np.ndarray)})
    print(f">>> -> {args.plots}")


if __name__ == "__main__":
    main()
