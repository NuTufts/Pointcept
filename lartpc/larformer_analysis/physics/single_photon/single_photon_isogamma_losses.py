#!/usr/bin/env python3
"""Why do lar-visible gun photons fail the single-photon cutflow?  Companion to
single_photon_isogamma_cutflow.py (same args). Per step, the failing events are
classified by what happened to the GUN photon's reco shower (truth match via
showerTrueTID), split by true photon energy.
"""
import argparse, os, sys
import numpy as np, awkward as ak
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import single_photon_selection as SPS, single_photon_diagnostics as SPD
SPD.BR_EXTRA = tuple(SPD.BR_EXTRA) + ("trueNuE", "showerTrueTID", "showerTruePurity", "trackTrueTID", "trackLArFormerPID", "trackStream", "trueVtxX", "trueVtxY", "trueVtxZ", "showerRecoE")


def main():
    ap = argparse.ArgumentParser(description=__doc__); SPD.add_args(ap); ap.set_defaults(signal_def="lar")
    for k, v in (("veto2g_emin", 10.0), ("veto2g_dcmax", 100.0), ("mb_mu_kemin", 40.0), ("mb_pi_kemin", 50.0), ("mb_p_kemax", 342.0), ("extra_emax", 10.0), ("cone_e2_min", 5.0), ("cone_e2_max", 600.0), ("contain_dwall", 10.0), ("dfw_cut", SPS.DFW_CUT), ("numu_cut", 0.25), ("enter_cut", 0.30), ("enter_cut_0p", 0.15)):
        ap.add_argument("--" + k.replace("_", "-"), type=float, default=v)
    ap.add_argument("--mb-chain-model", default=""); ap.add_argument("--mb-chain-model-extra", default=""); ap.add_argument("--unit-weight", action="store_true", default=True)
    args = ap.parse_args(); SPS.configure(args); os.makedirs(args.plots, exist_ok=True)
    M, a, pot, R = SPD.per_sample(args.mc_ntuple, True, args, None, "GUN")
    S, names = M["steps"], M["names"]; n = len(M["E"]); T = SPS.truth_tables(a)
    Eg = np.asarray(ak.to_numpy(a["trueNuE"]), np.float64) * 1000.0
    gun = (a["trueSimPartPDG"] == 22) & (a["trueSimPartProcess"] == 0)
    first = lambda x, fill=np.nan: np.asarray(ak.to_numpy(ak.fill_none(ak.firsts(x[gun]), fill)), np.float64)
    conv = np.stack([first(a["trueSimPartEDep" + c]) for c in "XYZ"], 1)
    converted = np.all(np.isfinite(conv), 1) & np.all(np.abs(conv) < 1e4, 1)
    evis = first(SPS.A_GAMMA * np.maximum(a["trueSimPartPixelSumQ"], 0.0)); visible = converted & (evis >= SPS.EVIS_MIN)
    gtid = ak.Array(first(a["trueSimPartTID"], -1).astype(np.int64))
    den = visible & T["in_fv"]
    # --- what happened to the photon's shower(s)?
    sm = (a["showerTrueTID"] == gtid) & (a["showerTruePurity"] > 0.5)         # shower prongs that ARE the photon
    tm = (a["trackTrueTID"] == gtid)                                            # the photon reconstructed as a TRACK
    nus = a["showerStream"] == 0; nut = a["trackStream"] == 0
    E = R["_E"]; spid = a["showerLArFormerPID"]
    has_nu_slice = ak.to_numpy(ak.any(nus, axis=1) | ak.any(nut, axis=1))
    any_sh = ak.to_numpy(ak.any(sm, axis=1)); any_tr = ak.to_numpy(ak.any(tm, axis=1))
    sh_nu = ak.to_numpy(ak.any(sm & nus, axis=1)); tr_nu = ak.to_numpy(ak.any(tm & nut, axis=1))
    sh_nu_g = ak.to_numpy(ak.any(sm & nus & (spid == 22), axis=1))
    sh_nu_g20 = ak.to_numpy(ak.any(sm & nus & (spid == 22) & (E > SPS.RECO_G_MIN), axis=1))
    pid_nu = ak.to_numpy(ak.fill_none(ak.firsts(spid[sm & nus]), -1))
    n_cand = ak.to_numpy(ak.sum(R["_is_g0"] & (R["_score"] >= args.shower_bdt_min), axis=1))
    n_cand_ph = ak.to_numpy(ak.sum(R["_is_g0"] & (R["_score"] >= args.shower_bdt_min) & sm, axis=1))
    n_g0_ph = ak.to_numpy(ak.sum(R["_is_g0"] & sm, axis=1))
    s1f = den & ~S[1]
    rows1 = [("no nu-stream slice at all (no nu prong)", s1f & ~has_nu_slice),
             ("nu slice, photon has NO matched prong (shower or track)", s1f & has_nu_slice & ~any_sh & ~any_tr),
             ("nu slice, photon only matched to a TRACK prong", s1f & has_nu_slice & ~any_sh & any_tr),
             ("nu slice, photon shower in a COSMIC stream", s1f & has_nu_slice & any_sh & ~sh_nu),
             ("nu slice, photon shower in nu stream, PID = electron", s1f & sh_nu & (pid_nu == 11)),
             ("nu slice, photon shower in nu stream, PID = other (not 22/11)", s1f & sh_nu & (pid_nu != 11) & ~sh_nu_g),
             ("nu slice, photon shower PID 22 but reco E <= 20 MeV", s1f & sh_nu_g & ~sh_nu_g20)]
    s2f = den & S[1] & ~S[2]
    rows2 = [("0 candidates: photon shower(s) fail the BDT", s2f & (n_cand == 0) & (n_g0_ph > 0)),
             ("0 candidates: photon not among photon showers (other shower < BDT)", s2f & (n_cand == 0) & (n_g0_ph == 0)),
             (">=2 candidates, >=2 of them are the photon (SPLIT shower)", s2f & (n_cand >= 2) & (n_cand_ph >= 2)),
             (">=2 candidates, one is the photon + other prong(s)", s2f & (n_cand >= 2) & (n_cand_ph == 1)),
             (">=2 candidates, none is the photon", s2f & (n_cand >= 2) & (n_cand_ph == 0))]
    s3f = den & S[2] & ~S[3]; s4f = den & S[3] & ~S[4]; s5f = den & S[4] & ~S[5]; s6f = den & S[5] & ~S[6]
    ebins = [(0, 100), (100, 200), (200, 400), (400, 800), (800, 1e9)]
    def line(lab, m):
        return f"{lab:66s}{int(m.sum()):7d}{m.sum() / den.sum():8.3f}" + "".join(f"{(m & (Eg >= lo) & (Eg < hi)).sum() / max((den & (Eg >= lo) & (Eg < hi)).sum(), 1):9.3f}" for lo, hi in ebins)
    L = [f"LOSS BREAKDOWN for lar-visible gun photons with ORIGIN in the WC FV (den {int(den.sum())}); columns: N, fraction of den, then fraction of den per true-E bin "
         + " ".join(f"[{lo:.0f},{hi:.0f})" for lo, hi in ebins), f"{'':66s}{'N':>7}{'frac':>8}" + "".join(f"{'E%d' % lo:>9}" for lo, hi in ebins)]
    for k in range(1, 7):
        L.append(line(f"pass {names[k]}", den & S[k]))
    L.append("\n-- fail S1 (no nu-stream photon shower > 20 MeV):"); L += [line("  " + l, m) for l, m in rows1]
    L.append(line("  (unclassified S1 failures)", s1f & ~np.any([m for _, m in rows1], 0)))
    L.append("\n-- fail S2 (exactly one BDT-passing candidate):"); L += [line("  " + l, m) for l, m in rows2]
    L.append("\n-- fail S3 (visible X): muon-like track / electron shower present"); L.append(line("  fail S3", s3f))
    L.append(line("  fail S3 and the photon is itself the 'X' (track prong matched to the photon)", s3f & any_tr))
    L.append("\n-- flash / vertex steps"); L += [line("  fail S4 flash chi2", s4f), line("  fail S5 PE ratio", s5f), line("  fail S6 vertex cosmic BDT", s6f)]
    # same for the entering (origin outside FV) sample, compactly
    L.append(f"\nFor reference, lar-visible photons with origin OUTSIDE the FV (den {int((visible & ~T['in_fv']).sum())}): pass S1 {((visible & ~T['in_fv']) & S[1]).sum() / (visible & ~T['in_fv']).sum():.3f}, "
             f"S2 {((visible & ~T['in_fv']) & S[2]).sum() / (visible & ~T['in_fv']).sum():.3f}, S6 {((visible & ~T['in_fv']) & S[6]).sum() / (visible & ~T['in_fv']).sum():.3f}")
    txt = "\n".join(L); print(txt); open(os.path.join(args.plots, "isogamma_losses.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
