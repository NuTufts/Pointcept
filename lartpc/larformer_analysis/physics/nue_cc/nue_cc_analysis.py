"""nue CC inclusive selection -- per-sample TABLE builder (ntuple-only).

Reads ONE gen2ntuple and writes a per-event selection table (.npz) that the
overlay script (nue_cc_overlay.py) stacks into a data/MC/EXT prediction. Mirrors
the pi0mass_peak two-stage pattern (pi0_mass_analysis.py -> datamc_ext_overlay.py).

Chain version v2_s1ep2p8cew6: recal3 is BAKED INTO showerRecoE, so no
analysis-side recal is applied here.

Truth (MC), per README:
  signal = true nue CC (|trueNuPDG|==12 & trueNuCCNC==0) with the true vertex in
  the WireCell FV (trueVtxInWCFV==1) and a primary electron with E>20 MeV.
  `is_nuecc` (|trueNuPDG|==12 & CC, ANY vertex) is the veto flag: the bnb-nu
  overlay's nue CC must be removed downstream so the intrinsic-nue sample is the
  ONLY source of nue CC (no double counting).

First-pass reco selection (nu stream):
  foundVertex & primaryVtxStream==0 & vtxIsFiducial==1, AND >=1 PRIMARY electron
  shower (showerLArFormerPID==11, showerIsSecondary==0, showerRecoE>ELE_E_MIN).
  Observable = leading (most energetic) primary-electron showerRecoE.
  flash_chi2 = the primary nu-stream reco vertex's recoVtxFlashChi2. The
  flash-chi2 CUT is applied downstream (nue_cc_common.base).

Also stores, for the leading electron candidate: the LArPID + LArFormer scores,
the cew6 handles (objectness, cosmic/novtx BDT scores, slice flash-chi2, vertex
quality), the sharpened photon-veto counts, and -- for MC -- the TRUTH CATEGORY
of what the candidate actually is (`bg_cat`, see categorize()).

Weights: MC w = xsecWeight * (--pot / sum potTree.totGoodPOT); data/EXT unit
(EXT gets its spill weight in the overlay).

    PYTHONPATH=./ python3 nue_cc_analysis.py --ntuple ....root --out tab.npz [--data]
"""
import argparse
import os
import sys

import numpy as np
import uproot
import awkward as ak

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nue_cc_common as C  # noqa: E402

ELE_E_MIN = 20.0      # reco primary-electron shower energy floor [MeV]
TRUE_ELE_E_MIN = 20.0  # true primary-electron energy floor [MeV] (signal def)
M_E = 0.000511         # GeV
A_GAMMA = 0.0253017    # MeV/ADC -- EM (e/gamma) visible-energy calib (pi0 conv.)
PHOTON_E_MIN = 20.0    # reco photon floor for the pi0 tags [MeV]

CAT = {name: i for i, name in enumerate(C.TRUTH_CATS)}


# --------------------------------------------------------------------------
# Truth categorisation
# --------------------------------------------------------------------------
def _ancestor_primary(tid, tid_index, T, M, maxdepth=12):
    """Walk MID up to the Process==0 ancestor (MID==TID). -9 if it breaks."""
    for _ in range(maxdepth):
        j = tid_index.get(int(tid))
        if j is None:
            return -9
        mid = int(M[j])
        if mid == int(T[j]):          # Process==0 <=> MID==TID (verified)
            return int(T[j])
        tid = mid
    return -9


def categorize(tp, ti, pur, unl, T, M, P, D, tid_index, sig_tid):
    """(category index into C.TRUTH_CATS, subcategory = mother PDG).

    The cosmic test runs FIRST and uses `showerTrueTID <= 0`, NOT
    `showerTruePID <= 0`. Two distinct bugs are avoided:

      * `<= 0` on the PID swallows every NEGATIVE PDG code. On cew6 electron
        candidates that mislabels 294 genuinely nu-matched showers as cosmic
        (257 positrons, 31 pi-, 6 mu+), none of which has unlabeledPurity>=0.5.
      * `== 0` on the PID still mislabels the ~2.5% of showers that have
        truePID==0 but TID>0 -- neutrino-origin particles mcreco did not save.

    In MCC9 overlay the cosmics are real off-beam DATA carrying no G4 labels, so
    `origin==1 <=> trackid>0` exactly; TID<=0 therefore *is* the cosmic tag, and
    showerTrueUnlabeledPurity is the cosmic-contamination fraction.
    """
    if ti <= 0 or (np.isfinite(unl) and unl >= C.TRUTH_PURITY_MIN):
        return CAT["cosmic"], 0
    j = tid_index.get(int(ti))
    if j is None:
        return CAT["unresolved"], 0          # nu-origin, absent from trueSimPart
    if np.isfinite(pur) and pur < C.TRUTH_PURITY_MIN:
        return CAT["nu other"], 0
    proc = int(P[j])
    mid = int(M[j])
    jm = tid_index.get(mid)
    mpdg = int(D[jm]) if jm is not None else 0   # 0 = mother absent (e.g. pi0)
    a = abs(int(tp))
    if a == 11:
        if proc == C.PROC_PRIMARY and sig_tid > 0 and int(ti) == sig_tid:
            return CAT["nu e (primary)"], mpdg
        if sig_tid > 0 and _ancestor_primary(ti, tid_index, T, M) == sig_tid:
            return CAT["nu e (fragment)"], mpdg
        return CAT["nu e (secondary)"], mpdg
    if a == 22:
        return CAT["nu photon"], mpdg
    if a == 13:
        return CAT["nu muon"], mpdg
    if a == 211:
        return CAT["nu pion"], mpdg
    if a == 2212:
        return CAT["nu proton"], mpdg
    return CAT["nu other"], mpdg


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ntuple", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pot", type=float, default=C.DEFAULT_POT,
                    help="target POT for MC scaling (Tufts bnb5e19 livetime)")
    ap.add_argument("--data", action="store_true",
                    help="real-data mode: unit weights, no truth tags")
    ap.add_argument("--shower-bdt-wp", type=float, default=C.SHOWER_BDT_WP,
                    help="cosmic-BDT threshold defining a 'good' vertex photon")
    ap.add_argument("--novtx-bdt-wp", type=float, default=C.NOVTX_BDT_WP,
                    help="vertex-free BDT threshold for vertex-less photons")
    ap.add_argument("--max-events", type=int, default=None,
                    help="debug: only read the first N entries")
    ap.add_argument("--sample-set", default=C.DEFAULT_SAMPLE_SET,
                    choices=sorted(C.SAMPLE_SETS),
                    help="recorded in the table; the plotters take the EXT "
                         "weight and hygiene rules from it")
    ap.add_argument("--exclude-rows", default=None,
                    help="npz with a 'rows' array of ntuple rows to drop "
                         "entirely (w=0, never selected, no truth category) -- "
                         "e.g. MC files that fed model training. If it carries "
                         "'kept_pot', that replaces the potTree POT sum unless "
                         "--sample-pot is given. Same semantics as "
                         "pi0_mass_analysis.py --exclude-rows.")
    ap.add_argument("--sample-pot", type=float, default=None,
                    help="override the sample POT (default: potTree sum)")
    args = ap.parse_args()
    excl = np.load(args.exclude_rows) if args.exclude_rows else None
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    fin = uproot.open(args.ntuple)
    if args.data:
        scale = 1.0
        pot_sum = 0.0
        print(">>> DATA mode: unit weights, no truth tags")
    else:
        pot = fin["potTree"].arrays(library="np")
        pot_sum = float(np.sum(pot["totGoodPOT"])) or float(np.sum(pot["totPOT"]))
        if args.sample_pot is not None:
            print(f">>> potTree POT {pot_sum:.4e} overridden by --sample-pot")
            pot_sum = args.sample_pot
        elif excl is not None and "kept_pot" in excl.files:
            print(f">>> potTree POT {pot_sum:.4e} -> kept POT after row "
                  f"exclusion {float(excl['kept_pot']):.4e}")
            pot_sum = float(excl["kept_pot"])
        scale = args.pot / pot_sum
        print(f">>> sample POT {pot_sum:.3e}, target {args.pot:.2e} "
              f"-> scale {scale:.4e}")

    t = fin["EventTree"]
    have = set(t.keys())
    want = ["run", "subrun", "event", "xsecWeight",
            "trueNuE", "trueNuPDG", "trueNuCCNC", "trueVtxInWCFV",
            "trueLepE", "trueLepPDG",     # final-state lepton (electron for nueCC)
            "truePrimPartPDG", "truePrimPartE",
            # true sim particles -> primary-e visible energy + truth categories
            "trueSimPartPDG", "trueSimPartProcess", "trueSimPartE",
            "trueSimPartPixelSumQ", "trueSimPartTID", "trueSimPartMID",
            # truth match of the reco e-shower
            "showerTruePID", "showerTrueTID", "showerTruePurity",
            "showerTrueComp", "showerTrueUnlabeledPurity",
            "foundVertex", "primaryVtxStream", "vtxIsFiducial",
            "vtxX", "vtxY", "vtxZ",
            "vtxDistToTrue",          # reco-vtx to SCE-corrected true-vtx [cm]
            "vtxScore", "vtxFracHitsOnCosmic",   # vertex-quality / cosmic handles
            "recoVtxX", "recoVtxY", "recoVtxZ", "recoVtxStream",
            "recoVtxFlashChi2",
            "showerLArFormerPID", "showerRecoE", "showerIsSecondary",
            "showerVtxIdx",
            # cew6 per-shower additions
            "showerCosmicScore", "showerNoVtxScore", "showerObjectness",
            "showerStream", "showerCharge", "showerChargeFrac", "showerNHits",
            "showerAttScore", "showerAttConfident",
            # slice-level flash chi2 (complement to the per-vertex value)
            "nuSliceFlashChi2", "fmSliceFlashChi2",
            "nuSliceNParticles", "fmSliceNParticles",
            # LArPID PID log-softmax [e, gamma, mu, pi, p] + process log-softmax
            "showerElScore", "showerPhScore", "showerMuScore",
            "showerPiScore", "showerPrScore",
            "showerPrimaryScore",
            "showerFromNeutralScore", "showerFromChargedScore",
            # tracks (for the "muon at the e-shower's vertex" background tag)
            "trackVtxIdx", "trackMuScore",
            # LArFormer (segmentation model) softmax PROBABILITIES; stored as
            # log(prob) so the same confidence formulas apply as for LArPID.
            "showerLArFormerElScore", "showerLArFormerPhScore",
            "showerLArFormerMuScore", "showerLArFormerPiScore",
            "showerLArFormerPrScore", "trackLArFormerMuScore"]
    missing = [b for b in want if b not in have]
    if missing:
        print(f">>> NOTE: {len(missing)} branch(es) absent from this ntuple "
              f"(old-chain file?): {', '.join(sorted(missing))}")
    a = t.arrays([b for b in want if b in have],
                 entry_stop=args.max_events)
    n = len(a["run"])
    print(f">>> {n} events")

    def has(b):
        return b in have

    # ---- weights ----------------------------------------------------------
    if args.data:
        w = np.ones(n, np.float64)
        is_nuecc = np.zeros(n, bool)
        is_nuecc_fv = np.zeros(n, bool)
        pdg = np.zeros(n, np.int64)
        ccnc = np.full(n, -1, np.int64)
    else:
        w0 = np.asarray(a["xsecWeight"], np.float64)
        w = np.where(w0 > 0, w0, 0.0) * scale
        pdg = np.asarray(a["trueNuPDG"])
        ccnc = np.asarray(a["trueNuCCNC"])
        is_nuecc = (np.abs(pdg) == 12) & (ccnc == 0)          # VETO flag
        # true primary electron with E>20 MeV (truePrimPartE is total E [GeV])
        has_true_e = ak.to_numpy(ak.any(
            (np.abs(a["truePrimPartPDG"]) == 11)
            & (a["truePrimPartE"] * 1000.0 - M_E * 1000.0 > TRUE_ELE_E_MIN),
            axis=1))
        is_nuecc_fv = (is_nuecc & (np.asarray(a["trueVtxInWCFV"]) == 1)
                       & has_true_e)                          # SIGNAL def

    # ---- reco selection ---------------------------------------------------
    vtx_ok = ((np.asarray(a["foundVertex"]) == 1)
              & (np.asarray(a["primaryVtxStream"]) == 0)
              & (np.asarray(a["vtxIsFiducial"]) == 1))
    keep = np.ones(n, bool)
    if excl is not None:
        rows = np.asarray(excl["rows"], np.int64)
        keep[rows[rows < n]] = False
        w[~keep] = 0.0
        is_nuecc_fv = is_nuecc_fv & keep       # out of the efficiency denominator
        vtx_ok = vtx_ok & keep                 # never selected, no category
        print(f">>> excluded {int((~keep).sum())} rows via {args.exclude_rows}")
    is_e = ((a["showerLArFormerPID"] == 11) & (a["showerIsSecondary"] == 0)
            & (a["showerRecoE"] > ELE_E_MIN))
    n_e = ak.to_numpy(ak.sum(is_e, axis=1))
    sel = vtx_ok & (n_e >= 1)

    nan = lambda: np.full(n, np.nan)                          # noqa: E731
    reco_ele_E = nan()
    el_score, ph_score, mu_score = nan(), nan(), nan()
    pi_score, pr_score = nan(), nan()
    prim_score, fromneut_score, fromchg_score = nan(), nan(), nan()
    flash_chi2 = nan()
    vtx_mu_score = nan()
    vtx_dist_true = (np.asarray(a["vtxDistToTrue"], np.float64)
                     if has("vtxDistToTrue") and not args.data else nan())
    # cew6 candidate handles
    cosmic_score, novtx_score, objectness = nan(), nan(), nan()
    charge, chargefrac, nhits = nan(), nan(), nan()
    att_score, att_confident = nan(), nan()
    stream = np.full(n, -1, np.int64)
    # counts
    n_photons = np.full(n, -1, np.int64)
    n_good_photons = np.full(n, -1, np.int64)
    n_novtx_photons = np.full(n, -1, np.int64)
    # truth of the candidate
    shower_true_pid = np.zeros(n, np.int64)
    shower_true_tid = np.zeros(n, np.int64)
    shower_true_purity, shower_true_comp = nan(), nan()
    shower_true_unl_purity = nan()
    bg_cat = np.full(n, -1, np.int64)
    bg_subcat = np.zeros(n, np.int64)
    # LArFormer log-probs
    lf_el, lf_ph, lf_mu, lf_pi, lf_pr = nan(), nan(), nan(), nan(), nan()
    vtx_lf_mu_score = nan()
    # truth kinematics
    true_ele_ke, true_ele_vise, true_nu_e = nan(), nan(), nan()

    have_lp = has("showerElScore")
    have_lf = has("showerLArFormerElScore")
    have_tpid = has("showerTruePID")
    have_truthcat = (not args.data and has("showerTrueTID")
                     and has("trueSimPartTID") and has("trueSimPartMID"))
    if not args.data and not have_truthcat:
        print(">>> NOTE: truth categories unavailable (old-chain ntuple lacks "
              "showerTrueTID / trueSimPartMID); bg_cat stays -1")

    def logclip(p):
        return float(np.log(np.clip(float(p), 1e-6, 1.0)))

    if not args.data:
        nuE = np.asarray(a["trueNuE"], float)
        true_nu_e = np.where(nuE > 0, nuE, np.nan)            # GeV
        lepE = np.asarray(a["trueLepE"], float)               # GeV, FS lepton
        leppdg = np.asarray(a["trueLepPDG"])
        true_ele_ke = np.where((np.abs(leppdg) == 11) & (lepE > 0),
                               (lepE - M_E) * 1000.0, np.nan)

    rvs, rvf = a["recoVtxStream"], a["recoVtxFlashChi2"]
    rvx, rvy, rvz = a["recoVtxX"], a["recoVtxY"], a["recoVtxZ"]
    vtx_score = (np.asarray(a["vtxScore"], float) if has("vtxScore") else nan())
    vtx_frac_cosmic = (np.asarray(a["vtxFracHitsOnCosmic"], float)
                       if has("vtxFracHitsOnCosmic") else nan())
    nu_slice_chi2 = (np.asarray(a["nuSliceFlashChi2"], float)
                     if has("nuSliceFlashChi2") else nan())
    fm_slice_chi2 = (np.asarray(a["fmSliceFlashChi2"], float)
                     if has("fmSliceFlashChi2") else nan())
    nu_slice_nparts = (np.asarray(a["nuSliceNParticles"], np.int64)
                       if has("nuSliceNParticles") else np.full(n, -1, np.int64))
    fm_slice_nparts = (np.asarray(a["fmSliceNParticles"], np.int64)
                       if has("fmSliceNParticles") else np.full(n, -1, np.int64))

    for i in np.nonzero(vtx_ok)[0]:
        # primary nu-stream vertex flash chi2 (nu-stream recoVtx closest to vtx)
        st = ak.to_numpy(rvs[i])
        nu = np.nonzero(st == 0)[0]
        if len(nu):
            d = ((ak.to_numpy(rvx[i])[nu] - a["vtxX"][i]) ** 2
                 + (ak.to_numpy(rvy[i])[nu] - a["vtxY"][i]) ** 2
                 + (ak.to_numpy(rvz[i])[nu] - a["vtxZ"][i]) ** 2)
            flash_chi2[i] = float(ak.to_numpy(rvf[i])[nu[int(np.argmin(d))]])
        if not sel[i]:
            continue
        ei = np.nonzero(ak.to_numpy(is_e[i]))[0]
        E_sh = ak.to_numpy(a["showerRecoE"][i])
        k = ei[int(np.argmax(E_sh[ei]))]
        reco_ele_E[i] = float(E_sh[k])
        if have_lp:
            el_score[i] = float(ak.to_numpy(a["showerElScore"][i])[k])
            ph_score[i] = float(ak.to_numpy(a["showerPhScore"][i])[k])
            mu_score[i] = float(ak.to_numpy(a["showerMuScore"][i])[k])
            pi_score[i] = float(ak.to_numpy(a["showerPiScore"][i])[k])
            pr_score[i] = float(ak.to_numpy(a["showerPrScore"][i])[k])
            prim_score[i] = float(ak.to_numpy(a["showerPrimaryScore"][i])[k])
            fromneut_score[i] = float(ak.to_numpy(a["showerFromNeutralScore"][i])[k])
            fromchg_score[i] = float(ak.to_numpy(a["showerFromChargedScore"][i])[k])
        if have_lf:
            lf_el[i] = logclip(ak.to_numpy(a["showerLArFormerElScore"][i])[k])
            lf_ph[i] = logclip(ak.to_numpy(a["showerLArFormerPhScore"][i])[k])
            lf_mu[i] = logclip(ak.to_numpy(a["showerLArFormerMuScore"][i])[k])
            lf_pi[i] = logclip(ak.to_numpy(a["showerLArFormerPiScore"][i])[k])
            lf_pr[i] = logclip(ak.to_numpy(a["showerLArFormerPrScore"][i])[k])
        for arr, br in ((cosmic_score, "showerCosmicScore"),
                        (novtx_score, "showerNoVtxScore"),
                        (objectness, "showerObjectness"),
                        (charge, "showerCharge"),
                        (chargefrac, "showerChargeFrac"),
                        (nhits, "showerNHits"),
                        (att_score, "showerAttScore"),
                        (att_confident, "showerAttConfident")):
            if has(br):
                arr[i] = float(ak.to_numpy(a[br][i])[k])
        if has("showerStream"):
            stream[i] = int(ak.to_numpy(a["showerStream"][i])[k])

        # --- photon counts at the e-vertex + vertex-less photons ------------
        svi = ak.to_numpy(a["showerVtxIdx"][i])
        pid_all = ak.to_numpy(a["showerLArFormerPID"][i])
        e_vtx = int(svi[k])
        gam = (pid_all == 22) & (E_sh > PHOTON_E_MIN)
        if e_vtx >= 0:
            at_vtx = gam & (svi == e_vtx)
            n_photons[i] = int(np.sum(at_vtx))
            if has("showerCosmicScore"):
                cs = ak.to_numpy(a["showerCosmicScore"][i])
                n_good_photons[i] = int(np.sum(at_vtx
                                               & (cs >= args.shower_bdt_wp)))
        if has("showerNoVtxScore") and has("showerStream"):
            nv = ak.to_numpy(a["showerNoVtxScore"][i])
            sstream = ak.to_numpy(a["showerStream"][i])
            n_novtx_photons[i] = int(np.sum(gam & (svi < 0) & (sstream == 0)
                                            & (nv >= args.novtx_bdt_wp)))

        # --- max muon score among the OTHER particles at the e-vertex -------
        if e_vtx >= 0:
            tvi = ak.to_numpy(a["trackVtxIdx"][i])
            other = (svi == e_vtx).copy()
            other[k] = False                       # exclude the e-shower itself
            cand = []
            if len(tvi):
                cand += ak.to_numpy(a["trackMuScore"][i])[tvi == e_vtx].tolist()
            cand += ak.to_numpy(a["showerMuScore"][i])[other].tolist()
            cand = [c for c in cand if np.isfinite(c)]
            if cand:
                vtx_mu_score[i] = float(max(cand))
            if have_lf:
                candlf = []
                if len(tvi):
                    candlf += ak.to_numpy(
                        a["trackLArFormerMuScore"][i])[tvi == e_vtx].tolist()
                candlf += ak.to_numpy(a["showerLArFormerMuScore"][i])[other].tolist()
                candlf = [c for c in candlf if np.isfinite(c)]
                if candlf:
                    vtx_lf_mu_score[i] = logclip(max(candlf))

        # --- truth of the candidate ----------------------------------------
        if args.data:
            continue
        if have_tpid:
            shower_true_pid[i] = int(ak.to_numpy(a["showerTruePID"][i])[k])
        for arr, br in ((shower_true_purity, "showerTruePurity"),
                        (shower_true_comp, "showerTrueComp"),
                        (shower_true_unl_purity, "showerTrueUnlabeledPurity")):
            if has(br):
                arr[i] = float(ak.to_numpy(a[br][i])[k])
        if has("showerTrueTID"):
            shower_true_tid[i] = int(ak.to_numpy(a["showerTrueTID"][i])[k])

        # trueSimPart tables for this event (also give the primary-e vis energy)
        spdg = ak.to_numpy(a["trueSimPartPDG"][i])
        sproc = ak.to_numpy(a["trueSimPartProcess"][i])
        sE = ak.to_numpy(a["trueSimPartE"][i])
        # the neutrino's primary electron: highest-E |PDG|==11 with Process==0.
        # Computed for EVERY event (not just signal) so the categoriser can ask
        # "is this THE primary electron?" in background events too.
        me = (np.abs(spdg) == 11) & (sproc == 0)
        sig_tid = -9
        if me.any():
            jj = np.nonzero(me)[0]
            jm = jj[int(np.argmax(sE[jj]))]
            if has("trueSimPartTID"):
                sig_tid = int(ak.to_numpy(a["trueSimPartTID"][i])[jm])
            if is_nuecc_fv[i] and has("trueSimPartPixelSumQ"):
                sq = ak.to_numpy(a["trueSimPartPixelSumQ"][i])
                if sq[jm] >= 0:
                    true_ele_vise[i] = A_GAMMA * float(sq[jm])
        if have_truthcat:
            T = ak.to_numpy(a["trueSimPartTID"][i])
            M = ak.to_numpy(a["trueSimPartMID"][i])
            tid_index = {int(v): j for j, v in enumerate(T)}
            bg_cat[i], bg_subcat[i] = categorize(
                shower_true_pid[i], shower_true_tid[i],
                shower_true_purity[i], shower_true_unl_purity[i],
                T, M, sproc, spdg, tid_index, sig_tid)

    # true_ele_vise for signal events that failed the reco selection (needed as
    # the efficiency DENOMINATOR, so it cannot be limited to selected events)
    if not args.data and has("trueSimPartPixelSumQ"):
        todo = np.nonzero(is_nuecc_fv & ~np.isfinite(true_ele_vise))[0]
        for i in todo:
            spdg = np.abs(ak.to_numpy(a["trueSimPartPDG"][i]))
            sproc = ak.to_numpy(a["trueSimPartProcess"][i])
            sE = ak.to_numpy(a["trueSimPartE"][i])
            sq = ak.to_numpy(a["trueSimPartPixelSumQ"][i])
            m = (spdg == 11) & (sproc == 0)
            if m.any():
                jj = np.nonzero(m)[0]
                jm = jj[int(np.argmax(sE[jj]))]
                if sq[jm] >= 0:
                    true_ele_vise[i] = A_GAMMA * float(sq[jm])

    # ---- cutflow ----------------------------------------------------------
    print("\n== CUTFLOW (raw | weighted) ==")
    for lab, m in (("all", np.ones(n, bool)),
                   ("reco nu-vtx in FV", vtx_ok),
                   (">=1 primary e shower (sel)", sel),
                   ("  of which true nueCC-FV", sel & is_nuecc_fv)):
        print(f"  {lab:32s} {int(m.sum()):7d} | {w[m].sum():10.2f}")
    if not args.data:
        print(f"  true nueCC (veto flag) total : {int(is_nuecc.sum())} "
              f"(FV+e signal {int(is_nuecc_fv.sum())})")
        if have_truthcat:
            print("\n== CANDIDATE TRUTH CATEGORIES (selected events) ==")
            msel = sel & np.isfinite(reco_ele_E)
            for ci, name in enumerate(C.TRUTH_CATS):
                mc = msel & (bg_cat == ci)
                if mc.any():
                    print(f"  {name:20s} {int(mc.sum()):7d} | "
                          f"{w[mc].sum():10.2f}")

    np.savez(args.out,
             row=np.arange(n, dtype=np.int64),
             run=np.asarray(a["run"]), subrun=np.asarray(a["subrun"]),
             event=np.asarray(a["event"]),
             w=w, sel=sel, reco_ele_E=reco_ele_E, flash_chi2=flash_chi2,
             nu_pdg=pdg, ccnc=ccnc,
             is_nuecc=is_nuecc, is_nuecc_fv=is_nuecc_fv,
             el_score=el_score, ph_score=ph_score, mu_score=mu_score,
             pi_score=pi_score, pr_score=pr_score, prim_score=prim_score,
             fromneut_score=fromneut_score, fromchg_score=fromchg_score,
             vtx_mu_score=vtx_mu_score, vtx_dist_true=vtx_dist_true,
             n_photons=n_photons, n_good_photons=n_good_photons,
             n_novtx_photons=n_novtx_photons,
             shower_true_pid=shower_true_pid, shower_true_tid=shower_true_tid,
             shower_true_purity=shower_true_purity,
             shower_true_comp=shower_true_comp,
             shower_true_unl_purity=shower_true_unl_purity,
             bg_cat=bg_cat, bg_subcat=bg_subcat,
             true_ele_ke=true_ele_ke, true_ele_vise=true_ele_vise,
             true_nu_e=true_nu_e,
             lf_el_score=lf_el, lf_ph_score=lf_ph, lf_mu_score=lf_mu,
             lf_pi_score=lf_pi, lf_pr_score=lf_pr,
             vtx_lf_mu_score=vtx_lf_mu_score,
             cosmic_score=cosmic_score, novtx_score=novtx_score,
             objectness=objectness, charge=charge, chargefrac=chargefrac,
             nhits=nhits, att_score=att_score, att_confident=att_confident,
             stream=stream,
             vtx_score=vtx_score, vtx_frac_cosmic=vtx_frac_cosmic,
             nu_slice_chi2=nu_slice_chi2, fm_slice_chi2=fm_slice_chi2,
             nu_slice_nparts=nu_slice_nparts, fm_slice_nparts=fm_slice_nparts,
             pot=(0.0 if args.data else pot_sum), is_data=args.data,
             sample_set=np.array(args.sample_set), kept=keep)
    print(f"\n>>> wrote {args.out}  (sel={int(sel.sum())}, "
          f"weighted={w[sel].sum():.2f})")


if __name__ == "__main__":
    main()
