"""nue CC inclusive selection -- per-sample TABLE builder (ntuple-only).

Reads ONE gen2ntuple and writes a per-event selection table (.npz) that the
overlay script (nue_cc_overlay.py) stacks into a data/MC/EXT prediction. Mirrors
the pi0mass_peak two-stage pattern (pi0_mass_analysis.py -> datamc_ext_overlay.py).

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
  flash_chi2 = the primary nu-stream reco vertex's recoVtxFlashChi2 (the samples
  were reprocessed with the dead-PMT + saturation flash fixes, so the stored
  value is already corrected). The flash-chi2 CUT is applied in the overlay.

Weights: MC w = xsecWeight * (--pot / sum potTree.totGoodPOT); data/EXT unit
(EXT gets its spill weight in the overlay).

    PYTHONPATH=./ python3 nue_cc_analysis.py --ntuple ....root --out tab.npz [--data]
"""
import argparse
import os

import numpy as np
import uproot
import awkward as ak

ELE_E_MIN = 20.0      # reco primary-electron shower energy floor [MeV]
TRUE_ELE_E_MIN = 20.0  # true primary-electron energy floor [MeV] (signal def)
M_E = 0.000511         # GeV
A_GAMMA = 0.0253017    # MeV/ADC -- EM (e/gamma) visible-energy calib (pi0 conv.)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ntuple", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pot", type=float, default=4.4e19,
                    help="target POT for MC scaling (Tufts bnb5e19 livetime)")
    ap.add_argument("--data", action="store_true",
                    help="real-data mode: unit weights, no truth tags")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    fin = uproot.open(args.ntuple)
    if args.data:
        scale = 1.0
        print(">>> DATA mode: unit weights, no truth tags")
    else:
        pot = fin["potTree"].arrays(library="np")
        pot_sum = float(np.sum(pot["totGoodPOT"])) or float(np.sum(pot["totPOT"]))
        scale = args.pot / pot_sum
        print(f">>> sample POT {pot_sum:.3e}, target {args.pot:.2e} "
              f"-> scale {scale:.4e}")

    t = fin["EventTree"]
    have = set(t.keys())
    want = ["run", "subrun", "event", "xsecWeight",
            "trueNuE", "trueNuPDG", "trueNuCCNC", "trueVtxInWCFV",
            "trueLepE", "trueLepPDG",     # final-state lepton (electron for nueCC)
            "truePrimPartPDG", "truePrimPartE",
            # true sim particles -> primary-electron visible energy (pixel charge)
            "trueSimPartPDG", "trueSimPartProcess", "trueSimPartE",
            "trueSimPartPixelSumQ",
            "showerTruePID",              # truth-matched PID of the reco e-shower
            "foundVertex", "primaryVtxStream", "vtxIsFiducial",
            "vtxX", "vtxY", "vtxZ",
            "vtxDistToTrue",          # reco-vtx to SCE-corrected true-vtx [cm]
            "recoVtxX", "recoVtxY", "recoVtxZ", "recoVtxStream",
            "recoVtxFlashChi2",
            "showerLArFormerPID", "showerRecoE", "showerIsSecondary",
            "showerVtxIdx",
            # LArPID PID log-softmax [e, gamma, mu, pi, p] + process log-softmax
            # [primary, from-neutral, from-charged] -- all LOG-probabilities.
            "showerElScore", "showerPhScore", "showerMuScore",
            "showerPiScore", "showerPrScore",
            "showerPrimaryScore",
            "showerFromNeutralScore", "showerFromChargedScore",
            # tracks (for the "muon at the e-shower's vertex" background tag)
            "trackVtxIdx", "trackMuScore",
            # LArFormer (particle-segmentation model) PID -- these are softmax
            # PROBABILITIES [e,gamma,mu,pi,p] (in-domain vs LArPID's out-of-domain
            # CNN). Stored as log(prob) so the same confidence formulas apply.
            "showerLArFormerElScore", "showerLArFormerPhScore",
            "showerLArFormerMuScore", "showerLArFormerPiScore",
            "showerLArFormerPrScore", "trackLArFormerMuScore"]
    a = t.arrays([b for b in want if b in have])
    n = len(a["run"])

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

    # ---- truth kinematics (MC): true e KE + visible E (for efficiency plots)
    true_ele_ke = np.full(n, np.nan)     # true electron KE [MeV]
    true_ele_vise = np.full(n, np.nan)   # true electron visible energy [MeV]
    true_nu_e = np.full(n, np.nan)       # true neutrino energy [GeV]
    if not args.data:
        nuE = np.asarray(a["trueNuE"], float)
        true_nu_e = np.where(nuE > 0, nuE, np.nan)            # GeV
        lepE = np.asarray(a["trueLepE"], float)               # GeV, FS lepton
        leppdg = np.asarray(a["trueLepPDG"])
        true_ele_ke = np.where((np.abs(leppdg) == 11) & (lepE > 0),
                               (lepE - M_E) * 1000.0, np.nan)
        # visible E = A_GAMMA * de-double-counted pixel charge of the PRIMARY
        # electron (highest-E trueSimPart with |PDG|==11, Process==0=primary).
        for i in np.nonzero(is_nuecc_fv)[0]:
            spdg = np.abs(ak.to_numpy(a["trueSimPartPDG"][i]))
            sproc = ak.to_numpy(a["trueSimPartProcess"][i])
            sE = ak.to_numpy(a["trueSimPartE"][i])
            sq = ak.to_numpy(a["trueSimPartPixelSumQ"][i])
            m = (spdg == 11) & (sproc == 0)
            if m.any():
                jj = np.nonzero(m)[0]; jm = jj[int(np.argmax(sE[jj]))]
                if sq[jm] >= 0:
                    true_ele_vise[i] = A_GAMMA * float(sq[jm])

    # ---- reco selection ---------------------------------------------------
    vtx_ok = ((np.asarray(a["foundVertex"]) == 1)
              & (np.asarray(a["primaryVtxStream"]) == 0)
              & (np.asarray(a["vtxIsFiducial"]) == 1))
    is_e = ((a["showerLArFormerPID"] == 11) & (a["showerIsSecondary"] == 0)
            & (a["showerRecoE"] > ELE_E_MIN))
    n_e = ak.to_numpy(ak.sum(is_e, axis=1))
    sel = vtx_ok & (n_e >= 1)

    # leading primary-electron shower energy + its LArPID scores
    reco_ele_E = np.full(n, np.nan)
    el_score = np.full(n, np.nan); ph_score = np.full(n, np.nan)
    mu_score = np.full(n, np.nan); pi_score = np.full(n, np.nan)
    pr_score = np.full(n, np.nan)
    prim_score = np.full(n, np.nan)
    fromneut_score = np.full(n, np.nan)
    fromchg_score = np.full(n, np.nan)
    flash_chi2 = np.full(n, np.nan)
    # (1) max muon score among the OTHER particles sharing the e-shower's vertex
    #     (targets true-e showers from a decay mu/pi merged into a track);
    #     NaN when the e-shower has no reco vertex or no other particle there.
    vtx_mu_score = np.full(n, np.nan)
    # (2) reco-vtx to SCE-corrected true-vtx distance [cm] (MC only). A long
    #     non-nueCC tail flags secondary-interaction (n -> pi->mu->e) topologies.
    vtx_dist_true = (np.asarray(a["vtxDistToTrue"], np.float64)
                     if "vtxDistToTrue" in have and not args.data
                     else np.full(n, np.nan))
    # reco photons (LArFormerPID==22, RecoE>20 MeV) attached to the nu vertex --
    # a pi0 / mis-identified-gamma tag (most photons come from pi0 decays, so
    # extra photons at the nu interaction flag pi0-containing backgrounds).
    n_photons = np.full(n, -1, np.int64)
    # truth-matched PID (PDG) of the reco'd leading e-shower (for the background
    # "what did the electron actually come from" plot). 0 = unmatched/data.
    shower_true_pid = np.zeros(n, np.int64)
    have_tpid = "showerTruePID" in have
    # LArFormer (log of the softmax probability) analogues of the LArPID scores
    lf_el = np.full(n, np.nan); lf_ph = np.full(n, np.nan)
    lf_mu = np.full(n, np.nan); lf_pi = np.full(n, np.nan)
    lf_pr = np.full(n, np.nan); vtx_lf_mu_score = np.full(n, np.nan)
    have_lp = "showerElScore" in have
    have_lf = "showerLArFormerElScore" in have

    def logclip(p):
        return float(np.log(np.clip(float(p), 1e-6, 1.0)))

    rvs = a["recoVtxStream"]; rvf = a["recoVtxFlashChi2"]
    rvx = a["recoVtxX"]; rvy = a["recoVtxY"]; rvz = a["recoVtxZ"]
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
        E = ak.to_numpy(a["showerRecoE"][i])[ei]
        k = ei[int(np.argmax(E))]
        reco_ele_E[i] = float(ak.to_numpy(a["showerRecoE"][i])[k])
        if have_tpid:
            shower_true_pid[i] = int(ak.to_numpy(a["showerTruePID"][i])[k])
        if have_lp:
            el_score[i] = float(ak.to_numpy(a["showerElScore"][i])[k])
            ph_score[i] = float(ak.to_numpy(a["showerPhScore"][i])[k])
            mu_score[i] = float(ak.to_numpy(a["showerMuScore"][i])[k])
            pi_score[i] = float(ak.to_numpy(a["showerPiScore"][i])[k])
            pr_score[i] = float(ak.to_numpy(a["showerPrScore"][i])[k])
            prim_score[i] = float(ak.to_numpy(a["showerPrimaryScore"][i])[k])
            fromneut_score[i] = float(ak.to_numpy(a["showerFromNeutralScore"][i])[k])
            fromchg_score[i] = float(ak.to_numpy(a["showerFromChargedScore"][i])[k])
        if have_lf:                       # LArFormer log-probs (same convention)
            lf_el[i] = logclip(ak.to_numpy(a["showerLArFormerElScore"][i])[k])
            lf_ph[i] = logclip(ak.to_numpy(a["showerLArFormerPhScore"][i])[k])
            lf_mu[i] = logclip(ak.to_numpy(a["showerLArFormerMuScore"][i])[k])
            lf_pi[i] = logclip(ak.to_numpy(a["showerLArFormerPiScore"][i])[k])
            lf_pr[i] = logclip(ak.to_numpy(a["showerLArFormerPrScore"][i])[k])
        # (1) max muon score among OTHER particles at the e-shower's vertex
        #     -- computed for BOTH LArPID and LArFormer (max prob -> log).
        e_vtx = int(ak.to_numpy(a["showerVtxIdx"][i])[k])
        if e_vtx >= 0:
            tvi = ak.to_numpy(a["trackVtxIdx"][i])
            svi = ak.to_numpy(a["showerVtxIdx"][i])
            other = (svi == e_vtx); other[k] = False   # exclude the e-shower
            # reco photons attached to the nu vertex (pi0 / mis-id-gamma tag)
            pid_all = ak.to_numpy(a["showerLArFormerPID"][i])
            E_all = ak.to_numpy(a["showerRecoE"][i])
            n_photons[i] = int(np.sum((pid_all == 22) & (E_all > 20.0)
                                      & (svi == e_vtx)))
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

    np.savez(args.out,
             run=np.asarray(a["run"]), subrun=np.asarray(a["subrun"]),
             event=np.asarray(a["event"]),
             w=w, sel=sel, reco_ele_E=reco_ele_E, flash_chi2=flash_chi2,
             nu_pdg=pdg, ccnc=ccnc,
             is_nuecc=is_nuecc, is_nuecc_fv=is_nuecc_fv,
             el_score=el_score, ph_score=ph_score, mu_score=mu_score,
             pi_score=pi_score, pr_score=pr_score, prim_score=prim_score,
             fromneut_score=fromneut_score, fromchg_score=fromchg_score,
             vtx_mu_score=vtx_mu_score, vtx_dist_true=vtx_dist_true,
             n_photons=n_photons, shower_true_pid=shower_true_pid,
             true_ele_ke=true_ele_ke, true_ele_vise=true_ele_vise,
             true_nu_e=true_nu_e,
             lf_el_score=lf_el, lf_ph_score=lf_ph, lf_mu_score=lf_mu,
             lf_pi_score=lf_pi, lf_pr_score=lf_pr,
             vtx_lf_mu_score=vtx_lf_mu_score,
             pot=(0.0 if args.data else pot_sum), is_data=args.data)
    print(f">>> wrote {args.out}  (sel={int(sel.sum())}, weighted={w[sel].sum():.2f})")


if __name__ == "__main__":
    main()
