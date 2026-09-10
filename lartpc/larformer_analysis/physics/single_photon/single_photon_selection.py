"""Inclusive single-photon (1gamma + X, MiniBooNE-style visibility) selection
on the LArFormer gen2ntuple ROOT files (v2_s1ep2p8 chain, ntuple-only).

TRUTH signal (MC overlay; per README):
  true vertex in the WireCell FV (trueVtxInWCFV==1) and, over ALL trueSimPart
  entries (primary + secondary; overlay => all nu-induced, no cosmic truth):
    visible photon   : |PDG|==22 and E_vis = A_GAMMA x trueSimPartPixelSumQ >= 20 MeV
    visible electron : |PDG|==11 and E_vis >= 20 MeV
    visible muon     : |PDG|==13 and KE = E - m_mu >= 100 MeV (Cherenkov thr.)
    charged pions / protons / other hadrons: NEVER visible (MiniBooNE-blind)
  Visible mu / X particles must also deposit charge in the TPC (PixelSumQ>0)
  so that particles of out-of-TPC interactions that never enter do not veto.
  signal (inclusive 1g+X) = exactly 1 visible photon, 0 visible e, 0 visible mu,
  for ANY true vertex position: an entering photon from a nu interaction
  outside the FV/TPC is signal too (MicroBooNE + SBND convention, 2026-09-05)
  and is tracked as its own category.
  1g0X (strict) sub-sample = in-FV signal AND no other charged particle above
  the MicroBooNE-detector thresholds: mu / charged mesons KE >= 30 MeV,
  protons / charged baryons KE >= 60 MeV (neutrons, K0, Lambda, nuclei never).
  "rest" = in-FV signal but not 1g0X (1g + N visible-to-MicroBooNE X).

MC truth categories (stack colours) in priority order:
  0 sig 1g0X (in FV) | 1 sig 1g+X (in FV) | 2 sig entering g, no muon above
  100 MeV anywhere | 3 sig entering g, CC: a muon with KE >= 100 MeV exists but
  never deposits in the TPC (2026-09-06 split; MiniBooNE would have seen it) |
  out-of-FV backgrounds: 4 entering >=2 visible photons | 5 entering CC
  (visible mu or e entered; takes priority) | 6 entering other (no visible
  photon, no lepton) | in-FV backgrounds: 7 visible mu (numu CC-like) |
  8 visible e, no mu (nue CC-like) | 9 >=2 visible photons | 10 no visible
  photon | 11 EXT-BNB cosmic (from the EXT ntuple)

RECO selection (cumulative cut steps, nu-stream only). The per-shower cosmic
BDT (showerCosmicScore) DEFINES the photon-candidate set (decision 2026-09-05:
cosmics dominate, so photons failing the BDT are treated as cosmic and do not
count in the multiplicity):
  S1 reco vertex : foundVertex & primaryVtxStream==0 & vtxIsFiducial
  S2 1 photon    : exactly ONE candidate = shower with showerLArFormerPID==22,
                   recal'd showerRecoE > 20 MeV and showerCosmicScore >=
                   --shower-bdt-min (opt. --require-confident / --primary-only)
  S3 no vis X    : no track (primary or secondary) with muon PID (segmenter /
                   larpid / union) and trackRecoE > --mu-ke-min; no PID==11
                   shower > 20 MeV
  S4 flash chi2  : nu-vertex recoVtxFlashChi2 < --flashchi2-cut
--bdt-after-multiplicity restores the earlier ordering (count all photons,
then cut the single candidate's score as a separate step).

VERTEX-LESS PATH (--novtx; needs the 2026-09-06 "novtx" ntuples that carry
showerVtxIdx=-1 prongs, showerNoVtxScore, showerStream, nuSliceFlashChi2):
photon showers of the nu-stream slice that are NOT attached to the primary
in-FV nu vertex (vertex-less, or attached to a non-primary / out-of-FV
vertex) become candidates via the vertex-free BDT (showerNoVtxScore >=
--novtx-bdt-min; --novtx-model <joblib> computes it from the ntuple
branches when the exported branch is -9). The steps then read
  S1 nu slice   : >=1 nu-stream photon shower > 20 MeV (either path)
  S2 1 photon   : exactly ONE candidate over both paths
  S3 no vis X   : as above, over ALL nu-stream tracks/showers (orphans too)
  S4 flash chi2 : nuSliceFlashChi2 < --flashchi2-cut
The vertex-free BDT was trained on EVEN MC events: pass --mc-odd-only
(odd events, w x2) whenever --novtx is used.
Shower energies: the novtx ntuples are exported with the deployed calib
baked in, so leave --recal-gamma-a unset for them; the older s1ep2p8
ntuples need --recal-gamma-a 0.01553 --recal-gamma-b -12.80.
The flash chi2 is read from the ntuple (recoVtxFlashChi2 of the recoVtxStream==0
vertex); it equals the dead-PMT/saturation-masked chi2 used by the pi0 study.

Weights: MC xsecWeight scaled to --pot (sum potTree totGoodPOT); EXT rows <
--ext-row-min (per-shower BDT training half) dropped, remaining x --ext-scale;
beam data (--data-ntuple, 4.4e19 POT) unit weights, overlaid as points.

Outputs (--plots): stacked candidate-photon energy per step (+ data points),
N-1 flash-chi2, efficiency vs true photon energy (E_vis and true E) per step,
purity vs reco photon energy per step, BDT / chi2 threshold scans,
cutflow.txt; --out-dir gets row-aligned selection tables ({mc,ext,data}_table.npz).

    python3 single_photon_selection.py --mc-ntuple ... --ext-ntuple ... \
        --data-ntuple ... --plots plots_v2_s1ep2p8 --out-dir workdir
"""
import argparse
import os

import numpy as np
import uproot
import awkward as ak

A_GAMMA = 0.0253017            # MeV/ADC truth visible-energy convention
EVIS_MIN = 20.0                # visible shower threshold [MeV]
MU_KE_VIS = 100.0              # MiniBooNE-mimic muon visibility [MeV KE]
X_KE_MESON = 30.0              # 1g0X strict: mu / charged mesons [MeV KE]
X_KE_BARYON = 60.0             # 1g0X strict: protons / charged baryons
RECO_G_MIN = 20.0              # reco photon energy cut [MeV]
NTUP_G_A = 0.020100999623537064   # deployed gamma calib in the s1ep2p8 ntuples
NTUP_G_B = -15.489999771118164

MASS = {11: 0.511, 13: 105.658, 211: 139.570, 321: 493.677, 2212: 938.272,
        2112: 939.565, 3222: 1189.37, 3112: 1197.45, 3312: 1321.71,
        3334: 1672.45, 3122: 1115.68, 22: 0.0, 111: 134.977}
CHARGED_MESONS = (13, 211, 321)              # mu counted with mesons (README)
CHARGED_BARYONS = (2212, 3222, 3112, 3312, 3334)

CATS = ["sig 1g0X (in FV)", "sig 1g+X (in FV)", "sig entering g (no mu>100)",
        "sig entering g, CC (mu>100 outside TPC)",
        "bkg: entering >=2 g", "bkg: entering CC (vis mu/e)",
        "bkg: entering other", "bkg: visible mu (CC numu)",
        "bkg: visible e (CC nue)", "bkg: >=2 visible g",
        "bkg: no visible g", "EXT cosmic"]
CAT_COLORS = ["#d62728", "#ff9896", "#ff7f0e", "#ffbb78", "#c49c94", "#17becf",
              "#bdbdbd", "#1f77b4", "#2ca02c", "#9467bd", "#8c564b", "#7f7f7f"]
N_MC_CAT = 11                # MC categories; index N_MC_CAT = EXT
N_SIG_CAT = 4                # categories 0..3 are signal

BR_RECO = ["run", "subrun", "event", "foundVertex", "primaryVtxStream",
           "vtxIsFiducial", "recoVtxStream", "recoVtxFlashChi2",
           "showerLArFormerPID", "showerRecoE", "showerCosmicScore",
           "showerAttConfident", "showerIsSecondary", "showerDistToVtx",
           "trackLArFormerPID", "trackIsSecondary", "trackRecoE",
           "trackClassified", "trackMuScore", "trackElScore", "trackPhScore",
           "trackPiScore", "trackPrScore"]
BR_NOVTX = ["showerVtxIdx", "showerStream", "showerNoVtxScore",
            "nuSliceFlashChi2", "nuSliceNParticles", "fmSliceNParticles",
            "trackVtxIdx", "trackStream",
            # features for --novtx-model (analysis-side scoring)
            "showerCosTheta", "showerCosThetaY", "showerStartPosX",
            "showerStartPosY", "showerStartPosZ", "showerObjectness",
            "showerLArFormerPhScore", "showerLArFormerElScore",
            "showerLArFormerMuScore", "showerLArFormerPiScore",
            "showerLArFormerPrScore", "showerNHits", "showerCharge",
            "showerChargeFrac"]
TPC_LO = np.array([0.0, -116.5, 0.0]); TPC_HI = np.array([256.35, 116.5, 1036.8])
BR_MC = ["xsecWeight", "trueVtxInWCFV", "trueNuCCNC", "trueSimPartPDG",
         "trueSimPartE", "trueSimPartPixelSumQ", "trueSimPartProcess",
         "trueSimPartMID", "trueSimPartTID"]


def _mass_of(pdg):
    m = ak.zeros_like(pdg, dtype=np.float64)
    for p, mm in MASS.items():
        m = ak.where(np.abs(pdg) == p, mm, m)
    return m


def truth_tables(a):
    """Per-event truth arrays from trueSimPart*; returns dict of numpy arrays."""
    pdg = a["trueSimPartPDG"]
    apdg = np.abs(pdg)
    E = a["trueSimPartE"]
    ke = E - _mass_of(pdg)
    q = a["trueSimPartPixelSumQ"]
    evis = A_GAMMA * np.maximum(q, 0.0)
    dep = q > 0                                  # deposits charge in the TPC
    vis_g = (apdg == 22) & (evis >= EVIS_MIN)
    vis_e = (apdg == 11) & (evis >= EVIS_MIN)
    vis_mu = (apdg == 13) & (ke >= MU_KE_VIS) & dep
    mu100_any = (apdg == 13) & (ke >= MU_KE_VIS)     # incl. muons outside the TPC
    is_meson = ak.zeros_like(pdg, dtype=bool)
    for p in CHARGED_MESONS:
        is_meson = is_meson | (apdg == p)
    is_baryon = ak.zeros_like(pdg, dtype=bool)
    for p in CHARGED_BARYONS:
        is_baryon = is_baryon | (apdg == p)
    x_strict = ((is_meson & (ke >= X_KE_MESON))
                | (is_baryon & (ke >= X_KE_BARYON))) & dep

    n = lambda m: ak.to_numpy(ak.sum(m, axis=1)).astype(np.int64)
    out = dict(n_vis_g=n(vis_g), n_vis_e=n(vis_e), n_vis_mu=n(vis_mu),
               n_x_strict=n(x_strict), n_mu100_any=n(mu100_any))
    fv = ak.to_numpy(a["trueVtxInWCFV"]) == 1
    sig = (out["n_vis_g"] == 1) & (out["n_vis_e"] == 0) & (out["n_vis_mu"] == 0)
    out["sig_incl"] = sig                        # any vertex (entering included)
    out["sig_1g0x"] = sig & fv & (out["n_x_strict"] == 0)
    out["sig_enter"] = sig & ~fv
    out["sig_enter_mu"] = out["sig_enter"] & (out["n_mu100_any"] >= 1)
    out["in_fv"] = fv
    # the (single) visible photon: E_vis, true E, process, TID, mother-in-table
    def first(x):
        v = ak.to_numpy(ak.fill_none(ak.firsts(x[vis_g]), np.nan))
        return np.asarray(v, np.float64)
    out["g_evis"] = first(evis)
    out["g_trueE"] = first(E)
    out["g_process"] = first(a["trueSimPartProcess"])
    out["g_tid"] = first(a["trueSimPartTID"])
    # mother of the signal photon present in the trueSimPart table? (a primary
    # pi0 is dropped from the table, so False = primary-pi0 / direct photon)
    mid_g = a["trueSimPartMID"][vis_g]
    tids = a["trueSimPartTID"]
    tid_in = np.zeros(len(fv), bool)
    for i in np.nonzero(sig)[0]:
        tid_in[i] = int(mid_g[i][0]) in set(np.asarray(tids[i]).tolist())
    out["g_mother_in_table"] = tid_in
    cat = np.full(len(fv), 10, np.int64)
    cat[out["n_vis_g"] >= 2] = 9
    cat[out["n_vis_e"] >= 1] = 8
    cat[out["n_vis_mu"] >= 1] = 7
    lep = (out["n_vis_mu"] >= 1) | (out["n_vis_e"] >= 1)
    cat[~fv] = 6
    cat[~fv & (out["n_vis_g"] >= 2)] = 4
    cat[~fv & lep] = 5
    cat[out["sig_enter"]] = 2
    cat[out["sig_enter_mu"]] = 3
    cat[out["sig_incl"] & fv] = 1
    cat[out["sig_1g0x"]] = 0
    out["cat"] = cat
    out["ccnc"] = ak.to_numpy(a["trueNuCCNC"]).astype(np.int64)
    return out


def recal_E(a, ga, gb):
    spid = a["showerLArFormerPID"]
    E = a["showerRecoE"]
    if ga is None:
        return E
    return ak.where(spid == 22, (E - NTUP_G_B) / NTUP_G_A * ga + gb, E)


def novtx_scores(a, model_path):
    """Analysis-side showerNoVtxScore from the ntuple branches (same feature
    names as the exporter's novtx_features / shower_novtx_bdt.py)."""
    import joblib
    M = joblib.load(model_path)
    st = [a["showerStartPosX"], a["showerStartPosY"], a["showerStartPosZ"]]
    dw = ak.zeros_like(a["showerRecoE"]) + 1e9
    for k, c in enumerate(st):
        dw = np.minimum(dw, np.minimum(c - TPC_LO[k], TPC_HI[k] - c))
    stream = a["showerStream"]
    nsp = ak.where(stream == 0, a["nuSliceNParticles"], a["fmSliceNParticles"])
    F = {"E": a["showerRecoE"], "cosZ": a["showerCosTheta"],
         "cosY": a["showerCosThetaY"], "sdwall": dw, "startX": st[0],
         "startY": st[1], "startZ": st[2], "objectness": a["showerObjectness"],
         "phS": a["showerLArFormerPhScore"], "elS": a["showerLArFormerElScore"],
         "muS": a["showerLArFormerMuScore"], "piS": a["showerLArFormerPiScore"],
         "prS": a["showerLArFormerPrScore"], "nhits": a["showerNHits"],
         "charge": a["showerCharge"], "chargefrac": a["showerChargeFrac"],
         "hasVtx": ak.values_astype(a["showerVtxIdx"] >= 0, np.float64),
         "stream": stream, "nSlicePart": nsp}
    cols = [ak.to_numpy(ak.flatten(F[k])).astype(np.float64) for k in M["feats"]]
    if not len(cols[0]):
        return a["showerNoVtxScore"]
    sc = M["clf"].predict_proba(np.column_stack(cols))[:, 1]
    flat = ak.flatten(a["showerRecoE"])
    is_g = ak.to_numpy(ak.flatten((a["showerLArFormerPID"] == 22)
                                  & (a["showerRecoE"] > 0)))
    sc = np.where(is_g, sc, -9.0)
    return ak.unflatten(sc, ak.num(a["showerRecoE"]))


def reco_tables(a, args):
    """Per-event reco arrays (numpy) + per-shower awkward arrays needed to
    (re)build the cut steps at any BDT threshold. Works for MC, EXT, data."""
    spid = a["showerLArFormerPID"]
    E = recal_E(a, args.recal_gamma_a, args.recal_gamma_b)
    vtx_ok = ((ak.to_numpy(a["foundVertex"]) == 1)
              & (ak.to_numpy(a["primaryVtxStream"]) == 0)
              & (ak.to_numpy(a["vtxIsFiducial"]) == 1))
    nu = a["recoVtxStream"] == 0
    chi2 = np.asarray(ak.to_numpy(ak.fill_none(
        ak.firsts(a["recoVtxFlashChi2"][nu]), np.nan)), np.float64)
    chi2[~vtx_ok] = np.nan

    is_g0 = (spid == 22) & (E > RECO_G_MIN)       # photon showers before the BDT
    if args.require_confident:
        is_g0 = is_g0 & (a["showerAttConfident"] == 1)
    if args.primary_only:
        is_g0 = is_g0 & (a["showerIsSecondary"] == 0)
    score = a["showerCosmicScore"]
    if args.novtx:
        # path A: attached to the primary in-FV nu vertex -> cosmic BDT;
        # path B: any other nu-stream photon shower -> vertex-free BDT
        vtxA = (a["showerVtxIdx"] == 0) & vtx_ok
        nu_stream = a["showerStream"] == 0
        nv = (a["showerNoVtxScore"] if args.novtx_model is None
              else novtx_scores(a, args.novtx_model))
        pathA = is_g0 & vtxA
        pathB = is_g0 & nu_stream & ~vtxA
        is_g0 = pathA | pathB
        # unify: a "score" that is compared to ONE threshold in build_steps:
        # path A showers carry cosmic score - thr_A + thr_B offset so the
        # single --shower-bdt-min comparison works for both paths
        score = ak.where(pathA, score,
                         nv - args.novtx_bdt_min + args.shower_bdt_min)
        chi2 = np.asarray(ak.to_numpy(a["nuSliceFlashChi2"]), np.float64)
        chi2[chi2 <= 0] = np.nan
        n_e = ak.to_numpy(ak.sum((spid == 11) & (E > RECO_G_MIN) & nu_stream,
                                 axis=1)).astype(np.int64)
        path_flag = ak.values_astype(pathB, np.int64)   # 1 = vertex-less path
    else:
        n_e = ak.to_numpy(ak.sum((spid == 11) & (E > RECO_G_MIN), axis=1)).astype(np.int64)
        path_flag = ak.zeros_like(spid)
    n_g0 = ak.to_numpy(ak.sum(is_g0, axis=1)).astype(np.int64)

    seg_mu = a["trackLArFormerPID"] == 13
    lp_mu = ((a["trackClassified"] == 1)
             & (a["trackMuScore"] > a["trackElScore"])
             & (a["trackMuScore"] > a["trackPhScore"])
             & (a["trackMuScore"] > a["trackPiScore"])
             & (a["trackMuScore"] > a["trackPrScore"]))
    mu_pid = {"segmenter": seg_mu, "larpid": lp_mu,
              "union": seg_mu | lp_mu}[args.muon_finder]
    is_mu = mu_pid & (a["trackRecoE"] > args.mu_ke_min)
    if args.mu_primary_only:
        is_mu = is_mu & (a["trackIsSecondary"] == 0)
    if args.novtx:
        is_mu = is_mu & (a["trackStream"] == 0)
    n_mu = ak.to_numpy(ak.sum(is_mu, axis=1)).astype(np.int64)
    s1 = vtx_ok if not args.novtx else (n_g0 > 0)
    return dict(run=ak.to_numpy(a["run"]), subrun=ak.to_numpy(a["subrun"]),
                event=ak.to_numpy(a["event"]), vtx_ok=vtx_ok, chi2=chi2,
                n_g0=n_g0, n_e=n_e, n_mu=n_mu, s1=s1,
                _is_g0=is_g0, _E=E, _score=score, _path=path_flag,
                _dist=a["showerDistToVtx"])


def build_steps(R, thr, chi2_cut, bdt_after=False):
    """Cumulative step masks + candidate-photon arrays for a BDT threshold.
    Returns (names, steps[nstep, n], E_cand, score_cand, n_g)."""
    is_g = R["_is_g0"] if bdt_after else (R["_is_g0"] & (R["_score"] >= thr))
    n_g = ak.to_numpy(ak.sum(is_g, axis=1)).astype(np.int64)
    Eg = ak.where(is_g, R["_E"], -1.0)
    idx = ak.argmax(Eg, axis=1, keepdims=True)

    def pick(x, fill=np.nan):
        v = ak.fill_none(ak.firsts(x[idx]), fill)
        v = np.asarray(ak.to_numpy(v), np.float64)
        v[n_g == 0] = np.nan
        return v
    E_cand, score_cand = pick(R["_E"]), pick(R["_score"], -9.0)
    path_cand = pick(R["_path"], 0.0)
    R["_cand_idx"] = idx                 # per-event leading-candidate index (diagnostics)
    n = len(n_g)
    s1 = R["s1"]
    s2 = s1 & (n_g == 1)
    s3 = s2 & (R["n_mu"] == 0) & (R["n_e"] == 0)
    names = ["all", "S1 nu slice" if R.get("novtx") else "S1 reco vtx",
             "S2 1 photon", "S3 no vis X"]
    steps = [np.ones(n, bool), s1, s2, s3]
    if bdt_after:
        s3b = s3 & (score_cand >= thr)
        names.append("S4 photon BDT")
        steps.append(s3b)
        s3 = s3b
    s_last = s3 & np.isfinite(R["chi2"]) & (R["chi2"] < chi2_cut)
    names.append(f"S{len(steps)} flash chi2")
    steps.append(s_last)
    return names, np.stack(steps), E_cand, score_cand, n_g, path_cand


def load(ntuple, is_mc, novtx=False):
    fin = uproot.open(ntuple)
    t = fin["EventTree"]
    have = set(t.keys())
    want = BR_RECO + (BR_MC if is_mc else []) + (BR_NOVTX if novtx else [])
    missing = set(want) - have
    if missing:
        raise RuntimeError(f"{ntuple}: missing branches {sorted(missing)}")
    a = t.arrays(want)
    pot = None
    if is_mc:
        p = fin["potTree"].arrays(library="np")
        pot = float(np.sum(p["totGoodPOT"])) or float(np.sum(p["totPOT"]))
    return a, pot


def binned_ratio(num_w, den_w, x, edges, n_raw=None):
    """weighted ratio per bin (+ binomial-ish error from raw counts)."""
    e, er = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = (x >= lo) & (x < hi)
        d = den_w[b].sum()
        v = num_w[b].sum() / d if d > 0 else np.nan
        N = int(b.sum()) if n_raw is None else int(n_raw[b].sum())
        e.append(v)
        er.append(np.sqrt(max(v * (1 - v), 0) / N) if N and np.isfinite(v) else 0)
    return np.array(e), np.array(er)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mc-ntuple", required=True)
    ap.add_argument("--ext-ntuple", default=None)
    ap.add_argument("--data-ntuple", default=None,
                    help="beam data ntuple (4.4e19 POT); unit weights, drawn "
                         "as points over the MC+EXT stacks")
    ap.add_argument("--ext-scale", type=float, default=1.1818,
                    help="EXT weight for rows >= --ext-row-min (200k subset "
                         "spill scale 0.5909 x2 for the analysis half)")
    ap.add_argument("--ext-row-min", type=int, default=100000)
    ap.add_argument("--pot", type=float, default=4.4e19)
    ap.add_argument("--recal-gamma-a", type=float, default=None,
                    help="invert the deployed OLD gamma calib and re-apply "
                         "E=a*Q+b (needed for the pre-2026-09-06 s1ep2p8 "
                         "ntuples: 0.01553); default None = energies as "
                         "exported (novtx ntuples have the calib baked in)")
    ap.add_argument("--recal-gamma-b", type=float, default=-12.80)
    ap.add_argument("--novtx", action="store_true",
                    help="enable the vertex-less candidate path (novtx ntuples)")
    ap.add_argument("--novtx-bdt-min", type=float, default=0.5,
                    help="showerNoVtxScore threshold for path-B candidates")
    ap.add_argument("--novtx-model", default=None,
                    help="joblib from shower_novtx_bdt.py: compute the "
                         "vertex-free score from ntuple branches instead of "
                         "reading showerNoVtxScore")
    ap.add_argument("--mc-odd-only", action="store_true",
                    help="use only odd-event MC (w x2): required with the "
                         "vertex-free BDT, which trained on even events")
    ap.add_argument("--shower-bdt-min", type=float, default=0.192)
    ap.add_argument("--bdt-after-multiplicity", action="store_true",
                    help="count ALL photons > 20 MeV for the multiplicity cut "
                         "and apply the BDT to the single candidate afterwards "
                         "(the 2026-09-05 first-pass ordering)")
    ap.add_argument("--flashchi2-cut", type=float, default=316.2,
                    help="keep nu-vertex flash chi2 below this (default "
                         "10^2.5; the N-1 plot shows EXT rising above it)")
    ap.add_argument("--mu-ke-min", type=float, default=100.0)
    ap.add_argument("--muon-finder", default="union",
                    choices=["segmenter", "larpid", "union"])
    ap.add_argument("--mu-primary-only", action="store_true")
    ap.add_argument("--require-confident", action="store_true",
                    help="photon candidates must be showerAttConfident==1")
    ap.add_argument("--primary-only", action="store_true",
                    help="photon candidates must be showerIsSecondary==0")
    ap.add_argument("--plots", required=True)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    os.makedirs(args.plots, exist_ok=True)
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    # ---- MC -----------------------------------------------------------------
    print(">>> loading MC ...", flush=True)
    a, pot = load(args.mc_ntuple, True, args.novtx)
    scale = args.pot / pot
    print(f">>> MC POT {pot:.3e} -> scale {scale:.4f} to {args.pot:.2e}")
    w0 = ak.to_numpy(a["xsecWeight"]).astype(np.float64)
    w = np.where(w0 > 0, w0, 0.0) * scale
    if args.mc_odd_only:
        odd = ak.to_numpy(a["event"]) % 2 == 1
        w = np.where(odd, 2.0 * w, 0.0)
        print(f">>> MC odd-event half only: {int(odd.sum())} events, w x2")
    T = truth_tables(a)
    R = reco_tables(a, args)
    cat = T["cat"]
    n = len(w)
    del a
    print(f">>> MC {n} events | sig incl {int(T['sig_incl'].sum())} "
          f"(w {w[T['sig_incl']].sum():.1f}) | sig 1g0X "
          f"{int(T['sig_1g0x'].sum())} (w {w[T['sig_1g0x']].sum():.1f}) | "
          f"entering {int(T['sig_enter'].sum())} (w {w[T['sig_enter']].sum():.1f})")
    print("    truth category census (raw | weighted):")
    for c in range(N_MC_CAT):
        m = cat == c
        print(f"      {CATS[c]:28s} {int(m.sum()):7d} | {w[m].sum():9.1f}")
    ps = T["g_process"][T["sig_incl"]]
    print(f"    signal photon process: primary {int((ps==0).sum())} decay "
          f"{int((ps==1).sum())} other {int((ps==2).sum())}; mother in table "
          f"(secondary pi0 etc.) {int(T['g_mother_in_table'][T['sig_incl']].sum())}")
    print(f"    signal CC/NC: CC {int((T['ccnc'][T['sig_incl']]==0).sum())} "
          f"NC {int((T['ccnc'][T['sig_incl']]==1).sum())}")

    # ---- EXT / data -----------------------------------------------------------
    RX = RD = None
    wx = wd = None
    if args.ext_ntuple:
        print(">>> loading EXT ...", flush=True)
        ax, _ = load(args.ext_ntuple, False, args.novtx)
        RX = reco_tables(ax, args)
        del ax
        nx = len(RX["run"])
        wx = np.zeros(nx)
        wx[args.ext_row_min:] = args.ext_scale
        print(f">>> EXT {nx} rows; analysis rows {nx - args.ext_row_min} "
              f"x {args.ext_scale}")
    if args.data_ntuple:
        print(">>> loading beam data ...", flush=True)
        ad, _ = load(args.data_ntuple, False, args.novtx)
        RD = reco_tables(ad, args)
        del ad
        wd = np.ones(len(RD["run"]))
        print(f">>> data {len(wd)} events (unit weight)")

    thr, ccut, after = args.shower_bdt_min, args.flashchi2_cut, args.bdt_after_multiplicity
    for RR in (R, RX, RD):
        if RR is not None:
            RR["novtx"] = args.novtx
    STEPS, S, E_cand, score_cand, n_g, path_cand = build_steps(R, thr, ccut, after)
    SX = EX = None
    if RX is not None:
        _, SX, EX, _, _, PX = build_steps(RX, thr, ccut, after)
    SD = ED = None
    if RD is not None:
        _, SD, ED, _, _, PD = build_steps(RD, thr, ccut, after)
    nstep = len(STEPS)
    k_phot = 2                                   # first step with a candidate

    # ---- cutflow --------------------------------------------------------------
    sig, sig0 = T["sig_incl"], T["sig_1g0x"]
    sig_fv, sig_en = sig & T["in_fv"], T["sig_enter"]
    sig_en_mu = T["sig_enter_mu"]; sig_en_nomu = sig_en & ~sig_en_mu
    DEN, DEN0 = w[sig].sum(), w[sig0].sum()
    DEN_FV, DEN_EN = w[sig_fv].sum(), w[sig_en].sum()
    DEN_ENM, DEN_ENN = w[sig_en_mu].sum(), w[sig_en_nomu].sum()
    mode = ("BDT applied after the multiplicity cut" if after
            else "BDT defines the photon-candidate set")
    lines = [f"cutflow (POT-weighted to {args.pot:.2e}; shower BDT >= {thr} "
             f"[{mode}], flash chi2 < {ccut:.0f}, mu KE > {args.mu_ke_min:.0f} "
             f"[{args.muon_finder}])",
             f"efficiency denominators (w): combined {DEN:.1f} | in-FV {DEN_FV:.1f} "
             f"| entering (out-of-FV) {DEN_EN:.1f} = no-mu {DEN_ENN:.1f} + CC-mu "
             f"{DEN_ENM:.1f} | strict 1g0X {DEN0:.1f}",
             f"{'step':<16}{'sig incl':>9}{'eff':>7}{'effFV':>7}{'effOut':>7}"
             f"{'effOutNoMu':>11}{'effOutMu':>9}"
             f"{'eff1g0X':>8}{'pur':>7}{'purFV':>7}"
             f"{'MC bkg':>9}{'EXT':>9}{'pred':>9}{'data':>7}{'d/p':>6}{'raw sig':>8}"]
    for k, nm in enumerate(STEPS):
        m = S[k]
        s, s0 = w[m & sig].sum(), w[m & sig0].sum()
        sf, se = w[m & sig_fv].sum(), w[m & sig_en].sum()
        sen, sem = w[m & sig_en_nomu].sum(), w[m & sig_en_mu].sum()
        b = w[m & ~sig].sum()
        e = wx[SX[k]].sum() if SX is not None else 0.0
        d = int(SD[k].sum()) if SD is not None else 0
        tot = max(s + b + e, 1e-9)
        lines.append(f"{nm:<16}{s:9.1f}{s/DEN:7.3f}{sf/DEN_FV:7.3f}{se/DEN_EN:7.3f}"
                     f"{sen/max(DEN_ENN,1e-9):11.3f}{sem/max(DEN_ENM,1e-9):9.3f}"
                     f"{s0/DEN0:8.3f}{s/tot:7.3f}{sf/tot:7.3f}{b:9.1f}{e:9.1f}"
                     f"{tot:9.1f}{d:7d}{d/tot:6.2f}{int((m & sig).sum()):8d}")
    m = S[-1]
    if args.novtx:
        lines.append("vertex-less (path B) candidate fraction at the final step: "
                     f"signal {w[m & sig & (path_cand==1)].sum()/max(w[m & sig].sum(),1e-9):.2f}"
                     f" | MC bkg {w[m & ~sig & (path_cand==1)].sum()/max(w[m & ~sig].sum(),1e-9):.2f}"
                     + (f" | EXT {wx[SX[-1] & (PX==1)].sum()/max(wx[SX[-1]].sum(),1e-9):.2f}"
                        if SX is not None else "")
                     + (f" | data {int((SD[-1] & (PD==1)).sum())}/{int(SD[-1].sum())}"
                        if SD is not None else ""))
    lines.append("final composition (weighted):")
    for c in range(N_MC_CAT):
        lines.append(f"    {CATS[c]:28s} {w[m & (cat==c)].sum():9.1f}")
    if SX is not None:
        lines.append(f"    {CATS[N_MC_CAT]:28s} {wx[SX[-1]].sum():9.1f}")
    if SD is not None:
        lines.append(f"    {'beam data':28s} {int(SD[-1].sum()):9d}")
    txt = "\n".join(lines)
    print("\n" + txt)
    with open(os.path.join(args.plots, "cutflow.txt"), "w") as fh:
        fh.write(txt + "\n")

    if args.out_dir:
        pub = lambda d: {k: v for k, v in d.items()
                         if not k.startswith("_") and k != "novtx"}
        np.savez(os.path.join(args.out_dir, "mc_table.npz"), w=w, steps=S,
                 step_names=np.array(STEPS), E_cand=E_cand, path_cand=path_cand,
                 score_cand=score_cand, n_g=n_g, **pub(T), **pub(R))
        if RX is not None:
            np.savez(os.path.join(args.out_dir, "ext_table.npz"), w=wx,
                     steps=SX, step_names=np.array(STEPS), E_cand=EX, **pub(RX))
        if RD is not None:
            np.savez(os.path.join(args.out_dir, "data_table.npz"), w=wd,
                     steps=SD, step_names=np.array(STEPS), E_cand=ED, **pub(RD))

    # ---- plots ------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def stack_hist(ax, vals_mc, m_mc, vals_ext, m_ext, vals_d, m_d, bins,
                   xlabel, title):
        lo, hi = bins[0], bins[-1] - 1e-6
        data = [np.clip(vals_mc[m_mc & (cat == c)], lo, hi) for c in range(N_MC_CAT)]
        ws = [w[m_mc & (cat == c)] for c in range(N_MC_CAT)]
        cols = list(CAT_COLORS[:N_MC_CAT])
        if vals_ext is not None:
            data.append(np.clip(vals_ext[m_ext], lo, hi))
            ws.append(wx[m_ext])
            cols.append(CAT_COLORS[N_MC_CAT])
        labs = [f"{CATS[c]} ({ws[c].sum():.1f})" for c in range(len(ws))]
        ax.hist(data, bins=bins, weights=ws, stacked=True, color=cols, label=labs)
        if vals_d is not None:
            hd, _ = np.histogram(np.clip(vals_d[m_d], lo, hi), bins=bins)
            ctr = 0.5 * (bins[:-1] + bins[1:])
            ax.errorbar(ctr[hd > 0], hd[hd > 0], yerr=np.sqrt(hd[hd > 0]),
                        fmt="ko", ms=3.5, label=f"beam data ({int(hd.sum())})")
        ax.set(xlabel=xlabel, ylabel=f"events / {args.pot:.1e} POT", title=title)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    ebins = np.linspace(0, 1000, 41)
    for k in range(k_phot, nstep):
        fig, ax = plt.subplots(figsize=(7.0, 4.6))
        stack_hist(ax, E_cand, S[k] & np.isfinite(E_cand),
                   EX, (SX[k] & np.isfinite(EX)) if SX is not None else None,
                   ED, (SD[k] & np.isfinite(ED)) if SD is not None else None,
                   ebins, "reco photon energy [MeV]",
                   f"candidate photon energy after {STEPS[k]}")
        fig.tight_layout()
        fig.savefig(f"{args.plots}/photon_energy_stacked_{k}_"
                    f"{STEPS[k].split()[0]}.png", dpi=110)
        plt.close(fig)

    # N-1 flash chi2 (all cuts but the chi2 cut)
    def lchi(RR):
        return np.log10(np.clip(np.nan_to_num(RR["chi2"], nan=1e-3), 1e-3, None))
    cbins = np.linspace(0, 7, 57)
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    stack_hist(ax, lchi(R), S[-2] & np.isfinite(R["chi2"]) & (R["chi2"] > 0),
               lchi(RX) if RX else None,
               (SX[-2] & np.isfinite(RX["chi2"]) & (RX["chi2"] > 0)) if RX else None,
               lchi(RD) if RD else None,
               (SD[-2] & np.isfinite(RD["chi2"]) & (RD["chi2"] > 0)) if RD else None,
               cbins, r"$\log_{10}$ flash $\chi^2$ (nu-slice vertex)",
               f"N-1: flash chi2 after {STEPS[-2]}, before the chi2 cut")
    ax.axvline(np.log10(ccut), color="k", ls="--", lw=1, label=f"cut {ccut:.0f}")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{args.plots}/flashchi2_nminus1.png", dpi=110)
    plt.close(fig)

    # efficiency vs true photon energy (E_vis and true E), per step
    tedges = np.array([20, 40, 60, 80, 100, 150, 200, 300, 400, 600, 1000])
    tctr = 0.5 * (tedges[:-1] + tedges[1:])
    styles = ["", "o-", "s-", "^-", "d-", "*-", "v-"]
    for xkey, xlab, stem in (("g_evis", "true photon E_vis [MeV] (A_GAMMA x Q)",
                              "eff_vs_true_evis"),
                             ("g_trueE", "true photon energy [MeV]",
                              "eff_vs_true_E")):
        fig, axs = plt.subplots(2, 3, figsize=(17, 9), sharey=True)
        for ax, den, dlab in ((axs[0, 0], sig_fv, "in-FV signal (1g+X, any X)"),
                              (axs[0, 1], sig_en_nomu, "entering photon, no mu > 100 MeV"),
                              (axs[0, 2], sig_en_mu, "entering photon, CC (mu > 100 MeV outside)"),
                              (axs[1, 0], sig, "combined in-FV + entering"),
                              (axs[1, 1], sig_en, "all entering (out-of-FV)"),
                              (axs[1, 2], sig0, "strict 1g0X signal (in FV)")):
            x = np.nan_to_num(T[xkey], nan=-1)
            for k in range(1, nstep):
                num = S[k] & den
                e, er = binned_ratio(w * num, w * den, x, tedges, n_raw=den)
                ax.errorbar(tctr, e, yerr=er, fmt=styles[k], ms=4,
                            label=f"{STEPS[k]} ({w[num].sum()/max(w[den].sum(),1e-9):.3f})")
            ax.set(xlabel=xlab, ylabel="efficiency", ylim=(0, 1.0), xscale="log",
                   title=f"efficiency vs true photon energy\n(denominator: {dlab})")
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3, which="both")
        fig.tight_layout()
        fig.savefig(f"{args.plots}/{stem}.png", dpi=110)
        plt.close(fig)

    # purity vs reco photon energy per step (signal / (MC + EXT))
    redges = tedges
    rctr = 0.5 * (redges[:-1] + redges[1:])
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, numsel, dlab in ((axs[0], sig, "inclusive 1g+X"),
                             (axs[1], sig0, "strict 1g0X")):
        for k in range(1, nstep):
            m = S[k] & np.isfinite(E_cand)
            xs, ws_num, ws_den = [E_cand[m]], [w[m] * numsel[m]], [w[m]]
            if SX is not None:
                mx = SX[k] & np.isfinite(EX)
                xs.append(EX[mx]); ws_num.append(np.zeros(mx.sum())); ws_den.append(wx[mx])
            p, pe = binned_ratio(np.concatenate(ws_num), np.concatenate(ws_den),
                                 np.concatenate(xs), redges)
            tot_num = sum(v.sum() for v in ws_num)
            tot_den = max(sum(v.sum() for v in ws_den), 1e-9)
            ax.errorbar(rctr, p, yerr=pe, fmt=styles[k], ms=4,
                        label=f"{STEPS[k]} ({tot_num/tot_den:.3f})")
        ax.set(xlabel="reco photon energy [MeV] (leading candidate)",
               ylabel="purity", ylim=(0, 1.0), xscale="log",
               title=f"purity vs reco photon energy\n(signal = {dlab}; "
                     f"bkg = other MC + EXT)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(f"{args.plots}/purity_vs_reco_E.png", dpi=110)
    plt.close(fig)

    # threshold scans: full selection re-evaluated per threshold
    def scan_bdt(thrs):
        eff, pur = [], []
        for t in thrs:
            _, Sm, *_ = build_steps(R, t, ccut, after)
            s = w[Sm[-1] & sig].sum(); b = w[Sm[-1] & ~sig].sum()
            e = 0.0
            if RX is not None:
                _, Sx, *_ = build_steps(RX, t, ccut, after)
                e = wx[Sx[-1]].sum()
            eff.append(s / DEN); pur.append(s / max(s + b + e, 1e-9))
        return np.array(eff), np.array(pur)

    tscan = np.linspace(0, 0.95, 20)
    eff, pur = scan_bdt(tscan)
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.plot(tscan, eff, "o-", ms=3, label="efficiency (incl. signal)")
    ax.plot(tscan, pur, "s-", ms=3, label="purity")
    ax.plot(tscan, eff * pur, "^-", ms=3, label="eff x purity")
    ax.axvline(thr, color="k", ls="--", lw=1, label="current")
    ax.set(xlabel="photon showerCosmicScore threshold", ylim=(0, 1),
           title=f"per-shower cosmic-BDT scan (full selection, chi2 < {ccut:.0f})")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{args.plots}/scan_shower_bdt.png", dpi=110)
    plt.close(fig)

    lthr = np.linspace(1.5, 6.0, 46)
    eff, pur = [], []
    base, basex = S[-2], (SX[-2] if SX is not None else None)
    for lt in lthr:
        pm = base & np.isfinite(R["chi2"]) & (R["chi2"] < 10 ** lt)
        s = w[pm & sig].sum(); b = w[pm & ~sig].sum()
        e = wx[basex & np.isfinite(RX["chi2"]) & (RX["chi2"] < 10 ** lt)].sum() if RX else 0.0
        eff.append(s / DEN); pur.append(s / max(s + b + e, 1e-9))
    eff, pur = np.array(eff), np.array(pur)
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.plot(lthr, eff, "o-", ms=3, label="efficiency (incl. signal)")
    ax.plot(lthr, pur, "s-", ms=3, label="purity")
    ax.plot(lthr, eff * pur, "^-", ms=3, label="eff x purity")
    ax.axvline(np.log10(ccut), color="k", ls="--", lw=1, label="current")
    ax.set(xlabel=r"$\log_{10}$ flash $\chi^2$ cut (keep below)", ylim=(0, 1),
           title=f"flash-chi2 scan ({STEPS[-2]} applied)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{args.plots}/scan_flashchi2.png", dpi=110)
    plt.close(fig)
    print(f">>> plots -> {args.plots}")


if __name__ == "__main__":
    main()
