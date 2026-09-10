"""SBND-comparable step-by-step cutflow for the CC1pi0 selection.

Cut order (user-specified, 2026-09-03, to line up with SBND-SPINE tables):
  1 all events | 2 reco vertex in tight FV | 3 exactly ONE primary muon
  above 143.4 MeV | 4 no primary charged pion above 25 MeV | 5 exactly 2
  photons (recal'd E>20, att-confident, showerCosmicScore>=0.192) |
  6 m_gg in [30,300] MeV | 7 flash chi2 < 1e4.

Signal = SBND truth cat 0 (sbnd_cc1pi0.sbnd_truth_cat), POT-weighted.
eff = signal passing / all true signal; purity = sig/(sig+MC other+EXT).
EXT = shower-BDT analysis half (rows>=100k) x --ext-scale; no event BDT
anywhere in this selection, so no odd/even split is needed.

    PYTHONPATH=./ python3 sbnd_cutflow.py --muon-finder union ...
"""
import argparse
import os
import sys

import numpy as np
import uproot
import awkward as ak

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sbnd_cc1pi0 as S  # noqa: E402

NTUP_G_A, NTUP_G_B = 0.020100999623537064, -15.489999771118164


def build(ntuple, table, is_data, args):
    t = uproot.open(ntuple)["EventTree"]
    br = list(S._BR) + ["showerCosmicScore"] + ([] if is_data else S._BR_MC)
    a = t.arrays([b for b in br if b in set(t.keys())])
    E = ak.where(a["showerLArFormerPID"] == 22,
                 (a["showerRecoE"] - NTUP_G_B) / NTUP_G_A
                 * args.recal_gamma_a + args.recal_gamma_b, a["showerRecoE"])
    a["showerRecoE"] = E
    n = len(a["run"])
    tab = np.load(table)
    w = np.ones(n) if is_data else np.asarray(tab["w"], np.float64)
    fchi2 = np.asarray(tab["flash_chi2"], np.float64)

    vtx_ok = ((np.asarray(a["foundVertex"]) == 1)
              & (np.asarray(a["primaryVtxStream"]) == 0)
              & S._in_fv(np.asarray(a["vtxX"]), np.asarray(a["vtxY"]),
                         np.asarray(a["vtxZ"])))
    seg_mu = ((a["trackLArFormerPID"] == 13) & (a["trackIsSecondary"] == 0)
              & (a["trackRecoE"] > S.MU_KE_MIN))
    lp_mu = ((a["trackClassified"] == 1)
             & (a["trackMuScore"] > a["trackElScore"])
             & (a["trackMuScore"] > a["trackPhScore"])
             & (a["trackMuScore"] > a["trackPiScore"])
             & (a["trackMuScore"] > a["trackPrScore"])
             & (a["trackIsSecondary"] == 0)
             & (a["trackRecoE"] > S.MU_KE_MIN))
    is_mu = {"segmenter": seg_mu, "larpid": lp_mu,
             "union": seg_mu | lp_mu}[args.muon_finder]
    n_mu = ak.to_numpy(ak.sum(is_mu, axis=1))
    is_cpi = ((a["trackLArFormerPID"] == 211) & (a["trackIsSecondary"] == 0)
              & (a["trackRecoE"] > S.CPI_KE_MIN))
    n_cpi = ak.to_numpy(ak.sum(is_cpi, axis=1))
    conf = a["showerAttConfident"] != 0
    is_g = ((a["showerLArFormerPID"] == 22)
            & (a["showerRecoE"] > S.GAMMA_RECO_MIN) & conf)
    if args.shower_bdt_min is not None:
        is_g = is_g & (a["showerCosmicScore"] >= args.shower_bdt_min)
    n_g = ak.to_numpy(ak.sum(is_g, axis=1))

    m_gg = np.full(n, np.nan)
    for i in np.nonzero(vtx_ok & (n_g == 2))[0]:
        gi = np.nonzero(ak.to_numpy(is_g[i]))[0]
        Ei = ak.to_numpy(a["showerRecoE"][i])[gi]
        v = np.array([a["vtxX"][i], a["vtxY"][i], a["vtxZ"][i]], np.float64)
        sp = np.stack([ak.to_numpy(a[f"showerStartPos{c}"][i])[gi]
                       for c in "XYZ"], 1).astype(np.float64)
        d = sp - v
        nn = np.linalg.norm(d, axis=1)
        if np.all(nn > 1e-3):
            d = d / nn[:, None]
            m_gg[i] = float(np.sqrt(max(
                2.0 * Ei[0] * Ei[1] * (1.0 - d[0] @ d[1]), 0.0)))

    cat = np.zeros(n, np.int64)
    if not is_data:
        for i in range(n):
            cat[i], _ = S.sbnd_truth_cat(a, i)

    steps = [
        ("all events",                np.ones(n, bool)),
        ("reco vtx in tight FV",      vtx_ok),
        ("exactly 1 primary mu >143", vtx_ok & (n_mu == 1)),
        ("no primary cpi >25",        vtx_ok & (n_mu == 1) & (n_cpi == 0)),
        ("exactly 2 photons",         vtx_ok & (n_mu == 1) & (n_cpi == 0)
                                      & (n_g == 2)),
        ("m_gg in [30,300]",          vtx_ok & (n_mu == 1) & (n_cpi == 0)
                                      & (n_g == 2) & (m_gg >= 30.0)
                                      & (m_gg < 300.0)),
        ("flash chi2 < 1e4",          vtx_ok & (n_mu == 1) & (n_cpi == 0)
                                      & (n_g == 2) & (m_gg >= 30.0)
                                      & (m_gg < 300.0)
                                      & np.isfinite(fchi2) & (fchi2 < 1e4)),
    ]
    return steps, cat, w, m_gg


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mc-ntuple", required=True)
    ap.add_argument("--mc-table", required=True)
    ap.add_argument("--data-ntuple", required=True)
    ap.add_argument("--data-table", required=True)
    ap.add_argument("--ext-ntuple", required=True)
    ap.add_argument("--ext-table", required=True)
    ap.add_argument("--ext-scale", type=float, default=1.1818)
    ap.add_argument("--ext-row-min", type=int, default=100000,
                    help="EXT rows below this (shower-BDT training half) are "
                         "excluded entirely")
    ap.add_argument("--recal-gamma-a", type=float, default=0.01553)
    ap.add_argument("--recal-gamma-b", type=float, default=-12.80)
    ap.add_argument("--shower-bdt-min", type=float, default=0.192)
    ap.add_argument("--muon-finder", default="union",
                    choices=["segmenter", "larpid", "union"])
    args = ap.parse_args()

    print(f">>> muon finder: {args.muon_finder} | shower score >= "
          f"{args.shower_bdt_min} | gamma recal a={args.recal_gamma_a}")
    mc_steps, cat, wmc, _ = build(args.mc_ntuple, args.mc_table, False, args)
    da_steps, _, _, _ = build(args.data_ntuple, args.data_table, True, args)
    ex_steps, _, _, _ = build(args.ext_ntuple, args.ext_table, True, args)
    sig = cat == 0
    DEN = wmc[sig].sum()
    n_ext = len(ex_steps[0][1])
    we = np.zeros(n_ext)
    we[args.ext_row_min:] = args.ext_scale
    print(f">>> true SBND signal (POT-wtd): {DEN:.1f} "
          f"(raw {int(sig.sum())})\n")
    print(f"{'cut':<28}{'signal':>8}{'eff':>7}{'purity':>8}"
          f"{'MC other':>10}{'EXT':>8}{'data':>7}{'d/p':>6}")
    for (nm, mm), (_, dm), (_, em) in zip(mc_steps, da_steps, ex_steps):
        s = wmc[mm & sig].sum()
        b = wmc[mm & ~sig].sum()
        e = we[em].sum()
        d = int(dm.sum())
        print(f"{nm:<28}{s:8.1f}{s/DEN:7.3f}{s/max(s+b+e,1e-9):8.3f}"
              f"{b:10.1f}{e:8.1f}{d:7d}{d/max(s+b+e,1e-9):6.2f}")


if __name__ == "__main__":
    main()
