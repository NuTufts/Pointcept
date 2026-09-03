"""SBND-SPINE-style CC 1pi0 selection on the LArFormer ntuples (a SECOND,
self-contained analysis -- does NOT touch the scripts behind
plots_ext_cut1e4_satfix).

Signal (truth), following the SBND CC 1pi0 definition (sbn-docdb 47857):
  - true vertex in a TIGHT fiducial volume: >=20 cm from the X,Y TPC walls,
    >=10 cm from the upstream (low-z) wall, >=50 cm from the downstream (high-z)
    wall of the MicroBooNE active TPC [0,256.35] x [-116.5,116.5] x [0,1036.8];
  - charged current (>=1 primary muon with KE > 143.425 MeV, = 50 cm range);
  - exactly 1 primary pi0 whose two photons are DETECTABLE (true visible energy
    A_GAMMA*pixelSumQ > 20 MeV each -- the repo convention, per the primary
    README, NOT the SBND literal true-KE>20);
  - 0 primary charged pions with KE > 25 MeV;
  - inclusive to all other particles (protons, neutrons, ...).

Reco selection (matched to the truth definition):
  - nu-stream vertex (primaryVtxStream==0), foundVertex, reco vtx in the same FV;
  - >=1 reco primary muon (trackLArFormerPID==13, primary, trackRecoE>143.425);
  - >=2 CONFIDENTLY ATTACHED reco photons (showerLArFormerPID==22,
    showerRecoE>20, showerAttConfident) -- two most energetic used;
  - reco m_gg < 400 MeV;
  - 0 reco primary charged pions (PID 211, primary, trackRecoE>25).

Observables: two-photon invariant mass m_gg and reco pi0 momentum
p_pi0 = |p_g1 + p_g2| (vertex->start photon directions). Two sets each: BEFORE
the flash cut and AFTER it (flash_chi2 < --flashchi2-cut, default 1e4, the CC
working point). MC is stacked as signal + background categories with the EXT
cosmic on top; beam data overlaid as points.

flash_chi2 and the POT weight w are read row-aligned from the existing
*_satfix_table.npz (row i == ntuple entry i), so no cascade rescan is needed.

    PYTHONPATH=./ python3 sbnd_cc1pi0.py \
        --mc-ntuple .. --mc-table mc_full_satfix_table.npz \
        --data-ntuple .. --data-table data_satfix_table.npz \
        --ext-ntuple .. --ext-table ext_satfix_table.npz \
        --ext-scale 0.17682554549 --plots plots_sbnd_cc1pi0
"""
import argparse
import os

import numpy as np
import uproot
import awkward as ak
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- SBND CC 1pi0 constants ------------------------------------------------
TPC_LO = np.array([0.0, -116.5, 0.0])
TPC_HI = np.array([256.35, 116.5, 1036.8])
FV_LO = np.array([TPC_LO[0] + 20.0, TPC_LO[1] + 20.0, TPC_LO[2] + 10.0])
FV_HI = np.array([TPC_HI[0] - 20.0, TPC_HI[1] - 20.0, TPC_HI[2] - 50.0])
MU_KE_MIN = 143.425        # primary muon KE [MeV] (50 cm range)
CPI_KE_MIN = 25.0          # charged-pion veto KE [MeV]
GAMMA_RECO_MIN = 20.0      # reco photon energy [MeV]
MGG_MAX = 400.0            # reco diphoton mass cut [MeV]
A_GAMMA = 0.0253017        # MeV/ADC (visible-energy photon detectability)
# deployed calib the ntuple's showerRecoE was produced with (for --recal-*):
NTUP_E_A = 0.019743114709854126
NTUP_G_A = 0.020100999623537064
NTUP_G_B = -15.489999771118164
RECAL = {"gamma_a": None, "e_a": None}     # set from --recal-* in main
EVIS_MIN = 20.0
M_MU = 0.105658            # GeV
M_CPI = 0.139570           # GeV
PI0_MASS = 134.977

CATS = ["signal CC 1pi0", "bkg: out-of-FV", "bkg: NC",
        "bkg: CC other-pi0", "bkg: CC w/ charged-pi", "bkg: CC soft-mu"]
CAT_COLORS = ["#d62728", "#bdbdbd", "#1f77b4", "#ff7f0e", "#9467bd", "#8c564b"]
EXT_COLOR = "#e5e5e5"

_BR = ["run", "subrun", "event", "foundVertex", "primaryVtxStream",
       "vtxX", "vtxY", "vtxZ",
       "showerLArFormerPID", "showerRecoE", "showerAttConfident",
       "showerStartPosX", "showerStartPosY", "showerStartPosZ",
       "trackLArFormerPID", "trackIsSecondary", "trackRecoE"]
_BR_MC = ["trueVtxX", "trueVtxY", "trueVtxZ", "trueNuCCNC",
          "truePrimPartPDG", "truePrimPartE",
          "trueSimPartPDG", "trueSimPartTID", "trueSimPartMID",
          "trueSimPartProcess", "trueSimPartPixelSumQ"]


def _in_fv(x, y, z):
    return ((x >= FV_LO[0]) & (x <= FV_HI[0]) & (y >= FV_LO[1])
            & (y <= FV_HI[1]) & (z >= FV_LO[2]) & (z <= FV_HI[2]))


def _pi0_detectable(a, i):
    """(n_primary_pi0, pair_ok, evis_total): the repo's detectable-pi0-photon-
    pair test (Process==1 photons whose mother id is absent from the TID table,
    one MID group of 2 with both visible energies > 20 MeV). evis_total is the
    sum of that pair's visible energies [MeV] (NaN if no detectable pair)."""
    npi0 = int(np.sum(np.asarray(a["truePrimPartPDG"][i]) == 111))
    pdg = np.asarray(a["trueSimPartPDG"][i])
    proc = np.asarray(a["trueSimPartProcess"][i])
    mid = np.asarray(a["trueSimPartMID"][i])
    tid = set(np.asarray(a["trueSimPartTID"][i]).tolist())
    q = np.asarray(a["trueSimPartPixelSumQ"][i])
    ph = (np.abs(pdg) == 22) & (proc == 1)
    orphan = ph & np.asarray([int(m) not in tid for m in mid], bool)
    mids, cnt = (np.unique(mid[orphan], return_counts=True)
                 if orphan.any() else (np.array([]), np.array([])))
    for m, c in zip(mids, cnt):
        if c != 2:
            continue
        evis = A_GAMMA * np.clip(q[orphan & (mid == m)], 0, None)
        if np.all(evis > EVIS_MIN):
            return npi0, True, float(evis.sum())
    return npi0, False, np.nan


def sbnd_truth_cat(a, i):
    """(cat, evis_2gamma): SBND CC-1pi0 truth category (0=signal, 1..5
    backgrounds by first-failed requirement) plus the signal photon pair's total
    true visible energy [MeV] (NaN unless a detectable pair exists)."""
    if not _in_fv(a["trueVtxX"][i], a["trueVtxY"][i], a["trueVtxZ"][i]):
        return 1, np.nan
    if a["trueNuCCNC"][i] == 1:
        return 2, np.nan                           # NC
    npi0, pair_ok, evis = _pi0_detectable(a, i)
    if not (npi0 == 1 and pair_ok):
        return 3, np.nan                           # CC, wrong/undetectable pi0
    pdg = np.abs(np.asarray(a["truePrimPartPDG"][i]))
    E = np.asarray(a["truePrimPartE"][i])          # GeV, total energy
    ke_cpi = (E - M_CPI) * 1000.0
    if np.any((pdg == 211) & (ke_cpi > CPI_KE_MIN)):
        return 4, evis                             # CC 1pi0 but charged pi
    ke_mu = (E - M_MU) * 1000.0
    if not np.any((pdg == 13) & (ke_mu > MU_KE_MIN)):
        return 5, evis                             # CC 1pi0 0cpi but soft muon
    return 0, evis                                  # signal


def load(ntuple, table, is_data):
    """Per-event SBND reco selection + observables + (MC) truth category, with w
    and flash_chi2 pulled row-aligned from the pre-built table."""
    t = uproot.open(ntuple)["EventTree"]
    br = list(_BR) + ([] if is_data else _BR_MC)
    a = t.arrays([b for b in br if b in set(t.keys())])
    if RECAL["gamma_a"] is not None or RECAL["e_a"] is not None:
        E, spid = a["showerRecoE"], a["showerLArFormerPID"]
        if RECAL["gamma_a"] is not None:
            E = ak.where(spid == 22,
                         (E - NTUP_G_B) / NTUP_G_A * RECAL["gamma_a"]
                         + RECAL.get("gamma_b", 0.0), E)
        if RECAL["e_a"] is not None:
            E = ak.where(spid == 11, E / NTUP_E_A * RECAL["e_a"] + RECAL.get("e_b", 0.0), E)
        a["showerRecoE"] = E
    n = len(a["run"])
    tab = np.load(table)
    assert len(tab["w"]) == n, "table/ntuple row mismatch"
    w = np.ones(n) if is_data else np.asarray(tab["w"], np.float64)
    fchi2 = np.asarray(tab["flash_chi2"], np.float64)

    # ---- reco selection ----------------------------------------------------
    vtx_ok = ((np.asarray(a["foundVertex"]) == 1)
              & (np.asarray(a["primaryVtxStream"]) == 0)
              & _in_fv(np.asarray(a["vtxX"]), np.asarray(a["vtxY"]),
                       np.asarray(a["vtxZ"])))
    conf = a["showerAttConfident"] != 0
    is_g = (a["showerLArFormerPID"] == 22) & (a["showerRecoE"] > GAMMA_RECO_MIN) & conf
    if RECAL.get("shower_bdt_min") is not None:
        sb = t.arrays(["showerCosmicScore"])["showerCosmicScore"]
        is_g = is_g & (sb >= RECAL["shower_bdt_min"])
    n_g = ak.to_numpy(ak.sum(is_g, axis=1))
    is_mu = ((a["trackLArFormerPID"] == 13) & (a["trackIsSecondary"] == 0)
             & (a["trackRecoE"] > MU_KE_MIN))
    has_mu = ak.to_numpy(ak.any(is_mu, axis=1))
    is_cpi = ((a["trackLArFormerPID"] == 211) & (a["trackIsSecondary"] == 0)
              & (a["trackRecoE"] > CPI_KE_MIN))
    n_cpi = ak.to_numpy(ak.sum(is_cpi, axis=1))

    m_gg = np.full(n, np.nan)
    p_pi0 = np.full(n, np.nan)
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
            m_gg[i] = float(np.sqrt(max(2.0 * E1 * E2 * (1.0 - d[0] @ d[1]), 0.0)))
            p_pi0[i] = float(np.linalg.norm(E1 * d[0] + E2 * d[1]))

    sel = (vtx_ok & (n_g >= 2) & has_mu & (n_cpi == 0)
           & np.isfinite(m_gg) & (m_gg < MGG_MAX))

    if is_data:
        cat = np.zeros(n, np.int64)
        evis2 = np.full(n, np.nan)
        n_sig_true = 0
    else:
        cat = np.zeros(n, np.int64)
        evis2 = np.full(n, np.nan)
        for i in range(n):
            cat[i], evis2[i] = sbnd_truth_cat(a, i)
        n_sig_true = float(w[cat == 0].sum())

    return dict(w=w, fchi2=fchi2, sel=sel, cat=cat, m_gg=m_gg, p_pi0=p_pi0,
                evis2=evis2, n=n, n_sig_true=n_sig_true)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc-ntuple", required=True)
    ap.add_argument("--mc-table", required=True)
    ap.add_argument("--data-ntuple", required=True)
    ap.add_argument("--data-table", required=True)
    ap.add_argument("--ext-ntuple", required=True)
    ap.add_argument("--ext-table", required=True)
    ap.add_argument("--ext-scale", type=float, default=0.17682554549)
    ap.add_argument("--flashchi2-cut", type=float, default=1e4)
    ap.add_argument("--pot", type=float, default=4.4e19)
    ap.add_argument("--recal-gamma-a", type=float, default=None,
                    help="analysis-level shower recal: E=(E-b)/a_old*a_new "
                         "for PID==22 (clustering unchanged -> exact)")
    ap.add_argument("--recal-e-a", type=float, default=None)
    ap.add_argument("--recal-gamma-b", type=float, default=0.0)
    ap.add_argument("--recal-e-b", type=float, default=0.0)
    ap.add_argument("--a-gamma", type=float, default=None,
                    help="override truth-side A_GAMMA (signal definition)")
    ap.add_argument("--shower-bdt-min", type=float, default=None,
                    help="opt-in per-shower cosmic-BDT cut (showerCosmicScore)")
    ap.add_argument("--plots", default="plots_sbnd_cc1pi0")
    args = ap.parse_args()
    if args.a_gamma is not None:
        globals()["A_GAMMA"] = args.a_gamma
        print(f">>> A_GAMMA (signal-def) -> {args.a_gamma:.6f}")
    RECAL["gamma_a"], RECAL["e_a"] = args.recal_gamma_a, args.recal_e_a
    RECAL["shower_bdt_min"] = args.shower_bdt_min
    if args.shower_bdt_min is not None:
        print(f">>> per-shower cosmic-BDT cut: score >= {args.shower_bdt_min}")
    RECAL["gamma_b"], RECAL["e_b"] = args.recal_gamma_b, args.recal_e_b
    if args.recal_gamma_a or args.recal_e_a:
        print(f">>> showerRecoE recal: gamma_a={args.recal_gamma_a} "
              f"e_a={args.recal_e_a}")
    os.makedirs(args.plots, exist_ok=True)

    print(">>> loading MC ...");   mc = load(args.mc_ntuple, args.mc_table, False)
    print(">>> loading data ...");  da = load(args.data_ntuple, args.data_table, True)
    print(">>> loading EXT ...");   ex = load(args.ext_ntuple, args.ext_table, True)

    def flashmask(d, after):
        m = d["sel"].copy()
        if after:
            m = m & np.isfinite(d["fchi2"]) & (d["fchi2"] < args.flashchi2_cut)
        return m

    def panel(obs, bins, xlabel, fbase, clip_hi, after):
        mm, dm, em = (flashmask(mc, after) & np.isfinite(mc[obs]),
                      flashmask(da, after) & np.isfinite(da[obs]),
                      flashmask(ex, after) & np.isfinite(ex[obs]))
        ctr = 0.5 * (bins[:-1] + bins[1:])
        stack = [np.clip(mc[obs][mm & (mc["cat"] == c)], bins[0], clip_hi)
                 for c in range(6)]
        ws = [mc["w"][mm & (mc["cat"] == c)] for c in range(6)]
        stack.append(np.clip(ex[obs][em], bins[0], clip_hi))
        ws.append(np.full(int(em.sum()), args.ext_scale))
        colors = CAT_COLORS + [EXT_COLOR]
        labels = [f"{CATS[c]} ({ws[c].sum():.0f})" for c in range(6)]
        labels.append(f"EXT cosmic ({ws[6].sum():.0f})")
        fig, ax = plt.subplots(figsize=(7.0, 4.8))
        ax.hist(stack, bins=bins, weights=ws, stacked=True, color=colors,
                label=labels)
        pred = sum(w.sum() for w in ws)
        dh, _ = np.histogram(np.clip(da[obs][dm], bins[0], clip_hi), bins=bins)
        ax.errorbar(ctr, dh, yerr=np.sqrt(np.clip(dh, 1, None)), fmt="ko",
                    ms=3.5, lw=1, capsize=0, label=f"beam data ({int(dh.sum())})")
        if obs == "m_gg":
            ax.axvline(PI0_MASS, color="0.4", ls=":", lw=1.1)
        tag = "after flash cut" if after else "before flash cut"
        # purity (signal / total pred) over this selection
        sig = mc["w"][mm & (mc["cat"] == 0)].sum()
        pur = sig / max(pred, 1e-9)
        ax.set(xlabel=xlabel, ylabel=f"events / {args.pot:.1e} POT",
               title=f"SBND-style CC 1$\\pi^0$: {fbase} ({tag})\n"
                     f"data {int(dh.sum())} vs pred {pred:.0f} (MC+EXT) | "
                     f"purity {pur:.2f}")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        suff = "afterflash" if after else "beforeflash"
        fig.savefig(f"{args.plots}/{fbase}_{suff}.png", dpi=110)
        plt.close(fig)
        return pred, sig, int(dh.sum())

    print("\n== SBND CC 1pi0 selection ==")
    for after in (False, True):
        tag = "AFTER flash cut " if after else "BEFORE flash cut"
        pm, sm, dm = panel("m_gg", np.linspace(0, 500, 51),
                           r"$m_{\gamma\gamma}$ [MeV]", "mgg_sbnd_cc1pi0",
                           499, after)
        panel("p_pi0", np.linspace(0, 1500, 51),
              r"reco $p_{\pi^0}$ [MeV/c]", "ppi0_sbnd_cc1pi0", 1499, after)
        eff = sm / max(mc["n_sig_true"], 1e-9)
        print(f"  [{tag}] MC pred {pm:.0f} | signal {sm:.0f} | data {dm} "
              f"| purity {sm/max(pm,1e-9):.3f} | efficiency {eff:.3f} "
              f"(true signal {mc['n_sig_true']:.0f})")

    # ---- log10(flash chi2) of the SELECTED sample, BEFORE the flash cut ------
    lbins = np.linspace(0, 8, 33)
    ctr = 0.5 * (lbins[:-1] + lbins[1:])

    def lchi(d, m):
        return np.log10(np.clip(d["fchi2"][m], 1, None))
    mm = mc["sel"] & np.isfinite(mc["fchi2"])
    dm = da["sel"] & np.isfinite(da["fchi2"])
    em = ex["sel"] & np.isfinite(ex["fchi2"])
    stack = [np.clip(lchi(mc, mm & (mc["cat"] == c)), 0, 7.999) for c in range(6)]
    ws = [mc["w"][mm & (mc["cat"] == c)] for c in range(6)]
    stack.append(np.clip(lchi(ex, em), 0, 7.999))
    ws.append(np.full(int(em.sum()), args.ext_scale))
    colors = CAT_COLORS + [EXT_COLOR]
    labels = [f"{CATS[c]} ({ws[c].sum():.0f})" for c in range(6)]
    labels.append(f"EXT cosmic ({ws[6].sum():.0f})")
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.hist(stack, bins=lbins, weights=ws, stacked=True, color=colors,
            label=labels)
    dh, _ = np.histogram(np.clip(lchi(da, dm), 0, 7.999), bins=lbins)
    ax.errorbar(ctr, dh, yerr=np.sqrt(np.clip(dh, 1, None)), fmt="ko", ms=3.5,
                lw=1, capsize=0, label=f"beam data ({int(dh.sum())})")
    ax.axvline(np.log10(args.flashchi2_cut), color="crimson", ls="--", lw=1.4,
               label=f"cut chi2={args.flashchi2_cut:.0f}")
    ax.set(xlabel=r"$\log_{10}$ flash $\chi^2$ (primary nu vtx)",
           ylabel=f"events / {args.pot:.1e} POT",
           title="SBND-style CC 1$\\pi^0$: flash $\\chi^2$ (before cut, tight FV)"
                 f"\ndata {int(dh.sum())} vs pred {sum(w.sum() for w in ws):.0f}")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{args.plots}/flashchi2_sbnd_cc1pi0_beforeflash.png", dpi=110)
    plt.close(fig)

    # ---- signal efficiency vs total true visible energy of the pi0 photons ---
    ebins = np.linspace(0, 1000, 21)
    ectr = 0.5 * (ebins[:-1] + ebins[1:])
    den = (mc["cat"] == 0) & np.isfinite(mc["evis2"])
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for after, sty, lab in ((False, "s--", "before flash cut"),
                            (True, "o-", "after flash cut")):
        num = den & flashmask(mc, after)
        eff, err = [], []
        for lo, hi in zip(ebins[:-1], ebins[1:]):
            b = den & (mc["evis2"] >= lo) & (mc["evis2"] < hi)
            N = int(b.sum())
            k = int((num & b).sum())
            p = k / N if N else np.nan
            eff.append(p)
            err.append(np.sqrt(p * (1 - p) / N) if N and np.isfinite(p) else 0)
        ax.errorbar(ectr, eff, yerr=err, fmt=sty, ms=4,
                    alpha=1.0 if after else 0.5, label=lab)
    ax.set(xlabel=r"total true visible energy of the $\pi^0$ photons [MeV]",
           ylabel="signal selection efficiency", ylim=(0, 1.02),
           title="SBND-style CC 1$\\pi^0$: efficiency vs true $\\gamma\\gamma$ "
                 "visible energy\n(unweighted; true signal events passing reco)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{args.plots}/efficiency_vs_evis_sbnd_cc1pi0.png", dpi=110)
    plt.close(fig)

    print(f">>> plots -> {args.plots}")


if __name__ == "__main__":
    main()
