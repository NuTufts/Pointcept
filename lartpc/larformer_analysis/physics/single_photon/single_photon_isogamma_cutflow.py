#!/usr/bin/env python3
"""Isotropic single-photon GUN sample through the single-photon cutflow and the
BDT-path tree (2026-10-10). Unit weights (xsecWeight = -1 in this sample), no
POT: everything is a FRACTION of a truth denominator.

Truth per event (gun conventions, versions doc 3d): the photon's true energy
(trueNuE), ORIGIN = trueVtx (in WC FV or not), CONVERSION point = the photon's
trueSimPart EDep (inf -> never converted in the TPC), deposited energy
a_gamma x PixelSumQ (lar 'visible' = >= 20 MeV). Reco: the shared selection
with the geometric 2nd-photon veto, chain BDT scores, 0p/Np, and whether the
selected candidate IS the gun photon (showerTrueTID == photon TID, purity > 0.5)
and whether the reco vertex sits on the conversion point (< 5 cm).

    python3 single_photon_isogamma_cutflow.py --mc-ntuple <gun ntuple> --novtx --unit-weight \\
        --veto2g-dcmax 100 --mb-chain-model plots_run1_mbchain/mb_bdt_chain_noflash_novtx_lar.joblib --plots plots_isogamma/cutflow
"""
import argparse
import os
import sys

import numpy as np
import awkward as ak

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import single_photon_selection as SPS                      # noqa: E402
import single_photon_diagnostics as SPD                    # noqa: E402
from mb_variable_study import features_from                # noqa: E402
from single_photon_bdt_paths import cells_for               # noqa: E402

SPD.BR_EXTRA = tuple(SPD.BR_EXTRA) + ("trueNuE", "showerTrueTID", "showerTruePurity", "trueVtxX", "trueVtxY", "trueVtxZ")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    SPD.add_args(ap); ap.set_defaults(signal_def="lar", unit_weight=True)
    for k, v in (("mb_mu_kemin", 40.0), ("mb_pi_kemin", 50.0), ("mb_p_kemax", 342.0), ("extra_emax", 10.0), ("cone_e2_min", 5.0), ("cone_e2_max", 600.0),
                 ("contain_dwall", 10.0), ("dfw_cut", SPS.DFW_CUT), ("veto2g_emin", 10.0), ("veto2g_dcmax", 100.0),
                 ("numu_cut", 0.25), ("enter_cut", 0.30), ("enter_cut_0p", 0.15)):
        ap.add_argument("--" + k.replace("_", "-"), type=float, default=v)
    ap.add_argument("--mb-chain-model", default=os.path.join(HERE, "plots_run1_mbchain", "mb_bdt_chain_noflash_novtx_lar.joblib"))
    ap.add_argument("--mb-chain-model-extra", default="")
    ap.add_argument("--unit-weight", action="store_true", default=True)
    args = ap.parse_args()
    SPS.configure(args); os.makedirs(args.plots, exist_ok=True)
    M, a, pot, R = SPD.per_sample(args.mc_ntuple, True, args, None, "GUN")
    n = len(M["E"]); w = np.ones(n)
    T = SPS.truth_tables(a)
    # ---- gun truth ---------------------------------------------------------------
    Eg = np.asarray(ak.to_numpy(a["trueNuE"]), np.float64) * 1000.0           # MeV
    pdg = a["trueSimPartPDG"]; proc = a["trueSimPartProcess"]
    gun = (pdg == 22) & (proc == 0)                                            # the gun photon row
    first = lambda x, fill=np.nan: np.asarray(ak.to_numpy(ak.fill_none(ak.firsts(x[gun]), fill)), np.float64)
    conv = np.stack([first(a["trueSimPartEDep" + c]) for c in "XYZ"], 1)
    converted = np.all(np.isfinite(conv), 1) & np.all(np.abs(conv) < 1e4, 1)
    conv_dwall = np.where(converted, np.minimum((conv - SPS.TPC_LO).min(1), (SPS.TPC_HI - conv).min(1)), np.nan)
    evis = first(SPS.A_GAMMA * np.maximum(a["trueSimPartPixelSumQ"], 0.0))
    visible = converted & (evis >= SPS.EVIS_MIN)
    gtid = first(a["trueSimPartTID"], -1).astype(np.int64)
    origin = np.stack([np.asarray(ak.to_numpy(a["trueVtx" + c]), np.float64) for c in "XYZ"], 1)
    origin_dwall = np.minimum((origin - SPS.TPC_LO).min(1), (SPS.TPC_HI - origin).min(1))
    in_fv = T["in_fv"]                                                          # gun ORIGIN in the WC FV
    conv_fv = converted & (conv_dwall >= 10.0)
    # candidate = the gun photon?  reco vertex on the conversion point?
    idx = R["_cand_idx"]; has = np.isfinite(M["E"])
    ctid = np.asarray(ak.to_numpy(ak.fill_none(ak.firsts(a["showerTrueTID"][idx]), -1)), np.int64); ctid[~has] = -1
    cpur = np.asarray(ak.to_numpy(ak.fill_none(ak.firsts(a["showerTruePurity"][idx]), 0.0)), np.float64)
    cand_is_photon = has & (ctid == gtid) & (gtid > 0) & (cpur > 0.5)
    nu_v = a["recoVtxStream"] == 0
    vx = np.stack([np.asarray(ak.to_numpy(ak.fill_none(ak.firsts(a["recoVtx" + c][nu_v]), np.nan)), np.float64) for c in "XYZ"], 1)
    vtx_on_photon = converted & np.all(np.isfinite(vx), 1) & (np.linalg.norm(vx - conv, axis=1) < 5.0)
    Mf = features_from(a, R, M["names"], M["steps"], M["E"], M["path"], args); del a
    C, OLD = cells_for(M, R, Mf, args)
    S, names = M["steps"], M["names"]
    DEN = [("all events", np.ones(n, bool)), ("photon converted in TPC", converted), ("converted, >= 20 MeV deposited (lar-visible)", visible),
           ("visible & conversion >= 10 cm from walls", visible & conv_fv), ("visible & ORIGIN in WC FV", visible & in_fv), ("visible & origin outside FV", visible & ~in_fv)]
    L = [f"ISOTROPIC 1-GAMMA GUN through the single-photon cutflow (unit weights; {n} events; truth '{SPS.SIGNAL_DEF}'; "
         f"veto E > {args.veto2g_emin:.0f} MeV AND dcmax < {args.veto2g_dcmax:.0f} cm; chain {os.path.basename(args.mb_chain_model)})",
         "denominators: " + " | ".join(f"{d} {int(m.sum())}" for d, m in DEN),
         f"photon energy [MeV] pct 5/50/95: {np.percentile(Eg, [5, 50, 95]).round(0)}; converted {converted.mean():.3f}; lar-visible {visible.mean():.3f}; "
         f"origin in WC FV {in_fv.mean():.3f}; lar category of visible events: " + ", ".join(f"{SPS.CATS[c]} {int((visible & (T['cat'] == c)).sum())}" for c in range(SPS.N_MC_CAT) if (visible & (T['cat'] == c)).any())]
    L.append("\nBASE CUTFLOW: fraction of each denominator passing (cumulative); last two columns = of the passing events, fraction whose candidate IS the gun photon / whose reco vertex is on the conversion point")
    L.append(f"{'step':36s}" + "".join(f"{d[:22]:>24}" for d, _ in DEN) + f"{'cand=photon':>13}{'vtx on conv':>13}")
    for k, nm in enumerate(names):
        m = S[k]
        L.append(f"{nm[:36]:36s}" + "".join(f"{(m & dm).sum() / max(dm.sum(), 1):24.3f}" for _, dm in DEN)
                 + f"{(m & cand_is_photon).sum() / max(m.sum(), 1):13.3f}{(m & vtx_on_photon).sum() / max(m.sum(), 1):13.3f}")
    L.append("\nBDT-PATH TREE: events in each cell | fraction of the denominators | cand=photon | vtx on conv | median true E of events in the cell")
    L.append(f"{'cell':52s}{'N':>7}" + "".join(f"{d[:22]:>24}" for d, _ in DEN) + f"{'cand=ph':>9}{'vtxconv':>9}{'medE':>7}")
    for lab, m in C + OLD:
        L.append(f"{lab[:52]:52s}{int(m.sum()):7d}" + "".join(f"{(m & dm).sum() / max(dm.sum(), 1):24.3f}" for _, dm in DEN)
                 + f"{(m & cand_is_photon).sum() / max(m.sum(), 1):9.3f}{(m & vtx_on_photon).sum() / max(m.sum(), 1):9.3f}{np.median(Eg[m]) if m.any() else np.nan:7.0f}")
    # where do the VISIBLE in-FV photons end up?  (exclusive leaves)
    leaves = [("vertex, veto FAIL", C[0][1]), ("vertex_pass, CC-numu-like", C[2][1]), ("non-CC-numu, ENTERING", C[4][1]), ("in-FV, 1g0p", C[6][1]), ("in-FV, 1gNp", C[7][1]),
              ("no-vertex, veto FAIL", C[8][1]), ("no-vertex, veto PASS", C[9][1])]
    L.append("\nLEAF OCCUPANCY (exclusive leaves + 'not selected'): fraction of each denominator")
    L.append(f"{'leaf':36s}" + "".join(f"{d[:22]:>24}" for d, _ in DEN))
    anyleaf = np.zeros(n, bool)
    for lab, m in leaves:
        anyleaf |= m
        L.append(f"{lab:36s}" + "".join(f"{(m & dm).sum() / max(dm.sum(), 1):24.3f}" for _, dm in DEN))
    L.append(f"{'not selected (fails base S1-S6)':36s}" + "".join(f"{(~anyleaf & dm).sum() / max(dm.sum(), 1):24.3f}" for _, dm in DEN))
    txt = "\n".join(L); print(txt)
    open(os.path.join(args.plots, "isogamma_cutflow.txt"), "w").write(txt + "\n")
    # ---- plots: efficiency vs true E and vs conversion dwall --------------------------
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ebins = np.array([20, 50, 100, 150, 200, 300, 400, 600, 800, 1000, 1500, 2000, 3000, 5000, 10000]); ectr = np.sqrt(ebins[:-1] * ebins[1:])
    dbins = np.array([0, 5, 10, 20, 30, 50, 75, 100, 130]); dctr = 0.5 * (dbins[:-1] + dbins[1:])
    targets = [("base final (S6)", S[-1]), ("vertex, veto PASS", C[1][1]), ("in-FV MB target", C[5][1]), ("in-FV 1g0p", C[6][1]), ("no-vertex, veto PASS", C[9][1])]
    for xname, x, bins, ctr, den, dlab in (("trueE", Eg, ebins, ectr, visible & in_fv, "lar-visible, origin in FV"), ("trueE_allconv", Eg, ebins, ectr, converted, "photon converted in TPC"),
                                            ("convDwall", conv_dwall, dbins, dctr, visible, "lar-visible (any origin)")):
        fig, ax = plt.subplots(figsize=(7.4, 4.8))
        for lab, m in targets:
            e, er = SPS.binned_ratio(w * (m & den), w * den, np.nan_to_num(x, nan=-1), bins, n_raw=den)
            ax.errorbar(ctr, e, yerr=er, fmt="o-", ms=4, label=lab)
        ax.set(xlabel="true photon energy [MeV]" if "trueE" in xname else "conversion point dwall [cm]", ylabel="fraction of denominator", ylim=(0, 1),
               title=f"gun photon selection fraction vs {'true E' if 'trueE' in xname else 'conversion dwall'} (den: {dlab})")
        if "trueE" in xname:
            ax.set_xscale("log")
        ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(args.plots, f"eff_vs_{xname}.png"), dpi=110); plt.close(fig)
    # candidate purity vs E in the MB cell; leaf occupancy bar
    fig, ax = plt.subplots(figsize=(8, 4.8))
    labs = [l for l, _ in leaves] + ["not selected"]; vals = [[(m & dm).sum() / max(dm.sum(), 1) for _, dm in DEN] for _, m in leaves] + [[(~anyleaf & dm).sum() / max(dm.sum(), 1) for _, dm in DEN]]
    vals = np.array(vals); xx = np.arange(len(labs)); wd = 0.13
    for j, (d, _) in enumerate(DEN):
        ax.bar(xx + (j - 2.5) * wd, vals[:, j], wd, label=d)
    ax.set_xticks(xx); ax.set_xticklabels(labs, rotation=25, ha="right", fontsize=8); ax.set(ylabel="fraction", title="where the gun photons end up (exclusive leaves)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3, axis="y"); fig.tight_layout(); fig.savefig(os.path.join(args.plots, "leaf_occupancy.png"), dpi=110); plt.close(fig)
    np.savez(os.path.join(args.plots, "isogamma_cells.npz"), Eg=Eg, converted=converted, visible=visible, in_fv=in_fv, conv_dwall=conv_dwall, origin_dwall=origin_dwall,
             cand_is_photon=cand_is_photon, vtx_on_photon=vtx_on_photon, steps=S, **{f"cell{k}": m for k, (_, m) in enumerate(C)}, run=M["run"], subrun=M["subrun"], event=M["event"], E=M["E"])
    print(f">>> -> {args.plots}")


if __name__ == "__main__":
    main()
