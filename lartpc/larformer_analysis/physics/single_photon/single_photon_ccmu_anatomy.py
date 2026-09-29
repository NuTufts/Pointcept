"""CC-numu background anatomy for the single-photon selection (MC only).

Population: events passing the full selection whose truth is "in-FV, visible
muon" (category 7) -- and, reported alongside, the entering-CC category (5),
where a muon from an out-of-FV vertex deposited in the TPC. For each event the
TRUE muon = the highest-KE muon with KE >= 100 MeV that deposits charge; its
prong is any track/shower prong whose truth TID is that muon (all nu-slice
segmenter particles are exported, attached and vertex-less).

MUON failure mode (why the muon veto did not fire):
  M0 not reconstructed   no prong carries the muon's TID
  M1 reco'd as SHOWER    the muon's prong is a shower (e/gamma label); flagged
                         separately when that shower IS the accepted photon
  M2 track, wrong PID    track prong, but neither segmenter mu nor LArPID
                         mu-argmax (labelled p / pi / other)
  M3 track, mu PID, soft, EXITING   mu PID but trackRecoE <= 100 MeV and the
                         true muon leaves the TPC (range energy is only the
                         contained part)
  M4 track, mu PID, soft, BROKEN    mu PID, E <= 100, contained muon but the
                         prong holds < --broken-comp of its charge (split /
                         truncated track)
  M5 track, mu PID, soft, other     mu PID, E <= 100, contained and complete
                         (genuinely soft after calibration; KE close to 100)
  M6 mu PID, E > 100 but not vetoed  (prong in the flashmatch stream, or a
                         veto-definition gap) -- should be ~0
PHOTON reality (what the accepted candidate is), from its truth match:
  true photon, visible (E_dep >= 20) | true photon, sub-threshold | the muon
  itself | electron | proton / pion / other | cosmic-unlabeled (purity < 0.5
  or unlabeled >= 0.5)
plus the event's true detectable-photon count (0 / 1 / >=2).

Outputs (--plots): counts table (raw + weighted) muon-mode x photon-reality,
per-mode muon true KE / trackRecoE / completeness / containment plots.

    python3 single_photon_ccmu_anatomy.py --mc-ntuple <run-1 halves> --novtx \
        --shower-bdt-min 0.164 --plots plots_run1_novtx_dep/ccmu_anatomy
"""
import argparse
import os
import sys

import numpy as np
import awkward as ak

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import single_photon_selection as SPS                      # noqa: E402
from single_photon_diagnostics import add_args             # noqa: E402

EXTRA = ["showerTrueTID", "showerTruePID", "showerTruePurity", "showerTrueUnlabeledPurity",
         "trackTrueTID", "trackTruePID", "trackTrueComp", "trackTruePurity",
         "trackLArFormerPID", "trackRecoE", "trackStream", "trackVtxIdx", "trackClassified",
         "trackMuScore", "trackElScore", "trackPhScore", "trackPiScore", "trackPrScore",
         "trueSimPartEndX", "trueSimPartEndY", "trueSimPartEndZ", "trueSimPartContained",
         "trackEndPosX", "trackEndPosY", "trackEndPosZ"]
MODES = ["M0 not reconstructed", "M1 reco'd as shower", "M2 track, wrong PID",
         "M3 mu PID, soft, EXITING", "M4 mu PID, soft, BROKEN", "M5 mu PID, soft, other",
         "M6 mu PID, E>100, not vetoed"]
MCOL = ["#7f7f7f", "#d62728", "#1f77b4", "#ff7f0e", "#9467bd", "#2ca02c", "#bcbd22"]
PHOT = ["true photon, visible", "true photon, sub-threshold", "the muon itself",
        "electron", "p / pi / other", "cosmic / unlabeled", "no candidate"]
TPC_LO, TPC_HI = SPS.TPC_LO, SPS.TPC_HI


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_args(ap)
    ap.add_argument("--broken-comp", type=float, default=0.6)
    ap.add_argument("--cats", default="7,5", help="truth categories to analyse")
    args = ap.parse_args()
    SPS.set_truth_scale(args.truth_scale)
    os.makedirs(args.plots, exist_ok=True)
    a, pot = SPS.load(args.mc_ntuple, True, args.novtx, extra=EXTRA)
    w0 = ak.to_numpy(a["xsecWeight"]).astype(np.float64)
    w = np.where(w0 > 0, w0, 0.0) * (args.pot / pot)
    T = SPS.truth_tables(a)
    R = SPS.reco_tables(a, args)
    R["novtx"] = args.novtx
    names, S, E_cand, sc, n_g, path = SPS.build_steps(
        R, args.shower_bdt_min, args.flashchi2_cut, args.bdt_after_multiplicity,
        getattr(args, 'flashpe_ratio_max', None))
    cidx = ak.to_numpy(ak.fill_none(ak.firsts(R["_cand_idx"]), -1)).astype(int)
    cat, fin = T["cat"], S[-1]
    pdg = a["trueSimPartPDG"]; apdg = np.abs(pdg); E = a["trueSimPartE"]; q = a["trueSimPartPixelSumQ"]
    ke = E - SPS._mass_of(pdg)
    evis_g = SPS.A_GAMMA * np.maximum(q, 0.0)
    mu_ok = (apdg == 13) & (ke >= SPS.MU_KE_VIS) & (q > 0)
    lines = []
    allrows = {}
    for c in [int(x) for x in args.cats.split(",")]:
        pop = fin & (cat == c)
        rows = []
        for i in np.nonzero(pop)[0]:
            mm = np.asarray(mu_ok[i])
            if not mm.any():
                continue
            kk = np.asarray(ke[i]); tids = np.asarray(a["trueSimPartTID"][i])
            j = np.nonzero(mm)[0][np.argmax(kk[mm])]
            mtid = int(tids[j]); mke = float(kk[j])
            end = np.array([a["trueSimPartEnd" + s][i][j] for s in "XYZ"], float)
            contained = bool(np.all(end >= TPC_LO) and np.all(end <= TPC_HI))
            ttid = np.asarray(a["trackTrueTID"][i]); stid = np.asarray(a["showerTrueTID"][i])
            jt = np.nonzero(ttid == mtid)[0]; js = np.nonzero(stid == mtid)[0]
            ci = cidx[i]
            r = dict(evt=i, w=w[i], cat=c, mke=mke, contained=contained, recoE=np.nan, comp=np.nan,
                     cand_is_mu=False, mode=-1)
            if len(jt) == 0 and len(js) == 0:
                r["mode"] = 0
            elif len(jt) == 0:
                r["mode"] = 1; r["cand_is_mu"] = (ci >= 0 and ci in js)
            else:
                # the track prong holding the most muon charge
                comps = np.asarray(a["trackTrueComp"][i])[jt]; k = jt[np.argmax(comps)]
                r["comp"] = float(comps.max()); r["recoE"] = float(a["trackRecoE"][i][k])
                te = np.array([a["trackEndPos" + s_][i][k] for s_ in "XYZ"], float)
                r["end_dwall"] = float(min((te - TPC_LO).min(), (TPC_HI - te).min()))
                seg_mu = int(a["trackLArFormerPID"][i][k]) == 13
                lp_mu = (int(a["trackClassified"][i][k]) == 1 and
                         all(float(a["trackMuScore"][i][k]) > float(a["track" + s + "Score"][i][k])
                             for s in ("El", "Ph", "Pi", "Pr")))
                is_mu_pid = seg_mu if args.muon_finder == "segmenter" else (
                    lp_mu if args.muon_finder == "larpid" else (seg_mu or lp_mu))
                if not is_mu_pid:
                    r["mode"] = 2
                elif r["recoE"] > args.mu_ke_min:
                    r["mode"] = 6
                elif not contained:
                    r["mode"] = 3
                elif r["comp"] < args.broken_comp:
                    r["mode"] = 4
                else:
                    r["mode"] = 5
            # photon reality of the accepted candidate
            if ci < 0:
                r["phot"] = 6
            else:
                tp = int(a["showerTruePID"][i][ci]); pur = float(a["showerTruePurity"][i][ci])
                unl = float(a["showerTrueUnlabeledPurity"][i][ci]); ctid = int(stid[ci])
                if unl >= 0.5 or tp == 0 or pur < 0.5:
                    r["phot"] = 5
                elif ctid == mtid or abs(tp) == 13:
                    r["phot"] = 2
                elif tp == 22:
                    kg = np.nonzero(tids == ctid)[0]
                    vis = len(kg) and float(evis_g[i][kg[0]]) >= SPS.EVIS_MIN
                    r["phot"] = 0 if vis else 1
                elif abs(tp) == 11:
                    r["phot"] = 3
                else:
                    r["phot"] = 4
            r["nvisg"] = int(T["n_vis_g"][i])
            r.setdefault("end_dwall", np.nan)
            rows.append(r)
        allrows[c] = rows
        W = np.array([r["w"] for r in rows]); mode = np.array([r["mode"] for r in rows])
        phot = np.array([r["phot"] for r in rows]); nvg = np.array([r["nvisg"] for r in rows])
        lines.append(f"\n===== {SPS.CATS[c]}: {len(rows)} selected events (w {W.sum():.1f}) =====")
        lines.append(f"{'muon failure mode':32s}{'raw':>6}{'w':>8}{'frac':>7}   photon reality of the candidate (w)")
        for m in range(len(MODES)):
            sel = mode == m
            if not sel.any():
                continue
            ph = " | ".join(f"{PHOT[p][:14]} {W[sel & (phot == p)].sum():.1f}" for p in range(len(PHOT)) if (sel & (phot == p)).any())
            lines.append(f"{MODES[m]:32s}{int(sel.sum()):6d}{W[sel].sum():8.1f}{W[sel].sum()/W.sum():7.3f}   {ph}")
        lines.append("photon reality, all modes:      " + " | ".join(
            f"{PHOT[p]} {W[phot == p].sum():.1f} ({W[phot == p].sum()/W.sum():.2f})" for p in range(len(PHOT)) if (phot == p).any()))
        lines.append("true detectable photons/event:  " + " | ".join(
            f"n={n} {W[nvg == n].sum():.1f} ({W[nvg == n].sum()/W.sum():.2f})" for n in (0, 1) if (nvg == n).any())
            + f" | n>=2 {W[nvg >= 2].sum():.1f} ({W[nvg >= 2].sum()/W.sum():.2f})")
        cim = np.array([r["cand_is_mu"] for r in rows])
        if cim.any():
            lines.append(f"  (M1 where the muon shower IS the accepted photon: {W[cim].sum():.1f} w, {int(cim.sum())} raw)")
        cont = np.array([r["contained"] for r in rows])
        lines.append(f"true muon contained (end in TPC): {W[cont].sum()/W.sum():.2f} | median true KE {np.median([r['mke'] for r in rows]):.0f} MeV")
        ed = np.array([r["end_dwall"] for r in rows], float)
        for m in (3, 4, 5):
            sel = (mode == m) & np.isfinite(ed)
            if sel.any():
                lines.append(f"  {MODES[m]}: reco track END dwall median {np.median(ed[sel]):.1f} cm; "
                             f"< 5 cm {W[sel & (ed < 5)].sum()/W[sel].sum():.2f}, < 10 cm {W[sel & (ed < 10)].sum()/W[sel].sum():.2f}, "
                             f"< 20 cm {W[sel & (ed < 20)].sum()/W[sel].sum():.2f}")

    # ---- EXITING-MUON veto scan: veto if any nu-stream mu-PID track (any E)
    # ends within d of a TPC wall -- rescues the range-energy loophole (M3)
    lines.append("\nEXITING-MUON VETO scan over the FULL final selection: veto = a nu-stream track "
                 "with mu PID (union finder, ANY reco E) whose reco end point is < d cm from a wall")
    tpid = a["trackLArFormerPID"]; tcl = a["trackClassified"]
    lp_mu = ((tcl == 1) & (a["trackMuScore"] > a["trackElScore"]) & (a["trackMuScore"] > a["trackPhScore"])
             & (a["trackMuScore"] > a["trackPiScore"]) & (a["trackMuScore"] > a["trackPrScore"]))
    mu_pid = (tpid == 13) | lp_mu
    if args.novtx:
        mu_pid = mu_pid & (a["trackStream"] == 0)
    tend = [a["trackEndPos" + s_] for s_ in "XYZ"]
    edw = ak.zeros_like(a["trackRecoE"]) + 1e9
    for k_, c_ in enumerate(tend):
        edw = np.minimum(edw, np.minimum(c_ - TPC_LO[k_], TPC_HI[k_] - c_))
    grp = {"sig 1g0X": cat == 0, "sig 1g+X": cat == 1, "sig entering": np.isin(cat, [2, 3]),
           "bkg CC numu": cat == 7, "bkg entering CC": cat == 5, "bkg >=2g (FV+ent)": np.isin(cat, [4, 9]),
           "bkg other MC": np.isin(cat, [6, 8, 10])}
    base = {k_: w[fin & m_].sum() for k_, m_ in grp.items()}
    lines.append(f"{'d [cm]':10s}" + "".join(f"{k_[:16]:>18}" for k_ in grp))
    lines.append(f"{'(final w)':10s}" + "".join(f"{base[k_]:18.1f}" for k_ in grp))
    for d_ in (3.0, 5.0, 10.0, 20.0):
        veto = ak.to_numpy(ak.any(mu_pid & (edw < d_), axis=1))
        lines.append(f"{d_:<10.0f}" + "".join(f"{w[fin & m_ & ~veto].sum()/max(base[k_],1e-9):18.3f}" for k_, m_ in grp.items())
                     + "   (fraction KEPT)")
    txt = "\n".join(lines); print(txt)
    with open(os.path.join(args.plots, "ccmu_anatomy.txt"), "w") as fh:
        fh.write(txt + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for c, rows in allrows.items():
        if not rows:
            continue
        W = np.array([r["w"] for r in rows]); mode = np.array([r["mode"] for r in rows])
        tag = SPS.CATS[c].replace(" ", "_").replace(":", "").replace("/", "").replace("(", "").replace(")", "")
        for key, xlab, bins in (("mke", "true muon KE [MeV]", np.linspace(0, 1500, 31)),
                                ("recoE", "muon track prong reco KE [MeV]", np.linspace(0, 600, 31)),
                                ("comp", "muon track prong charge completeness", np.linspace(0, 1.2, 25))):
            v = np.array([r[key] for r in rows], float); ok = np.isfinite(v)
            ms = [m for m in range(len(MODES)) if (ok & (mode == m)).any()]
            fig, ax = plt.subplots(figsize=(7.2, 4.6))
            ax.hist([np.clip(v[ok & (mode == m)], bins[0], bins[-1] - 1e-6) for m in ms], bins=bins,
                    weights=[W[ok & (mode == m)] for m in ms], stacked=True, color=[MCOL[m] for m in ms],
                    label=[f"{MODES[m]} ({W[ok & (mode == m)].sum():.1f})" for m in ms])
            ax.set(xlabel=xlab, ylabel=f"events / {args.pot:.1e} POT", title=f"{SPS.CATS[c]}: {xlab}")
            ax.legend(fontsize=7); ax.grid(alpha=0.3)
            fig.tight_layout(); fig.savefig(f"{args.plots}/{tag}_{key}.png", dpi=110); plt.close(fig)
    print(f">>> plots -> {args.plots}")


if __name__ == "__main__":
    main()
