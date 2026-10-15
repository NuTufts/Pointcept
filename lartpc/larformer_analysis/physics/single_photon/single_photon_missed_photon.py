"""Where does the SECOND photon go?  Missed-photon anatomy for the in-FV
">=2 visible photon" background of the single-photon selection (MC only).

Population: events passing the full selection whose true vertex is in the FV
and that have >= 2 true detectable photons (E_vis >= 20 MeV). The accepted
candidate is truth-matched to one of them (or to nothing); every OTHER
detectable photon is a "missed photon". Because the exporter writes every
nu-slice segmenter particle (attached AND vertex-less prongs) with its truth
TID, each missed photon can be followed through the reco from the ntuple:

  stage 0  no prong        no shower/track prong carries its TID (not found in
                           the nu slice, deghosted, mis-sliced, or absorbed
                           without a truth signature -- ntuple cannot split these)
  stage 1  merged          no prong, but the accepted candidate's photon-species
                           charge purity exceeds its matched-TID purity by
                           > --merge-frac: the two photons share one prong
  stage 2  track prong     the segmenter mask exists but was labelled a track
  stage 3  electron prong  mask exists, labelled electron shower
  stage 4  photon, soft    photon prong with reco E <= 20 MeV (the truth
                           visibility uses A_GAMMA x Q ~ 2.2 x E_true, so a
                           "visible" photon can be < 20 MeV in calibrated reco E)
  stage 5  attached, BDT   photon prong on the primary in-FV vertex, cosmic BDT
                           below --shower-bdt-min
  stage 6  unattached, BDT photon prong NOT on the primary vertex, vertex-free
                           BDT below --novtx-bdt-min
For every missed photon: true E, E_vis, distance true vertex -> conversion
point, conversion x/y/z + dwall, stacked by stage. For stages 2-6 (a prong
exists): reco E, its BDT score, and the PROVISIONAL two-photon invariant mass
with the accepted candidate (vertex->start directions when the primary vertex
exists, else the prongs' start directions), next to the truth m_gg from the
true photon momenta. Finally a SECOND-PHOTON VETO scan: veto the event if any
other nu-stream photon (or e) shower prong has reco E > e_min [and m_gg in a
window]; reports background removed vs signal kept, per category.

    python3 single_photon_missed_photon.py --mc-ntuple <run-1 halves> --novtx \
        --shower-bdt-min 0.164 --plots plots_run1_novtx/missed_photon
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

EXTRA = ["showerTrueTID", "showerTruePurity", "showerTruePhPurity", "showerTrueComp",
         "showerStartPosX", "showerStartPosY", "showerStartPosZ",
         "showerStartDirX", "showerStartDirY", "showerStartDirZ",
         "trackTrueTID", "trackLArFormerPID", "trackRecoE", "trackVtxIdx", "trackStream",
         "trueSimPartEDepX", "trueSimPartEDepY", "trueSimPartEDepZ",
         "trueSimPartPx", "trueSimPartPy", "trueSimPartPz",
         "trueVtxX", "trueVtxY", "trueVtxZ", "vtxX", "vtxY", "vtxZ"]
STAGES = ["no prong", "merged into candidate", "track prong (mis-PID)",
          "electron prong", "photon prong, reco E <= 20", "attached, cosmic-BDT fail",
          "unattached, vertex-free-BDT fail", "other"]
ST_COL = ["#7f7f7f", "#8c564b", "#1f77b4", "#2ca02c", "#ff7f0e", "#d62728",
          "#9467bd", "#bcbd22"]
PRONG_STAGES = (2, 3, 4, 5, 6)


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else None


def mgg(E1, d1, E2, d2):
    if d1 is None or d2 is None:
        return np.nan
    return float(np.sqrt(max(2.0 * E1 * E2 * (1.0 - float(d1 @ d2)), 0.0)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_args(ap)
    ap.add_argument("--merge-frac", type=float, default=0.2)
    ap.add_argument("--veto-emins", default="5,10,15,20,30")
    ap.add_argument("--mgg-window", default="50,250")
    args = ap.parse_args()
    SPS.configure(args)
    os.makedirs(args.plots, exist_ok=True)
    a, pot = SPS.load(args.mc_ntuple, True, args.novtx, extra=EXTRA)
    w0 = ak.to_numpy(a["xsecWeight"]).astype(np.float64)
    w = np.where(w0 > 0, w0, 0.0) * (args.pot / pot)
    if args.mc_odd_only:
        w = np.where(ak.to_numpy(a["event"]) % 2 == 1, 2.0 * w, 0.0)
    T = SPS.truth_tables(a)
    R = SPS.reco_tables(a, args)
    R["novtx"] = args.novtx
    names, S, E_cand, sc_cand, n_g, path = SPS.build_steps(
        R, args.shower_bdt_min, args.flashchi2_cut, args.bdt_after_multiplicity, args.flashpe_ratio_max)
    cidx = ak.to_numpy(ak.fill_none(ak.firsts(R["_cand_idx"]), -1)).astype(int)
    cat, sig = T["cat"], T["sig_incl"]
    fin = S[-1]
    pop = fin & T["in_fv"] & (T["n_vis_g"] >= 2)
    print(f">>> selected in-FV events with >=2 detectable photons: {int(pop.sum())} raw, "
          f"{w[pop].sum():.1f} w; by category: "
          + ", ".join(f"{SPS.CATS[c]} {int((pop & (cat == c)).sum())}"
                      for c in np.unique(cat[pop])))

    pdg = a["trueSimPartPDG"]; q = a["trueSimPartPixelSumQ"]
    evis_all = SPS.A_GAMMA * np.maximum(q, 0.0)
    vis_g = (np.abs(pdg) == 22) & (evis_all >= SPS.EVIS_MIN)
    E_rec = R["_E"]                                  # recal'd shower energies
    lo, hi = np.array([0.0, -116.5, 0.0]), np.array([256.35, 116.5, 1036.8])

    rows = []                    # one per missed photon
    for i in np.nonzero(pop)[0]:
        vm = np.asarray(vis_g[i])
        tids = np.asarray(a["trueSimPartTID"][i])[vm]
        tE = np.asarray(a["trueSimPartE"][i])[vm]
        tQ = np.asarray(evis_all[i])[vm]
        ED = np.stack([np.asarray(a["trueSimPartEDep" + c][i])[vm] for c in "XYZ"], 1)
        P = np.stack([np.asarray(a["trueSimPartP" + c][i])[vm] for c in ("x", "y", "z")], 1)
        tv = np.array([a["trueVtxX"][i], a["trueVtxY"][i], a["trueVtxZ"][i]], float)
        ci = cidx[i]
        stid = np.asarray(a["showerTrueTID"][i]); ttid = np.asarray(a["trackTrueTID"][i])
        spid = np.asarray(a["showerLArFormerPID"][i]); sE = np.asarray(E_rec[i])
        svi = np.asarray(a["showerVtxIdx"][i]); scs = np.asarray(a["showerCosmicScore"][i])
        snv = np.asarray(a["showerNoVtxScore"][i])
        sp = np.stack([np.asarray(a["showerStartPos" + c][i]) for c in "XYZ"], 1)
        sd = np.stack([np.asarray(a["showerStartDir" + c][i]) for c in "XYZ"], 1)
        ctid = int(stid[ci]) if ci >= 0 else -999
        cpur = float(a["showerTruePurity"][i][ci]) if ci >= 0 else np.nan
        cphp = float(a["showerTruePhPurity"][i][ci]) if ci >= 0 else np.nan
        vtx = (np.array([a["vtxX"][i], a["vtxY"][i], a["vtxZ"][i]], float)
               if R["vtx_ok"][i] else None)
        # accepted candidate direction + true direction of the matched photon
        cE = float(sE[ci]) if ci >= 0 else np.nan
        cdir_v = unit(sp[ci] - vtx) if (ci >= 0 and vtx is not None) else None
        cdir_s = unit(sd[ci]) if ci >= 0 else None
        kc = np.nonzero(tids == ctid)[0]
        ctrue_p = P[kc[0]] if len(kc) else None
        for k, t in enumerate(tids):
            if t == ctid:
                continue
            js = np.nonzero(stid == t)[0]; jt = np.nonzero(ttid == t)[0]
            r = dict(evt=i, cat=int(cat[i]), w=w[i], trueE=float(tE[k]), evis=float(tQ[k]),
                     dconv=float(np.linalg.norm(ED[k] - tv)), cx=ED[k, 0], cy=ED[k, 1], cz=ED[k, 2],
                     dwall=float(min((ED[k] - lo).min(), (hi - ED[k]).min())),
                     recoE=np.nan, score=np.nan, mgg_v=np.nan, mgg_s=np.nan, mgg_true=np.nan,
                     path=int(path[i]))
            if ctrue_p is not None:
                d1, d2 = unit(ctrue_p), unit(P[k])
                if d1 is not None and d2 is not None:
                    r["mgg_true"] = mgg(float(np.linalg.norm(ctrue_p)), d1,
                                        float(np.linalg.norm(P[k])), d2)
            if len(js) == 0 and len(jt) == 0:
                r["stage"] = 1 if (np.isfinite(cpur) and cphp - cpur > args.merge_frac) else 0
            elif len(js) == 0:
                r["stage"] = 2
                r["recoE"] = float(a["trackRecoE"][i][jt[0]])
            else:
                j = js[0]
                r["recoE"] = float(sE[j])
                if spid[j] == 11:
                    r["stage"] = 3
                elif spid[j] != 22:
                    r["stage"] = 7
                elif sE[j] <= SPS.RECO_G_MIN:
                    r["stage"] = 4
                elif svi[j] == 0 and R["vtx_ok"][i]:
                    r["stage"] = 5 if scs[j] < args.shower_bdt_min else 7
                    r["score"] = float(scs[j])
                else:
                    r["stage"] = 6 if snv[j] < args.novtx_bdt_min else 7
                    r["score"] = float(snv[j])
                if ci >= 0:
                    d2v = unit(sp[j] - vtx) if vtx is not None else None
                    r["mgg_v"] = mgg(cE, cdir_v, float(sE[j]), d2v)
                    r["mgg_s"] = mgg(cE, cdir_s, float(sE[j]), unit(sd[j]))
            rows.append(r)
    keys = list(rows[0].keys())
    D = {k: np.array([r[k] for r in rows], dtype=(np.int64 if k in ("evt", "cat", "stage", "path") else np.float64))
         for k in keys}
    nrow, W = len(rows), D["w"]
    lines = [f"missed photons: {nrow} raw in {int(pop.sum())} events",
             f"{'stage':36s}{'raw':>6}{'frac':>7}{'w':>8}{'wfrac':>7}"]
    for s in range(len(STAGES)):
        m = D["stage"] == s
        if m.any():
            lines.append(f"{STAGES[s]:36s}{int(m.sum()):6d}{m.mean():7.3f}{W[m].sum():8.1f}{W[m].sum()/W.sum():7.3f}")
    lines.append("by event category (weighted, stage fractions):")
    for c in np.unique(D["cat"]):
        mc = D["cat"] == c
        lines.append(f"  {SPS.CATS[c]:28s} w={W[mc].sum():6.1f}: " + " | ".join(
            f"{STAGES[s][:14]} {W[mc & (D['stage']==s)].sum()/W[mc].sum():.2f}"
            for s in range(len(STAGES)) if (mc & (D["stage"] == s)).any()))

    # ---- plots ---------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def stacked(key, xlab, bins, sel=None, fname=None, logy=False):
        m = np.isfinite(D[key]) if sel is None else (np.isfinite(D[key]) & sel)
        lo_, hi_ = bins[0], bins[-1] - 1e-6
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        st = [s for s in range(len(STAGES)) if (m & (D["stage"] == s)).any()]
        ax.hist([np.clip(D[key][m & (D["stage"] == s)], lo_, hi_) for s in st], bins=bins,
                weights=[W[m & (D["stage"] == s)] for s in st], stacked=True,
                color=[ST_COL[s] for s in st],
                label=[f"{STAGES[s]} ({W[m & (D['stage']==s)].sum():.1f})" for s in st])
        ax.set(xlabel=xlab, ylabel=f"missed photons / {args.pot:.1e} POT",
               title=f"missed 2nd photon: {xlab}")
        if logy:
            ax.set_yscale("log")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(f"{args.plots}/{fname or key}.png", dpi=110); plt.close(fig)

    stacked("trueE", "missed photon true E [MeV]", np.linspace(0, 600, 41))
    stacked("evis", "missed photon deposited E [MeV] (a_gamma x Q)", np.linspace(0, 400, 41))
    stacked("dconv", "true vertex -> conversion point [cm]", np.linspace(0, 200, 41))
    stacked("cx", "conversion x [cm]", np.linspace(0, 256.35, 33))
    stacked("cy", "conversion y [cm]", np.linspace(-116.5, 116.5, 30))
    stacked("cz", "conversion z [cm]", np.linspace(0, 1036.8, 40))
    stacked("dwall", "conversion point dwall [cm]", np.linspace(0, 130, 27))
    stacked("mgg_true", "TRUE m_gg (candidate photon, missed photon) [MeV]",
            np.linspace(0, 400, 41))
    has = np.isin(D["stage"], PRONG_STAGES)
    stacked("recoE", "missed photon prong reco E [MeV]", np.linspace(0, 300, 61), has)
    stacked("score", "missed photon prong BDT score (cosmic or vertex-free)", np.linspace(0, 1, 41),
            np.isin(D["stage"], (5, 6)))
    stacked("mgg_v", "provisional m_gg, vertex->start dirs [MeV]", np.linspace(0, 500, 51), has)
    stacked("mgg_s", "provisional m_gg, prong start dirs [MeV]", np.linspace(0, 500, 51), has)
    # reco vs true m_gg for the prong stages
    m = has & np.isfinite(D["mgg_v"]) & np.isfinite(D["mgg_true"])
    fig, ax = plt.subplots(figsize=(5.6, 5))
    ax.scatter(D["mgg_true"][m], D["mgg_v"][m], s=6, alpha=0.5, c=[ST_COL[s] for s in D["stage"][m]])
    ax.plot([0, 400], [0, 400], "k--", lw=1)
    ax.set(xlabel="true m_gg [MeV]", ylabel="provisional reco m_gg [MeV]", xlim=(0, 400), ylim=(0, 500),
           title="missed-photon prong: provisional m_gg vs truth")
    ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(f"{args.plots}/mgg_reco_vs_true.png", dpi=110)
    plt.close(fig)

    # ---- second-photon veto scan (all selected events) -----------------------
    # second photon = any OTHER nu-stream shower prong with PID 22 (or 11 when
    # --veto-electrons... kept simple: photon-only and photon+electron variants)
    wlo, whi = [float(x) for x in args.mgg_window.split(",")]
    emins = [float(x) for x in args.veto_emins.split(",")]
    sel_all = np.nonzero(fin)[0]
    spid_all = a["showerLArFormerPID"]; strm = a["showerStream"]
    # precompute, per selected event, the max reco E of other photon / e prongs
    # and whether any of them has m_gg in the window with the candidate
    maxE_g = np.full(len(w), np.nan); maxE_ge = np.full(len(w), np.nan)
    inwin_g = np.zeros(len(w), bool)
    for i in sel_all:
        ci = cidx[i]
        if ci < 0:
            continue
        spid = np.asarray(spid_all[i]); sE = np.asarray(E_rec[i]); st = np.asarray(strm[i])
        sp = np.stack([np.asarray(a["showerStartPos" + c][i]) for c in "XYZ"], 1)
        sd = np.stack([np.asarray(a["showerStartDir" + c][i]) for c in "XYZ"], 1)
        vtx = (np.array([a["vtxX"][i], a["vtxY"][i], a["vtxZ"][i]], float) if R["vtx_ok"][i] else None)
        cdir = unit(sp[ci] - vtx) if vtx is not None else unit(sd[ci])
        others = [j for j in range(len(spid)) if j != ci and st[j] == 0]
        og = [j for j in others if spid[j] == 22]; oge = [j for j in others if spid[j] in (22, 11)]
        if og:
            maxE_g[i] = max(sE[j] for j in og)
        if oge:
            maxE_ge[i] = max(sE[j] for j in oge)
        for j in og:
            d2 = unit(sp[j] - vtx) if vtx is not None else unit(sd[j])
            mm = mgg(float(sE[ci]), cdir, float(sE[j]), d2)
            if wlo <= mm <= whi:
                inwin_g[i] = True
    grp = {"sig in-FV 1g0X": cat == 0, "sig in-FV 1g+X": cat == 1, "sig entering (both)": np.isin(cat, [2, 3]),
           "bkg in-FV >=2g": cat == 9, "bkg in-FV CC numu": cat == 7, "bkg entering >=2g": cat == 4,
           "bkg all other MC": np.isin(cat, [5, 6, 8, 10])}
    base = {k: w[fin & m].sum() for k, m in grp.items()}
    lines.append("\nSECOND-PHOTON VETO scan: fraction of the FINAL selection KEPT per group "
                 "(veto = another nu-stream photon prong with reco E > e_min; "
                 f"'+win' additionally requires m_gg(candidate, that prong) in [{wlo:.0f},{whi:.0f}]; "
                 "'g+e' counts electron-labelled prongs too)")
    lines.append(f"{'veto':22s}" + "".join(f"{k[:16]:>18}" for k in grp))
    lines.append(f"{'(final counts, w)':22s}" + "".join(f"{base[k]:18.1f}" for k in grp))
    for em in emins:
        for lab, keep in ((f"g E>{em:.0f}", ~(maxE_g > em)),
                          (f"g E>{em:.0f} +win", ~((maxE_g > em) & inwin_g)),
                          (f"g+e E>{em:.0f}", ~(maxE_ge > em))):
            lines.append(f"{lab:22s}" + "".join(
                f"{w[fin & m & keep].sum() / max(base[k], 1e-9):18.3f}" for k, m in grp.items()))
    txt = "\n".join(lines)
    print("\n" + txt)
    with open(os.path.join(args.plots, "missed_photon_stages.txt"), "w") as fh:
        fh.write(txt + "\n")
    np.savez(os.path.join(args.plots, "missed_photons.npz"), **D)
    print(f">>> plots -> {args.plots}")


if __name__ == "__main__":
    main()
