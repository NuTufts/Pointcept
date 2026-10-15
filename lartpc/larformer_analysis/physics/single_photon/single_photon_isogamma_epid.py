#!/usr/bin/env python3
"""Why are gun photons PID'd as electrons?  (2026-10-10)

One row per lar-visible TRUE photon whose conversion point is >= 10 cm inside the
TPC, in the isotropic 1-gamma GUN and (for comparison) in the held-out run-1 BNB
overlay (photons of any origin, mostly pi0). The photon's nu-stream shower is
found by truth match (showerTrueTID, purity > 0.5; highest reco E if several).
Recorded: LArFormer PID + el/ph scores, LArPID (showerPID + el/ph scores), the
shower-to-vertex gap (showerDistToVtx), the reco vertex distance to the TRUE
conversion point, the photon flight distance (origin -> conversion), energies,
completeness / purity.  Outputs: epid_summary.txt + plots in --plots.

    python3 single_photon_isogamma_epid.py --mc-ntuple <gun> --overlay-ntuple <ovl> --plots plots_isogamma/epid
"""
import argparse, os, sys
import numpy as np, awkward as ak, uproot
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import single_photon_selection as SPS

BR = ["run", "subrun", "event", "foundVertex", "primaryVtxStream", "vtxIsFiducial", "recoVtxX", "recoVtxY", "recoVtxZ", "recoVtxStream",
      "trueVtxX", "trueVtxY", "trueVtxZ", "trueSimPartPDG", "trueSimPartTID", "trueSimPartE", "trueSimPartPixelSumQ", "trueSimPartProcess",
      "trueSimPartEDepX", "trueSimPartEDepY", "trueSimPartEDepZ", "trueSimPartPx", "trueSimPartPy", "trueSimPartPz",
      "showerStream", "showerVtxIdx", "showerIsSecondary", "showerLArFormerPID", "showerLArFormerElScore", "showerLArFormerPhScore",
      "showerPID", "showerElScore", "showerPhScore", "showerClassified", "showerDistToVtx", "showerRecoE", "showerTrueTID", "showerTruePurity",
      "showerTrueComp", "showerNHits", "showerCosmicScore", "showerAttConfident", "showerCosTheta", "showerStartPosX", "showerStartPosY", "showerStartPosZ",
      "trackStream", "trackTrueTID", "trackLArFormerPID", "trackRecoE"]
EBINS = np.array([20, 50, 100, 150, 200, 300, 400, 600, 800, 1000, 1500, 2000, 3000, 5000]); ECTR = np.sqrt(EBINS[:-1] * EBINS[1:])


def photon_table(path, gun, max_events=None):
    t = uproot.open(path)["EventTree"]
    a = t.arrays(BR, entry_stop=max_events)
    pdg, proc = a["trueSimPartPDG"], a["trueSimPartProcess"]
    ph = (pdg == 22) & ((proc == 0) if gun else True)
    conv = [a["trueSimPartEDep" + c] for c in "XYZ"]
    fin = np.ones_like(ph, dtype=bool)
    for c in conv:
        fin = fin & np.isfinite(c) & (abs(c) < 1e4)
    edep = SPS.A_GAMMA * ak.where(a["trueSimPartPixelSumQ"] > 0, a["trueSimPartPixelSumQ"], 0.0)
    lo, hi = SPS.TPC_LO, SPS.TPC_HI
    cx, cy, cz = [ak.fill_none(ak.nan_to_none(c), -999.0) for c in conv]
    dwall = ak.min(ak.concatenate([(cx - lo[0])[..., None], (hi[0] - cx)[..., None], (cy - lo[1])[..., None], (hi[1] - cy)[..., None], (cz - lo[2])[..., None], (hi[2] - cz)[..., None]], axis=-1), axis=-1)
    sel = ph & fin & (edep >= SPS.EVIS_MIN) & (dwall >= 10.0)
    # ---- match photons to nu-stream showers / tracks (truth TID) ----------------
    SHB = [b for b in BR if b.startswith("shower")]
    sh = ak.zip({b: a[b] for b in SHB})
    pairs = ak.cartesian({"p": a["trueSimPartTID"], "s": sh}, nested=True)             # [event][photon][shower] records
    ps = pairs["s"]
    good = (pairs["p"] == ps["showerTrueTID"]) & (ps["showerTruePurity"] > 0.5) & (ps["showerStream"] == 0)
    goodE = ak.where(good, ps["showerRecoE"], -1.0)
    best = ak.argmax(goodE, axis=-1, keepdims=True)                                    # per photon: best shower
    has = ak.fill_none(ak.firsts(ak.max(goodE, axis=-1, keepdims=True), axis=-1), -1.0) > -0.5
    msh = ak.firsts(ps[best], axis=-1)                                                           # [event][photon] matched-shower record (None if no showers)
    # photon reconstructed as a track?
    tpairs = ak.cartesian({"p": a["trueSimPartTID"], "t": ak.zip({"tid": a["trackTrueTID"], "st": a["trackStream"]})}, nested=True)
    trk = ak.any((tpairs["p"] == tpairs["t"]["tid"]) & (tpairs["t"]["st"] == 0), axis=-1)
    # ---- per-event reco vertex (nu stream) -------------------------------------
    nuv = a["recoVtxStream"] == 0
    vx = [np.asarray(ak.to_numpy(ak.fill_none(ak.firsts(a["recoVtx" + c][nuv]), np.nan)), np.float64) for c in "XYZ"]
    vok = (ak.to_numpy(a["foundVertex"]) == 1) & (ak.to_numpy(a["primaryVtxStream"]) == 0)
    for v in vx:
        v[~vok] = np.nan
    ovx = [np.asarray(ak.to_numpy(a["trueVtx" + c]), np.float64) for c in "XYZ"]
    # ---- flatten to photon rows ------------------------------------------------
    ne = ak.to_numpy(ak.num(a["trueSimPartTID"])); evt_of = np.repeat(np.arange(len(ne)), ne)
    f = lambda x: np.asarray(ak.to_numpy(ak.flatten(x)), np.float64)
    def fsh(b, fill=np.nan):          # shower quantity of the matched shower (per photon)
        v = ak.flatten(ak.fill_none(msh[b], fill)); return np.asarray(ak.to_numpy(v), np.float64)
    keep = f(sel) > 0.5
    T = dict(evt=evt_of[keep], E=f(a["trueSimPartE"])[keep], Edep=f(edep)[keep], dwall=f(dwall)[keep], matched=(f(has) > 0.5)[keep], astrack=(f(trk) > 0.5)[keep])
    cxyz = np.stack([f(cx), f(cy), f(cz)], 1)[keep]
    T["conv"] = cxyz
    T["vtx_conv"] = np.linalg.norm(np.stack(vx, 1)[T["evt"]] - cxyz, axis=1)              # NaN = no nu vertex
    T["flight"] = np.linalg.norm(np.stack(ovx, 1)[T["evt"]] - cxyz, axis=1)              # origin -> conversion
    p = np.stack([f(a["trueSimPart" + c]) for c in ("Px", "Py", "Pz")], 1)[keep]
    T["cosBeam"] = p[:, 2] / np.maximum(np.linalg.norm(p, axis=1), 1e-9)
    for k, b in (("lfpid", "showerLArFormerPID"), ("lfel", "showerLArFormerElScore"), ("lfph", "showerLArFormerPhScore"), ("lpid", "showerPID"), ("lel", "showerElScore"),
                 ("lph", "showerPhScore"), ("lcls", "showerClassified"), ("gap", "showerDistToVtx"), ("recoE", "showerRecoE"), ("comp", "showerTrueComp"), ("pur", "showerTruePurity"),
                 ("nhits", "showerNHits"), ("sec", "showerIsSecondary"), ("vtxidx", "showerVtxIdx"), ("cosmic", "showerCosmicScore"), ("conf", "showerAttConfident"), ("cos", "showerCosTheta"),
                 ("sx", "showerStartPosX"), ("sy", "showerStartPosY"), ("sz", "showerStartPosZ")):
        v = fsh(b)[keep]; v[~T["matched"]] = np.nan; T[k] = v
    T["start_conv"] = np.linalg.norm(np.stack([T["sx"], T["sy"], T["sz"]], 1) - cxyz, axis=1)
    T["n_events"] = len(ne)
    return T


def frac(mask, den, x, bins):
    e, er = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        b = den & (x >= lo) & (x < hi); n = b.sum(); k = (mask & b).sum()
        e.append(k / n if n else np.nan); er.append(np.sqrt(max(k, 1) * (1 - k / n)) / n if n else np.nan)
    return np.array(e), np.array(er)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mc-ntuple", required=True, help="gun ntuple"); ap.add_argument("--overlay-ntuple", default="")
    ap.add_argument("--overlay-max-events", type=int, default=None); ap.add_argument("--plots", required=True)
    args = ap.parse_args(); os.makedirs(args.plots, exist_ok=True)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    S = {"gun": photon_table(args.mc_ntuple, True)}
    if args.overlay_ntuple:
        S["overlay"] = photon_table(args.overlay_ntuple, False, args.overlay_max_events)
    L = []
    for nm, T in S.items():
        m = T["matched"]; e = m & (T["lfpid"] == 11); g = m & (T["lfpid"] == 22)
        L.append(f"\n=== {nm}: {T['n_events']} events; lar-visible photons with conversion >= 10 cm inside the TPC: {len(m)}; matched to a nu-stream shower {m.mean():.3f}; "
                 f"matched only as a track {(T['astrack'] & ~m).mean():.3f}; unmatched {(~m & ~T['astrack']).mean():.3f}")
        L.append(f"  LArFormer PID of the matched shower: photon {g.sum() / m.sum():.3f}, electron {e.sum() / m.sum():.3f}, other {(m & ~e & ~g).sum() / m.sum():.3f}")
        lc = m & (T["lcls"] == 1)
        L.append(f"  LArPID (showerPID) available (showerClassified) {lc.sum() / m.sum():.3f}; of those photon {((T['lpid'] == 22) & lc).sum() / max(lc.sum(), 1):.3f}, electron {((T['lpid'] == 11) & lc).sum() / max(lc.sum(), 1):.3f}")
        L.append("  LArFormer x LArPID confusion (matched & classified), rows LArFormer e/g, cols LArPID e/g/other:")
        for lf in (11, 22):
            r = lc & (T["lfpid"] == lf)
            L.append(f"    LArFormer {lf:2d}: " + "  ".join(f"LArPID {lp}: {((T['lpid'] == lp) & r).sum():6d}" for lp in (11, 22)) + f"  other: {(r & (T['lpid'] != 11) & (T['lpid'] != 22)).sum():6d}")
        if nm == "gun":
            L.append(f"  recoverable if LArPID photon overrides LArFormer electron: {((T['lfpid'] == 11) & (T['lpid'] == 22) & lc).sum()} of {e.sum()} electron-tagged photons; "
                     f"lost the other way (LArFormer g, LArPID e): {((T['lfpid'] == 22) & (T['lpid'] == 11) & lc).sum()} of {g.sum()}")
        L.append("  electron fraction (of matched showers) by true E bin:")
        fe, _ = frac(e, m, T["E"], EBINS); L.append("    " + " ".join(f"[{lo:.0f},{hi:.0f}) {v:.2f}" for lo, hi, v in zip(EBINS[:-1], EBINS[1:], fe)))
        for lab, x, bins in (("shower-to-vertex gap showerDistToVtx [cm]", T["gap"], np.array([0, 0.5, 1, 2, 3, 5, 10, 20, 50, 1e9])),
                             ("reco vertex -> true conversion point [cm]", T["vtx_conv"], np.array([0, 1, 2, 3, 5, 10, 20, 50, 1e9])),
                             ("photon flight distance origin->conversion [cm]", T["flight"], np.array([0, 1, 2, 3, 5, 10, 20, 50, 100, 1e9])),
                             ("shower start -> true conversion point [cm]", T["start_conv"], np.array([0, 1, 2, 3, 5, 10, 20, 50, 1e9])),
                             ("showerTrueComp", T["comp"], np.array([0, .2, .4, .6, .8, .9, 1.01])), ("showerTruePurity", T["pur"], np.array([.5, .7, .8, .9, .95, 1.01])),
                             ("|cos theta_beam| (true)", np.abs(T["cosBeam"]), np.array([0, .2, .4, .6, .8, .9, 1.01]))):
            fe, _ = frac(e, m & np.isfinite(x), x, bins); nn = [int((m & np.isfinite(x) & (x >= lo) & (x < hi)).sum()) for lo, hi in zip(bins[:-1], bins[1:])]
            L.append(f"  electron fraction vs {lab}: " + " ".join(f"[{lo:g},{hi:g}) {v:.2f} (n {k})" for lo, hi, v, k in zip(bins[:-1], bins[1:], fe, nn)))
        L.append(f"  medians (electron-tagged | photon-tagged): gap {np.nanmedian(T['gap'][e]):.2f} | {np.nanmedian(T['gap'][g]):.2f} cm; vtx->conv {np.nanmedian(T['vtx_conv'][e]):.2f} | {np.nanmedian(T['vtx_conv'][g]):.2f} cm; "
                 f"flight {np.nanmedian(T['flight'][e]):.1f} | {np.nanmedian(T['flight'][g]):.1f} cm; recoE/trueE {np.nanmedian(T['recoE'][e] / T['E'][e]):.2f} | {np.nanmedian(T['recoE'][g] / T['E'][g]):.2f}; "
                 f"comp {np.nanmedian(T['comp'][e]):.2f} | {np.nanmedian(T['comp'][g]):.2f}; secondary {np.nanmean(T['sec'][e]):.2f} | {np.nanmean(T['sec'][g]):.2f}; attached to vtx 0 {np.nanmean(T['vtxidx'][e] == 0):.2f} | {np.nanmean(T['vtxidx'][g] == 0):.2f}")
        L.append(f"  LArFormer photon score (ph / (ph+el)) median e-tagged {np.nanmedian((T['lfph'] / (T['lfph'] + T['lfel']))[e]):.2f}, g-tagged {np.nanmedian((T['lfph'] / (T['lfph'] + T['lfel']))[g]):.2f}; "
                 f"fraction of e-tagged with ph/(ph+el) > 0.3: {np.nanmean(((T['lfph'] / (T['lfph'] + T['lfel'])) > 0.3)[e]):.2f}")
    txt = "\n".join(L); print(txt); open(os.path.join(args.plots, "epid_summary.txt"), "w").write(txt + "\n")
    # ---- plots -----------------------------------------------------------------
    cols = {"gun": "C3", "overlay": "C0"}
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for nm, T in S.items():
        m = T["matched"]; e = m & (T["lfpid"] == 11)
        fe, er = frac(e, m, T["E"], EBINS); ax.errorbar(ECTR, fe, yerr=er, fmt="o-", ms=4, color=cols[nm], label=f"{nm}: LArFormer PID = electron")
        lc = m & (T["lcls"] == 1); fe, er = frac(lc & (T["lpid"] == 11), lc, T["E"], EBINS); ax.errorbar(ECTR, fe, yerr=er, fmt="s--", ms=4, color=cols[nm], alpha=0.6, label=f"{nm}: LArPID = electron")
    ax.set(xscale="log", xlabel="true photon energy [MeV]", ylabel="fraction of matched nu-stream photon showers", ylim=(0, 1), title="photon showers tagged as electrons vs true E"); ax.grid(alpha=.3, which="both"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(args.plots, "efrac_vs_trueE.png"), dpi=110); plt.close(fig)
    for key, lab, bins, fn in (("gap", "shower-to-vertex gap showerDistToVtx [cm]", np.linspace(0, 30, 31), "efrac_vs_gap"), ("vtx_conv", "reco vertex to true conversion point [cm]", np.linspace(0, 30, 31), "efrac_vs_vtxconv"),
                               ("flight", "photon flight distance origin to conversion [cm]", np.linspace(0, 100, 26), "efrac_vs_flight"), ("comp", "showerTrueComp (completeness)", np.linspace(0, 1, 21), "efrac_vs_comp"),
                               ("start_conv", "shower start to true conversion point [cm]", np.linspace(0, 30, 31), "efrac_vs_startconv")):
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
        for nm, T in S.items():
            m = T["matched"] & np.isfinite(T[key]); e = m & (T["lfpid"] == 11); x = np.clip(T[key], bins[0], bins[-1] - 1e-6)
            fe, er = frac(e, m, x, bins); ax.errorbar(0.5 * (bins[:-1] + bins[1:]), fe, yerr=er, fmt="o-", ms=3, color=cols[nm], label=nm)
            ax2.hist(x[m & (T["lfpid"] == 22)], bins=bins, histtype="step", color=cols[nm], label=f"{nm} PID photon", density=True)
            ax2.hist(x[e], bins=bins, histtype="step", color=cols[nm], ls="--", label=f"{nm} PID electron", density=True)
        ax.set(xlabel=lab, ylabel="electron-tagged fraction", ylim=(0, 1)); ax.grid(alpha=.3); ax.legend(fontsize=8)
        ax2.set(xlabel=lab, ylabel="density (last bin = overflow)"); ax2.grid(alpha=.3); ax2.legend(fontsize=7); fig.tight_layout(); fig.savefig(os.path.join(args.plots, fn + ".png"), dpi=110); plt.close(fig)
    # score distributions
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
    for nm, T in S.items():
        m = T["matched"]; r = T["lfph"] / np.maximum(T["lfph"] + T["lfel"], 1e-9)
        axs[0].hist(r[m], bins=np.linspace(0, 1, 41), histtype="step", color=cols[nm], label=nm, density=True)
        lc = m & (T["lcls"] == 1); r2 = T["lph"] - T["lel"]
        axs[1].hist(np.clip(r2[lc], -20, 20), bins=np.linspace(-20, 20, 41), histtype="step", color=cols[nm], label=nm, density=True)
    axs[0].set(xlabel="LArFormer ph / (ph + el) score of the photon shower", ylabel="density", yscale="log"); axs[1].set(xlabel="LArPID showerPhScore - showerElScore (raw, clipped)", yscale="log")
    for ax in axs:
        ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(args.plots, "pid_scores.png"), dpi=110); plt.close(fig)
    # 2-D: electron fraction vs (true E, gap) for the gun
    T = S["gun"]; m = T["matched"] & np.isfinite(T["gap"]); e = m & (T["lfpid"] == 11)
    gb = np.array([0, 1, 2, 5, 10, 1e9]); H = np.full((len(gb) - 1, len(EBINS) - 1), np.nan)
    for i in range(len(gb) - 1):
        for j in range(len(EBINS) - 1):
            b = m & (T["gap"] >= gb[i]) & (T["gap"] < gb[i + 1]) & (T["E"] >= EBINS[j]) & (T["E"] < EBINS[j + 1])
            if b.sum() >= 20: H[i, j] = (e & b).sum() / b.sum()
    fig, ax = plt.subplots(figsize=(8, 4.2)); im = ax.imshow(H, origin="lower", aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(EBINS) - 1)); ax.set_xticklabels([f"{lo:.0f}" for lo in EBINS[:-1]], fontsize=7); ax.set_yticks(range(len(gb) - 1)); ax.set_yticklabels([f"{lo:g}-{hi:g}" for lo, hi in zip(gb[:-1], gb[1:])])
    ax.set(xlabel="true E bin lower edge [MeV]", ylabel="shower-to-vertex gap [cm]", title="gun: electron-tagged fraction (cells with >= 20 photons)")
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            if np.isfinite(H[i, j]): ax.text(j, i, f"{H[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color="w" if H[i, j] < 0.6 else "k")
    fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(os.path.join(args.plots, "efrac_E_vs_gap_gun.png"), dpi=110); plt.close(fig)
    np.savez(os.path.join(args.plots, "epid_tables.npz"), **{f"{nm}_{k}": v for nm, T in S.items() for k, v in T.items() if isinstance(v, np.ndarray)})
    print(">>> ->", args.plots)


if __name__ == "__main__":
    main()
