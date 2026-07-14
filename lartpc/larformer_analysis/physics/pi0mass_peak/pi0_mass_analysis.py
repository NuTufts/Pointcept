"""pi0 mass-peak benchmark analysis -- ntuple-only (analyzer-facing test).

Signal (truth, per README): true vertex in the WireCell FV; exactly ONE
primary final-state pi0 (truePrimPartPDG==111) decaying to two DETECTABLE
photons -- E_vis = 0.0253 MeV/ADC x trueSimPartPixelSumQ > 20 MeV for both
(the compare_to_legacy_ntuple convention). The pi0's photon pair is the
MID group of Process==1 photons whose mother id is absent from the
trueSimPartTID table (the primary pi0 itself is dropped from trueSimPart;
secondary pi0s appear and their photons form separate MID groups).
Split by trueNuCCNC. Backgrounds split: out-of-FV / no-primary-pi0 /
multi-pi0 / pi0-but-undetectable-photons.

Reco selection (nu stream, default = every ATTACHED shower counts):
  foundVertex & primaryVtxStream==0 & vtxIsFiducial==1;
  photons = showers with showerLArFormerPID==22 and showerRecoE > 20 MeV;
  default >= 2 photons (two most energetic used); 'exactly 2' variant too;
  reco-CC = any primary track (trackIsSecondary==0) with
  trackLArFormerPID==13 and trackRecoE > 100 MeV.

m_gg = sqrt(2 E1 E2 (1 - cos theta12)), with the photon direction computed
two ways: (a) vertex -> shower start (default candidate), (b) the trunk
StartDir; a truth direction-resolution comparison picks the winner.

Weights: xsecWeight scaled to --pot (4.4e19, the Tufts beam-data livetime);
sample POT = sum(potTree totGoodPOT).

    PYTHONPATH=./ python3 .../pi0_mass_analysis.py \
        --ntuple .../dlgen2_larformer_ntuple_*.root --plots plots/
"""
import argparse
import os

import numpy as np
import uproot
import awkward as ak

A_GAMMA = 0.0253017          # MeV/ADC (trajfit calo calib, gamma)
EVIS_MIN = 20.0              # detectable-photon threshold [MeV]
RECO_G_MIN = 20.0            # reco photon energy cut [MeV]
MU_KE_MIN = 100.0            # reco muon KE for the CC tag [MeV]
PI0_MASS = 134.977

CATS = ["signal CC", "signal NC", "bkg: out-of-FV", "bkg: no primary pi0",
        "bkg: multi pi0", "bkg: pi0 undetectable"]
CAT_COLORS = ["#d62728", "#1f77b4", "#bdbdbd", "#ff7f0e", "#9467bd",
              "#8c564b"]


def truth_category(a, i):
    """Per-event truth tag (index into CATS)."""
    if a["trueVtxInWCFV"][i] != 1:
        return 2
    npi0 = int(np.sum(np.asarray(a["truePrimPartPDG"][i]) == 111))
    if npi0 == 0:
        return 3
    if npi0 > 1:
        return 4
    # primary pi0 photon pair: Process==1 photons whose MID is not a TID
    pdg = np.asarray(a["trueSimPartPDG"][i])
    proc = np.asarray(a["trueSimPartProcess"][i])
    mid = np.asarray(a["trueSimPartMID"][i])
    tid = set(np.asarray(a["trueSimPartTID"][i]).tolist())
    q = np.asarray(a["trueSimPartPixelSumQ"][i])
    ph = (np.abs(pdg) == 22) & (proc == 1)
    orphan = ph & np.asarray([int(m) not in tid for m in mid])
    mids, cnt = (np.unique(mid[orphan], return_counts=True)
                 if orphan.any() else (np.array([]), np.array([])))
    pair_ok = False
    for m, c in zip(mids, cnt):
        if c != 2:
            continue
        sel = orphan & (mid == m)
        evis = A_GAMMA * np.clip(q[sel], 0, None)
        if np.all(evis > EVIS_MIN):
            pair_ok = True
            break
    if not pair_ok:
        return 5
    return 0 if a["trueNuCCNC"][i] == 0 else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ntuple", required=True)
    ap.add_argument("--plots", required=True)
    ap.add_argument("--pot", type=float, default=4.4e19)
    ap.add_argument("--out", default=None)
    ap.add_argument("--cascade-dir", default=None,
                    help="keypoint2_streams dir; if given, saves a dead-PMT-"
                         "masked nu-slice flash_chi2 per event in the --out npz "
                         "(analysis-level flash-fix preview)")
    ap.add_argument("--dead-channels", default="15")
    ap.add_argument("--data", action="store_true",
                    help="real-data mode: unit weights (no xsecWeight/POT "
                         "scaling), no truth categories (all events tagged "
                         "'data'), truth-dependent plots skipped")
    args = ap.parse_args()
    os.makedirs(args.plots, exist_ok=True)

    fin = uproot.open(args.ntuple)
    if args.data:
        scale = 1.0
        print(">>> DATA mode: unit weights, no truth tags")
    else:
        pot = fin["potTree"].arrays(library="np")
        pot_sum = (float(np.sum(pot["totGoodPOT"]))
                   or float(np.sum(pot["totPOT"])))
        scale = args.pot / pot_sum
        print(f">>> sample POT {pot_sum:.3e}, target {args.pot:.2e} "
              f"-> scale {scale:.4f}")

    t = fin["EventTree"]
    have = set(t.keys())
    a = t.arrays([b for b in ("showerAttConfident", "showerAttScore")
                  if b in have] + ["run", "subrun", "event", "xsecWeight",
                  "trueVtxInWCFV", "trueNuCCNC",
                  "trueSimPartPDG", "trueSimPartTID", "trueSimPartMID",
                  "trueSimPartProcess", "trueSimPartPixelSumQ",
                  "trueSimPartPx", "trueSimPartPy", "trueSimPartPz",
                  "truePrimPartPDG", "truePrimPartPx", "truePrimPartPy",
                  "truePrimPartPz",
                  "foundVertex", "primaryVtxStream", "vtxIsFiducial",
                  "vtxX", "vtxY", "vtxZ",
                  "showerLArFormerPID", "showerRecoE",
                  "showerStartPosX", "showerStartPosY", "showerStartPosZ",
                  "showerStartDirX", "showerStartDirY", "showerStartDirZ",
                  "showerTrueTID", "showerTruePID",
                  "trackLArFormerPID", "trackIsSecondary", "trackRecoE"])
    n = len(a["run"])
    if args.data:
        w = np.ones(n, np.float64)
        cat = np.zeros(n, np.int64)      # single tag; truth plots skipped
    else:
        w0 = np.asarray(a["xsecWeight"], np.float64)
        w = np.where(w0 > 0, w0, 0.0) * scale
        cat = np.array([truth_category(a, i) for i in range(n)], np.int64)
        print(f">>> {n} events | true signal CC {int((cat==0).sum())} "
              f"NC {int((cat==1).sum())}")

    # ---- reco selection ---------------------------------------------------
    vtx_ok = ((np.asarray(a["foundVertex"]) == 1)
              & (np.asarray(a["primaryVtxStream"]) == 0)
              & (np.asarray(a["vtxIsFiducial"]) == 1))
    is_g = (a["showerLArFormerPID"] == 22) & (a["showerRecoE"] > RECO_G_MIN)
    n_g = ak.to_numpy(ak.sum(is_g, axis=1))
    is_mu = ((a["trackLArFormerPID"] == 13) & (a["trackIsSecondary"] == 0)
             & (a["trackRecoE"] > MU_KE_MIN))
    reco_cc = ak.to_numpy(ak.any(is_mu, axis=1))

    # per-event two most-energetic photons -> E, dirs (both variants)
    E1 = np.full(n, np.nan); E2 = np.full(n, np.nan)
    cosA = np.full(n, np.nan)     # vertex->start direction
    cosB = np.full(n, np.nan)     # trunk StartDir
    p_reco = np.full(n, np.nan)   # |p(g1)+p(g2)| with vertex->start dirs
    dres = {"vtx2start": [], "trunkdir": []}   # direction resolution [deg]
    for i in np.nonzero(vtx_ok & (n_g >= 2))[0]:
        gi = np.nonzero(ak.to_numpy(is_g[i]))[0]
        E = ak.to_numpy(a["showerRecoE"][i])[gi]
        o = gi[np.argsort(E)[::-1][:2]]
        E1[i], E2[i] = (ak.to_numpy(a["showerRecoE"][i])[o]).tolist()
        v = np.array([a["vtxX"][i], a["vtxY"][i], a["vtxZ"][i]], np.float64)
        sp = np.stack([ak.to_numpy(a[f"showerStartPos{c}"][i])[o]
                       for c in "XYZ"], 1).astype(np.float64)
        sd = np.stack([ak.to_numpy(a[f"showerStartDir{c}"][i])[o]
                       for c in "XYZ"], 1).astype(np.float64)
        dA = sp - v
        nA = np.linalg.norm(dA, axis=1)
        if np.all(nA > 1e-3):
            dA = dA / nA[:, None]
            cosA[i] = float(dA[0] @ dA[1])
            p_reco[i] = float(np.linalg.norm(E1[i] * dA[0] + E2[i] * dA[1]))
        nB = np.linalg.norm(sd, axis=1)
        if np.all(nB > 1e-3):
            sd = sd / nB[:, None]
            cosB[i] = float(sd[0] @ sd[1])
        # direction resolution vs true photon momentum (truth-matched only)
        ttid = ak.to_numpy(a["showerTrueTID"][i])[o]
        tpid = ak.to_numpy(a["showerTruePID"][i])[o]
        tids = ak.to_numpy(a["trueSimPartTID"][i])
        P = np.stack([ak.to_numpy(a[f"trueSimPart{c}"][i])
                      for c in ("Px", "Py", "Pz")], 1).astype(np.float64)
        for k in range(2):
            if abs(tpid[k]) != 22:
                continue
            j = np.nonzero(tids == ttid[k])[0]
            if not len(j):
                continue
            td = P[j[0]]
            tn = np.linalg.norm(td)
            if tn < 1e-6:
                continue
            td = td / tn
            if np.all(nA > 1e-3):
                dres["vtx2start"].append(np.degrees(
                    np.arccos(np.clip(dA[k] @ td, -1, 1))))
            if np.all(nB > 1e-3):
                dres["trunkdir"].append(np.degrees(
                    np.arccos(np.clip(sd[k] @ td, -1, 1))))

    # true pi0 momentum [MeV/c] for events with exactly one primary pi0
    # (truePrimPart 4-vectors are in GeV; invariant mass validates at 0.1350)
    p_true = np.full(n, np.nan)
    if not args.data:
        for i in range(n):
            pd = np.asarray(a["truePrimPartPDG"][i])
            k = np.nonzero(pd == 111)[0]
            if len(k) == 1:
                p_true[i] = 1000.0 * float(np.sqrt(
                    a["truePrimPartPx"][i][k[0]] ** 2
                    + a["truePrimPartPy"][i][k[0]] ** 2
                    + a["truePrimPartPz"][i][k[0]] ** 2))

    def mgg(cth):
        return np.sqrt(np.clip(2.0 * E1 * E2 * (1.0 - cth), 0, None))

    mA, mB = mgg(cosA), mgg(cosB)
    sel2p = vtx_ok & (n_g >= 2) & np.isfinite(mA)
    sel2x = sel2p & (n_g == 2)

    # analysis-level flash-fix: dead-PMT-masked nu-slice flash chi2 (per event,
    # matched to cascade by run/subrun/event), for a provisional flash-chi2 cut
    flash_chi2 = np.full(n, np.nan)
    if args.cascade_dir:
        import os as _os
        from flash_correction import corrected_chi2_by_rse
        dead = tuple(int(x) for x in args.dead_channels.split(",") if x != "")
        run_ = np.asarray(a["run"]); sub_ = np.asarray(a["subrun"])
        evt_ = np.asarray(a["event"])
        ent = np.nonzero(vtx_ok & (n_g >= 2))[0]
        cache = _os.path.join(
            _os.path.dirname(_os.path.abspath(__file__)),
            "rse_" + _os.path.basename(_os.path.dirname(
                args.cascade_dir.rstrip("/"))) + "_"
            + _os.path.basename(args.cascade_dir.rstrip("/")) + ".npz")
        cc = corrected_chi2_by_rse(args.cascade_dir, run_, sub_, evt_, ent,
                                   dead, cache)
        for i in ent:
            flash_chi2[i] = cc.get(int(i), np.nan)
        print(f">>> flash-fix: corrected chi2 for "
              f"{int(np.isfinite(flash_chi2[sel2p]).sum())}/{int(sel2p.sum())} "
              f"selected events")

    # ---- cutflow + selection summary --------------------------------------
    print("\n== CUTFLOW (raw | POT-weighted) ==")
    for lab, m in (("all events", np.ones(n, bool)),
                   ("reco vtx (nu-stream, in FV)", vtx_ok),
                   (">=2 reco photons >20 MeV", vtx_ok & (n_g >= 2)),
                   ("default selection (mass ok)", sel2p),
                   ("exactly-2 variant", sel2x)):
        print(f"  {lab:30s}: {int(m.sum()):6d} | {w[m].sum():9.1f}")
    if args.data:
        for lab, m in (("reco-CC", sel2p & reco_cc),
                       ("reco-NC", sel2p & ~reco_cc)):
            print(f"  {lab}: N={int(m.sum())}")
    if not args.data:
        print("\n== selected composition (default, >=2) ==")
    for lab, m in ([] if args.data else
                   [("reco-CC", sel2p & reco_cc),
                    ("reco-NC", sel2p & ~reco_cc)]):
        tot = w[m].sum()
        parts = " | ".join(f"{CATS[c]} {w[m & (cat==c)].sum()/max(tot,1e-9):.2f}"
                           for c in range(6) if (m & (cat == c)).any())
        print(f"  {lab} (N={int(m.sum())}, w={tot:.1f}): {parts}")
    for cc, lab in ([] if args.data else ((0, "CC"), (1, "NC"))):
        den = (cat == cc)
        num = den & sel2p & (reco_cc if cc == 0 else ~reco_cc)
        print(f"  signal {lab} efficiency (sel+right CC tag): "
              f"{num.sum()/max(den.sum(),1):.3f} "
              f"(any-sel {(den & sel2p).sum()/max(den.sum(),1):.3f})")

    if args.out:
        np.savez(args.out, cat=cat, sel_ge2=sel2p, sel_eq2=sel2x,
                 reco_cc=reco_cc, m_vtx2start=mA, m_trunkdir=mB,
                 n_gamma=n_g, w=w, E1=E1, E2=E2,
                 p_reco=p_reco, p_true=p_true, flash_chi2=flash_chi2)

    # ---- plots -------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bins = np.linspace(0, 500, 51)

    ncat = 1 if args.data else 6
    cat_names = ["beam data"] if args.data else CATS
    cat_colors = ["#1f77b4"] if args.data else CAT_COLORS

    def stacked(mass, sel, tag, fname):
        for ccm, cclab in ((reco_cc, "recoCC"), (~reco_cc, "recoNC")):
            m = sel & ccm & np.isfinite(mass)
            fig, ax = plt.subplots(figsize=(6.4, 4.4))
            data = [np.clip(mass[m & (cat == c)], 0, 499) for c in range(ncat)]
            ws = [w[m & (cat == c)] for c in range(ncat)]
            ax.hist(data, bins=bins, weights=ws, stacked=True,
                    color=cat_colors,
                    label=[f"{cat_names[c]} ({ws[c].sum():.0f})"
                           for c in range(ncat)])
            ax.axvline(PI0_MASS, color="k", ls=":", lw=1.2,
                       label=f"pi0 mass {PI0_MASS:.0f}")
            sig = m & (cat <= 1) if not args.data else m
            if sig.sum() > 10:
                med = np.median(mass[sig])
                ax.axvline(med, color="r", ls="--", lw=1,
                           label=f"{'data' if args.data else 'signal'} "
                                 f"median {med:.0f}")
            ax.set(xlabel=r"$m_{\gamma\gamma}$ [MeV]",
                   ylabel=("events" if args.data
                           else f"events / {args.pot:.1e} POT"),
                   title=f"{cclab}: two-photon invariant mass ({tag})")
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(f"{args.plots}/{fname}_{cclab}.png", dpi=110)
            plt.close(fig)

    stacked(mA, sel2p, "vertex->start dir, >=2 gamma", "mgg_vtx2start_ge2")
    stacked(mA, sel2x, "vertex->start dir, exactly 2", "mgg_vtx2start_eq2")
    stacked(mB, sel2p, "trunk dir, >=2 gamma", "mgg_trunkdir_ge2")

    # reconstructed pi0 momentum |p(g1)+p(g2)| (vertex->start dirs), stacked
    # by truth category per reco-CC/NC (works in data mode too)
    pbins = np.linspace(0, 1200, 49)
    for ccm, cclab in ((reco_cc, "recoCC"), (~reco_cc, "recoNC")):
        m = sel2p & ccm & np.isfinite(p_reco)
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        data = [np.clip(p_reco[m & (cat == c)], 0, 1199) for c in range(ncat)]
        ws = [w[m & (cat == c)] for c in range(ncat)]
        ax.hist(data, bins=pbins, weights=ws, stacked=True, color=cat_colors,
                label=[f"{cat_names[c]} ({ws[c].sum():.0f})"
                       for c in range(ncat)])
        ax.set(xlabel=r"reco $p_{\pi^0}$ [MeV/c]",
               ylabel=("events" if args.data
                       else f"events / {args.pot:.1e} POT"),
               title=f"{cclab}: reconstructed pi0 momentum (>=2 gamma)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{args.plots}/ppi0_reco_{cclab}.png", dpi=110)
        plt.close(fig)

    if args.data:
        print(f">>> plots -> {args.plots}")
        return

    # 2D reco vs true pi0 momentum (selected true-signal events)
    m2 = sel2p & (cat <= 1) & np.isfinite(p_reco) & np.isfinite(p_true)
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    h = ax.hist2d(np.clip(p_true[m2], 0, 1199), np.clip(p_reco[m2], 0, 1199),
                  bins=[np.linspace(0, 1200, 40)] * 2, cmin=1, cmap="viridis")
    ax.plot([0, 1200], [0, 1200], "r--", lw=1, label="reco = true")
    r = p_reco[m2] / p_true[m2]
    ax.set(xlabel=r"true $p_{\pi^0}$ [MeV/c]",
           ylabel=r"reco $p_{\pi^0}$ [MeV/c]",
           title=f"pi0 momentum: reco vs true (selected signal, N={int(m2.sum())})\n"
                 f"median reco/true = {np.median(r):.3f}, "
                 f"16-84%: [{np.percentile(r,16):.2f}, {np.percentile(r,84):.2f}]")
    fig.colorbar(h[3], ax=ax, label="events")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{args.plots}/ppi0_reco_vs_true.png", dpi=110)
    plt.close(fig)

    # selection efficiency vs true pi0 momentum (denominator = true signal)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    pedges = np.array([0, 100, 200, 300, 400, 500, 650, 800, 1000, 1200])
    pctr = 0.5 * (pedges[:-1] + pedges[1:])
    den_all = (cat <= 1) & np.isfinite(p_true)
    for num, lab, sty in (
            (sel2p, "selected (>=2 gamma)", "o-"),
            (sel2p & ((cat == 0) == reco_cc), "selected + correct CC/NC tag",
             "s--"),
            (sel2x, "selected (exactly 2)", "^:")):
        eff, err = [], []
        for lo, hi in zip(pedges[:-1], pedges[1:]):
            b = den_all & (p_true >= lo) & (p_true < hi)
            N = int(b.sum())
            e = num[b].mean() if N else np.nan
            eff.append(e)
            err.append(np.sqrt(e * (1 - e) / N) if N else 0)
        ax.errorbar(pctr, eff, yerr=err, fmt=sty, ms=4, label=lab)
    ax.set(xlabel=r"true $p_{\pi^0}$ [MeV/c]", ylabel="efficiency",
           ylim=(0, 1.05),
           title="pi0 selection efficiency vs true momentum\n"
                 "(denominator: true signal, both photons E_vis>20 MeV)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{args.plots}/ppi0_efficiency.png", dpi=110)
    plt.close(fig)

    # direction-estimator comparison on signal
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    sig = sel2p & (cat <= 1)
    for mass, lab in ((mA[sig], "vertex->start"), (mB[sig], "trunk StartDir")):
        ax.hist(np.clip(mass, 0, 499), bins=bins, weights=w[sig],
                histtype="step", lw=1.8, label=f"{lab} "
                f"(median {np.median(mass):.0f} MeV)")
    ax.axvline(PI0_MASS, color="k", ls=":", lw=1.2)
    ax.set(xlabel=r"$m_{\gamma\gamma}$ [MeV]", ylabel="weighted events",
           title="signal m_gg: direction estimator comparison")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{args.plots}/mgg_direction_comparison.png", dpi=110)
    plt.close(fig)

    # direction resolution
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for k, lab in (("vtx2start", "vertex->start"), ("trunkdir", "trunk dir")):
        r = np.asarray(dres[k])
        ax.hist(np.clip(r, 0, 60), bins=np.linspace(0, 60, 61),
                histtype="step", lw=1.8, density=True,
                label=f"{lab} (median {np.median(r):.1f} deg, N={len(r)})")
    ax.set(xlabel="angle(reco dir, true photon dir) [deg]", ylabel="density",
           title="photon direction resolution (truth-matched sel. photons)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{args.plots}/direction_resolution.png", dpi=110)
    plt.close(fig)

    # photon-count fidelity: reco N_gamma vs true N detectable decay photons
    n_true_g = np.zeros(n, np.int64)
    for i in range(n):
        pdg = np.asarray(a["trueSimPartPDG"][i])
        proc = np.asarray(a["trueSimPartProcess"][i])
        q = np.asarray(a["trueSimPartPixelSumQ"][i])
        n_true_g[i] = int(np.sum((np.abs(pdg) == 22)
                                 & (A_GAMMA * np.clip(q, 0, None) > EVIS_MIN)
                                 & (proc <= 1)))
    m = vtx_ok & (np.asarray(a["trueVtxInWCFV"]) == 1)
    K = 4
    M = np.zeros((K, K))
    for i in np.nonzero(m)[0]:
        M[min(int(n_true_g[i]), K - 1), min(int(n_g[i]), K - 1)] += w[i]
    M = M / np.maximum(M.sum(1, keepdims=True), 1e-9)
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    im = ax.imshow(M, origin="lower", cmap="Blues", vmin=0, vmax=1)
    for r in range(K):
        for c in range(K):
            ax.text(c, r, f"{M[r, c]:.2f}", ha="center", va="center",
                    fontsize=9, color="k")
    lab = ["0", "1", "2", ">=3"]
    ax.set(xticks=range(K), xticklabels=lab, yticks=range(K),
           yticklabels=lab, xlabel="N reco photons (>20 MeV)",
           ylabel="N true detectable photons",
           title="photon multiplicity: row-normalized (in-FV, reco vtx)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(f"{args.plots}/photon_multiplicity.png", dpi=110)
    plt.close(fig)
    print(f">>> plots -> {args.plots}")


if __name__ == "__main__":
    main()
