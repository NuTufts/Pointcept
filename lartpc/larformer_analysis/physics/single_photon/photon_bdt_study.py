"""Per-shower cosmic-BDT (showerCosmicScore) performance for the single-photon
selection, as a function of the number of TRUE detectable photons in the event.

Question (2026-09-05): the BDT was trained on single reco photons vs cosmics;
does it also keep photons in multi-photon (pi0) events, and how much EXT does it
reject? Per reco photon shower (showerLArFormerPID==22, recal'd E > 20 MeV) in
events with a nu-stream FV vertex, on the MC overlay the shower truth class is
  nu photon   : showerTruePID==22 & showerTruePurity>=0.5 & unlabeled<0.5
  nu electron : |showerTruePID|==11 & purity>=0.5
  nu other    : any other labeled particle
  cosmic (MC) : showerTruePID<=0 or showerTrueUnlabeledPurity>=0.5 (overlay cosmic)
and the event context is n_vis_g = number of true detectable photons
(E_vis >= 20 MeV, single_photon_selection.truth_tables) and the inclusive-signal
flag. EXT-BNB photon showers (rows >= --ext-row-min, x --ext-scale) are cosmic
by construction.

Outputs (--plots): score distributions by class; per-shower pass fraction vs
threshold split by n_vis_g (1 signal / 1 all / 2 / >=3) with EXT + MC-cosmic
rejection; per-event curves (signal: exactly-one-passing-and-it-is-the-true-
photon; EXT: >=1 passing photon); ROC; pass fraction vs reco E and vs
conversion distance at a few thresholds; a working-point table (stdout +
bdt_table.txt).

    python3 photon_bdt_study.py --mc-ntuple ... --ext-ntuple ... --plots ...
"""
import argparse
import os
import sys

import numpy as np
import uproot
import awkward as ak

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from single_photon_selection import (truth_tables, recal_E, BR_MC,  # noqa: E402
                                     RECO_G_MIN, binned_ratio)

BR = ["run", "subrun", "event", "foundVertex", "primaryVtxStream",
      "vtxIsFiducial", "showerLArFormerPID", "showerRecoE",
      "showerCosmicScore", "showerDistToVtx", "showerTruePID",
      "showerTruePurity", "showerTrueUnlabeledPurity", "showerTrueTID"]
CLASSES = ["nu photon", "nu electron", "nu other", "cosmic (overlay)"]
CLS_COL = ["#d62728", "#2ca02c", "#9467bd", "#7f7f7f"]


def per_shower(a, args, is_mc):
    """Flatten photon showers of S1 events into per-shower numpy arrays."""
    E = recal_E(a, args.recal_gamma_a, args.recal_gamma_b)
    vtx_ok = ((a["foundVertex"] == 1) & (a["primaryVtxStream"] == 0)
              & (a["vtxIsFiducial"] == 1))
    is_g = (a["showerLArFormerPID"] == 22) & (E > RECO_G_MIN) & vtx_ok
    n = len(a["run"])
    # event index per shower
    evi = ak.flatten(ak.broadcast_arrays(np.arange(n), is_g)[0][is_g])
    out = dict(evt=ak.to_numpy(evi).astype(np.int64),
               E=ak.to_numpy(ak.flatten(E[is_g])).astype(np.float64),
               score=ak.to_numpy(ak.flatten(a["showerCosmicScore"][is_g])).astype(np.float64),
               dist=ak.to_numpy(ak.flatten(a["showerDistToVtx"][is_g])).astype(np.float64),
               n_g_event=ak.to_numpy(ak.sum(is_g, axis=1)).astype(np.int64),
               vtx_ok=ak.to_numpy(vtx_ok))
    if is_mc:
        tp = ak.to_numpy(ak.flatten(a["showerTruePID"][is_g]))
        pur = ak.to_numpy(ak.flatten(a["showerTruePurity"][is_g]))
        unl = ak.to_numpy(ak.flatten(a["showerTrueUnlabeledPurity"][is_g]))
        cls = np.full(len(tp), 2, np.int64)
        cls[(tp == 22) & (pur >= 0.5) & (unl < 0.5)] = 0
        cls[(np.abs(tp) == 11) & (pur >= 0.5) & (unl < 0.5)] = 1
        cls[(tp <= 0) | (unl >= 0.5)] = 3
        out["cls"] = cls
        out["tid"] = ak.to_numpy(ak.flatten(a["showerTrueTID"][is_g])).astype(np.int64)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mc-ntuple", required=True)
    ap.add_argument("--ext-ntuple", required=True)
    ap.add_argument("--ext-scale", type=float, default=1.1818)
    ap.add_argument("--ext-row-min", type=int, default=100000)
    ap.add_argument("--pot", type=float, default=4.4e19)
    ap.add_argument("--recal-gamma-a", type=float, default=0.01553)
    ap.add_argument("--recal-gamma-b", type=float, default=-12.80)
    ap.add_argument("--plots", required=True)
    args = ap.parse_args()
    os.makedirs(args.plots, exist_ok=True)

    print(">>> loading MC ...", flush=True)
    fin = uproot.open(args.mc_ntuple)
    p = fin["potTree"].arrays(library="np")
    pot = float(np.sum(p["totGoodPOT"])) or float(np.sum(p["totPOT"]))
    scale = args.pot / pot
    a = fin["EventTree"].arrays(BR + BR_MC)
    T = truth_tables(a)
    w0 = ak.to_numpy(a["xsecWeight"]).astype(np.float64)
    wev = np.where(w0 > 0, w0, 0.0) * scale
    M = per_shower(a, args, True)
    del a
    M["w"] = wev[M["evt"]]
    nvg = T["n_vis_g"][M["evt"]]
    sig = T["sig_incl"][M["evt"]]
    is_sig_photon = sig & (M["cls"] == 0) & (M["tid"] == T["g_tid"][M["evt"]])
    print(f">>> MC photon showers in S1 events: {len(M['E'])}; class census "
          + ", ".join(f"{CLASSES[c]} {int((M['cls']==c).sum())}" for c in range(4)))

    print(">>> loading EXT ...", flush=True)
    ax = uproot.open(args.ext_ntuple)["EventTree"].arrays(BR)
    X = per_shower(ax, args, False)
    del ax
    nx = len(X["vtx_ok"])
    wx_ev = np.zeros(nx); wx_ev[args.ext_row_min:] = args.ext_scale
    X["w"] = wx_ev[X["evt"]]
    keepx = X["evt"] >= args.ext_row_min
    print(f">>> EXT photon showers (analysis rows, S1 events): {int(keepx.sum())}")

    # ------------------------------------------------------------------ groups
    groups = [("n_vis_g=1, signal photon", is_sig_photon, "#d62728", "o-"),
              ("n_vis_g=1, all nu photons", (M["cls"] == 0) & (nvg == 1), "#ff9896", "s--"),
              ("n_vis_g=2 nu photons", (M["cls"] == 0) & (nvg == 2), "#9467bd", "^-"),
              ("n_vis_g>=3 nu photons", (M["cls"] == 0) & (nvg >= 3), "#8c564b", "d-")]
    thrs = np.linspace(0, 0.98, 50)

    def passfrac(sel_w, sc, thrs):
        tot = sel_w.sum()
        return np.array([sel_w[sc >= t].sum() / max(tot, 1e-9) for t in thrs])

    ext_pass = passfrac(X["w"][keepx], X["score"][keepx], thrs)
    mccos = M["cls"] == 3
    mccos_pass = passfrac(M["w"][mccos], M["score"][mccos], thrs)
    # per-event curves: signal events with S1 -> exactly one passing photon AND
    # it is the true one; EXT S1 events -> at least one passing photon
    vtx_ok_ev = M["vtx_ok"]
    den_sig = wev[T["sig_incl"]].sum()
    ev_curve = []
    ext_ev_curve = []
    ext_den = wx_ev[X["vtx_ok"] & (np.arange(nx) >= args.ext_row_min)].sum()
    for t in thrs:
        pas = M["score"] >= t
        npass = np.bincount(M["evt"][pas], minlength=len(wev))
        ntrue = np.bincount(M["evt"][pas & is_sig_photon], minlength=len(wev))
        ok = T["sig_incl"] & vtx_ok_ev & (npass == 1) & (ntrue == 1)
        ev_curve.append(wev[ok].sum() / max(den_sig, 1e-9))
        xpass = np.bincount(X["evt"][(X["score"] >= t)], minlength=nx)
        ext_ev_curve.append(wx_ev[(xpass >= 1)].sum() / max(ext_den, 1e-9))
    ev_curve, ext_ev_curve = np.array(ev_curve), np.array(ext_ev_curve)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1. score distributions
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    bins = np.linspace(0, 1, 41)
    for c in range(4):
        m = M["cls"] == c
        ax.hist(np.clip(M["score"][m], 0, 1 - 1e-6), bins=bins, weights=M["w"][m],
                histtype="step", lw=1.8, color=CLS_COL[c],
                label=f"MC {CLASSES[c]} ({M['w'][m].sum():.0f})")
    ax.hist(np.clip(X["score"][keepx], 0, 1 - 1e-6), bins=bins, weights=X["w"][keepx],
            histtype="stepfilled", alpha=0.3, color="k",
            label=f"EXT photons ({X['w'][keepx].sum():.0f})")
    m = is_sig_photon
    ax.hist(np.clip(M["score"][m], 0, 1 - 1e-6), bins=bins, weights=M["w"][m],
            histtype="step", lw=2.2, ls="--", color="#d62728",
            label=f"MC signal photon ({M['w'][m].sum():.0f})")
    ax.set(xlabel="showerCosmicScore", ylabel=f"showers / {args.pot:.1e} POT",
           yscale="log", title="per-shower cosmic-BDT score (photon showers, S1 events)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{args.plots}/bdt_score_distributions.png", dpi=110)
    plt.close(fig)

    # 2. pass fraction vs threshold, by n_vis_g; EXT / MC-cosmic rejection
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axs[0]
    for lab, m, col, sty in groups:
        ax.plot(thrs, passfrac(M["w"][m], M["score"][m], thrs), sty, ms=3, color=col,
                label=f"{lab} (N={int(m.sum())})")
    ax.plot(thrs, 1 - ext_pass, "k-", lw=2, label="EXT photon rejection")
    ax.plot(thrs, 1 - mccos_pass, "k:", lw=1.5, label="MC overlay-cosmic rejection")
    ax.axvline(0.192, color="gray", ls="--", lw=1, label="0.192 (pi0 WP)")
    ax.set(xlabel="score threshold", ylabel="per-shower pass fraction / rejection",
           ylim=(0, 1.02), title="per-shower BDT efficiency by true photon multiplicity")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    ax = axs[1]
    ax.plot(thrs, ev_curve, "o-", ms=3, color="#d62728",
            label="signal events: exactly 1 passing photon = the true photon | S1")
    ax.plot(thrs, ext_ev_curve, "k-", lw=2, label="EXT events: >=1 passing photon | S1")
    ax.axvline(0.192, color="gray", ls="--", lw=1)
    ax.set(xlabel="score threshold", ylabel="per-event fraction", ylim=(0, 1.02),
           title="event-level effect of the BDT (S1 = nu-stream FV vertex)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{args.plots}/bdt_eff_vs_threshold.png", dpi=110)
    plt.close(fig)

    # 3. ROC
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    for lab, m, col, sty in groups:
        ax.plot(1 - ext_pass, passfrac(M["w"][m], M["score"][m], thrs), sty[:-1] + "-",
                ms=3, color=col, label=lab)
    for t in (0.1, 0.192, 0.5, 0.8):
        k = int(np.argmin(np.abs(thrs - t)))
        ax.annotate(f"{t}", (1 - ext_pass[k], passfrac(M["w"][groups[0][1]],
                    M["score"][groups[0][1]], thrs)[k]), fontsize=7)
    ax.set(xlabel="EXT photon-shower rejection", ylabel="nu photon pass fraction",
           xlim=(0, 1.02), ylim=(0, 1.02), title="ROC (per shower)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{args.plots}/bdt_roc.png", dpi=110)
    plt.close(fig)

    # 4. pass fraction vs reco E and vs conversion distance at fixed thresholds
    eedges = np.array([20, 40, 60, 80, 100, 150, 200, 300, 400, 600, 1000])
    dedges = np.array([0, 2, 5, 10, 20, 30, 50, 80, 120, 200])
    for xkey, edges, xlab, stem in (("E", eedges, "reco photon energy [MeV]", "bdt_eff_vs_recoE"),
                                    ("dist", dedges, "shower distance to vertex [cm]",
                                     "bdt_eff_vs_dist")):
        ctr = 0.5 * (edges[:-1] + edges[1:])
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
        for ax, t in zip(axs, (0.192, 0.5, 0.8)):
            for lab, m, col, sty in groups:
                e, er = binned_ratio(M["w"] * (m & (M["score"] >= t)), M["w"] * m,
                                     M[xkey], edges, n_raw=m)
                ax.errorbar(ctr, e, yerr=er, fmt=sty, ms=3, color=col, label=lab)
            ex, exr = binned_ratio(X["w"] * (keepx & (X["score"] >= t)), X["w"] * keepx,
                                   X[xkey], edges, n_raw=keepx)
            ax.errorbar(ctr, ex, yerr=exr, fmt="k-", lw=2, label="EXT photon pass frac")
            ax.set(xlabel=xlab, ylim=(0, 1.02), title=f"score >= {t}",
                   xscale="log" if xkey == "E" else "linear")
            ax.grid(alpha=0.3, which="both")
        axs[0].set_ylabel("pass fraction")
        axs[0].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(f"{args.plots}/{stem}.png", dpi=110)
        plt.close(fig)

    # 5. table
    lines = [f"{'thr':>6}{'sig-photon':>11}{'nvg1 all':>10}{'nvg2':>8}{'nvg>=3':>8}"
             f"{'nu e':>8}{'EXT rej':>9}{'MCcos rej':>10}{'sig evt S2':>11}{'EXT evt>=1':>11}"]
    for t in (0.0, 0.05, 0.1, 0.192, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        k = int(np.argmin(np.abs(thrs - t)))
        vals = [passfrac(M["w"][m], M["score"][m], [thrs[k]])[0] for _, m, _, _ in groups]
        ne = passfrac(M["w"][M["cls"] == 1], M["score"][M["cls"] == 1], [thrs[k]])[0]
        lines.append(f"{thrs[k]:6.3f}{vals[0]:11.3f}{vals[1]:10.3f}{vals[2]:8.3f}"
                     f"{vals[3]:8.3f}{ne:8.3f}{1-ext_pass[k]:9.3f}{1-mccos_pass[k]:10.3f}"
                     f"{ev_curve[k]:11.3f}{ext_ev_curve[k]:11.3f}")
    txt = "\n".join(lines)
    print("\nper-shower pass fractions (weighted) by group; per-event columns as in "
          "bdt_eff_vs_threshold.png\n" + txt)
    with open(os.path.join(args.plots, "bdt_table.txt"), "w") as fh:
        fh.write(txt + "\n")
    print(f">>> plots -> {args.plots}")


if __name__ == "__main__":
    main()
