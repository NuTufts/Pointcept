#!/usr/bin/env python3
"""Variable-insight study for the MiniBooNE-targeted selection: which
KINEMATICS-FREE variables separate mb-truth signal from (a) neutrino
background and (b) EXT cosmics in the primary cell? Trains two gradient-boosted
classifiers (odd/even event folds, xsec/EXT weights), reports test AUC,
permutation importances and single-variable AUCs, and plots the score
distributions. For INSIGHT, not for cutting (MB_SELECTION_PLAN.md).

    python3 mb_variable_study.py --mc-ntuple ... --ext-ntuple ... --novtx --plots plots_run1_mbsel_vars
"""
import argparse
import os
import sys

import numpy as np
import awkward as ak

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import single_photon_selection as SPS            # noqa: E402
import single_photon_mb_selection as MBS         # noqa: E402
from single_photon_diagnostics import add_args   # noqa: E402

# kinematics-free feature list (name -> description); candidate E / direction are excluded
FEATURES = [
    ("n_ring", "n ring-making mu/pi/p tracks"), ("n_bad_mu", "n mu-PID tracks >= 40 MeV"),
    ("n_bad_pi", "n pi-PID tracks >= 50 MeV"), ("n_bad_p", "n proton tracks >= 342 MeV"),
    ("n_p", "n accepted protons >= 35 MeV"), ("p_kemax", "max proton KE"), ("mupi_kemax", "max mu/pi KE"),
    ("extra_E", "other-prong energy sum"), ("extra_E_showers", "other-shower energy sum"),
    ("n_soft", "n soft other showers"), ("soft_min_angle", "min angle soft cluster-candidate"),
    ("n_cone", "n soft clusters in pi0 cone"), ("cone_e2_min", "min pi0-cone E2"),
    ("vtx_dwall", "reco vertex dwall"), ("start_dwall", "candidate start dwall"),
    ("dfw", "backward wall distance"), ("conv_dist", "candidate start to vertex"),
    ("bdt_cosmic", "candidate cosmic BDT"), ("pe_ratio", "flash PE obs/pred"), ("logchi2", "log10 flash chi2"),
    ("track_min_dwall", "min track start/end dwall"), ("n_tracks", "n nu-stream tracks"),
    ("n_showers", "n nu-stream showers"), ("n_vtxless", "n vertex-less nu-stream prongs"),
    ("slice_nparticles", "nu-slice n particles"), ("cand_objectness", "candidate objectness"),
    ("cand_phscore", "candidate photon score"), ("cand_chargefrac", "candidate charge fraction"),
    ("cand_disttovtx", "candidate dist to vertex"), ("cand_secondary", "candidate is secondary"),
    # path-split quantities (vertex / no-vertex, 2nd-photon veto, containment, in-TPC / entering)
    ("path", "candidate path (0 vertex, 1 vertex-less)"), ("vtx_ok", "primary in-FV reco vertex found"),
    ("other_gE_max", "max E of another photon prong (2nd-photon tag)"), ("veto2g_fail", "fails 2nd-photon veto"),
    ("track_min_dwall", "min track start/end dwall (containment)"), ("pc", "partially contained"),
    ("dtw", "forward wall distance"), ("cand_score", "unified selection score"), ("cand_novtx", "vertex-free BDT score"),
    # muon-aware containment / PID (2026-10-05)
    ("mu_kemax", "max mu-PID track KE"), ("pi_kemax", "max pi-PID track KE"),
    ("mu_min_dwall", "min wall distance of mu-PID tracks (start/end)"), ("pi_min_dwall", "min wall distance of pi-PID tracks"),
    ("mu_exiting", "a mu-PID track ends/starts < contain-dwall of a wall"), ("mu_score_max", "max LArPID muon score (classified tracks)"),
    ("n_mu_tracks", "n mu-PID tracks"),
]
# background classes targeted one at a time (WC-style), by truth category
TARGETS = [("numu", "signal vs CC-numu-like (visible mu / ring-making + entering CC)", [5, 7]),
           ("pi0", "signal vs >=2-photon (in-FV + entering)", [4, 9]),
           ("other", "signal vs other nu (entering other, CC nue, no visible photon)", [6, 8, 10]),
           ("allnu", "signal vs ALL nu background", list(range(4, 11)))]


def features_from(a, R, names, S, E, path, args):
    """MB feature dict from already-built reco tables / steps (shared with the diagnostics)."""
    M = MBS.mb_tables(a, R, args, names, S, E, path)
    idx = R["_cand_idx"]; has = np.isfinite(E)
    nu_t = a["trackStream"] == 0; nu_s = a["showerStream"] == 0
    cnt = lambda m: np.asarray(ak.to_numpy(ak.sum(m, axis=1)), np.float64)
    M["track_min_dwall"] = R["track_min_dwall"]
    M["n_tracks"] = cnt(nu_t); M["n_showers"] = cnt(nu_s)
    M["n_vtxless"] = cnt(nu_t & (a["trackVtxIdx"] < 0)) + cnt(nu_s & (a["showerVtxIdx"] < 0))
    M["slice_nparticles"] = np.asarray(ak.to_numpy(a["nuSliceNParticles"]), np.float64)
    M["cand_objectness"] = MBS.pick(a["showerObjectness"], idx, has)
    M["cand_phscore"] = MBS.pick(a["showerLArFormerPhScore"], idx, has)
    M["cand_chargefrac"] = MBS.pick(a["showerChargeFrac"], idx, has)
    M["cand_disttovtx"] = MBS.pick(a["showerDistToVtx"], idx, has)
    M["cand_secondary"] = MBS.pick(a["showerIsSecondary"], idx, has)
    return M


def features(ntuple, is_mc, args):
    a, pot = SPS.load(ntuple, is_mc, args.novtx, extra=MBS.BR_EXTRA + ("showerIsSecondary",))
    R = SPS.reco_tables(a, args); R["novtx"] = args.novtx
    names, S, E, _, _, path = SPS.build_steps(R, args.shower_bdt_min, args.flashchi2_cut,
                                              args.bdt_after_multiplicity, args.flashpe_ratio_max)
    M = features_from(a, R, names, S, E, path, args)
    ev = np.asarray(ak.to_numpy(a["event"]), np.int64)
    return a, pot, M, ev


def auc_w(y, s, w):
    from sklearn.metrics import roc_auc_score
    ok = np.isfinite(s)
    if y[ok].min() == y[ok].max():
        return np.nan
    v = roc_auc_score(y[ok], s[ok], sample_weight=w[ok])
    return max(v, 1 - v)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(ap); ap.set_defaults(signal_def="mb")
    for k, v in (("veto2g_emin", 10.0), ("contain_dwall", 10.0), ("dfw_cut", SPS.DFW_CUT), ("mb_mu_kemin", 40.0),
                 ("mb_pi_kemin", 50.0), ("mb_p_kemax", 342.0), ("extra_emax", 10.0), ("cone_e2_min", 5.0),
                 ("cone_e2_max", 600.0)):
        ap.add_argument("--" + k.replace("_", "-"), type=float, default=v)
    for k, v in (("tight_dwall", 10.0), ("dfw_tight", 100.0), ("conv_dist_min", 0.0), ("cand_bdt_min_mb", 0.7),
                 ("mb_pe_ratio_max", 2.0), ("mb_logchi2_max", 2.25)):
        ap.add_argument("--" + k.replace("_", "-"), type=float, default=v)
    ap.add_argument("--pi0-cone", action="store_true")
    ap.add_argument("--stage", default="pre", choices=["S3", "pre", "vpass", "M0", "M1"],
                    help="S3 = pre-selection before the flash cuts (all paths); pre = base final step, all paths "
                         "(before any path split); M0 = primary cell; M1 = after the hadron vetoes")
    args = ap.parse_args()
    SPS.configure(args); os.makedirs(args.plots, exist_ok=True)
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.inspection import permutation_importance

    a, pot, M, ev = features(args.mc_ntuple, True, args)
    w0 = ak.to_numpy(a["xsecWeight"]).astype(np.float64); w = np.where(w0 > 0, w0, 0.0) * (args.pot / pot)
    T = SPS.truth_tables(a); sig = T["sig_incl"]; cat = T["cat"]; del a
    _, _, MX, evx = features(args.ext_ntuple, False, args)
    wx = np.zeros(len(evx)); wx[args.ext_row_min:] = args.ext_scale
    for d in (M, MX):
        d["cell"] = MBS.mb_conditions(d, args)[1][0]
    def stage_mask(d):
        vpass = d["base"] & (np.asarray(d["path"]) == 0) & (np.asarray(d["veto2g_fail"]) == 0)
        return {"S3": d["base_S3"], "pre": d["base"], "vpass": vpass, "M0": d["cell"],
                "M1": d["cell"] & (d["n_ring"] == 0)}[args.stage]
    cell, cellx = stage_mask(M), stage_mask(MX)
    for d in (M, MX):
        d["pc"] = d["pc"].astype(np.float64); d["veto2g_fail"] = d["veto2g_fail"].astype(np.float64)
    keys = [k for k, _ in FEATURES]
    fin_ = lambda v: np.where(np.isfinite(v), v, np.nan)      # +-inf -> NaN (HistGB handles NaN)
    X = np.stack([fin_(np.asarray(M[k], np.float64)) for k in keys], 1)[cell]
    Xx = np.stack([fin_(np.asarray(MX[k], np.float64)) for k in keys], 1)[cellx]
    y = sig[cell].astype(int); wc = w[cell]; evc = ev[cell]; catc = cat[cell]
    wxc = wx[cellx]; evxc = evx[cellx]
    L = [f"MB variable study, stage {args.stage}: signal {wc[y == 1].sum():.1f} ({int((y == 1).sum())} raw) | "
         f"nu bkg {wc[y == 0].sum():.1f} ({int((y == 0).sum())} raw) | EXT {wxc.sum():.1f} ({len(wxc)} raw)"]
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    L.append("  signal composition: " + " | ".join(f"{SPS.CATS[c][4:]} {wc[y == 1][catc[y == 1] == c].sum():.1f}" for c in range(SPS.N_SIG_CAT)))
    L.append("  nu background composition: " + " | ".join(f"{SPS.CATS[c][5:]} {wc[catc == c].sum():.1f}" for c in range(SPS.N_SIG_CAT, SPS.N_MC_CAT) if wc[catc == c].sum() > 0))

    def train(Xa, ya, wa, eva, lab, tag):
        wtr = wa.copy(); wtr[ya == 1] *= 0.5 / max(wtr[ya == 1].sum(), 1e-9); wtr[ya == 0] *= 0.5 / max(wtr[ya == 0].sum(), 1e-9)
        score = np.full(len(ya), np.nan); imps = []
        for fold in (0, 1):
            tr = (eva % 2) == fold; te = ~tr
            if ya[tr].min() == ya[tr].max() or ya[te].min() == ya[te].max():
                continue
            clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=3, l2_regularization=1.0,
                                                 min_samples_leaf=20, random_state=1)
            clf.fit(Xa[tr], ya[tr], sample_weight=wtr[tr])
            score[te] = clf.predict_proba(Xa[te])[:, 1]
            pi = permutation_importance(clf, Xa[te], ya[te], sample_weight=wtr[te], n_repeats=5, random_state=1, scoring="roc_auc")
            imps.append(pi.importances_mean)
        imp = np.mean(imps, 0) if imps else np.zeros(Xa.shape[1])
        auc = auc_w(ya, score, wa)
        ok = np.isfinite(score)
        fpr, tpr, _ = roc_curve(ya[ok], score[ok], sample_weight=wa[ok])
        rej = {e: 1 - np.interp(e, tpr, fpr) for e in (0.9, 0.8, 0.7, 0.5)}
        L.append(f"\n== {lab} [{tag}]: test AUC {auc:.3f}; background rejection at signal eff 0.9/0.8/0.7/0.5 = "
                 + " / ".join(f"{rej[e]:.2f}" for e in (0.9, 0.8, 0.7, 0.5)))
        L.append(f"{'variable':22s}{'perm. importance':>18}{'1-var AUC':>11}  description")
        for i in np.argsort(-imp)[:12]:
            L.append(f"{keys[i]:22s}{imp[i]:18.4f}{auc_w(ya, Xa[:, i], wa):11.3f}  {FEATURES[i][1]}")
        bins = np.linspace(0, 1, 26)
        fig, ax = plt.subplots(figsize=(7, 4.6))
        ax.hist(score[ya == 1], bins=bins, weights=wa[ya == 1], histtype="step", lw=2, color="#d62728", label=f"class 1 ({wa[ya == 1].sum():.1f})")
        ax.hist(score[ya == 0], bins=bins, weights=wa[ya == 0], histtype="step", lw=2, color="#1f77b4", label=f"class 0 ({wa[ya == 0].sum():.1f})")
        ax.set(xlabel=f"diagnostic BDT score ({lab})", ylabel=f"events / {args.pot:.1e} POT", title=f"{lab}, stage {args.stage} (test folds)")
        ax.legend(); ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(os.path.join(args.plots, f"score_{tag}.png"), dpi=110); plt.close(fig)
        return score

    S1 = y == 1
    # (a) targeted background classifiers: all mb signal vs one background class
    for tag, lab, cats_ in TARGETS:
        B = np.isin(catc, cats_) & (y == 0)
        if wc[B].sum() <= 0:
            continue
        Xa = np.vstack([X[S1], X[B]]); ya = np.r_[np.ones(S1.sum()), np.zeros(B.sum())]
        sc = train(Xa, ya, np.r_[wc[S1], wc[B]], np.r_[evc[S1], evc[B]], lab, tag)
        if tag == "allnu":
            sc_b = sc[S1.sum():]; cat_b = catc[B]; w_b = wc[B]
            L.append("   mean score per background category (lower = easier to reject): " + " | ".join(
                f"{SPS.CATS[c][5:24]} {np.average(sc_b[(cat_b == c) & np.isfinite(sc_b)], weights=w_b[(cat_b == c) & np.isfinite(sc_b)]):.3f}"
                for c in range(SPS.N_SIG_CAT, SPS.N_MC_CAT) if w_b[(cat_b == c) & np.isfinite(sc_b)].sum() > 0))
    # (b) signal vs EXT
    Xa = np.vstack([X[S1], Xx]); ya = np.r_[np.ones(S1.sum()), np.zeros(len(Xx))]
    train(Xa, ya, np.r_[wc[S1], wxc], np.r_[evc[S1], evxc], "signal vs EXT", "ext")
    # (c) in-FV mb signal vs ENTERING mb signal (the entering sideband separator)
    infv = S1 & np.isin(catc, [0, 1]); ent = S1 & np.isin(catc, [2, 3])
    if wc[infv].sum() > 0 and wc[ent].sum() > 0:
        Xa = np.vstack([X[infv], X[ent]]); ya = np.r_[np.ones(infv.sum()), np.zeros(ent.sum())]
        sc = train(Xa, ya, np.r_[wc[infv], wc[ent]], np.r_[evc[infv], evc[ent]], "in-FV mb signal vs ENTERING mb signal", "enter")
        # how the backgrounds would score on this separator: refit on all signal, score backgrounds
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=3, l2_regularization=1.0, min_samples_leaf=20, random_state=1)
        wtr = np.r_[wc[infv] * 0.5 / wc[infv].sum(), wc[ent] * 0.5 / wc[ent].sum()]
        clf.fit(Xa, ya, sample_weight=wtr)
        sb = clf.predict_proba(X[y == 0])[:, 1]; cb = catc[y == 0]; wb = wc[y == 0]
        L.append("   in-FV-vs-entering separator applied to the backgrounds (mean score; 1 = in-FV-like): " + " | ".join(
            f"{SPS.CATS[c][5:24]} {np.average(sb[cb == c], weights=wb[cb == c]):.3f}" for c in range(SPS.N_SIG_CAT, SPS.N_MC_CAT) if wb[cb == c].sum() > 0))
        sx = clf.predict_proba(Xx)[:, 1]
        L.append(f"   EXT mean score on the separator: {np.average(sx, weights=wxc) if wxc.sum() > 0 else np.nan:.3f}")
    txt = "\n".join(L); print(txt)
    with open(os.path.join(args.plots, "mb_variable_study.txt"), "w") as fh:
        fh.write(txt + "\n")


if __name__ == "__main__":
    main()
