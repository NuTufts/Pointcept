#!/usr/bin/env python3
"""WC-style MiniBooNE-targeted selection: pre-selection + three BDTs
(MB_SELECTION_PLAN.md, third pass). Kinematics-free inputs only.

  P0  pre-selection  : the shared S3 (nu slice, exactly one photon candidate on
                       either path, no visible X) -- or --stage pre = S5.
  B1  BDT_ext        : mb signal vs EXT cosmics            >= --t-ext
  B2  BDT_numu       : mb signal vs CC-numu-like nu bkg    >= --t-numu
                       (visible-mu / ring-making + entering CC)
  B3  BDT_enter      : in-FV mb signal vs ENTERING mb signal: >= --t-enter is
                       the IN-FV-LIKE sample, below is the ENTERING sideband
  split              : in-FV-like -> 1g0p / 1gNp (accepted protons >= --proton-ke-min)

Training: --train-mc (run-1 overlay TRAINPOOL half) + --train-ext (run-1 EXT
first half). Evaluation / plots: --mc-ntuple (RESERVED half), --ext-ntuple
(EXT second half), --data-ntuple. Nothing in the evaluation sample is used for
training. --no-flash drops the flash variables (log chi2, PE ratio) from every
BDT (poor data/MC agreement); the score plots at P0 carry data points and a
data / prediction ratio so the agreement can be judged for each BDT.

    python3 single_photon_mb_bdt_chain.py --train-mc A.root --train-ext X1.root \
        --mc-ntuple B.root --ext-ntuple X2.root --data-ntuple D.root --novtx \
        --ext-scale 1.202 --plots plots_run1_mbchain [--no-flash]
"""
import argparse
import itertools
import os
import sys

import numpy as np
import awkward as ak

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import single_photon_selection as SPS            # noqa: E402
import single_photon_mb_selection as MBS         # noqa: E402
from single_photon_diagnostics import add_args   # noqa: E402
from mb_variable_study import features, FEATURES  # noqa: E402

FLASH_VARS = {"logchi2", "pe_ratio"}
NUMU_CATS = [5, 7]


def chain_scores(a, R, names, S, E, path, args, model_path):
    """Flattened BDT_ext / BDT_numu / BDT_enter scores per event from a saved chain
    joblib (NaN without a candidate); None if the model file is absent. Needs the
    ntuple loaded with MBS.BR_EXTRA."""
    if not model_path or not os.path.exists(model_path):
        return None
    import joblib
    from mb_variable_study import features_from
    import __main__
    if not hasattr(__main__, "FoldEnsemble"):          # the joblib pickles the class under __main__
        __main__.FoldEnsemble = FoldEnsemble
    mod = joblib.load(model_path)
    Mf = features_from(a, R, names, S, E, path, args)
    has = np.isfinite(E); X = matrix(Mf, mod["keys"], has); out = {}
    for nm in ("ext", "numu", "enter"):
        v = np.full(len(E), np.nan)
        if "fold_models" in mod:                        # average the plain sklearn fold models
            raw = np.mean([m.predict_proba(X)[:, 1] for m in mod["fold_models"][nm]], 0)
        else:
            raw = mod[nm].predict_proba(X)[:, 1]
        if "flat" in mod:
            raw = np.interp(raw, mod["flat"][nm][0], mod["flat"][nm][1], left=0.0, right=1.0)
        v[has] = raw; out[nm] = v
    out["_features"] = Mf
    return out


def matrix(M, keys, mask):
    fin_ = lambda v: np.where(np.isfinite(v), v, np.nan)
    return np.stack([fin_(np.asarray(M[k], np.float64)) for k in keys], 1)[mask]


class FoldEnsemble:
    """K-fold cross-fitted gradient-boosted classifier (2026-10-02 retrain).

    The training sample is split by event-number parity; each fold model is
    fitted on the other fold with early stopping on a held-out 20% and stronger
    regularisation (min_samples_leaf 40, l2 3). predict_proba on NEW data
    averages the fold models; oof_scores() returns the out-of-fold score of
    every training event, which is what the flattening map is built from --
    so "flattened >= 1 - e" keeps a fraction e of UNSEEN signal, not of the
    memorised training signal (the first chain was overtrained: 17.5% of the
    held-out in-FV signal sat below flattened 0.05)."""

    def __init__(self, Xa, ya, wa, folds, seed=1):
        from sklearn.ensemble import HistGradientBoostingClassifier
        self.models, self.oof = [], np.full(len(ya), np.nan)
        for f in sorted(set(folds.tolist())):
            tr = folds != f
            clf = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, max_depth=3,
                                                 l2_regularization=3.0, min_samples_leaf=40,
                                                 early_stopping=True, validation_fraction=0.2,
                                                 n_iter_no_change=25, random_state=seed + f)
            clf.fit(Xa[tr], ya[tr], sample_weight=wa[tr])
            self.models.append(clf)
            self.oof[~tr] = clf.predict_proba(Xa[~tr])[:, 1]
        self.n_iter = [m.n_iter_ for m in self.models]

    def predict_proba(self, X):
        p = np.mean([m.predict_proba(X)[:, 1] for m in self.models], 0)
        return np.stack([1 - p, p], 1)


def fit(X1, w1, X0, w0, ev1, ev0, seed=1):
    Xa = np.vstack([X1, X0]); ya = np.r_[np.ones(len(X1)), np.zeros(len(X0))]
    # class-balanced weights normalised to a MEAN weight of 1: with tiny absolute weights the
    # L2-regularised leaf values (G / (H + lambda)) collapse and the scores compress into ~0.2-0.8
    wa = np.r_[w1 * 0.5 / max(w1.sum(), 1e-9), w0 * 0.5 / max(w0.sum(), 1e-9)]
    wa *= len(wa) / wa.sum()
    folds = (np.r_[ev1, ev0] % 2).astype(int)
    return FoldEnsemble(Xa, ya, wa, folds, seed)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(ap); ap.set_defaults(signal_def="mb")
    ap.add_argument("--train-mc", required=True); ap.add_argument("--train-ext", required=True)
    ap.add_argument("--train-ext-scale", type=float, default=1.202)
    for k, v in (("veto2g_emin", 10.0), ("contain_dwall", 10.0), ("dfw_cut", SPS.DFW_CUT), ("mb_mu_kemin", 40.0),
                 ("mb_pi_kemin", 50.0), ("mb_p_kemax", 342.0), ("extra_emax", 10.0), ("cone_e2_min", 5.0),
                 ("cone_e2_max", 600.0), ("tight_dwall", 10.0), ("dfw_tight", 100.0), ("conv_dist_min", 0.0),
                 ("cand_bdt_min_mb", 0.7), ("mb_pe_ratio_max", 2.0), ("mb_logchi2_max", 2.25)):
        ap.add_argument("--" + k.replace("_", "-"), type=float, default=v)
    ap.add_argument("--pi0-cone", action="store_true")
    ap.add_argument("--stage", default="S3", choices=["S3", "pre"])
    ap.add_argument("--no-flash", action="store_true", help="drop log chi2 and flash PE ratio from every BDT")
    ap.add_argument("--drop-vars", default="", help="comma-separated extra inputs to drop (e.g. pe_ratio)")
    ap.add_argument("--tag", default=None, help="output tag (default: flash / noflash)")
    ap.add_argument("--t-ext", type=float, default=0.3, help="FLATTENED score cut = 1 - signal efficiency")
    ap.add_argument("--t-numu", type=float, default=0.3); ap.add_argument("--t-enter", type=float, default=0.3)
    args = ap.parse_args()
    SPS.configure(args); os.makedirs(args.plots, exist_ok=True)
    import joblib
    drop = set(x for x in args.drop_vars.split(",") if x) | (FLASH_VARS if args.no_flash else set())
    keys = [k for k, _ in FEATURES if k not in drop]
    tag = args.tag or ("noflash" if args.no_flash else "flash")

    def prep(ntuple, is_mc):
        a, pot, M, ev = features(ntuple, is_mc, args)
        M["pre"] = M["base_S3"] if args.stage == "S3" else M["base"]
        M["pc"] = M["pc"].astype(np.float64); M["veto2g_fail"] = M["veto2g_fail"].astype(np.float64)
        return a, pot, M, ev

    # ---- training ---------------------------------------------------------------
    print(">>> training sample", flush=True)
    a, pot_t, Mt, evt = prep(args.train_mc, True)
    w0 = ak.to_numpy(a["xsecWeight"]).astype(np.float64); wt = np.where(w0 > 0, w0, 0.0) * (args.pot / pot_t)
    Tt = SPS.truth_tables(a); del a
    pre_t = Mt["pre"]; sig_t = Tt["sig_incl"] & pre_t; cat_t = Tt["cat"]
    _, _, MtX, evtX = prep(args.train_ext, False)
    preX_t = MtX["pre"]; wX_t = np.full(len(preX_t), args.train_ext_scale)
    Xs = matrix(Mt, keys, sig_t); ws = wt[sig_t]; evs = evt[sig_t]
    clf_ext = fit(Xs, ws, matrix(MtX, keys, preX_t), wX_t[preX_t], evs, evtX[preX_t])
    numu_t = pre_t & np.isin(cat_t, NUMU_CATS)
    clf_numu = fit(Xs, ws, matrix(Mt, keys, numu_t), wt[numu_t], evs, evt[numu_t])
    infv_t = sig_t & Tt["in_fv"]; ent_t = sig_t & ~Tt["in_fv"]
    clf_enter = fit(matrix(Mt, keys, infv_t), wt[infv_t], matrix(Mt, keys, ent_t), wt[ent_t], evt[infv_t], evt[ent_t])
    flat = {}
    for nm, clf, nsig, wq in (("ext", clf_ext, len(Xs), ws), ("numu", clf_numu, len(Xs), ws),
                              ("enter", clf_enter, int(infv_t.sum()), wt[infv_t])):
        sq = clf.oof[:nsig]; o = np.argsort(sq)                # OUT-OF-FOLD scores of the class-1 training events
        flat[nm] = (sq[o], np.cumsum(wq[o]) / wq[o].sum())      # raw score -> signal ECDF (0..1)
    joblib.dump({"keys": keys, "ext": clf_ext, "numu": clf_numu, "enter": clf_enter, "stage": args.stage,
                 "fold_models": {nm: c.models for nm, c in (("ext", clf_ext), ("numu", clf_numu), ("enter", clf_enter))},
                 "flat": {nm: (np.asarray(flat[nm][0]), np.asarray(flat[nm][1])) for nm in flat}},
                os.path.join(args.plots, f"mb_bdt_chain_{tag}.joblib"))
    def wquant(v, wv, q):
        o = np.argsort(v); cw = np.cumsum(wv[o]) / wv[o].sum()
        return float(v[o][np.searchsorted(cw, q)])
    # score thresholds that keep a given fraction of the TRAINING signal (in-FV signal for BDT_enter)
    EFFS = (0.95, 0.9, 0.8, 0.7, 0.6, 0.5)
    thr = {}
    for nm, clf, nsig, wq in (("ext", clf_ext, len(Xs), ws), ("numu", clf_numu, len(Xs), ws),
                              ("enter", clf_enter, int(infv_t.sum()), wt[infv_t])):
        sq = clf.oof[:nsig]
        thr[nm] = {e: 1.0 - e for e in EFFS}
        thr[nm + "_raw"] = {e: wquant(sq, wq, 1.0 - e) for e in EFFS}
    L = [f"MB BDT chain [{tag}], pre-selection {args.stage}; features ({len(keys)}): {', '.join(keys)}",
         "  plotted / cut scores are FLATTENED: raw score mapped to the training-signal ECDF, so the signal is "
         "uniform in [0,1] and a cut at s >= 1-e keeps a fraction e of the signal",
         "  score thresholds at training-signal efficiency " + ", ".join(f"{e:.2f}" for e in EFFS) + ": "
         + " | ".join(f"{nm} raw " + "/".join(f"{thr[nm + '_raw'][e]:.3f}" for e in EFFS) for nm in ("ext", "numu", "enter"))
         + " (flattened: 1 - efficiency)",
         f"  training: mb signal {ws.sum():.1f} ({int(sig_t.sum())} raw; in-FV {int(infv_t.sum())}, entering {int(ent_t.sum())}) | "
         f"CC-numu-like {wt[numu_t].sum():.1f} ({int(numu_t.sum())}) | EXT {wX_t[preX_t].sum():.1f} ({int(preX_t.sum())}) @ {args.pot:.1e} POT",
         "  2-fold cross-fitted ensembles, early stopping; boosting iterations per fold: "
         + " | ".join(f"{nm} {c.n_iter}" for nm, c in (("ext", clf_ext), ("numu", clf_numu), ("enter", clf_enter)))]

    # ---- evaluation samples -----------------------------------------------------
    def score(M, mask):
        X = matrix(M, keys, mask)
        out = {}
        for nm, clf in (("ext", clf_ext), ("numu", clf_numu), ("enter", clf_enter)):
            v = np.full(len(mask), np.nan); raw = clf.predict_proba(X)[:, 1]
            v[mask] = np.interp(raw, flat[nm][0], flat[nm][1], left=0.0, right=1.0); out[nm] = v
            r_ = np.full(len(mask), np.nan); r_[mask] = raw; out[nm + "_raw"] = r_
        return out

    print(">>> evaluation MC", flush=True)
    a, pot, M, _ = prep(args.mc_ntuple, True)
    w0 = ak.to_numpy(a["xsecWeight"]).astype(np.float64); w = np.where(w0 > 0, w0, 0.0) * (args.pot / pot)
    T = SPS.truth_tables(a); cat = T["cat"]; sig = T["sig_incl"]; infv = sig & T["in_fv"]; del a
    SC = score(M, M["pre"]); is_np = M["n_p"] >= 1
    MX = SCX = MD = SCD = None; wx = None
    if args.ext_ntuple:
        _, _, MX, _ = prep(args.ext_ntuple, False); SCX = score(MX, MX["pre"])
        wx = np.zeros(len(MX["pre"])); wx[args.ext_row_min:] = args.ext_scale
    if args.data_ntuple:
        _, _, MD, _ = prep(args.data_ntuple, False); SCD = score(MD, MD["pre"])
    CATS, NSIG, NMC = SPS.CATS, SPS.N_SIG_CAT, SPS.N_MC_CAT
    den_fv = w[infv].sum()
    # flattening calibration on the EVALUATION sample: fraction of the relevant signal below x
    # (should equal x if the out-of-fold map generalises)
    L.append("  flattening check on the evaluation sample, fraction of signal below flattened x = "
             "0.05 / 0.10 / 0.20 / 0.50 (ideal = x):")
    for nm, ref in (("ext", sig), ("numu", sig), ("enter", infv)):
        mm = M["pre"] & ref & np.isfinite(SC[nm]); W = w[mm].sum()
        L.append(f"    {nm:6s} " + " / ".join(f"{w[mm & (SC[nm] < x)].sum() / max(W, 1e-9):.3f}" for x in (0.05, 0.1, 0.2, 0.5)))

    def chain(Mm, Ss, te, tn, tent):
        nz = lambda v: np.nan_to_num(v, nan=-1.0)
        s0 = Mm["pre"]; s1 = s0 & (nz(Ss["ext"]) >= te); s2 = s1 & (nz(Ss["numu"]) >= tn)
        infl = s2 & (nz(Ss["enter"]) >= tent); entl = s2 & (nz(Ss["enter"]) < tent)
        return [s0, s1, s2, infl, entl]

    names = [f"P0 pre-selection ({args.stage})", f"B1 BDT_ext >= {args.t_ext:.2f}", f"B2 BDT_numu >= {args.t_numu:.2f}",
             f"B3 IN-FV-LIKE (BDT_enter >= {args.t_enter:.2f})", f"B3 ENTERING sideband (BDT_enter < {args.t_enter:.2f})"]
    S = chain(M, SC, args.t_ext, args.t_numu, args.t_enter)
    SX = chain(MX, SCX, args.t_ext, args.t_numu, args.t_enter) if MX is not None else None
    SD = chain(MD, SCD, args.t_ext, args.t_numu, args.t_enter) if MD is not None else None
    npx = (MX["n_p"] >= 1) if MX is not None else None; npd = (MD["n_p"] >= 1) if MD is not None else None

    def row(lab, m, mx, md):
        st, sf, b = w[m & sig].sum(), w[m & infv].sum(), w[m & ~sig].sum()
        e = wx[mx].sum() if mx is not None else 0.0; d = int(md.sum()) if md is not None else 0
        tot = max(st + b + e, 1e-9)
        return (f"{lab[:46]:46s}{st:8.1f}{sf:7.1f}{sf / max(den_fv, 1e-9):7.3f}{st / tot:7.3f}{sf / tot:7.3f}"
                f"{b:8.1f}{e:7.1f}{tot:8.1f}{d:6d}{d / tot:6.2f}")
    L.append(f"\nevaluation (RESERVED halves + data) @ {args.pot:.1e} POT; in-FV mb signal denominator {den_fv:.1f}")
    L.append(f"{'step':46s}{'signal':>8}{'in-FV':>7}{'effFV':>7}{'pur':>7}{'purFV':>7}{'MC bkg':>8}{'EXT':>7}{'pred':>8}{'data':>6}{'d/p':>6}")
    for k, nm in enumerate(names):
        L.append(row(nm, S[k], None if SX is None else SX[k], None if SD is None else SD[k]))
        L.append("    bkg: " + " | ".join(f"{CATS[c][5:26]} {w[S[k] & (cat == c)].sum():.1f}" for c in range(NSIG, NMC) if w[S[k] & (cat == c)].sum() > 0))
    for lab, mm, mmx, mmd in (("in-FV-like 1g0p", ~is_np, None if npx is None else ~npx, None if npd is None else ~npd), ("in-FV-like 1gNp", is_np, npx, npd)):
        L.append(row("  " + lab, S[3] & mm, None if SX is None else SX[3] & mmx, None if SD is None else SD[3] & mmd))
        L.append("    bkg: " + " | ".join(f"{CATS[c][5:26]} {w[S[3] & mm & (cat == c)].sum():.1f}" for c in range(NSIG, NMC) if w[S[3] & mm & (cat == c)].sum() > 0))
        L.append("    signal: " + " | ".join(f"{CATS[c][4:]} {w[S[3] & mm & (cat == c)].sum():.1f}" for c in range(NSIG)))

    # ---- threshold scan (in-FV-like sample) -------------------------------------
    L.append("\nSCAN (in-FV-like sample), thresholds by TRAINING-signal efficiency e_ext e_numu e_enter | "
             "all: sig effFV pur MCbkg EXT | 1g0p: sig pur bkg+EXT | 1gNp: sig pur bkg+EXT")
    for ee, en, eent in itertools.product((0.9, 0.8, 0.7), (0.9, 0.8, 0.7), (0.9, 0.8, 0.7, 0.6, 0.5)):
        te, tn, tent = thr["ext"][ee], thr["numu"][en], thr["enter"][eent]
        m = chain(M, SC, te, tn, tent)[3]; mx = chain(MX, SCX, te, tn, tent)[3] if MX is not None else None
        def nums(mm, mmx):
            st, b = w[mm & sig].sum(), w[mm & ~sig].sum(); e = wx[mmx].sum() if mmx is not None else 0.0
            return st, w[mm & infv].sum() / max(den_fv, 1e-9), st / max(st + b + e, 1e-9), b, e
        A = nums(m, mx); P0 = nums(m & ~is_np, None if mx is None else mx & ~npx); PN = nums(m & is_np, None if mx is None else mx & npx)
        L.append(f"  {ee:.1f} {en:.1f} {eent:.1f} (t {te:.2f} {tn:.2f} {tent:.2f}) | {A[0]:6.1f} {A[1]:.3f} {A[2]:.3f} {A[3]:6.1f} {A[4]:5.1f} | "
                 f"{P0[0]:5.1f} {P0[2]:.3f} {P0[3] + P0[4]:5.1f} | {PN[0]:5.1f} {PN[2]:.3f} {PN[3] + PN[4]:5.1f}")
    txt = "\n".join(L); print(txt)
    with open(os.path.join(args.plots, f"mb_bdt_chain_{tag}.txt"), "w") as fh:
        fh.write(txt + "\n")

    # ---- plots: scores at P0 with data / prediction ratio; final energy spectra ---
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    COL = SPS.CAT_COLORS

    def stack(vals, valsx, valsd, m, mx, md, bins, xlab, fname, title, ratio=True):
        lo, hi = bins[0], bins[-1] - 1e-6; ok = m & np.isfinite(vals)
        data = [np.clip(vals[ok & (cat == c)], lo, hi) for c in range(NMC)]; ws = [w[ok & (cat == c)] for c in range(NMC)]; cols = list(COL[:NMC])
        if valsx is not None:
            okx = mx & np.isfinite(valsx); data.append(np.clip(valsx[okx], lo, hi)); ws.append(wx[okx]); cols.append(COL[NMC])
        fig, axs = plt.subplots(2 if ratio else 1, 1, figsize=(7.6, 6.2 if ratio else 4.8), sharex=True,
                                gridspec_kw={"height_ratios": [3, 1]} if ratio else None)
        ax = axs[0] if ratio else axs
        ax.hist(data, bins=bins, weights=ws, stacked=True, color=cols, label=[f"{CATS[c]} ({ws[c].sum():.1f})" for c in range(len(ws))])
        hp = np.sum([np.histogram(d_, bins=bins, weights=w_)[0] for d_, w_ in zip(data, ws)], 0)
        if valsd is not None:
            okd = md & np.isfinite(valsd); hd, _ = np.histogram(np.clip(valsd[okd], lo, hi), bins=bins); ctr = 0.5 * (bins[:-1] + bins[1:])
            ax.errorbar(ctr[hd > 0], hd[hd > 0], yerr=np.sqrt(hd[hd > 0]), fmt="ko", ms=3.5, label=f"beam data ({int(hd.sum())})")
            if ratio:
                r = np.where(hp > 0, hd / np.maximum(hp, 1e-9), np.nan); er = np.where(hp > 0, np.sqrt(hd) / np.maximum(hp, 1e-9), np.nan)
                axs[1].errorbar(ctr, r, yerr=er, fmt="ko", ms=3); axs[1].axhline(1, color="r", lw=1); axs[1].set(ylim=(0, 2), ylabel="data / pred", xlabel=xlab)
                chi2 = np.nansum((hd - hp) ** 2 / np.maximum(hp, 1.0)); ndf = int((hp > 0).sum())
                ax.set_title(f"{title}  (data/pred {hd.sum() / max(hp.sum(), 1e-9):.2f}, chi2/ndf {chi2:.0f}/{ndf})")
        if not (valsd is not None and ratio):
            ax.set(title=title); (axs[1] if ratio else ax).set(xlabel=xlab)
        ax.set(ylabel=f"events / {args.pot:.1e} POT"); ax.legend(fontsize=5.5); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(args.plots, fname), dpi=110)
        ymax = max(hp.max() if len(hp) else 1.0, 1.0)
        ax.set_yscale("log"); ax.set_ylim(0.02, ymax * 3.0)
        fig.savefig(os.path.join(args.plots, fname.replace(".png", "_logy.png")), dpi=110); plt.close(fig)

    sb = np.linspace(0, 1, 26)
    if SCD is not None and SCX is not None:
        agree = []
        for nm in ("ext", "numu", "enter"):
            m0, mx0, md0 = S[0] & np.isfinite(SC[nm]), SX[0] & np.isfinite(SCX[nm]), SD[0] & np.isfinite(SCD[nm])
            hp = np.histogram(SC[nm][m0], bins=sb, weights=w[m0])[0] + np.histogram(SCX[nm][mx0], bins=sb, weights=wx[mx0])[0]
            hd = np.histogram(SCD[nm][md0], bins=sb)[0]
            chi2 = np.sum((hd - hp) ** 2 / np.maximum(hp, 1.0)); ndf = int((hp > 0).sum())
            agree.append(f"{nm}: data/pred {hd.sum() / max(hp.sum(), 1e-9):.3f}, chi2/ndf {chi2:.1f}/{ndf}")
        line = "score data/MC agreement at P0 (25 bins): " + " | ".join(agree)
        print(line)
        with open(os.path.join(args.plots, f"mb_bdt_chain_{tag}.txt"), "a") as fh:
            fh.write(line + "\n")
    for k, nm in enumerate(("ext", "numu", "enter")):
        prev = S[k] if k < 2 else S[2]; prevx = None if SX is None else (SX[k] if k < 2 else SX[2]); prevd = None if SD is None else (SD[k] if k < 2 else SD[2])
        xl = f"BDT_{nm} flattened score (signal-uniform; cut at 1 - eff)"
        stack(SC[nm], None if SCX is None else SCX[nm], None if SCD is None else SCD[nm], S[0], None if SX is None else SX[0], None if SD is None else SD[0],
              sb, xl, f"score_{nm}_P0_{tag}.png", f"BDT_{nm} at pre-selection [{tag}]")
        stack(SC[nm], None if SCX is None else SCX[nm], None if SCD is None else SCD[nm], prev, prevx, prevd,
              sb, xl, f"score_{nm}_prev_{tag}.png", f"BDT_{nm} after the previous steps [{tag}]")
        stack(SC[nm + "_raw"], None if SCX is None else SCX[nm + "_raw"], None if SCD is None else SCD[nm + "_raw"], S[0],
              None if SX is None else SX[0], None if SD is None else SD[0], sb, f"BDT_{nm} RAW score (classifier probability)",
              f"score_{nm}_P0_raw_{tag}.png", f"BDT_{nm} raw score at pre-selection [{tag}]")
    eb = np.linspace(0, 1000, 41)
    for lab, mm, mmx, mmd in (("infv_0p", S[3] & ~is_np, None if SX is None else SX[3] & ~npx, None if SD is None else SD[3] & ~npd),
                              ("infv_Np", S[3] & is_np, None if SX is None else SX[3] & npx, None if SD is None else SD[3] & npd),
                              ("entering", S[4], None if SX is None else SX[4], None if SD is None else SD[4])):
        stack(M["E"], None if MX is None else MX["E"], None if MD is None else MD["E"], mm, mmx, mmd, eb,
              "reco photon energy [MeV] (reported, not cut)", f"final_{lab}_E_{tag}.png", f"final {lab} [{tag}]", ratio=False)
    print(f">>> -> {args.plots}")


if __name__ == "__main__":
    main()
