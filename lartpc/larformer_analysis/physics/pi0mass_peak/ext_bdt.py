"""Event-level BDT to reject EXT (cosmic) background in the two-photon
selection, trained signal-vs-EXT on the post-working-point sample.

Signal = MC truth-signal events (table cat 0/1), POT weights; background =
EXT events (spill-scaled). Features are the topological/geometry/flash
variables from datamc_diagnostics.load() — m_gg is deliberately EXCLUDED so
the score cannot sculpt the mass peak. Even/odd event split for train/test.
Model: sklearn HistGradientBoostingClassifier (NaN-native, so events with a
missing cascade flash PE are handled without imputation).

Outputs (into --plots): ROC, score distributions (signal / EXT / MC
non-signal / data), permutation feature importances, working-point table,
mass-sculpting check (signal m_gg shape at score cuts), and
bdt_scores.npz with per-event scores for all four populations.

    PYTHONPATH=./ python3 ext_bdt.py --mc-ntuple ... --mc-table ... \
        --mc-cascade ... --ext-ntuple ... --ext-table ... --ext-cascade ... \
        --data-ntuple ... --data-table ... --data-cascade ... \
        --ext-scale 0.5909 --plots plots_s1ep2p8_diag
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datamc_diagnostics import load, add_flash_pe  # noqa: E402

FEATS = ["cosY1", "cosY2", "cosZ1", "cosZ2", "vtxY", "dwall", "vtxScore",
         "dist1", "dist2", "E1", "E2", "cos12", "nPrimTrk", "nPhot",
         "flashPE", "logchi2", "vtxX", "vtxZ"]


def fmat(d):
    return np.column_stack([d[f].astype(np.float64) for f in FEATS])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for s in ("mc", "data", "ext"):
        ap.add_argument(f"--{s}-ntuple", required=True)
        ap.add_argument(f"--{s}-table", required=True)
        ap.add_argument(f"--{s}-cascade", required=True)
    ap.add_argument("--ext-scale", type=float, required=True)
    ap.add_argument("--recal-gamma-a", type=float, default=0.01556)
    ap.add_argument("--recal-gamma-b", type=float, default=-11.47)
    ap.add_argument("--mu-ke-min", type=float, default=50.0)
    ap.add_argument("--chi2-cut", type=float, default=1e4)
    ap.add_argument("--chi2-cut-nc", type=float, default=1778.0)
    ap.add_argument("--plots", required=True)
    ap.add_argument("--tag", default="",
                    help="suffix for output files (variant runs)")
    ap.add_argument("--drop-feats", default="",
                    help="comma-sep features to EXCLUDE (e.g. logchi2,flashPE)")
    ap.add_argument("--holdout-plots", action="store_true",
                    help="score npz, plots and impact table use ONLY the "
                         "held-out (odd-event) half of signal-MC and EXT, "
                         "weights x2 — no BDT-training events leak into any "
                         "plotted distribution. mcbkg/data (never trained on) "
                         "stay full.")
    ap.add_argument("--save-model", default=None,
                    help="joblib path to persist the trained classifier + "
                         "feature list + thresholds (official-cut deployment)")
    ap.add_argument("--smear", default="",
                    help="comma-sep feat:sigma — Gaussian noise added to that "
                         "feature in the TRAINING set only, de-sensitizing the "
                         "BDT to its fine (possibly mismodeled) shape")
    args = ap.parse_args()
    drop = {x for x in args.drop_feats.split(",") if x}
    global FEATS
    FEATS = [f for f in FEATS if f not in drop]
    smear = {}
    for tok in args.smear.split(","):
        if tok:
            k, v = tok.split(":"); smear[k] = float(v)
    if drop:  print(f">>> dropped features: {sorted(drop)}")
    if smear: print(f">>> training-time smear: {smear}")
    os.makedirs(args.plots, exist_ok=True)
    here = os.path.dirname(os.path.abspath(__file__))

    def cache_for(cdir):
        return os.path.join(here, "rse_" + os.path.basename(os.path.dirname(
            cdir.rstrip("/"))) + "_" + os.path.basename(
            cdir.rstrip("/")) + ".npz")

    smp = {}
    for s in ("mc", "data", "ext"):
        print(f">>> loading {s} ...", flush=True)
        smp[s] = load(getattr(args, f"{s}_ntuple"),
                      getattr(args, f"{s}_table"),
                      args.recal_gamma_a, args.recal_gamma_b,
                      args.mu_ke_min, args.chi2_cut, args.chi2_cut_nc)
        smp[s] = add_flash_pe(smp[s], getattr(args, f"{s}_cascade"),
                              cache_for(getattr(args, f"{s}_cascade")))
    smp["ext"]["w"] = np.full(len(smp["ext"]["run"]), args.ext_scale)
    cat = np.load(args.mc_table)["cat"][smp["mc"]["row"]]
    sig_m = cat < 2

    Xs, ws = fmat(smp["mc"])[sig_m], smp["mc"]["w"][sig_m]
    Xb, wb = fmat(smp["ext"]), smp["ext"]["w"]
    ev_s = smp["mc"]["event"][sig_m]
    ev_b = smp["ext"]["event"]
    tr_s, tr_b = (ev_s % 2 == 0), (ev_b % 2 == 0)
    X_tr = np.vstack([Xs[tr_s], Xb[tr_b]])
    y_tr = np.r_[np.ones(tr_s.sum()), np.zeros(tr_b.sum())]
    w_tr = np.r_[ws[tr_s], wb[tr_b]]
    # balance total class weight so neither dominates the loss
    w_tr[y_tr == 1] *= w_tr[y_tr == 0].sum() / w_tr[y_tr == 1].sum()
    if smear:
        rng = np.random.default_rng(11)
        for k, sg in smear.items():
            j = FEATS.index(k)
            X_tr[:, j] = X_tr[:, j] + rng.normal(0, sg, len(X_tr))
    print(f">>> train: sig {int(tr_s.sum())} | ext {int(tr_b.sum())}; "
          f"test: sig {int((~tr_s).sum())} | ext {int((~tr_b).sum())}")

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.inspection import permutation_importance
    clf = HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.08, max_iter=400,
        early_stopping=True, validation_fraction=0.2, random_state=7)
    clf.fit(X_tr, y_tr, sample_weight=w_tr)
    print(f">>> trained: {clf.n_iter_} iters")

    def score(X):
        return clf.predict_proba(X)[:, 1]

    s_te, b_te = score(Xs[~tr_s]), score(Xb[~tr_b])
    ws_te, wb_te = ws[~tr_s], wb[~tr_b]
    ths = np.unique(np.r_[s_te, b_te])
    eff = np.array([ws_te[s_te >= t].sum() for t in ths]) / ws_te.sum()
    rej = 1 - np.array([wb_te[b_te >= t].sum() for t in ths]) / wb_te.sum()
    auc = float(np.trapz(np.clip(eff, 0, 1)[np.argsort(1 - rej)],
                         np.sort(1 - rej)))
    print(f">>> test AUC ~ {auc:.3f}")
    print(f"{'sig eff':>8} {'EXT rej':>8} {'thr':>7}")
    wps = {}
    for target in (0.99, 0.97, 0.95, 0.90, 0.85):
        k = int(np.argmin(np.abs(eff - target)))
        wps[target] = float(ths[k])
        print(f"{eff[k]:8.3f} {rej[k]:8.3f} {ths[k]:7.3f}")

    imp = permutation_importance(
        clf, np.vstack([Xs[~tr_s], Xb[~tr_b]]),
        np.r_[np.ones((~tr_s).sum()), np.zeros((~tr_b).sum())],
        sample_weight=np.r_[ws_te * wb_te.sum() / ws_te.sum(), wb_te],
        n_repeats=5, random_state=7)
    order = np.argsort(imp.importances_mean)[::-1]
    print("\n== permutation importance (test half) ==")
    for k in order[:12]:
        print(f"  {FEATS[k]:10s} {imp.importances_mean[k]:.4f}")

    if args.save_model:
        import joblib
        joblib.dump({"clf": clf, "feats": FEATS, "wps": wps,
                     "drop": sorted(drop), "tag": args.tag},
                    args.save_model)
        print(f">>> model -> {args.save_model}")
    if args.holdout_plots:
        ho_s, ho_b = ~tr_s, ~tr_b
        sig_pack = (score(Xs[ho_s]), 2.0 * ws[ho_s],
                    smp["mc"]["mgg"][sig_m][ho_s], ev_s[ho_s])
        ext_pack = (score(Xb[ho_b]), 2.0 * wb[ho_b],
                    smp["ext"]["mgg"][ho_b], ev_b[ho_b])
        print(">>> holdout-plots: sig/EXT restricted to odd-event half, w x2")
    else:
        sig_pack = (score(Xs), ws, smp["mc"]["mgg"][sig_m], ev_s)
        ext_pack = (score(Xb), wb, smp["ext"]["mgg"], ev_b)
    sc = {"sig": sig_pack,
          "mcbkg": (score(fmat(smp["mc"])[~sig_m]), smp["mc"]["w"][~sig_m],
                    smp["mc"]["mgg"][~sig_m], smp["mc"]["event"][~sig_m]),
          "ext": ext_pack,
          "data": (score(fmat(smp["data"])), smp["data"]["w"],
                   smp["data"]["mgg"], smp["data"]["event"])}
    np.savez(os.path.join(args.plots, f"bdt_scores{args.tag}.npz"),
             feats=np.array(FEATS),
             **{f"{k}_{n}": v for k, (s_, w_, m_, e_) in sc.items()
                for n, v in (("score", s_), ("w", w_), ("mgg", m_),
                             ("event", e_))})

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.plot(1 - rej, eff, lw=1.8)
    for t, m in ((0.97, "o"), (0.90, "s")):
        k = int(np.argmin(np.abs(eff - t)))
        ax.plot(1 - rej[k], eff[k], m, ms=6,
                label=f"eff {eff[k]:.2f} / rej {rej[k]:.2f}")
    ax.set(xlabel="EXT acceptance", ylabel="signal efficiency",
           title=f"EXT-rejection BDT ROC (test half, AUC~{auc:.3f})",
           xscale="log")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.plots, f"bdt_roc{args.tag}.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    bins = np.linspace(0, 1, 41)
    for k, lab, sty in (("sig", "signal MC", dict(color="#d62728")),
                        ("mcbkg", "MC non-signal", dict(color="#1f77b4")),
                        ("ext", "EXT", dict(color="#7f7f7f"))):
        s_, w_, _, _ = sc[k]
        ax.hist(s_, bins, weights=w_, density=True, histtype="step",
                lw=1.8, label=lab, **sty)
    s_, w_, _, _ = sc["data"]
    h, _ = np.histogram(s_, bins)
    ctr = 0.5 * (bins[:-1] + bins[1:])
    ax.errorbar(ctr, h / h.sum() / np.diff(bins), fmt="ko", ms=3,
                yerr=np.sqrt(h) / h.sum() / np.diff(bins), label="data")
    ax.set(xlabel="BDT score (signal-like ->)", ylabel="density",
           title="EXT-rejection BDT score")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.plots, f"bdt_score{args.tag}.png"), dpi=120)
    plt.close(fig)

    # mass-sculpting check: signal m_gg shape at score cuts
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    mb = np.linspace(0, 400, 41)
    s_, w_, m_, _ = sc["sig"]
    for t, lab in ((None, "no cut"), (wps[0.97], "eff 0.97 cut"),
                   (wps[0.90], "eff 0.90 cut")):
        m = np.ones(len(s_), bool) if t is None else s_ >= t
        ax.hist(m_[m], mb, weights=w_[m], density=True, histtype="step",
                lw=1.6, label=lab)
    ax.set(xlabel="signal m_gg [MeV]", ylabel="density",
           title="mass-sculpting check (signal shape vs score cut)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.plots, f"bdt_sculpt{args.tag}.png"), dpi=120)
    plt.close(fig)
    print("\n== near-peak (100-170 MeV) impact ==")
    print(f"{'cut':>10} {'sig':>7} {'MCbkg':>7} {'EXT':>7} {'pred':>7} "
          f"{'data':>6} {'d/p':>5} {'sig-frac':>8}")
    for nm, t in [("no cut", -1.0)] + [(f"eff {k}", v)
                                       for k, v in wps.items()]:
        row = []
        for k in ("sig", "mcbkg", "ext", "data"):
            s_, w_, m_, _ = sc[k]
            m = (s_ >= t) & (m_ > 100) & (m_ < 170)
            row.append(w_[m].sum())
        sg, bk, ex, da = row
        pr = sg + bk + ex
        print(f"{nm:>10} {sg:7.0f} {bk:7.0f} {ex:7.0f} {pr:7.0f} "
              f"{da:6.0f} {da/pr:5.2f} {sg/pr:8.3f}")
    print(f">>> plots + bdt_scores{args.tag}.npz -> {args.plots}")


if __name__ == "__main__":
    main()
