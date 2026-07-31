#!/usr/bin/env python3
"""
DCTR-style embedding-space reweighting demo (plan figure F5).

On a tier-controlled feature pair (default: the method-symmetric clean
pair), derive per-event density-ratio weights w = p/(1-p) from an
out-of-fold domain classifier trained on PCA-reduced embeddings, then:

  1. report weight health vs PCA dimensionality: effective sample size
     (ESS), clip fraction, AUC before;
  2. closure test: retrain a fresh classifier on REWEIGHTED MC vs data --
     AUC should fall toward 0.5 where the reweighting is usable;
  3. propagate to observables the classifier never saw directly (charge
     scale q90_pix, points/event from the source h5) -- does reweighting
     in embedding space move physics observables toward data?

With near-separable domains (event-level AUC ~1) full-dimensional weights
are expected to be ill-conditioned; the dimensionality scan documents
where the method IS usable, which is itself the F5 message.

Outputs: one JSON + figures (weight histogram, ESS/closure vs dim,
reweighted-observable comparison). Run via run_dctr_tufts.sbatch.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_features import LM_KEYS  # noqa: E402  (h5 key convention)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", required=True, help="MC-side .npz (source)")
    ap.add_argument("--b", required=True, help="data-side .npz (target)")
    ap.add_argument("--mc-list", required=True,
                    help="filelist mapping MC event names to h5 paths")
    ap.add_argument("--data-list", required=True)
    ap.add_argument("--dims", default="4,8,16,32,64,128")
    ap.add_argument("--report-dim", type=int, default=16,
                    help="dimensionality used for the weight/observable "
                         "figures (overridden to the best-closure dim if "
                         "that one has higher ESS)")
    ap.add_argument("--clip-pct", type=float, default=99.0,
                    help="clip weights at this percentile before use")
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fig-dir", default="figures")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def oof_probs(X, y, n_dim, seed, sample_weight=None):
    """Out-of-fold P(domain=data) from logreg on PCA-n_dim features."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    p = np.zeros(len(X))
    for tr, te in StratifiedKFold(5, shuffle=True,
                                  random_state=seed).split(X, y):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        if n_dim < X.shape[1]:
            pca = PCA(n_components=n_dim, random_state=0).fit(Xtr)
            Xtr, Xte = pca.transform(Xtr), pca.transform(Xte)
        clf = LogisticRegression(max_iter=2000)
        sw = sample_weight[tr] if sample_weight is not None else None
        clf.fit(Xtr, y[tr], sample_weight=sw)
        p[te] = clf.predict_proba(Xte)[:, 1]
    return p


def weights_from_probs(p_mc, clip_pct):
    eps = 1e-6
    w = np.clip(p_mc, eps, 1 - eps)
    w = w / (1.0 - w)
    clip_val = np.percentile(w, clip_pct)
    n_clipped = int((w > clip_val).sum())
    w = np.minimum(w, clip_val)
    w = w * len(w) / w.sum()  # normalize mean to 1
    ess = float((w.sum() ** 2) / (w ** 2).sum() / len(w))
    return w, ess, float(clip_val), n_clipped


def event_observables(names, list_file):
    import h5py
    path = {}
    with open(list_file) as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                path[os.path.splitext(os.path.basename(ln))[0]] = ln
    n_pts, q90 = [], []
    for i, name in enumerate(names):
        with h5py.File(path[str(name)], "r") as f:
            pix = np.array(f["/entry_0/triplet_data/pixval"],
                           dtype=np.float32)
            n_pts.append(pix.shape[0])
            q90.append(float(np.quantile(pix, 0.90)))
        if (i + 1) % 250 == 0:
            print(f"  obs {i + 1}/{len(names)}", flush=True)
    return np.array(n_pts, float), np.array(q90, float)


def wmean_ci(x, w, n_boot, seed):
    rng = np.random.default_rng(seed)
    est = float(np.sum(w * x) / np.sum(w))
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        boots.append(np.sum(w[i] * x[i]) / np.sum(w[i]))
    return est, float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def main():
    args = parse_args()
    from sklearn.metrics import roc_auc_score

    fa, fb = np.load(args.a), np.load(args.b)
    Xa, Xb = fa["pooled"], fb["pooled"]
    X = np.concatenate([Xa, Xb])
    y = np.concatenate([np.zeros(len(Xa)), np.ones(len(Xb))])
    dims = [int(d) for d in args.dims.split(",")]
    print(f"[dctr] {len(Xa)} MC vs {len(Xb)} data; dims={dims}")

    scan, best = [], None
    for d in dims:
        p = oof_probs(X, y, d, args.seed)
        auc_before = roc_auc_score(y, p)
        w_mc, ess, clip_val, n_clip = weights_from_probs(
            p[:len(Xa)], args.clip_pct)
        sw = np.concatenate([w_mc, np.ones(len(Xb))])
        p2 = oof_probs(X, y, d, args.seed + 1, sample_weight=sw)
        auc_after = roc_auc_score(y, p2)
        row = dict(dim=d, auc_before=float(auc_before),
                   auc_after_reweight=float(auc_after), ess_frac=ess,
                   clip_value=clip_val, n_clipped=n_clip)
        scan.append(row)
        print(f"  d={d:4d} AUC {auc_before:.4f} -> {auc_after:.4f} "
              f"(closure)  ESS={ess:.3f}  clipped={n_clip}")
        if best is None or (abs(auc_after - 0.5) <
                            abs(best["auc_after_reweight"] - 0.5)):
            best = row

    d_rep = best["dim"] if best["ess_frac"] >= 0.05 else args.report_dim
    print(f"[dctr] report dim = {d_rep} (best closure: d={best['dim']})")
    p = oof_probs(X, y, d_rep, args.seed)
    w_mc, ess, clip_val, _ = weights_from_probs(p[:len(Xa)], args.clip_pct)

    print("[obs] h5 loops (charge scale + points/event) ...")
    n_a, q_a = event_observables(fa["names"], args.mc_list)
    n_b, q_b = event_observables(fb["names"], args.data_list)
    ones = np.ones(len(q_a))
    obs = {}
    for key, xa, xb in (("q90_pix", q_a, q_b), ("n_pts", n_a, n_b)):
        mc_e, mc_lo, mc_hi = wmean_ci(xa, ones, args.n_boot, args.seed)
        rw_e, rw_lo, rw_hi = wmean_ci(xa, w_mc, args.n_boot, args.seed)
        da_e, da_lo, da_hi = wmean_ci(xb, np.ones(len(xb)), args.n_boot,
                                      args.seed)
        obs[key] = dict(
            mc=[mc_e, mc_lo, mc_hi], mc_reweighted=[rw_e, rw_lo, rw_hi],
            data=[da_e, da_lo, da_hi],
            closure_frac=float((rw_e - mc_e) / (da_e - mc_e))
            if abs(da_e - mc_e) > 0 else float("nan"))
        print(f"  {key}: mc={mc_e:.4g} -> reweighted={rw_e:.4g} "
              f"(data={da_e:.4g}; moved {obs[key]['closure_frac']:+.0%} "
              f"of the gap)")

    res = dict(meta=dict(a=args.a, b=args.b, clip_pct=args.clip_pct,
                         report_dim=d_rep, seed=args.seed),
               dim_scan=scan, ess_report_dim=ess,
               observables=obs)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)

    os.makedirs(args.fig_dir, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ds = [r["dim"] for r in scan]
    ax1.plot(ds, [r["auc_before"] for r in scan], "o-",
             label="AUC before")
    ax1.plot(ds, [r["auc_after_reweight"] for r in scan], "s-",
             label="AUC after reweight (closure)")
    ax1.axhline(0.5, ls="--", c="gray", lw=1)
    ax1.set_xscale("log")
    ax1.set_xlabel("PCA dimensionality")
    ax1.set_ylabel("domain AUC")
    ax2 = ax1.twinx()
    ax2.plot(ds, [r["ess_frac"] for r in scan], "^--", c="tab:green",
             label="ESS fraction")
    ax2.set_ylabel("effective sample size / N")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="center left")
    fig.tight_layout()
    fig.savefig(os.path.join(args.fig_dir, "dctr_dim_scan.png"), dpi=150)
    plt.close(fig)

    plt.figure(figsize=(6, 4))
    plt.hist(w_mc, bins=np.geomspace(max(w_mc.min(), 1e-3),
                                     w_mc.max(), 40))
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(f"MC event weight (d={d_rep}, clipped at "
               f"p{args.clip_pct:g})")
    plt.ylabel("events")
    plt.title(f"ESS/N = {ess:.3f}")
    plt.tight_layout()
    plt.savefig(os.path.join(args.fig_dir, "dctr_weights.png"), dpi=150)
    plt.close()

    for key, xa, xb, xlabel in (
            ("q90pix", q_a, q_b, "q90 pixval (charge scale)"),
            ("npts", n_a, n_b, "points per event (raw triplet count)")):
        plt.figure(figsize=(6.5, 4.5))
        bins = np.linspace(0, np.percentile(np.concatenate([xa, xb]), 99),
                           40)
        plt.hist(xa, bins=bins, histtype="step", density=True, lw=1.8,
                 label="MC")
        plt.hist(xa, bins=bins, weights=w_mc, histtype="step", density=True,
                 lw=1.8, ls="--", label=f"MC reweighted (d={d_rep})")
        plt.hist(xb, bins=bins, histtype="step", density=True, lw=1.8,
                 label="data")
        plt.xlabel(xlabel)
        plt.ylabel("normalized")
        plt.title(f"DCTR closure (ESS/N = {ess:.2f})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.fig_dir, f"dctr_{key}_closure.png"),
                    dpi=150)
        plt.close()

    print(f"[done] -> {args.out}")


if __name__ == "__main__":
    main()
