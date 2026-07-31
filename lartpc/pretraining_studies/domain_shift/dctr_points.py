#!/usr/bin/env python3
"""
Point-level DCTR reweighting with physics-observable closure
(POINT_LEVEL_PLAN.md item 5; observable list agreed with PI 2026-07-31).

Weights: out-of-fold P(data|point) from logistic regression on PCA-d
point embeddings with EVENT-GROUPED folds; w = p/(1-p) on MC points,
clipped at p99, mean-normalized. Dim scan reports ESS + closure (refit
with weights -> AUC toward 0.5). Local AUC ~0.6 means healthy support
overlap, so ESS should stay usable well past the event-level d~8 wall.

Closure observables (all truth-free, identical construction both domains,
attached at extraction with --attach-observables):
  - pixval spectra per plane (U, V, Y) and plane ratios U/Y, V/Y
  - larmatch score distribution
  - profiles of mean Y-plane pixval and mean lm score vs x (drift
    coordinate; NOTE: apparent x for out-of-time cosmics = true drift
    diluted by unknown t0 -- claim is "drift-coordinate-dependent
    response", not raw lifetime/diffusion), and vs y, z as the
    flux-geometry control axes.
Profiles carry event-blocked bootstrap errors and per-bin closure
fractions (rw-mc)/(data-mc).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from point_metrics import _grouped_folds  # noqa: E402

AXES = {
    "x": (0, np.linspace(-50, 290, 18)),
    "y": (1, np.linspace(-117, 117, 14)),
    "z": (2, np.linspace(0, 1036, 15)),
}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", required=True, help="MC *_ptsobs.npz")
    ap.add_argument("--b", required=True, help="data *_ptsobs.npz")
    ap.add_argument("--dims", default="8,16,32,64,128")
    ap.add_argument("--fit-cap", type=int, default=200000,
                    help="max training points per fold fit")
    ap.add_argument("--clip-pct", type=float, default=99.0)
    ap.add_argument("--n-boot", type=int, default=60)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fig-dir", default="figures")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def oof_probs_grouped(X, y, groups, n_dim, fit_cap, seed,
                      sample_weight=None):
    """Event-grouped OOF probabilities; fit on a capped subsample per fold,
    predict on every test point."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(seed)
    p = np.zeros(len(X))
    for tr, te in _grouped_folds(y, groups, 5, seed):
        fit_idx = tr if len(tr) <= fit_cap else \
            tr[rng.choice(len(tr), fit_cap, replace=False)]
        sc = StandardScaler().fit(X[fit_idx])
        Xf = sc.transform(X[fit_idx])
        pca = PCA(n_components=n_dim, random_state=0).fit(Xf) \
            if n_dim < X.shape[1] else None
        Xf = pca.transform(Xf) if pca is not None else Xf
        clf = LogisticRegression(max_iter=1000)
        sw = sample_weight[fit_idx] if sample_weight is not None else None
        clf.fit(Xf, y[fit_idx], sample_weight=sw)
        Xt = sc.transform(X[te])
        Xt = pca.transform(Xt) if pca is not None else Xt
        p[te] = clf.predict_proba(Xt)[:, 1]
    return p


def weights_from_probs(p_mc, clip_pct):
    eps = 1e-6
    w = np.clip(p_mc, eps, 1 - eps)
    w = w / (1.0 - w)
    clip_val = np.percentile(w, clip_pct)
    w = np.minimum(w, clip_val)
    w = w * len(w) / w.sum()
    ess = float((w.sum() ** 2) / (w ** 2).sum() / len(w))
    return w, ess, float(clip_val)


def event_boot_binned(vals, bins_idx, n_bins, ev, w, n_boot, seed):
    """Event-blocked bootstrap of per-bin weighted means."""
    rng = np.random.default_rng(seed)
    uev = np.unique(ev)
    by_ev = {e: np.flatnonzero(ev == e) for e in uev}

    def binned(idx):
        ww, vv, bb = w[idx], vals[idx], bins_idx[idx]
        num = np.bincount(bb, weights=ww * vv, minlength=n_bins)
        den = np.bincount(bb, weights=ww, minlength=n_bins)
        return np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)

    est = binned(np.arange(len(vals)))
    boots = []
    for _ in range(n_boot):
        take = rng.choice(uev, len(uev), replace=True)
        idx = np.concatenate([by_ev[e] for e in take])
        boots.append(binned(idx))
    boots = np.array(boots)
    return est, np.nanstd(boots, axis=0)


def main():
    args = parse_args()
    from sklearn.metrics import roc_auc_score

    fa, fb = np.load(args.a), np.load(args.b)
    Xa = fa["point_feats"].astype(np.float32)
    Xb = fb["point_feats"].astype(np.float32)
    ev_a, ev_b = fa["point_event"], fb["point_event"]
    X = np.concatenate([Xa, Xb])
    y = np.concatenate([np.zeros(len(Xa)), np.ones(len(Xb))])
    groups = np.concatenate([ev_a, ev_b + ev_a.max() + 1])
    print(f"[dctr-pts] {len(Xa)} MC vs {len(Xb)} data points")

    dims = [int(d) for d in args.dims.split(",")]
    scan, best = [], None
    for d in dims:
        p = oof_probs_grouped(X, y, groups, d, args.fit_cap, args.seed)
        auc_before = roc_auc_score(y, p)
        w_mc, ess, clip_val = weights_from_probs(p[:len(Xa)],
                                                 args.clip_pct)
        sw = np.concatenate([w_mc, np.ones(len(Xb))])
        p2 = oof_probs_grouped(X, y, groups, d, args.fit_cap,
                               args.seed + 1, sample_weight=sw)
        auc_after = roc_auc_score(y, p2)
        row = dict(dim=d, auc_before=float(auc_before),
                   auc_after_reweight=float(auc_after), ess_frac=ess,
                   clip_value=clip_val)
        scan.append(row)
        print(f"  d={d:4d} AUC {auc_before:.4f} -> {auc_after:.4f}  "
              f"ESS={ess:.3f}", flush=True)
        if (ess >= 0.2 and (best is None or
                            abs(row["auc_after_reweight"] - 0.5) <
                            abs(best["auc_after_reweight"] - 0.5))):
            best = row
    d_rep = (best or scan[0])["dim"]
    print(f"[dctr-pts] report dim = {d_rep}")
    p = oof_probs_grouped(X, y, groups, d_rep, args.fit_cap, args.seed)
    w_mc, ess, _ = weights_from_probs(p[:len(Xa)], args.clip_pct)
    ones_b = np.ones(len(Xb))

    pv_a, pv_b = fa["point_pixval"], fb["point_pixval"]
    lm_a, lm_b = fa["point_lm"], fb["point_lm"]
    xyz_a, xyz_b = fa["point_rawpos"], fb["point_rawpos"]

    res = dict(meta=dict(a=args.a, b=args.b, clip_pct=args.clip_pct,
                         report_dim=d_rep, ess=ess, seed=args.seed),
               dim_scan=scan, profiles={}, spectra_summary={})

    # ---- profiles: mean Y-plane pixval and mean lm vs x, y, z ----------
    for obs_name, va, vb in (("pixY", pv_a[:, 2], pv_b[:, 2]),
                             ("lm", lm_a, lm_b)):
        for ax_name, (col, edges) in AXES.items():
            nb = len(edges) - 1
            ba = np.clip(np.digitize(xyz_a[:, col], edges) - 1, 0, nb - 1)
            bb = np.clip(np.digitize(xyz_b[:, col], edges) - 1, 0, nb - 1)
            mc, mc_e = event_boot_binned(va, ba, nb, ev_a,
                                         np.ones(len(va)),
                                         args.n_boot, args.seed)
            rw, rw_e = event_boot_binned(va, ba, nb, ev_a, w_mc,
                                         args.n_boot, args.seed + 1)
            da, da_e = event_boot_binned(vb, bb, nb, ev_b, ones_b,
                                         args.n_boot, args.seed + 2)
            with np.errstate(invalid="ignore", divide="ignore"):
                closure = (rw - mc) / (da - mc)
            res["profiles"][f"{obs_name}_vs_{ax_name}"] = dict(
                edges=edges.tolist(), mc=mc.tolist(), mc_err=mc_e.tolist(),
                rw=rw.tolist(), rw_err=rw_e.tolist(), data=da.tolist(),
                data_err=da_e.tolist(), closure=closure.tolist(),
                mar_before=float(np.nanmean(np.abs(da - mc))),
                mar_after=float(np.nanmean(np.abs(da - rw))))
            print(f"  profile {obs_name} vs {ax_name}: mean|resid| "
                  f"{res['profiles'][f'{obs_name}_vs_{ax_name}']['mar_before']:.3f}"
                  f" -> "
                  f"{res['profiles'][f'{obs_name}_vs_{ax_name}']['mar_after']:.3f}",
                  flush=True)

    # ---- spectra summaries (weighted means) -----------------------------
    def wmean(v, w):
        return float(np.sum(v * w) / np.sum(w))

    ratios_a = {"U_over_Y": np.where(pv_a[:, 2] > 4,
                                     pv_a[:, 0] / np.maximum(pv_a[:, 2], 4),
                                     np.nan),
                "V_over_Y": np.where(pv_a[:, 2] > 4,
                                     pv_a[:, 1] / np.maximum(pv_a[:, 2], 4),
                                     np.nan)}
    ratios_b = {"U_over_Y": np.where(pv_b[:, 2] > 4,
                                     pv_b[:, 0] / np.maximum(pv_b[:, 2], 4),
                                     np.nan),
                "V_over_Y": np.where(pv_b[:, 2] > 4,
                                     pv_b[:, 1] / np.maximum(pv_b[:, 2], 4),
                                     np.nan)}
    spectra = {"pixU": (pv_a[:, 0], pv_b[:, 0]),
               "pixV": (pv_a[:, 1], pv_b[:, 1]),
               "pixY": (pv_a[:, 2], pv_b[:, 2]),
               "lm": (lm_a, lm_b),
               "U_over_Y": (ratios_a["U_over_Y"], ratios_b["U_over_Y"]),
               "V_over_Y": (ratios_a["V_over_Y"], ratios_b["V_over_Y"])}
    for k, (va, vb) in spectra.items():
        ok_a, ok_b = np.isfinite(va), np.isfinite(vb)
        mc_m = wmean(va[ok_a], np.ones(ok_a.sum()))
        rw_m = wmean(va[ok_a], w_mc[ok_a])
        da_m = wmean(vb[ok_b], np.ones(ok_b.sum()))
        res["spectra_summary"][k] = dict(
            mc=mc_m, rw=rw_m, data=da_m,
            closure_frac=float((rw_m - mc_m) / (da_m - mc_m))
            if abs(da_m - mc_m) > 1e-9 else float("nan"))
        print(f"  {k}: mc={mc_m:.4g} -> rw={rw_m:.4g} (data={da_m:.4g}, "
              f"moved {res['spectra_summary'][k]['closure_frac']:+.0%})")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)

    # ---- figures --------------------------------------------------------
    os.makedirs(args.fig_dir, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ds = [r["dim"] for r in scan]
    ax1.plot(ds, [r["auc_before"] for r in scan], "o-", label="AUC before")
    ax1.plot(ds, [r["auc_after_reweight"] for r in scan], "s-",
             label="AUC after reweight")
    ax1.axhline(0.5, ls="--", c="gray", lw=1)
    ax1.set_xscale("log")
    ax1.set_xlabel("PCA dim")
    ax1.set_ylabel("point domain AUC (grouped)")
    ax2 = ax1.twinx()
    ax2.plot(ds, [r["ess_frac"] for r in scan], "^--", c="tab:green",
             label="ESS/N")
    ax2.set_ylabel("ESS / N")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="center left")
    fig.tight_layout()
    fig.savefig(os.path.join(args.fig_dir, "dctr_pts_dim_scan.png"),
                dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    ranges = {"pixU": (0, 250), "pixV": (0, 250), "pixY": (0, 250),
              "lm": (0.1, 1.0), "U_over_Y": (0, 3), "V_over_Y": (0, 3)}
    for ax, k in zip(axes.flat, spectra):
        va, vb = spectra[k]
        ok_a, ok_b = np.isfinite(va), np.isfinite(vb)
        bins = np.linspace(*ranges[k], 40)
        ax.hist(va[ok_a], bins=bins, histtype="step", density=True,
                lw=1.6, label="MC")
        ax.hist(va[ok_a], bins=bins, weights=w_mc[ok_a], histtype="step",
                density=True, lw=1.6, ls="--", label="MC reweighted")
        ax.hist(vb[ok_b], bins=bins, histtype="step", density=True,
                lw=1.6, label="data")
        ax.set_xlabel(k)
        ax.set_yscale("log")
        ax.legend(fontsize=8)
    fig.suptitle(f"point-DCTR closure spectra (d={d_rep}, ESS/N={ess:.2f})")
    fig.tight_layout()
    fig.savefig(os.path.join(args.fig_dir, "dctr_pts_spectra.png"), dpi=150)
    plt.close(fig)

    for obs_name, ylab in (("pixY", "mean Y-plane pixval [ADC]"),
                           ("lm", "mean larmatch score")):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        for ax, ax_name in zip(axes, AXES):
            pr = res["profiles"][f"{obs_name}_vs_{ax_name}"]
            c = 0.5 * (np.array(pr["edges"])[:-1]
                       + np.array(pr["edges"])[1:])
            ax.errorbar(c, pr["mc"], yerr=pr["mc_err"], fmt="o-",
                        ms=3, label="MC")
            ax.errorbar(c, pr["rw"], yerr=pr["rw_err"], fmt="s--",
                        ms=3, label="MC reweighted")
            ax.errorbar(c, pr["data"], yerr=pr["data_err"], fmt="^-",
                        ms=3, label="data")
            ax.set_xlabel(f"{ax_name} [cm]")
            ax.set_ylabel(ylab)
            ax.set_title(f"|resid| {pr['mar_before']:.3f} -> "
                         f"{pr['mar_after']:.3f}")
            ax.legend(fontsize=8)
        fig.suptitle(f"point-DCTR profile closure: {obs_name} "
                     f"(d={d_rep})")
        fig.tight_layout()
        fig.savefig(os.path.join(args.fig_dir,
                                 f"dctr_pts_profile_{obs_name}.png"),
                    dpi=150)
        plt.close(fig)

    print(f"[done] -> {args.out}")


if __name__ == "__main__":
    main()
