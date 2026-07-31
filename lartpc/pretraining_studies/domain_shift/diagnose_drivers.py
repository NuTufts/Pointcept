#!/usr/bin/env python3
"""
What drives the domain classifier? Turn "AUC ~= 1" into physics statements.

Given the Tier-1 feature pair (nu-masked MC vs raw data), this script:
  1. computes out-of-fold logistic discriminant scores per event;
  2. builds simple per-event observables from the source h5 files
     (point count, charge scale, lm-score stats, MC ghost/nu fractions);
  3. reports Spearman correlations (observable vs score, within-domain)
     and single-observable cross-domain AUCs -- "how much of the
     separation does this one variable already give you?";
  4. ranks prototypes by cross-domain frequency ratio (Tier-1 pair) and
     lists the data-exclusive prototypes of the cleaned pair, with a
     scatter figure of their sampled points;
  5. compares the lm-score distributions of MC vs data (the direct
     LArMatch-response check that pairs with the cosmic-lmclean tier).

CPU-only. Run inside the container (needs numpy/sklearn/scipy/matplotlib
+ h5py); submit via run_drivers_tufts.sbatch -- the h5 loop reads ~10 GB.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_features import LM_KEYS  # noqa: E402

LM_CUT = 0.45
PIX_Q = 0.90


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--feat-a", required=True, help="MC tier-1 .npz")
    ap.add_argument("--feat-b", required=True, help="data raw .npz")
    ap.add_argument("--clean-a", required=True, help="MC cosmic-clean .npz")
    ap.add_argument("--clean-b", required=True, help="data cosmic-clean .npz")
    ap.add_argument("--mc-list", required=True)
    ap.add_argument("--data-list", required=True)
    ap.add_argument("--out", required=True, help="output .json")
    ap.add_argument("--fig-dir", default=None)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def oof_scores(Xa, Xb, seed=0):
    """Out-of-fold logistic decision scores, one per event."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    X = np.concatenate([Xa, Xb])
    y = np.concatenate([np.zeros(len(Xa)), np.ones(len(Xb))])
    s = np.zeros(len(X))
    for tr, te in StratifiedKFold(5, shuffle=True,
                                  random_state=seed).split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000).fit(sc.transform(X[tr]),
                                                    y[tr])
        s[te] = clf.decision_function(sc.transform(X[te]))
    return s[:len(Xa)], s[len(Xa):]


def name_to_path(list_file):
    m = {}
    with open(list_file) as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                m[os.path.splitext(os.path.basename(ln))[0]] = ln
    return m


def event_observables(names, path_map, is_mc, lm_hist_bins):
    """Per-event scalars + accumulated lm-score histogram for the domain."""
    import h5py
    obs = {k: [] for k in ("n_pts", "mean_pix", "q90_pix",
                           "mean_lm", "frac_lm_gt")}
    if is_mc:
        obs["ghost_frac"], obs["nu_frac"] = [], []
    lm_hist = np.zeros(len(lm_hist_bins) - 1, dtype=np.float64)
    for i, name in enumerate(names):
        with h5py.File(path_map[str(name)], "r") as f:
            td = f["/entry_0/triplet_data"]
            pix = np.array(td["pixval"], dtype=np.float32)
            obs["n_pts"].append(pix.shape[0])
            obs["mean_pix"].append(float(pix.mean()))
            obs["q90_pix"].append(float(np.quantile(pix, PIX_Q)))
            lm = None
            for k in LM_KEYS:
                if k in td:
                    lm = np.array(td[k], dtype=np.float32)
                    break
            if lm is not None:
                obs["mean_lm"].append(float(lm.mean()))
                obs["frac_lm_gt"].append(float((lm > LM_CUT).mean()))
                lm_hist += np.histogram(lm, bins=lm_hist_bins)[0]
            else:
                obs["mean_lm"].append(np.nan)
                obs["frac_lm_gt"].append(np.nan)
            if is_mc:
                hm = np.array(td["hasmatch"])
                org = np.array(td["origin"])
                obs["ghost_frac"].append(float((hm == 0).mean()))
                obs["nu_frac"].append(float((org == 1).mean()))
        if (i + 1) % 200 == 0:
            print(f"  observables {'mc' if is_mc else 'data'}: "
                  f"{i + 1}/{len(names)}", flush=True)
    return {k: np.array(v, dtype=np.float64) for k, v in obs.items()}, lm_hist


def auc_rank(x_a, x_b):
    """Mann-Whitney AUC of a single scalar separating the two domains."""
    from scipy.stats import mannwhitneyu
    ok_a, ok_b = np.isfinite(x_a), np.isfinite(x_b)
    if ok_a.sum() == 0 or ok_b.sum() == 0:
        return float("nan")
    u = mannwhitneyu(x_b[ok_b], x_a[ok_a], alternative="two-sided").statistic
    return float(u / (ok_a.sum() * ok_b.sum()))


def main():
    args = parse_args()
    from scipy.stats import spearmanr

    fa, fb = np.load(args.feat_a), np.load(args.feat_b)
    print(f"[oof] discriminant scores on {len(fa['pooled'])} + "
          f"{len(fb['pooled'])} events")
    sa, sb = oof_scores(fa["pooled"], fb["pooled"], args.seed)

    bins = np.linspace(0.0, 1.0, 51)
    print("[obs] MC h5 loop ...")
    obs_a, lmh_a = event_observables(
        fa["names"], name_to_path(args.mc_list), True, bins)
    print("[obs] data h5 loop ...")
    obs_b, lmh_b = event_observables(
        fb["names"], name_to_path(args.data_list), False, bins)

    res = {"meta": {"feat_a": args.feat_a, "feat_b": args.feat_b,
                    "clean_a": args.clean_a, "clean_b": args.clean_b,
                    "lm_cut": LM_CUT, "seed": args.seed}}

    # --- 3. correlations + single-observable AUCs -------------------------
    res["observables"] = {}
    for k in ("n_pts", "mean_pix", "q90_pix", "mean_lm", "frac_lm_gt"):
        row = {
            "auc_alone": auc_rank(obs_a[k], obs_b[k]),
            "spearman_score_mc": float(spearmanr(
                obs_a[k], sa, nan_policy="omit").statistic),
            "spearman_score_data": float(spearmanr(
                obs_b[k], sb, nan_policy="omit").statistic),
            "mean_mc": float(np.nanmean(obs_a[k])),
            "mean_data": float(np.nanmean(obs_b[k])),
        }
        res["observables"][k] = row
    for k in ("ghost_frac", "nu_frac"):
        res["observables"][k] = {
            "spearman_score_mc": float(spearmanr(
                obs_a[k], sa, nan_policy="omit").statistic),
            "mean_mc": float(np.nanmean(obs_a[k])),
        }

    # --- 4. prototype frequency ratios (tier-1 pair) ----------------------
    ca = fa["proto_hist"].sum(0).astype(np.float64)
    cb = fb["proto_hist"].sum(0).astype(np.float64)
    pa, pb = ca / ca.sum(), cb / cb.sum()
    eps = 0.5 / min(ca.sum(), cb.sum())
    ratio = np.log2((pb + eps) / (pa + eps))
    order = np.argsort(ratio)
    top = {}
    for label, idxs in (("data_enriched", order[::-1][:20]),
                        ("mc_enriched", order[:20])):
        top[label] = [{"proto": int(i), "log2_ratio": float(ratio[i]),
                       "n_mc": int(ca[i]), "n_data": int(cb[i])}
                      for i in idxs]
    res["proto_ratio_tier1"] = top

    clean_a, clean_b = np.load(args.clean_a), np.load(args.clean_b)
    cca = clean_a["proto_hist"].sum(0).astype(np.int64)
    ccb = clean_b["proto_hist"].sum(0).astype(np.int64)
    excl = np.flatnonzero((ccb >= 10) & (cca == 0))
    res["clean_data_exclusive"] = [
        {"proto": int(i), "n_data": int(ccb[i])}
        for i in excl[np.argsort(ccb[excl])[::-1]]]
    print(f"[proto] {len(excl)} data-exclusive prototypes in cleaned pair")

    # --- 5. lm-score histograms ------------------------------------------
    res["lm_hist"] = {"bins": bins.tolist(),
                      "mc": (lmh_a / max(lmh_a.sum(), 1)).tolist(),
                      "data": (lmh_b / max(lmh_b.sum(), 1)).tolist()}

    # --- figures ----------------------------------------------------------
    if args.fig_dir:
        os.makedirs(args.fig_dir, exist_ok=True)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        c = 0.5 * (bins[:-1] + bins[1:])
        plt.figure(figsize=(6, 4))
        plt.step(c, res["lm_hist"]["mc"], where="mid",
                 label="MC (lm_score)")
        plt.step(c, res["lm_hist"]["data"], where="mid",
                 label="data (larmatch_score)")
        plt.axvline(LM_CUT, ls="--", c="gray", lw=1)
        plt.yscale("log")
        plt.xlabel("larmatch score")
        plt.ylabel("normalized")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.fig_dir, "lm_score_mc_vs_data.png"),
                    dpi=150)
        plt.close()

        pts = clean_b["point_proto"]
        sel = np.isin(pts, excl)
        res["excl_points_in_sample"] = int(sel.sum())
        if sel.sum() > 10:
            xyz = clean_b["point_coord"][sel]
            pid = pts[sel]
            fig, axes = plt.subplots(1, 2, figsize=(11, 5))
            for ax, (i, j, lx, ly) in zip(
                    axes, [(2, 1, "z (norm)", "y (norm)"),
                           (2, 0, "z (norm)", "x (norm)")]):
                sc = ax.scatter(xyz[:, i], xyz[:, j], c=pid % 20,
                                cmap="tab20", s=4)
                ax.set_xlabel(lx)
                ax.set_ylabel(ly)
            fig.suptitle(
                f"sampled points of {len(excl)} data-exclusive prototypes "
                f"(n={sel.sum()})")
            fig.tight_layout()
            fig.savefig(os.path.join(args.fig_dir,
                                     "data_exclusive_protos.png"), dpi=150)
            plt.close(fig)
        else:
            print(f"[warn] only {sel.sum()} exclusive-proto points in the "
                  f"per-point sample; targeted re-extraction needed for a "
                  f"useful display")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)

    print("\n[report] single-observable AUCs (0.5 = uninformative):")
    for k, v in res["observables"].items():
        if "auc_alone" in v:
            print(f"  {k:12s} auc={v['auc_alone']:.3f}  "
                  f"mc={v['mean_mc']:.4g}  data={v['mean_data']:.4g}  "
                  f"rho(score)|mc={v['spearman_score_mc']:+.2f} "
                  f"|data={v['spearman_score_data']:+.2f}")
    print(f"[done] -> {args.out}")


if __name__ == "__main__":
    main()
