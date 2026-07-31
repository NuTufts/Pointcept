#!/usr/bin/env python3
"""
Point-level domain-gap battery (POINT_LEVEL_PLAN.md items 3+4).

Inputs: a dense-point feature pair (*_pts2048.npz). Computes:
  1. Point-level PAD with event-grouped folds (linear + HistGB on PCA-64),
     with event-blocked bootstrap CI on the linear AUC -- reported next to
     the EVENT-level linear AUC from the same files: the CLT comparison.
  2. Point-level MMD^2 with event-block permutation p-value + event
     bootstrap CI.
  3. Prototype-conditional shifts: for every prototype with >= --min-proto
     points on each side, the L2 norm of the standardized within-prototype
     mean difference; calibrated against a same-domain (MC half vs MC
     half) null computed identically. Ranked table + figure.

Outputs one JSON + figures. CPU-heavy; run via run_point_battery sbatch.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from domain_metrics import pad as event_pad  # noqa: E402
from point_metrics import (  # noqa: E402
    _standardize_pca, bootstrap_events, mmd2_points, pad_points)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", required=True, help="MC-side *_pts2048.npz")
    ap.add_argument("--b", required=True, help="data-side *_pts2048.npz")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fig-dir", default="figures")
    ap.add_argument("--n-pca", type=int, default=64)
    ap.add_argument("--max-points", type=int, default=150000)
    ap.add_argument("--n-boot", type=int, default=30)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--min-proto", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def load_points(path):
    f = np.load(path)
    return (f["point_feats"].astype(np.float32), f["point_event"],
            f["point_proto"], f["pooled"], f)


def proto_shifts(Xa, Xb, pa, pb, n_pca, min_pts, seed):
    """Standardized within-prototype mean-shift norms + same-domain null."""
    Xa2, Xb2 = _standardize_pca(Xa, Xb, n_pca, seed)
    ca = np.bincount(pa, minlength=4096)
    cb = np.bincount(pb, minlength=4096)
    protos = np.flatnonzero((ca >= min_pts) & (cb >= min_pts))
    rng = np.random.default_rng(seed)
    rows = []
    for pr in protos:
        A, B = Xa2[pa == pr], Xb2[pb == pr]
        shift = float(np.linalg.norm(A.mean(0) - B.mean(0)))
        # same-domain finite-sample floor: split the MC points in half
        h = rng.permutation(len(A))
        half = len(A) // 2
        null = float(np.linalg.norm(A[h[:half]].mean(0)
                                    - A[h[half:2 * half]].mean(0)))
        rows.append(dict(proto=int(pr), shift=shift, null_split=null,
                         excess=shift - null, n_mc=int(ca[pr]),
                         n_data=int(cb[pr])))
    rows.sort(key=lambda r: -r["excess"])
    return rows


def main():
    args = parse_args()
    Xa, ev_a, pa, pooled_a, fa = load_points(args.a)
    Xb, ev_b, pb, pooled_b, fb = load_points(args.b)
    print(f"[points] a: {Xa.shape}  b: {Xb.shape}")

    res = {"meta": dict(a=args.a, b=args.b, label=args.label,
                        n_pca=args.n_pca, max_points=args.max_points,
                        n_boot=args.n_boot, n_perm=args.n_perm,
                        seed=args.seed)}

    print("[event-level reference AUC]")
    res["event_pad"] = event_pad(pooled_a, pooled_b, seed=args.seed)

    print("[point PAD] grouped folds, linear + histgb ...")
    res["point_pad"] = pad_points(
        Xa, Xb, ev_a, ev_b, n_pca=args.n_pca,
        max_points=args.max_points, seed=args.seed)
    print(f"  point linear={res['point_pad']['auc_linear']:.4f} "
          f"histgb={res['point_pad']['auc_histgb']:.4f} "
          f"(event linear={res['event_pad']['auc_linear']:.4f})")

    print("[point PAD bootstrap] linear only ...")
    res["point_pad_boot"] = bootstrap_events(
        pad_points, Xa, Xb, ev_a, ev_b, n_boot=args.n_boot,
        seed=args.seed, n_pca=args.n_pca, max_points=args.max_points,
        classifiers=("linear",))

    print("[point MMD] block permutation ...")
    res["point_mmd"] = mmd2_points(
        Xa, Xb, ev_a, ev_b, n_pca=args.n_pca, n_perm=args.n_perm,
        seed=args.seed)
    print(f"  mmd2={res['point_mmd']['mmd2']:.5g} "
          f"p={res['point_mmd']['mmd2_p']:.4f} "
          f"null95={res['point_mmd']['mmd2_null_95']:.5g}")

    print("[prototype-conditional shifts] ...")
    rows = proto_shifts(Xa, Xb, pa, pb, args.n_pca, args.min_proto,
                        args.seed)
    res["proto_shift_top30"] = rows[:30]
    res["proto_shift_summary"] = dict(
        n_protos_tested=len(rows),
        median_shift=float(np.median([r["shift"] for r in rows])),
        median_null=float(np.median([r["null_split"] for r in rows])),
        n_excess_pos=int(sum(r["excess"] > 0 for r in rows)))
    print(f"  {len(rows)} prototypes tested; median shift "
          f"{res['proto_shift_summary']['median_shift']:.3f} vs null "
          f"{res['proto_shift_summary']['median_null']:.3f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)

    os.makedirs(args.fig_dir, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # figure filenames carry the label so parallel batteries don't clobber
    slug = "".join(c if c.isalnum() else "_" for c in args.label) or "pair"

    # CLT comparison figure
    plt.figure(figsize=(5.5, 4.2))
    labels = ["point\nlinear", "point\nHistGB", "event\nlinear",
              "event\nkNN"]
    vals = [res["point_pad"]["auc_linear"], res["point_pad"]["auc_histgb"],
            res["event_pad"]["auc_linear"], res["event_pad"]["auc_knn"]]
    b = res["point_pad_boot"]["auc_linear"]
    yerr = [[vals[0] - b["lo"], 0, 0, 0], [b["hi"] - vals[0], 0, 0, 0]]
    plt.bar(labels, vals, yerr=yerr, capsize=4,
            color=["tab:blue", "tab:blue", "tab:orange", "tab:orange"])
    plt.axhline(0.5, ls="--", c="gray", lw=1)
    plt.ylim(0.45, 1.02)
    plt.ylabel("domain AUC")
    plt.title(f"per-point vs event-level gap  ({args.label})")
    plt.tight_layout()
    plt.savefig(os.path.join(args.fig_dir, f"point_vs_event_auc_{slug}.png"),
                dpi=150)
    plt.close()

    # prototype shift ranking
    plt.figure(figsize=(6.5, 4.5))
    sh = np.array([r["shift"] for r in rows])
    nl = np.array([r["null_split"] for r in rows])
    n = np.array([min(r["n_mc"], r["n_data"]) for r in rows])
    plt.scatter(n, sh, s=8, label="MC vs data shift")
    plt.scatter(n, nl, s=8, alpha=0.5, label="MC split-half null")
    for r in rows[:8]:
        plt.annotate(str(r["proto"]),
                     (min(r["n_mc"], r["n_data"]), r["shift"]),
                     fontsize=7)
    plt.xscale("log")
    plt.xlabel("min(points per side) in prototype")
    plt.ylabel("standardized mean-shift norm")
    plt.legend()
    plt.title("within-prototype domain shift (top-8 labeled)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.fig_dir, f"proto_shift_ranking_{slug}.png"),
                dpi=150)
    plt.close()
    print(f"[done] -> {args.out}")


if __name__ == "__main__":
    main()
