#!/usr/bin/env python3
"""
F3: domain gap vs images seen, from the ladder battery JSONs.

  python3 plot_ladder.py --runs P5B.1 P1A.2 P1A.3 --tier tier1 \
      --results-dir results --out figures/f3_gap_vs_images.png

Plots, per run: MMD^2 / perm-null-95 (log scale; the scale-free effect
size comparable across snapshots AND models), kNN AUC, and proto JSD.
Points are labeled by images seen parsed from the result filenames
(<RUN>_img<N>_<tier>.json).
"""
import argparse
import glob
import json
import os
import re

import numpy as np


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--tier", default="tier1")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def collect(run, tier, rdir):
    rows = []
    for path in glob.glob(os.path.join(rdir, f"{run}_img*_{tier}.json")):
        m = re.search(r"_img(\d+)_", os.path.basename(path))
        if not m:
            continue
        r = json.load(open(path))
        mmd = r["mmd"]["mmd2"]["value"]
        n95 = r["mmd"]["mmd2_null_95"]["value"]
        rows.append(dict(
            img=int(m.group(1)),
            mmd_norm=mmd / n95 if n95 > 0 else np.nan,
            mmd_lo=r["mmd"]["mmd2"]["lo"] / n95 if n95 > 0 else np.nan,
            mmd_hi=r["mmd"]["mmd2"]["hi"] / n95 if n95 > 0 else np.nan,
            auc_knn=r["pad"]["auc_knn"]["value"],
            jsd=r["proto"]["proto_jsd"]["value"],
            jsd_lo=r["proto"]["proto_jsd"]["lo"],
            jsd_hi=r["proto"]["proto_jsd"]["hi"],
        ))
    return sorted(rows, key=lambda d: d["img"])


def main():
    args = parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    any_rows = False
    for run in args.runs:
        rows = collect(run, args.tier, args.results_dir)
        if not rows:
            print(f"[warn] no results for {run} {args.tier}")
            continue
        any_rows = True
        x = [r["img"] for r in rows]

        def _err(lo_hi):
            # percentile-bootstrap CIs can straddle the point estimate
            # (bias); clip negative bar lengths to zero
            return [[max(a, 0.0) for a in lo_hi[0]],
                    [max(a, 0.0) for a in lo_hi[1]]]

        axes[0].errorbar(
            x, [r["mmd_norm"] for r in rows],
            yerr=_err([[r["mmd_norm"] - r["mmd_lo"] for r in rows],
                       [r["mmd_hi"] - r["mmd_norm"] for r in rows]]),
            marker="o", capsize=3, label=run)
        axes[1].plot(x, [r["auc_knn"] for r in rows], "o-", label=run)
        axes[2].errorbar(
            x, [r["jsd"] for r in rows],
            yerr=_err([[r["jsd"] - r["jsd_lo"] for r in rows],
                       [r["jsd_hi"] - r["jsd"] for r in rows]]),
            marker="o", capsize=3, label=run)
    if not any_rows:
        raise SystemExit("no ladder results found")

    for ax, (ylab, yscale) in zip(axes, [
            ("MMD$^2$ / null$_{95}$", "log"),
            ("kNN domain AUC", "linear"),
            ("prototype JSD", "linear")]):
        ax.set_xscale("log")
        ax.set_yscale(yscale)
        ax.set_xlabel("images seen")
        ax.set_ylabel(ylab)
        ax.legend()
    axes[1].axhline(0.5, ls="--", c="gray", lw=1)
    fig.suptitle(f"MC-data gap vs pretraining budget ({args.tier})")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"[done] -> {args.out}")


if __name__ == "__main__":
    main()
