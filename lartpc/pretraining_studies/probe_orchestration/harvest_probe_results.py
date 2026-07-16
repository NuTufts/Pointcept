#!/usr/bin/env python3
"""
Harvest probe results into representation-quality curves (RUNS AT TUFTS).

Walks exp/probes/<run_id>/img<N>/train.log, extracts the best and final val
mIoU plus the per-class IoU block at the best eval, and writes:

  <outdir>/probe_curves.csv     one row per (run, images_seen)
  <outdir>/probe_curves.png     mIoU vs images-seen per run (log x)
  <outdir>/probe_classes.png    pion / proton IoU vs images-seen
  <outdir>/summary.md           latest-common-point comparison table

Supervised ceiling reference lines (P05A, Isambard 2026-07-16):
  A.1 mIoU 0.8576 (pion 0.773, proton 0.933); A.2 geometry-only 0.8152.
Fraction-of-ceiling (M6) = probe mIoU / 0.8576 on the same split.

Usage:  python3 harvest_probe_results.py --repo <REPO> --outdir <REPO>/exp/probes/analysis
"""
import argparse
import csv
import glob
import os
import re

VAL_RE = re.compile(
    r"Val result: mIoU/mAcc/allAcc (\d\.\d+)/(\d\.\d+)/(\d\.\d+)")
CLS_RE = re.compile(
    r"Class_(\d)-(\w+) Result: iou/accuracy (\d\.\d+)/(\d\.\d+)")
CEILING = {"mIoU": 0.8576, "pion": 0.7727, "proton": 0.9328}


def parse_probe_log(path):
    """Return (best_eval, final_eval) where each is dict(mIoU=..., cls={...})."""
    evals = []
    with open(path) as f:
        cur = None
        for line in f:
            m = VAL_RE.search(line)
            if m:
                cur = dict(mIoU=float(m.group(1)), mAcc=float(m.group(2)),
                           allAcc=float(m.group(3)), cls={})
                evals.append(cur)
                continue
            m = CLS_RE.search(line)
            if m and cur is not None:
                cur["cls"][m.group(2)] = float(m.group(3))
    if not evals:
        return None, None
    best = max(evals, key=lambda e: e["mIoU"])
    return best, evals[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.environ.get(
        "LOCAL_REPO",
        "/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/isambard_pointcept"))
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    outdir = args.outdir or os.path.join(args.repo, "exp/probes/analysis")
    os.makedirs(outdir, exist_ok=True)

    rows = []
    for log in sorted(glob.glob(
            os.path.join(args.repo, "exp/probes/*/img*/train.log"))):
        probe_dir = os.path.dirname(log)
        run_id = os.path.basename(os.path.dirname(probe_dir))
        img = int(os.path.basename(probe_dir)[3:])
        best, final = parse_probe_log(log)
        done = os.path.exists(os.path.join(probe_dir, ".done"))
        if best is None:
            print(f"WARN no evals yet in {log}")
            continue
        row = dict(run_id=run_id, images_seen=img,
                   best_mIoU=best["mIoU"], final_mIoU=final["mIoU"],
                   frac_of_ceiling=best["mIoU"] / CEILING["mIoU"],
                   job_done=done)
        for cname in ("electron", "muon", "pion", "proton",
                      "gamma", "michel", "delta", "led"):
            row[f"iou_{cname}"] = best["cls"].get(cname, float("nan"))
        rows.append(row)

    if not rows:
        print("no probe results found"); return
    rows.sort(key=lambda r: (r["run_id"], r["images_seen"]))
    with open(os.path.join(outdir, "probe_curves.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} probe points -> {outdir}/probe_curves.csv")

    # ---------------------------------------------------------------- plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    runs = sorted(set(r["run_id"] for r in rows))
    fig, ax = plt.subplots(figsize=(8, 5))
    for run in runs:
        pts = [(r["images_seen"], r["best_mIoU"]) for r in rows
               if r["run_id"] == run]
        ax.plot(*zip(*pts), marker="o", ms=3, label=run)
    ax.axhline(CEILING["mIoU"], color="k", ls="--", lw=1,
               label="P05A.1 supervised ceiling")
    ax.set_xscale("log"); ax.set_xlabel("images seen")
    ax.set_ylabel("linear-probe val mIoU"); ax.legend(fontsize=7)
    ax.set_title("Representation quality vs pretraining images seen")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "probe_curves.png"), dpi=130)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for axi, cname in zip(axes, ("pion", "proton")):
        for run in runs:
            pts = [(r["images_seen"], r[f"iou_{cname}"]) for r in rows
                   if r["run_id"] == run]
            axi.plot(*zip(*pts), marker="o", ms=3, label=run)
        axi.axhline(CEILING[cname], color="k", ls="--", lw=1)
        axi.set_xscale("log"); axi.set_xlabel("images seen")
        axi.set_title(f"{cname} IoU (dashed = supervised ceiling)")
    axes[0].set_ylabel("probe IoU"); axes[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "probe_classes.png"), dpi=130)

    # -------------------------------------------------------------- summary
    latest_common = min(max(r["images_seen"] for r in rows
                            if r["run_id"] == run) for run in runs)
    lines = ["# Probe sweep summary", "",
             f"probe points: {len(rows)}; runs: {len(runs)}",
             f"latest common images-seen point: {latest_common:,}", "",
             "| run | mIoU@common | frac of ceiling | pion | proton |",
             "|---|---|---|---|---|"]
    for run in runs:
        cands = [r for r in rows if r["run_id"] == run
                 and r["images_seen"] <= latest_common]
        if not cands:
            continue
        r = max(cands, key=lambda r: r["images_seen"])
        lines.append(f"| {run} | {r['best_mIoU']:.4f} "
                     f"| {r['frac_of_ceiling']:.3f} "
                     f"| {r['iou_pion']:.3f} | {r['iou_proton']:.3f} |")
    with open(os.path.join(outdir, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
