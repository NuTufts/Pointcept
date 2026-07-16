#!/usr/bin/env python3
"""
Stage-2b of the input-distribution study (P05F): tune the asinh scale.

Why this exists: stage 2 compares transform *families* at fixed parameters,
and its `make_t_asinh` default (scale=25, xmax=2e4) was a guess, not a fit.
Two corrections come out of tuning it, and both change what gets implemented:

  1. xmax should be 1000, not 2e4. Only 0.19% of pixels exceed 1000 ADC, so
     an asinh with xmax=2e4 spends ~40% of its bounded output range on that
     0.19% -- the bulk of the data is squeezed into [-1, 0.19]. Since the
     noise-aware d' penalizes width relative to a FIXED augmentation sigma,
     wasting output range costs real separation. xmax=1000 also matches the
     loader's existing clip, so no loader change is needed.

     (Note: applying the loader's clip(0,1000) *before* an xmax=2e4 asinh
     does NOT change d' -- the bulk mapping is identical either way and only
     0.19% of points move. The range waste, not the clip, is the problem.)

  2. scale=50 beats scale=25. The optimum is broad (50-75) and lies where
     the asinh knee sits near the muon MPV.

Run after merge_and_analyze_pixval_hists.py, against the same shard dir:

  python3 sweep_asinh_scale.py --indir <shard dir> [--channel y]

Outputs a table to stdout (and --outdir/asinh_scale_sweep.csv if given).
"""
import argparse
import csv
import glob
import os

import numpy as np

SIGMA_AUG = 0.05    # MultiplicativeRandomJitter sigma, as in stage 2
TRAIN_CLIP = 1000.0
SCALES = [10, 25, 50, 75, 100, 150, 200, 300, 500, 1000]
PAIRS = [("muon", "pion"), ("pion", "proton"), ("muon", "proton")]


def load_merged(indir):
    shards = sorted(glob.glob(os.path.join(indir, "pixval_hists_shard*.npz")))
    assert shards, f"no shard files in {indir}"
    first = np.load(shards[0], allow_pickle=False)
    hists = np.zeros_like(first["hists"])
    for s in shards:
        hists += np.load(s, allow_pickle=False)["hists"]
    edges = first["bin_edges"]
    centers = np.empty(hists.shape[-1])
    centers[1:-1] = np.sqrt(edges[:-1] * edges[1:])
    centers[0], centers[-1] = edges[0], edges[-1]
    return (hists, centers,
            [str(c) for c in first["class_names"]],
            [str(c) for c in first["channels"]],
            len(shards))


def dprime_noise(centers_t, ha, hb):
    def mom(h):
        p = h.astype(np.float64) / max(h.sum(), 1)
        m = (p * centers_t).sum()
        return m, (p * (centers_t - m) ** 2).sum()
    ma, va = mom(ha)
    mb, vb = mom(hb)
    return abs(ma - mb) / np.sqrt(0.5 * (va + vb) + SIGMA_AUG ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--channel", default="y")
    args = ap.parse_args()

    hists, centers, class_names, channels, n_shards = load_merged(args.indir)
    idx = {n: i for i, n in enumerate(class_names)}
    ci = channels.index(args.channel)
    print(f"merged {n_shards} shards; channel={args.channel}, "
          f"noise-aware d' (sigma_aug={SIGMA_AUG})\n")

    def report(name, ct):
        vals = [dprime_noise(ct, hists[idx[a], ci], hists[idx[b], ci])
                for a, b in PAIRS]
        print(f"{name:32s}" + "".join(f"{v:>10.3f}" for v in vals)
              + f"{np.mean(vals):>10.3f}")
        return vals

    header = f"{'transform':32s}" + "".join(
        f"{a[:2]+'-'+b[:2]:>10s}" for a, b in PAIRS) + f"{'mean':>10s}"
    print(header)
    print("-" * len(header))

    rows = []
    for s in SCALES:
        ct = 2 * np.arcsinh(np.clip(centers, 0, TRAIN_CLIP) / s) \
            / np.arcsinh(TRAIN_CLIP / s) - 1
        vals = report(f"asinh scale={s}, xmax=1000", ct)
        rows.append(dict(transform=f"asinh{s}", scale=s, xmax=TRAIN_CLIP,
                         **{f"dprime_{a}_vs_{b}": v
                            for (a, b), v in zip(PAIRS, vals)},
                         mean_dprime=float(np.mean(vals))))

    print("-" * len(header))
    # references
    y0, y1 = np.log10(0.01), np.log10(TRAIN_CLIP + 0.01)
    report("log(current)", 2 * (np.log10(np.clip(centers, 0, TRAIN_CLIP) + 0.01)
                                - y0) / (y1 - y0) - 1)
    cdf = np.cumsum(hists.sum(axis=0)[ci]).astype(np.float64)
    cdf /= cdf[-1]
    report("quantile", 2 * cdf - 1)
    report("asinh scale=25, xmax=2e4 (stage-2)",
           2 * np.arcsinh(np.clip(centers, 0, 2e4) / 25.0)
           / np.arcsinh(2e4 / 25.0) - 1)

    best = max(rows, key=lambda r: r["mean_dprime"])
    print(f"\nbest asinh: scale={best['scale']} "
          f"(mean d'={best['mean_dprime']:.3f})")

    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)
        out = os.path.join(args.outdir, "asinh_scale_sweep.csv")
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
