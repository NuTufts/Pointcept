#!/usr/bin/env python3
"""
Stage 2 (reduce) of the input-distribution study (P05F): merge the per-shard
histograms and compare candidate input transforms on class-separability.

Key methodological point: any MONOTONE transform (log, linear, sqrt,
quantile, ...) leaves ranking-based separability (AUC) and the density
overlap coefficient unchanged. What a transform DOES change is the geometry
the network sees: where class modes land in the bounded input range, how
wide they are relative to the augmentation noise, and how much resolution
is spent on each ADC region. So we report, per class-pair / channel:

  - AUC + overlap (transform-invariant; the ceiling any scalar encoding has)
  - Fisher d' in transformed units (mean separation / pooled width)
  - noise-aware d': same, with the training-time strength jitter
    (MultiplicativeRandomJitter sigma=0.05) added to the pooled width
  - median separation in transformed units and per-class central-80% widths
    (dynamic-range usage)
  - clip fraction at the training clip (1000 ADC) — information destroyed
    BEFORE any transform (relevant for protons / overlaps)

Usage:
  python3 merge_and_analyze_pixval_hists.py --indir <shard dir> --outdir <dir>

Outputs: metrics.csv, clip_fractions.csv, dist_<channel>.png, summary.md
"""
import argparse
import glob
import os

import numpy as np

TRAIN_CLIP = 1000.0
MIN_VAL = 0.01          # LogTransform min_val + loader add_min_pixval
MAX_VAL = 1000.0        # LogTransform max_val
SIGMA_AUG = 0.05        # MultiplicativeRandomJitter sigma (log_space), approx
                        # treated as additive noise in transformed units.

PAIRS = [("muon", "pion"), ("muon", "proton"), ("pion", "proton"),
         ("electron", "gamma")]


# ---------------------------------------------------------------- transforms
def t_log(x):
    """Current training pipeline: clip -> +min -> log10 -> [-1, 1]."""
    y0, y1 = np.log10(MIN_VAL), np.log10(MAX_VAL + MIN_VAL)
    return 2 * (np.log10(np.clip(x, 0, TRAIN_CLIP) + MIN_VAL) - y0) / (y1 - y0) - 1


def t_linear(x):
    return 2 * np.clip(x, 0, TRAIN_CLIP) / TRAIN_CLIP - 1


def t_sqrt(x):
    return 2 * np.sqrt(np.clip(x, 0, TRAIN_CLIP) / TRAIN_CLIP) - 1


def make_t_asinh(scale=25.0, xmax=2e4):
    denom = np.arcsinh(xmax / scale)
    return lambda x: 2 * np.arcsinh(np.clip(x, 0, xmax) / scale) / denom - 1


def make_t_quantile(centers, total_hist):
    """Empirical-CDF (quantile) transform from the GLOBAL distribution."""
    cdf = np.cumsum(total_hist).astype(np.float64)
    cdf /= cdf[-1]
    return lambda x: 2 * np.interp(x, centers, cdf) - 1


# ------------------------------------------------------------------- metrics
def hist_moments(centers_t, h):
    w = h.astype(np.float64)
    if w.sum() == 0:
        return np.nan, np.nan
    p = w / w.sum()
    m = (p * centers_t).sum()
    v = (p * (centers_t - m) ** 2).sum()
    return m, v


def hist_quantile(centers, h, q):
    c = np.cumsum(h).astype(np.float64)
    if c[-1] == 0:
        return np.nan
    return np.interp(q * c[-1], c, centers)


def auc_from_hists(h1, h2):
    """P(x2 > x1) via histogram CDFs (monotone-transform invariant)."""
    p1 = h1.astype(np.float64) / max(h1.sum(), 1)
    p2 = h2.astype(np.float64) / max(h2.sum(), 1)
    cdf1 = np.cumsum(p1)
    # P(X2 > X1) + 0.5 P(tie in same bin)
    auc = (p2 * (cdf1 - 0.5 * p1)).sum()
    return max(auc, 1 - auc)


def overlap_coeff(h1, h2):
    p1 = h1.astype(np.float64) / max(h1.sum(), 1)
    p2 = h2.astype(np.float64) / max(h2.sum(), 1)
    return np.minimum(p1, p2).sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    shards = sorted(glob.glob(os.path.join(args.indir, "pixval_hists_shard*.npz")))
    assert shards, f"no shard files in {args.indir}"
    first = np.load(shards[0], allow_pickle=False)
    edges = first["bin_edges"]
    channels = [str(c) for c in first["channels"]]
    class_names = [str(c) for c in first["class_names"]]
    hists = np.zeros_like(first["hists"])
    clip_counts = np.zeros_like(first["clip_counts"])
    class_counts = np.zeros_like(first["class_counts"])
    n_files = n_bad = n_points = 0
    for s in shards:
        d = np.load(s, allow_pickle=False)
        hists += d["hists"]
        clip_counts += d["clip_counts"]
        class_counts += d["class_counts"]
        n_files += int(d["n_files"]); n_bad += int(d["n_bad_files"])
        n_points += int(d["n_points"])
    print(f"merged {len(shards)} shards: {n_files} files ({n_bad} bad), "
          f"{n_points:,} true points")

    # interior bin centers (geometric mean); under/overflow pinned to edges
    centers = np.empty(hists.shape[-1])
    centers[1:-1] = np.sqrt(edges[:-1] * edges[1:])
    centers[0], centers[-1] = edges[0], edges[-1]

    idx = {n: i for i, n in enumerate(class_names)}
    ch_idx = {n: i for i, n in enumerate(channels)}

    total_by_channel = hists.sum(axis=0)  # (channel, bins) global distribution
    transforms = {
        "log(current)": t_log,
        "linear1000": t_linear,
        "sqrt": t_sqrt,
        "asinh25": make_t_asinh(25.0),
    }

    rows = []
    for ch in ["u", "v", "y", "sum"]:
        ci = ch_idx[ch]
        tset = dict(transforms)
        tset["quantile"] = make_t_quantile(centers, total_by_channel[ci])
        for a, b in PAIRS:
            ha, hb = hists[idx[a], ci], hists[idx[b], ci]
            auc = auc_from_hists(ha, hb)
            ovl = overlap_coeff(ha, hb)
            for tname, tf in tset.items():
                ct = tf(centers)
                ma, va = hist_moments(ct, ha)
                mb, vb = hist_moments(ct, hb)
                pooled = np.sqrt(0.5 * (va + vb))
                dprime = abs(ma - mb) / pooled if pooled > 0 else np.nan
                dnoise = abs(ma - mb) / np.sqrt(0.5 * (va + vb) + SIGMA_AUG**2)
                med_a = tf(np.array([hist_quantile(centers, ha, 0.5)]))[0]
                med_b = tf(np.array([hist_quantile(centers, hb, 0.5)]))[0]
                w80a = (tf(np.array([hist_quantile(centers, ha, 0.9)]))[0]
                        - tf(np.array([hist_quantile(centers, ha, 0.1)]))[0])
                w80b = (tf(np.array([hist_quantile(centers, hb, 0.9)]))[0]
                        - tf(np.array([hist_quantile(centers, hb, 0.1)]))[0])
                rows.append(dict(
                    channel=ch, pair=f"{a}-vs-{b}", transform=tname,
                    auc=auc, overlap=ovl, fisher_dprime=dprime,
                    dprime_with_aug_noise=dnoise,
                    median_sep=abs(med_a - med_b),
                    width80_a=w80a, width80_b=w80b,
                ))

    os.makedirs(args.outdir, exist_ok=True)
    import csv
    with open(os.path.join(args.outdir, "metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    with open(os.path.join(args.outdir, "clip_fractions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "plane", "frac_above_1000ADC"])
        for c, cname in enumerate(class_names):
            for p, pname in enumerate(["u", "v", "y"]):
                tot = hists[c, p].sum()
                w.writerow([cname, pname,
                            clip_counts[c, p] / tot if tot else np.nan])

    # ------------------------------------------------------------- figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plot_classes = ["muon", "pion", "proton", "electron", "gamma"]
    for ch in ["u", "y", "sum"]:
        ci = ch_idx[ch]
        tset = dict(transforms)
        tset["quantile"] = make_t_quantile(centers, total_by_channel[ci])
        fig, axes = plt.subplots(1, len(tset), figsize=(5 * len(tset), 4),
                                 sharey=False)
        for ax, (tname, tf) in zip(axes, tset.items()):
            ct = tf(centers)
            order = np.argsort(ct)
            for cname in plot_classes:
                h = hists[idx[cname], ci].astype(np.float64)
                if h.sum() == 0:
                    continue
                # density in transformed units
                ct_s, h_s = ct[order], h[order]
                dx = np.gradient(ct_s)
                dens = h_s / h.sum() / np.maximum(np.abs(dx), 1e-12)
                ax.plot(ct_s, dens, label=cname, lw=1.2)
            ax.set_title(f"{ch}-plane, {tname}")
            ax.set_xlabel("transformed value")
            ax.set_xlim(-1.05, 1.05)
        axes[0].set_ylabel("density")
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, f"dist_{ch}.png"), dpi=130)
        plt.close(fig)

    # ------------------------------------------------------------- summary
    lines = ["# Input-distribution study (P05F) summary", "",
             f"files: {n_files} ({n_bad} bad), true points: {n_points:,}", "",
             "## Class counts", ""]
    for c, cname in enumerate(class_names):
        lines.append(f"- {cname}: {class_counts[c]:,}")
    lines += ["", "## Clip fractions above 1000 ADC (y-plane)", ""]
    for c, cname in enumerate(class_names):
        tot = hists[c, ch_idx["y"]].sum()
        frac = clip_counts[c, ch_idx["y"]] / tot if tot else float("nan")
        lines.append(f"- {cname}: {frac:.4f}")
    lines += ["", "## Best transform per pair (y-plane, by aug-noise-aware d')", ""]
    for a, b in PAIRS:
        sub = [r for r in rows if r["channel"] == "y"
               and r["pair"] == f"{a}-vs-{b}"]
        best = max(sub, key=lambda r: (r["dprime_with_aug_noise"]
                                       if np.isfinite(r["dprime_with_aug_noise"])
                                       else -1))
        cur = next(r for r in sub if r["transform"] == "log(current)")
        lines.append(
            f"- {a} vs {b}: AUC={best['auc']:.3f} (transform-invariant); "
            f"current log d'={cur['dprime_with_aug_noise']:.3f}; "
            f"best={best['transform']} d'={best['dprime_with_aug_noise']:.3f}")
    with open(os.path.join(args.outdir, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
