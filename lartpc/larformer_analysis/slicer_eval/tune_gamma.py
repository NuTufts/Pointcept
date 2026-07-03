"""Photon-library γ tuning from per-event analysis H5s — no rerun needed.

`analyze_event.py` ran with γ_beam = γ_cosmic = 1.0 by default; the
recorded `pe_pred` values therefore obey

    pe_pred_recorded[pmt] = readout_factor × Σ_sp (q_sp × visibility[sp,pmt])

i.e., linear in γ. So at any other γ_new,

    pe_pred[pmt; γ_new] = γ_new × pe_pred_recorded[pmt]

— meaning total_PE scales linearly too, and a single γ estimate falls
out of comparing observed in-time-flash total PE against the recorded
total predicted PE.

This script reads a directory of `perevent_*.h5` files, builds three
view-pairs:

  (A) GT-nu mask → predicted PE                 (truth-side reference)
  (B) M1 panoptic nu slice → predicted PE       (model's chosen slice)
  (C) Sum of ALL model-labeled-nu slices → PE   (multi-slice fallback;
                                                  matches what an
                                                  analyzer would do
                                                  without a tiebreaker)

For each view it reports several γ estimators and produces plots:

  scatter:   total_pe_obs vs total_pe_pred (colored by category)
  histogram: ratio pe_obs / pe_pred (the per-event γ estimate)
  histogram: total_pe_obs by itself

Estimators reported (each robust against different failure modes):
    γ_median        — median(pe_obs_total / pe_pred_total)
    γ_mean          — mean of the same ratios
    γ_ratio_of_sums — Σ pe_obs_total / Σ pe_pred_total
                      (de-weights small events; most robust)
    γ_lsq_through_0 — least-squares slope through origin
                      pe_obs = γ · pe_pred  (per-event weights from
                      pe_pred itself; uses every PMT, not the totals)

A quick rule of thumb: if γ_median, γ_mean, γ_ratio_of_sums all agree
within ~5%, you can take any. If γ_median deviates a lot from γ_lsq
something is wrong (a tail of failed events; check the ratio
histogram).

Usage:
  python tune_gamma.py \\
      --perevent-dir /path/to/analysis/<TAG>/ \\
      --output-dir   /path/to/gamma_tune/<TAG>/ \\
      [--min-pe-obs 50]            # drop events with obs PE below this
      [--min-pe-pred 1]            # drop events with pred PE below this
      [--per-pmt-fit]              # also fit using per-PMT pairs, not
                                    # just per-event totals

Pure h5py + numpy + matplotlib. Safe to run outside the container.
"""

import argparse
import glob
import os
import sys

import h5py
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from lartpc.larformer_analysis.slicer_eval.lib import categorize  # noqa: E402


def _safe_log10(x, floor=1e-6):
    return np.log10(np.maximum(np.asarray(x, dtype=np.float64), floor))


def _gamma_estimators(pe_obs_total, pe_pred_total):
    """Return a dict of γ estimators on per-event totals.

    All assume `pe_pred_total` was computed with γ=1.0.
    """
    out = dict(n_events=int(pe_obs_total.size))
    if pe_obs_total.size == 0:
        return dict(out, gamma_median=float("nan"),
                    gamma_mean=float("nan"),
                    gamma_ratio_of_sums=float("nan"))
    ratio = pe_obs_total / np.maximum(pe_pred_total, 1e-9)
    finite = np.isfinite(ratio)
    if not finite.any():
        return dict(out, gamma_median=float("nan"),
                    gamma_mean=float("nan"),
                    gamma_ratio_of_sums=float("nan"))
    out["gamma_median"]        = float(np.median(ratio[finite]))
    out["gamma_mean"]          = float(np.mean(ratio[finite]))
    sum_obs = float(pe_obs_total[finite].sum())
    sum_pred = float(pe_pred_total[finite].sum())
    out["gamma_ratio_of_sums"] = sum_obs / sum_pred if sum_pred > 0 else float("nan")
    return out


def _per_pmt_lsq_through_zero(pe_obs_arr, pe_pred_arr):
    """Least-squares slope through origin on per-PMT pairs.

    pe_obs_arr / pe_pred_arr are flat (N_events * N_pmts,) arrays.
    Returns γ such that pe_obs ≈ γ · pe_pred minimizing Σ (pe_obs - γ·pe_pred)².
    Closed form: γ = (x·y) / (x·x).
    """
    x = np.asarray(pe_pred_arr, dtype=np.float64).ravel()
    y = np.asarray(pe_obs_arr, dtype=np.float64).ravel()
    m = np.isfinite(x) & np.isfinite(y) & (x > 0)
    if not m.any():
        return float("nan")
    x = x[m]; y = y[m]
    denom = float(np.dot(x, x))
    if denom <= 0:
        return float("nan")
    return float(np.dot(x, y) / denom)


def load_perevent_dir(perevent_dir, glob_pat="perevent_*.h5"):
    """Walk per-event H5s, extract the per-event totals (and per-PMT vectors)
    we need to tune γ. Returns a flat dict of (N,) / (N, 32) arrays."""
    paths = sorted(glob.glob(os.path.join(perevent_dir, glob_pat)))
    if not paths:
        sys.exit(f"no files matched {perevent_dir}/{glob_pat}")
    N = len(paths)
    pe_obs        = np.zeros((N, 32), dtype=np.float32)
    pe_pred_gt    = np.zeros((N, 32), dtype=np.float32)
    pe_pred_m1    = np.full((N, 32), np.nan, dtype=np.float32)
    pe_pred_nu_sum = np.full((N, 32), np.nan, dtype=np.float32)
    category_mask = np.zeros(N, dtype=np.uint8)
    has_nu_pred   = np.zeros(N, dtype=bool)
    n_pred_nu     = np.zeros(N, dtype=np.int32)
    n_nu_gt       = np.zeros(N, dtype=np.int32)
    run    = np.zeros(N, dtype=np.int64)
    subrun = np.zeros(N, dtype=np.int64)
    event  = np.zeros(N, dtype=np.int64)

    for i, p in enumerate(paths):
        with h5py.File(p, "r") as f:
            run[i]    = int(f.attrs["run"])
            subrun[i] = int(f.attrs["subrun"])
            event[i]  = int(f.attrs["event"])
            category_mask[i] = int(f["truth"].attrs.get("category_mask", 0))
            has_nu_pred[i]   = bool(f.attrs.get("has_nu_prediction", False))
            pe_obs[i, :]     = f["in_time_flash/pe_obs"][:]
            pe_pred_gt[i, :] = f["gt_baseline/pe_pred"][:]
            ps = f["pred_slices"]
            n_pred_nu[i] = int(ps.attrs.get("n_pred_nu_slices", 0))
            if "overclaim" in f:
                n_nu_gt[i] = int(
                    f["overclaim"].attrs.get("n_nu_gt_instances", 0)
                )
            # M1 (panoptic-view) slice pe_pred
            m1_qid = int(f["metrics"].attrs.get("m1_slice_id", -1))
            qids = ps["query_id"][:]
            classes = ps["class_argmax"][:]
            pe_pred_per_slice = ps["pe_pred"][:]
            if m1_qid >= 0:
                row = np.flatnonzero(qids == m1_qid)
                if row.size > 0:
                    pe_pred_m1[i, :] = pe_pred_per_slice[int(row[0])]
            # Sum across all model-labeled-nu slices
            nu_rows = np.flatnonzero(classes == 0)
            if nu_rows.size > 0:
                pe_pred_nu_sum[i, :] = pe_pred_per_slice[nu_rows].sum(axis=0)
    return dict(
        paths=paths, run=run, subrun=subrun, event=event,
        category_mask=category_mask,
        has_nu_pred=has_nu_pred,
        n_pred_nu=n_pred_nu, n_nu_gt=n_nu_gt,
        pe_obs=pe_obs, pe_pred_gt=pe_pred_gt,
        pe_pred_m1=pe_pred_m1, pe_pred_nu_sum=pe_pred_nu_sum,
    )


# -----------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------

def _save(fig, output_dir, stem):
    fig.savefig(os.path.join(output_dir, f"{stem}.png"),
                bbox_inches="tight", dpi=130)
    fig.savefig(os.path.join(output_dir, f"{stem}.pdf"), bbox_inches="tight")
    plt.close(fig)


def _scatter_obs_vs_pred(pe_obs_total, pe_pred_total, label, gamma_est,
                         output_dir, stem):
    if pe_obs_total.size == 0:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    # linear
    ax = axes[0]
    ax.scatter(pe_pred_total, pe_obs_total, s=8, alpha=0.4)
    lo = 1.0
    hi = max(1.0, float(np.nanmax(pe_pred_total) * 1.05))
    xs = np.array([lo, hi])
    for est in ("median", "ratio_of_sums", "lsq"):
        g = gamma_est.get(f"gamma_{est}", float("nan"))
        if np.isfinite(g):
            ax.plot(xs, g * xs, label=f"γ_{est}={g:.3g}", linewidth=1.0)
    ax.set_xlabel("predicted total PE  (γ = 1.0 at analysis time)")
    ax.set_ylabel("observed total PE (in-time flash)")
    ax.set_title(f"{label} — linear")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    # log-log
    ax = axes[1]
    ax.scatter(pe_pred_total, pe_obs_total, s=8, alpha=0.4)
    for est in ("median", "ratio_of_sums", "lsq"):
        g = gamma_est.get(f"gamma_{est}", float("nan"))
        if np.isfinite(g):
            ax.plot(xs, g * xs, label=f"γ_{est}={g:.3g}", linewidth=1.0)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("predicted total PE (γ = 1.0)")
    ax.set_ylabel("observed total PE")
    ax.set_title(f"{label} — log-log")
    ax.legend(fontsize=7)
    ax.grid(which="both", alpha=0.3)
    fig.tight_layout()
    _save(fig, output_dir, stem)


def _ratio_hist(pe_obs_total, pe_pred_total, label, gamma_est,
                output_dir, stem, log_bins=True):
    if pe_obs_total.size == 0:
        return
    ratio = pe_obs_total / np.maximum(pe_pred_total, 1e-9)
    finite = np.isfinite(ratio) & (ratio > 0)
    if not finite.any():
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    r = ratio[finite]
    if log_bins:
        bins = np.logspace(np.log10(max(1e-3, r.min())),
                           np.log10(r.max()), 60)
        ax.set_xscale("log")
    else:
        bins = 60
    ax.hist(r, bins=bins, edgecolor="black", alpha=0.85)
    for est, color in (("median", "tab:red"),
                       ("ratio_of_sums", "tab:green"),
                       ("lsq", "tab:purple")):
        g = gamma_est.get(f"gamma_{est}", float("nan"))
        if np.isfinite(g):
            ax.axvline(g, color=color, linewidth=1.2,
                       label=f"γ_{est}={g:.3g}")
    ax.set_xlabel("pe_obs_total / pe_pred_total  (= per-event γ estimate)")
    ax.set_ylabel("events")
    ax.set_title(f"{label}  n={int(r.size)}  "
                 f"med={float(np.median(r)):.3g}  "
                 f"p10={float(np.percentile(r, 10)):.3g}  "
                 f"p90={float(np.percentile(r, 90)):.3g}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, output_dir, stem)


def _total_pe_obs_hist(pe_obs_total, output_dir, stem="pe_obs_hist"):
    if pe_obs_total.size == 0:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    finite = np.isfinite(pe_obs_total) & (pe_obs_total > 0)
    v = pe_obs_total[finite]
    bins = np.logspace(np.log10(max(1.0, v.min())), np.log10(v.max()), 60)
    ax.hist(v, bins=bins, edgecolor="black", alpha=0.85)
    ax.set_xscale("log")
    ax.set_xlabel("total observed PE (in-time flash)")
    ax.set_ylabel("events")
    ax.set_title(f"total pe_obs  n={int(v.size)}  "
                 f"med={float(np.median(v)):.1f}  "
                 f"p10={float(np.percentile(v, 10)):.1f}  "
                 f"p90={float(np.percentile(v, 90)):.1f}")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    _save(fig, output_dir, stem)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("--perevent-dir", required=True,
                    help="Directory of perevent_*.h5")
    ap.add_argument("--output-dir",   required=True,
                    help="Where to write plots + summary.txt")
    ap.add_argument("--glob", default="perevent_*.h5")
    ap.add_argument("--min-pe-obs",  type=float, default=10.0,
                    help="Drop events where total pe_obs < this. Default 10.")
    ap.add_argument("--min-pe-pred", type=float, default=0.1,
                    help="Drop events where total pe_pred < this. Default 0.1.")
    ap.add_argument("--per-pmt-fit", action="store_true",
                    help="Also fit γ on per-PMT pairs (not just per-event totals)")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading per-event H5s from {args.perevent_dir}…")
    es = load_perevent_dir(args.perevent_dir, glob_pat=args.glob)
    N = es["pe_obs"].shape[0]
    print(f"  loaded N={N} events")

    # Per-event totals
    pe_obs_tot      = es["pe_obs"].sum(axis=-1).astype(np.float64)
    pe_pred_gt_tot  = es["pe_pred_gt"].sum(axis=-1).astype(np.float64)
    pe_pred_m1_tot  = es["pe_pred_m1"].sum(axis=-1).astype(np.float64)
    pe_pred_nu_tot  = es["pe_pred_nu_sum"].sum(axis=-1).astype(np.float64)

    # Quality cut for the fit pool
    base_ok = (pe_obs_tot >= args.min_pe_obs) & np.isfinite(pe_obs_tot)
    print(f"  events passing pe_obs >= {args.min_pe_obs}: "
          f"{int(base_ok.sum())} / {N}")

    summary_lines = [
        f"# γ tuning summary",
        f"# perevent_dir = {args.perevent_dir}",
        f"# N total events = {N}",
        f"# min_pe_obs = {args.min_pe_obs}  min_pe_pred = {args.min_pe_pred}",
        "",
    ]

    # ----- Three view-pairs ------------------------------------------
    for label, key, pe_pred_tot, extra_ok in (
        ("(A) GT-nu baseline",       "A_gt_baseline",  pe_pred_gt_tot,
         np.ones(N, dtype=bool)),
        ("(B) M1 panoptic-nu slice", "B_m1_panoptic", pe_pred_m1_tot,
         es["has_nu_pred"]),
        ("(C) Σ all model-nu slices","C_nu_sum",       pe_pred_nu_tot,
         es["has_nu_pred"]),
    ):
        ok = base_ok & extra_ok & (pe_pred_tot >= args.min_pe_pred)
        sub_obs  = pe_obs_tot[ok]
        sub_pred = pe_pred_tot[ok]

        est = _gamma_estimators(sub_obs, sub_pred)

        # Per-PMT LSQ (optional).
        if args.per_pmt_fit:
            if key == "A_gt_baseline":
                pe_pred_per_pmt = es["pe_pred_gt"]
            elif key == "B_m1_panoptic":
                pe_pred_per_pmt = es["pe_pred_m1"]
            else:
                pe_pred_per_pmt = es["pe_pred_nu_sum"]
            est["gamma_lsq"] = _per_pmt_lsq_through_zero(
                es["pe_obs"][ok], pe_pred_per_pmt[ok],
            )

        # Output
        print(f"\n=== {label} ===")
        print(f"  n_events used: {est['n_events']}")
        for k in ("gamma_median", "gamma_mean",
                  "gamma_ratio_of_sums", "gamma_lsq"):
            if k in est:
                print(f"  {k:<24s} = {est[k]:.4g}")

        summary_lines.append(f"=== {label} ===")
        summary_lines.append(f"  n_events_used = {est['n_events']}")
        for k in ("gamma_median", "gamma_mean",
                  "gamma_ratio_of_sums", "gamma_lsq"):
            if k in est:
                summary_lines.append(f"  {k:<24s} = {est[k]:.4g}")
        summary_lines.append("")

        # Plots
        _scatter_obs_vs_pred(
            sub_obs, sub_pred, label, est,
            args.output_dir, f"scatter_{key}",
        )
        _ratio_hist(
            sub_obs, sub_pred, label, est,
            args.output_dir, f"ratio_hist_{key}",
        )

    # pe_obs (independent of any pe_pred view)
    _total_pe_obs_hist(pe_obs_tot[base_ok], args.output_dir)

    # Write summary
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"\nPlots + summary written to {args.output_dir}")


if __name__ == "__main__":
    main()
