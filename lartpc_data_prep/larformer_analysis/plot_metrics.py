"""Stage 5: matplotlib plots from event_summary.h5 + category_summary.h5.

Produces PDF + PNG plots:

  m1_iou_hist_<cat>.{pdf,png}
      Histogram of M1 IoU (best-nu-pred IoU vs GT-nu mask) per category.
      One panel per category; events with has_nu_prediction=False land in
      the 0-bin.

  m3_delta_chi2_box_<cat>.{pdf,png}
      Box-plot of Δχ² = chi2_nu - chi2_gt per OOB threshold, per category.
      Negative Δχ² = model's nu prediction is a BETTER flash match than
      the GT-truth slice (suspicious, usually means OOB filter dropped
      the GT). Positive Δχ² = expected (GT-truth is a better match).

  m4_rank1_frac.{pdf,png}
      Bar chart: per category, per OOB threshold, fraction of events
      where the GT-best-match predicted slice ranks #1 by chi-2.
      Side-by-side bars for ALL pool vs nu-only pool. The ALL pool
      shows how informative chi-2 is when ignoring class; nu pool
      shows the joint quality of class + chi-2.

  headline_table.txt
      ASCII table of per-category counts + headline scalars at the
      default OOB threshold.

Usage:
  python plot_metrics.py \\
      --event-summary    /path/to/event_summary.h5 \\
      --category-summary /path/to/category_summary.h5 \\
      --output-dir       /path/to/plots/

If neither --no-pdf nor --no-png is set, both formats are written.
"""

import argparse
import os

import h5py
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_event_summary(p):
    with h5py.File(p, "r") as f:
        out = dict(
            oob_thresholds  = np.asarray(f.attrs["oob_thresholds"],
                                         dtype=np.float32),
            default_oob_idx = int(f.attrs["default_oob_idx"]),
            category_names  = [
                n.decode() if isinstance(n, bytes) else str(n)
                for n in f.attrs["category_names"]
            ],
            model_tag       = str(f.attrs.get("model_tag", "unknown")),
            n_events        = int(f.attrs["n_events"]),
        )
        e = f["events"]
        out["run"]    = e["run"][:]
        out["subrun"] = e["subrun"][:]
        out["event"]  = e["event"][:]
        out["category_mask"]     = e["category_mask"][:]
        out["has_nu_prediction"] = e["has_nu_prediction"][:]
        out["m1_iou"]            = e["m1_iou"][:]
        out["m1_slice_id"]       = e["m1_slice_id"][:]
        out["m1_iou_intent"]     = (e["m1_iou_intent"][:]
                                    if "m1_iou_intent" in e
                                    else np.zeros(out["n_events"], np.float32))
        out["m3_chi2_gt"]        = e["m3_chi2_gt"][:]
        out["m3_chi2_nu"]        = e["m3_chi2_nu"][:]
        out["m3_delta_chi2"]     = e["m3_delta_chi2"][:]
        out["m4_rank_all"]       = e["m4_rank_all"][:]
        out["m4_rank_nu"]        = e["m4_rank_nu"][:]
        out["n_pred_slices"]     = e["n_pred_slices"][:]
        out["n_pred_nu_slices"]  = e["n_pred_nu_slices"][:]
        out["sp_level_nu_recall"]    = (
            e["sp_level_nu_recall"][:]
            if "sp_level_nu_recall" in e
            else np.full(out["n_events"], np.nan, np.float32))
        out["sp_level_nu_precision"] = (
            e["sp_level_nu_precision"][:]
            if "sp_level_nu_precision" in e
            else np.full(out["n_events"], np.nan, np.float32))
        out["n_nu_gt_instances"]     = (
            e["n_nu_gt_instances"][:]
            if "n_nu_gt_instances" in e
            else np.zeros(out["n_events"], np.int32))
        out["n_nu_gt_matched_to_nu"] = (
            e["n_nu_gt_matched_to_nu"][:]
            if "n_nu_gt_matched_to_nu" in e
            else np.zeros(out["n_events"], np.int32))
        out["n_nu_match_survived"]   = (
            e["n_nu_match_survived"][:]
            if "n_nu_match_survived" in e
            else np.zeros(out["n_events"], np.int32))
        # nu_pairs/ (flat per-pair arrays across events)
        if "nu_pairs" in f:
            np_g = f["nu_pairs"]
            out["pair_event_idx"]      = np_g["event_idx"][:]
            out["pair_category_mask"]  = np_g["category_mask"][:]
            out["pair_iou"]            = np_g["pair_iou"][:]
            out["pair_argmax_iou"]     = np_g["argmax_iou"][:]
            out["pair_overclaim_gap"]  = np_g["overclaim_gap"][:]
            out["pair_matched_class"]  = np_g["matched_class"][:]
        else:
            out["pair_event_idx"]      = np.zeros(0, np.int32)
            out["pair_category_mask"]  = np.zeros(0, np.uint8)
            out["pair_iou"]            = np.zeros(0, np.float32)
            out["pair_argmax_iou"]     = np.zeros(0, np.float32)
            out["pair_overclaim_gap"]  = np.zeros(0, np.float32)
            out["pair_matched_class"]  = np.zeros(0, np.int32)
    return out


def category_event_masks(es):
    """Return {category_name: bool mask over events}."""
    cats = {"all": np.ones(es["n_events"], dtype=bool)}
    for ci, name in enumerate(es["category_names"]):
        bit = 1 << ci
        cats[name] = (es["category_mask"] & bit) != 0
    return cats


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def _save(fig, output_dir, stem, want_pdf, want_png):
    if want_pdf:
        fig.savefig(os.path.join(output_dir, f"{stem}.pdf"), bbox_inches="tight")
    if want_png:
        fig.savefig(os.path.join(output_dir, f"{stem}.png"),
                    bbox_inches="tight", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_m1_iou_hist(es, cats, output_dir, want_pdf, want_png):
    names = [n for n in cats.keys() if cats[n].any()]
    n_cat = len(names)
    if n_cat == 0:
        return
    n_cols = min(3, n_cat)
    n_rows = (n_cat + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows),
                             squeeze=False)
    bins = np.linspace(0, 1, 21)
    for i, name in enumerate(names):
        ax = axes[i // n_cols][i % n_cols]
        m = cats[name]
        vals = es["m1_iou"][m]
        has_nu = es["has_nu_prediction"][m]
        ax.hist(vals, bins=bins, edgecolor="black", alpha=0.85)
        n = int(m.sum())
        n_with = int(has_nu.sum())
        ax.set_title(f"{name}  n={n}  ({n_with} w/ nu-pred)")
        ax.set_xlabel("M1 IoU (best-nu-pred vs GT-nu)")
        ax.set_ylabel("events")
        ax.set_xlim(0, 1)
    # Hide spare axes.
    for j in range(n_cat, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")
    fig.suptitle(f"M1: best-nu-pred IoU vs GT-nu  ({es['model_tag']})")
    fig.tight_layout()
    _save(fig, output_dir, "m1_iou_hist", want_pdf, want_png)


def plot_m3_delta_chi2_box(es, cats, output_dir, want_pdf, want_png):
    """Box plot of Δχ² per category, per OOB threshold."""
    th = es["oob_thresholds"]
    n_th = len(th)
    names = [n for n in cats.keys() if cats[n].any()]
    n_cat = len(names)
    if n_cat == 0:
        return
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * n_cat * n_th), 4))
    positions = []
    data = []
    labels = []
    colors = []
    cmap = plt.get_cmap("tab10")
    for ci, name in enumerate(names):
        m = cats[name]
        for ti in range(n_th):
            col = es["m3_delta_chi2"][m, ti]
            col = col[np.isfinite(col)]
            data.append(col)
            positions.append(ci * (n_th + 1) + ti)
            labels.append(f"{name}\nthr={float(th[ti]):.2f}")
            colors.append(cmap(ti % 10))
    if any(len(d) > 0 for d in data):
        bp = ax.boxplot(
            data, positions=positions, widths=0.7, patch_artist=True,
            showfliers=False,
        )
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c); patch.set_alpha(0.5)
    ax.axhline(0, color="red", linewidth=0.7)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel(r"$\Delta\chi^2 = \chi^2_{\rm pred-nu} - \chi^2_{\rm GT-nu}$")
    ax.set_title(f"M3 Δχ² per category × OOB threshold  ({es['model_tag']})")
    fig.tight_layout()
    _save(fig, output_dir, "m3_delta_chi2_box", want_pdf, want_png)


def plot_m4_rank1_frac(es, cats, output_dir, want_pdf, want_png):
    """Bar chart of rank-1 fraction (M4) per category × OOB threshold,
    one cluster of bars per (pool ∈ {all, nu})."""
    th = es["oob_thresholds"]
    n_th = len(th)
    names = [n for n in cats.keys() if cats[n].any()]
    n_cat = len(names)
    if n_cat == 0:
        return

    def _rank1(col):
        col = np.asarray(col)
        denom = int((col >= 1).sum())
        if denom == 0:
            return float("nan")
        return float((col == 1).sum()) / denom

    fig, axes = plt.subplots(1, 2, figsize=(max(10, 1.5 * n_cat * n_th), 4),
                             sharey=True)
    cmap = plt.get_cmap("viridis")
    for pi, (pool, key) in enumerate((
        ("all queries", "m4_rank_all"),
        ("nu-only queries", "m4_rank_nu"),
    )):
        ax = axes[pi]
        bar_width = 0.8 / n_th
        for ti in range(n_th):
            fracs = []
            for name in names:
                m = cats[name]
                fracs.append(_rank1(es[key][m, ti]))
            xpos = np.arange(n_cat) + ti * bar_width
            ax.bar(xpos, fracs, width=bar_width,
                   color=cmap(ti / max(1, n_th - 1)),
                   label=f"thr={float(th[ti]):.2f}",
                   edgecolor="black", linewidth=0.5)
        ax.set_xticks(np.arange(n_cat) + (n_th - 1) * bar_width / 2)
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("rank-1 fraction")
        ax.set_title(f"M4 pool: {pool}")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")
    fig.suptitle(f"M4: GT-best-match rank-1 fraction by category  "
                 f"({es['model_tag']})")
    fig.tight_layout()
    _save(fig, output_dir, "m4_rank1_frac", want_pdf, want_png)


def plot_m1_panoptic_vs_intent(es, cats, output_dir, want_pdf, want_png):
    """Side-by-side bars per category: M1 IoU (panoptic) vs M1 IoU (intent).

    Surfaces the over-claim gap: when intent >> panoptic, the model's
    class-correct nu query lost the panoptic argmax. When panoptic >>
    intent, Hungarian matching picked the wrong query for the nu GT.
    """
    names = [n for n in cats.keys() if cats[n].any()]
    if not names:
        return
    pan_means    = []
    intent_means = []
    counts       = []
    for name in names:
        m = cats[name]
        # Only average over events with at least one nu GT.
        valid = m & (es["n_nu_gt_instances"] > 0)
        if not valid.any():
            pan_means.append(0.0); intent_means.append(0.0)
            counts.append(0); continue
        pan_means.append(float(es["m1_iou"][valid].mean()))
        intent_means.append(float(es["m1_iou_intent"][valid].mean()))
        counts.append(int(valid.sum()))
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(names)), 4))
    x = np.arange(len(names))
    w = 0.4
    b1 = ax.bar(x - w/2, pan_means, w, label="M1 panoptic (analyzer view)",
                color="tab:blue",  edgecolor="black", linewidth=0.5)
    b2 = ax.bar(x + w/2, intent_means, w, label="M1 intent (matched-query pair IoU)",
                color="tab:orange", edgecolor="black", linewidth=0.5)
    for xi, (p, i, n) in enumerate(zip(pan_means, intent_means, counts)):
        ax.text(xi, max(p, i) + 0.03, f"n={n}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("mean IoU vs GT-nu")
    ax.set_title(f"M1 panoptic vs intent  ({es['model_tag']})  "
                 f"— gap = over-claim impact")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, output_dir, "m1_panoptic_vs_intent", want_pdf, want_png)


def plot_overclaim_gap_hist(es, cats, output_dir, want_pdf, want_png):
    """Per-nu-GT histogram of (pair_iou - argmax_iou) per category.

    Negative values = over-claim (matched query lost SPs in panoptic
    that it would have wanted). Zero = no contention. Positive (rare)
    = panoptic argmax happens to be a tighter slice than the pre-arg
    above-threshold mask (mostly numerical edge cases).

    The histogram is the per-event view of what `tools/measure_overclaim
    .py` reports across an inference output dir — here split per
    analysis category.
    """
    pair_iou      = es.get("pair_iou", np.zeros(0))
    pair_arg_iou  = es.get("pair_argmax_iou", np.zeros(0))
    pair_gap      = es.get("pair_overclaim_gap", np.zeros(0))
    pair_cat_mask = es.get("pair_category_mask", np.zeros(0, np.uint8))
    if pair_iou.size == 0:
        return
    names = [n for n in cats.keys() if cats[n].any()]
    n_cat = len(names)
    n_cols = min(3, n_cat)
    n_rows = (n_cat + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4 * n_cols, 3 * n_rows), squeeze=False)
    bins = np.linspace(-0.5, 0.2, 36)
    for i, name in enumerate(names):
        ax = axes[i // n_cols][i % n_cols]
        if name == "all":
            sel = np.ones_like(pair_gap, dtype=bool)
        else:
            try:
                ci = es["category_names"].index(name)
            except ValueError:
                ax.axis("off"); continue
            bit = 1 << ci
            sel = (pair_cat_mask & bit) != 0
        sub = pair_gap[sel]
        ax.hist(sub, bins=bins, edgecolor="black", alpha=0.85)
        ax.axvline(0, color="red", linewidth=0.7)
        n = int(sub.size)
        mean = float(sub.mean()) if n else 0.0
        median = float(np.median(sub)) if n else 0.0
        ax.set_title(f"{name}  n_pairs={n}  mean={mean:+.3f}  med={median:+.3f}")
        ax.set_xlabel(r"pair_iou $-$ argmax_iou (per nu GT)")
        ax.set_ylabel("count")
    for j in range(n_cat, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")
    fig.suptitle(f"Over-claim gap per nu-GT  ({es['model_tag']})  "
                 f"— negative = matched query lost SPs in panoptic")
    fig.tight_layout()
    _save(fig, output_dir, "overclaim_gap_hist", want_pdf, want_png)


def plot_sp_level_nu_recall(es, cats, output_dir, want_pdf, want_png):
    """Per-category histogram of `sp_level_nu_recall` — the analyzer-
    facing 'fraction of true-nu SPs labeled nu by panoptic argmax'.

    1.0 = panoptic-argmax labels every nu SP nu. 0.0 = the nu prediction
    is entirely missing in the panoptic view. Paired with
    `frac_nu_match_survived_panoptic` in the headline table to
    distinguish 'model didn't predict nu' from 'model predicted nu
    but lost the per-SP competition'.
    """
    names = [n for n in cats.keys() if cats[n].any()]
    if not names:
        return
    n_cat = len(names)
    n_cols = min(3, n_cat)
    n_rows = (n_cat + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4 * n_cols, 3 * n_rows), squeeze=False)
    bins = np.linspace(0, 1, 21)
    for i, name in enumerate(names):
        ax = axes[i // n_cols][i % n_cols]
        m = cats[name]
        valid = m & (es["n_nu_gt_instances"] > 0)
        sub = es["sp_level_nu_recall"][valid]
        sub = sub[np.isfinite(sub)]
        ax.hist(sub, bins=bins, edgecolor="black", alpha=0.85)
        n = int(sub.size)
        mean = float(sub.mean()) if n else float("nan")
        ax.set_title(f"{name}  n={n}  mean={mean:.3f}")
        ax.set_xlabel("sp_level_nu_recall")
        ax.set_ylabel("events")
        ax.set_xlim(0, 1)
    for j in range(n_cat, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")
    fig.suptitle(f"SP-level nu recall per category  ({es['model_tag']})")
    fig.tight_layout()
    _save(fig, output_dir, "sp_level_nu_recall_hist", want_pdf, want_png)


def write_headline_table(es, cats, output_dir):
    th = es["oob_thresholds"]
    default_idx = es["default_oob_idx"]
    th_default = float(th[default_idx])

    def _safe(arr):
        v = np.asarray(arr); v = v[np.isfinite(v)]
        return float(v.mean()) if v.size else float("nan")

    def _rank1_frac(col):
        col = np.asarray(col)
        denom = int((col >= 1).sum())
        if denom == 0:
            return float("nan")
        return float((col == 1).sum()) / denom

    lines = [
        f"# Headline metrics  ({es['model_tag']})",
        f"# n_events_total = {es['n_events']}",
        f"# default OOB threshold = {th_default:.2f}",
        "",
        f"{'category':<22s} {'n':>5s}  {'nu_pred%':>8s}  "
        f"{'iou_pan':>8s}  {'iou_int':>8s}  "
        f"{'sp_rec':>7s}  {'surv%':>6s}  "
        f"{'mean_dchi2':>10s}  {'rank1_all':>9s}  {'rank1_nu':>9s}",
        "-" * 118,
    ]
    for name, m in cats.items():
        n = int(m.sum())
        if n == 0:
            lines.append(f"{name:<22s} {n:>5d}")
            continue
        nu_pred_frac = float(es["has_nu_prediction"][m].mean())
        mean_iou_pan = float(es["m1_iou"][m].mean())
        mean_iou_int = float(es["m1_iou_intent"][m].mean())
        sp_rec = _safe(es["sp_level_nu_recall"][m])
        denom = int(es["n_nu_gt_matched_to_nu"][m].sum())
        num   = int(es["n_nu_match_survived"][m].sum())
        surv_frac = (float(num) / float(denom)) if denom > 0 else float("nan")
        mean_dchi2 = _safe(es["m3_delta_chi2"][m, default_idx])
        rank1_all = _rank1_frac(es["m4_rank_all"][m, default_idx])
        rank1_nu  = _rank1_frac(es["m4_rank_nu"][m,  default_idx])
        lines.append(
            f"{name:<22s} {n:>5d}  {nu_pred_frac:>7.1%}  "
            f"{mean_iou_pan:>8.3f}  {mean_iou_int:>8.3f}  "
            f"{sp_rec:>7.3f}  {surv_frac:>5.1%}  "
            f"{mean_dchi2:>+10.2f}  {rank1_all:>8.1%}  {rank1_nu:>8.1%}"
        )
    text = "\n".join(lines) + "\n"
    with open(os.path.join(output_dir, "headline_table.txt"), "w") as f:
        f.write(text)
    print(text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--event-summary",    required=True,
                    help="event_summary.h5 from aggregate_metrics.py")
    ap.add_argument("--category-summary", default=None,
                    help="(optional, unused for now) category_summary.h5")
    ap.add_argument("--output-dir",       required=True)
    ap.add_argument("--no-pdf", action="store_true")
    ap.add_argument("--no-png", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    es = load_event_summary(args.event_summary)
    cats = category_event_masks(es)
    print(f"Loaded {args.event_summary}: N={es['n_events']}  "
          f"model_tag={es['model_tag']}")
    print(f"  category counts: " + ", ".join(
        f"{name}={int(m.sum())}" for name, m in cats.items()
    ))
    plot_m1_iou_hist(es, cats, args.output_dir,
                     not args.no_pdf, not args.no_png)
    plot_m1_panoptic_vs_intent(es, cats, args.output_dir,
                               not args.no_pdf, not args.no_png)
    plot_m3_delta_chi2_box(es, cats, args.output_dir,
                           not args.no_pdf, not args.no_png)
    plot_m4_rank1_frac(es, cats, args.output_dir,
                       not args.no_pdf, not args.no_png)
    plot_sp_level_nu_recall(es, cats, args.output_dir,
                            not args.no_pdf, not args.no_png)
    plot_overclaim_gap_hist(es, cats, args.output_dir,
                            not args.no_pdf, not args.no_png)
    write_headline_table(es, cats, args.output_dir)
    print(f"plots written to {args.output_dir}")


if __name__ == "__main__":
    main()
