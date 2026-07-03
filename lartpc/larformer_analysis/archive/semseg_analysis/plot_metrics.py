"""Read a combined semseg-eval .npz and produce metrics + plots.

Outputs (under --outdir):
    metrics_summary.txt           — per-class accuracy, IoU, recall, precision,
                                     totals, broken down by origin (nu / cosmic
                                     / all).
    confusion_*.png               — confusion matrices, raw + row-normalized,
                                     one panel per origin and an "all" panel.
    iou_per_class.png             — IoU per class, grouped bars (nu vs cosmic
                                     vs all).
    recall_per_class.png          — per-class recall (TP / support_true) split
                                     by origin (this is "per-class accuracy"
                                     as commonly used in semseg).
    score_distributions.png       — softmax-score distribution of the "in
                                     class" column for each true class.
    roc_one_vs_rest.png           — one-vs-rest ROC per class.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from histograms import (  # noqa: E402
    CLASS_NAMES, ORIGIN_NAMES, N_CLASSES, N_ORIGINS,
    confusion_metrics, confusion_origins_sum,
    score_hist_to_roc, trapezoidal_auc,
)


def _fmt(x, width=8, prec=4):
    if np.isnan(x):
        return f"{'n/a':>{width}s}"
    return f"{x:{width}.{prec}f}"


def _write_summary(out_path: str, confusion: np.ndarray, n_events: int):
    """Write the per-(class, origin) metrics table to a plain-text file."""
    m_all_origins = confusion_metrics(confusion)

    # Marginals: nu only, cosmic only, all origins combined
    cm_nu = confusion_origins_sum(confusion, [1])
    cm_cos = confusion_origins_sum(confusion, [2])
    cm_unk = confusion_origins_sum(confusion, [0])
    cm_all = confusion_origins_sum(confusion, [0, 1, 2])
    m_nu = confusion_metrics(cm_nu)
    m_cos = confusion_metrics(cm_cos)
    m_unk = confusion_metrics(cm_unk)
    m_all = confusion_metrics(cm_all)

    lines = []
    lines.append("Semseg evaluation summary")
    lines.append("=========================")
    lines.append(f"events processed: {n_events}")
    lines.append(f"total scored spacepoints: {int(confusion.sum()):,}")
    lines.append(f"  origin=unknown: {int(m_unk['total'][0]):,}")
    lines.append(f"  origin=nu     : {int(m_nu['total'][0]):,}")
    lines.append(f"  origin=cosmic : {int(m_cos['total'][0]):,}")
    lines.append("")
    lines.append("Overall accuracy (trace / total):")
    lines.append(f"  nu     : {_fmt(float(m_nu['overall_accuracy'][0]))}")
    lines.append(f"  cosmic : {_fmt(float(m_cos['overall_accuracy'][0]))}")
    lines.append(f"  all    : {_fmt(float(m_all['overall_accuracy'][0]))}")
    lines.append("")

    # Mean-IoU (averaged over classes with support)
    def _miou(metrics):
        iou = metrics["iou"][:, 0]
        return float(np.nanmean(iou)) if np.isfinite(iou).any() else float("nan")
    lines.append("mIoU (macro, classes with support):")
    lines.append(f"  nu     : {_fmt(_miou(m_nu))}")
    lines.append(f"  cosmic : {_fmt(_miou(m_cos))}")
    lines.append(f"  all    : {_fmt(_miou(m_all))}")
    lines.append("")

    header = (f"{'class':>10s}  {'origin':>7s}  "
              f"{'support_true':>13s}  {'support_pred':>13s}  "
              f"{'TP':>10s}  "
              f"{'IoU':>8s}  {'recall':>8s}  {'precision':>9s}  {'acc_ovr':>8s}")
    lines.append(header)
    lines.append("-" * len(header))
    pairs = [("nu", m_nu), ("cosmic", m_cos), ("all", m_all)]
    for c in range(N_CLASSES):
        for label, metrics in pairs:
            lines.append(
                f"{CLASS_NAMES[c]:>10s}  {label:>7s}  "
                f"{int(metrics['support_true'][c, 0]):>13d}  "
                f"{int(metrics['support_pred'][c, 0]):>13d}  "
                f"{int(metrics['tp'][c, 0]):>10d}  "
                f"{_fmt(float(metrics['iou'][c, 0]))}  "
                f"{_fmt(float(metrics['recall'][c, 0]))}  "
                f"{_fmt(float(metrics['precision'][c, 0]), width=9)}  "
                f"{_fmt(float(metrics['accuracy_ovr'][c, 0]))}"
            )
        lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\n[wrote] {out_path}")
    return m_nu, m_cos, m_unk, m_all


def _plot_confusion(confusion_per_origin: np.ndarray, outdir: str):
    """Plot 4-panel confusion: nu / cosmic / unknown / all, raw and normalized."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        ("nu", confusion_per_origin[:, :, 1]),
        ("cosmic", confusion_per_origin[:, :, 2]),
        ("unknown", confusion_per_origin[:, :, 0]),
        ("all", confusion_per_origin.sum(axis=2)),
    ]

    # --- raw counts --------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    for ax, (label, cm) in zip(axes.ravel(), panels):
        im = ax.imshow(cm, cmap="viridis", aspect="auto")
        for i in range(N_CLASSES):
            for j in range(N_CLASSES):
                v = cm[i, j]
                if v > 0:
                    ax.text(j, i, f"{int(v)}", ha="center", va="center",
                            color="white" if v < cm.max() / 2 else "black",
                            fontsize=7)
        ax.set_xticks(range(N_CLASSES))
        ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(N_CLASSES))
        ax.set_yticklabels(CLASS_NAMES, fontsize=8)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"origin={label}   N={int(cm.sum()):,}")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Confusion matrix (raw counts)", fontsize=14)
    fig.tight_layout()
    out = os.path.join(outdir, "confusion_counts.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[wrote] {out}")

    # --- row-normalized (recall view) --------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    for ax, (label, cm) in zip(axes.ravel(), panels):
        row_sum = cm.sum(axis=1, keepdims=True)
        norm = np.where(row_sum > 0, cm / np.maximum(row_sum, 1), 0.0)
        im = ax.imshow(norm, cmap="viridis", aspect="auto", vmin=0, vmax=1)
        for i in range(N_CLASSES):
            for j in range(N_CLASSES):
                if cm[i, j] > 0:
                    ax.text(j, i, f"{norm[i, j]:.2f}",
                            ha="center", va="center",
                            color="white" if norm[i, j] < 0.5 else "black",
                            fontsize=7)
        ax.set_xticks(range(N_CLASSES))
        ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(N_CLASSES))
        ax.set_yticklabels(CLASS_NAMES, fontsize=8)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"origin={label}   N={int(cm.sum()):,}")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Confusion matrix (row-normalized = recall)", fontsize=14)
    fig.tight_layout()
    out = os.path.join(outdir, "confusion_rownorm.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[wrote] {out}")


def _plot_metric_bars(m_nu: dict, m_cos: dict, m_all: dict,
                      metric_key: str, ylabel: str, outdir: str, fname: str):
    """Grouped bar chart of `metric_key` per class, split by origin."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vals_nu = m_nu[metric_key][:, 0]
    vals_cos = m_cos[metric_key][:, 0]
    vals_all = m_all[metric_key][:, 0]

    x = np.arange(N_CLASSES)
    width = 0.27

    fig, ax = plt.subplots(figsize=(11, 5))
    bars_nu = ax.bar(x - width, np.nan_to_num(vals_nu, nan=0.0),
                     width, label="nu", color="tab:red", alpha=0.8)
    bars_cos = ax.bar(x, np.nan_to_num(vals_cos, nan=0.0),
                      width, label="cosmic", color="tab:blue", alpha=0.8)
    bars_all = ax.bar(x + width, np.nan_to_num(vals_all, nan=0.0),
                      width, label="all", color="tab:gray", alpha=0.8)

    # Annotate NaN bars
    for bars, vals in [(bars_nu, vals_nu), (bars_cos, vals_cos), (bars_all, vals_all)]:
        for b, v in zip(bars, vals):
            if np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, 0.02,
                        "n/a", ha="center", va="bottom",
                        fontsize=7, rotation=90, color="black")
            else:
                ax.text(b.get_x() + b.get_width() / 2, v + 0.01,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-0.5, N_CLASSES - 0.5)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="lower right")
    ax.set_title(f"{ylabel} per class, by origin")
    fig.tight_layout()
    out = os.path.join(outdir, fname)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[wrote] {out}")


def _plot_score_distributions(score_hist: np.ndarray, edges: np.ndarray,
                              outdir: str):
    """For each true class c, plot the distribution of softmax[:, c] for
    points with true class c (positives) vs all other true classes
    (negatives), per origin nu/cosmic.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    centers = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]

    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True)
    for c, ax in zip(range(N_CLASSES), axes.ravel()):
        # Sum over origins nu+cosmic for "all"
        s_c = score_hist[:, c, :, :].sum(axis=1)        # (C_true, B)
        pos = s_c[c]
        neg = s_c.sum(axis=0) - pos
        if pos.sum() == 0 and neg.sum() == 0:
            ax.set_title(f"{CLASS_NAMES[c]}  (no data)")
            ax.axis("off")
            continue
        ax.bar(centers, pos, width=width, alpha=0.6,
               color="tab:orange", label=f"true={CLASS_NAMES[c]} ({int(pos.sum()):,})")
        ax.bar(centers, neg, width=width, alpha=0.4,
               color="tab:gray", label=f"other classes ({int(neg.sum()):,})")
        ax.set_yscale("log")
        ax.set_title(f"softmax[:, {CLASS_NAMES[c]}]")
        ax.set_xlabel("softmax score")
        ax.legend(loc="upper center", fontsize=7)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Per-class softmax distributions (one-vs-rest)", fontsize=13)
    fig.tight_layout()
    out = os.path.join(outdir, "score_distributions.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[wrote] {out}")


def _plot_roc_one_vs_rest(score_hist: np.ndarray, edges: np.ndarray,
                          outdir: str):
    """One-vs-rest ROC per class, drawn separately for nu and cosmic strata."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(18, 9), sharex=True, sharey=True)
    for c, ax in zip(range(N_CLASSES), axes.ravel()):
        for label, origins, color in [
            ("nu", [1], "tab:red"),
            ("cosmic", [2], "tab:blue"),
            ("all", [0, 1, 2], "k"),
        ]:
            roc = score_hist_to_roc(score_hist, c, origins)
            if roc is None:
                continue
            au = trapezoidal_auc(roc["fpr"], roc["tpr"])
            ax.plot(roc["fpr"], roc["tpr"], color=color, lw=1.6,
                    label=f"{label}  AUC={au:.3f}  "
                          f"(P={roc['n_pos']:,}, N={roc['n_neg']:,})")
        ax.plot([0, 1], [0, 1], "k--", lw=0.6, alpha=0.4)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.01)
        ax.set_title(CLASS_NAMES[c])
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("FPR")
    for ax in axes[:, 0]:
        ax.set_ylabel("TPR")
    fig.suptitle("One-vs-rest ROC per class", fontsize=13)
    fig.tight_layout()
    out = os.path.join(outdir, "roc_one_vs_rest.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[wrote] {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True,
                   help="Combined .npz from merge_shards.py (or a single shard)")
    p.add_argument("--outdir", required=True)
    p.add_argument("--no-plots", action="store_true",
                   help="Skip writing PNGs; just print + write metrics_summary.txt")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    z = np.load(args.input, allow_pickle=False)
    confusion = z["confusion"]          # (C, C, O)
    score_hist = z["score_hist"]        # (C, C, O, B)
    edges = z["score_edges"]
    n_events = int(z["per_event"].size) if "per_event" in z.files else 0
    print(f"input: {args.input}")
    print(f"  events: {n_events}")
    print(f"  confusion shape : {confusion.shape}")
    print(f"  score_hist shape: {score_hist.shape}")

    out_txt = os.path.join(args.outdir, "metrics_summary.txt")
    m_nu, m_cos, m_unk, m_all = _write_summary(out_txt, confusion, n_events)

    if args.no_plots:
        return

    _plot_confusion(confusion, args.outdir)
    _plot_metric_bars(m_nu, m_cos, m_all, "iou", "IoU", args.outdir,
                      "iou_per_class.png")
    _plot_metric_bars(m_nu, m_cos, m_all, "recall", "Recall (per-class accuracy)",
                      args.outdir, "recall_per_class.png")
    _plot_score_distributions(score_hist, edges, args.outdir)
    _plot_roc_one_vs_rest(score_hist, edges, args.outdir)


if __name__ == "__main__":
    main()
