"""Read a combined deghost-eval .npz and produce metrics + plots.

Outputs:
    - {outdir}/metrics_summary.txt  — text table of accuracy, IoU, AUC at key
                                       thresholds for each (predictor, stratum)
    - {outdir}/roc_all.png           — ROC curves: TPR(real_all) vs FPR(ghost)
    - {outdir}/roc_by_origin.png     — ROC curves: real_nu / real_cosmic
                                       (with FPR computed on all ghosts)
    - {outdir}/roc_by_ssnet.png      — ROC curves per ssnet class
    - {outdir}/iou_vs_threshold.png  — IoU_real / IoU_ghost vs threshold
    - {outdir}/recall_vs_threshold.png — TPR per stratum vs threshold
    - {outdir}/score_histograms.png  — score distributions, real vs ghost,
                                       per predictor
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from histograms import (  # noqa: E402
    cumulative_counts, metrics_from_counts, SSNET_NUM_CLASSES,
)


SSNET_CLASS_LABELS = [
    "background", "electron", "photon", "muon", "proton",
    "pion", "michel", "delta", "led", "other",
]


def auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """Trapezoidal AUC. Inputs must be ordered with respect to threshold;
    we sort by fpr to be safe."""
    order = np.argsort(fpr)
    trap = getattr(np, "trapezoid", None) or np.trapz  # numpy >= 2 / < 2
    return float(trap(tpr[order], fpr[order]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="combined .npz from merge_shards.py")
    p.add_argument("--outdir", required=True)
    p.add_argument("--no-plots", action="store_true",
                   help="Skip writing PNGs; just print + write metrics_summary.txt")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    z = np.load(args.input, allow_pickle=False)
    hist_real = z["hist_real"]       # (P, S, B)
    hist_ghost = z["hist_ghost"]     # (P, B)
    edges = z["score_edges"]         # (B+1,)
    centers = 0.5 * (edges[:-1] + edges[1:])
    predictor_names = [str(s) for s in z["predictor_names"]]
    stratum_names = [str(s) for s in z["stratum_names"]]
    n_events = int(z["per_event"].size)
    print(f"input: {args.input}")
    print(f"  events: {n_events}")
    print(f"  predictors: {predictor_names}")
    print(f"  strata: {stratum_names}")
    print(f"  hist_real {hist_real.shape}  hist_ghost {hist_ghost.shape}")

    # ---- cumulative counts as a function of threshold index ----
    TP, FN, FP, TN = cumulative_counts(hist_real, hist_ghost)
    # TP shape: (P, S, B). For thresholds we use the LEFT bin edge: any score
    # >= edges[k] gets counted. So index k corresponds to threshold edges[k].
    thresholds = edges[:-1]

    # FP/TN are (P, B); to broadcast with TP/FN (P, S, B) we add a stratum axis.
    FP_b = FP[:, None, :]
    TN_b = TN[:, None, :]

    m = metrics_from_counts(TP, FN, FP_b, TN_b)
    # tpr (P, S, B); fpr (P, S, B); iou_real (P, S, B); iou_ghost (P, S, B); accuracy (P, S, B)

    # ----------------------------------------------------------------
    # 1. Text summary
    # ----------------------------------------------------------------

    lines = []
    lines.append(f"Deghoster evaluation summary")
    lines.append(f"============================")
    lines.append(f"events processed: {n_events}")
    real_all_total = int(hist_real[0, 0].sum())
    ghost_total = int(hist_ghost[0].sum())
    lines.append(f"real points (hasmatch=1) total: {real_all_total:,}")
    lines.append(f"ghost points (hasmatch=0) total: {ghost_total:,}")
    lines.append("")

    # AUC and best-IoU operating point per predictor x stratum
    lines.append(f"{'predictor':>18s} {'stratum':>18s}  "
                 f"{'AUC':>6s}  {'τ*':>5s}  {'acc@τ*':>7s}  "
                 f"{'IoU_real@τ*':>11s}  {'IoU_ghost@τ*':>12s}  "
                 f"{'TPR@τ*':>7s}  {'FPR@τ*':>7s}  {'TPR@0.5':>7s}")
    for p in range(len(predictor_names)):
        # Use real_all stratum for the AUC / best-τ selection
        fpr_all = m["fpr"][p, 0]
        tpr_all = m["tpr"][p, 0]
        auc_all = auc(fpr_all, tpr_all)
        # best τ = argmax IoU_real
        best_k = int(np.argmax(m["iou_real"][p, 0]))
        tau_star = float(thresholds[best_k])
        for s in range(len(stratum_names)):
            iou_real_at = float(m["iou_real"][p, s, best_k])
            iou_ghost_at = float(m["iou_ghost"][p, s, best_k])
            acc_at = float(m["accuracy"][p, s, best_k])
            tpr_at = float(m["tpr"][p, s, best_k])
            fpr_at = float(m["fpr"][p, s, best_k])
            # TPR@0.5 — use the bin containing 0.5
            k_half = int(np.searchsorted(thresholds, 0.5))
            k_half = min(k_half, m["tpr"].shape[-1] - 1)
            tpr_half = float(m["tpr"][p, s, k_half])
            # AUC: use this stratum's TPR vs FPR(global ghost)
            auc_s = auc(m["fpr"][p, s], m["tpr"][p, s])
            lines.append(
                f"{predictor_names[p]:>18s} {stratum_names[s]:>18s}  "
                f"{auc_s:>6.4f}  {tau_star:>5.3f}  {acc_at:>7.4f}  "
                f"{iou_real_at:>11.4f}  {iou_ghost_at:>12.4f}  "
                f"{tpr_at:>7.4f}  {fpr_at:>7.4f}  {tpr_half:>7.4f}"
            )
        lines.append("")

    out_txt = os.path.join(args.outdir, "metrics_summary.txt")
    with open(out_txt, "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\n[wrote] {out_txt}")

    if args.no_plots:
        return

    # ----------------------------------------------------------------
    # 2. Plots
    # ----------------------------------------------------------------

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pred_colors = ["#d62728", "#1f77b4"]  # LANTERN red, LoRA blue

    def _save(fig, name):
        out = os.path.join(args.outdir, name)
        fig.tight_layout()
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f"[wrote] {out}")

    # --- ROC: all real vs all ghost (one panel) ---
    fig, ax = plt.subplots(figsize=(6, 6))
    for p in range(len(predictor_names)):
        ax.plot(m["fpr"][p, 0], m["tpr"][p, 0],
                color=pred_colors[p], lw=2,
                label=f"{predictor_names[p]} (AUC={auc(m['fpr'][p, 0], m['tpr'][p, 0]):.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("FPR (ghost predicted real)")
    ax.set_ylabel("TPR (real predicted real)")
    ax.set_title(f"ROC — all real vs all ghost   N_events={n_events}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    _save(fig, "roc_all.png")

    # --- ROC by origin (real_nu vs real_cosmic), with global ghost FPR ---
    fig, ax = plt.subplots(figsize=(6, 6))
    for p in range(len(predictor_names)):
        for s, label, ls in [(1, "real_nu", "-"), (2, "real_cosmic", "--")]:
            ax.plot(m["fpr"][p, s], m["tpr"][p, s],
                    color=pred_colors[p], lw=2, ls=ls,
                    label=f"{predictor_names[p]} / {label} "
                          f"(AUC={auc(m['fpr'][p, s], m['tpr'][p, s]):.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("FPR (ghost predicted real)")
    ax.set_ylabel("TPR (real predicted real)")
    ax.set_title("ROC by origin (real_nu solid, real_cosmic dashed)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, "roc_by_origin.png")

    # --- ROC by ssnet class ---
    ssnet_colors = plt.cm.tab10(np.linspace(0, 1, SSNET_NUM_CLASSES))
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    for ax, p in zip(axes, range(len(predictor_names))):
        for c in range(SSNET_NUM_CLASSES):
            s = 3 + c   # real_ssnet_c offset in the stratum list
            if hist_real[p, s].sum() == 0:
                continue
            ax.plot(m["fpr"][p, s], m["tpr"][p, s],
                    color=ssnet_colors[c], lw=1.5,
                    label=f"{c}={SSNET_CLASS_LABELS[c]}  "
                          f"AUC={auc(m['fpr'][p, s], m['tpr'][p, s]):.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=0.6, alpha=0.4)
        ax.set_xlabel("FPR")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
        ax.set_title(f"ROC by ssnet — {predictor_names[p]}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=7)
    axes[0].set_ylabel("TPR")
    _save(fig, "roc_by_ssnet.png")

    # --- IoU vs threshold (real_all stratum) ---
    fig, ax = plt.subplots(figsize=(7, 5))
    for p in range(len(predictor_names)):
        ax.plot(thresholds, m["iou_real"][p, 0],
                color=pred_colors[p], lw=2,
                label=f"{predictor_names[p]} IoU_real")
        ax.plot(thresholds, m["iou_ghost"][p, 0],
                color=pred_colors[p], lw=2, ls="--",
                label=f"{predictor_names[p]} IoU_ghost")
    ax.set_xlabel("threshold τ (predict real if score ≥ τ)")
    ax.set_ylabel("IoU")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower center", fontsize=9)
    ax.set_title("IoU vs threshold (real_all stratum)")
    _save(fig, "iou_vs_threshold.png")

    # --- Recall (TPR) vs threshold per stratum ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, p in zip(axes, range(len(predictor_names))):
        for s, label, ls, c in [
            (0, "real_all", "-", "k"),
            (1, "real_nu", "-", "tab:red"),
            (2, "real_cosmic", "-", "tab:blue"),
        ]:
            ax.plot(thresholds, m["tpr"][p, s], color=c, ls=ls, lw=2, label=label)
        ax.set_xlabel("threshold τ")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Recall vs τ — {predictor_names[p]}")
        ax.legend(loc="lower left", fontsize=9)
    axes[0].set_ylabel("TPR (recall on real)")
    _save(fig, "recall_vs_threshold.png")

    # --- Score histograms (real vs ghost) per predictor ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, p in zip(axes, range(len(predictor_names))):
        ax.bar(centers, hist_real[p, 0], width=(centers[1] - centers[0]),
               alpha=0.5, color="tab:orange", label=f"real ({int(hist_real[p,0].sum()):,})")
        ax.bar(centers, hist_ghost[p], width=(centers[1] - centers[0]),
               alpha=0.5, color="tab:gray", label=f"ghost ({int(hist_ghost[p].sum()):,})")
        ax.set_yscale("log")
        ax.set_xlabel("real-class score")
        ax.set_ylabel("count")
        ax.set_title(f"Score distribution — {predictor_names[p]}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    _save(fig, "score_histograms.png")


if __name__ == "__main__":
    main()
