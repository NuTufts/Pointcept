"""
Basic performance plots for shower origin inference results.

Usage:
    python tools/plot_shower_origin_results.py results.h5 -o plots/
"""

import argparse
import os

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


CLASS_NAMES_3 = {0: "inside", 1: "outside", 2: "on_track"}
CLASS_NAMES_5 = {0: "inside", 1: "outside", 2: "cosmic_inside",
                 3: "ghost", 4: "true_track"}

# GT origin types that have no meaningful ground truth origin point.
# Distance/score metrics are NaN for these.
NO_GT_ORIGIN_TYPES = {-1, 3, 4}


def detect_class_names(df):
    """Auto-detect 3-class vs 5-class from the data."""
    gt_types = set(df["gt_origin_type"].unique())
    pred_types = set(df["pred_class"].unique())
    all_types = gt_types | pred_types
    if all_types & {3, 4}:
        return CLASS_NAMES_5
    return CLASS_NAMES_3


def load_results(path):
    with h5py.File(path, "r") as f:
        df = pd.DataFrame({k: f[k][:] for k in f.keys()})
    # Decode byte strings if needed
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str)
    return df


# ── Plot 1: Confusion matrix ─────────────────────────────────────────────────

def plot_confusion_matrix(df, outdir, class_names):
    """Confusion matrix over all classes (GT vs predicted).

    For reco 5-class mode, ghost/track GT types have no origin-score GT
    but *do* have a GT class label, so they appear in the confusion matrix.
    """
    # Use the union of GT types and predicted classes as labels
    all_labels = sorted(set(df["gt_origin_type"].unique()) |
                        set(df["pred_class"].unique()))
    display_labels = [class_names.get(l, str(l)) for l in all_labels]

    cm = confusion_matrix(df["gt_origin_type"], df["pred_class"],
                          labels=all_labels)

    fig, ax = plt.subplots(figsize=(max(6, len(all_labels) + 2),
                                    max(5, len(all_labels) + 1)))
    disp = ConfusionMatrixDisplay(cm, display_labels=display_labels)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Origin Type Classification")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")

    # Overall accuracy
    acc = (df["gt_origin_type"] == df["pred_class"]).mean()
    ax.text(
        0.02, 0.98, f"Overall accuracy: {acc:.1%}",
        transform=ax.transAxes, va="top", fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "confusion_matrix.png"), dpi=150)
    plt.close(fig)
    print(f"  confusion_matrix.png  (accuracy={acc:.1%})")


# ── Plot 2: Peak-to-GT distance histograms per class ─────────────────────────

def plot_distance_histograms(df, outdir, class_names,
                             coord_scale=179.44046366413568):
    """Distance histograms only for classes with a meaningful GT origin.

    Ghost (3), true_track (4), and unknown (-1) are skipped since their
    peak_gt_dist values are NaN.
    """
    # Filter to classes that have GT origin
    classes = sorted(c for c in df["gt_origin_type"].unique()
                     if c not in NO_GT_ORIGIN_TYPES)
    if not classes:
        print("  distance_by_class.png  SKIPPED (no classes with GT origin)")
        return

    n = len(classes)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, cls in zip(axes, classes):
        subset = df[df["gt_origin_type"] == cls]
        dists = subset["peak_gt_dist"].dropna() * coord_scale
        name = class_names.get(cls, str(cls))

        if len(dists) == 0:
            ax.set_title(f"{name} (n=0)")
            continue

        ax.hist(dists, bins=50, range=(0, dists.quantile(0.98)), alpha=0.8,
                edgecolor="black", linewidth=0.5)
        ax.set_title(f"{name} (n={len(dists)})")
        ax.set_xlabel("Peak-to-GT distance (cm)")
        ax.axvline(dists.median(), color="red", linestyle="--", linewidth=1.5,
                   label=f"median={dists.median():.1f} cm")
        ax.legend(fontsize=9)

    axes[0].set_ylabel("Fragments")
    fig.suptitle("Predicted Peak to GT Origin Distance", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "distance_by_class.png"), dpi=150)
    plt.close(fig)
    print("  distance_by_class.png")


# ── Plot 3: Max score histograms (predicted peak vs GT nearest) per class ────

def plot_score_histograms(df, outdir, class_names):
    """Score histograms only for classes with a meaningful GT origin.

    Ghost/track/unknown types are skipped since gt_nearest_score is NaN.
    """
    classes = sorted(c for c in df["gt_origin_type"].unique()
                     if c not in NO_GT_ORIGIN_TYPES)
    if not classes:
        print("  scores_by_class.png  SKIPPED (no classes with GT origin)")
        return

    n = len(classes)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    bins = np.linspace(0, 1, 51)

    for ax, cls in zip(axes, classes):
        subset = df[df["gt_origin_type"] == cls]
        name = class_names.get(cls, str(cls))

        ax.hist(subset["pred_peak_score"].dropna(), bins=bins, alpha=0.6,
                label="pred peak score", edgecolor="black", linewidth=0.3)
        ax.hist(subset["gt_nearest_score"].dropna(), bins=bins, alpha=0.6,
                label="GT nearest score", edgecolor="black", linewidth=0.3)
        ax.set_title(f"{name} (n={len(subset)})")
        ax.set_xlabel("Score")
        ax.legend(fontsize=9)

    axes[0].set_ylabel("Fragments")
    fig.suptitle("Predicted Peak Score vs GT Nearest Score", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "scores_by_class.png"), dpi=150)
    plt.close(fig)
    print("  scores_by_class.png")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot shower origin inference results"
    )
    parser.add_argument("input", type=str, help="Results HDF5 file")
    parser.add_argument(
        "-o", "--outdir", type=str, default="plots",
        help="Output directory for plots (default: plots/)"
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = load_results(args.input)
    class_names = detect_class_names(df)
    is_reco = class_names is CLASS_NAMES_5

    print(f"Loaded {len(df)} fragments from {args.input}")
    print(f"  Mode: {'reco 5-class' if is_reco else 'standard 3-class'}")
    print(f"  GT type counts: { {class_names.get(int(k), int(k)): v for k, v in df['gt_origin_type'].value_counts().items()} }")
    print()

    plot_confusion_matrix(df, args.outdir, class_names)
    plot_distance_histograms(df, args.outdir, class_names)
    plot_score_histograms(df, args.outdir, class_names)

    print(f"\nAll plots saved to {args.outdir}/")


if __name__ == "__main__":
    main()
