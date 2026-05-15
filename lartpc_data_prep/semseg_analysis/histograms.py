"""Shared accumulator schema for the semantic-segmentation evaluation pipeline.

Mirrors deghost_analysis/histograms.py but stores multi-class confusion
matrices + per-class softmax score histograms instead of binary scores.

We never save per-spacepoint scores. Each shard sums:

    confusion   shape (C, C, O)        int64
                  rows = true class, cols = predicted (argmax) class,
                  origin = {0=unknown, 1=neutrino, 2=cosmic}
    score_hist  shape (C, C, O, B)     int64
                  for each (true_class t, "softmax column" s, origin o, bin b)
                  the count of spacepoints with true class t, origin o whose
                  softmax probability for class s falls in bin b.
                  Enables one-vs-rest ROC and threshold sweeps post-hoc.

with:
    C = 8 semantic classes (electron, muon, pion, proton, gamma, michel, delta, led)
        — order matches the training config
        Pointcept/configs/lartpc/lorafinetune-sonata-v1m1-lartpc-v6-seg.py
    O = 3 origin strata (unknown / nu / cosmic)
    B = score-bin count, default 1000, edges = linspace(0, 1, B+1)

Per-class accuracy/IoU/recall per origin are reductions of `confusion`:
    TP(c, o)           = confusion[c, c, o]
    support_true(c, o) = confusion[c, :, o].sum()
    support_pred(c, o) = confusion[:, c, o].sum()
    FN(c, o)           = support_true - TP
    FP(c, o)           = support_pred - TP
    IoU(c, o)          = TP / (TP + FN + FP)
    recall(c, o)       = TP / support_true       (per-class accuracy)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


# Order matches the training config's `data.names`.
CLASS_NAMES = (
    "electron", "muon", "pion", "proton", "gamma", "michel", "delta", "led",
)
N_CLASSES = len(CLASS_NAMES)

# Order matches the H5 `origin` field (0=unknown, 1=neutrino, 2=cosmic).
ORIGIN_NAMES = ("unknown", "nu", "cosmic")
N_ORIGINS = len(ORIGIN_NAMES)

DEFAULT_SCORE_BINS = 1000


def make_score_edges(n_bins: int = DEFAULT_SCORE_BINS) -> np.ndarray:
    return np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float64)


PER_EVENT_DTYPE = np.dtype([
    ("file_idx", np.int32),
    ("n_input", np.int64),           # spacepoints fed to the model
    ("n_scored", np.int64),          # spacepoints with valid truth (in {0..C-1})
    ("n_ignored", np.int64),         # spacepoints with truth == -1
    ("n_origin_unknown", np.int64),
    ("n_origin_nu", np.int64),
    ("n_origin_cosmic", np.int64),
    ("n_correct", np.int64),         # argmax == truth, among scored
    ("status", np.int8),             # 0=ok, 1=skipped, 2=error
])


@dataclass
class SemSegAccumulator:
    """Sums confusion + score_hist across many events for one shard."""

    n_bins: int = DEFAULT_SCORE_BINS
    confusion: np.ndarray = field(init=False)       # (C, C, O) int64
    score_hist: np.ndarray = field(init=False)      # (C, C, O, B) int64
    per_event: list = field(default_factory=list)

    def __post_init__(self):
        self.confusion = np.zeros(
            (N_CLASSES, N_CLASSES, N_ORIGINS), dtype=np.int64
        )
        self.score_hist = np.zeros(
            (N_CLASSES, N_CLASSES, N_ORIGINS, self.n_bins), dtype=np.int64
        )

    def add_event(
        self,
        softmax: np.ndarray,        # (N, C) float32
        truth_class: np.ndarray,    # (N,) int, in {-1, 0..C-1}
        origin: np.ndarray,         # (N,) int, expected 0/1/2
        file_idx: int,
        status: int = 0,
    ) -> None:
        N = int(truth_class.shape[0])
        n_unk = int((origin == 0).sum())
        n_nu = int((origin == 1).sum())
        n_cos = int((origin == 2).sum())
        n_ignored = int((truth_class == -1).sum())
        n_scored_pre = max(N - n_ignored, 0)

        if status != 0 or N == 0:
            self.per_event.append((
                np.int32(file_idx), np.int64(N), np.int64(0), np.int64(n_ignored),
                np.int64(n_unk), np.int64(n_nu), np.int64(n_cos),
                np.int64(0), np.int8(status),
            ))
            return

        pred_class = softmax.argmax(axis=1).astype(np.int64)
        valid = (
            (truth_class >= 0)
            & (truth_class < N_CLASSES)
            & (origin >= 0)
            & (origin < N_ORIGINS)
        )
        if not valid.any():
            self.per_event.append((
                np.int32(file_idx), np.int64(N), np.int64(0), np.int64(n_ignored),
                np.int64(n_unk), np.int64(n_nu), np.int64(n_cos),
                np.int64(0), np.int8(status),
            ))
            return

        t = truth_class[valid].astype(np.int64)
        p = pred_class[valid]
        o = origin[valid].astype(np.int64)
        n_correct = int((t == p).sum())

        # Confusion via a single bincount on a flattened (t, p, o) index.
        flat_cm = t * (N_CLASSES * N_ORIGINS) + p * N_ORIGINS + o
        cm_counts = np.bincount(
            flat_cm, minlength=N_CLASSES * N_CLASSES * N_ORIGINS,
        )
        self.confusion += cm_counts[: N_CLASSES * N_CLASSES * N_ORIGINS].reshape(
            N_CLASSES, N_CLASSES, N_ORIGINS
        )

        # Score-hist update: one bincount per softmax column. Memory per
        # iteration is M ints (vs M*C for a single-flatten approach).
        sc = np.clip(
            softmax[valid].astype(np.float32, copy=False), 0.0, 1.0 - 1e-9
        )
        bins = (sc * self.n_bins).astype(np.int64)  # (M, C)
        for s in range(N_CLASSES):
            flat = t * (N_ORIGINS * self.n_bins) + o * self.n_bins + bins[:, s]
            counts = np.bincount(
                flat, minlength=N_CLASSES * N_ORIGINS * self.n_bins,
            )
            self.score_hist[:, s, :, :] += counts[
                : N_CLASSES * N_ORIGINS * self.n_bins
            ].reshape(N_CLASSES, N_ORIGINS, self.n_bins)

        self.per_event.append((
            np.int32(file_idx), np.int64(N),
            np.int64(n_scored_pre), np.int64(n_ignored),
            np.int64(n_unk), np.int64(n_nu), np.int64(n_cos),
            np.int64(n_correct), np.int8(status),
        ))

    # ---------- I/O ----------

    def save(self, path: str, file_list: list, extra: Optional[dict] = None) -> None:
        edges = make_score_edges(self.n_bins)
        per_event = (
            np.asarray(self.per_event, dtype=PER_EVENT_DTYPE)
            if self.per_event
            else np.zeros(0, dtype=PER_EVENT_DTYPE)
        )
        payload = dict(
            confusion=self.confusion,
            score_hist=self.score_hist,
            score_edges=edges,
            n_bins=np.int64(self.n_bins),
            class_names=np.array(CLASS_NAMES),
            origin_names=np.array(ORIGIN_NAMES),
            per_event=per_event,
            file_list=np.array(file_list),
        )
        if extra:
            for k, v in extra.items():
                payload[k] = v
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: str) -> "SemSegAccumulator":
        z = np.load(path, allow_pickle=False)
        acc = cls(n_bins=int(z["n_bins"]))
        acc.confusion = z["confusion"].astype(np.int64)
        acc.score_hist = z["score_hist"].astype(np.int64)
        acc.per_event = list(z["per_event"]) if z["per_event"].size else []
        return acc


# ---------------------------------------------------------------------------
# Metric helpers (used by plot_metrics.py)
# ---------------------------------------------------------------------------

def confusion_metrics(confusion: np.ndarray) -> dict:
    """Reduce a (C, C, O) confusion to per-class, per-origin metrics.

    Conventions:
        rows = true class, cols = predicted class.
        TP(c, o)           = confusion[c, c, o]
        support_true(c, o) = confusion[c, :, o].sum()
        support_pred(c, o) = confusion[:, c, o].sum()
    """
    C, _, O = confusion.shape
    tp = np.array([confusion[c, c] for c in range(C)], dtype=np.int64)  # (C, O)
    support_true = confusion.sum(axis=1)
    support_pred = confusion.sum(axis=0)
    total = confusion.sum(axis=(0, 1))                  # (O,)
    fn = support_true - tp
    fp = support_pred - tp
    tn = total[None, :] - tp - fp - fn

    denom_iou = tp + fn + fp
    iou = np.where(denom_iou > 0, tp / np.maximum(denom_iou, 1), np.nan)
    recall = np.where(support_true > 0, tp / np.maximum(support_true, 1), np.nan)
    precision = np.where(support_pred > 0, tp / np.maximum(support_pred, 1), np.nan)
    accuracy_ovr = np.where(
        total[None, :] > 0, (tp + tn) / np.maximum(total[None, :], 1), np.nan,
    )

    overall_acc = np.array([
        np.trace(confusion[:, :, o]) / total[o] if total[o] > 0 else np.nan
        for o in range(O)
    ], dtype=np.float64)

    return dict(
        tp=tp, fn=fn, fp=fp, tn=tn,
        iou=iou.astype(np.float64),
        recall=recall.astype(np.float64),
        precision=precision.astype(np.float64),
        accuracy_ovr=accuracy_ovr.astype(np.float64),
        support_true=support_true, support_pred=support_pred,
        total=total, overall_accuracy=overall_acc,
    )


def confusion_origins_sum(confusion: np.ndarray,
                          origins: Optional[List[int]] = None) -> np.ndarray:
    """Marginalise a (C, C, O) confusion over a subset of origins.

    Returns a (C, C, 1) array so it can be passed back through
    ``confusion_metrics`` unchanged.
    """
    if origins is None:
        origins = list(range(confusion.shape[2]))
    return confusion[:, :, origins].sum(axis=2, keepdims=True)


def score_hist_to_roc(
    score_hist: np.ndarray,
    class_idx: int,
    origin_filter: Optional[List[int]] = None,
) -> Optional[dict]:
    """One-vs-rest ROC for ``class_idx`` from the score_hist.

    score_hist[t, s, o, b] = count of points with true class t, origin o, whose
    softmax probability for class s falls in bin b.

    Positive = true class == class_idx
    Negative = true class != class_idx, summed across other classes
    Scan threshold over the column s == class_idx.
    """
    if origin_filter is None:
        origin_filter = list(range(score_hist.shape[2]))
    # (C, B) sliced for column s=class_idx, summed over selected origins
    s_c = score_hist[:, class_idx, :, :][:, origin_filter, :].sum(axis=1)
    pos = s_c[class_idx]
    neg = s_c.sum(axis=0) - pos
    P = int(pos.sum())
    N = int(neg.sum())
    if P == 0 or N == 0:
        return None
    # Cumulative-from-the-right: count of points with score >= edges[k].
    pos_cum = np.cumsum(pos[::-1])[::-1].astype(np.float64)
    neg_cum = np.cumsum(neg[::-1])[::-1].astype(np.float64)
    tpr = pos_cum / max(P, 1)
    fpr = neg_cum / max(N, 1)
    return dict(tpr=tpr, fpr=fpr, n_pos=P, n_neg=N)


def trapezoidal_auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    order = np.argsort(fpr)
    trap = getattr(np, "trapezoid", None) or np.trapz
    return float(trap(tpr[order], fpr[order]))
