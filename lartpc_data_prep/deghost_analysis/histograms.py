"""Shared accumulator schema for the deghoster evaluation pipeline.

We never save per-spacepoint scores to disk. Instead each shard sums the
following counts across all events it processes:

    hist_real  shape (P, S_real, B)   counts of *real* (hasmatch==1) points,
                                       broken out by predictor x stratum x
                                       score-bin.
    hist_ghost shape (P, B)           counts of *ghost* (hasmatch==0)
                                       points by predictor x score-bin.

with:
    P = 2  predictors:
          0 = LANTERN  (triplet_data/lm_score, higher = more real)
          1 = LoRA     (softmax(deghost_head)[:, real_class_idx])
    B      score-bin count (default 1000), edges = linspace(0, 1, B+1)

Strata indexes for real points (S_real = 13):
    0:  real_all                          (hasmatch==1)
    1:  real_nu                           (hasmatch==1, origin==1)
    2:  real_cosmic                       (hasmatch==1, origin==2)
    3..12: real_ssnet_{0..9}              (hasmatch==1, ssnet_label==c)

Ghost points are not split by origin/ssnet (origin==0 by construction for
ghosts; ssnet labels there carry no calorimetric meaning).

From the summed histograms, at any threshold τ ∈ [0, 1]:
    TP(p, s, τ) = hist_real[p, s, score_bin >= τ].sum()
    FN(p, s, τ) = hist_real[p, s, score_bin <  τ].sum()
    FP(p, τ)    = hist_ghost[p, score_bin >= τ].sum()
    TN(p, τ)    = hist_ghost[p, score_bin <  τ].sum()

(thus "true neutrino accuracy" really means TPR over the nu real subset:
TPR_nu = TP(p, real_nu, τ) / (TP+FN). For overall accuracy use s=real_all.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_PREDICTORS = 2
PREDICTOR_NAMES = ("lantern_lm_score", "lora_real_score")

# Real-point strata. Order must stay stable so different shards line up.
SSNET_NUM_CLASSES = 10  # 0=background, 1=electron, 2=photon, 3=muon, 4=proton,
                        # 5=pion, 6=michel, 7=delta, 8=led, 9=other
STRATUM_NAMES: List[str] = (
    ["real_all", "real_nu", "real_cosmic"]
    + [f"real_ssnet_{c}" for c in range(SSNET_NUM_CLASSES)]
)
N_STRATA = len(STRATUM_NAMES)  # 13

DEFAULT_SCORE_BINS = 1000


def make_score_edges(n_bins: int = DEFAULT_SCORE_BINS) -> np.ndarray:
    return np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float64)


# ---------------------------------------------------------------------------
# Per-event tally row (one struct per processed event, kept short)
# ---------------------------------------------------------------------------

PER_EVENT_DTYPE = np.dtype([
    ("file_idx", np.int32),
    ("n_total", np.int64),
    ("n_real", np.int64),       # truth-label "real" count (definition depends on mode)
    ("n_ghost", np.int64),      # truth-label "ghost" count
    ("n_real_nu", np.int64),    # real & origin==1
    ("n_real_cosmic", np.int64),
    ("n_lora_covered", np.int64),  # spacepoints with a finite LoRA score
    ("status", np.int8),        # 0=ok, 1=skipped, 2=error
    # (hasmatch x ssnet==8) cross-tab so we can quantify the gap between
    # the two ground-truth label conventions at any time:
    ("n_h1_s8", np.int64),      # hasmatch==1 AND ssnet==8 (would be "real" by hasmatch, "ghost" by ssnet)
    ("n_h0_s8", np.int64),      # hasmatch==0 AND ssnet==8 ("ghost" by both — usually most ghosts)
    ("n_h1_sNot8", np.int64),   # hasmatch==1 AND ssnet!=8 ("real" by both)
    ("n_h0_sNot8", np.int64),   # hasmatch==0 AND ssnet!=8 (would be "ghost" by hasmatch, "real" by ssnet)
    ("n_ssnet_9", np.int64),    # ssnet==9 (ignored in val pipeline)
    ("n_ignored", np.int64),    # spacepoints excluded by ignore_mask (e.g., val-mimic ignore=9)
])


# ---------------------------------------------------------------------------
# Accumulator class
# ---------------------------------------------------------------------------

@dataclass
class DeghostAccumulator:
    """Sums hist_real / hist_ghost across many events for one shard."""

    n_bins: int = DEFAULT_SCORE_BINS
    hist_real: np.ndarray = field(init=False)   # (P, S_real, B) int64
    hist_ghost: np.ndarray = field(init=False)  # (P, B) int64
    per_event: list = field(default_factory=list)

    def __post_init__(self):
        self.hist_real = np.zeros((N_PREDICTORS, N_STRATA, self.n_bins), dtype=np.int64)
        self.hist_ghost = np.zeros((N_PREDICTORS, self.n_bins), dtype=np.int64)

    # ---------- accumulate one event ----------

    def add_event(
        self,
        scores: np.ndarray,        # (N, P) float32, columns = PREDICTOR_NAMES
        truth_real: np.ndarray,    # (N,) bool/int  — 1 = "real" truth, 0 = "ghost" truth
                                   # (caller decides label source: hasmatch or ssnet-based)
        origin: np.ndarray,        # (N,) int       — 1=nu, 2=cosmic, 0=other (raw)
        ssnet_label: np.ndarray,   # (N,) int       — 0..9 (raw)
        hasmatch: np.ndarray,      # (N,) int       — raw hasmatch, used only for the
                                   # cross-tab diagnostic; not used as truth here.
        file_idx: int,
        n_lora_covered: int,
        ignore_mask: np.ndarray | None = None,   # (N,) bool, True = drop from hist
        status: int = 0,
    ) -> None:
        N = int(truth_real.shape[0])
        # Cross-tab diagnostic on raw labels (over ALL points, not just kept)
        s8 = (ssnet_label == 8)
        h1 = (hasmatch == 1)
        n_h1_s8 = int((h1 & s8).sum())
        n_h0_s8 = int((~h1 & s8).sum())
        n_h1_sNot8 = int((h1 & ~s8).sum())
        n_h0_sNot8 = int((~h1 & ~s8).sum())
        n_ssnet_9 = int((ssnet_label == 9).sum())
        n_ignored = int(ignore_mask.sum()) if ignore_mask is not None else 0

        n_real_total = int((truth_real == 1).sum())
        n_ghost_total = int((truth_real == 0).sum())
        n_real_nu = int(((truth_real == 1) & (origin == 1)).sum())
        n_real_cosmic = int(((truth_real == 1) & (origin == 2)).sum())
        self.per_event.append((
            np.int32(file_idx), np.int64(N),
            np.int64(n_real_total), np.int64(n_ghost_total),
            np.int64(n_real_nu), np.int64(n_real_cosmic),
            np.int64(n_lora_covered), np.int8(status),
            np.int64(n_h1_s8), np.int64(n_h0_s8),
            np.int64(n_h1_sNot8), np.int64(n_h0_sNot8),
            np.int64(n_ssnet_9), np.int64(n_ignored),
        ))
        if status != 0 or N == 0:
            return

        # Clip scores into [0, 1] and digitize into [0, n_bins-1]
        sc = np.clip(scores.astype(np.float32, copy=False), 0.0, 1.0 - 1e-9)
        bins = (sc * self.n_bins).astype(np.int64)  # (N, P)

        keep = (~ignore_mask) if ignore_mask is not None else np.ones(N, dtype=bool)
        real_mask = (truth_real == 1) & keep
        ghost_mask = (truth_real == 0) & keep
        nu_mask = real_mask & (origin == 1)
        cos_mask = real_mask & (origin == 2)

        for p in range(N_PREDICTORS):
            b_p = bins[:, p]
            # Ghosts
            if ghost_mask.any():
                self.hist_ghost[p] += np.bincount(
                    b_p[ghost_mask], minlength=self.n_bins,
                ).astype(np.int64)[: self.n_bins]
            # Real (all)
            if real_mask.any():
                self.hist_real[p, 0] += np.bincount(
                    b_p[real_mask], minlength=self.n_bins,
                ).astype(np.int64)[: self.n_bins]
            # Real (nu)
            if nu_mask.any():
                self.hist_real[p, 1] += np.bincount(
                    b_p[nu_mask], minlength=self.n_bins,
                ).astype(np.int64)[: self.n_bins]
            # Real (cosmic)
            if cos_mask.any():
                self.hist_real[p, 2] += np.bincount(
                    b_p[cos_mask], minlength=self.n_bins,
                ).astype(np.int64)[: self.n_bins]
            # Per-ssnet strata (real only)
            for c in range(SSNET_NUM_CLASSES):
                m = real_mask & (ssnet_label == c)
                if m.any():
                    self.hist_real[p, 3 + c] += np.bincount(
                        b_p[m], minlength=self.n_bins,
                    ).astype(np.int64)[: self.n_bins]

    # ---------- I/O ----------

    def save(self, path: str, file_list: list, extra: dict | None = None) -> None:
        edges = make_score_edges(self.n_bins)
        per_event = (np.asarray(self.per_event, dtype=PER_EVENT_DTYPE)
                     if self.per_event
                     else np.zeros(0, dtype=PER_EVENT_DTYPE))
        payload = dict(
            hist_real=self.hist_real,
            hist_ghost=self.hist_ghost,
            score_edges=edges,
            n_bins=np.int64(self.n_bins),
            predictor_names=np.array(PREDICTOR_NAMES),
            stratum_names=np.array(STRATUM_NAMES),
            per_event=per_event,
            file_list=np.array(file_list),
        )
        if extra:
            for k, v in extra.items():
                payload[k] = v
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: str) -> "DeghostAccumulator":
        z = np.load(path, allow_pickle=False)
        acc = cls(n_bins=int(z["n_bins"]))
        acc.hist_real = z["hist_real"].astype(np.int64)
        acc.hist_ghost = z["hist_ghost"].astype(np.int64)
        acc.per_event = list(z["per_event"]) if z["per_event"].size else []
        return acc


# ---------------------------------------------------------------------------
# Metric helpers (used by plot_metrics.py)
# ---------------------------------------------------------------------------

def cumulative_counts(hist_real: np.ndarray, hist_ghost: np.ndarray):
    """Return cumulative TP/FN/FP/TN as a function of threshold index.

    For each predictor p and stratum s and bin index k:
        TP[p, s, k] = number of REAL points (in stratum s) with score in
                      bins [k, B-1]                          (predicted real)
        FN[p, s, k] = number of REAL points with score in bins [0, k-1]
        FP[p, k]    = number of GHOST points with score in bins [k, B-1]
        TN[p, k]    = number of GHOST points with score in bins [0, k-1]
    """
    # Sum from the right: counts with score >= threshold[k]
    right_real = np.cumsum(hist_real[:, :, ::-1], axis=-1)[:, :, ::-1]
    right_ghost = np.cumsum(hist_ghost[:, ::-1], axis=-1)[:, ::-1]
    total_real = hist_real.sum(axis=-1, keepdims=True)
    total_ghost = hist_ghost.sum(axis=-1, keepdims=True)
    TP = right_real
    FN = total_real - right_real
    FP = right_ghost
    TN = total_ghost - right_ghost
    return TP, FN, FP, TN


def metrics_from_counts(TP, FN, FP, TN, eps=1e-12):
    """Standard derived metrics. TP/FN are (P, S, B); FP/TN are (P, 1, B)
    (ghost counts are global, not stratified). Outputs are broadcast to
    (P, S, B) for uniform indexing."""
    tpr = TP / (TP + FN + eps)            # recall on real
    fpr = FP / (FP + TN + eps)            # global; broadcast for convenience
    fpr = np.broadcast_to(fpr, tpr.shape).copy()
    precision = TP / (TP + FP + eps)
    iou_real = TP / (TP + FP + FN + eps)
    iou_ghost_b = TN / (TN + FP + FN + eps)
    iou_ghost = np.broadcast_to(iou_ghost_b, tpr.shape).copy()
    accuracy = (TP + TN) / (TP + FN + FP + TN + eps)
    return dict(tpr=tpr, fpr=fpr, precision=precision,
                iou_real=iou_real, iou_ghost=iou_ghost, accuracy=accuracy)
