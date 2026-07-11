"""Production shower-attachment LLR scorer (Phase 2 alternative to hard cuts).

Loads the histogram log-likelihood-ratio tables fitted by
eval/fit_attachment_likelihood.py (--save-tables) and scores a
(shower point cloud, connection point) pair with the same variables the
study recorded:

  cosine / log_sin_tk / log_gap / trunk_q   from the vertex-biased trunk,
  pca_cosine / log_sin_pca                  from the full-shower 1st PCA,
  cone_qfrac / ang_rms                      cone-shape prior about the axis
                                            CP -> trunk start (charge-weighted
                                            when weights are given).

Motivated by the attachment study: hard cuts (impact<=10, cos>=0.9, gap<=60)
keep ~42% of correct pairs and collapse for far-converting / small showers;
the size-binned LLR at the same false-attach rate recovers far photons
(0.15 -> 0.79) with efficiency flat past 100 cm conversion distance.
"""
import os

import numpy as np

from .shower_trunk import trunk_vertex_biased, _pca
from .shower_connect import connection_geometry

EPS = 1e-3
CONE_HALF_DEG = 30.0
DEFAULT_TABLES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "data", "attachment_llr_tables.npz")


def cone_vars(pts, q, apex, axis, min_r=1.0):
    """Charge fraction within CONE_HALF_DEG of `axis` from `apex` + weighted
    RMS angle [deg]; points within min_r of the apex are angle-undefined."""
    r = pts - apex
    dist = np.linalg.norm(r, axis=1)
    m = dist > min_r
    if not m.any() or q[m].sum() <= 0:
        return np.nan, np.nan
    cosang = np.clip((r[m] @ axis) / dist[m], -1, 1)
    ang = np.degrees(np.arccos(cosang))
    w = q[m]
    return (float(w[ang < CONE_HALF_DEG].sum() / w.sum()),
            float(np.sqrt((w * ang ** 2).sum() / w.sum())))


class AttachLLR:
    """Histogram-LLR attachment scorer. `thr` = the fitted matched-false-rate
    working point (callers may override)."""

    def __init__(self, path=None):
        path = path or DEFAULT_TABLES
        z = np.load(path, allow_pickle=False)
        self.size_edges = z["size_edges"]
        self.thr = float(z["thr_matched_false"])
        self.var_names = [str(v) for v in z["var_names"]]
        nbin = len(self.size_edges) - 1
        self.tables = [
            {name: (z[f"bins_{bi}_{name}"], z[f"lr_{bi}_{name}"])
             for name in self.var_names if f"lr_{bi}_{name}" in z}
            for bi in range(nbin)]

    # -- variable computation (must mirror eval/shower_attachment_study) ----
    def variables(self, pts, cp, q=None, tk=None):
        pts = np.asarray(pts, np.float64)
        cp = np.asarray(cp, np.float64)
        if q is None:
            q = np.ones(len(pts), np.float64)
        if tk is None:
            tk = trunk_vertex_biased(pts, cp)
        g = connection_geometry(tk.start, tk.direction, cp)
        c, evals, evecs = _pca(pts)
        e1 = evecs[:, 0]
        d1 = e1 if (c - cp) @ e1 >= 0 else -e1
        rv = cp - c
        a1 = float(rv @ d1)
        pca_imp = float(np.linalg.norm(rv - a1 * d1))
        v2c = c - cp
        nn = np.linalg.norm(v2c)
        pca_cos = float((d1 @ v2c) / nn) if nn > 1e-9 else 1.0
        ax = tk.start - cp
        axn = np.linalg.norm(ax)
        axis = ax / axn if axn > 1e-6 else tk.direction
        cone_f, ang_rms = cone_vars(pts, q, tk.start, axis)
        gap = max(g["gap"], 0.1)
        return tk, {
            "cosine": g["cosine"],
            "pca_cosine": pca_cos,
            "log_sin_tk": np.log10(np.clip(g["impact"] / gap, EPS, 2.0)),
            "log_sin_pca": np.log10(np.clip(pca_imp / gap, EPS, 20.0)),
            "log_gap": np.log10(np.clip(gap, 0.1, 500)),
            "trunk_q": tk.quality,
            "cone_qfrac": cone_f,
            "ang_rms": (np.clip(ang_rms, 0, 120)
                        if np.isfinite(ang_rms) else np.nan),
        }, g

    def score_vars(self, n_pts, v):
        bi = int(np.clip(np.digitize(n_pts, self.size_edges) - 1,
                         0, len(self.tables) - 1))
        s = 0.0
        for name, (bins, lr) in self.tables[bi].items():
            x = v.get(name, np.nan)
            if np.isfinite(x):
                s += float(lr[int(np.clip(np.digitize(x, bins) - 1,
                                          0, len(lr) - 1))])
        return s

    def score(self, pts, cp, q=None, tk=None):
        """(llr, trunk, geometry dict, variables dict) for one pair."""
        tk, v, g = self.variables(pts, cp, q=q, tk=tk)
        return self.score_vars(len(pts), v), tk, g, v
