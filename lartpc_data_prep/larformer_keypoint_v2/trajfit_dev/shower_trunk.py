"""Shower trunk-direction reconstruction — three methods (Phase 1).

For one predicted shower instance (cls in {e, gamma}), estimate the trunk
direction used to match the shower back to the nu vertex / interaction. See
`../shower_reco_spec.md`. All methods return a `ShowerTrunk`.

Methods:
  (1) trunk_pca           — leading PCA axis of the whole cluster (baseline).
  (2) trunk_elpigraph     — ElPiGraph skeleton of the trunk fragment; initial
                            straight segment direction.
  (3) trunk_vertex_biased — LANTERN NuVertexShowerReco port: local PCA anchored
                            at the cluster point nearest the vertex.

Anchor: methods (1)/(2) start at the predicted shower-start keypoint; method (3)
starts at the vertex-nearest cluster point (its defining bias).
"""
import time
from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import DBSCAN


@dataclass
class ShowerTrunk:
    start: np.ndarray            # (3,) trunk start, cm
    direction: np.ndarray        # (3,) unit, pointing away from start (down shower)
    trunk_points: np.ndarray     # (K,3) points used for the direction fit
    length_cm: float
    n_cluster: int
    quality: float               # direction confidence (PCA elongation in [0,1])
    method: str
    runtime_s: float
    extra: dict = field(default_factory=dict)


def _pca(P, weights=None):
    """Charge-weighted PCA. Returns (center, eigvals desc, eigvecs cols desc)."""
    P = np.asarray(P, np.float64)
    if weights is not None and np.sum(weights) > 0:
        w = np.asarray(weights, np.float64)
        c = (P * w[:, None]).sum(0) / w.sum()
        X = P - c
        cov = (X.T * w) @ X / w.sum()
    else:
        c = P.mean(0)
        X = P - c
        cov = X.T @ X / max(len(P), 1)
    evals, evecs = np.linalg.eigh(cov)        # ascending
    order = np.argsort(evals)[::-1]
    return c, evals[order], evecs[:, order]


def _orient(direction, start, ref):
    """Flip `direction` so it points from `start` toward `ref` (the shower body)."""
    return direction if np.dot(direction, ref - start) >= 0 else -direction


def _extent_along(P, start, d):
    s = (np.asarray(P, np.float64) - start) @ d
    return float(s.max() - s.min())


def cluster_fragments(points, eps=2.0, min_samples=3):
    """DBSCAN the shower cloud into continuous fragments. Returns list of (label,
    point-index-array), largest first; noise (-1) dropped."""
    lab = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points)
    frags = [(c, np.where(lab == c)[0]) for c in sorted(set(lab[lab >= 0]))]
    frags.sort(key=lambda t: -len(t[1]))
    return frags


def trunk_fragment(points, start, eps=2.0):
    """The fragment whose closest point is nearest `start` (the keypoint tags the
    trunk). Returns the fragment's point indices (or all points if no fragment)."""
    frags = cluster_fragments(points, eps=eps)
    if not frags:
        return np.arange(len(points))
    best, bestd = None, np.inf
    for _, idx in frags:
        d = np.linalg.norm(points[idx] - start, axis=1).min()
        if d < bestd:
            bestd, best = d, idx
    return best


# ---------------------------------------------------------------------------
# (1) whole-cluster PCA
# ---------------------------------------------------------------------------
def trunk_pca(points, start, weights=None):
    t0 = time.perf_counter()
    P = np.asarray(points, np.float64)
    c, evals, evecs = _pca(P, weights)
    d = _orient(evecs[:, 0], start, c)
    quality = float(evals[0] / (evals.sum() + 1e-12))
    return ShowerTrunk(start=np.asarray(start, np.float64), direction=d,
                       trunk_points=P, length_cm=_extent_along(P, start, d),
                       n_cluster=len(P), quality=quality, method="pca",
                       runtime_s=time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# (3) vertex-biased local trunk  (LANTERN NuVertexShowerReco::_make_trunk_cand)
# ---------------------------------------------------------------------------
def trunk_vertex_biased(points, vertex, trunk_maxdist_cm=5.0, eps_sub=2.0,
                        min_pts=5, min_frag_pts=10):
    t0 = time.perf_counter()
    P = np.asarray(points, np.float64)
    V = np.asarray(vertex, np.float64)
    # 0. fragment the cluster and keep only fragments >= min_frag_pts, so the
    #    trunk is NOT anchored on a tiny stray fragment that floats near the
    #    connection point — skip to the next real fragment instead.
    keep = [idx for _, idx in cluster_fragments(P, eps=eps_sub)
            if len(idx) >= min_frag_pts]
    pool = P[np.concatenate(keep)] if keep else P
    # 1. pool point nearest the vertex = trunk-start anchor
    dv = np.linalg.norm(pool - V, axis=1)
    minpos = pool[int(dv.argmin())]
    # 2. local trunk region within radius of minpos
    local = pool[np.linalg.norm(pool - minpos, axis=1) <= trunk_maxdist_cm]
    if len(local) < min_pts:
        local = pool
    # 3. sub-cluster the local region; 4/5. PCA each, score by vertex alignment
    v2m = minpos - V
    v2m = v2m / (np.linalg.norm(v2m) + 1e-12)
    best = None
    subs = cluster_fragments(local, eps=eps_sub) or [(0, np.arange(len(local)))]
    for _, idx in subs:
        cand = local[idx]
        if len(cand) < min_pts:
            continue
        c, evals, evecs = _pca(cand)
        d = _orient(evecs[:, 0], minpos, c)         # outward from minpos
        score = float(d @ v2m)                       # alignment with vertex line
        if best is None or score > best[0]:
            best = (score, d, cand, float(evals[0] / (evals.sum() + 1e-12)))
    if best is None:                                 # degenerate: PCA the local
        c, evals, evecs = _pca(local)
        d = _orient(evecs[:, 0], minpos, c)
        best = (float(d @ v2m), d, local, float(evals[0] / (evals.sum() + 1e-12)))
    _, d, cand, quality = best
    return ShowerTrunk(start=minpos, direction=d, trunk_points=cand,
                       length_cm=_extent_along(cand, minpos, d),
                       n_cluster=len(P), quality=quality, method="vertex_biased",
                       runtime_s=time.perf_counter() - t0,
                       extra=dict(align_score=best[0]))


# ---------------------------------------------------------------------------
# (2) ElPiGraph trunk finding
# ---------------------------------------------------------------------------
def trunk_elpigraph(points, start, eps=2.0, min_trunk_len=5.0, num_nodes=0):
    from run_elpigraph import fit_elpigraph, trace_path  # noqa (container only)
    t0 = time.perf_counter()
    P = np.asarray(points, np.float64)
    idx = trunk_fragment(P, np.asarray(start, np.float64), eps=eps)
    F = P[idx]
    if len(F) < 6:                                   # too small to skeletonize
        tk = trunk_pca(F if len(F) else P, start)
        tk.method = "elpigraph"
        tk.runtime_s = time.perf_counter() - t0
        tk.extra["fallback"] = "pca_small_fragment"
        return tk
    nn = num_nodes or int(np.clip(np.ptp(F, 0).max() / 2.0, 5, 40))
    try:
        poly, info = fit_elpigraph(F, num_nodes=nn)
    except Exception as e:                           # pragma: no cover
        tk = trunk_pca(F, start)
        tk.method = "elpigraph"
        tk.runtime_s = time.perf_counter() - t0
        tk.extra["fallback"] = f"pca_elpi_err:{type(e).__name__}"
        return tk
    poly = np.asarray(poly, np.float64)
    s = np.asarray(start, np.float64)
    if np.linalg.norm(poly[0] - s) > np.linalg.norm(poly[-1] - s):
        poly = poly[::-1]                            # end nearest start first
    # walk the initial straight segment up to min_trunk_len
    acc, prev, tgt = 0.0, poly[0], poly[-1]
    for q in poly[1:]:
        acc += np.linalg.norm(q - prev)
        prev = q
        if acc >= min_trunk_len:
            tgt = q
            break
    d = tgt - poly[0]
    n = np.linalg.norm(d)
    d = d / n if n > 1e-9 else np.array([1.0, 0, 0])
    d = _orient(d, s, F.mean(0))
    return ShowerTrunk(start=s, direction=d, trunk_points=F,
                       length_cm=min(acc, _extent_along(F, s, d)),
                       n_cluster=len(P), quality=float(min(acc / max(min_trunk_len, 1e-6), 1.0)),
                       method="elpigraph", runtime_s=time.perf_counter() - t0,
                       extra=dict(n_nodes=info.get("n_nodes")))


METHODS = {"pca": trunk_pca, "elpigraph": trunk_elpigraph,
           "vertex_biased": trunk_vertex_biased}
