"""Shower -> connection-point geometry & greedy nu-vertex connection (Phase 1).

Showers connect to the interaction by POINTING BACK across a (possibly large)
conversion gap, not by touching — so the test is impact parameter + back-pointing
cosine, not endpoint proximity. See `../shower_reco_spec.md` §5.
"""
import numpy as np


def connection_geometry(start, direction, point):
    """Geometry of a trunk (start, unit direction) relative to a `point` (vertex
    or track connection point).

    Returns dict with:
      gap     = |start - point|                         (conversion gap)
      impact  = perpendicular distance of `point` to the trunk LINE
      along   = signed distance of `point` projected on the trunk axis (from start;
                <0 means `point` is "behind" the start, i.e. up-shower — expected
                for a shower originating at `point`)
      cosine  = cos angle between `direction` and (start - point): ~+1 when the
                trunk heads away from `point` (originates there)
    """
    start = np.asarray(start, np.float64)
    d = np.asarray(direction, np.float64)
    P = np.asarray(point, np.float64)
    r = P - start
    along = float(r @ d)
    impact = float(np.linalg.norm(r - along * d))
    v2s = start - P
    n = np.linalg.norm(v2s)
    cosine = float((d @ v2s) / n) if n > 1e-9 else 1.0
    return dict(gap=float(np.linalg.norm(r)), impact=impact, along=along,
                cosine=cosine)


def connects(start, direction, point, d_impact=5.0, cos_min=0.9,
             d_gap=np.inf):
    """Greedy connection decision (Phase 1: point = nu vertex).

    Attach when the trunk line passes close to `point` (impact) AND the trunk
    points back at it (cosine) AND the gap is within bound (default unbounded —
    showers can convert far from the vertex). `cos_min`/`d_gap` are the
    class-specific knobs left open for later e-vs-gamma tuning.
    """
    g = connection_geometry(start, direction, point)
    ok = (g["impact"] <= d_impact and g["cosine"] >= cos_min
          and g["gap"] <= d_gap)
    return bool(ok), g
