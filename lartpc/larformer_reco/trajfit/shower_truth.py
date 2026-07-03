"""Provenance truth for shower attachment (item a).

A shower should attach to the nu vertex iff its TRUE origin (creation vertex of
the shower-initiating particle) is at the nu vertex. That origin is NOT the
matched trackid's mc origin (shower electrons are created mid-shower, 100s of cm
away) — it is the shower-level `originpt` in `merged_sp/shower_fragments`. We map
a predicted shower instance to its fragment by nearest `startpt` to the instance's
GT conversion point (`gt_start`), then test `|originpt - nu_vertex| <= R`.
"""
import numpy as np
import h5py

from . import trajfit_io as tio


def load_shower_fragments(keypoint2_path, merged_sp_dir):
    """Load shower_fragments (startpt/originpt/type/pid) from the parent merged_sp,
    or None if unavailable."""
    if not merged_sp_dir:
        return None
    with h5py.File(keypoint2_path, "r") as f:
        src = f.attrs.get("src_file", "")
    if isinstance(src, bytes):
        src = src.decode()
    entry, fh = tio._open_merged_sp(merged_sp_dir, src)
    if entry is None:
        if fh is not None:
            fh.close()
        return None
    try:
        sf = entry.get("shower_fragments")
        if sf is None or "startpt" not in sf or sf["startpt"].shape[0] == 0:
            return None
        return dict(startpt=np.asarray(sf["startpt"][()], np.float32),
                    originpt=np.asarray(sf["originpt"][()], np.float32),
                    type=np.asarray(sf["type"][()], np.int64).reshape(-1),
                    pid=np.asarray(sf["pid"][()], np.int64).reshape(-1))
    finally:
        fh.close()


def shower_is_primary(gt_start, frag, nu_vertex, R=5.0, max_match_cm=10.0):
    """Returns (is_primary, origin, match_dist). is_primary is None when there is
    no truth (no fragments / no GT start / no reliable startpt match)."""
    if frag is None or gt_start is None or not np.all(np.isfinite(gt_start)) \
            or not np.all(np.isfinite(nu_vertex)):
        return None, None, None
    d = np.linalg.norm(frag["startpt"] - gt_start, axis=1)
    j = int(d.argmin())
    if d[j] > max_match_cm:                       # no fragment matches this start
        return None, None, float(d[j])
    origin = frag["originpt"][j]
    return bool(np.linalg.norm(origin - nu_vertex) <= R), origin, float(d[j])
