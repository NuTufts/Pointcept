"""Per-instance track loader for the trajectory-fitting spike.

Yields one record per predicted particle instance from a `keypoint2_out` event
H5 (the product of tools/larformer/run_larformer_keypoint2_cascade_inference.py), with
optional enrichment from the parent `merged_sp` event H5 (charge weights + MC
truth). See `../trajectory_fitting_brief.md` §2 for the schema.

A `keypoint2_out` file is self-contained for geometry + GT endpoints; pass a
`merged_sp` directory to also attach per-point charge (`triplet_data/pixval`),
true PDG/momentum, and true kink (hard-scatter) vertices.

CLI smoke test:
    python trajfit_io.py <keypoint2_event*.h5> [--merged-sp-dir DIR]
"""
import os
import glob
import argparse
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import h5py

# Stage-3 particle segmenter class list (see brief §2.4).
CLASS_NAMES = ["e", "gamma", "mu", "pi", "p", "other", "(unused)", "no_object"]
TRACK_CLASSES = (2, 3, 4)          # mu, pi, p
SHOWER_CLASSES = (0, 1)            # e, gamma

# PDG code -> rest mass [MeV] for the few species we see (for true_p).
_PDG_MASS = {
    11: 0.511, -11: 0.511, 22: 0.0,
    13: 105.658, -13: 105.658,
    211: 139.570, -211: 139.570, 111: 134.977,
    2212: 938.272, 2112: 939.565,
    1000010030: 2808.921,  # triton
}


@dataclass
class InstanceRecord:
    event_file: str
    inst_idx: int
    points: np.ndarray              # (n, 3) f32 cm  -- the fit input cloud
    pred_cls: int                   # particle class (CLASS_NAMES)
    pred_cls_name: str
    is_track: bool
    pred_start: np.ndarray          # (3,) f32, may be NaN
    pred_end: np.ndarray            # (3,) f32, may be NaN
    gt_trackid: int
    has_match: bool
    iou: float
    gt_start: np.ndarray            # (3,) f32, may be NaN
    gt_end: np.ndarray              # (3,) f32, may be NaN
    truth_cloud: np.ndarray         # (m, 3) f32 -- matched-GT cloud (slice idx)
    # --- merged_sp enrichment (None / NaN if not provided or unavailable) ---
    charge: Optional[np.ndarray] = None     # (n, 3) per-plane pixval
    weights: Optional[np.ndarray] = None     # (n,) scalar charge weight (Y plane)
    pid: int = 0
    energy_mev: float = float("nan")
    true_p_mev: float = float("nan")
    true_kinks: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))

    @property
    def n_points(self):
        return self.points.shape[0]


def momentum_from_ke(ke_mev: float, pid: int) -> float:
    """p = sqrt(T^2 + 2 T m), T = kinetic energy, m = PDG rest mass."""
    m = _PDG_MASS.get(int(pid))
    if m is None or not np.isfinite(ke_mev):
        return float("nan")
    return float(np.sqrt(max(ke_mev, 0.0) ** 2 + 2.0 * max(ke_mev, 0.0) * m))


def _read_keypoint2(path):
    """Pull the per-instance arrays out of a keypoint2_out event H5."""
    with h5py.File(path, "r") as f:
        slice_coord = f["slice/coord_cm"][()].astype(np.float32)
        n = int(f.attrs["n_particles"])
        insts = []
        for i in range(n):
            g = f[f"particle/{i}"]
            insts.append(dict(
                inst_idx=i,
                cls=int(g.attrs["cls"]),
                gt_trackid=int(g.attrs["gt_trackid"]),
                has_match=bool(g.attrs["has_match"]),
                iou=float(g.attrs["iou"]),
                point_idx=g["point_idx"][()].astype(np.int64),
                gt_point_idx=g["gt_point_idx"][()].astype(np.int64),
                start=g["start_cm"][()].astype(np.float32),
                end=g["end_cm"][()].astype(np.float32),
                gt_start=g["gt_start_cm"][()].astype(np.float32),
                gt_end=g["gt_end_cm"][()].astype(np.float32),
            ))
        src_file = f.attrs.get("src_file", "")
        if isinstance(src_file, bytes):
            src_file = src_file.decode()
    return slice_coord, insts, src_file


def _open_merged_sp(merged_sp_dir, src_file):
    """Find + open the parent merged_sp file; return (entry_group, file) or (None, None)."""
    if not merged_sp_dir or not src_file:
        return None, None
    cand = os.path.join(merged_sp_dir, src_file)
    if not os.path.exists(cand):
        hits = glob.glob(os.path.join(merged_sp_dir, "**", src_file), recursive=True)
        cand = hits[0] if hits else None
    if not cand or not os.path.exists(cand):
        return None, None
    f = h5py.File(cand, "r")
    # single top group "entry_0" (or first entry_*)
    key = "entry_0" if "entry_0" in f else next(
        (k for k in f.keys() if k.startswith("entry_")), None)
    return (f[key] if key else None), f


def _build_charge_match(slice_coord, entry):
    """KD-match slice points into triplet_data/pos; return per-slice pixval (N,3)
    or None. Distance check guards against a mismatched parent file."""
    from scipy.spatial import cKDTree
    tpos = entry["triplet_data/pos"][()].astype(np.float32)
    pixval = entry["triplet_data/pixval"][()].astype(np.float32)
    tree = cKDTree(tpos)
    d, idx = tree.query(slice_coord, k=1)
    if np.median(d) > 0.05:   # cm; slice must be an exact subset (see brief §2.1)
        print(f"  [warn] slice<->triplet_data median match dist={np.median(d):.3f} cm "
              f"(>0.05) -- charge match unreliable, skipping")
        return None
    return pixval[idx]


def _true_kinks_for(entry, trackid):
    """Hard-scatter / decay vertices on a parent track: start_pos_sce of the track
    itself + of all its direct daughters, deduped (brief §2.6)."""
    mt = entry["mc_particle_tree"]
    tid = mt["trackid"][()]
    parent = mt["parent_trackid"][()]
    spos = mt["start_pos_sce"][()].astype(np.float32)
    verts = []
    sel_self = tid == trackid
    if sel_self.any():
        verts.append(spos[sel_self][0])
    sel_dau = parent == trackid
    verts.extend(spos[sel_dau])
    if not verts:
        return np.zeros((0, 3), np.float32)
    verts = np.asarray(verts, np.float32)
    # dedup coincident vertices (< 0.5 cm)
    keep = []
    for v in verts:
        if not any(np.linalg.norm(v - k) < 0.5 for k in keep):
            keep.append(v)
    return np.asarray(keep, np.float32)


def load_instances(keypoint2_path, merged_sp_dir=None, tracks_only=False,
                   min_points=1):
    """Yield InstanceRecord for every predicted particle in one event.

    merged_sp_dir: if given, attach charge weights + MC truth from the parent.
    tracks_only:   keep only cls in TRACK_CLASSES.
    """
    slice_coord, insts, src_file = _read_keypoint2(keypoint2_path)
    entry, fh = _open_merged_sp(merged_sp_dir, src_file)
    pixval_slice = None
    if entry is not None:
        try:
            pixval_slice = _build_charge_match(slice_coord, entry)
        except Exception as e:                       # pragma: no cover
            print(f"  [warn] charge match failed: {e}")

    # cache mc_particle_tree pid/energy lookups
    pid_by_tid, ke_by_tid = {}, {}
    if entry is not None:
        mt = entry["mc_particle_tree"]
        for t, p, e in zip(mt["trackid"][()], mt["pid"][()], mt["energy_mev"][()]):
            pid_by_tid[int(t)] = int(p)
            ke_by_tid[int(t)] = float(e)

    recs = []
    try:
        for d in insts:
            if tracks_only and d["cls"] not in TRACK_CLASSES:
                continue
            pidx = d["point_idx"]
            if pidx.size < min_points:
                continue
            pts = slice_coord[pidx]
            charge = pixval_slice[pidx] if pixval_slice is not None else None
            weights = charge[:, 2].copy() if charge is not None else None  # Y plane
            pid = pid_by_tid.get(d["gt_trackid"], 0)
            ke = ke_by_tid.get(d["gt_trackid"], float("nan"))
            kinks = (_true_kinks_for(entry, d["gt_trackid"])
                     if entry is not None and d["gt_trackid"] >= 0
                     else np.zeros((0, 3), np.float32))
            recs.append(InstanceRecord(
                event_file=os.path.basename(keypoint2_path),
                inst_idx=d["inst_idx"], points=pts,
                pred_cls=d["cls"], pred_cls_name=CLASS_NAMES[d["cls"]]
                if 0 <= d["cls"] < len(CLASS_NAMES) else str(d["cls"]),
                is_track=d["cls"] in TRACK_CLASSES,
                pred_start=d["start"], pred_end=d["end"],
                gt_trackid=d["gt_trackid"], has_match=d["has_match"],
                iou=d["iou"], gt_start=d["gt_start"], gt_end=d["gt_end"],
                truth_cloud=slice_coord[d["gt_point_idx"]],
                charge=charge, weights=weights, pid=pid, energy_mev=ke,
                true_p_mev=momentum_from_ke(ke, pid), true_kinks=kinks))
    finally:
        if fh is not None:
            fh.close()
    return recs


def _main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("keypoint2_file")
    ap.add_argument("--merged-sp-dir", default=None)
    ap.add_argument("--tracks-only", action="store_true")
    args = ap.parse_args()
    recs = load_instances(args.keypoint2_file, args.merged_sp_dir,
                          tracks_only=args.tracks_only)
    print(f"{len(recs)} instances in {os.path.basename(args.keypoint2_file)}")
    for r in recs:
        chg = "n/a" if r.weights is None else f"{r.weights.sum():.0f}"
        print(f"  inst {r.inst_idx:2d} cls={r.pred_cls}({r.pred_cls_name:5s}) "
              f"track={r.is_track} n={r.n_points:4d} iou={r.iou:.2f} "
              f"pid={r.pid:5d} p={r.true_p_mev:7.1f}MeV kinks={len(r.true_kinks)} "
              f"Qsum={chg}")


if __name__ == "__main__":
    _main()
