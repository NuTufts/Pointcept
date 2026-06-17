"""Keypoint label helpers for the LArFormer keypoint module.

Shared, dependency-light builders for keypoint training targets. Used by the
LArFormer data loaders (raw merged_h5 path, and the stage-12 cache path once
augmented with the `mckeypoints` group).

The canonical per-spacepoint *dense* target is the same Gaussian-proximity
score MicroBooNE's LArMatch keypoint net trained on (see
`ubdl/larflow/larflow/PrepFlowMatchData/SimChTripletLabelMaker.cxx::make_keypoint_labels`):

    kpscore_t(sp) = exp( -0.5 * (d_t / sigma)^2 )   if >= threshold else 0

where `d_t` is the Euclidean distance (cm) from the spacepoint to the nearest
keypoint of type `t`. We compute it on-the-fly from the `mckeypoints` group
rather than reading a stored `triplet_data/kpscores` field (which is absent
from the current merged_h5 production), so `sigma` is tunable and the target
is available regardless of how the H5 was produced.

Keypoint type indices (match `mckeypoints/kptype` and the `kpscores` columns):
    0 nu_vertex, 1 track_start, 2 track_end, 3 shower, 4 michel, 5 delta
"""

import numpy as np

N_KEYPOINT_TYPES = 6
KEYPOINT_TYPE_NAMES = (
    "nu_vertex",
    "track_start",
    "track_end",
    "shower",
    "michel",
    "delta",
)
KPTYPE_NU_VERTEX = 0
KPTYPE_TRACK_START = 1
KPTYPE_TRACK_END = 2
KPTYPE_SHOWER = 3
KPTYPE_MICHEL = 4
KPTYPE_DELTA = 5


def compute_kpscores(
    coord_cm,
    kp_pos_cm,
    kp_type,
    sigma_cm=3.0,
    score_threshold=0.01,
    n_types=N_KEYPOINT_TYPES,
    return_offsets=False,
):
    """Dense per-spacepoint keypoint proximity scores (and optional offsets).

    Args:
        coord_cm:        (N, 3) float — spacepoint positions in detector cm.
        kp_pos_cm:       (K, 3) float — keypoint positions in detector cm
                          (use the *appearance* point `mckeypoints/pos`, not
                          `startpos`, since that's what spacepoints sit near).
        kp_type:         (K,) int — keypoint type per keypoint (0..n_types-1).
        sigma_cm:        Gaussian width (cm). Production default 3.0.
        score_threshold: scores below this are zeroed. Production default 0.01.
        n_types:         number of keypoint-type columns to emit (default 6).
        return_offsets:  if True, also return (N, n_types, 3) offsets from
                          each spacepoint to the nearest keypoint of each type
                          (the VoteNet/PPN offset-regression target). Offsets
                          are 0 for type columns with no keypoints.

    Returns:
        scores (N, n_types) float32, or (scores, offsets) if return_offsets.
        A type column with no keypoints in this event is all-zero.
    """
    from scipy.spatial import cKDTree

    coord_cm = np.ascontiguousarray(coord_cm, dtype=np.float32)
    N = coord_cm.shape[0]
    scores = np.zeros((N, n_types), dtype=np.float32)
    offsets = (np.zeros((N, n_types, 3), dtype=np.float32)
               if return_offsets else None)
    if N == 0 or kp_pos_cm is None or len(kp_pos_cm) == 0:
        return (scores, offsets) if return_offsets else scores

    kp_pos_cm = np.ascontiguousarray(kp_pos_cm, dtype=np.float32)
    kp_type = np.asarray(kp_type, dtype=np.int64)
    inv_2sig2 = 1.0 / (2.0 * float(sigma_cm) * float(sigma_cm))

    for t in range(n_types):
        sel = kp_type == t
        if not np.any(sel):
            continue
        kpt = kp_pos_cm[sel]
        tree = cKDTree(kpt)
        d, idx = tree.query(coord_cm, k=1)
        d = np.asarray(d, dtype=np.float32)
        s = np.exp(-(d * d) * inv_2sig2).astype(np.float32)
        s[s < score_threshold] = 0.0
        scores[:, t] = s
        if return_offsets:
            offsets[:, t, :] = (kpt[idx] - coord_cm).astype(np.float32)

    return (scores, offsets) if return_offsets else scores


def copy_mckeypoints_group(src_h5_path, dst_file, entry="entry_0",
                           overwrite=True):
    """Copy `<entry>/mckeypoints` from a source merged_h5 into an open dst file.

    Used to make a stage-12 cache file self-describing for keypoint training:
    the cache reader then computes `kpscores` + endpoints on-the-fly from this
    group (single source of truth, σ tunable at read time) exactly like the
    raw-merged_h5 path.

    Args:
        src_h5_path: path to the source merged_h5.
        dst_file:    an open `h5py.File` (mode r+/w) to copy into.
        entry:       entry group name (default "entry_0").
        overwrite:   if a group already exists at the destination, replace it.

    Returns:
        number of keypoints copied, or -1 if the source has no mckeypoints
        group (caller decides whether that's an error).
    """
    import h5py

    with h5py.File(src_h5_path, "r") as src:
        if entry not in src or "mckeypoints" not in src[entry]:
            return -1
        src_grp = src[entry]["mckeypoints"]
        dst_entry = dst_file.require_group(entry)
        if "mckeypoints" in dst_entry:
            if not overwrite:
                return int(dst_entry["mckeypoints"]["pos"].shape[0])
            del dst_entry["mckeypoints"]
        # Cross-file group copy (datasets pos, startpos, kptype, trackid,
        # pid, imgcoord) — keeps the cm frame + every field of the source.
        src.copy(src_grp, dst_entry, name="mckeypoints")
        return int(dst_entry["mckeypoints"]["pos"].shape[0])


def endpoint_by_trackid(kp_type, kp_trackid, kp_pos, kptype_value):
    """Map trackid → keypoint position for keypoints of one type.

    Used to attach a `track_end` (kptype 2) — or any type-by-trackid point —
    to a GT particle instance keyed by its Geant4 trackid. First occurrence
    wins. `kp_pos` may be cm or normalized; the caller picks the frame.
    """
    out = {}
    kt = np.asarray(kp_type, dtype=np.int64)
    tid = np.asarray(kp_trackid, dtype=np.int64)
    for i in range(kt.shape[0]):
        if int(kt[i]) == int(kptype_value):
            out.setdefault(int(tid[i]), np.asarray(kp_pos[i], dtype=np.float32))
    return out
