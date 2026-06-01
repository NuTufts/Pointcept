"""Slice (event-instance) labels for LArTPC spacepoints.

A "slice" is the set of spacepoints whose contributing Geant4 track shares the
same primary ancestor — i.e., one slice per cosmic ray entering the detector
and one slice per neutrino interaction. Slices are the instance-segmentation
target for the event-slicer model.

This module derives slice IDs from the existing per-spacepoint `trackid` and
the `mc_particle_tree` group already saved in the H5 files. The producer
(`SimChTripletLabelMaker` since Nov 2025) also writes a per-spacepoint `aid`
field; this utility works whether `aid` is present or not.

Definitions
-----------
- primary_trackid : Geant4 trackid of the slice's ancestor. Found by walking
  `parent_trackid` upward until parent is -1, 0, self, or absent from the
  graph. The sentinel root node (trackid=0, parent=-1, pid=-1, origin=-1) is
  *not* considered a valid primary; its descendants resolve to themselves.
- slice_id : an integer per spacepoint equal to its `primary_trackid`. Ghosts
  (hasmatch==0) and spacepoints whose trackid is not in the MC graph get -1.
- Neutrino slice: in mc_particle_tree the genie final-state particles from a
  single nu interaction appear as multiple sibling primaries (e.g., mu, p, n,
  pi). With ``merge_nu_slices=True`` (default) all primaries with
  ``origin == 1`` are merged into a single neutrino slice per nu_vertex,
  keyed by the smallest member trackid. The merged slice's
  ``primary_start_pos`` becomes the nu_vertex position.
- Cosmic slice: any slice whose primary's ``origin == 2``. Multiple per event,
  never merged.

Multi-nu_vertex events (rare for MicroBooNE, possible for SBND/DUNE-ND):
  This implementation assigns each nu primary to its closest nu_vertex by
  Euclidean distance on ``start_pos``. With a single vertex (the MicroBooNE
  case) all nu primaries collapse to one slice.
"""

from collections import defaultdict

import numpy as np


GHOST_SLICE_ID = -1


def _walk_to_primary(parent_of, tids_in_graph):
    """Build trackid -> primary_trackid for every node in the graph.

    `parent_of` maps trackid -> parent_trackid. `tids_in_graph` is the set of
    trackids present in mc_particle_tree (so we can detect orphans).
    """
    primary_of = {}
    for t in parent_of:
        seen = []
        cur = t
        while True:
            if cur in primary_of:
                root = primary_of[cur]
                break
            seen.append(cur)
            p = parent_of.get(cur, None)
            if p is None or p == -1 or p == 0 or p == cur or p not in tids_in_graph:
                root = cur
                break
            cur = p
        for s in seen:
            primary_of[s] = root
    return primary_of


def compute_slice_labels(mpt_group, sp_trackid, sp_hasmatch=None,
                         merge_nu_slices=True):
    """Compute per-spacepoint slice IDs and per-slice metadata.

    Parameters
    ----------
    mpt_group : h5py.Group or dict-like
        The ``mc_particle_tree`` group. Must expose datasets: ``trackid``,
        ``parent_trackid``, ``pid``, ``origin``, ``start_pos``. Optionally
        ``start_pos_sce`` and ``nu_vertices``.
    sp_trackid : (N,) int array
        Per-spacepoint Geant4 trackid (from ``triplet_data/trackid``).
    sp_hasmatch : (N,) int array, optional
        Per-spacepoint ghost flag (from ``triplet_data/hasmatch``). When
        provided, points with ``hasmatch == 0`` are forced to ``GHOST_SLICE_ID``.
    merge_nu_slices : bool, default True
        If True, all ``origin == 1`` primaries belonging to the same nu_vertex
        are merged into one slice keyed by the smallest member trackid; the
        merged slice's ``primary_start_pos`` is the corresponding nu_vertex.

    Returns
    -------
    dict with:
        slice_id              (N,) int64   slice key per spacepoint, -1 ghost/orphan
        primary_trackid       (S,) int64   sorted unique slice keys (excludes -1)
        primary_pid           (S,) int64   PDG of slice's lead primary; 0 for merged-nu slice
        primary_origin        (S,) int64   1=nu, 2=cosmic
        primary_start_pos     (S, 3) float32
        primary_start_pos_sce (S, 3) float32  or zeros if not in H5
        primary_n_spacepoints (S,) int64
        slice_member_trackids list[ list[int] ]  the original primary trackids
                                                 collapsed into each slice
        nu_vertices           (V, 3) float32  copied from mpt_group (V usually 1)
    """
    mc_tids = np.asarray(mpt_group["trackid"][:]).astype(np.int64)
    mc_parents = np.asarray(mpt_group["parent_trackid"][:]).astype(np.int64)
    mc_pids = np.asarray(mpt_group["pid"][:]).astype(np.int64)
    mc_origin = np.asarray(mpt_group["origin"][:]).astype(np.int64)
    mc_start = np.asarray(mpt_group["start_pos"][:]).astype(np.float32)
    if "start_pos_sce" in mpt_group:
        mc_start_sce = np.asarray(mpt_group["start_pos_sce"][:]).astype(np.float32)
    else:
        mc_start_sce = np.zeros_like(mc_start)

    tid_to_idx = {int(t): i for i, t in enumerate(mc_tids)}
    parent_of = {int(t): int(p) for t, p in zip(mc_tids, mc_parents)}
    tids_in_graph = set(parent_of.keys())

    primary_of = _walk_to_primary(parent_of, tids_in_graph)

    if "nu_vertices" in mpt_group:
        nu_vertices = np.asarray(mpt_group["nu_vertices"][:]).astype(np.float32)
    else:
        nu_vertices = np.zeros((0, 3), dtype=np.float32)

    # Optional merge: collapse all origin==1 primaries to one canonical key
    # per nu_vertex. Each nu primary is assigned to its closest nu_vertex.
    merged_primary = dict(primary_of)
    nu_slice_members = defaultdict(list)
    if merge_nu_slices and len(nu_vertices) > 0:
        nu_primary_tids = [
            int(t) for i, t in enumerate(mc_tids)
            if int(mc_origin[i]) == 1 and primary_of.get(int(t), -1) == int(t)
        ]
        if len(nu_primary_tids) > 0:
            assigned_vertex = {}
            for ptid in nu_primary_tids:
                idx = tid_to_idx[ptid]
                p_pos = mc_start[idx]
                d = np.linalg.norm(nu_vertices - p_pos[None, :], axis=1)
                assigned_vertex[ptid] = int(np.argmin(d))
            grouped = defaultdict(list)
            for ptid, vidx in assigned_vertex.items():
                grouped[vidx].append(ptid)
            for vidx, members in grouped.items():
                canonical = min(members)
                for ptid in members:
                    nu_slice_members[canonical].append(ptid)
            remap = {}
            for canonical, members in nu_slice_members.items():
                for m in members:
                    remap[m] = canonical
            merged_primary = {
                t: remap.get(p, p) for t, p in primary_of.items()
            }

    sp_trackid = np.asarray(sp_trackid).astype(np.int64)
    slice_id = np.full(sp_trackid.shape, GHOST_SLICE_ID, dtype=np.int64)
    for i, t in enumerate(sp_trackid):
        ti = int(t)
        if ti in merged_primary:
            prim = merged_primary[ti]
            if prim != 0 and prim != -1:
                slice_id[i] = prim

    if sp_hasmatch is not None:
        sp_hasmatch = np.asarray(sp_hasmatch).astype(np.int64)
        slice_id[sp_hasmatch == 0] = GHOST_SLICE_ID

    unique_slices = np.array(
        sorted(int(s) for s in np.unique(slice_id) if int(s) != GHOST_SLICE_ID),
        dtype=np.int64,
    )

    counts = defaultdict(int)
    for s in slice_id:
        si = int(s)
        if si != GHOST_SLICE_ID:
            counts[si] += 1

    primary_pid = np.zeros(len(unique_slices), dtype=np.int64)
    primary_origin = np.zeros(len(unique_slices), dtype=np.int64)
    primary_start_pos = np.zeros((len(unique_slices), 3), dtype=np.float32)
    primary_start_pos_sce = np.zeros((len(unique_slices), 3), dtype=np.float32)
    primary_n = np.zeros(len(unique_slices), dtype=np.int64)
    slice_member_trackids = [[] for _ in unique_slices]
    for k, s in enumerate(unique_slices):
        si = int(s)
        if si in nu_slice_members:
            members = nu_slice_members[si]
            slice_member_trackids[k] = list(members)
            primary_pid[k] = 0
            primary_origin[k] = 1
            first_member_idx = tid_to_idx[members[0]]
            first_member_pos = mc_start[first_member_idx]
            d = np.linalg.norm(nu_vertices - first_member_pos[None, :], axis=1)
            primary_start_pos[k] = nu_vertices[int(np.argmin(d))]
            primary_start_pos_sce[k] = nu_vertices[int(np.argmin(d))]
        else:
            idx = tid_to_idx[si]
            slice_member_trackids[k] = [si]
            primary_pid[k] = mc_pids[idx]
            primary_origin[k] = mc_origin[idx]
            primary_start_pos[k] = mc_start[idx]
            primary_start_pos_sce[k] = mc_start_sce[idx]
        primary_n[k] = counts[si]

    return {
        "slice_id": slice_id,
        "primary_trackid": unique_slices,
        "primary_pid": primary_pid,
        "primary_origin": primary_origin,
        "primary_start_pos": primary_start_pos,
        "primary_start_pos_sce": primary_start_pos_sce,
        "primary_n_spacepoints": primary_n,
        "slice_member_trackids": slice_member_trackids,
        "nu_vertices": nu_vertices,
    }


# ----------------------------------------------------------------------------
# Per-particle labels (Stage 3 particle segmenter — see
# docs/LArFormer_particlesegment_stage.md §2a)
# ----------------------------------------------------------------------------

# Per-PDG (by abs(pid)) minimum KE to count as a separate GT instance.
# Particles below threshold get merged into their nearest above-threshold
# nu-origin ancestor (walking parent_trackid up). Cosmic-origin SPs and
# fully-orphan SPs land in GHOST_SLICE_ID (= -1).
DEFAULT_PARTICLE_KE_THRESH_MeV = {
    11:   10.0,   # e±
    22:   10.0,   # γ
    13:   30.0,   # μ±
    211:  30.0,   # π±
    2212: 60.0,   # p
    2112: 60.0,   # n
    321:  60.0,   # K±
}
DEFAULT_PARTICLE_OTHER_KE_THRESH_MeV = 60.0


def compute_particle_labels(
    mpt_group, sp_trackid, sp_hasmatch=None,
    ke_thresholds=None,
    other_ke_threshold=DEFAULT_PARTICLE_OTHER_KE_THRESH_MeV,
    nu_origin=1,
):
    """Per-spacepoint particle slice IDs for the Stage 3 particle segmenter.

    Walks `mc_particle_tree` for tracks with ``origin == nu_origin``. Each
    track is "visible" iff ``energy_mev >= ke_thresholds[abs(pid)]`` (or
    ``other_ke_threshold`` for PDGs not in the table). A track that is not
    visible is collapsed into its nearest visible ancestor by walking up
    ``parent_trackid``. SPs whose chain finds no visible nu-origin
    ancestor (orphans, cosmic primaries, etc.) get ``GHOST_SLICE_ID``.

    Parameters
    ----------
    mpt_group : h5py.Group or dict-like
        Same ``mc_particle_tree`` shape `compute_slice_labels` consumes,
        plus the ``energy_mev`` dataset (KE in MeV per Geant4 track).
    sp_trackid : (N,) int array
        Per-SP Geant4 trackid (from ``triplet_data/trackid``).
    sp_hasmatch : (N,) int array, optional
        Ghost flag; ``hasmatch == 0`` forces ``GHOST_SLICE_ID``.
    ke_thresholds : dict, optional
        Mapping ``abs(pid) -> KE_threshold_MeV``. Defaults to
        ``DEFAULT_PARTICLE_KE_THRESH_MeV`` (10 MeV e±/γ; 30 MeV μ±/π±;
        60 MeV p/n/K).
    other_ke_threshold : float
        Threshold for PDGs not in ``ke_thresholds``.
    nu_origin : int, default 1
        Which ``origin`` value counts as "nu-origin". Cosmics
        (``origin == 2``) are dropped (-> ``GHOST_SLICE_ID``).

    Returns
    -------
    dict with:
        slice_id              (N,) int64       particle slice id per SP
        primary_trackid       (S,) int64       sorted unique slice keys
        primary_pid           (S,) int64       PDG of each surviving particle
        primary_ke_MeV        (S,) float32     KE of each surviving particle
        primary_origin        (S,) int64       always ``nu_origin`` here
        primary_start_pos     (S, 3) float32
        primary_n_spacepoints (S,) int64
        slice_member_trackids list[ list[int] ]  trackids merged into each
                                                  surviving particle
        ke_thresholds         dict             the thresholds actually used
        other_ke_threshold    float            ditto
    """
    if ke_thresholds is None:
        ke_thresholds = dict(DEFAULT_PARTICLE_KE_THRESH_MeV)

    mc_tids    = np.asarray(mpt_group["trackid"][:]).astype(np.int64)
    mc_parents = np.asarray(mpt_group["parent_trackid"][:]).astype(np.int64)
    mc_pids    = np.asarray(mpt_group["pid"][:]).astype(np.int64)
    mc_origin  = np.asarray(mpt_group["origin"][:]).astype(np.int64)
    mc_start   = np.asarray(mpt_group["start_pos"][:]).astype(np.float32)
    mc_ke      = np.asarray(mpt_group["energy_mev"][:]).astype(np.float32)

    tid_to_idx = {int(t): i for i, t in enumerate(mc_tids)}
    parent_of  = {int(t): int(p) for t, p in zip(mc_tids, mc_parents)}

    def _is_visible(idx):
        if int(mc_origin[idx]) != int(nu_origin):
            return False
        pdg = abs(int(mc_pids[idx]))
        ke  = float(mc_ke[idx])
        thresh = float(ke_thresholds.get(pdg, other_ke_threshold))
        return ke >= thresh

    # Walk each track up to its nearest visible nu-origin ancestor.
    # particle_of: trackid -> surviving slice id (or -1 if no visible ancestor).
    particle_of = {}

    def _walk(tid):
        cur, seen = int(tid), []
        while True:
            if cur in particle_of:
                root = particle_of[cur]; break
            if cur not in tid_to_idx:
                root = GHOST_SLICE_ID; break
            idx = tid_to_idx[cur]
            if _is_visible(idx):
                root = cur; break
            seen.append(cur)
            parent = parent_of.get(cur, -1)
            if parent == -1 or parent == 0 or parent == cur:
                root = GHOST_SLICE_ID; break
            cur = parent
        for s in seen:
            particle_of[s] = root
        particle_of[cur] = root
        return root

    for t in mc_tids:
        ti = int(t)
        if ti not in particle_of:
            _walk(ti)

    sp_trackid = np.asarray(sp_trackid).astype(np.int64)
    slice_id = np.full(sp_trackid.shape, GHOST_SLICE_ID, dtype=np.int64)
    for i, t in enumerate(sp_trackid):
        ti = int(t)
        if ti in particle_of:
            slice_id[i] = particle_of[ti]

    if sp_hasmatch is not None:
        sp_hasmatch = np.asarray(sp_hasmatch).astype(np.int64)
        slice_id[sp_hasmatch == 0] = GHOST_SLICE_ID

    unique_slices = np.array(
        sorted(int(s) for s in np.unique(slice_id) if int(s) != GHOST_SLICE_ID),
        dtype=np.int64,
    )

    # Reverse map: slice id -> list of merged trackids.
    inv = defaultdict(list)
    for tid, sid in particle_of.items():
        if sid != GHOST_SLICE_ID:
            inv[int(sid)].append(int(tid))

    primary_pid       = np.zeros(len(unique_slices), dtype=np.int64)
    primary_origin    = np.full (len(unique_slices), int(nu_origin), dtype=np.int64)
    primary_ke_MeV    = np.zeros(len(unique_slices), dtype=np.float32)
    primary_start_pos = np.zeros((len(unique_slices), 3), dtype=np.float32)
    primary_n         = np.zeros(len(unique_slices), dtype=np.int64)
    slice_member_trackids = [[] for _ in unique_slices]
    for k, s in enumerate(unique_slices):
        si = int(s)
        idx = tid_to_idx[si]
        primary_pid[k]       = int(mc_pids[idx])
        primary_ke_MeV[k]    = float(mc_ke[idx])
        primary_start_pos[k] = mc_start[idx]
        primary_n[k]         = int((slice_id == si).sum())
        slice_member_trackids[k] = sorted(inv[si])

    return {
        "slice_id":              slice_id,
        "primary_trackid":       unique_slices,
        "primary_pid":           primary_pid,
        "primary_origin":        primary_origin,
        "primary_ke_MeV":        primary_ke_MeV,
        "primary_start_pos":     primary_start_pos,
        "primary_n_spacepoints": primary_n,
        "slice_member_trackids": slice_member_trackids,
        "ke_thresholds":         dict(ke_thresholds),
        "other_ke_threshold":    float(other_ke_threshold),
    }


def summarize_particle_labels(particle_info):
    """Return a short multi-line summary string for printing."""
    s = particle_info
    n_inst = len(s["primary_trackid"])
    n_assigned = int((s["slice_id"] != GHOST_SLICE_ID).sum())
    n_ghost = int((s["slice_id"] == GHOST_SLICE_ID).sum())
    # Per-PDG count
    from collections import Counter
    pdg_counts = Counter(int(p) for p in s["primary_pid"])
    lines = [
        f"particles: {n_inst} instances",
        f"spacepoints assigned to a particle: {n_assigned}",
        f"spacepoints unassigned (ghost/cosmic/orphan/sub-threshold-without-parent): {n_ghost}",
        "per-PDG instance count: " + ", ".join(
            f"{pdg}={c}" for pdg, c in sorted(pdg_counts.items())
        ),
    ]
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Legacy slice-label summary (unchanged below)
# ----------------------------------------------------------------------------

def summarize_slices(slice_info):
    """Return a short multi-line summary string for printing."""
    s = slice_info
    n_nu = int((s["primary_origin"] == 1).sum())
    n_cos = int((s["primary_origin"] == 2).sum())
    n_other = len(s["primary_trackid"]) - n_nu - n_cos
    n_ghost = int((s["slice_id"] == GHOST_SLICE_ID).sum())
    lines = [
        f"slices: {len(s['primary_trackid'])} "
        f"(nu={n_nu}, cosmic={n_cos}, other={n_other})",
        f"spacepoints with no slice (ghost/orphan): {n_ghost}",
    ]
    return "\n".join(lines)
