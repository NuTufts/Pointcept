"""
Shower fragment merger — groups reco shower fragments into reconstructed
particle showers using predicted origin proximity and cone-based merging.

All operations use detector coordinates (cm). No PyTorch dependency.

Algorithm:
  1. Group fragments by predicted class (merge only within same class).
  2. Within each class, cluster predicted origins by fixed-radius proximity.
  3. For each cluster: identify trunk fragment, compute PCA shower axis,
     define a cone, merge fragments whose points overlap the cone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FragmentInfo:
    """Per-fragment data after inference, all coordinates in detector cm."""

    fragment_idx: int
    pred_origin_cm: np.ndarray          # (3,) predicted origin peak
    pred_class: int                     # 0-4
    pred_scores: np.ndarray             # (N,) origin scores per model point
    pred_coords_cm: np.ndarray          # (N, 3) model output coords in cm
    raw_coords_cm: np.ndarray           # (M, 3) raw spacepoints in cm
    trackid: int
    pid: int
    gt_origin_cm: np.ndarray            # (3,)
    gt_origin_type: int
    gt_pret0_origin: np.ndarray         # (4,) x,y,z,t true MC origin
    istrunk: int
    n_points_raw: int


@dataclass
class MergedShower:
    """A reconstructed shower = a group of merged fragments."""

    shower_id: int
    trunk_fragment_idx: int
    fragment_indices: List[int]
    predicted_origin_cm: np.ndarray     # (3,) cluster mean origin
    shower_axis: np.ndarray             # (3,) PCA first component (unit)
    start_point_cm: np.ndarray          # (3,)
    pred_class: int


@dataclass
class MergerConfig:
    """Tunable parameters for the fragment merger."""

    origin_proximity_radius: float = 5.0    # cm
    cone_half_angle: float = 20.0           # degrees
    cone_max_length: float = 300.0          # cm
    min_points_fraction_in_cone: float = 0.5


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def merge_fragments(
    fragments: List[FragmentInfo],
    config: MergerConfig | None = None,
) -> Tuple[List[MergedShower], List[int]]:
    """Merge fragments into reconstructed showers.

    Returns
    -------
    merged_showers : list[MergedShower]
    unmerged_fragment_indices : list[int]
        Indices into *fragments* that were not assigned to any shower.
    """
    if config is None:
        config = MergerConfig()

    if not fragments:
        return [], []

    # Group fragments by predicted class
    class_groups: dict[int, list[int]] = {}
    for i, frag in enumerate(fragments):
        class_groups.setdefault(frag.pred_class, []).append(i)

    merged_showers: list[MergedShower] = []
    assigned: set[int] = set()
    shower_id_counter = 0

    for pred_class, frag_indices in class_groups.items():
        if len(frag_indices) == 0:
            continue

        # Cluster predicted origins within this class
        origins = np.array(
            [fragments[i].pred_origin_cm for i in frag_indices]
        )
        clusters = _cluster_origins(origins, config.origin_proximity_radius)

        for cluster_local_ids in clusters:
            # Map local cluster indices back to global fragment indices
            cluster_global = [frag_indices[j] for j in cluster_local_ids]

            # Iteratively extract showers from this cluster.
            # After merging one shower (trunk + cone members), remove those
            # fragments and repeat on the remainder.  This handles cases
            # like pi0 -> 2 photons where two showers share a common origin.
            remaining = list(cluster_global)

            while remaining:
                # Mean predicted origin for remaining fragments
                cluster_origins = np.array(
                    [fragments[gi].pred_origin_cm for gi in remaining]
                )
                mean_origin = cluster_origins.mean(axis=0)

                # Identify trunk fragment
                trunk_global = _identify_trunk(
                    fragments, remaining, mean_origin
                )

                # Compute shower axis from trunk PCA
                trunk_frag = fragments[trunk_global]
                axis = _compute_shower_axis(trunk_frag.raw_coords_cm)

                # Find start point
                start_point = _find_start_point(
                    trunk_frag.raw_coords_cm,
                    mean_origin,
                    all_pred_scores=[fragments[gi].pred_scores for gi in remaining],
                    all_pred_coords_cm=[fragments[gi].pred_coords_cm for gi in remaining],
                )

                # Orient axis away from origin (toward shower body)
                trunk_centroid = trunk_frag.raw_coords_cm.mean(axis=0)
                if np.dot(axis, trunk_centroid - start_point) < 0:
                    axis = -axis

                # Cone membership test for non-trunk fragments
                merged_indices = [trunk_global]
                for gi in remaining:
                    if gi == trunk_global:
                        continue
                    frag = fragments[gi]
                    in_cone = _test_cone_membership(
                        frag.raw_coords_cm,
                        apex=start_point,
                        axis=axis,
                        half_angle_deg=config.cone_half_angle,
                        max_length=config.cone_max_length,
                    )
                    fraction_in_cone = in_cone.sum() / max(len(in_cone), 1)
                    if fraction_in_cone >= config.min_points_fraction_in_cone:
                        merged_indices.append(gi)

                shower = MergedShower(
                    shower_id=shower_id_counter,
                    trunk_fragment_idx=trunk_global,
                    fragment_indices=merged_indices,
                    predicted_origin_cm=mean_origin,
                    shower_axis=axis,
                    start_point_cm=start_point,
                    pred_class=pred_class,
                )
                merged_showers.append(shower)
                assigned.update(merged_indices)
                shower_id_counter += 1

                # Remove merged fragments from remaining
                merged_set = set(merged_indices)
                remaining = [gi for gi in remaining if gi not in merged_set]

                # If only the trunk was merged (no additional fragments joined),
                # and there are remaining fragments, they each failed the cone
                # test against this trunk.  Pick a new trunk from the remainder
                # on the next iteration. But if a single-fragment "shower" was
                # created and nothing else changed, the next iteration will
                # pick a different trunk -- so we just continue.

    unmerged = [i for i in range(len(fragments)) if i not in assigned]
    return merged_showers, unmerged


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _cluster_origins(
    origins: np.ndarray,
    radius: float,
) -> list[list[int]]:
    """Fixed-radius clustering of origin points.

    Uses a greedy approach: pick the first unvisited point, find all
    neighbours within *radius*, form a cluster, repeat.

    Returns list of clusters, each a list of indices into *origins*.
    """
    n = len(origins)
    if n == 0:
        return []

    tree = cKDTree(origins)
    visited = np.zeros(n, dtype=bool)
    clusters: list[list[int]] = []

    for i in range(n):
        if visited[i]:
            continue
        # Find all points within radius of point i
        neighbours = tree.query_ball_point(origins[i], radius)
        # Only include unvisited neighbours
        cluster = [j for j in neighbours if not visited[j]]
        for j in cluster:
            visited[j] = True
        clusters.append(cluster)

    return clusters


def _identify_trunk(
    fragments: list[FragmentInfo],
    cluster_indices: list[int],
    cluster_mean_origin: np.ndarray,
) -> int:
    """Return the fragment index whose closest point to *cluster_mean_origin*
    is smallest.  This fragment is the trunk."""
    best_idx = cluster_indices[0]
    best_dist = np.inf

    for gi in cluster_indices:
        coords = fragments[gi].raw_coords_cm
        if len(coords) == 0:
            continue
        dists = np.linalg.norm(coords - cluster_mean_origin, axis=1)
        min_dist = dists.min()
        if min_dist < best_dist:
            best_dist = min_dist
            best_idx = gi

    return best_idx


def _compute_shower_axis(trunk_coords: np.ndarray) -> np.ndarray:
    """PCA on trunk spacepoints; returns the first principal component
    as a unit vector."""
    if len(trunk_coords) < 2:
        return np.array([1.0, 0.0, 0.0])

    centroid = trunk_coords.mean(axis=0)
    centered = trunk_coords - centroid

    # Covariance matrix
    cov = centered.T @ centered / (len(centered) - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)

    # eigh returns eigenvalues in ascending order; last = largest
    axis = eigvecs[:, -1].astype(np.float64)
    norm = np.linalg.norm(axis)
    if norm > 0:
        axis /= norm

    return axis


def _find_start_point(
    trunk_coords: np.ndarray,
    mean_origin: np.ndarray,
    all_pred_scores: list[np.ndarray] | None = None,
    all_pred_coords_cm: list[np.ndarray] | None = None,
    disconnected_threshold: float = 30.0,
) -> np.ndarray:
    """Start point = trunk point closest to predicted origin.

    If the origin is far from all trunk points (> *disconnected_threshold* cm),
    fall back to the highest origin_score point across all fragments in the
    cluster.
    """
    if len(trunk_coords) == 0:
        return mean_origin.copy()

    dists = np.linalg.norm(trunk_coords - mean_origin, axis=1)
    min_idx = int(np.argmin(dists))
    min_dist = dists[min_idx]

    if min_dist <= disconnected_threshold:
        return trunk_coords[min_idx].copy()

    # Disconnected — use highest origin score point across all fragments
    if all_pred_scores is not None and all_pred_coords_cm is not None:
        best_score = -np.inf
        best_coord = trunk_coords[min_idx].copy()
        for scores, coords in zip(all_pred_scores, all_pred_coords_cm):
            if len(scores) == 0:
                continue
            peak_idx = int(np.argmax(scores))
            if scores[peak_idx] > best_score:
                best_score = scores[peak_idx]
                best_coord = coords[peak_idx].copy()
        return best_coord

    return trunk_coords[min_idx].copy()


def _test_cone_membership(
    points: np.ndarray,
    apex: np.ndarray,
    axis: np.ndarray,
    half_angle_deg: float,
    max_length: float,
) -> np.ndarray:
    """Test which *points* fall inside a cone.

    Parameters
    ----------
    apex : (3,) cone apex (start point of shower)
    axis : (3,) unit direction vector (pointing away from origin)
    half_angle_deg : half-angle of the cone in degrees
    max_length : maximum distance from apex along the axis

    Returns
    -------
    mask : (M,) boolean array
    """
    if len(points) == 0:
        return np.zeros(0, dtype=bool)

    cos_threshold = np.cos(np.radians(half_angle_deg))

    # Vector from apex to each point
    diff = points - apex  # (M, 3)

    # Projection along axis
    proj = diff @ axis  # (M,)

    # Distance from axis
    proj_vec = np.outer(proj, axis)  # (M, 3)
    perp = diff - proj_vec  # (M, 3)
    perp_dist = np.linalg.norm(perp, axis=1)  # (M,)

    # Distance from apex to point
    dist = np.linalg.norm(diff, axis=1)  # (M,)

    # Cosine of angle between diff and axis
    # Avoid division by zero
    safe_dist = np.maximum(dist, 1e-9)
    cos_angle = proj / safe_dist

    # Cone membership: positive projection, within angle, within length
    mask = (proj >= 0) & (cos_angle >= cos_threshold) & (proj <= max_length)

    return mask
