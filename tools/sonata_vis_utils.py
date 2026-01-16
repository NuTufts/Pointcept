"""
Shared utilities for Sonata visualization scripts.

This module provides common functions used across visualization tools
to ensure consistent behavior (e.g., label assignment, color schemes).
"""

import numpy as np


# LArTPC particle class definitions
CLASS_NAMES = [
    "electron",   # 0: e+/e-
    "muon",       # 1: mu+/mu-
    "pion",       # 2: pi+/pi-
    "proton",     # 3: proton
    "gamma",      # 4: photon
    "michel",     # 5: michel
    "delta",      # 6: delta
    "led",        # 7: low energy deposit
    "ghost",      # 8: ghost
    "other"       # 9: everything else
]
NUM_CLASSES = len(CLASS_NAMES)
GHOST_LABEL = 8

# High-contrast color palette for particle classes (ColorBrewer Set1-based)
CLASS_COLORS = {
    0: "#E41A1C",  # electron - red
    1: "#377EB8",  # muon - blue
    2: "#4DAF4A",  # pion - green
    3: "#FF7F00",  # proton - orange
    4: "#984EA3",  # gamma - purple
    5: "#F682E9",  # michel - pink
    6: "#673A91",  # delta - maroon
    7: "#178787",  # led - teal
    8: "#999999",  # ghost - gray
    9: "#999999",  # other - gray
}

# RGBA versions for plotly (with alpha)
CLASS_COLORS_RGBA = {
    0: "rgba(228, 26, 28, 1.0)",   # electron - red
    1: "rgba(55, 126, 184, 1.0)",  # muon - blue
    2: "rgba(77, 175, 74, 1.0)",   # pion - green
    3: "rgba(255, 127, 0, 1.0)",   # proton - orange
    4: "rgba(152, 78, 163, 1.0)",  # gamma - purple
    5: "rgba(153, 153, 153, 1.0)", # ghost - gray
    6: "rgba(153, 153, 153, 1.0)", # ghost - gray
    7: "rgba(153, 153, 153, 1.0)", # ghost - gray
    8: "rgba(153, 153, 153, 1.0)", # ghost - gray
    9: "rgba(153, 153, 153, 1.0)", # ghost - gray
}

# SSNet label definitions (different indexing than CLASS_NAMES)
# These are the labels stored in HDF5 ssnet_label field
SSNET_CLASS_NAMES = {
    0: 'bg',
    1: 'electron',
    2: 'photon',
    3: 'muon',
    4: 'proton',
    5: 'pion',
    6: 'michel',
    7: 'delta',
    8: 'led',
    9: 'other'
}

SSNET_CLASS_COLORS = {
    0: 'rgba(50,50,50,1.0)',    # background
    1: 'rgba(255,0,0,1.0)',     # electron
    2: 'rgba(200,125,0,1)',     # photon
    3: 'rgba(0,0,255,1)',       # muon
    4: 'rgba(0,125,255,1)',     # proton
    5: 'rgba(125,0,255,1)',     # pion/kaon
    6: 'rgba(255,51,255,1)',    # shower michel
    7: 'rgba(125,100,255,1)',   # shower delta
    8: 'rgba(125,200,255,1)',   # low energy deposits
    9: 'rgba(125,125,0,1)',     # other
}


def assign_labels_to_output_points(
    output_coords: np.ndarray,
    input_coords: np.ndarray,
    input_labels: np.ndarray,
    grid_size: float = 0.25,
    threshold_factor: float = 4.0,
    ghost_label: int = GHOST_LABEL,
    use_gpu: bool = True,
    verbose: bool = False,
) -> np.ndarray:
    """
    Assign particle labels to output (pooled) points based on input (full resolution) labels.

    This function addresses the label assignment problem that arises when the model
    pools/downsamples points. An output point may have both ghost and non-ghost input
    points nearby. We prioritize non-ghost labels when available within a threshold.

    Algorithm:
        1. Split input points into ghost and non-ghost sets
        2. Build nearest neighbor index from non-ghost points only
        3. For each output point, find nearest non-ghost input point
        4. Assign non-ghost label if within threshold distance
        5. Otherwise, assign ghost label

    Args:
        output_coords: Output point coordinates, shape (N_out, 3)
        input_coords: Input point coordinates, shape (N_in, 3)
        input_labels: Input point labels, shape (N_in,)
        grid_size: Voxelization grid size (default 0.25 cm)
        threshold_factor: Multiplier for grid_size to get distance threshold (default 4.0)
        ghost_label: Label value for ghost points (default 5)
        use_gpu: If True, use cuML GPU-accelerated KNN; else use scipy (default True)
        verbose: If True, print label assignment statistics (default False)

    Returns:
        labels: Assigned labels for output points, shape (N_out,)
    """
    n_output = len(output_coords)
    label_threshold = grid_size * threshold_factor

    # Split input points into ghost and non-ghost
    non_ghost_mask = input_labels != ghost_label
    non_ghost_coords = input_coords[non_ghost_mask]
    non_ghost_labels = input_labels[non_ghost_mask]

    # Initialize all labels as ghost
    labels = np.full(n_output, ghost_label, dtype=np.int64)

    if len(non_ghost_coords) == 0:
        if verbose:
            print(f"No non-ghost points found, all {n_output} output points labeled as ghost")
        return labels

    # Find nearest non-ghost point for each output point
    if use_gpu:
        try:
            from cuml.neighbors import NearestNeighbors

            nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
            nn.fit(non_ghost_coords.astype(np.float32))
            distances, indices = nn.kneighbors(output_coords.astype(np.float32))

            # cuML returns 2D arrays even for k=1
            distances = distances.flatten()
            indices = indices.flatten()
        except ImportError:
            if verbose:
                print("cuML not available, falling back to scipy")
            use_gpu = False

    if not use_gpu:
        from scipy.spatial import cKDTree

        tree = cKDTree(non_ghost_coords)
        distances, indices = tree.query(output_coords, k=1)

    # Assign non-ghost label if within threshold distance
    within_threshold = distances <= label_threshold
    labels[within_threshold] = non_ghost_labels[indices[within_threshold]]

    if verbose:
        n_assigned = within_threshold.sum()
        print(f"Label assignment: {n_assigned} non-ghost, {n_output - n_assigned} ghost "
              f"(threshold={label_threshold:.3f}, grid_size={grid_size:.3f}, factor={threshold_factor})")

    return labels


def get_class_color(class_idx: int, format: str = "hex") -> str:
    """
    Get color for a particle class.

    Args:
        class_idx: Class index (0-5)
        format: "hex" for matplotlib, "rgba" for plotly

    Returns:
        Color string in requested format
    """
    if format == "rgba":
        return CLASS_COLORS_RGBA.get(class_idx, "rgba(0, 0, 0, 1.0)")
    return CLASS_COLORS.get(class_idx, "#000000")


def get_class_name(class_idx: int) -> str:
    """Get name for a particle class."""
    if 0 <= class_idx < len(CLASS_NAMES):
        return CLASS_NAMES[class_idx]
    return f"unknown({class_idx})"
