"""
Virtual Grid Generator for Shower Origin Prediction V3

Generates virtual grid points in 3D space and computes interpolated features
at those points from nearby real (measured) points. This enables the model to
predict origin scores at locations where no physical measurement exists,
critical for showers that originate outside the detector active volume.

Two grid resolutions:
- Dense grid: Fine spacing (2 cm) within a sphere around each shower centroid
- Sparse grid: Coarse spacing (10 cm) covering the full MicroBooNE TPC volume

Features at virtual points are computed via inverse-distance-weighted kNN
interpolation from real backbone features, augmented with learned positional
encodings.

Author: Claude Code Assistant
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# MicroBooNE TPC default bounds (cm)
# ============================================================================
DEFAULT_TPC_BOUNDS = dict(
    x_min=0.0,
    x_max=255.6,
    y_min=-116.5,
    y_max=116.5,
    z_min=0.5,
    z_max=1035.5,
)


# ============================================================================
# Helper functions
# ============================================================================

def generate_dense_grid(center, radius, spacing):
    """
    Generate 3D grid points within a sphere around center.

    Args:
        center: (3,) tensor, center of the sphere
        radius: float, radius of the sphere
        spacing: float, grid spacing

    Returns:
        grid_points: (M, 3) tensor of grid point coordinates
    """
    device = center.device
    dtype = center.dtype

    # Create 1D ranges for each axis
    n_steps = int(math.ceil(radius / spacing))
    offsets = torch.arange(-n_steps, n_steps + 1, device=device, dtype=dtype) * spacing

    # Create 3D meshgrid
    gx, gy, gz = torch.meshgrid(offsets, offsets, offsets, indexing="ij")
    grid = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=-1)  # (M_full, 3)

    # Filter to sphere
    dists = torch.norm(grid, dim=-1)
    mask = dists <= radius
    grid = grid[mask]  # (M, 3)

    # Shift to center
    grid = grid + center.unsqueeze(0)

    return grid


def generate_sparse_tpc_grid(bounds, spacing, device=None, dtype=None):
    """
    Generate coarse 3D grid covering TPC volume.

    Args:
        bounds: dict with x_min, x_max, y_min, y_max, z_min, z_max
        spacing: float, grid spacing
        device: torch device
        dtype: torch dtype

    Returns:
        grid_points: (M, 3) tensor of grid point coordinates
    """
    if device is None:
        device = torch.device("cpu")
    if dtype is None:
        dtype = torch.float32

    xs = torch.arange(bounds["x_min"], bounds["x_max"] + spacing, spacing,
                       device=device, dtype=dtype)
    ys = torch.arange(bounds["y_min"], bounds["y_max"] + spacing, spacing,
                       device=device, dtype=dtype)
    zs = torch.arange(bounds["z_min"], bounds["z_max"] + spacing, spacing,
                       device=device, dtype=dtype)

    gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing="ij")
    grid = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=-1)

    return grid


def knn_interpolate_features(virtual_coords, real_coords, real_features, k=8):
    """
    Interpolate features at virtual points from k nearest real points.

    Uses inverse-distance weighted average of k nearest neighbors.

    Args:
        virtual_coords: (V, 3) virtual point coordinates
        real_coords: (N, 3) real point coordinates
        real_features: (N, D) features at real points
        k: int, number of nearest neighbors

    Returns:
        interpolated: (V, D) interpolated features at virtual points
    """
    V = virtual_coords.shape[0]
    N = real_coords.shape[0]
    D = real_features.shape[1]

    # Clamp k to not exceed available real points
    k_actual = min(k, N)

    # Compute pairwise distances: (V, N)
    # Process in chunks to avoid OOM for large V * N
    chunk_size = 4096
    interpolated_chunks = []

    for start in range(0, V, chunk_size):
        end = min(start + chunk_size, V)
        v_chunk = virtual_coords[start:end]  # (chunk, 3)

        # Pairwise distances for this chunk
        dists = torch.cdist(v_chunk, real_coords)  # (chunk, N)

        # Find k nearest neighbors
        topk_dists, topk_idx = torch.topk(dists, k_actual, dim=-1, largest=False)  # (chunk, k)

        # Inverse-distance weights
        # Add small epsilon to avoid division by zero for coincident points
        eps = 1e-8
        inv_dists = 1.0 / (topk_dists + eps)  # (chunk, k)
        weights = inv_dists / inv_dists.sum(dim=-1, keepdim=True)  # (chunk, k), normalized

        # Gather neighbor features
        # topk_idx: (chunk, k) -> expand to (chunk, k, D)
        neighbor_feats = real_features[topk_idx.reshape(-1)].reshape(
            end - start, k_actual, D
        )  # (chunk, k, D)

        # Weighted average
        weighted = (weights.unsqueeze(-1) * neighbor_feats).sum(dim=1)  # (chunk, D)
        interpolated_chunks.append(weighted)

    interpolated = torch.cat(interpolated_chunks, dim=0)  # (V, D)
    return interpolated


# ============================================================================
# Modules
# ============================================================================

class PositionalEncoding3D(nn.Module):
    """
    Learned MLP that maps (x, y, z) coordinates to a feature vector.

    Uses sinusoidal positional features as input augmentation before the MLP:
    [x, y, z, sin(x/s1), cos(x/s1), sin(y/s1), cos(y/s1), sin(z/s1), cos(z/s1), ...]
    with multiple frequency scales for multi-resolution encoding.

    Args:
        pos_dim: Spatial coordinate dimension (default: 3)
        embed_dim: Output embedding dimension (default: 64)
        hidden_dim: Hidden layer dimension (default: 128)
        num_frequencies: Number of sinusoidal frequency bands (default: 8)
    """

    def __init__(self, pos_dim=3, embed_dim=64, hidden_dim=128, num_frequencies=8):
        super().__init__()
        self.pos_dim = pos_dim
        self.num_frequencies = num_frequencies

        # Frequency scales: logarithmically spaced from 1.0 to 512.0
        # These define the wavelengths of the sinusoidal encoding
        scales = torch.logspace(0.0, math.log10(512.0), num_frequencies)
        self.register_buffer("scales", scales)  # (num_frequencies,)

        # Input dimension: raw coords + sin/cos for each coord and each frequency
        input_dim = pos_dim + pos_dim * num_frequencies * 2

        # MLP: input_dim -> hidden -> hidden -> embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, coords):
        """
        Map coordinates to positional embeddings.

        Args:
            coords: (M, 3) spatial coordinates

        Returns:
            embeddings: (M, embed_dim)
        """
        # coords: (M, 3)
        features = [coords]

        for s in self.scales:
            # For each coordinate dimension, compute sin and cos at this frequency
            features.append(torch.sin(coords / s))  # (M, 3)
            features.append(torch.cos(coords / s))  # (M, 3)

        # Concatenate all features: (M, pos_dim + pos_dim * num_frequencies * 2)
        x = torch.cat(features, dim=-1)

        return self.mlp(x)


class VirtualGridGenerator(nn.Module):
    """
    Generate virtual grid points and compute their features via kNN interpolation.

    Creates a two-level grid:
    1. Dense grid near each shower centroid (fine resolution for precise origin localization)
    2. Sparse grid covering the full TPC volume (coarse coverage for distant origins)

    Features at virtual points are obtained by:
    - kNN interpolation from real backbone features (inverse-distance weighted)
    - Addition of learned positional encoding
    - Linear projection to match backbone feature dimension

    Args:
        backbone_dim: Feature dimension from backbone (496 for Sonata upcast)
        dense_spacing: Grid spacing near showers in cm (default: 2.0)
        dense_radius: Radius around shower centroid for dense grid in cm (default: 30.0)
        sparse_spacing: Coarse grid spacing for full TPC in cm (default: 10.0)
        tpc_bounds: Dict with TPC volume bounds (default: MicroBooNE)
        k_neighbors: Number of nearest neighbors for interpolation (default: 8)
        pos_embed_dim: Dimension of positional encoding (default: 64)
    """

    def __init__(
        self,
        backbone_dim,
        dense_spacing=2.0,
        dense_radius=30.0,
        sparse_spacing=10.0,
        tpc_bounds=None,
        k_neighbors=8,
        pos_embed_dim=64,
    ):
        super().__init__()

        self.backbone_dim = backbone_dim
        self.dense_spacing = dense_spacing
        self.dense_radius = dense_radius
        self.sparse_spacing = sparse_spacing
        self.k_neighbors = k_neighbors

        # TPC bounds (default: MicroBooNE)
        if tpc_bounds is None:
            tpc_bounds = DEFAULT_TPC_BOUNDS
        self.tpc_bounds = tpc_bounds

        # Positional encoding
        self.pos_encoding = PositionalEncoding3D(
            pos_dim=3,
            embed_dim=pos_embed_dim,
            hidden_dim=128,
        )

        # Projection to combine interpolated features + positional encoding
        # into backbone_dim
        self.feature_proj = nn.Sequential(
            nn.Linear(backbone_dim + pos_embed_dim, backbone_dim),
            nn.LayerNorm(backbone_dim),
            nn.GELU(),
        )

        # Generate and cache the sparse TPC grid (it is fixed)
        sparse_grid = generate_sparse_tpc_grid(
            bounds=self.tpc_bounds,
            spacing=self.sparse_spacing,
        )
        self.register_buffer("sparse_grid", sparse_grid)  # (M_sparse, 3)

    def forward(self, real_coords, real_features, shower_centroids, batch_offsets=None):
        """
        Generate virtual grid and compute features.

        Args:
            real_coords: (N, 3) real point coordinates
            real_features: (N, D) backbone features at real points
            shower_centroids: list of (3,) tensors, one centroid per shower in this event
            batch_offsets: optional (B,) tensor of cumulative point counts per batch element

        Returns:
            virtual_coords: (V, 3) virtual point coordinates
            virtual_features: (V, D) features at virtual points
            virtual_batch_indices: (V,) batch index for each virtual point (if batched, else None)
        """
        device = real_coords.device
        dtype = real_coords.dtype

        # ------------------------------------------------------------------
        # 1. Generate dense grids near each shower centroid
        # ------------------------------------------------------------------
        dense_grids = []
        for centroid in shower_centroids:
            centroid = centroid.to(device=device, dtype=dtype)
            dg = generate_dense_grid(centroid, self.dense_radius, self.dense_spacing)
            dense_grids.append(dg)

        if len(dense_grids) > 0:
            dense_points = torch.cat(dense_grids, dim=0)  # (M_dense, 3)
        else:
            dense_points = torch.empty(0, 3, device=device, dtype=dtype)

        # ------------------------------------------------------------------
        # 2. Get sparse grid (cached, move to device if needed)
        # ------------------------------------------------------------------
        sparse_points = self.sparse_grid.to(device=device, dtype=dtype)  # (M_sparse, 3)

        # ------------------------------------------------------------------
        # 3. Remove sparse points that fall within any dense region
        #    (dense overrides sparse within dense_radius of any centroid)
        # ------------------------------------------------------------------
        if dense_points.shape[0] > 0 and sparse_points.shape[0] > 0:
            keep_sparse = torch.ones(sparse_points.shape[0], dtype=torch.bool, device=device)
            for centroid in shower_centroids:
                centroid = centroid.to(device=device, dtype=dtype)
                dist_to_centroid = torch.norm(
                    sparse_points - centroid.unsqueeze(0), dim=-1
                )
                keep_sparse = keep_sparse & (dist_to_centroid > self.dense_radius)
            sparse_points = sparse_points[keep_sparse]

        # ------------------------------------------------------------------
        # 4. Combine dense and sparse grids
        # ------------------------------------------------------------------
        if dense_points.shape[0] > 0 and sparse_points.shape[0] > 0:
            virtual_coords = torch.cat([dense_points, sparse_points], dim=0)
        elif dense_points.shape[0] > 0:
            virtual_coords = dense_points
        else:
            virtual_coords = sparse_points

        # If no virtual points generated, return empty tensors
        if virtual_coords.shape[0] == 0:
            return (
                torch.empty(0, 3, device=device, dtype=dtype),
                torch.empty(0, self.backbone_dim, device=device, dtype=dtype),
                None,
            )

        # ------------------------------------------------------------------
        # 5. kNN interpolation: features at virtual points from real points
        # ------------------------------------------------------------------
        interpolated_feats = knn_interpolate_features(
            virtual_coords, real_coords, real_features, k=self.k_neighbors
        )  # (V, D)

        # ------------------------------------------------------------------
        # 6. Positional encoding at virtual point locations
        # ------------------------------------------------------------------
        pos_feats = self.pos_encoding(virtual_coords)  # (V, pos_embed_dim)

        # ------------------------------------------------------------------
        # 7. Project combined features to backbone_dim
        # ------------------------------------------------------------------
        combined = torch.cat([interpolated_feats, pos_feats], dim=-1)  # (V, D + pos_embed_dim)
        virtual_features = self.feature_proj(combined)  # (V, D)

        # ------------------------------------------------------------------
        # 8. Batch indices (simple single-batch case for now)
        # ------------------------------------------------------------------
        virtual_batch_indices = None
        if batch_offsets is not None:
            # For multi-batch: assign all virtual points to batch 0
            # TODO: Extend to per-batch virtual grid generation
            virtual_batch_indices = torch.zeros(
                virtual_coords.shape[0], dtype=torch.long, device=device
            )

        return virtual_coords, virtual_features, virtual_batch_indices
