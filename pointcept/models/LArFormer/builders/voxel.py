"""
VoxelBuilder — quantize the spacepoint cloud onto a regular grid, mean-pool
backbone features per voxel, project to token_dim.

Voxelization is **model-side**: the builder derives voxel IDs from
`coord_norm` at every forward, using its configured `voxel_size_cm` and
the dataset's normalization scale. This lets a config flip resolutions
without touching the dataset / regenerating H5 caches.

Token coords are voxel-center coords in the same normalized frame as the
input `coord_norm`, so the decoder's shared (3)→(D) positional MLP can be
applied uniformly across all levels.
"""

import torch
import torch.nn as nn

from .base import BUILDERS, LevelBuilder, LevelOutput


@BUILDERS.register_module()
class VoxelBuilder(LevelBuilder):
    """
    Args:
        in_dim:        backbone feature dim
        token_dim:     output token dim
        voxel_size_cm: voxel edge length in detector cm
        coord_scale:   the same `coord_scale` the dataset used to build
                       coord_norm (default 179.55 — matches
                       `ShowerClusteringDataset.DEFAULT_COORD_SCALE`).
                       voxel_size_norm = voxel_size_cm / coord_scale.
    """

    def __init__(
        self,
        in_dim: int,
        token_dim: int,
        voxel_size_cm: float = 5.0,
        coord_scale: float = 179.55,
    ):
        super().__init__(in_dim=in_dim, token_dim=token_dim)
        self.voxel_size_cm = float(voxel_size_cm)
        self.coord_scale = float(coord_scale)
        self.voxel_size_norm = float(voxel_size_cm) / float(coord_scale)
        self.proj = nn.Linear(in_dim, token_dim)

    def _voxelize(self, coord_norm: torch.Tensor):
        """Quantize coord_norm onto the voxel grid.

        Returns:
            sp_to_level_id: (N,) long in [0, M)
            voxel_keys:     (M, 3) long — integer grid coords
        """
        v_int = torch.floor(coord_norm / self.voxel_size_norm).to(torch.long)
        # `unique` on a 2-D tensor along dim=0 with return_inverse gives us
        # voxel_keys (M, 3) and sp_to_level_id (N,) in one shot.
        voxel_keys, sp_to_level_id = torch.unique(
            v_int, dim=0, return_inverse=True,
        )
        return sp_to_level_id.to(torch.long), voxel_keys

    def forward(
        self,
        sp_feat: torch.Tensor,
        coord_norm: torch.Tensor,
        event_dict: dict,
    ) -> LevelOutput:
        n = sp_feat.shape[0]
        if n == 0:
            return LevelOutput(
                tokens=sp_feat.new_zeros(0, self.token_dim),
                coords=sp_feat.new_zeros(0, 3),
                sp_to_level_id=torch.zeros(0, dtype=torch.long,
                                           device=sp_feat.device),
                name=self.name,
            )

        sp_to_level_id, voxel_keys = self._voxelize(coord_norm)
        m = int(voxel_keys.shape[0])

        # Mean-pool features per voxel via scatter_add. Faster + lower memory
        # than torch_scatter and avoids the optional dependency.
        v_sum = sp_feat.new_zeros(m, self.in_dim)
        v_sum.index_add_(0, sp_to_level_id, sp_feat)
        v_count = sp_feat.new_zeros(m)
        v_count.index_add_(0, sp_to_level_id, sp_feat.new_ones(n))
        v_mean = v_sum / v_count.clamp(min=1.0).unsqueeze(-1)
        tokens = self.proj(v_mean)

        # Voxel centers in normalized coords
        coords = (voxel_keys.to(coord_norm.dtype) + 0.5) * self.voxel_size_norm
        return LevelOutput(
            tokens=tokens,
            coords=coords,
            sp_to_level_id=sp_to_level_id,
            name=self.name,
        )
