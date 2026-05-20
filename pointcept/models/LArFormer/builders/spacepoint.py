"""
SpacepointBuilder — identity level.

Tokens are a linear projection of the per-spacepoint backbone features;
coords are the spacepoints' own normalized positions; sp_to_level_id is
the identity arange.

The dominant per-pair mask loss is usually evaluated at this level
(set in the LArFormerLoss `primary_level` field).
"""

import torch
import torch.nn as nn

from .base import BUILDERS, LevelBuilder, LevelOutput


@BUILDERS.register_module()
class SpacepointBuilder(LevelBuilder):
    def __init__(self, in_dim: int, token_dim: int):
        super().__init__(in_dim=in_dim, token_dim=token_dim)
        self.proj = nn.Linear(in_dim, token_dim)

    def forward(
        self,
        sp_feat: torch.Tensor,
        coord_norm: torch.Tensor,
        event_dict: dict,
    ) -> LevelOutput:
        n = sp_feat.shape[0]
        tokens = self.proj(sp_feat)
        sp_to_level_id = torch.arange(n, device=sp_feat.device, dtype=torch.long)
        return LevelOutput(
            tokens=tokens,
            coords=coord_norm,
            sp_to_level_id=sp_to_level_id,
            name=self.name,
        )
