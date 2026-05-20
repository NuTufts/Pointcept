"""
LevelBuilder ABC + LevelOutput dataclass + BUILDERS registry.

A `LevelBuilder` is a small nn.Module that consumes the per-event
backbone-features + coords (+ optional per-event auxiliary fields) and emits
one "level" of tokens for the Mask2Former-style decoder. See
`Pointcept/docs/LArFormer.md` for the design rationale.

`LevelOutput.sp_to_level_id` is the only mechanism the rest of LArFormer
uses to lift per-spacepoint truth arrays into per-level GT masks / class
targets. Entries of −1 mark spacepoints that do not map to any token at
this level (e.g. spacepoints outside any DBSCAN fragment).
"""

from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from pointcept.utils.registry import Registry


BUILDERS = Registry("larformer_builders")


@dataclass
class LevelOutput:
    """Per-event output of one LevelBuilder.

    Attributes:
        tokens:           (M, D) per-token feature vector
        coords:           (M, 3) per-token position in normalized coords,
                           same frame as spacepoint coord_norm
        sp_to_level_id:   (N,) long. For each spacepoint, the index in
                           [0, M) of the level token it maps to, or −1
                           if the spacepoint is unmapped at this level.
        name:             stable string key (matches the level's config name)
    """
    tokens: torch.Tensor
    coords: torch.Tensor
    sp_to_level_id: torch.Tensor
    name: str

    @property
    def n_tokens(self) -> int:
        return int(self.tokens.shape[0])

    @property
    def valid_sp_mask(self) -> torch.Tensor:
        """(N,) bool: True iff this spacepoint maps to a level token."""
        return self.sp_to_level_id >= 0


class LevelBuilder(nn.Module):
    """ABC for per-event level builders.

    Subclasses implement `forward(sp_feat, coord_norm, event_dict)` returning a
    `LevelOutput`. The builder owns its projection from backbone dim to
    `token_dim`. `name` is supplied by the composite tokenizer (matches the
    config-level "name" field).

    `event_dict` carries any per-event auxiliary tensors that specific
    builders need (e.g. fragment indices for the fragment builder). Builders
    must ignore keys they don't recognize so that the same dict can be passed
    uniformly across configs.
    """

    def __init__(self, in_dim: int, token_dim: int):
        super().__init__()
        self.in_dim = int(in_dim)
        self.token_dim = int(token_dim)
        self._name: Optional[str] = None

    @property
    def name(self) -> str:
        if self._name is None:
            raise RuntimeError(
                "LevelBuilder.name accessed before composite tokenizer set it"
            )
        return self._name

    def set_name(self, name: str) -> None:
        self._name = str(name)

    @abstractmethod
    def forward(
        self,
        sp_feat: torch.Tensor,
        coord_norm: torch.Tensor,
        event_dict: dict,
    ) -> LevelOutput:
        ...
