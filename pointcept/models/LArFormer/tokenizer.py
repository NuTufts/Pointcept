"""
CompositeTokenizer — instantiates one LevelBuilder per declared level and
runs them all on a single event's backbone features. Returns an ordered
dict[level_name → LevelOutput] keyed by the level's stable config name.

The decoder + loss consume that dict to dispatch cross-attention by name and
to build per-level GT masks / cls targets.
"""

from collections import OrderedDict
from typing import List, Sequence

import torch
import torch.nn as nn

from .builders import BUILDERS, LevelBuilder, LevelOutput


def _normalize_levels_cfg(levels_cfg: Sequence[dict]) -> List[dict]:
    """Validate + freeze a copy of the user's `levels` config block.

    Required keys per entry:
        name:        unique stable string key
        builder:     registry key for a LevelBuilder
        builder_cfg: kwargs forwarded to the builder ctor (default {})

    Other keys (notably `supervision`) are preserved verbatim so downstream
    consumers (LArFormerLoss, the per-level cls heads, the GT visualizer) can
    read them off the same normalized list.
    """
    out = []
    seen_names = set()
    for i, lc in enumerate(levels_cfg):
        if "name" not in lc:
            raise ValueError(f"levels[{i}] is missing 'name'")
        if "builder" not in lc:
            raise ValueError(f"levels[{i}] is missing 'builder'")
        name = str(lc["name"])
        if name in seen_names:
            raise ValueError(f"duplicate level name {name!r}")
        seen_names.add(name)
        entry = dict(lc)  # shallow copy preserves supervision et al
        entry["name"] = name
        entry["builder"] = str(lc["builder"])
        entry["builder_cfg"] = dict(lc.get("builder_cfg", {}))
        out.append(entry)
    return out


class CompositeTokenizer(nn.Module):
    """Owns one LevelBuilder per declared level.

    Args:
        levels_cfg: list of level dicts (see `_normalize_levels_cfg`).
        in_dim:     backbone output dim (forwarded to each builder).
        token_dim:  decoder token dim (forwarded to each builder).
    """

    def __init__(
        self,
        levels_cfg: Sequence[dict],
        in_dim: int,
        token_dim: int,
    ):
        super().__init__()
        self.levels_cfg = _normalize_levels_cfg(levels_cfg)
        self.in_dim = int(in_dim)
        self.token_dim = int(token_dim)

        # nn.ModuleDict preserves insertion order on iteration in py3.7+
        self.builders = nn.ModuleDict()
        for lc in self.levels_cfg:
            bcfg = dict(lc["builder_cfg"])
            bcfg.setdefault("in_dim", in_dim)
            bcfg.setdefault("token_dim", token_dim)
            builder = BUILDERS.build(dict(type=lc["builder"], **bcfg))
            if not isinstance(builder, LevelBuilder):
                raise TypeError(
                    f"builder {lc['builder']!r} did not produce a "
                    f"LevelBuilder (got {type(builder).__name__})"
                )
            builder.set_name(lc["name"])
            self.builders[lc["name"]] = builder

    @property
    def level_names(self) -> List[str]:
        return [lc["name"] for lc in self.levels_cfg]

    def forward(
        self,
        sp_feat: torch.Tensor,
        coord_norm: torch.Tensor,
        event_dict: dict,
    ) -> "OrderedDict[str, LevelOutput]":
        """Run every builder on this event. Returned dict preserves config order."""
        out: "OrderedDict[str, LevelOutput]" = OrderedDict()
        for name in self.level_names:
            out[name] = self.builders[name](sp_feat, coord_norm, event_dict)
        return out
