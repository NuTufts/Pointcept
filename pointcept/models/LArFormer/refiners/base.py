"""TokenRefiner ABC + REFINERS registry + IdentityRefiner + builder.

`TokenRefiner` is a drop-in transform on the tokenizer's output:

    OrderedDict[level_name → LevelOutput]  ->  OrderedDict[level_name → LevelOutput]

The contract is: same keys in, same keys out; per-level (`coords`,
`sp_to_level_id`, `name`) are preserved; only `tokens` may change. Token
count `M_level` MUST stay the same (the decoder's mask logits and the
loss's per-level GT masks are indexed by token position).
"""

from collections import OrderedDict
from typing import Optional

import torch.nn as nn

from pointcept.utils.registry import Registry

from ..builders import LevelOutput


REFINERS = Registry("larformer_refiners")


class TokenRefiner(nn.Module):
    """ABC for token refiners.

    Subclasses implement `forward(levels) -> levels`.

    Naming: "refiner" rather than "decoder" to avoid collision with the
    existing Mask2Former decoder that operates on queries. The refiner
    operates on the token/key side — between tokenizer and decoder.
    """

    def forward(
        self,
        levels: "OrderedDict[str, LevelOutput]",
    ) -> "OrderedDict[str, LevelOutput]":
        raise NotImplementedError


@REFINERS.register_module()
class IdentityRefiner(TokenRefiner):
    """Pass-through. Equivalent to LArFormer pre-refiner behavior (the
    decoder consumes static, mean-pooled backbone features directly)."""

    def forward(self, levels):
        return levels


def build_token_refiner(cfg: Optional[dict]) -> TokenRefiner:
    """Construct a TokenRefiner from its config dict.

    `cfg=None` or `cfg={"type": "IdentityRefiner"}` returns a pass-through
    (matches pre-refiner behavior; safe default).
    """
    if cfg is None:
        return IdentityRefiner()
    cfg = dict(cfg)
    typ = cfg.pop("type", "IdentityRefiner")
    cls = REFINERS.get(typ)
    if cls is None:
        raise KeyError(
            f"unknown TokenRefiner type {typ!r}; registered: "
            f"{list(REFINERS.module_dict.keys())!r}"
        )
    return cls(**cfg)
