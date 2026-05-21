"""LArFormer token refiners. Importing this package registers all v1 refiners.

The TokenRefiner abstraction sits between the CompositeTokenizer (which
emits one static `LevelOutput` per level by pooling backbone features) and
the Mask2FormerDecoder (which cross-attends queries to those level tokens).

Without a refiner, level tokens are exactly the pooled backbone features —
two distant voxels with content-similar member SPs are indistinguishable
to the mask head except through `pos_emb`. A refiner gets a chance to mix
information (within a level, across levels) so that the level tokens
themselves encode the spatial/contextual structure the mask head needs.

See `docs/LArFormer.md` §15 for the failure-mode analysis motivating the
abstraction and the menu of refiner options.
"""

from .base import REFINERS, TokenRefiner, IdentityRefiner, build_token_refiner
from .cross_level import CrossLevelAttn
from .per_level_sa import PerLevelSelfAttn

__all__ = [
    "REFINERS",
    "TokenRefiner",
    "IdentityRefiner",
    "PerLevelSelfAttn",
    "CrossLevelAttn",
    "build_token_refiner",
]
