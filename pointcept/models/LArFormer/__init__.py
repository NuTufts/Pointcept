"""LArFormer — configurable Mask2Former-style model over named LArTPC token levels.

See `Pointcept/docs/LArFormer.md` for the design.
"""

from .builders import (
    BUILDERS,
    FragmentBuilder,
    FragmentContentEnricher,
    FragmentPool,
    LevelBuilder,
    LevelOutput,
    SpacepointBuilder,
    VoxelBuilder,
)
from .decoder import Mask2FormerDecoder
from .heads import PerTokenClsHead
from .losses import (
    LArFormerLoss,
    build_per_level_cls_target,
    build_per_level_gt,
    build_per_level_instance_mask,
)
from .cascade_filter import drop_empty_events, filter_batch_by_keep_mask
from .cascaded import CascadedSlicer
from .matcher import HungarianMatcher
from .model import LArFormer
from .tokenizer import CompositeTokenizer

__all__ = [
    # Builders + registry
    "BUILDERS",
    "LevelBuilder",
    "LevelOutput",
    "SpacepointBuilder",
    "VoxelBuilder",
    "FragmentBuilder",
    "FragmentPool",
    "FragmentContentEnricher",
    # Components
    "CompositeTokenizer",
    "Mask2FormerDecoder",
    "PerTokenClsHead",
    "HungarianMatcher",
    "LArFormerLoss",
    # Public GT helpers (used by the visualizer)
    "build_per_level_instance_mask",
    "build_per_level_cls_target",
    "build_per_level_gt",
    # Cascade (Stage-2 slicer wrapping a frozen deghoster)
    "CascadedSlicer",
    "filter_batch_by_keep_mask",
    "drop_empty_events",
    # Top-level
    "LArFormer",
]
