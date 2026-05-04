"""Shower-clustering Mask2Former model components.

See pointcept/docs/shower_clustering_design.md.

Phase 3 (this file): tokenizer — turns per-spacepoint backbone features into
fragment, voxel, and spacepoint token sets that the Mask2Former decoder
will cross-attend over.
"""

from .tokenizer import (
    FragmentPool,
    VoxelPool,
    FragmentPositionalEncoder,
    VoxelPositionalEncoder,
    ShowerClusteringTokenizer,
)
from .decoder import (
    QueryPositionalEncoder,
    Mask2FormerDecoder,
    DEFAULT_SCALE_PATTERN,
)
from .matcher import HungarianMatcher
from .losses import ShowerClusteringLoss
from .model import ShowerClusteringMask2Former
# NOTE: ShowerClusteringTrainer is NOT imported here. It depends on
# pointcept.engines.train.TRAINERS, which doesn't exist yet at this point in
# the import chain (engines.train imports pointcept.models before defining
# TRAINERS, so importing the trainer here triggers a circular import).
# The training config (configs/lartpc/shower-cluster-sonata-v1.py) imports it
# explicitly, which runs after engines.train is fully initialized.

__all__ = [
    "FragmentPool",
    "VoxelPool",
    "FragmentPositionalEncoder",
    "VoxelPositionalEncoder",
    "ShowerClusteringTokenizer",
    "QueryPositionalEncoder",
    "Mask2FormerDecoder",
    "DEFAULT_SCALE_PATTERN",
    "HungarianMatcher",
    "ShowerClusteringLoss",
    "ShowerClusteringMask2Former",
]
