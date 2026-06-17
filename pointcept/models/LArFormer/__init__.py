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
from .cascade_particle_filter import (
    build_nu_keep_mask,
    filter_batch_for_particle_segmenter,
    recenter_coord_norm_to_per_event_centroid,
)
from .cascaded import CascadedSlicer
from .cascaded_particle import CascadedParticleSegmenter
from .matcher import HungarianMatcher
from .model import LArFormer
from .keypoint import LArFormerKeypoint
from .keypoint_heads import KeypointScoreHead, KeypointOffsetHead
from .keypoint_query import KeypointQueryHead, keypoint_query_loss
from .keypoint_refine import KeypointRefinementDecoder
from .keypoint_eval import (
    decode_dense_votes, decode_query_points, decode_nu_vertex,
    accumulate_metrics, format_metrics_table,
    query_points_to_typed, reconcile_keypoints, gt_keypoints_by_type,
    decode_event_keypoints, keypoint_arrays_for_h5, dedup_query_effective_argmax,
)
from .keypoint_particle_evaluator import LArFormerKeypointEvaluator
from .keypoint2_evaluator import LArFormerLevelKeypointEvaluator
from .keypoint2_cascade import CascadedKeypoint, predicted_masks_to_instances
from .keypoint_hooks import KeypointRegWarmupHook
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
    # Cascade (Stage-3 particle segmenter wrapping CascadedSlicer)
    "CascadedParticleSegmenter",
    "build_nu_keep_mask",
    "filter_batch_for_particle_segmenter",
    "recenter_coord_norm_to_per_event_centroid",
    # Top-level
    "LArFormer",
    # Keypoint module (Phase 1: dense per-SP score head)
    "LArFormerKeypoint",
    "KeypointScoreHead",
    "KeypointOffsetHead",
    # Keypoint module (Phase 2: query-based start/end/vertex head)
    "KeypointQueryHead",
    "keypoint_query_loss",
    "KeypointRefinementDecoder",
    # Keypoint module (Phase 4: decode + PPN-style metrics)
    "decode_dense_votes",
    "decode_query_points",
    "decode_nu_vertex",
    "accumulate_metrics",
    "format_metrics_table",
    "query_points_to_typed",
    "reconcile_keypoints",
    "gt_keypoints_by_type",
    "decode_event_keypoints",
    "keypoint_arrays_for_h5",
    "dedup_query_effective_argmax",
    "LArFormerKeypointEvaluator",
    "LArFormerLevelKeypointEvaluator",
    "CascadedKeypoint",
    "predicted_masks_to_instances",
    "KeypointRegWarmupHook",
]
