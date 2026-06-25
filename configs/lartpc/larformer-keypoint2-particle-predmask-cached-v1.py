"""LArFormer Keypoint v2 — predicted-mask training from a CACHED stage-1+2+3
cache (fast variant of larformer-keypoint2-particle-predmask-v1.py).

Same training as the live predicted-mask config — per-particle MAIN path on the
segmenter's PREDICTED masks, DN path on GT masks — but the predicted masks are
READ from the cache (precomputed by tools/augment_stage12_cache_pred_masks.py)
instead of running the frozen segmenter live each step. So NO second backbone in
VRAM and NO ~2x per-step cost.

PREREQUISITE: augment the (no-cap) cache with predicted masks first:
    for SPLIT in train val; do
      python tools/augment_stage12_cache_pred_masks.py \\
        --keypoint-config configs/lartpc/larformer-keypoint2-particle-v1.py \\
        --cache-root <CACHE_ROOT>/$SPLIT \\
        --particle-config configs/lartpc/larformer-particle-fullcascade-ptv3crosslevel.py \\
        --particle-weights <particle ckpt>
    done
"""

# The top-level config is larformer-keypoint2-slice-v1.py due to the staged keypoint config
# - larformer-keypoint2-slice-v1.py: top-level, defines the nu vertex prediction mask and the object/no-object mask
# - larformer-keypoint2-particle-v1.py: defines the per-particle keypoint prediction mask
# - larformer-keypoint2-particle-predmask-cached-v1.py: configures the dataset to use a cached stage 1+2+3 files
_base_ = ["./larformer-keypoint2-particle-v1.py"]

# MAIN path reads cached predicted masks (no live segmenter built/run); DN path
# still uses GT masks. dataset supplies them via particle_instances_per_event.
model = dict(
    particle_keypoint_cfg=dict(main_source="predicted_cached"),
)

# Turn on cached-predicted-mask loading for every split (the dataset then emits
# `particle_instances` → collate → particle_instances_per_event).
data = dict(
    train=dict(load_pred_instances=True),
    val=dict(load_pred_instances=True),
    test=dict(load_pred_instances=True),
)

save_path = "exp/larformer_keypoint2_particle_predmask_cached_v1"
