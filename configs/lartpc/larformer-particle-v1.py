"""LArFormer Stage 3 — particle segmenter.

Wraps the production Stage-1+2 cascade (LoRA deghoster + ptv3hybrid
crosslevel slicer) and adds a Stage-3 LArFormer that instance-segments
the nu-candidate spacepoints into per-particle masks.

This config is BOTH the S3.2 benchmark target (loaded by
`tools/benchmark_larformer_s3_cascade.py`) AND the eventual S3.3
training config. See `docs/LArFormer_particlesegment_stage.md` for the
design.

Stage 2 → Stage 3 boundary (set in the outer CascadedParticleSegmenter):
    nu_class_id              = 0    (matches the Stage 2 slicer)
    mask_prob_threshold      = 0.5  (τ_loose — start at 0.5; §3 / S3.9
                                       sweeps 0.3 / 0.5 / 0.7)
    recenter_to_slice_centroid = True  (§1f)
    freeze_cascaded_slicer   = True

Stage 3 LArFormer:
    7-class taxonomy {e±, γ, μ±, π±, p, other_track, no_object}
    K = 32 queries (~4× the typical particle count per event)
    enable_origin_head = True  (per-particle starting point regression)
    mixed_query_selection ON, mask_denoising ON  (mirrors Stage 2)
    backbone: Sonata-v1m1 (frozen), same checkpoint as Stage 2
"""

_base_ = ["../_base_/default_runtime.py"]

# Side-effect: register LArFormerTrainer + LArFormerSlicerEvaluator.
from pointcept.models.LArFormer import trainer as _larformer_trainer_module
from pointcept.models.LArFormer import evaluator as _larformer_evaluator_module
del _larformer_trainer_module
del _larformer_evaluator_module

# =============================================================================
# Paths — set to your cluster's locations or override at run time.
# =============================================================================
#TRAIN_FILE_LIST = "/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/lartpc_data_prep/lantern_scripts/h5lists/h5list_mcall_lantern_train.txt"
#VAL_FILE_LIST   = "/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/lartpc_data_prep/lantern_scripts/h5lists/h5list_mcall_lantern_val.txt"
#TRAIN_FILE_LIST = "/home/twongjirad/working/larbys/gen2/container_u22/Pointcept/devdata_mergedh5_pi0filter_10files.txt"
#VAL_FILE_LIST   = "/home/twongjirad/working/larbys/gen2/container_u22/Pointcept/devdata_mergedh5_pi0filter_10files.txt"
TRAIN_FILE_LIST = "/home/twongjirad/working/larbys/gen2/container_u22/Pointcept/exp/cache_stage12_devdata/train/"
VAL_FILE_LIST   = "/home/twongjirad/working/larbys/gen2/container_u22/Pointcept/exp/cache_stage12_devdata/val/"

# Stage-1 LoRA deghoster checkpoint (BOTH backbone + LoRA adapter weights).
deghoster_weight = "/home/twongjirad/working/larbys/gen2/container_u22/Pointcept/sonata/lora_deghost_v6_hasmatch/model/epoch_30.pth"

# Sonata-v1m1 pretrain used by:
#   - the slicer's frozen backbone (Stage 2)
#   - the particle segmenter's frozen backbone (Stage 3)
sonata_pretrain_weight = (
    "/home/twongjirad/working/larbys/gen2/container_u22/Pointcept/"
    "sonata/lartpc_v6_h200_noghosts_pretrain_logspace_resume/model/epoch_42.pth"
)

# Stage-2 (CascadedSlicer) trained checkpoint — the user's most recent
# ptv3hybrid crosslevel slicer with mixed query + mask denoising.
cascaded_slicer_weight = (
    "/home/twongjirad/working/larbys/gen2/container_u22/Pointcept/"
    "exp/larformer_slicer_v1_cascaded_ptv3hybrid_crosslevel_nonzeroinit_maskdn_noamp/"
    "model/model_ptv3crosslevel_iter_37625.pth"
)

# =============================================================================
# Toggles — mirror the slicer config for consistency, plus Stage-3 specifics.
# =============================================================================
USE_SINUSOIDAL_POS_EMB                = False
ENABLE_ORIGIN_HEAD_WITH_CENTROID      = False    # Stage 2 (slicer) origin-head config
PURE_RANDOM_NEGATIVES                 = False
HARD_NEG_FRACTION_OF_IMPORTANCE       = 0.5

_IMPORTANCE_BUDGET   = 0.0 if PURE_RANDOM_NEGATIVES else 0.75
_IMPORTANCE_RATIO    = _IMPORTANCE_BUDGET * (1.0 - HARD_NEG_FRACTION_OF_IMPORTANCE)
_HARD_NEG_RATIO      = _IMPORTANCE_BUDGET * HARD_NEG_FRACTION_OF_IMPORTANCE

# Stage-3 specific knobs
STAGE3_NUM_QUERIES                    = 32
# 8-way per-query class head:
#   0 = e±,    1 = γ,        2 = μ±,         3 = π±,
#   4 = p,     5 = other_track,
#   6 = (unused — reserved for future split, e.g. shower vs MIP)
#   7 = no_object
STAGE3_NUM_CLASSES                    = 8
STAGE3_MASK_PROB_THRESHOLD            = 0.5
STAGE3_RECENTER_TO_SLICE_CENTROID     = True
STAGE3_TOKEN_DIM                      = 256
STAGE3_BACKBONE_OUT_CH                = 1232      # Sonata-v1m1 + PT-v3m2 head

# =============================================================================
# Shared geometry / backbone config (same as the slicer).
# =============================================================================
coord_center = (125.0, 0.0, 518.0)
coord_scale  = 179.55
flash_backend = "flash_attn"

# Slicer-side knobs (the trained checkpoint). Kept verbatim from the
# ptv3hybrid-crosslevel config so the slicer ckpt loads cleanly.
_SLICER_PTV3_DEC_CHANNELS = (64, 64, 128, 256)
slicer_backbone_out_channels = _SLICER_PTV3_DEC_CHANNELS[0]   # 64 = dec0 width
slicer_token_dim = 256

# =============================================================================
# Dataset — for both training and the S3.2 benchmark when --real-data is set.
# =============================================================================
_dataset_common = dict(
    type="LArFormerDataset",
    coord_center=coord_center,
    coord_scale=coord_scale,
    # NOTE: gt_source="slice" until S3.1.b lands the "particle" plug. This
    # is FINE for the S3.2 benchmark (we just need realistic SP counts).
    # S3.3 will swap to gt_source="particle" using compute_particle_labels.
    gt_source="slice",
    emit_fragments=False,
    slice_class_map={1: 0, 2: 1},
    slice_origin_kind=("centroid" if ENABLE_ORIGIN_HEAD_WITH_CENTROID
                       else "primary_start_pos"),
    merge_nu_slices=True,
    lm_score_aug_low=0.0,
    lm_score_aug_high=0.0,
    lm_score_val_threshold=0.0,
    wire_scale=1.0 / 3456.0,
    min_fragment_points_post_filter=50,
)

data = dict(
    num_classes=STAGE3_NUM_CLASSES,
    ignore_index=-1,
    names=["e", "gamma", "mu", "pi", "p", "other", "(unused)", "no_object"],
    train=dict(split="train", data_root="/", data_list_file=TRAIN_FILE_LIST,
               loop=1, max_spacepoints=100_000, **_dataset_common),
    val=dict(split="val", data_root="/", data_list_file=VAL_FILE_LIST,
             loop=1, max_spacepoints=150_000, **_dataset_common),
    test=dict(split="test", data_root="/", data_list_file=VAL_FILE_LIST,
              loop=1, max_spacepoints=None, **_dataset_common),
)

# =============================================================================
# Stage 1 — SonataLoRADeghostSegmentor (unchanged from the Stage-2 config)
# =============================================================================
deghoster_cfg = dict(
    type="SonataLoRADeghostSegmentor",
    backbone_out_channels=1232,
    lora_rank=16,
    lora_alpha=32.0,
    lora_dropout=0.05,
    lora_target_modules=["qkv", "proj"],
    freeze_backbone_non_lora=True,
    ghost_class_index=1,
    backbone=dict(
        type="Sonata-v1m1",
        backbone=dict(
            type="PT-v3m2",
            in_channels=6,
            order=("z", "z-trans", "hilbert", "hilbert-trans"),
            stride=(2, 2, 2, 2),
            enc_depths=(3, 3, 3, 9, 3),
            enc_channels=(48, 96, 192, 384, 512),
            enc_num_head=(3, 6, 12, 24, 32),
            enc_patch_size=(256, 256, 256, 256, 256),
            mlp_ratio=4, qkv_bias=True, qk_scale=None,
            attn_drop=0.0, proj_drop=0.0, drop_path=0.3,
            shuffle_orders=True, pre_norm=True,
            enable_rpe=False, enable_flash=False, flash_backend=flash_backend,
            upcast_attention=False, upcast_softmax=False,
            traceable=True, enc_mode=True, mask_token=True,
        ),
        head_in_channels=1088, head_hidden_channels=2048,
        head_embed_channels=256, head_num_prototypes=4096,
        num_global_view=2, num_local_view=6,
        up_cast_level=4,
    ),
    criteria=[
        dict(type="FocalLoss", gamma=2.0, alpha=0.5,
             loss_weight=1.0, ignore_index=-1, reduction="mean"),
        dict(type="LovaszLoss", mode="multiclass",
             loss_weight=1.0, ignore_index=-1),
    ],
)

# =============================================================================
# Stage 2 — ptv3hybrid crosslevel slicer (unchanged from its config)
# =============================================================================
_token_refiner_cfg = dict(
    type="CrossLevelAttn",
    num_layers=2,
    num_heads=4,
    mlp_ratio=4.0,
    target_levels=["voxel_16cm", "voxel_8cm", "ptv3_dec3", "ptv3_dec2"],
    max_source_tokens_per_level=8192,
)

slicer_levels = [
    dict(name="voxel_16cm",
         builder="VoxelBuilder",
         builder_cfg=dict(voxel_size_cm=16.0, coord_scale=coord_scale),
         supervision=dict(mask=dict(weight=1.0, mode="aux"))),
    dict(name="voxel_8cm",
         builder="VoxelBuilder",
         builder_cfg=dict(voxel_size_cm=8.0, coord_scale=coord_scale),
         supervision=dict(
             mask=dict(weight=1.0, mode="aux"),
             cls=dict(num_classes=3, label_src="origin_label",
                      label_remap={0: 0, 1: 2, 2: 1}, reduce="amax",
                      weight=0.5, loss="ce", ignore_index=-1),
         )),
    dict(name="ptv3_dec3",
         builder="PTv3DecoderStageLevel",
         builder_cfg=dict(stage_key="dec3", in_dim=_SLICER_PTV3_DEC_CHANNELS[3]),
         supervision=dict(mask=dict(weight=1.0, mode="aux"))),
    dict(name="ptv3_dec2",
         builder="PTv3DecoderStageLevel",
         builder_cfg=dict(stage_key="dec2", in_dim=_SLICER_PTV3_DEC_CHANNELS[2]),
         supervision=dict(mask=dict(weight=1.0, mode="aux"))),
    dict(name="spacepoint",
         builder="SpacepointBuilder",
         supervision=dict(mask=dict(weight=5.0, mode="primary"))),
]
slicer_scale_pattern = [
    "voxel_16cm", "voxel_8cm",
    "ptv3_dec3", "ptv3_dec2",
    "spacepoint", "spacepoint",
]

slicer_cfg = dict(
    type="LArFormer",
    backbone=dict(
        type="Sonata-v1m1",
        backbone=dict(
            type="PT-v3m2",
            in_channels=6,
            order=("z", "z-trans", "hilbert", "hilbert-trans"),
            stride=(2, 2, 2, 2),
            enc_depths=(3, 3, 3, 9, 3),
            enc_channels=(48, 96, 192, 384, 512),
            enc_num_head=(3, 6, 12, 24, 32),
            enc_patch_size=(256, 256, 256, 256, 256),
            mlp_ratio=4, qkv_bias=True, qk_scale=None,
            attn_drop=0.0, proj_drop=0.0, drop_path=0.0,
            shuffle_orders=False, pre_norm=True,
            enable_rpe=False, enable_flash=True, flash_backend=flash_backend,
            upcast_attention=False, upcast_softmax=False,
            traceable=True, enc_mode=False, mask_token=False,
        ),
        head_in_channels=1088, head_hidden_channels=2048,
        head_embed_channels=256, head_num_prototypes=4096,
        num_global_view=2, num_local_view=6,
        up_cast_level=0,
    ),
    backbone_out_channels=slicer_backbone_out_channels,
    levels=slicer_levels,
    scale_pattern=slicer_scale_pattern,
    token_dim=slicer_token_dim,
    num_queries=128,
    num_classes=3,
    freeze_backbone=True,
    unfreeze_decoder=True,
    capture_decoder_stages=True,
    ptv3_decoder_init_scale=0.01,
    enable_origin_head=ENABLE_ORIGIN_HEAD_WITH_CENTROID,
    token_refiner=_token_refiner_cfg,
    decoder_kwargs=dict(
        num_heads=4, mlp_ratio=4.0,
        zero_init_output_proj=False,
        **(dict(pos_emb_kind="sinusoidal") if USE_SINUSOIDAL_POS_EMB else {}),
    ),
    loss_kwargs=dict(
        weight_class=2.0,
        weight_mask_primary=5.0,
        weight_dice_primary=5.0,
        weight_aux_mask=0.7,
        weight_per_level_cls=0.5,
        weight_origin=(0.5 if ENABLE_ORIGIN_HEAD_WITH_CENTROID else 0.0),
        num_sample_points=16392,
        use_importance_sampling=(not PURE_RANDOM_NEGATIVES),
        importance_oversample_ratio=3.0,
        importance_ratio=_IMPORTANCE_RATIO,
        importance_hard_neg_ratio=_HARD_NEG_RATIO,
        aux_max_tokens=20_000,
        no_object_weight=0.5,
    ),
    mixed_query_selection=dict(
        source_level="voxel_8cm",
        score_source="cls_head",
        selection_mode="top_m_then_fps",
        score_filter_multiplier=4,
    ),
    mask_denoising=dict(
        dn_groups=3,
        max_dn_per_event=96,
        anchor_jitter_std=0.05,
    ),
)

cascaded_slicer_cfg = dict(
    type="CascadedSlicer",
    deghoster=deghoster_cfg,
    deghoster_weight=deghoster_weight,
    slicer=slicer_cfg,
    slicer_backbone_weight=sonata_pretrain_weight,
    deghost_threshold_min=0.4,
    deghost_threshold_max=0.6,
    deghost_threshold_val=0.5,
    freeze_deghoster=True,
    deghoster_class_index_real=0,
    report_keep_frac=True,
)

# =============================================================================
# Stage 3 — Particle segmenter (a fresh LArFormer)
# =============================================================================
# Lighter weight than the Stage 2 slicer: the input is just the nu-slice
# spacepoints (typically ~5–15K vs ~50–150K full event). Voxel pyramid
# at 16 / 8 cm is more than enough — the SP-level primary mask carries
# the bulk of the supervision.
particle_levels = [
    dict(name="voxel_16cm",
         builder="VoxelBuilder",
         builder_cfg=dict(voxel_size_cm=16.0, coord_scale=coord_scale),
         supervision=dict(mask=dict(weight=1.0, mode="aux"))),
    dict(name="voxel_8cm",
         builder="VoxelBuilder",
         builder_cfg=dict(voxel_size_cm=8.0, coord_scale=coord_scale),
         supervision=dict(
             mask=dict(weight=1.0, mode="aux"),
             cls=dict(num_classes=STAGE3_NUM_CLASSES,
                      label_src="origin_label",
                      label_remap={0: 0, 1: STAGE3_NUM_CLASSES - 1, 2: STAGE3_NUM_CLASSES - 1},
                      reduce="amax",
                      weight=0.3, loss="ce", ignore_index=-1),
         )),
    dict(name="spacepoint",
         builder="SpacepointBuilder",
         supervision=dict(mask=dict(weight=5.0, mode="primary"))),
]
particle_scale_pattern = [
    "voxel_16cm", "voxel_8cm",
    "spacepoint", "spacepoint",
]

# Per-level refiner — same architecture as Stage 2 but smaller depth.
_particle_token_refiner_cfg = dict(
    type="CrossLevelAttn",
    num_layers=2,
    num_heads=4,
    mlp_ratio=4.0,
    target_levels=["voxel_16cm", "voxel_8cm"],
    max_source_tokens_per_level=4096,
)

particle_segmenter_cfg = dict(
    type="LArFormer",
    backbone=dict(
        type="Sonata-v1m1",
        backbone=dict(
            type="PT-v3m2",
            in_channels=6,
            order=("z", "z-trans", "hilbert", "hilbert-trans"),
            stride=(2, 2, 2, 2),
            enc_depths=(3, 3, 3, 9, 3),
            enc_channels=(48, 96, 192, 384, 512),
            enc_num_head=(3, 6, 12, 24, 32),
            enc_patch_size=(256, 256, 256, 256, 256),
            mlp_ratio=4, qkv_bias=True, qk_scale=None,
            attn_drop=0.0, proj_drop=0.0, drop_path=0.0,
            shuffle_orders=False, pre_norm=True,
            enable_rpe=False, enable_flash=True, flash_backend=flash_backend,
            upcast_attention=False, upcast_softmax=False,
            traceable=True, enc_mode=True, mask_token=False,
        ),
        head_in_channels=1088, head_hidden_channels=2048,
        head_embed_channels=256, head_num_prototypes=4096,
        num_global_view=2, num_local_view=6,
        up_cast_level=4,
    ),
    backbone_out_channels=STAGE3_BACKBONE_OUT_CH,
    levels=particle_levels,
    scale_pattern=particle_scale_pattern,
    token_dim=STAGE3_TOKEN_DIM,
    num_queries=STAGE3_NUM_QUERIES,
    num_classes=STAGE3_NUM_CLASSES,
    freeze_backbone=True,
    enable_origin_head=True,   # §6 — per-particle starting-point regression
    token_refiner=_particle_token_refiner_cfg,
    decoder_kwargs=dict(
        num_heads=4, mlp_ratio=4.0,
        zero_init_output_proj=False,
        **(dict(pos_emb_kind="sinusoidal") if USE_SINUSOIDAL_POS_EMB else {}),
    ),
    loss_kwargs=dict(
        weight_class=2.0,
        weight_mask_primary=5.0,
        weight_dice_primary=5.0,
        weight_aux_mask=0.5,
        weight_per_level_cls=0.3,
        weight_origin=0.5,
        num_sample_points=8192,
        use_importance_sampling=(not PURE_RANDOM_NEGATIVES),
        importance_oversample_ratio=3.0,
        importance_ratio=_IMPORTANCE_RATIO,
        importance_hard_neg_ratio=_HARD_NEG_RATIO,
        aux_max_tokens=10_000,
        no_object_weight=0.1,   # Stage 3 has fewer GT instances per event
        weight_dn_loss=1.0,
    ),
    mixed_query_selection=dict(
        source_level="voxel_8cm",
        score_source="cls_head",
        selection_mode="top_m_then_fps",
        score_filter_multiplier=4,
    ),
    mask_denoising=dict(
        dn_groups=3,
        max_dn_per_event=64,    # particle GT counts are smaller than slice GT
        anchor_jitter_std=0.05,
    ),
)

# =============================================================================
# Top-level CascadedParticleSegmenter wrapping Stage 1+2 + Stage 3.
# =============================================================================
model = dict(
    type="CascadedParticleSegmenter",
    cascaded_slicer=cascaded_slicer_cfg,
    particle_segmenter=particle_segmenter_cfg,
    cascaded_slicer_weight=cascaded_slicer_weight,
    particle_segmenter_backbone_weight=sonata_pretrain_weight,
    freeze_cascaded_slicer=True,
    nu_class_id=0,
    mask_prob_threshold=STAGE3_MASK_PROB_THRESHOLD,
    spacepoint_level="spacepoint",
    recenter_to_slice_centroid=STAGE3_RECENTER_TO_SLICE_CENTROID,
    report_keep_frac=True,
)

# =============================================================================
# Training loop knobs (placeholders — S3.3 will retune)
# =============================================================================
weight = None
save_path        = "exp/larformer_particle_v1"
epoch            = 50
eval_epoch       = 50
batch_size       = 4
batch_size_val   = 4
num_worker       = 8
num_worker_val   = 8
evaluate         = True
enable_amp       = False
amp_dtype        = "bfloat16"
empty_cache      = False
clip_grad        = 1.0
enable_wandb     = False
find_unused_parameters = True
