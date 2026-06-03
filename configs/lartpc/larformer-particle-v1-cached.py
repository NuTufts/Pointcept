"""LArFormer Stage-3 training — directly from a Stage-1+2 cache.

Companion to `larformer-particle-v1.py` (which defined the production
CascadedParticleSegmenter, used for inference and for cache building).
This config trains JUST the Stage-3 particle segmenter LArFormer using
per-event HDF5 caches produced by
`tools/build_stage12_cache_shard.py`.

Key delta from `larformer-particle-v1.py`:

  - `model` is a standalone `LArFormer`, NOT `CascadedParticleSegmenter`.
    Stage 1+2 was run once at cache time; the cache files supply the
    filtered SPs + Stage-2 telemetry + particle-level GT directly. So
    training spends 100 % of wall-clock on Stage-3 forward/backward
    (≈17× speedup measured on RTX 3080, see the S3.2 benchmark).
  - `data` uses `LArFormerStage12CacheDataset` with `source_set_filter`.
    Default `"stage2_pass"` ⇒ inference-realistic input set. Swap to
    `"union"` / `"stage2_plus_gt_dropout"` for mask-denoising ablations
    or curriculum-training experiments.
  - Centroid recentering is applied at load time (`recenter_to_centroid=
    True`) — match-time recentering keeps the model invariant to where
    the nu slice lives in the detector.

For inference deployment, the trained weights load back into the
`particle_segmenter` slot of `CascadedParticleSegmenter` from
`larformer-particle-v1.py` — same shape, same backbone, same
tokenizer/decoder/heads.
"""

_base_ = ["../_base_/default_runtime.py"]

# Side-effect: register LArFormerTrainer + LArFormerParticleEvaluator.
from pointcept.models.LArFormer import trainer as _larformer_trainer_module
from pointcept.models.LArFormer import particle_evaluator as _larformer_particle_evaluator_module
del _larformer_trainer_module
del _larformer_particle_evaluator_module

# =============================================================================
# Paths
# =============================================================================
# Stage-1+2 cache locations. Update these to your cluster paths or
# override on the command line. The shard driver writes per-event H5s
# under `<cache_root>/<split>/<idx//1000>/<idx//100>/...`.
CACHE_ROOT  = "/cluster/tufts/wongjiradlabnu/twongj01/stage12_cache_v2"
TRAIN_ROOT  = f"{CACHE_ROOT}/train"
VAL_ROOT    = f"{CACHE_ROOT}/val"

# Sonata pretrain for the Stage-3 backbone. Loaded via the LArFormer's
# own `backbone_weight` knob.
sonata_pretrain_weight = (
    "/home/twongjirad/working/larbys/gen2/container_u22/Pointcept/"
    "sonata/lartpc_v6_h200_noghosts_pretrain_logspace_resume/model/epoch_42.pth"
)

# =============================================================================
# Toggles — mirror larformer-particle-v1.py for parity
# =============================================================================
USE_SINUSOIDAL_POS_EMB      = False
PURE_RANDOM_NEGATIVES       = False
HARD_NEG_FRACTION_OF_IMPORT = 0.5

_IMPORTANCE_BUDGET = 0.0 if PURE_RANDOM_NEGATIVES else 0.75
_IMPORTANCE_RATIO  = _IMPORTANCE_BUDGET * (1.0 - HARD_NEG_FRACTION_OF_IMPORT)
_HARD_NEG_RATIO    = _IMPORTANCE_BUDGET * HARD_NEG_FRACTION_OF_IMPORT

STAGE3_NUM_QUERIES      = 32
# 8-way per-query class head — must match larformer-particle-v1.py:
#   0=e±, 1=γ, 2=μ±, 3=π±, 4=p, 5=other_track,
#   6=(unused), 7=no_object
STAGE3_NUM_CLASSES      = 8
STAGE3_TOKEN_DIM        = 256
STAGE3_BACKBONE_OUT_CH  = 1232    # Sonata-v1m1 + PT-v3m2 head

# =============================================================================
# Geometry
# =============================================================================
coord_center = (125.0, 0.0, 518.0)
coord_scale  = 179.55
flash_backend = "flash_attn"

# =============================================================================
# Dataset (cache reader)
# =============================================================================
data = dict(
    num_classes=STAGE3_NUM_CLASSES,
    ignore_index=-1,
    names=["e", "gamma", "mu", "pi", "p", "other", "(unused)", "no_object"],
    train=dict(
        type="LArFormerStage12CacheDataset",
        split="train",
        data_root=TRAIN_ROOT,
        source_set_filter="stage2_pass",   # default: inference-realistic
        # The other modes are documented in
        # pointcept/datasets/larformer_stage12_cache.py — switch via
        # config override when running ablations.
        recenter_to_centroid=True,
        coord_center=coord_center,
        coord_scale=coord_scale,
        loop=1,
    ),
    # ---- Optional: dual-loader mask denoising path -------------------
    # When `train_dn` is set, LArFormerTrainer runs TWO forwards per
    # iter: the matcher forward on `train` (stage2_pass = inference-
    # realistic) and the denoising forward on `train_dn` (union = full
    # GT anchor for mask perturbation). The matcher loss comes from the
    # first forward; the `loss_dn_*` components from the second. Doubles
    # per-iter wall-clock; the 17× cache speedup from S3.2 absorbs it.
    #
    # Leave the line below commented out to train with a single forward
    # (the same `source_set_filter` for both the matcher and mask
    # denoising). Recommended for the first sanity-train; enable for
    # production quality runs.
    #
    # train_dn=dict(
    #     type="LArFormerStage12CacheDataset",
    #     split="train",
    #     data_root=TRAIN_ROOT,
    #     source_set_filter="union",
    #     recenter_to_centroid=True,
    #     coord_center=coord_center,
    #     coord_scale=coord_scale,
    #     loop=1,
    # ),
    val=dict(
        type="LArFormerStage12CacheDataset",
        split="val",
        data_root=VAL_ROOT,
        source_set_filter="stage2_pass",
        recenter_to_centroid=True,
        coord_center=coord_center,
        coord_scale=coord_scale,
        loop=1,
    ),
    test=dict(
        type="LArFormerStage12CacheDataset",
        split="test",
        data_root=VAL_ROOT,             # placeholder; replace per dataset
        source_set_filter="stage2_pass",
        recenter_to_centroid=True,
        coord_center=coord_center,
        coord_scale=coord_scale,
        loop=1,
    ),
)

# =============================================================================
# Particle segmenter — identical to the Stage-3 sub-config in
# larformer-particle-v1.py, except `backbone_weight` is set here so the
# standalone model loads the Sonata pretrain at __init__ time.
# =============================================================================
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
                      label_remap={0: 0, 1: STAGE3_NUM_CLASSES - 1,
                                   2: STAGE3_NUM_CLASSES - 1},
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

_particle_token_refiner_cfg = dict(
    type="CrossLevelAttn",
    num_layers=2,
    num_heads=4,
    mlp_ratio=4.0,
    target_levels=["voxel_16cm", "voxel_8cm"],
    max_source_tokens_per_level=4096,
)

model = dict(
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
    backbone_weight=sonata_pretrain_weight,
    levels=particle_levels,
    scale_pattern=particle_scale_pattern,
    token_dim=STAGE3_TOKEN_DIM,
    num_queries=STAGE3_NUM_QUERIES,
    num_classes=STAGE3_NUM_CLASSES,
    freeze_backbone=True,
    enable_origin_head=True,
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
        no_object_weight=0.1,
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
        max_dn_per_event=64,
        anchor_jitter_std=0.05,
    ),
)

# =============================================================================
# Trainer + evaluator
# =============================================================================
train = dict(type="LArFormerTrainer")

# Validation hook — strips nu-specific metrics; best_metric =
# mask_iou_mean (matched-pair IoU averaged across all matched classes).
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="LArFormerParticleEvaluator",
         best_metric="mask_iou_mean",
         class_names=["e", "gamma", "mu", "pi", "p", "other",
                      "(unused)", "no_object"],
         coord_scale=coord_scale),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]

# =============================================================================
# Training loop knobs
# =============================================================================
weight = None
save_path        = "exp/larformer_particle_v1_cached"
epoch            = 50
eval_epoch       = 50
batch_size       = 8          # Stage-3 input is small (~1K SPs/event)
batch_size_val   = 8
num_worker       = 8
num_worker_val   = 8
evaluate         = True
enable_amp       = False
amp_dtype        = "bfloat16"
empty_cache      = False
clip_grad        = 1.0
enable_wandb     = False
wandb_project    = "pointcept-larformer"
find_unused_parameters = True

# Mid-epoch resume strategy — same convention as the slicer config.
skip_dataloader_on_resume = True
resume_seed_strategy = "per_resume"

# =============================================================================
# Optimizer / scheduler
# =============================================================================
# Conservative defaults; tune in follow-up runs. AdamW with cosine to a
# fixed minimum is the same default used by the slicer training.
optimizer = dict(
    type="AdamW", lr=2e-4, weight_decay=0.05,
    betas=(0.9, 0.95),
)
scheduler = dict(
    type="OneCycleLR",
    max_lr=2e-4,
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)
param_dicts = [
    # Backbone is frozen, so excluded from optimizer entirely. The
    # decoder + tokenizer + heads + (optional) refiner + denoiser all
    # train at the default lr.
    dict(keyword="backbone", lr=0.0),
]
