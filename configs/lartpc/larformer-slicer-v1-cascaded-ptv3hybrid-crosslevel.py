"""
LArFormer Stage-2 cascaded slicer — PTv3-decoder + custom-voxel HYBRID variant.

Combines:
  - Two USER-DEFINED COARSE VOXEL LEVELS (voxel_16cm, voxel_8cm) pooled
    from PTv3's per-SP decoder output (dec0 / 64-channel features). The
    voxel pyramid extends the spatial coverage beyond PTv3's natural
    pyramid (which only reaches 2 cm at dec3 given the 0.25 cm input
    grid).
  - Two PTv3 NATIVE DECODER STAGES (dec3 @ 2 cm, dec2 @ 1 cm) consumed
    via `PTv3DecoderStageLevel`. These give the queries transformer-
    refined features at the fine end of the pyramid (features whose
    representations PTv3's decoder Blocks actually processed at that
    scale, vs. averaged dec0 features the voxel levels carry).
  - The SPACEPOINT LEVEL (= dec0 here, since enc_mode=False +
    up_cast_level=0 makes dec0 the final per-SP output).

Refinement: PerLevelSelfAttn on all four non-SP levels (voxel_16cm,
voxel_8cm, ptv3_dec3, ptv3_dec2). The PTv3 decoder already does within-
stage transformer mixing on dec2/dec3; this adds another pass on top
plus brings the voxel levels (which the PTv3 decoder doesn't touch)
into the same refinement regime.

Aux per-token cls supervision is attached to voxel_8cm — at 8 cm voxels
the token count is dense enough that the plurality vote over per-SP
origin labels gives a clean signal.

6-layer scale_pattern (coarse → fine):
    voxel_16cm → voxel_8cm → ptv3_dec3 → ptv3_dec2 → spacepoint → spacepoint

This is the same depth as the other cascaded configs so head-to-head
comparisons share their decoder budget.

Pretrained encoder weights load cleanly (frozen). The PTv3 decoder
trains from scratch (Sonata pretrain was enc_mode=True so decoder
weights aren't in the checkpoint). The refiner and Mask2Former decoder
+ heads also train from scratch as in the other variants.
"""

_base_ = ["../_base_/default_runtime.py"]

# Side-effect: register LArFormerTrainer + LArFormerSlicerEvaluator.
from pointcept.models.LArFormer import trainer as _larformer_trainer_module
from pointcept.models.LArFormer import evaluator as _larformer_evaluator_module
del _larformer_trainer_module
del _larformer_evaluator_module

# =============================================================================
# Paths
# =============================================================================
_DEFAULT_LIST = "/home/twongjirad/working/larbys/gen2/container_u22/Pointcept/devdata_mergedh5_pi0filter_10files.txt"

# Trained SonataLoRADeghostSegmentor checkpoint for Stage 1.
deghoster_weight = "/home/twongjirad/working/larbys/gen2/container_u22/Pointcept/sonata/lora_deghost_v6_hasmatch/model/epoch_30.pth"

# =============================================================================
# Toggles (kept here so this hybrid config is independently reproducible
# and so the toggle semantics match the sibling configs).
# =============================================================================
USE_SINUSOIDAL_POS_EMB = False
ENABLE_ORIGIN_HEAD_WITH_CENTROID = False

# Per-pair negative sampler — default to halo-only (current behavior). Flip
# HARD_NEG_FRACTION_OF_IMPORTANCE to 0.5 to half-and-half with confident-FP
# mining; see docs/LArFormer.md §15.
PURE_RANDOM_NEGATIVES = False
HARD_NEG_FRACTION_OF_IMPORTANCE = 0.5

_IMPORTANCE_BUDGET = 0.0 if PURE_RANDOM_NEGATIVES else 0.75
_IMPORTANCE_RATIO = _IMPORTANCE_BUDGET * (1.0 - HARD_NEG_FRACTION_OF_IMPORTANCE)
_HARD_NEG_RATIO = _IMPORTANCE_BUDGET * HARD_NEG_FRACTION_OF_IMPORTANCE

# =============================================================================
# Backbone shape — locked to PTv3-decoder mode
# =============================================================================
# enc_mode=False enables PT-v3m2's learned decoder (self.dec). The
# Sonata-v1m1 wrapper's up_cast must be disabled (up_cast_level=0) so it
# doesn't try to upcast the already-decoded output (the decoder consumed
# the pool chain that up_cast needs).
#
# With dec_channels = (64, 64, 128, 256):
#   - dec0 @ stride 1   → 64 ch  → per-SP feature width = backbone_out_channels
#   - dec1 @ stride 2   → 64 ch  (~0.5 cm grid; barely collapses, unused here)
#   - dec2 @ stride 4   → 128 ch (~1 cm grid)
#   - dec3 @ stride 8   → 256 ch (~2 cm grid)
_PTV3_DEC_CHANNELS = (64, 64, 128, 256)
backbone_out_channels = _PTV3_DEC_CHANNELS[0]   # 64 = dec0 width

coord_center = (125.0, 0.0, 518.0)
coord_scale = 179.55
#flash_backend = "xformers"
flash_backend = "flash_attn"
token_dim = 256

# =============================================================================
# Dataset
# =============================================================================
_dataset_common = dict(
    type="LArFormerDataset",
    coord_center=coord_center,
    coord_scale=coord_scale,
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
    num_classes=3,
    ignore_index=-1,
    names=["nu", "cosmic", "no_object"],
    train=dict(split="train", data_root="/", data_list_file=_DEFAULT_LIST,
               loop=1, max_spacepoints=100_000, **_dataset_common),
    val=dict(split="val", data_root="/", data_list_file=_DEFAULT_LIST,
             loop=1, max_spacepoints=150_000, **_dataset_common),
    test=dict(split="test", data_root="/", data_list_file=_DEFAULT_LIST,
              loop=1, max_spacepoints=None, **_dataset_common),
)

# =============================================================================
# Stage 1 — SonataLoRADeghostSegmentor (unchanged from the loradeghost config)
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
# Stage 2 — Slicer subconfig (HYBRID levels)
# =============================================================================
# Five levels, six decoder layers. The voxel pyramid (16 cm, 8 cm) extends
# spatial coverage beyond what the PTv3 decoder can reach at our 0.25 cm
# input grid; the PTv3 decoder stages (dec3 @ 2 cm, dec2 @ 1 cm) give the
# queries decoder-refined features at the fine end of the pyramid.
levels = [
    dict(name="voxel_16cm",
         builder="VoxelBuilder",
         builder_cfg=dict(voxel_size_cm=16.0, coord_scale=coord_scale),
         supervision=dict(mask=dict(weight=1.0, mode="aux"))),
    dict(name="voxel_8cm",
         builder="VoxelBuilder",
         builder_cfg=dict(voxel_size_cm=8.0, coord_scale=coord_scale),
         supervision=dict(
             mask=dict(weight=1.0, mode="aux"),
             # cls aux supervision — at 8 cm voxels there are typically
             # hundreds-to-low-thousands of tokens, dense enough for the
             # per-SP origin label plurality vote to give a clean signal.
             # `label_remap={0:0, 1:2, 2:1}` puts priority-pool order:
             # nu (label 1) gets the largest remapped id so reduce="amax"
             # makes any-nu voxel render as nu.
             cls=dict(num_classes=3, label_src="origin_label",
                      label_remap={0: 0, 1: 2, 2: 1}, reduce="amax",
                      weight=0.5, loss="ce", ignore_index=-1),
         )),
    dict(name="ptv3_dec3",
         builder="PTv3DecoderStageLevel",
         builder_cfg=dict(stage_key="dec3", in_dim=_PTV3_DEC_CHANNELS[3]),
         supervision=dict(mask=dict(weight=1.0, mode="aux"))),
    dict(name="ptv3_dec2",
         builder="PTv3DecoderStageLevel",
         builder_cfg=dict(stage_key="dec2", in_dim=_PTV3_DEC_CHANNELS[2]),
         supervision=dict(mask=dict(weight=1.0, mode="aux"))),
    dict(name="spacepoint",
         builder="SpacepointBuilder",
         supervision=dict(mask=dict(weight=5.0, mode="primary"))),
]
# 6 layers, coarse→fine. Same depth as the other cascaded configs.
scale_pattern = [
    "voxel_16cm", "voxel_8cm",
    "ptv3_dec3",  "ptv3_dec2",
    "spacepoint", "spacepoint",
]

# =============================================================================
# Token refiner — CrossLevelAttn on all non-SP level
# =============================================================================

_token_refiner_cfg = dict(
    type="CrossLevelAttn",
    num_layers=2,
    num_heads=4,
    mlp_ratio=4.0,
    target_levels=["voxel_16cm", "voxel_8cm","ptv3_dec3","ptv3_dec2"],
    # source_levels=None,                  # None = all levels (incl. SP)
    max_source_tokens_per_level=16392,      # cap SP's K/V contribution
    # pos_emb_kind="sinusoidal",
)

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
    backbone_out_channels=backbone_out_channels,
    levels=levels,
    scale_pattern=scale_pattern,
    token_dim=token_dim,
    num_queries=128,
    num_classes=3,
    freeze_backbone=True,
    unfreeze_decoder=True,             # PTv3 decoder trains from scratch
    capture_decoder_stages=True,       # hooks on dec3/dec2/dec1 (dec1 unused)
    enable_origin_head=ENABLE_ORIGIN_HEAD_WITH_CENTROID,
    token_refiner=_token_refiner_cfg,
    decoder_kwargs=dict(
        num_heads=4, mlp_ratio=4.0,
        **(dict(pos_emb_kind="sinusoidal") if USE_SINUSOIDAL_POS_EMB else {}),
    ),
    loss_kwargs=dict(
        weight_class=2.0,
        weight_mask_primary=5.0,
        weight_dice_primary=5.0,
        weight_aux_mask=0.7,
        weight_per_level_cls=0.5,
        weight_origin=(0.5 if ENABLE_ORIGIN_HEAD_WITH_CENTROID else 0.0),
        num_sample_points=16384,
        use_importance_sampling=(not PURE_RANDOM_NEGATIVES),
        importance_oversample_ratio=3.0,
        importance_ratio=_IMPORTANCE_RATIO,
        importance_hard_neg_ratio=_HARD_NEG_RATIO,
        aux_max_tokens=20_000,
        no_object_weight=0.5,
    ),
)

# =============================================================================
# CascadedSlicer
# =============================================================================
slicer_backbone_weight = (
    "/home/twongjirad/working/larbys/gen2/container_u22/Pointcept/"
    "sonata/lartpc_v6_h200_noghosts_pretrain_logspace_resume/model/epoch_42.pth"
)

model = dict(
    type="CascadedSlicer",
    deghoster=deghoster_cfg,
    deghoster_weight=deghoster_weight,
    slicer=slicer_cfg,
    slicer_backbone_weight=slicer_backbone_weight,
    deghost_threshold_min=0.4,
    deghost_threshold_max=0.6,
    deghost_threshold_val=0.5,
    freeze_deghoster=True,
    deghoster_class_index_real=0,
    report_keep_frac=True,
)

# =============================================================================
# Training loop knobs
# =============================================================================
weight = None

save_path        = "exp/larformer_slicer_v1_cascaded_ptv3hybrid_crosslevel"
epoch            = 1000
eval_epoch       = 200
batch_size       = 1
batch_size_val   = 1
num_worker       = 2
num_worker_val   = 2
evaluate         = True
enable_amp       = False
empty_cache      = True
# Gradient clipping — important for this config because the PTv3 decoder
# + refiner + Mask2Former decoder all initialize from random and can
# produce extreme activations / gradients in the first ~hundred iters.
# Without this, an early exploding gradient can wreck the weights such
# that the model never recovers; the matcher's NaN-sanitizer keeps the
# loss STEP from crashing but doesn't protect downstream optimization.
clip_grad        = 1.0
enable_wandb     = True
wandb_project    = "pointcept-larformer"
find_unused_parameters = True

# =============================================================================
# Optimizer / scheduler
# =============================================================================
base_lr = 1.0e-4
param_dicts = None
optimizer = dict(type="AdamW", lr=base_lr, weight_decay=0.01)
scheduler = dict(
    type="FlatWithDecayLR",
    mode="plateau",
    gamma=0.5,
    min_lr=1e-7,
    step_period_epochs=200,
    patience_epochs=12,
    min_delta=1e-4,
    cooldown_epochs=2,
    # Linear warmup over the first 500 training iters (~100 epochs on the
    # 10-event dev sample at batch_size=2). PTv3 decoder + refiner +
    # Mask2Former decoder all initialize randomly here — gradients in the
    # first ~50 iters are noisy enough to spike the loss curve, so we
    # don't want the plateau detector touching anything during that
    # phase. step_epoch is a no-op while in warmup (counters frozen).
    warmup_iters=500,
    warmup_start_lr=0.0,
    # EMA over the val/loss for plateau detection. A single lucky-low
    # raw val_loss (which fluctuates a few % epoch-to-epoch on this
    # small dev sample) was pinning best_val_loss too tight, collapsing
    # the LR prematurely. alpha=0.3 means each new val_loss contributes
    # 30% to the smoothed signal; one outlier moves the EMA by at most
    # 30% of the gap, not the full distance. Set None to disable
    # smoothing (raw val_loss tracked, current pre-EMA behavior).
    ema_alpha=0.3,
    # No reset_lr by default — set on resume only.
    reset_lr=None,
    reset_counters=False,
)

hooks = [
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="LArFormerSlicerEvaluator",
         eval_freq=0,
         best_metric="nu_mIoU",
         class_names=["nu", "cosmic"],
         nu_class_id=0,
         empty_cache=True,
         log_per_event=False),
    dict(type="LREpochScheduler"),
    dict(type="CheckpointSaver", save_freq=None),
]

train = dict(type="LArFormerTrainer")
