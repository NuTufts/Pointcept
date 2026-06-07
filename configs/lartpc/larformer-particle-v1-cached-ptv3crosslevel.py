"""LArFormer Stage-3 training — HYBRID PTv3-decoder + crosslevel refiner.

Companion / sibling to `larformer-particle-v1-cached.py`. Same Stage-3
particle-segmenter targets (per-particle masks + per-query 7-class +
origin head), same `LArFormerStage12CacheDataset` cache reader, same
trainer + evaluator hooks. The DELTA from the v1 cached config is the
backbone + level pyramid:

  - PT-v3m2 NATIVE DECODER turned on (`enc_mode=False` +
    `up_cast_level=0`). The Sonata pretrain only contains encoder
    weights, so the decoder trains from scratch (small-mag init via
    `ptv3_decoder_init_scale=0.01`, see
    `LArFormer._init_ptv3_decoder_blocks`).
  - Two USER-DEFINED COARSE VOXEL LEVELS (voxel_8cm, voxel_4cm) pooled
    off the PTv3 decoder's per-SP dec0 output (64 ch). Voxel coverage
    extends past PTv3's natural pyramid (which only reaches ~2 cm at
    dec3 given the 0.25 cm input grid).
  - Two PTv3 NATIVE DECODER STAGES (dec3 @ ~2 cm, dec2 @ ~1 cm)
    consumed via `PTv3DecoderStageLevel`. These give queries
    transformer-refined features at the fine end of the pyramid
    (representations the PTv3 decoder Blocks actually processed at
    that scale, vs. averaged dec0 features the voxel levels carry).
  - The SPACEPOINT level reads dec0 (= the final per-SP output now
    that `up_cast_level=0`).
  - The CrossLevelAttn refiner now also runs over the two PTv3
    decoder stages, not just the voxel levels.

6-layer scale_pattern (coarse → fine):
    voxel_8cm → voxel_4cm → ptv3_dec3 → ptv3_dec2 → spacepoint → spacepoint

This mirrors `larformer-slicer-v1-cascaded-ptv3hybrid-crosslevel.py`'s
structure on the Stage-3 side, so head-to-head ablations of Stage 2
vs Stage 3 use the same backbone / refiner geometry.

For inference deployment, the trained weights load back into the
`particle_segmenter` slot of a `CascadedParticleSegmenter` configured
with the same `levels` / `scale_pattern` / `enc_mode=False`.

Notes vs the simpler cached config:

  - `param_dicts` is set to `None` (NOT `keyword="backbone"`) because
    `unfreeze_decoder=True` makes the PT-v3 decoder trainable. The
    cached config's `keyword="backbone"` pattern would zero its lr.
    `freeze_backbone=True` plus `unfreeze_decoder=True` sets
    requires_grad correctly per-param, and the optimizer respects it.
  - Trainable param count goes up vs the v1-cached config because the
    PT-v3 decoder is now trained. Memory + per-iter compute increase
    accordingly. Drop `batch_size` if it OOMs.
  - `clip_grad=1.0` is critical here (same as the slicer hybrid
    config): the PTv3 decoder, the cross-level refiner, and the M2F
    decoder all start random, so gradients can spike in the first
    ~hundred iters.
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
#CACHE_ROOT  = "/home/twongjirad/working/larbys/gen2/container_u22/Pointcept/exp/cache_stage12_devdata/"
#CACHE_ROOT  = "/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/exp/cache_stage12_devdata"
CACHE_ROOT  = "/cluster/tufts/wongjiradlab/larbys/data/ub_on_tufts/hdf5/larformer_cache_stage12__ptv3crosslevelslicer_iter_75750/"
TRAIN_ROOT  = f"{CACHE_ROOT}/train"
VAL_ROOT    = f"{CACHE_ROOT}/val"
#VAL_ROOT    = f"{CACHE_ROOT}/val_train_copy"

# Sonata pretrain for the Stage-3 backbone. Loaded via the LArFormer's
# own `backbone_weight` knob. NOTE: the Sonata pretrain was saved with
# enc_mode=True, so it only contains encoder weights — the PT-v3
# decoder trains from scratch in this config.
sonata_pretrain_weight = (
    #"/home/twongjirad/working/larbys/gen2/container_u22/Pointcept/"
    "/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/"
    "sonata/lartpc_v6_h200_noghosts_pretrain_logspace_resume/model/epoch_42.pth"
)

# =============================================================================
# Toggles — mirror larformer-particle-v1-cached.py for parity
# =============================================================================
USE_SINUSOIDAL_POS_EMB      = False
PURE_RANDOM_NEGATIVES       = False
HARD_NEG_FRACTION_OF_IMPORT = 0.5

_IMPORTANCE_BUDGET = 0.0 if PURE_RANDOM_NEGATIVES else 0.75
_IMPORTANCE_RATIO  = _IMPORTANCE_BUDGET * (1.0 - HARD_NEG_FRACTION_OF_IMPORT)
_HARD_NEG_RATIO    = _IMPORTANCE_BUDGET * HARD_NEG_FRACTION_OF_IMPORT

STAGE3_NUM_QUERIES      = 32
# 8-way per-query class head:
#   0=e±, 1=γ, 2=μ±, 3=π±, 4=p, 5=other_track,
#   6=(unused), 7=no_object
STAGE3_NUM_CLASSES      = 8
STAGE3_TOKEN_DIM        = 256

# =============================================================================
# Backbone shape — locked to PTv3-decoder mode (delta vs v1-cached)
# =============================================================================
# enc_mode=False enables PT-v3m2's learned decoder (self.dec). The
# Sonata-v1m1 wrapper's up_cast must be disabled (up_cast_level=0) so it
# doesn't try to upcast the already-decoded output (the decoder consumed
# the pool chain that up_cast needs).
#
# With dec_channels = (64, 64, 128, 256):
#   - dec0 @ stride 1   → 64 ch  → per-SP feature width = backbone_out_channels
#   - dec1 @ stride 2   → 64 ch  (~0.5 cm grid; not used as a level)
#   - dec2 @ stride 4   → 128 ch (~1 cm grid)  → ptv3_dec2 level
#   - dec3 @ stride 8   → 256 ch (~2 cm grid)  → ptv3_dec3 level
_PTV3_DEC_CHANNELS    = (64, 64, 128, 256)
STAGE3_BACKBONE_OUT_CH = _PTV3_DEC_CHANNELS[0]   # 64 = dec0 width

# =============================================================================
# Geometry
# =============================================================================
coord_center = (125.0, 0.0, 518.0)
coord_scale  = 179.55
flash_backend = "flash_attn"
#flash_backend = "xformers"

# =============================================================================
# Dataset (cache reader) — identical to the v1-cached config.
# =============================================================================
data = dict(
    num_classes=STAGE3_NUM_CLASSES,
    ignore_index=-1,
    names=["e", "gamma", "mu", "pi", "p", "other", "(unused)", "no_object"],
    train=dict(
        type="LArFormerStage12CacheDataset",
        split="train",
        data_root=TRAIN_ROOT,
        source_set_filter="stage2_pass",
        recenter_to_centroid=True,
        coord_center=coord_center,
        coord_scale=coord_scale,
        min_spacepoints=20,
        loop=1,
    ),
    # ---- Optional: dual-loader mask denoising path -------------------
    # See larformer-particle-v1-cached.py for the explanation.
    #
    # train_dn=dict(
    #     type="LArFormerStage12CacheDataset",
    #     split="train",
    #     data_root=TRAIN_ROOT,
    #     source_set_filter="union",
    #     recenter_to_centroid=True,
    #     coord_center=coord_center,
    #     coord_scale=coord_scale,
    #     min_spacepoints=20,
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
        min_spacepoints=20,
        loop=1,
    ),
)

# =============================================================================
# Stage-3 particle segmenter — HYBRID levels (the delta vs v1-cached).
# =============================================================================
# Five levels, six decoder layers. The voxel pyramid (16 cm, 8 cm) covers
# the coarse end; the PTv3 decoder stages (dec3 @ ~2 cm, dec2 @ ~1 cm)
# cover the fine end with decoder-refined features; the spacepoint level
# carries the per-SP primary mask supervision.
particle_levels = [
    dict(name="voxel_8cm",
         builder="VoxelBuilder",
         builder_cfg=dict(voxel_size_cm=8.0, coord_scale=coord_scale),
         supervision=dict(mask=dict(weight=1.0, mode="aux"))),
    dict(name="voxel_4cm",
         builder="VoxelBuilder",
         builder_cfg=dict(voxel_size_cm=4.0, coord_scale=coord_scale),
         supervision=dict(
             mask=dict(weight=1.0, mode="aux"),
             # cls aux supervision at 4 cm voxels — dense enough for the
             # per-SP origin label plurality vote to give a clean signal.
             # Stage-3 only sees nu-slice SPs after the cache filter, so
             # cosmic / no_object SPs are rare; the label_remap pushes
             # any non-nu origin into the no_object slot (= 7).
             cls=dict(num_classes=STAGE3_NUM_CLASSES,
                      label_src="origin_label",
                      label_remap={0: 0, 1: STAGE3_NUM_CLASSES - 1,
                                   2: STAGE3_NUM_CLASSES - 1},
                      reduce="amax",
                      weight=0.3, loss="ce", ignore_index=-1),
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
# 6 layers, coarse → fine. Same depth as the slicer hybrid config.
particle_scale_pattern = [
    "voxel_8cm", "voxel_4cm",
    "ptv3_dec3",  "ptv3_dec2",
    "spacepoint", "spacepoint",
]

# Cross-level refiner now also covers the two PTv3 decoder stages.
# `max_source_tokens_per_level` is bumped from 4096 → 8192 because the
# PTv3 stages and the SP level can both carry more tokens than the 16
# / 8 cm voxel levels did when the refiner was voxel-only.
_particle_token_refiner_cfg = dict(
    type="CrossLevelAttn",
    num_layers=2,
    num_heads=4,
    mlp_ratio=4.0,
    target_levels=["voxel_8cm", "voxel_4cm", "ptv3_dec3", "ptv3_dec2"],
    max_source_tokens_per_level=8192,
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
            traceable=True,
            # delta: PT-v3 decoder ON, Sonata-side up_cast OFF.
            enc_mode=False,
            mask_token=False,
        ),
        head_in_channels=1088, head_hidden_channels=2048,
        head_embed_channels=256, head_num_prototypes=4096,
        num_global_view=2, num_local_view=6,
        # delta: was 4 (Sonata-natural upcast); 0 = decoder owns the
        # final per-SP output, no further upcast.
        up_cast_level=0,
    ),
    # delta: was 1232 (Sonata head); 64 = PTv3 dec0 width.
    backbone_out_channels=STAGE3_BACKBONE_OUT_CH,
    backbone_weight=sonata_pretrain_weight,
    levels=particle_levels,
    scale_pattern=particle_scale_pattern,
    token_dim=STAGE3_TOKEN_DIM,
    num_queries=STAGE3_NUM_QUERIES,
    num_classes=STAGE3_NUM_CLASSES,
    freeze_backbone=True,
    # delta: keep PT-v3 decoder trainable while the encoder stays frozen.
    # `unfreeze_decoder=True` sets requires_grad=True on `*.dec.*` params
    # only — the encoder + the rest stay frozen. Used alongside
    # `ptv3_decoder_init_scale` for stable from-scratch training (the
    # Sonata pretrain doesn't contain decoder weights).
    unfreeze_decoder=True,
    capture_decoder_stages=True,        # hooks on dec3/dec2 for the
                                        # PTv3DecoderStageLevel builders.
    ptv3_decoder_init_scale=0.01,       # small-mag init for the decoder
                                        # Block weights (attn.qkv,
                                        # attn.proj, mlp.fc2). std=0.01
                                        # is ~6× smaller than default
                                        # Linear init — near-identity at
                                        # iter 0 but gradient flows.
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
        source_level="voxel_4cm",
        score_source="cls_head",
        selection_mode="top_m_then_fps",
        score_filter_multiplier=4,
    ),
    mask_denoising=dict(
        dn_groups=3,
        max_dn_per_event=64,
        anchor_jitter_std=0.05,
    ),
    #mask_denoising=None,
)

# =============================================================================
# Trainer + evaluator (unchanged from v1-cached).
# =============================================================================
train = dict(type="LArFormerTrainer")

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="LArFormerParticleEvaluator",
         best_metric="mask_iou_mean",
         class_names=["e", "gamma", "mu", "pi", "p", "other",
                      "(unused)", "no_object"],
         coord_scale=coord_scale),
    dict(type="CheckpointSaver", save_freq=1),
    # Iteration-level checkpointing so SLURM jobs killed at the 24h wall-clock
    # cap can resume mid-epoch (epochs are ~9h on the Isambard-AI allocation).
    # Writes to the same model_last.pth that CheckpointSaver uses, plus
    # iter_in_epoch + RNG state so CheckpointLoader can pick up mid-epoch.
    # save_iter_freq=500 ≈ ~9 min between saves at ~30k batches/epoch, so the
    # worst-case wasted compute on a kill is bounded by that.
    dict(type="IterCheckpointSaver", save_iter_freq=50, keep_history=False),
    # Catch SLURM's pre-timeout SIGUSR1 (sent via --signal=USR1@1800), save a
    # checkpoint, write a RESUBMIT marker so the batch script can chain the
    # next job, and exit cleanly before SLURM kills the process.
    dict(type="SignalCheckpointHook", check_every_n_iter=30),
    dict(type="PreciseEvaluator", test_last=False),
]

# =============================================================================
# Training loop knobs
# =============================================================================
weight = None
save_path        = "exp/larformer_particle_v1_cached_ptv3crosslevel_smallbatch_lr1e4"
epoch            = 20
eval_epoch       = 20
# delta vs v1-cached: PT-v3 decoder is now trained and consumes more
# memory/compute, so the default batch_size is halved. Bump it back up
# if your GPU has the headroom.
batch_size       = 16
batch_size_val   = 40
num_worker       = 12
num_worker_val   = 8
evaluate         = True
enable_amp       = False    
amp_dtype        = "bfloat16"
empty_cache      = False
# Same convention as the slicer hybrid config: the PT-v3 decoder, the
# cross-level refiner, and the M2F decoder all initialize randomly →
# first-few-iter gradients can spike. clip_grad=1.0 keeps them sane.
clip_grad        = 1.0
enable_wandb     = True
wandb_project    = "pointcept-larformer-stage3"
find_unused_parameters = True

# Mid-epoch resume strategy — same convention as the slicer config.
skip_dataloader_on_resume = True
resume_seed_strategy = "per_resume"

# =============================================================================
# Optimizer / scheduler
# =============================================================================
# IMPORTANT: param_dicts is None here (NOT keyword="backbone"). The
# v1-cached config used `keyword="backbone"` to zero the lr of the
# frozen Sonata head — but with `unfreeze_decoder=True` the PT-v3
# decoder lives under the `backbone.*` prefix too, and that pattern
# would silently zero its lr. `freeze_backbone=True` + `unfreeze_decoder=
# True` set requires_grad correctly per-param, and the optimizer skips
# params with requires_grad=False; no param_dicts override needed.
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
    min_lr=5e-8,
    step_period_epochs=50,
    patience_epochs=2,
    min_delta=1e-4,
    cooldown_epochs=1,
    # Linear warmup over the first 500 training iters (~100 epochs on the
    # 10-event dev sample at batch_size=2). PTv3 decoder + refiner +
    # Mask2Former decoder all initialize randomly here — gradients in the
    # first ~50 iters are noisy enough to spike the loss curve, so we
    # don't want the plateau detector touching anything during that
    # phase. step_epoch is a no-op while in warmup (counters frozen).
    warmup_iters=25625, # 1 epoch of warm up: 410k/16
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

# base_lr=2.0e-5
# optimizer = dict(
#     type="AdamW", lr=base_lr, weight_decay=0.05,
#     betas=(0.9, 0.95),
# )
# scheduler = dict(
#     type="OneCycleLR",
#     max_lr=base_lr,
#     pct_start=0.1,
#     anneal_strategy="cos",
#     div_factor=10.0,
#     final_div_factor=100.0,
# )
#param_dicts = None
