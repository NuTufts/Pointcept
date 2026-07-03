"""LArFormer Stage-3 training — v1.1 HYBRID: PTv3-decoder + crosslevel refiner.

Companion / sibling to `larformer-particle-v1-cached.py`. Same Stage-3
particle-segmenter targets (per-particle masks + per-query 7-class +
origin head), same `LArFormerStage12CacheDataset` cache reader, same
trainer + evaluator hooks.

Deltas from `larformer-particle-v1-cached-ptv3crosslevel.py` (v1.0):

  - **Level pyramid pruned + extended one stage deeper.** v1.0 had two
    voxel levels (voxel_16cm, voxel_8cm) plus dec3 / dec2 / spacepoint.
    v1.1 drops both voxel levels in favor of a single voxel_4cm + adds
    PT-v3's dec1 (~0.5 cm) so the cross-level refiner sees decoder-
    refined features at four scales (~4 / 2 / 1 / 0.5 cm). The model
    now gets all its medium-to-coarse spatial context from the PT-v3
    decoder stages rather than a separate user-defined voxel pyramid.
  - **Per-token cls supervision is now grounded in particle truth.**
    v1.0's cls block read `origin_label` (per-SP nu/cosmic) and ran
    `reduce="amax"` with a label_remap that pushed everything to
    no_object — a degenerate signal. v1.1 reads `particle_class_id`
    (the per-SP class label written to caches by
    `tools/larformer/augment_stage12_cache_particle_class_id.py`) and uses
    `reduce="soft_presence"`, which produces a per-voxel uniform-over-
    present-classes target. Easier to learn than the count-proportional
    `soft_distribution` variant; doesn't ask the model to memorize the
    exact SP-count mix per voxel.
  - **`mixed_query_selection.source_level` aligned with the cls block.**
    v1.0 had `source_level="voxel_8cm"` but the cls supervision was on
    voxel_4cm — the query selector was scoring queries against an
    untrained cls head. v1.1 puts both at voxel_4cm.

Backbone shape unchanged from v1.0:
  - PT-v3m2 NATIVE DECODER turned on (`enc_mode=False` +
    `up_cast_level=0`). The Sonata pretrain only contains encoder
    weights, so the decoder trains from scratch (small-mag init via
    `ptv3_decoder_init_scale=0.01`, see
    `LArFormer._init_ptv3_decoder_blocks`).
  - dec0 (~0.25 cm) is the per-SP output (read by the SPACEPOINT level).
  - dec1 / dec2 / dec3 are exposed as level tokens via
    `PTv3DecoderStageLevel` builders.
  - voxel_4cm pools off the per-SP dec0 output (64 ch).

5 levels, 6-layer scale_pattern (coarse → fine):
    voxel_4cm → ptv3_dec3 → ptv3_dec2 → ptv3_dec1 → spacepoint → spacepoint

For inference deployment, the trained weights load back into the
`particle_segmenter` slot of a `CascadedParticleSegmenter` configured
with the same `levels` / `scale_pattern` / `enc_mode=False`.

Notes:

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
  - Caches MUST be augmented with `entry_0/particle_class_id` (run
    `tools/larformer/augment_stage12_cache_particle_class_id.py` on the cache
    root) before training. Without that, the dataset emits all -1 for
    `particle_class_id` and the cls supervision becomes a no-op.
"""

_base_ = ["../../../../_base_/default_runtime.py"]

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
CACHE_ROOT  = "/home/twongjirad/working/larbys/gen2/container_u22/Pointcept/exp/cache_stage12_devdata/"
TRAIN_ROOT  = f"{CACHE_ROOT}/train"
VAL_ROOT    = f"{CACHE_ROOT}/val"

# Sonata pretrain for the Stage-3 backbone. Loaded via the LArFormer's
# own `backbone_weight` knob. NOTE: the Sonata pretrain was saved with
# enc_mode=True, so it only contains encoder weights — the PT-v3
# decoder trains from scratch in this config.
sonata_pretrain_weight = (
    "/home/twongjirad/working/larbys/gen2/container_u22/Pointcept/"
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
#   - dec1 @ stride 2   → 64 ch  (~0.5 cm grid)  → ptv3_dec1 level
#   - dec2 @ stride 4   → 128 ch (~1 cm grid)    → ptv3_dec2 level
#   - dec3 @ stride 8   → 256 ch (~2 cm grid)    → ptv3_dec3 level
_PTV3_DEC_CHANNELS    = (64, 64, 128, 256)
STAGE3_BACKBONE_OUT_CH = _PTV3_DEC_CHANNELS[0]   # 64 = dec0 width

# =============================================================================
# Geometry
# =============================================================================
coord_center = (125.0, 0.0, 518.0)
coord_scale  = 179.55
flash_backend = "flash_attn"

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
# Stage-3 particle segmenter — HYBRID levels (the delta vs v1-cached).
# =============================================================================
# Five levels, six decoder layers. The voxel_4cm level covers the coarse
# end (~4 cm tokens for the M2F decoder's query-anchor selection); the
# PTv3 decoder stages (dec3 @ ~2 cm, dec2 @ ~1 cm, dec1 @ ~0.5 cm) cover
# the medium-to-fine end with decoder-refined features the queries can
# cross-attend to; the spacepoint level (dec0 @ ~0.25 cm) carries the
# per-SP primary mask supervision.
particle_levels = [
    dict(name="voxel_4cm",
         builder="VoxelBuilder",
         builder_cfg=dict(voxel_size_cm=4.0, coord_scale=coord_scale),
         supervision=dict(
            mask=dict(weight=1.0, mode="aux"),
            # Per-token cls aux supervision at 4 cm voxels.
            #
            #   label_src: per-SP `particle_class_id` (0..5 visible
            #     classes, -1 = SP not in any GT particle = Stage-2
            #     false positive). Must be present in the cache —
            #     run `tools/larformer/augment_stage12_cache_particle_class_id.py`
            #     to write it.
            #
            #   reduce: "soft_presence" produces a per-voxel uniform-
            #     over-present-classes distribution. A voxel with γ
            #     SPs + p SPs + some -1 FPs → target [γ=1/3, p=1/3,
            #     no_object=1/3]. The -1 SPs are mapped to the
            #     no_object slot (=7) so FP-heavy voxels learn high
            #     p(no_object), which `mixed_query_selection`'s
            #     `1 - p(no_object)` scoring needs.
            #
            #   Soft-presence is the recommended supervision: it
            #     captures "which classes are here" without asking
            #     the model to memorize the exact SP-count mix
            #     (which depends on voxel boundary placement and is
            #     essentially unpredictable from voxel appearance).
            #     If you want count-proportional supervision, switch
            #     to reduce="soft_distribution".
            #cls=dict(num_classes=STAGE3_NUM_CLASSES,
            #         label_src="particle_class_id",
            #         reduce="soft_presence",
            #         weight=0.3, loss="ce",
            #         ignore_index=-1),
        ),
    ),
    dict(name="ptv3_dec3",
         builder="PTv3DecoderStageLevel",
         builder_cfg=dict(stage_key="dec3", in_dim=_PTV3_DEC_CHANNELS[3]),
         supervision=dict(
            mask=dict(weight=1.0, mode="aux"),
            cls=dict(num_classes=STAGE3_NUM_CLASSES,
                     label_src="particle_class_id",
                     reduce="soft_presence",
                     weight=0.3, loss="ce",
                     ignore_index=-1),
         ),
    ),
    dict(name="ptv3_dec2",
         builder="PTv3DecoderStageLevel",
         builder_cfg=dict(stage_key="dec2", in_dim=_PTV3_DEC_CHANNELS[2]),
         supervision=dict(mask=dict(weight=1.0, mode="aux"))),
    dict(name="ptv3_dec1",
         builder="PTv3DecoderStageLevel",
         builder_cfg=dict(stage_key="dec1", in_dim=_PTV3_DEC_CHANNELS[1]),
         supervision=dict(mask=dict(weight=1.0, mode="aux"))),
    dict(name="spacepoint",
         builder="SpacepointBuilder",
         supervision=dict(mask=dict(weight=5.0, mode="primary"))),
]
# 6 layers, coarse → fine. Same depth as the slicer hybrid config.
particle_scale_pattern = [
    "voxel_4cm",  # 4.0   cm
    "ptv3_dec3",  # 2.0   cm
    "ptv3_dec2",  # 1.0   cm
    "ptv3_dec1",  # 0.50  cm
    "spacepoint", # 0.25  cm
    "spacepoint", # 0.25  cm
]

# Cross-level refiner runs over the voxel_4cm level + all three PT-v3
# decoder stages (dec3 / dec2 / dec1). The spacepoint level is excluded
# from the refiner's target_levels — it has the most tokens (one per
# input SP) and contributes via the primary mask supervision, not the
# refiner's cross-level attention.
#
# `max_source_tokens_per_level=8192` caps the K/V contribution from any
# single source level. The finer PT-v3 stages (dec1 @ ~0.5 cm) can
# easily exceed a few thousand tokens for a dense nu slice; the cap
# keeps memory bounded.
_particle_token_refiner_cfg = dict(
    type="CrossLevelAttn",
    num_layers=2,
    num_heads=4,
    mlp_ratio=4.0,
    target_levels=["voxel_4cm", "ptv3_dec3", "ptv3_dec2", "ptv3_dec1"],
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
        #source_level="voxel_4cm",
        source_level="ptv3_dec3",
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
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]

# =============================================================================
# Training loop knobs
# =============================================================================
weight = None
save_path        = "exp/larformer_particle_v1.1_cached_ptv3crosslevel_10eventtest"
epoch            = 1000
eval_epoch       = 1000
# delta vs v1-cached: PT-v3 decoder is now trained and consumes more
# memory/compute, so the default batch_size is halved. Bump it back up
# if your GPU has the headroom.
batch_size       = 10
batch_size_val   = 10
num_worker       = 2
num_worker_val   = 2
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
optimizer = dict(
    type="AdamW", lr=1e-5, weight_decay=0.05,
    betas=(0.9, 0.95),
)
scheduler = dict(
    type="OneCycleLR",
    max_lr=1e-5,
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)
param_dicts = None
