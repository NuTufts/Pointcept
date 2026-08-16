"""LArFormer Stage-3 particle segmenter V2 — M2F-recipe retrain.

Applies the full recipe validated by the Stage-2 slicer m2frecipe-v2 retrain
(see configs/lartpc/larformer/stage2_slicer/...-m2frecipe{,-v2}.py and the
review notes) to the Stage-3 particle segmenter. Model / levels / dataset
reader are UNCHANGED from `larformer-particle-v1-cached-ptv3crosslevel.py`;
only the training recipe moves:

  1. use_vectorized_pair_loss=True — batched per-pair BCE/Dice (2.04x
     end-to-end on the slicer, bench job 1809369). NOTE the small-instance
     corner: masks with < num_sample_points/2 positives get an under-filled
     sample, so diag_*_rand values are not comparable to loop-based runs.
  2. match_per_layer=True — per-layer Hungarian re-matching (standard
     DETR/M2F deep supervision). On the slicer, the shared-final-match
     scheme showed the churn signature (diag_match_agreement 0.3 -> 0.1
     regression); per-layer matching + fewer near-duplicate queries fixed
     the v2 run. Stage-3 already runs 32 queries, so no query reduction
     is needed here.
  3. Weight-decay hygiene: no_decay_on_1d_and_embeddings excludes norms,
     biases (1D rule) and the learnable queries (query_content, query_pos,
     DN class_embedding) by explicit keyword list (the builder default's
     bare "embed" would wrongly exempt the mask_embed head MLP); decay
     group raised 0.01 -> 0.05 (M2F default).
  4. OneCycleLR cosine replaces FlatWithDecayLR-plateau. The plateau decay
     NEVER fired on the v1 stage-3 runs (stability doc §1.4: LREpochScheduler
     wasn't even hooked) — the production checkpoint needed a mid-run
     DelayedCosineLR rescue. Horizon sized to epoch=8 below; warmup =
     pct_start * total_steps ≈ 2.7k optimizer steps (replaces the absurd
     1-epoch warmup_iters=25625 in the v1 config).
  5. clip_grad 1.0 -> 0.1 (aggressive clipping is what makes lr 1e-4 safe
     through the noisy early-matching phase; drop to 0.01 if grad_norm
     spikes persist past warmup).
  6. cost_origin=0.5 == weight_origin — matcher costs now literally
     identical to loss weights (matcher default cost_origin=1.0 silently
     mismatched the 0.5 loss weight in v1; the origin head is ACTIVE in
     stage 3, so unlike the slicer this was a real, if mild, mismatch).
  7. log_diagnostics=True + the intra-epoch valprobe (probe_freq below;
     forwarded through LArFormerParticleEvaluator as of this retrain).
     The slicer overfit by epoch ~3 and the valprobe is what showed it —
     watch valprobe/loss vs train loss here for the same onset.
  8. Hooks: plain CheckpointLoader (v1's reset_optimizer=True was a
     warm-start artifact and would wipe Adam state on EVERY chained
     resume).

Already recipe-compliant in v1, carried over unchanged: lr 1e-4, eos
no_object_weight=0.1, 32 queries, 2/5/5 class/mask/dice weights = matcher
costs, mask denoising, mixed query selection, importance+hard-neg sampling.

*** CACHE DEPENDENCY — READ BEFORE LAUNCHING ***
CACHE_ROOT now points at the built+verified v2 cache (2026-08-09). Original note: The v1 cache was built with the OLD
slicer (iter_75750); training stage 3 against the NEW m2frecipe-v2 slicer
requires rebuilding the stage-1+2 cache with the chosen v2 slicer
checkpoint first:
    tools/larformer/build_stage12_cache_shard.py   (stage-1+2 inference)
    tools/larformer/augment_stage12_cache_particle_class_id.py
        (required by voxel_4cm soft-presence cls supervision)
Then point CACHE_ROOT at the new cache. The config intentionally fails
fast (missing path) if launched before that.
"""

_base_ = ["../../../_base_/default_runtime.py"]

# Side-effect: register LArFormerTrainer + LArFormerParticleEvaluator.
from pointcept.models.LArFormer import trainer as _larformer_trainer_module
from pointcept.models.LArFormer import particle_evaluator as _larformer_particle_evaluator_module
del _larformer_trainer_module
del _larformer_particle_evaluator_module

# =============================================================================
# Paths
# =============================================================================
# v2 production cache (built 2026-08-09, VERIFIED: 392,131 files, 0 corrupt,
# exact coverage, particle_class_id augmented): ft-deghoster @ tau=0.20 +
# m2frecipe-v2 ep4 slicer; 382,768 train / 9,363 val events.
CACHE_ROOT  = "/cluster/tufts/wongjiradlab/larbys/data/ub_on_tufts/hdf5/larformer_cache_stage12__m2fv2ep4_ftdeghost_tau020/"
TRAIN_ROOT  = f"{CACHE_ROOT}/train"
VAL_ROOT    = f"{CACHE_ROOT}/val"

# Sonata pretrain for the Stage-3 backbone (encoder only; the PT-v3 decoder
# trains from scratch — see ptv3_decoder_init_scale).
sonata_pretrain_weight = (
    "/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/"
    "sonata/lartpc_v6_h200_noghosts_pretrain_logspace_resume/model/epoch_42.pth"
)

# =============================================================================
# Toggles — mirror the v1 config for parity
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
# Backbone shape — PTv3-decoder mode (unchanged from v1)
# =============================================================================
_PTV3_DEC_CHANNELS    = (64, 64, 128, 256)
STAGE3_BACKBONE_OUT_CH = _PTV3_DEC_CHANNELS[0]   # 64 = dec0 width

# =============================================================================
# Geometry
# =============================================================================
coord_center = (125.0, 0.0, 518.0)
coord_scale  = 179.55
flash_backend = "flash_attn"

# =============================================================================
# Dataset (cache reader) — identical to v1 except the cache root.
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
# Stage-3 particle segmenter — HYBRID levels (unchanged from v1)
# =============================================================================
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
             # Per-token soft-presence cls supervision at 4 cm voxels; per-SP
             # `particle_class_id` must be present in the cache (see the
             # augment tool named in the module docstring). -1 SPs (Stage-2
             # FPs) map to the no_object slot for mixed_query_selection's
             # `1 - p(no_object)` scoring.
             cls=dict(num_classes=STAGE3_NUM_CLASSES,
                      label_src="particle_class_id",
                      reduce="soft_presence",
                      weight=0.3, loss="ce",
                      ignore_index=-1),
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
            enc_mode=False,
            mask_token=False,
        ),
        head_in_channels=1088, head_hidden_channels=2048,
        head_embed_channels=256, head_num_prototypes=4096,
        num_global_view=2, num_local_view=6,
        up_cast_level=0,
    ),
    backbone_out_channels=STAGE3_BACKBONE_OUT_CH,
    backbone_weight=sonata_pretrain_weight,
    levels=particle_levels,
    scale_pattern=particle_scale_pattern,
    token_dim=STAGE3_TOKEN_DIM,
    num_queries=STAGE3_NUM_QUERIES,
    num_classes=STAGE3_NUM_CLASSES,
    freeze_backbone=True,
    unfreeze_decoder=True,
    capture_decoder_stages=True,
    ptv3_decoder_init_scale=0.01,
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
        # --- M2F-RECIPE additions (see module docstring items 1, 2, 6, 7) ---
        use_vectorized_pair_loss=True,
        match_per_layer=True,
        # Matcher costs ≡ loss weights: origin head is active with
        # weight_origin=0.5; the matcher's default cost_origin=1.0 silently
        # broke the invariant in v1.
        cost_origin=0.5,
        log_diagnostics=True,
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
)

# =============================================================================
# Trainer + evaluator
# =============================================================================
train = dict(type="LArFormerTrainer")

hooks = [
    # Plain loader — v1's reset_optimizer=True was a warm-start artifact and
    # would wipe Adam moments on every chained resume.
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="LArFormerParticleEvaluator",
         best_metric="mask_iou_mean",
         class_names=["e", "gamma", "mu", "pi", "p", "other",
                      "(unused)", "no_object"],
         coord_scale=coord_scale,
         # Intra-epoch val-loss probe (overfitting watch) — the slicer
         # overfit by ~epoch 3 and the probe is what showed it clearly.
         probe_freq=1000,
         probe_max_events=256),
    dict(type="CheckpointSaver", save_freq=1),
    dict(type="IterCheckpointSaver", save_iter_freq=100, keep_history=False),
    dict(type="SignalCheckpointHook", check_every_n_iter=30),
    dict(type="PreciseEvaluator", test_last=False),
    dict(
        type="AdamStateMonitor",
        log_frequency=10,
        prefix="adam_state",
        track_layers=True,
        track_histograms=False,
        histogram_frequency=500,
    ),
]

# =============================================================================
# Training loop knobs
# =============================================================================
weight = None
save_path        = "exp/larformer_particle_v2_cached_ptv3crosslevel_m2frecipe"
# Horizon sized to the recipe, not to the v1 nominal 20: at lr 1e-4 the
# slicer converged (and began overfitting) by ~epoch 3 of a comparable
# step budget. 8 epochs x 22588 iters (v1 cache; recompute for the new
# cache) ≈ 181k optimizer steps with the cosine completing at the end.
# epoch and eval_epoch MUST move together (OneCycleLR horizon =
# len(train_loader) * eval_epoch). Watch valprobe/loss for the overfit
# onset; CheckpointSaver keeps every epoch for post-hoc selection.
epoch            = 8
eval_epoch       = 8
batch_size       = 16
batch_size_val   = 40
num_worker       = 12
num_worker_val   = 8
evaluate         = True
enable_amp       = False
amp_dtype        = "bfloat16"
empty_cache      = False
# M2F-RECIPE: 1.0 -> 0.1 (see module docstring item 5).
clip_grad        = 0.1
enable_wandb     = True
wandb_project    = "pointcept-larformer-stage3"
find_unused_parameters = True

# Mid-epoch resume strategy — same convention as the slicer configs.
skip_dataloader_on_resume = True
resume_seed_strategy = "per_resume"

# =============================================================================
# Optimizer / scheduler — M2F-RECIPE block
# =============================================================================
# param_dicts stays None: freeze_backbone=True + unfreeze_decoder=True set
# requires_grad per-param correctly (see the v1 config's warning about the
# keyword="backbone" pattern zeroing the trainable PT-v3 decoder's lr).
base_lr = 1.0e-4
param_dicts = None
optimizer = dict(
    type="AdamW",
    lr=base_lr,
    # wd 0.01 -> 0.05 (M2F default) on the decay group only.
    weight_decay=0.05,
    # Exclude from decay: 1D params (norms, biases) via the builder's ndim
    # rule + the learnable queries by explicit name. Explicit list rather
    # than True: the builder default's bare "embed" keyword would also
    # exempt the mask_embed head MLP, which should keep decaying.
    no_decay_on_1d_and_embeddings=["query_content", "query_pos",
                                   "class_embedding"],
)
# OneCycleLR replaces FlatWithDecayLR-plateau (which never fired on the v1
# stage-3 runs — stability doc §1.4). Pure function of the optimizer-step
# counter: chained SLURM resumes land on exactly the right LR with no
# plateau state. total_steps injected by the trainer
# (= len(train_loader) * eval_epoch // grad accum).
scheduler = dict(
    type="OneCycleLR",
    max_lr=base_lr,
    # ~0.015 * 181k ≈ 2.7k optimizer steps of warmup (4e-6 -> 1e-4),
    # replacing the v1 config's 1-epoch warmup_iters=25625.
    pct_start=0.015,
    anneal_strategy="cos",
    div_factor=25.0,        # start LR = max_lr / 25 = 4e-6
    final_div_factor=100.0, # end LR = start / 100 = 4e-8
)
