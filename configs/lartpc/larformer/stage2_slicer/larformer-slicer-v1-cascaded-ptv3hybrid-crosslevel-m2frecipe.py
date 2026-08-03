"""
LArFormer Stage-2 cascaded slicer — PTv3-decoder + custom-voxel HYBRID variant,
M2F-RECIPE retrain (cap300k run 2).

Identical model/data to `larformer-slicer-v1-cascaded-ptv3hybrid-crosslevel.py`
(the cap300k retrain config) but with the training recipe aligned to standard
Mask2Former practice after the first cap300k run converged too slowly at flat
lr 1e-5 (~45 h/epoch wall, plateau scheduler never in a position to act).
Full review + rationale: ~/.claude/plans/zany-whistling-pine.md (2026-07-25).

Recipe deltas vs the parent config (model & data untouched):
  1. base_lr 1e-5 -> 1e-4 (the M2F recipe at effective batch 16), with
     OneCycleLR replacing FlatWithDecayLR-plateau. Plateau decisions took
     >= patience(2) + cooldown(1) epochs ~ 1 week at this cadence, so the
     LR never decayed in practice; OneCycle is a pure function of the
     optimizer-step counter (robust across the 48 h SLURM resume chain)
     and its warmup phase replaces warmup_iters.
  2. epoch 50 -> 5: size the cosine horizon to the real wall-clock budget
     (~5 x 45 h ~ 10 days). 5 epochs = 128,125 optimizer steps at bs16 —
     about a third of Mask2Former's 368k-step COCO schedule, reasonable
     for 3 classes on a frozen pretrained encoder.
  3. Weight-decay hygiene: no_decay_on_1d_and_embeddings excludes norms,
     biases (1D rule) and the learnable queries (query_content, query_pos,
     DN class_embedding) from decay; decay group raised 0.01 -> 0.05 (M2F).
     Explicit keyword list — the builder's default bare "embed" keyword
     would also exempt the mask_embed head MLP, which should keep decaying.
  4. no_object_weight 0.5 -> 0.1 (M2F eos_coef; with 128 queries and
     typically <= 20 GT slices, 0.5 let the no-object CE dominate 5x the
     standard recipe and suppress query specialization).
  5. clip_grad 1.0 -> 0.1 — aggressive clipping is what makes the 10x LR
     safe through the noisy early-matching phase. (M2F uses 0.01, but its
     num_masks-sum loss normalization yields larger grads than this
     codebase's mean-over-pairs/events; drop to 0.01 if grad_norm spikes.)
  6. log_diagnostics=True — matching-stability + mask diagnostics
     (diag_match_agreement, diag_mask_iou_matched, ...) logged every iter.
  7. cost_origin=0.0 housekeeping so matcher costs are literally identical
     to the loss weights (origin head is disabled; the term was a benign
     column-constant).

Deliberately NOT in this run (later ablations): num_queries reduction,
per-layer Hungarian matching, vectorized pair loss, bf16 AMP.

-------------------------------------------------------------------------------
Parent-config description (model unchanged):

Combines:
  - Two USER-DEFINED COARSE VOXEL LEVELS (voxel_16cm, voxel_8cm) pooled
    from PTv3's per-SP decoder output (dec0 / 64-channel features). The
    voxel pyramid extends the spatial coverage beyond PTv3's natural
    pyramid (which only reaches 2 cm at dec3 given the 0.25 cm input
    grid).
  - Two PTv3 NATIVE DECODER STAGES (dec3 @ 2 cm, dec2 @ 1 cm) consumed
    via `PTv3DecoderStageLevel`. These give the queries transformer-
    refined features at the fine end of the pyramid.
  - The SPACEPOINT LEVEL (= dec0 here, since enc_mode=False +
    up_cast_level=0 makes dec0 the final per-SP output).

Refinement: CrossLevelAttn on all four non-SP levels. Aux per-token cls
supervision on voxel_8cm. 6-layer scale_pattern (coarse -> fine):
    voxel_16cm -> voxel_8cm -> ptv3_dec3 -> ptv3_dec2 -> spacepoint -> spacepoint

Pretrained encoder weights load cleanly (frozen). The PTv3 decoder trains
from scratch. The refiner and Mask2Former decoder + heads also train from
scratch.
"""

_base_ = ["../../../_base_/default_runtime.py"]

# Side-effect: register LArFormerTrainer + LArFormerSlicerEvaluator.
from pointcept.models.LArFormer import trainer as _larformer_trainer_module
from pointcept.models.LArFormer import evaluator as _larformer_evaluator_module
del _larformer_trainer_module
del _larformer_evaluator_module

# =============================================================================
# Paths
# =============================================================================
TRAIN_FILE_LIST = "/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/lartpc_data_prep/lantern_scripts/h5lists/h5list_mcall_lantern_train.txt"
VAL_FILE_LIST   = "/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/lartpc_data_prep/lantern_scripts/h5lists/h5list_mcall_lantern_val.txt"

# Trained SonataLoRADeghostSegmentor checkpoint for Stage 1.
deghoster_weight = "/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/sonata/lora_deghost_v6_hasmatch/model/epoch_30.pth"

slicer_backbone_weight = (
    "/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/"
    "sonata/lartpc_v6_h200_noghosts_pretrain_logspace_resume/model/epoch_42.pth"
)

# =============================================================================
# Toggles (kept here so this config is independently reproducible and so the
# toggle semantics match the sibling configs).
# =============================================================================
USE_SINUSOIDAL_POS_EMB = False
ENABLE_ORIGIN_HEAD_WITH_CENTROID = False

# Per-pair negative sampler — halo + confident-FP mining half-and-half.
# See docs/LArFormer.md §15.
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
# doesn't try to upcast the already-decoded output.
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
    # max_spacepoints caps the PRE-DEGHOST point count (after 0.25cm dedup); an
    # over-cap event is RANDOMLY thinned before the backbone. The old 100k train
    # cap bit 66.7% of events (median 117.5k SP) -- see
    # configs/lartpc/larformer/README.md. Raised to 300k (bite -> 1.6%) to stop
    # decimating sparse/soft showers, which the slicer was losing. val runs
    # under no_grad (cheaper) so it carries a larger 450k cap (1.5x, matching the
    # old train:val ratio) for a nearly-unthinned validation signal.
    train=dict(split="train", data_root="/", data_list_file=TRAIN_FILE_LIST,
               loop=1, max_spacepoints=300_000, **_dataset_common),
    val=dict(split="val", data_root="/", data_list_file=VAL_FILE_LIST,
             loop=1, max_spacepoints=450_000, **_dataset_common),
    test=dict(split="test", data_root="/", data_list_file=VAL_FILE_LIST,
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
# Stage 2 — Slicer subconfig (HYBRID levels) — unchanged from parent
# =============================================================================
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
# Token refiner — CrossLevelAttn on all non-SP levels
# =============================================================================

_token_refiner_cfg = dict(
    type="CrossLevelAttn",
    num_layers=2,
    num_heads=4,
    mlp_ratio=4.0,
    target_levels=["voxel_16cm", "voxel_8cm", "ptv3_dec3", "ptv3_dec2"],
    max_source_tokens_per_level=16392,      # cap SP's K/V contribution
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
    # Small-mag init for PT-v3m2 decoder Block weights. See parent config /
    # _init_ptv3_decoder_blocks docstring.
    ptv3_decoder_init_scale=0.01,
    enable_origin_head=ENABLE_ORIGIN_HEAD_WITH_CENTROID,
    token_refiner=_token_refiner_cfg,
    decoder_kwargs=dict(
        num_heads=4, mlp_ratio=4.0,
        # M2F decoder uses random init (zero-init was a known
        # convergence-rate regression); the PT-v3m2 inner decoder's
        # small-mag init is what keeps the from-scratch cascade stable.
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
        # M2F-RECIPE: eos_coef 0.5 -> 0.1 (standard M2F). With 128 queries
        # and typically <=20 GT slices, ~85% of queries are no-object every
        # iter; 0.5 let that CE term dominate 5x the standard recipe.
        no_object_weight=0.1,
        # M2F-RECIPE: origin head is disabled, so zero the matcher's origin
        # cost too — matching costs now literally identical to loss weights.
        # (Was benign — zero-filled origin makes the term column-constant —
        # but this keeps the invariant explicit.)
        cost_origin=0.0,
        # M2F-RECIPE: log matching-stability + mask diagnostics
        # (diag_match_agreement, diag_mask_iou_matched, diag_mask_bce_rand,
        # diag_frac_confident_fp, ...) — no_grad, final layer only, never
        # contributes to the loss. Watch diag_match_agreement: if the
        # init-vs-final Hungarian assignment still churns late in training,
        # that's the problem, not the loss weights.
        log_diagnostics=True,
    ),
    # Phase A: DINO/Mask-DINO mixed query selection (see docs/LArFormer.md §17).
    mixed_query_selection=dict(
        source_level="voxel_8cm",
        score_source="cls_head",
        selection_mode="top_m_then_fps",
        score_filter_multiplier=4,
    ),
    # Phase B: Mask DINO-style mask denoising (see docs/LArFormer.md §18).
    mask_denoising=dict(
        dn_groups=3,
        max_dn_per_event=96,
        anchor_jitter_std=0.05,
    ),
)

# =============================================================================
# CascadedSlicer
# =============================================================================

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

save_path        = "exp/larformer_slicer_v1_cascaded_ptv3hybrid_crosslevel_cap300k_m2frecipe"
# M2F-RECIPE: size the schedule to the wall clock, not to a nominal 50.
# 5 epochs x ~45 h ~ 10 days; 5 x 25625 = 128,125 optimizer steps at bs16
# (~1/3 of Mask2Former's 368k-step COCO schedule). eval_epoch must
# equal epoch — the trainer injects OneCycleLR's total_steps as
# len(train_loader) * eval_epoch // gradient_accumulation_steps, and the
# cosine must complete exactly at the end of training.
epoch            = 5
eval_epoch       = 5
# Memory budget at the 300k cap: 4x A100-80GB, batch_size TOTAL across GPUs;
# 16 total = 4/GPU physical, no accumulation. Worst-case smoke (40 biggest
# events, 231k-481k SP) peaked at 41.5 GB/GPU at 4/GPU — ~2x headroom.
batch_size       = 16
gradient_accumulation_steps = 1
# val runs under no_grad but at the larger 450k cap; keep the per-GPU val batch
# small so a cosmic-pileup val event doesn't OOM the eval forward.
batch_size_val   = 4
num_worker       = 12
num_worker_val   = 12
evaluate         = True
enable_amp       = False
amp_dtype        = "bfloat16" # use this for default 'flash_attn' backend
empty_cache      = False
# M2F-RECIPE: clip 1.0 -> 0.1. Aggressive norm-clipping is what makes the
# 10x LR raise safe through the noisy early-matching phase (M2F itself
# clips at 0.01, but its num_masks-sum loss normalization produces larger
# gradients than this codebase's mean-over-pairs/events aggregation).
# grad_norm is logged pre-clip — if it spikes persistently or the loss
# destabilizes early, drop to 0.01.
clip_grad        = 0.1
enable_wandb     = True
wandb_project    = "pointcept-larformer"
find_unused_parameters = True

# Mid-epoch resume strategy (see parent config for details).
skip_dataloader_on_resume = True
resume_seed_strategy = "per_resume"

# =============================================================================
# Optimizer / scheduler — M2F-RECIPE block (see module docstring)
# =============================================================================
# lr 1e-5 -> 1e-4: the standard Mask2Former recipe at effective batch 16.
# Everything that trains (PTv3 decoder, refiner, M2F decoder, heads) is
# from-scratch, so no backbone LR multiplier is needed (encoder frozen).
# The NaN protections that motivated the old conservative setting (logit
# clamps, matcher sanitizer, ptv3_decoder_init_scale=0.01) are unchanged.
base_lr = 1.0e-4
param_dicts = None
optimizer = dict(
    type="AdamW",
    lr=base_lr,
    # wd 0.01 -> 0.05 (M2F default) on the decay group only.
    weight_decay=0.05,
    # Exclude from weight decay: all 1D params (norm gains, biases) via the
    # builder's ndim rule, plus the learnable queries by name. Explicit
    # keyword list rather than `True`: the builder's default list includes
    # a bare "embed" keyword that would also exempt the mask_embed head MLP
    # (which should keep decaying). `class_embedding` is the MaskDenoiser's
    # DN-query content init (query-feature-like -> no decay).
    no_decay_on_1d_and_embeddings=["query_content", "query_pos",
                                   "class_embedding"],
)
# OneCycleLR replaces FlatWithDecayLR-plateau: at ~45 h/epoch the plateau
# machinery (once-per-epoch decisions, patience 2 + cooldown 1 on an EMA)
# could never act inside a realistic run window — the old run trained flat
# at base_lr throughout. OneCycle's LR is a pure function of the
# optimizer-step counter, so every 48 h chain resume lands on exactly the
# right LR with no plateau state to restore. total_steps is injected by
# the trainer (= 128,125 for 5 epochs at bs16).
scheduler = dict(
    type="OneCycleLR",
    max_lr=base_lr,
    # Warmup = pct_start * total_steps = 0.02 * 128,125 ~ 2,560 optimizer
    # steps (~4.5 h wall) ramping 4e-6 -> 1e-4. Replaces the old
    # warmup_iters=25625 (a full ~45 h epoch, far too long).
    pct_start=0.02,
    anneal_strategy="cos",
    div_factor=25.0,        # start LR = max_lr / 25 = 4e-6
    final_div_factor=100.0, # end LR = start / 100 = 4e-8
)

hooks = [
    dict(type="CheckpointLoader", extend_scheduler=False),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="LArFormerSlicerEvaluator",
         eval_freq=0,            # full metric eval + best-ckpt: once per epoch
         best_metric="nu_mIoU",
         class_names=["nu", "cosmic"],
         nu_class_id=0,
         empty_cache=True,
         log_per_event=False,
         # Intra-epoch val-LOSS probe for overfitting watch (epochs are ~45h;
         # the once-per-epoch val/loss is far too coarse). Loss-only eval over
         # the first `probe_max_events` val events every `probe_freq` global
         # iters, logged as `valprobe/loss[_*]` on the training-step axis.
         probe_freq=1000,
         probe_max_events=128),
    # Harmless no-op with OneCycleLR (no step_epoch method) — kept because its
    # before_train hook logs the actual starting LR after any resume.
    dict(type="LREpochScheduler"),
    dict(type="CheckpointSaver", save_freq=1),
    # Iteration-level checkpointing so SLURM jobs killed at the 48h wall-clock
    # cap can resume mid-epoch (epochs are ~45h). Writes to the same
    # model_last.pth that CheckpointSaver uses, plus iter_in_epoch + RNG state
    # so CheckpointLoader can pick up mid-epoch.
    dict(type="IterCheckpointSaver", save_iter_freq=50, keep_history=False),
    # SIGUSR1 does not propagate through apptainer here (see submit script);
    # kept for environments where it does.
    dict(type="SignalCheckpointHook", check_every_n_iter=30),
]

train = dict(type="LArFormerTrainer")
