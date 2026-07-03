"""
Config for the shower-clustering Mask2Former model. Phases 2–8 of the design
(see pointcept/docs/shower_clustering_design.md).

Builds:
  - ShowerClusteringDataset                   (Phase 2)
  - ShowerClusteringMask2Former               (Phase 7)
  - ShowerClusteringTrainer                   (Phase 8)
  - SonataCheckpointLoader hook for backbone  (loads cfg.weight)

Defaults are tuned for a Phase 8 one-event smoke test on a single P100:
batch_size=1, epoch=1, num_worker=0, evaluate=False. Bump these for real
training runs.

The visualizer at tools/visualize_shower_clustering.py reads this config so
the GT labels shown match what the model sees during training.
"""

_base_ = ["../../../_base_/default_runtime.py"]

# Side-effect import: triggers `@TRAINERS.register_module()` on
# ShowerClusteringTrainer. Done here (not in the package __init__.py) to
# avoid a circular import — pointcept.engines.train imports pointcept.models
# before defining TRAINERS. We use `import X as _name` and then `del _name`
# because Pointcept's `Config._file2dict` captures every non-dunder name in
# the config's module namespace and dumps it; a leftover module / class
# object would crash the dumper's yapf round-trip.
import pointcept.models.shower_clustering.trainer as _trainer_module
del _trainer_module

# =============================================================================
# Paths (override in subclass configs / via CLI)
# =============================================================================
# Default: the 3 fixed-merge test events. Real training will replace these
# with the production train/val/test split lists once the full re-merge job
# completes (~1 day per pi0 sample, in flight as of 2026-05-04).
_DEFAULT_TRAIN_LIST = (
    "/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/"
    "lartpc/data_prep/training_data/h5lists/"
    "h5list_bnbnu_pi0filter_validated_train.txt"
    #"h5list_showercluster_testsample.txt"
)
_DEFAULT_VAL_LIST = (
    "/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/"
    "lartpc/data_prep/training_data/h5lists/"
    "h5list_bnbnu_pi0filter_validated_val.txt"
    #"h5list_showercluster_testsample.txt"
)

# Backbone pretrain (frozen Sonata-v1m1 / PT-v3m2). Reused as-is from V3.
weight = (
    "/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/"
    "sonata/lartpc_v6_h200_noghosts_pretrain_logspace_resume/model/epoch_42.pth"
)
# up_cast_level=4 returns features at the *input* spacepoint resolution.
# The V3 default of up_cast_level=2 returns features at level-2 (4× downsampled
# in the GridPooling chain), which works for V3's tiny cropped-fragment
# inference but NOT for our full-event Mask2Former that needs per-spacepoint
# masks. With up_cast_level=4 the feature dim is 48+96+192+384+512 = 1232.
backbone_out_channels = 1232

# =============================================================================
# Coordinate normalization (must match Sonata pretraining)
# =============================================================================
coord_center = (125.0, 0.0, 518.0)
coord_scale = 179.55

# =============================================================================
# Voxel / fragment parameters (see design doc §3, §4)
# =============================================================================
voxel_size_cm = 5.0
min_fragment_points_post_filter = 50  # matches DBSCAN's min_fragment_points

# =============================================================================
# lm_score threshold augmentation (design doc §4e)
# =============================================================================
lm_score_aug_low = 0.40        # production deghoster floor
lm_score_aug_high = 0.80       # cap to avoid emptying events
lm_score_val_threshold = 0.60  # fixed on val for run-to-run comparability

# =============================================================================
# Common dataset kwargs
# =============================================================================
_dataset_common = dict(
    type="ShowerClusteringDataset",
    coord_center=coord_center,
    coord_scale=coord_scale,
    voxel_size_cm=voxel_size_cm,
    lm_score_aug_low=lm_score_aug_low,
    lm_score_aug_high=lm_score_aug_high,
    lm_score_val_threshold=lm_score_val_threshold,
    min_fragment_points_post_filter=min_fragment_points_post_filter,
    log_transform_strength=True,
    wire_scale=1.0 / 3456.0,
    # GT spacepoint labeling source:
    #   "truth"    — true descendants of trunk trackid (ghosts excluded;
    #                forces the spacepoint mask head to learn de-ghosting
    #                jointly with clustering).
    #   "fragment" — union of surviving reco fragments whose plurality
    #                trackid descends from trunk. Ghosts that DBSCAN
    #                grouped with shower points are INCLUDED; shower
    #                points DBSCAN missed are excluded. Matches the
    #                reco-system clustering output and removes the
    #                de-ghosting burden from the spacepoint mask head.
    #   "union"    — set-union of the above. Includes reco-clustered
    #                ghosts AND diffuse true shower points DBSCAN missed.
    #                Most permissive: model isn't penalized for following
    #                DBSCAN's ghost grouping OR for picking up missed
    #                shower points.
    gt_label_source="union",
    transform=None,  # augmentation is handled inside the dataset itself
    # `max_spacepoints` is set per-split below (NOT here) so each split can
    # use a different cap. Conflict with **_dataset_common would raise a
    # duplicate-keyword TypeError otherwise.
)

dataset_type = "ShowerClusteringDataset"
data_root = "/"  # absolute paths only

data = dict(
    num_classes=5,  # inside / outside / on_track / ghost / true_track
    ignore_index=-1,
    names=["inside", "outside", "on_track", "ghost", "true_track"],
    train=dict(
        split="train",
        data_root=data_root,
        data_list_file=_DEFAULT_TRAIN_LIST,
        loop=1,
        # Cap train events at 100k spacepoints — bounds the per-event
        # backward-pass memory on P100. Drops a few small GT instances
        # on the largest events but most signal stays. Bump higher (or
        # to None) if you have an H200 budget.
        max_spacepoints=100_000,
        **_dataset_common,
    ),
    val=dict(
        split="val",
        data_root=data_root,
        data_list_file=_DEFAULT_VAL_LIST,
        loop=1,
        # Val is forward-only — needs ~half the memory of training. Cap
        # higher so reported val metrics aren't biased by the cap on small
        # GT instances. Set to None for full events if memory permits.
        max_spacepoints=150_000,
        **_dataset_common,
    ),
    test=dict(
        split="test",
        data_root=data_root,
        data_list_file=_DEFAULT_VAL_LIST,
        loop=1,
        # Test split: full events for production-quality inference.
        max_spacepoints=None,
        **_dataset_common,
    ),
)

# =============================================================================
# Model — Phase 7 assembly. ShowerClusteringMask2Former wires backbone +
# tokenizer + decoder + loss into one nn.Module registered with the MODELS
# registry. See pointcept/docs/shower_clustering_design.md.
# =============================================================================
# V3 uses 'flash_attn' on H200 and 'xformers' on P100. We default to
# 'xformers' so the same config works on either, since the user prototypes
# on the 6×P100 box.
flash_backend = "xformers"

model = dict(
    type="ShowerClusteringMask2Former",
    backbone=dict(
        type="Sonata-v1m1",
        backbone=dict(
            type="PT-v3m2",
            in_channels=6,  # coord(3) + log(strength)(3)
            order=("z", "z-trans", "hilbert", "hilbert-trans"),
            stride=(2, 2, 2, 2),
            enc_depths=(3, 3, 3, 9, 3),
            enc_channels=(48, 96, 192, 384, 512),
            enc_num_head=(3, 6, 12, 24, 32),
            enc_patch_size=(256, 256, 256, 256, 256),
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            drop_path=0.0,  # frozen → no drop
            shuffle_orders=False,  # deterministic on inference
            pre_norm=True,
            enable_rpe=False,
            enable_flash=True,
            flash_backend=flash_backend,
            upcast_attention=False,
            upcast_softmax=False,
            traceable=True,
            enc_mode=True,
            mask_token=False,  # not pretraining
        ),
        head_in_channels=1088,         # unused in return_point=True path
        head_hidden_channels=2048,     # ditto
        head_embed_channels=256,       # ditto
        head_num_prototypes=4096,      # ditto
        num_global_view=2,             # ditto
        num_local_view=6,              # ditto
        up_cast_level=4,               # full input resolution (see comment above)
    ),
    backbone_out_channels=backbone_out_channels,
    token_dim=256,
    num_queries=64,
    num_classes=6,            # 5 origin types + 1 no_object
    no_object_class_id=5,
    num_decoder_layers=6,
    scale_pattern=("voxel", "fragment", "voxel", "fragment",
                   "spacepoint", "spacepoint"),
    num_heads=8,
    mlp_ratio=4.0,
    voxel_size_cm=voxel_size_cm,
    coord_scale=coord_scale,
    freeze_backbone=True,
    frag_pool_layers=2,
    frag_pool_heads=8,
    frag_pool_max_points=512,
    # Shared (3) → (D) pos-embedding MLP hidden width. None = token_dim (256).
    pos_emb_hidden_dim=None,
    # Origin regression head. When False: per-layer origin MLP is not
    # built, the dynamic pos_emb(prev_origin) feedback into query_pos is
    # skipped, and loss_kwargs["weight_origin"] is forced to 0. Use this
    # to ablate the origin objective (e.g. when downstream tasks only
    # need clustering / class).
    enable_origin_head=False,
    loss_kwargs=dict(
        weight_class=2.0,
        weight_mask=5.0,
        weight_dice=5.0,
        # weight_origin bumped from 1.0 → 3.0 to compensate for switching
        # the origin loss formula from .sum(dim=-1) (sum over 3 axes) to
        # .mean() (per-axis), which by itself reduced the loss magnitude
        # by 3×. With weight 1.0 + .mean(), the v2 100-epoch run regressed
        # origin error from ~100 cm to ~240 cm. 3.0 restores the original
        # effective weight while keeping the per-axis interpretable units.
        weight_origin=3.0,
        weight_aux_mask_voxel=1.0,
        weight_aux_mask_fragment=1.0,
        no_object_weight=0.1,
        num_sample_points=4096,
        deep_supervision=True,
        match_layer="final",
        # Phase B — importance / PointRend negative sampling. Off by default
        # to preserve the v4 baseline. Flip to True to focus the negative
        # half of each per-pair sample on points where the model is uncertain
        # (the over-prediction halo around the GT mask). The first ablation
        # to try when val/mask_iou plateaus.
        use_importance_sampling=True,
        importance_oversample_ratio=3.0,
        importance_ratio=0.75,
    ),
)

# =============================================================================
# Optimizer / scheduler
# =============================================================================
base_lr = 1e-3
optimizer = dict(type="AdamW", lr=base_lr, weight_decay=0.01)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[base_lr],
    pct_start=0.1,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)

# =============================================================================
# Training loop knobs (one-event smoke-test defaults; bump for real runs)
# =============================================================================
save_path = "exp/shower_clustering/run2"
epoch = 1000
eval_epoch = 100
batch_size = 16
batch_size_val  = 8
batch_size_test = 8
num_worker = 16          # 0 to keep stack traces sane during smoke test
num_worker_val = 8
evaluate = True          # flip to True (and uncomment evaluator hook below)
                         # for real training runs with val metrics
clip_grad = 5.0          # mild gradient clipping
enable_amp = True       # AMP roughly halves memory on P100 (xformers fp16).
                         # Watch for instability — see GradScalerMonitor hook
                         # below + the comment block at the end of this file.
amp_dtype = "float16"    # P100 has no bf16 hardware; bf16 falls back to fp32.
enable_wandb = True     # turn on for real runs (ShowerClusteringEvaluator
                         # will then push val/* metrics to wandb)
wandb_project = "pointcept-shower-clustering"
empty_cache = True       # 16 GB P100 — free CUDA cache between fwd / bwd

# =============================================================================
# Hooks: SonataCheckpointLoader loads cfg.weight into the model.backbone with
# the "backbone." key prefix that our wrapper requires.
#
# ShowerClusteringEvaluator runs after each epoch (eval_freq=0) and computes
# the val metrics: matched-pair class accuracy, spacepoint-level mask IoU
# (overall + per class), origin error in cm, matched fraction, plus the
# average per-component val loss. It also picks `mask_iou_mean` as the metric
# the CheckpointSaver tracks for "best so far". The hook is gated by
# cfg.evaluate; leaving it in the list while evaluate=False is harmless.
#
# Comment-out the evaluator entry while doing one-event smoke tests so the
# val_loader build is skipped entirely (we set evaluate=False above for the
# same reason).
# =============================================================================
hooks = [
    dict(type="SonataCheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    # GradScalerMonitor logs the AMP loss-scale + overflow events. Active
    # only when enable_amp=True. Look at val/wandb for:
    #   grad_scaler/scale         — should stabilize after a few iters; if it
    #                                drops to 1.0 and stays there, AMP is
    #                                fighting overflows constantly.
    #   grad_scaler/overflow_event — 1 on overflow, 0 otherwise. A few in
    #                                the first ~50 iters is normal (warmup).
    #                                Sustained > 1% of steps means trouble.
    dict(type="GradScalerMonitor", log_frequency=10, warn_on_low_scale=4.0),
    dict(type="InformationWriter"),
    dict(type="ShowerClusteringEvaluator",
         eval_freq=0,
         best_metric="mask_iou_mean",
         empty_cache=True,
         log_per_event=False),
    dict(type="CheckpointSaver", save_freq=10),
]


# =============================================================================
# AMP-instability watch list
# =============================================================================
# When `enable_amp = True`, watch these signals (in stdout / wandb) — any one
# of them means fall back to fp32:
#
#   1. NaN / Inf in `train/loss` or any component (`train/loss_*`).
#      InformationWriter prints them; PyTorch's GradScaler will SKIP the
#      optimizer step when it detects non-finite gradients, but if the
#      forward itself produces NaN, the step won't be skipped — you'll see
#      the loss go NaN and stay there.
#
#   2. `grad_scaler/scale` collapsing to 1.0 (or warn_on_low_scale) and
#      staying there. Means GradScaler keeps detecting overflows; AMP isn't
#      adding value. Could indicate fp16 saturation in mask_embed @ keys.T
#      or in the dice denominator.
#
#   3. `grad_scaler/overflow_event` firing more than ~1% of steps after
#      warmup. Same diagnosis as (2).
#
#   4. Train loss diverges (rising) while LR is still positive. Could be
#      AMP-induced instability OR genuine optimization issue — disable AMP
#      to disambiguate.
#
#   5. `val/loss_dice` plateauing at a much higher value than fp32 baseline
#      runs. Dice has a +1 epsilon and is mostly fp16-safe, but a sustained
#      gap is a signal something's off.
#
# If you hit any of (1)-(4), set `enable_amp = False`, drop `max_spacepoints`
# to ~80_000 instead, and re-launch — gives you ~the same memory headroom
# without the precision risk.
# =============================================================================

# =============================================================================
# Trainer
# =============================================================================
train = dict(type="ShowerClusteringTrainer")
