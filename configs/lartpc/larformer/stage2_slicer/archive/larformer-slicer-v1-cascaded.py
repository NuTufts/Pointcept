"""
LArFormer Stage-2 cascaded slicer (v1).

Wraps a frozen Stage-1 deghoster + a Stage-2 slicer into a single
`CascadedSlicer`. Forward chain:

    raw spacepoints
        → deghoster (frozen LArFormer with per-token cls on hasmatch)
        → per-SP P(real); filter at threshold τ
        → slicer (LArFormer with Mask2Former scale_pattern over voxels +
                  spacepoints; gt_source="slice")

τ is sampled per-batch (your design call):
    train:  τ ~ U(deghost_threshold_min, deghost_threshold_max)
    val:    τ = deghost_threshold_val

The slicer's backbone is the vanilla pretrained Sonata (independent of the
deghoster's LoRA-folded backbone — two backbones in v0). See
`docs/LArFormer.md` §6 + §11 for the "deghost-decoder" alternative that
would let us share one backbone.

The scale_pattern below uses (voxel_20cm, voxel_10cm, voxel_5cm, spacepoint).
You've flagged that this may be too coarse — flipping to (voxel_10cm,
voxel_5cm, voxel_2cm, spacepoint) is a config-only edit; the model-side
voxel builders re-compute the grid from coord_norm at every forward.
"""

_base_ = ["../../../../_base_/default_runtime.py"]

# Side-effect: register LArFormerTrainer + LArFormerSlicerEvaluator.
# See larformer-deghost-v0.py for why we use the `from X.Y import Z as _name`
# form instead of `import X.Y.Z`.
from pointcept.models.LArFormer import trainer as _larformer_trainer_module
from pointcept.models.LArFormer import evaluator as _larformer_evaluator_module
del _larformer_trainer_module
del _larformer_evaluator_module

# =============================================================================
# Paths
# =============================================================================
_DEFAULT_LIST = "/home/twongjirad/working/larbys/gen2/container_u22/Pointcept/devdata_mergedh5_pi0filter_10files.txt"

# Path to the trained Stage-1 deghoster checkpoint. The
# CascadedSlicer._load_deghoster_weight strips the SonataCheckpointLoader-
# style "backbone." prefix automatically; works on either prefix layout.
deghoster_weight = "exp/larformer_deghost_v0/model/model_last.pth"

# Vanilla Sonata pretrain (for the slicer's backbone). Same checkpoint the
# legacy shower-clustering config uses.
slicer_backbone_weight = (
    "/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/"
    "sonata/lartpc_v6_h200_noghosts_pretrain_logspace_resume/model/epoch_42.pth"
)

# =============================================================================
# Coords + backbone shape
# =============================================================================
coord_center = (125.0, 0.0, 518.0)
coord_scale = 179.55
flash_backend = "xformers"
token_dim = 256                       # bumped from 128 for slicer capacity
backbone_out_channels = 1232          # Sonata-v1m1 up_cast_level=4

# =============================================================================
# Dataset — LArFormerDataset(gt_source="slice")
# =============================================================================
_dataset_common = dict(
    type="LArFormerDataset",
    coord_center=coord_center,
    coord_scale=coord_scale,
    gt_source="slice",
    emit_fragments=False,             # slicer doesn't use the fragment level
    slice_class_map={1: 0, 2: 1},      # nu→0, cosmic→1; no_object=2
    merge_nu_slices=True,
    # No lm_score pre-filter: the cascade's deghoster (Stage 1) is the
    # sole ghost discriminator. Doubling up (LArMatch pre-filter +
    # deghoster) feeds the deghoster a non-representative input subset
    # that depresses its measured recall (verified on the dev sample:
    # τ=0.5 recall jumps from 0.13 with lm_threshold=0.10 to 0.65 with
    # lm_threshold=0.0, matching the inference-script measurement of
    # 0.71 across 100 events).
    lm_score_aug_low=0.0,
    lm_score_aug_high=0.0,
    lm_score_val_threshold=0.0,
    wire_scale=1.0 / 3456.0,
    min_fragment_points_post_filter=50,
)

data = dict(
    num_classes=3,                   # nu, cosmic, no_object
    ignore_index=-1,
    names=["nu", "cosmic", "no_object"],
    train=dict(
        split="train", data_root="/", data_list_file=_DEFAULT_LIST, loop=1,
        max_spacepoints=100_000, **_dataset_common,
    ),
    val=dict(
        split="val", data_root="/", data_list_file=_DEFAULT_LIST, loop=1,
        max_spacepoints=150_000, **_dataset_common,
    ),
    test=dict(
        split="test", data_root="/", data_list_file=_DEFAULT_LIST, loop=1,
        max_spacepoints=None, **_dataset_common,
    ),
)

# =============================================================================
# Shared Sonata-v1m1 backbone block (used by BOTH deghoster + slicer in v0;
# they're independent instances with separate weights — two passes).
# =============================================================================
def _sonata_v1m1_backbone():
    return dict(
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
    )

# =============================================================================
# Stage 1 — Deghoster subconfig (frozen; weight loaded by CascadedSlicer)
# =============================================================================
# Mirrors larformer-deghost-v0.py's model block: num_queries=0 (no decoder),
# single spacepoint level with a 2-class cls head on `hasmatch`.
deghoster_cfg = dict(
    type="LArFormer",
    backbone=_sonata_v1m1_backbone(),
    backbone_out_channels=backbone_out_channels,
    levels=[
        dict(
            name="spacepoint",
            builder="SpacepointBuilder",
            supervision=dict(
                cls=dict(num_classes=2, label_src="hasmatch",
                         reduce="amax", weight=1.0, loss="ce",
                         hidden_dim=256, ignore_index=-1),
            ),
        ),
    ],
    scale_pattern=[],
    token_dim=256,                    # match deghoster training config
    num_queries=0,
    num_classes=2,
    freeze_backbone=True,
    enable_origin_head=False,
    loss_kwargs=dict(weight_per_level_cls=1.0),
)

# =============================================================================
# Stage 2 — Slicer subconfig
# =============================================================================
levels = [
    dict(name="voxel_20cm",
         builder="VoxelBuilder",
         builder_cfg=dict(voxel_size_cm=20.0, coord_scale=coord_scale),
         supervision=dict(mask=dict(weight=1.0, mode="aux"))),
    dict(name="voxel_10cm",
         builder="VoxelBuilder",
         builder_cfg=dict(voxel_size_cm=10.0, coord_scale=coord_scale),
         supervision=dict(
             mask=dict(weight=1.0, mode="aux"),
             # Same priority-pool idiom as larformer-slicer-v0.py:
             # raw origin_label = {0=ghost, 1=nu, 2=cosmic}; remap so
             # amax-reduction acts as "any nu wins, else any cosmic, else ghost".
             cls=dict(num_classes=3, label_src="origin_label",
                      label_remap={0: 0, 1: 2, 2: 1}, reduce="amax",
                      weight=0.5, loss="ce", ignore_index=-1),
         )),
    dict(name="voxel_5cm",
         builder="VoxelBuilder",
         builder_cfg=dict(voxel_size_cm=5.0, coord_scale=coord_scale),
         supervision=dict(mask=dict(weight=1.0, mode="aux"))),
    dict(name="spacepoint",
         builder="SpacepointBuilder",
         supervision=dict(mask=dict(weight=5.0, mode="primary"))),
]
scale_pattern = [
    "voxel_20cm", "voxel_10cm", "voxel_10cm",
    "voxel_5cm",  "spacepoint", "spacepoint",
]

slicer_cfg = dict(
    type="LArFormer",
    backbone=_sonata_v1m1_backbone(),
    backbone_out_channels=backbone_out_channels,
    levels=levels,
    scale_pattern=scale_pattern,
    token_dim=token_dim,
    num_queries=64,                   # bumped from 32: Hungarian needs slack
                                       # vs the ~30 GT slices per event.
    num_classes=3,
    freeze_backbone=True,
    enable_origin_head=False,         # slicer doesn't need vertex regression
    decoder_kwargs=dict(num_heads=4, mlp_ratio=4.0),
    loss_kwargs=dict(
        weight_class=2.0,
        weight_mask_primary=5.0,
        weight_dice_primary=5.0,
        weight_aux_mask=0.3,           # dropped from 1.0: aux at 3 levels
                                        # was diluting the primary signal.
        weight_per_level_cls=0.5,
        weight_origin=0.0,
        num_sample_points=8192,        # bumped from 4096 — slicer sees ~3x
                                        # more SPs after dropping lm_score
                                        # pre-filter; sample budget needs
                                        # to scale or per-pair BCE undersamples.
        # PointRend-style hard-negative mining (ported from shower_clustering)
        # to bias negatives toward the over-prediction halo region. Cheap,
        # measurable uplift on the same set-prediction loss shape.
        use_importance_sampling=True,
        importance_oversample_ratio=3.0,
        importance_ratio=0.75,
        aux_max_tokens=20_000,
        no_object_weight=0.1,
    ),
)

# =============================================================================
# Top-level model — CascadedSlicer wraps both stages
# =============================================================================
model = dict(
    type="CascadedSlicer",
    deghoster=deghoster_cfg,
    deghoster_weight=deghoster_weight,
    slicer=slicer_cfg,
    # Loaded into self.slicer.backbone only — see CascadedSlicer
    # docstring + cascaded-loradeghost.py for the rationale.
    slicer_backbone_weight=slicer_backbone_weight,
    deghost_threshold_min=0.3,
    deghost_threshold_max=0.7,
    deghost_threshold_val=0.5,
    freeze_deghoster=True,
    deghoster_class_index_real=1,    # hasmatch=1 = real in P5 training
    report_keep_frac=True,
)

# =============================================================================
# Training loop knobs
# =============================================================================
# Top-level `weight` is unused — both submodule checkpoints are loaded
# inside CascadedSlicer.__init__. For RESUME, use the generic
# `CheckpointLoader` hook (not SonataCheckpointLoader) and set
# `weight = "exp/.../model/model_last.pth"` + `resume = True`.
weight = None

save_path        = "exp/larformer_slicer_v1_cascaded"
epoch            = 200
eval_epoch       = 200            # eval cycle = full training run
batch_size       = 2              # smaller than deghost v0 — slicer adds
                                  # decoder cost on top of 2 backbone passes
batch_size_val   = 1
num_worker       = 2
num_worker_val   = 1
evaluate         = True           # LArFormerSlicerEvaluator runs after each epoch
enable_amp       = False
empty_cache      = True
enable_wandb     = True
wandb_project    = "pointcept-larformer"
find_unused_parameters = True

# =============================================================================
# Optimizer & Scheduler
# =============================================================================
base_lr = 5e-4
param_dicts = None    # single param group: slicer's cls + tokenizer proj +
                       # decoder are all trainable; deghoster is frozen.

optimizer = dict(type="AdamW", lr=base_lr, weight_decay=0.01)

# Same caveat as the deghost config: CosineAnnealingLR is forgiving with
# small total_steps (dev sample is ~5 batches/epoch). Swap to OneCycleLR
# for a production-scale dataset.
scheduler = dict(
    type="CosineAnnealingLR",
    eta_min=base_lr * 0.01,
)

# =============================================================================
# Hooks
# =============================================================================
hooks = [
    # No CheckpointLoader hook: both deghoster + slicer-backbone weights
    # are loaded inside CascadedSlicer.__init__ (see deghoster_weight +
    # slicer_backbone_weight on the model config). Add `dict(type=
    # "CheckpointLoader")` here (NOT SonataCheckpointLoader) only when
    # resuming a previous cascaded-training run.
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="LArFormerSlicerEvaluator",
         eval_freq=0,                # after each epoch
         best_metric="nu_mIoU",       # matched-pair IoU on the nu class
         class_names=["nu", "cosmic"],
         nu_class_id=0,
         empty_cache=True,
         log_per_event=False),
    dict(type="CheckpointSaver", save_freq=None),
]

train = dict(type="LArFormerTrainer")
