"""
LArFormer Stage-2 cascaded slicer — variant that uses the existing trained
SonataLoRADeghostSegmentor as the Stage-1 deghoster.

Same scaffold as `larformer-slicer-v1-cascaded.py`, but `deghoster=` points
at the LoRA-finetuned Sonata deghoster from
`configs/lartpc/lorafinetune-sonata-v1m1-lartpc-v6-deghost.py`. Two changes
make this work:

  1. CascadedSlicer._run_deghoster_p_real already handles both output
     conventions (LArFormer's per-event `predictions` dict and the LoRA
     deghoster's flat `seg_logits` dict).
  2. `deghoster_class_index_real=0` is set on the cascade. The LoRA
     deghoster uses class 0=real, class 1=ghost (via the HasmatchAsGhost
     transform: hasmatch=1→0, hasmatch=0→1). LArFormer's per-token cls
     deghoster uses the opposite convention (class 1=real). The cascade
     just needs to know which column of the softmax is the real-class
     probability.

Usage:
    1. Edit `deghoster_weight` below to point at your trained LoRA
       deghoster checkpoint (e.g. `exp/<run-name>/model/model_best.pth`).
    2. Run:
           python tools/train.py --config $THIS_FILE
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

# Path to the trained SonataLoRADeghostSegmentor checkpoint. Replace with
# your actual run path; defaults are placeholders.
deghoster_weight = "/home/twongjirad/working/larbys/gen2/container_u22/Pointcept/sonata/lora_deghost_v6_hasmatch/model/epoch_30.pth"

# =============================================================================
# Coords + backbone shape
# =============================================================================
coord_center = (125.0, 0.0, 518.0)
coord_scale = 179.55
flash_backend = "xformers"
token_dim = 128
backbone_out_channels = 1232          # Sonata-v1m1 up_cast_level=4 (both stages)

# =============================================================================
# Dataset — LArFormerDataset(gt_source="slice")
# =============================================================================
# NOTE: the LoRA deghoster was trained on a different dataset class
# (LArTPCDataset) with its own augmentation pipeline. For the cascade we
# feed it LArFormerDataset output. The relevant inputs (coord, feat,
# offset, grid_coord) are identical in shape and meaning across the two
# datasets, and the deghoster never sees the dataset-specific augmentation
# transforms (it only reads from data_dict in eval mode). Coord scale is
# 179.55 here vs ~179.44 in the LoRA training — a ~0.06% mismatch that
# is well within the model's robustness margin.
_dataset_common = dict(
    type="LArFormerDataset",
    coord_center=coord_center,
    coord_scale=coord_scale,
    gt_source="slice",
    emit_fragments=False,
    slice_class_map={1: 0, 2: 1},
    merge_nu_slices=True,
    lm_score_aug_low=0.05,
    lm_score_aug_high=0.30,
    lm_score_val_threshold=0.10,
    log_transform_strength=True,
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
# Stage 1 — SonataLoRADeghostSegmentor subconfig
# =============================================================================
# Copy of the model block from
# `configs/lartpc/lorafinetune-sonata-v1m1-lartpc-v6-deghost.py`. Must match
# the trained model architecture exactly (lora_rank, lora_target_modules,
# backbone params including drop_path / enable_flash / mask_token) so the
# checkpoint loads with missing=0 / unexpected=0.
deghoster_cfg = dict(
    type="SonataLoRADeghostSegmentor",
    backbone_out_channels=1232,
    lora_rank=16,
    lora_alpha=32.0,
    lora_dropout=0.05,
    lora_target_modules=["qkv", "proj"],
    freeze_backbone_non_lora=True,
    ghost_class_index=1,             # LoRA model's own convention (unused
                                      # by cascade; cascade uses
                                      # deghoster_class_index_real below)
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
    # criteria is required by SonataLoRADeghostSegmentor.__init__ but never
    # invoked in our cascade (we call the deghoster in eval mode + the
    # dataset doesn't emit `segment`, so the loss path is skipped).
    criteria=[
        dict(type="FocalLoss", gamma=2.0, alpha=0.5,
             loss_weight=1.0, ignore_index=-1, reduction="mean"),
        dict(type="LovaszLoss", mode="multiclass",
             loss_weight=1.0, ignore_index=-1),
    ],
)

# =============================================================================
# Stage 2 — Slicer subconfig (same as larformer-slicer-v1-cascaded.py)
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
    backbone_out_channels=backbone_out_channels,
    levels=levels,
    scale_pattern=scale_pattern,
    token_dim=token_dim,
    num_queries=32,
    num_classes=3,
    freeze_backbone=True,
    enable_origin_head=False,
    decoder_kwargs=dict(num_heads=4, mlp_ratio=4.0),
    loss_kwargs=dict(
        weight_class=2.0,
        weight_mask_primary=5.0,
        weight_dice_primary=5.0,
        weight_aux_mask=1.0,
        weight_per_level_cls=0.5,
        weight_origin=0.0,
        num_sample_points=4096,
        aux_max_tokens=20_000,
        no_object_weight=0.1,
    ),
)

# =============================================================================
# CascadedSlicer
# =============================================================================
# Slicer backbone pretrain. Loaded by CascadedSlicer.__init__ into
# self.slicer.backbone only (NOT through SonataCheckpointLoader — that
# hook would prepend "backbone." and overwrite the deghoster too).
# Replace with your vanilla Sonata pretrain path; the keys should look
# like `{teacher,student}.backbone.*` (the standard Sonata-pretrain
# layout). Set to None to leave the slicer's backbone at random init.
slicer_backbone_weight = (
    #"/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/"
    "/home/twongjirad/working/larbys/gen2/container_u22/Pointcept/"
    "sonata/lartpc_v6_h200_noghosts_pretrain_logspace_resume/model/epoch_42.pth"
)

model = dict(
    type="CascadedSlicer",
    deghoster=deghoster_cfg,
    deghoster_weight=deghoster_weight,
    slicer=slicer_cfg,
    slicer_backbone_weight=slicer_backbone_weight,
    deghost_threshold_min=0.3,
    deghost_threshold_max=0.7,
    deghost_threshold_val=0.5,
    freeze_deghoster=True,
    # KEY: the LoRA deghoster has class 0 = real, class 1 = ghost (from
    # HasmatchAsGhost: hasmatch=1→0 = real, hasmatch=0→1 = ghost).
    # The LArFormer-flavored deghoster (larformer-slicer-v1-cascaded.py)
    # uses class 1 = real instead.
    deghoster_class_index_real=0,
    report_keep_frac=True,
)

# =============================================================================
# Training loop knobs
# =============================================================================
# Top-level `weight` is unused: both submodule checkpoints are loaded by
# CascadedSlicer.__init__ via `deghoster_weight` and `slicer_backbone_weight`
# above. For RESUME of a cascaded run, swap SonataCheckpointLoader for the
# generic `CheckpointLoader` (which does no prefix munging) and set
# `weight = "exp/.../model/model_last.pth"` + `resume = True`.
weight = None

save_path        = "exp/larformer_slicer_v1_cascaded_loradeghost"
epoch            = 200
eval_epoch       = 200
batch_size       = 2
batch_size_val   = 1
num_worker       = 2
num_worker_val   = 1
evaluate         = True          # LArFormerSlicerEvaluator runs after each epoch
enable_amp       = False
empty_cache      = True
enable_wandb     = True
wandb_project    = "pointcept-larformer"
find_unused_parameters = True

base_lr = 5e-5
param_dicts = None

optimizer = dict(type="AdamW", lr=base_lr, weight_decay=0.01)
scheduler = dict(
    type="CosineAnnealingLR",
    eta_min=base_lr * 0.01,
)

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
