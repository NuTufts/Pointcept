"""LoRA Deghoster — SINGLE-EVENT OVERFIT TEST.

Goal: verify the deghoster pipeline (data → model → loss → evaluator) is
correct end-to-end by overfitting to a single dev event. After ~50-100
epochs the model should reach near-perfect mIoU on this one event. If it
doesn't, something structural in the pipeline is broken.

Differences from the production config:
  - Train and val both point at the same single-event filelist
    (Pointcept/lartpc_data_prep/deghost_analysis/overfit_one.txt).
  - batch_size = batch_size_val = 1; no DDP scaling.
  - No random augmentations (RandomRotate/Flip/Jitter, Multiplicative
    strength jitter) so the same input is presented every epoch.
  - No BiasedSphereCrop on train or val (point_max set huge so the full
    event passes through). The whole event is the training example.
  - drop_cosmics=False (correct for deghosting).
  - HasmatchAsGhost transform sets the segment label directly from the
    producer's hasmatch field — no dependence on the dataset's
    SSNETLABEL_TO_CLASS chain.
  - FocalLoss alpha=0.5 (neutral; LovaszLoss handles imbalance).
  - epoch=200, eval_epoch=10 — small run, eval periodically.
  - WandB disabled (no need to log a debug run).

Once this overfits, the same transform list and loss should be safe to
use at scale with the production filelist + augmentations re-enabled.
"""

wire_projections = None

_base_ = ["../../../_base_/default_runtime.py"]

# ============================================================================
# Hyper-parameters
# ============================================================================
find_unused_parameters = True
batch_size       = 1
batch_size_val   = 1
num_worker       = 4
num_worker_val   = 4
mix_prob         = 0.0
empty_cache      = False
enable_amp       = False
enable_wandb     = True
wandb_project    = "pointcept-lora-deghost "
save_path        = "sonata/lora_deghost_overfit_one"
epoch            = 200
eval_epoch       = 10
base_lr          = 5e-4
lora_lr          = 2e-4
head_lr          = base_lr

flash_backend = 'xformers'
amp_dtype     = "float16"

# Grid size and coordinate normalization (same as production)
grid_size    = 0.25
coord_scale  = 1036.0 * 3**0.5 / 2.0 / 5.0

# Sphere crop: keep the transform in the pipeline but turn off the cropping
# behavior by setting point_max to "always larger than any real event".
# (BiasedSphereCrop returns the data unchanged when n_points <= point_max.)
_NO_CROP_POINT_MAX = 100_000_000
_NO_CROP_POINT_MIN = 1
biased_spherecrop_radius = 20.0

TRAIN_FILE_LIST = (
    "/home/twongjirad/working/larbys/gen2/container_u22/Pointcept/"
    "lartpc_data_prep/deghost_analysis/overfit_one.txt"
)
VAL_FILE_LIST = TRAIN_FILE_LIST   # same single event for train & val

pretrain_model_path = (
    "sonata/lartpc_v6_h200_noghosts_pretrain_logspace_resume/model/epoch_42.pth"
)

# ============================================================================
# Model (identical to production)
# ============================================================================
model = dict(
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
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            drop_path=0.3,
            shuffle_orders=True,
            pre_norm=True,
            enable_rpe=False,
            enable_flash=False,
            flash_backend=flash_backend,
            upcast_attention=False,
            upcast_softmax=False,
            traceable=True,
            enc_mode=True,
            mask_token=True,
        ),
        head_in_channels=1088,
        head_hidden_channels=2048,
        head_embed_channels=256,
        head_num_prototypes=4096,
        num_global_view=2,
        num_local_view=6,
        up_cast_level=4,
    ),
    # Focal (neutral alpha) + Lovász. LovaszLoss does the class-balance work.
    criteria=[
        dict(
            type="FocalLoss",
            gamma=2.0,
            alpha=0.5,
            loss_weight=1.0,
            ignore_index=-1,
            reduction='mean',
        ),
        dict(
            type="LovaszLoss",
            mode="multiclass",
            loss_weight=1.0,
            ignore_index=-1,
        ),
    ],
)

# ============================================================================
# Optimizer & Scheduler
# ============================================================================
optimizer = dict(
    type="AdamW",
    lr=base_lr,
    weight_decay=0.01,
)

param_dicts = [
    dict(keyword="lora_", lr=lora_lr, weight_decay=0.01),
    dict(keyword="deghost_head", lr=head_lr, weight_decay=0.0),
]

# Flat (constant) LR for the overfit test. FlatWithDecayLR is a no-op on
# per-iteration step() and only decays via the per-epoch hook — with
# step_period_epochs huge and gamma=1.0, no decay ever fires, so the LR
# stays at base_lr the whole run. (OneCycleLR was annealing to ~0 by the
# end of epoch=200, freezing the model mid-fit at mIoU ≈ 0.61.)
scheduler = dict(
    type="FlatWithDecayLR",
    mode="epoch",
    gamma=1.0,
    min_lr=base_lr,
    step_period_epochs=10_000,
)

# ============================================================================
# Dataset
# ============================================================================
dataset_type = "LArTPCDataset"
data_root    = "data/lartpc"

# Shared transform: deterministic, full-event, hasmatch-as-target.
_shared_transform = [
    # Keep BiasedSphereCrop in the pipeline but neutralize it so the full
    # event survives. Useful so we exercise the same code path the
    # production train uses; can be deleted entirely if preferred.
    dict(
        type="BiasedSphereCrop",
        anchor_points_key="nu_vertices",
        anchor_pdf_key=None,
        radius=biased_spherecrop_radius,
        point_max=_NO_CROP_POINT_MAX,
        point_min=_NO_CROP_POINT_MIN,
        prob_random=0.0,
        max_retries=1,
        fallback_to_random=True,
    ),
    dict(
        type="GridSample",
        grid_size=grid_size,
        hash_type="fnv",
        mode="train",
        return_grid_coord=True,
    ),
    dict(
        type="NormalizeCoord",
        center=[125.0, 0.0, 518.0],
        scale=coord_scale,
    ),
    dict(
        type="LogTransform",
        min_val=0.01,
        max_val=1000.0,
        log=True,
        keys=("strength",),
    ),
    # Direct ghost target from producer-side hasmatch. No reliance on
    # SSNETLABEL_TO_CLASS or RemapGhostLabel.
    dict(
        type="HasmatchAsGhost",
        real_target_index=0,
        ghost_target_index=1,
        ignore_index=-1,
    ),
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=("coord", "grid_coord", "segment", "segment_counts"),
        feat_keys=("coord", "strength"),
    ),
]

data = dict(
    num_classes=2,
    ignore_index=-1,
    names=["real", "ghost"],
    train=dict(
        type=dataset_type,
        data_list_file=TRAIN_FILE_LIST,
        use_reco_coords=True,
        use_edep_as_strength=True,
        label_mode="ssnet",
        coord_scale=1.0,
        include_ghosts=True,
        true_points_only=False,
        exclude_other=True,
        drop_cosmics=False,         # ← critical correction
        drop_cosmics_prob=0.0,
        transform=_shared_transform,
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_list_file=VAL_FILE_LIST,
        use_reco_coords=True,
        use_edep_as_strength=True,
        label_mode="ssnet",
        include_ghosts=True,
        true_points_only=False,
        coord_scale=1.0,
        exclude_other=True,
        drop_cosmics=False,
        drop_cosmics_prob=0.0,
        transform=_shared_transform,
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        data_list_file=VAL_FILE_LIST,
        use_reco_coords=True,
        use_edep_as_strength=True,
        label_mode="ssnet",
        include_ghosts=True,
        true_points_only=False,
        coord_scale=1.0,
        exclude_other=True,
        drop_cosmics=False,
        drop_cosmics_prob=0.0,
        transform=[],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
            ),
            crop=None,
            post_transform=[
                dict(
                    type="NormalizeCoord",
                    center=[125.0, 0.0, 518.0],
                    scale=coord_scale,
                ),
                dict(
                    type="LogTransform",
                    min_val=0.01,
                    max_val=1000.0,
                    log=True,
                    keys=("strength",),
                ),
                dict(
                    type="HasmatchAsGhost",
                    real_target_index=0,
                    ghost_target_index=1,
                    ignore_index=-1,
                ),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index"),
                    feat_keys=("coord", "strength"),
                ),
            ],
            aug_transform=[],
        ),
    ),
)

# ============================================================================
# Hooks
# ============================================================================
hooks = [
    dict(
        type="LoRASonataCheckpointLoader",
        pretrained_path=pretrain_model_path,
    ),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator", write_cls_iou=True),
    dict(type="CheckpointSaver", save_freq=10),
]
