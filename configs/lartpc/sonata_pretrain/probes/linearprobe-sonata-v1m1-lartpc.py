"""
Linear Probing Evaluation for SONATA Pretrained Backbone

Evaluates the quality of self-supervised pretrained features by training
only a linear classifier on frozen backbone features.

This is the standard evaluation protocol for self-supervised learning:
1. Load pretrained SONATA model (full model with student/teacher)
2. Freeze all backbone parameters (no gradient updates)
3. Train only a linear classification head
4. Evaluate semantic segmentation performance

The SonataSegmentor wraps the full SONATA model and uses its up_cast mechanism
to get point-resolution features. This avoids loading an untrained decoder.

Usage:
    python tools/train.py --config-file configs/lartpc/sonata_pretrain/probes/linearprobe-sonata-v1m1-lartpc.py \
        --options weight=exp/sonata_pretrain/model/model_best.pth

Data reuse note:
    It's acceptable to use the same data used for pretraining, since pretraining
    was unsupervised (no labels). The linear probe tests whether the learned
    features capture label-relevant structure.
"""

# =============================================================================
# Wire plane projection parameters (must match pretraining)
# =============================================================================
wire_projections = [
    ((0.0,    0.0,  -338.6334821387676), (0.0, -0.866, 0.5)),
    ((0.0,    0.0,  -333.0331845276306), (0.0,  0.866, 0.5)),
    ((0.0,    0.0,  0.33), (0.0, 0.0, 1.0))
]

_base_ = ["../../../_base_/default_runtime.py"]

# misc custom setting
batch_size = 16
num_worker = 4
mix_prob = 0.0
empty_cache = False
enable_amp = True
enable_wandb = True
wandb_project = "pointcept"
save_path = "sonata/linearprobe"
epoch=100
eval_epoch=100
base_lr=0.01

# Grid size must match pretraining
grid_size = 0.25

max_points_spherecrop=98304
min_points_spherecrop=20480

# Small dataset test
TRAIN_FILE_LIST="pi0_train_files.txt"        # 1880 events
VAL_FILE_LIST="pi0_test_files_100events.txt" # 100 events

# =============================================================================
# Model settings - Linear probe using SonataSegmentor with nested SONATA backbone
#
# The SONATA model's up_cast_level determines feature resolution:
#   up_cast_level=2: Features at 4x downsampled resolution (stride 2^2=4)
#   up_cast_level=4: Features at original point resolution (all strides undone)
#
# backbone_out_channels calculation (features concatenate during upcast):
# With enc_channels=(16, 32, 64, 128, 256) and stride=(2,2,2,2):
#   Level 4 (deepest): 256 channels at 16x downsampled
#   After 1 upcast: 128 + 256 = 384 channels at 8x downsampled
#   After 2 upcasts: 64 + 384 = 448 channels at 4x downsampled (pretraining default)
#   After 3 upcasts: 32 + 448 = 480 channels at 2x downsampled
#   After 4 upcasts: 16 + 480 = 496 channels at ORIGINAL resolution
#
# For semantic segmentation, we want features at original point resolution,
# so we use up_cast_level=4 with backbone_out_channels=496.
# =============================================================================
model = dict(
    type="SonataSegmentor",
    num_classes=6,  # electron, muon, pion, proton, gamma, ghost
    backbone_out_channels=496,  # 16 + 32 + 64 + 128 + 256 for up_cast_level=4
    backbone=dict(
        type="Sonata-v1m1",
        # PT-v3m2 backbone config - MUST match pretrained model exactly
        backbone=dict(
            type="PT-v3m2",
            in_channels=6,
            order=("z", "z-trans"),
            stride=(2, 2, 2, 2),
            enc_depths=(2, 2, 2, 6, 2),
            enc_channels=(16, 32, 64, 128, 256),  # Halved model
            enc_num_head=(2, 2, 4, 8, 16),
            enc_patch_size=(1024, 1024, 1024, 1024, 1024),
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            drop_path=0.0,  # No dropout for frozen features
            shuffle_orders=True,
            pre_norm=True,
            enable_rpe=False,
            enable_flash=True,
            upcast_attention=False,
            upcast_softmax=False,
            traceable=True,
            enc_mode=True,  # Encoder-only (matches pretraining)
            mask_token=True,  # Must match pretraining
        ),
        # SONATA head config - must match pretraining
        head_in_channels=448,
        head_hidden_channels=1024,
        head_embed_channels=128,
        head_num_prototypes=1024,
        # View config (not used for inference, but needed for model init)
        num_global_view=2,
        num_local_view=4,
        # Masking config (not used for inference)
        mask_size_start=1.0,
        mask_size_base=5.0,
        mask_ratio_start=0.3,
        mask_ratio_base=0.7,
        mask_jitter=0.3,
        # Temperature (not used for inference)
        teacher_temp_start=0.04,
        teacher_temp_base=0.07,
        student_temp=0.1,
        # Loss weights (not used for inference)
        mask_loss_weight=2/8,
        roll_mask_loss_weight=2/8,
        unmask_loss_weight=4/8,
        # EMA (not used for inference)
        momentum_base=0.994,
        momentum_final=1.0,
        # Matching (not used for inference)
        match_max_k=8,
        match_max_r=5.0,
        # IMPORTANT: up_cast_level determines feature resolution
        # 2 = features at 4x downsampled (pretraining default), 4 = original resolution
        # For segmentation, use 4 to get features at the same resolution as labels
        up_cast_level=4,
    ),
    criteria=[
        dict(
            type="FocalLoss",
            gamma=2.0,
            alpha=0.5,
            loss_weight=1.0,
            ignore_index=-1,
            reduction='mean',  # Use sum when using weights from per-batch class counts
        ),
    ],
    # IMPORTANT: Freeze backbone to only train the linear head
    freeze_backbone=True,
    # Initialize linear head bias with log-prior for faster convergence
    # Class order: electron=0, muon=1, pion=2, proton=3, gamma=4, ghost=5
    # Approximate class frequencies from data (adjust based on actual statistics):
    #   electron: ~4%, muon: ~39%, pion: ~0.064%, proton: ~0.5%, gamma: ~0%, ghost: ~70%
    class_priors=[0.03871, 0.30867, 0.00064, 0.00269, 0.00369, 0.64566],
)

# scheduler settings - faster training since only linear layer
optimizer = dict(type="AdamW", lr=base_lr, weight_decay=0.0)  # Higher LR, no weight decay for linear
scheduler = dict(
    type="OneCycleLR",
    max_lr=[base_lr],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)
# No param_dicts needed - backbone is frozen, only head trains

# dataset settings
dataset_type = "LArTPCDataset"
data_root = "data/lartpc"

data = dict(
    num_classes=6,
    ignore_index=-1,
    names=[
        "electron",
        "muon",
        "pion",
        "proton",
        "gamma",
        "ghost",
    ],
    train=dict(
        type=dataset_type,
        data_list_file=TRAIN_FILE_LIST,
        use_reco_coords=True,
        use_edep_as_strength=True,
        label_mode="pid",
        coord_scale=1.0,
        wire_scale=1.0/3456.0,  # Must match pretraining
        include_ghosts=True,
        exclude_other=True,
        log_transform_edep=True,
        transform=[
            # Minimal augmentation for linear probe - just test feature quality
            # Can add augmentations matching pretraining if desired
            dict(type="RandomScale", scale=[0.95, 1.05]),
            dict(type="RandomFlipAxis", p=0.5, axis="y", center="mean",
                 wire_projections=wire_projections),
            dict(type="RandomFlipAxis", p=0.5, axis="z", center="mean",
                 swap_strength_columns=(0, 1), wire_projections=wire_projections),
            # Grid sampling - must match pretraining
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="SphereCrop", point_max=max_points_spherecrop, point_min=min_points_spherecrop, mode="random"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "segment_counts"),
                feat_keys=("strength", "color"),
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_list_file=VAL_FILE_LIST,
        use_reco_coords=True,
        use_edep_as_strength=True,
        label_mode="pid",
        include_ghosts=True,
        exclude_other=True,
        coord_scale=1.0,
        wire_scale=1.0/3456.0,  # Must match pretraining
        log_transform_edep=True,
        transform=[
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "origin_segment", "inverse", "segment_counts"),
                feat_keys=("strength", "color"),
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        data_list_file=VAL_FILE_LIST,
        use_reco_coords=True,
        use_edep_as_strength=True,
        label_mode="pid",
        include_ghosts=True,
        exclude_other=True,
        coord_scale=1.0,
        wire_scale=1.0/3456.0,  # Must match pretraining
        log_transform_edep=True,
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
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index"),
                    feat_keys=("strength", "color"),
                ),
            ],
            aug_transform=[],
        ),
    ),
)

hooks = [
    # Use SonataCheckpointLoader to properly remap SONATA checkpoint keys
    # It prepends "backbone." to all keys (student.* -> backbone.student.*, teacher.* -> backbone.teacher.*)
    dict(type="SonataCheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator", write_cls_iou=True),  # Log per-class IoU to wandb
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]
