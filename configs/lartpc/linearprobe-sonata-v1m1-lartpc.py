"""
Linear Probing Evaluation for SONATA Pretrained Backbone

Evaluates the quality of self-supervised pretrained features by training
only a linear classifier on frozen backbone features.

This is the standard evaluation protocol for self-supervised learning:
1. Load pretrained encoder weights
2. Freeze all encoder parameters (no gradient updates)
3. Train only a linear classification head
4. Evaluate semantic segmentation performance

Usage:
    python tools/train.py --config-file configs/lartpc/linearprobe-sonata-v1m1-lartpc.py \
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
    ((0.0,-117.0, 0.0), (0.0, 0.866, 0.5)),
    ((0.0, 117.0, 0.0), (0.0,-0.866, 0.5)),
    ((0.0,   0.0, 0.0), (0.0, 0.000, 1.0))
]

_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 1
num_worker = 1
mix_prob = 0.0
empty_cache = False
enable_amp = True
enable_wandb = True
wandb_project = "pointcept"
save_path = "sonata/linearprobe"

# Grid size must match pretraining
grid_size = 0.0002

# model settings - Linear probe on frozen encoder
model = dict(
    type="DefaultSegmentorV2",
    num_classes=6,  # electron, muon, pion, proton, gamma, ghost
    backbone_out_channels=16,  # First value of dec_channels (decoder output)
    backbone=dict(
        type="PT-v3m2",
        in_channels=6,
        order=("z", "z-trans"),
        stride=(2, 2, 2, 2),
        # Encoder architecture - MUST match pretrained model
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(16, 32, 64, 128, 256),  # Halved model
        enc_num_head=(2, 2, 4, 8, 16),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        # No decoder - use encoder features directly
        dec_depths=(2, 2, 2, 2),
        dec_channels=(16, 32, 64, 128),
        dec_num_head=(2, 2, 4, 8),
        dec_patch_size=(1024, 1024, 1024, 1024),
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
        traceable=False,
        enc_mode=False,  # Need decoder to upsample features back to point resolution
        mask_token=False,
    ),
    criteria=[
        dict(
            type="FocalLoss",
            gamma=2.0,
            alpha=0.5,
            loss_weight=1.0,
            ignore_index=-1,
            reduction='sum',
        ),
    ],
    # IMPORTANT: Freeze backbone to only train the linear head
    freeze_backbone=True,
)

# scheduler settings - faster training since only linear layer
epoch = 100  # Fewer epochs needed for linear probe
eval_epoch = 10
optimizer = dict(type="AdamW", lr=0.01, weight_decay=0.0)  # Higher LR, no weight decay for linear
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.01],
    pct_start=0.1,
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
        split="train",
        data_root=data_root,
        use_reco_coords=True,
        use_edep_as_strength=True,
        label_mode="pid",
        coord_scale=0.001,
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
            dict(type="SphereCrop", point_max=102400, mode="random"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "segment_weights"),
                feat_keys=("strength", "color"),
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        use_reco_coords=True,
        use_edep_as_strength=True,
        label_mode="pid",
        include_ghosts=True,
        exclude_other=True,
        coord_scale=0.001,
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
                keys=("coord", "grid_coord", "segment", "origin_segment", "inverse", "segment_weights"),
                feat_keys=("strength", "color"),
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        use_reco_coords=True,
        use_edep_as_strength=True,
        label_mode="pid",
        include_ghosts=True,
        exclude_other=True,
        coord_scale=0.001,
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
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]
