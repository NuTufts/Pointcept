"""
LArTPC Semantic Segmentation with PointTransformerV3 - No Voxelization

Same as base config but WITHOUT GridSample voxelization.
Instead, grid_size is passed directly for serialization (space-filling curve ordering).

This preserves all original points and their spatial relationships.
May require more GPU memory and be slower, but maintains full resolution.

Trade-offs vs voxelized version:
  - More points (~285k vs ~72k)
  - Higher memory usage
  - Slower training
  - No spatial regularization from voxel merging
  - Preserves all detail
"""

_base_ = ["../../_base_/default_runtime.py"]

# misc custom setting
batch_size = 1  # bs: total bs in all gpus
num_worker = 1
mix_prob = 0.0  # disable mixing for now
empty_cache = False
enable_amp = True
enable_wandb = True

# model settings
model = dict(
    type="DefaultSegmentorV2",
    num_classes=5,  # electron, muon, pion, proton, gamma (unknown PIDs are ignored)
    backbone_out_channels=32,  # Must match first value in dec_channels
    backbone=dict(
        type="PT-v3m1",
        in_channels=6,  # strength(3) + wire_coords(3)
        order=("z", "z-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(16, 32, 64, 128, 256),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(32, 32, 64, 64),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        enc_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("LArTPC",),  # Single dataset condition
    ),
    # Focal Loss to handle class imbalance
    # Classes: 0-electron, 1-muon, 2-pion, 3-proton, 4-gamma
    criteria=[
        dict(
            type="FocalLoss",
            gamma=2.0,
            alpha=[0.6, 0.15, 0.9, 0.7, 0.7],  # electron, muon, pion, proton, gamma
            loss_weight=1.0,
            ignore_index=-1,
        ),
    ],
)

# scheduler settings
epoch = 1000
optimizer = dict(type="AdamW", lr=0.06, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.06, 0.006],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=100.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=0.006)]

# dataset settings
dataset_type = "LArTPCDataset"
data_root = "data/lartpc"

# Grid size for serialization (space-filling curve ordering)
# This does NOT reduce points - just used for computing point ordering
serialization_grid_size = 0.0002

data = dict(
    num_classes=5,
    ignore_index=-1,
    names=[
        "electron",
        "muon",
        "pion",
        "proton",
        "gamma",
    ],
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        use_reco_coords=True,
        use_edep_as_strength=True,
        label_mode="pid",
        coord_scale=0.001,
        log_transform_edep=True,
        transform=[
            dict(type="RandomScale", scale=[0.9, 1.1]),
            # No GridSample - just set grid_size for serialization
            # The model will compute grid_coord on-the-fly from coord + grid_size
            dict(type="Copy", keys_dict={"grid_size": serialization_grid_size}),
            # Limit points per sample for memory management
            # May need to reduce this if running out of memory
            dict(type="SphereCrop", point_max=102400, mode="random"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "segment"),  # No grid_coord - computed during serialization
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
        coord_scale=0.001,
        log_transform_edep=True,
        transform=[
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(type="Copy", keys_dict={"grid_size": serialization_grid_size}),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "segment", "origin_segment"),
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
        coord_scale=0.001,
        log_transform_edep=True,
        transform=[
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(type="Copy", keys_dict={"grid_size": serialization_grid_size}),
        ],
        test_mode=True,
        test_cfg=dict(
            # No voxelization - just pass grid_size for serialization
            voxelize=dict(
                type="Copy",
                keys_dict={"grid_size": serialization_grid_size},
            ),
            crop=None,
            post_transform=[
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "index"),
                    feat_keys=("strength", "color"),
                ),
            ],
            aug_transform=[],  # No test-time augmentation
        ),
    ),
)
