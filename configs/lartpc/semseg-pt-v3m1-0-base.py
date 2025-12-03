"""
LArTPC Semantic Segmentation with PointTransformerV3

Base configuration for training particle-type classification on LArTPC point clouds.
Uses HDF5 data from SimChTripletLabelMaker.

Features used:
    - coord: 3D spatial coordinates (reconstructed positions)
    - strength: Energy deposition (log-transformed)
    - color: Wire coordinates (u, v, y planes) as 3-channel feature

Total input channels: 6 (color: 3 + normal placeholder: 3, or strength: 1 + wire: 3 = 4)
Adjust in_channels in model config based on feat_keys in Collect transform.
"""

_base_ = ["../_base_/default_runtime.py"]

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
    # gamma=2.0: focus on hard examples (standard value)
    # alpha: per-class weights (higher for rare classes like pion)
    criteria=[
        dict(
            type="FocalLoss",
            gamma=2.0,
            alpha=[0.6, 0.15, 0.9, 0.7, 0.7],  # electron, muon, pion, proton, gamma
            loss_weight=1.0,
            ignore_index=-1,
        ),
        # Lovasz loss disabled for now - can be unstable early in training
        # Re-enable once model is stable, or use for fine-tuning
        # dict(type="LovaszLoss", mode="multiclass", loss_weight=0.1, ignore_index=-1),
    ],
)

# scheduler settings
epoch = 1000  # Fewer epochs for initial testing; increase for production
optimizer = dict(type="AdamW", lr=0.06, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.06,0.006],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=100.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=0.006)]

# dataset settings
dataset_type = "LArTPCDataset"
data_root = "data/lartpc"

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
        coord_scale=0.001,  # Set to 0.01 if converting cm to m
        log_transform_edep=True,
        transform=[
            #dict(type="CenterShift", apply_z=True),
            #dict(
            #    type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2
            #),
            # Full rotation around beam axis (z for typical LArTPC orientation)
            #dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            # Small rotations around other axes
            #dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            #dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            #dict(type="RandomFlip", p=0.5),
            #dict(type="RandomJitter", sigma=0.005, clip=0.02),
            # Grid sampling - adjust grid_size based on detector resolution
            # 0.02 = 2cm voxels; may need adjustment for LArTPC
            dict(
                type="GridSample",
                grid_size=0.0002,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            # Limit points per sample for memory management
            dict(type="SphereCrop", point_max=102400, mode="random"),
            #dict(type="CenterShift", apply_z=False),
            # Note: NormalizeColor expects values in 0-255 range
            # Wire coordinates may need different normalization
            # dict(type="NormalizeColor"),  # Skip if using wire coords
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("strength", "color"),  # strength + wire_coords = 6 channels
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
        coord_scale=0.001,  # Must match train
        log_transform_edep=True,
        transform=[
            #dict(type="CenterShift", apply_z=True),
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(
                type="GridSample",
                grid_size=0.0002,  # Must match train
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            #dict(type="CenterShift", apply_z=False),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "origin_segment", "inverse"),
                feat_keys=("strength", "color"),
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="val",  # Use val split for testing; change to "test" if available
        data_root=data_root,
        use_reco_coords=True,
        use_edep_as_strength=True,
        label_mode="pid",
        coord_scale=0.001,  # Must match train
        log_transform_edep=True,
        transform=[
            # Copy segment to origin_segment before voxelization
            # This preserves original labels for proper evaluation
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.0002,  # Must match train
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
                return_inverse=True,  # Required to map predictions back to original points
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
            aug_transform=[
                # [
                #     dict(
                #         type="RandomRotateTargetAngle",
                #         angle=[0],
                #         axis="z",
                #         center=[0, 0, 0],
                #         p=1,
                #     )
                # ],
                # [
                #     dict(
                #         type="RandomRotateTargetAngle",
                #         angle=[1 / 2],
                #         axis="z",
                #         center=[0, 0, 0],
                #         p=1,
                #     )
                # ],
                # [
                #     dict(
                #         type="RandomRotateTargetAngle",
                #         angle=[1],
                #         axis="z",
                #         center=[0, 0, 0],
                #         p=1,
                #     )
                # ],
                # [
                #     dict(
                #         type="RandomRotateTargetAngle",
                #         angle=[3 / 2],
                #         axis="z",
                #         center=[0, 0, 0],
                #         p=1,
                #     )
                # ],
            ],
        ),
    ),
)
