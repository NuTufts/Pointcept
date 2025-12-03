"""
LArTPC Semantic Segmentation - Fine-tuning from SONATA Pretrained Backbone

Fine-tunes a PointTransformerV3 backbone pretrained with SONATA self-supervision
for particle-type semantic segmentation.

To use:
1. First pretrain with: configs/lartpc/pretrain-sonata-v1m1-lartpc.py
2. Then fine-tune with this config, specifying the pretrained checkpoint

Usage:
    python tools/train.py --config-file configs/lartpc/semseg-sonata-v1m1-lartpc-finetune.py \
        --options weight=exp/sonata_pretrain/model/model_best.pth

Features (6 channels):
  - strength (3): Pixel values from u, v, y wire plane images
  - color (3): Wire indices for u, v, y planes
"""

# =============================================================================
# Wire plane projection parameters for recalculating wire coordinates after flips
# Each tuple: ((origin_x, origin_y, origin_z), (pitch_dir_x, pitch_dir_y, pitch_dir_z))
# - origin: A reference point on the wire plane
# - pitch_dir: Direction perpendicular to wires (direction of increasing wire index)
#
# TODO: Fill in with your detector's actual wire plane geometry
# Example for a detector with U/V at ±60° from vertical and Y vertical:
# wire_projections = [
#     ((0.0, 0.0, 0.0), (0.0, 0.866, 0.5)),   # U plane
#     ((0.0, 0.0, 0.0), (0.0, 0.866, -0.5)),  # V plane
#     ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),     # Y plane (vertical wires)
# ]
# =============================================================================
wire_projections = None  # Set to None to disable wire reprojection, or define as above

_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 1
num_worker = 1
mix_prob = 0.0
empty_cache = False
enable_amp = True
enable_wandb = True

# Grid size must match pretraining
grid_size = 0.0003

# model settings - architecture MUST match pretrained backbone
model = dict(
    type="DefaultSegmentorV2",
    num_classes=6,  # electron, muon, pion, proton, gamma, ghost
    backbone_out_channels=32,  # First value of dec_channels
    backbone=dict(
        type="PT-v3m2",  # Must match pretraining backbone
        in_channels=6,  # Must match pretraining
        order=("z", "z-trans"),
        stride=(2, 2, 2, 2),
        # Encoder architecture - MUST match pretrained model
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        # Decoder for segmentation (not used in pretraining)
        dec_depths=(2, 2, 2, 2),
        dec_channels=(32, 64, 128, 256),
        dec_num_head=(2, 4, 8, 16),
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
        traceable=False,
        enc_mode=False,  # Need decoder for segmentation
        mask_token=False,  # Not needed for fine-tuning
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
        # Lovasz loss can help with fine-tuning
        dict(type="LovaszLoss", mode="multiclass", loss_weight=0.1, ignore_index=-1),
    ],
    freeze_backbone=False,  # Set True initially if you want to train only the head first
)

# scheduler settings - lower LR for fine-tuning
epoch = 500
eval_epoch = 50
optimizer = dict(type="AdamW", lr=0.001, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.001, 0.0001],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=0.0001)]

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
            # Physics-appropriate augmentations (same as pretraining)
            dict(type="RandomScale", scale=[0.95, 1.05]),
            dict(type="RandomFlipAxis", p=0.5, axis="y", center="mean",
                 wire_projections=wire_projections),
            # Z-flip swaps u/v wire signals (columns 0,1 of strength)
            dict(type="RandomFlipAxis", p=0.5, axis="z", center="mean",
                 swap_strength_columns=(0, 1), wire_projections=wire_projections),
            dict(type="RandomJitter", sigma=0.0005, clip=0.002),
            # Grid sampling
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
