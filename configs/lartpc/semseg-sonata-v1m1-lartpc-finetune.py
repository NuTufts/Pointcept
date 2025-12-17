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
#wire_projections = None  # Set to None to disable wire reprojection, or define as above
wire_projections = [
    ((0.0,    0.0,  -338.6334821387676), (0.0, -0.866, 0.5)),
    ((0.0,    0.0,  -333.0331845276306), (0.0,  0.866, 0.5)),
    ((0.0,    0.0,  0.33), (0.0, 0.0, 1.0))
]

_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 8
num_worker = 4
mix_prob = 0.0
clip_grad = 3.0
empty_cache = False
enable_amp = True

enable_wandb = True
wandb_project = "pointcept"
save_path = "sonata/semseg-finetune"

TRAIN_FILE_LIST="pi0_train_files.txt"        # 1880 events
VAL_FILE_LIST="pi0_test_files_100events.txt" # 100 events

max_points_per_view=98304
max_points_spherecrop=98304
min_points_spherecrop=20480

# Grid size must match pretraining
grid_size = 0.25 # cm
jitter_sigma=0.06 # cm
jitter_clip=0.25 # cm
wire_scale=1.0/3456.0 # normalize the wire indices which range from 0-3456


# model settings - architecture MUST match pretrained backbone
model = dict(
    type="DefaultSegmentorV2",
    num_classes=6,  # electron, muon, pion, proton, gamma, ghost
    backbone_out_channels=16,  # First value of dec_channels (halved)
    backbone=dict(
        type="PT-v3m2",  # Must match pretraining backbone
        in_channels=6,  # Must match pretraining
        order=("z", "z-trans"),
        stride=(2, 2, 2, 2),
        # Encoder architecture - MUST match pretrained model (halved for 16GB GPU)
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(16, 32, 64, 128, 256),  # Halved to match pretraining
        enc_num_head=(2, 2, 4, 8, 16),  # Adjusted to divide channels evenly
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        # Decoder for segmentation (not used in pretraining)
        dec_depths=(2, 2, 2, 2),
        dec_channels=(16, 32, 64, 128),  # Halved to match encoder
        dec_num_head=(2, 2, 4, 8),  # Adjusted to divide channels evenly
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
            reduction='mean',
        ),
        # Lovasz loss can help with fine-tuning
        #dict(type="LovaszLoss", mode="multiclass", loss_weight=0.1, ignore_index=-1),
    ],
    freeze_backbone=False,  # Set True initially if you want to train only the head first
    # Initialize linear head bias with log-prior for faster convergence
    # Class order: electron=0, muon=1, pion=2, proton=3, gamma=4, ghost=5
    # Approximate class frequencies from data (adjust based on actual statistics):
    #   electron: ~4%, muon: ~25%, pion: ~0.5%, proton: ~0.5%, gamma: ~0%, ghost: ~70%
    class_priors=[0.04, 0.25, 0.005, 0.005, 0.001, 0.70],
)

# scheduler settings - lower LR for fine-tuning pretrained encoder
epoch = 100
eval_epoch = 100

# Learning rates:
#   - Decoder + seg_head (randomly initialized): higher LR (base_lr)
#   - Encoder + embedding (pretrained): lower LR (base_lr / 10)
base_lr = 0.001
pretrained_lr = base_lr / 100.0  # 0.0001

optimizer = dict(type="AdamW", lr=base_lr, weight_decay=0.05)

# param_dicts defines parameter groups with different LRs
# Group 0 (default): params NOT matching any keyword -> base_lr (decoder, seg_head)
# Group 1: params matching "backbone.enc" -> pretrained_lr (encoder)
# Group 2: params matching "backbone.embedding" -> pretrained_lr (embedding)
param_dicts = [
    dict(keyword="backbone.enc", lr=pretrained_lr),
    dict(keyword="backbone.embedding", lr=pretrained_lr),
]

# OneCycleLR max_lr must have one value per param group
# Order: [default_group, enc_group, embedding_group]
scheduler = dict(
    type="OneCycleLR",
    max_lr=[base_lr, pretrained_lr, pretrained_lr],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

# dataset settings
dataset_type = "LArTPCDataset"

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
        wire_scale=1.0/3456.0,
        include_ghosts=True,
        exclude_other=True,
        log_transform_edep=True,
        transform=[
            # Physics-appropriate augmentations (same as pretraining)
            dict(type="RandomScale", scale=[0.95, 1.05]),
            dict(type="RandomFlipAxis", p=0.5, axis="y", center="mean",
                 coord_scale=1.0,
                 swap_strength_columns=(0, 1),
                 wire_projections=wire_projections),
            # Z-flip swaps u/v wire signals (columns 0,1 of strength)
            dict(type="RandomFlipAxis", p=0.5, axis="z", center="mean",
                 coord_scale=1.0,
                 swap_strength_columns=(0, 1), 
                 wire_projections=wire_projections),
            dict(type="RandomJitter", sigma=jitter_sigma, clip=jitter_clip),
            # Grid sampling
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
        data_list_file=VAL_FILE_LIST,
        use_reco_coords=True,
        use_edep_as_strength=True,
        label_mode="pid",
        include_ghosts=True,
        exclude_other=True,
        coord_scale=1.0,
        wire_scale=1.0/3456.0,
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
    # Use SonataFinetuneCheckpointLoader to properly load SONATA pretrained weights
    # Maps student.backbone.* -> backbone.* (encoder + embedding only)
    # Decoder and seg_head remain randomly initialized
    dict(type="SonataFinetuneCheckpointLoader", use_teacher=True),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator", write_cls_iou=True),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]
