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
wire_projections = None

_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
#batch_size = 512
batch_size = 384
batch_size_val = 512
num_worker = 20
mix_prob = 0.0
clip_grad = 1.0
empty_cache = False
enable_amp = True

enable_wandb = True
wandb_project = "pointcept"
save_path = "sonata/semseg-decoder-finetune-v3-noghost-p100-resume-dropcosmics"

epoch = 40
eval_epoch = 40
base_lr = 0.001

TRAIN_FILE_LIST="/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/train_split_combined_prod3_validated.txt"
VAL_FILE_LIST="/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/val_split_combined_prod3_validated_shuffled.txt"
TEST_FILE_LIST="/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/test_split_combined_prod3_validated_shuffled.txt"

#flash_backend='flash_attn' # default backend
#amp_dtype = "bfloat16" # use this for default 'flash_attn' backend

flash_backend='xformers'   # backend needed to run on P100
amp_dtype = "float16"      # use this with xformer backend on P100

#max_points_per_view=20480
#max_points_spherecrop=20480
max_points_per_view=10240
max_points_spherecrop=10240
min_points_spherecrop=4096
biased_spherecrop_radius=20.0

# Grid size must match pretraining
grid_size = 0.25 # cm
jitter_sigma=0.06 # cm
jitter_clip=0.25 # cm
wire_scale=1.0/3456.0 # normalize the wire indices which range from 0-3456


# model settings - architecture MUST match pretrained backbone
model = dict(
    type="DefaultSegmentorV2",
    num_classes=8,  # electron, muon, pion, proton, gamma, michel, delta, led; ghost (removed), other (removed)
    backbone_out_channels=16,  # First value of dec_channels (halved)
    backbone=dict(
        type="PT-v3m2",  # Must match pretraining backbone
        in_channels=3,  # Must match pretraining
        order=("z", "z-trans","hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        # Encoder architecture - MUST match pretrained model (halved for 16GB GPU)
        enc_depths=(3, 3, 3, 9, 3), # voxel size at each encoder layer: (0.25, 0.5, 1.0, 2.0, 4.0); panda depths: (3, 3, 3, 9, 3)
        enc_channels=(16, 32, 64, 128, 256),  # Halved for 16GB GPU; panda channels: (48, 96, 192, 384, 512)
        enc_num_head=(2, 2, 4, 8, 16),  # Adjusted to divide channels evenly
        enc_patch_size=(256, 256, 256, 256, 256),
        # Decoder for segmentation (not used in pretraining)
        dec_depths=(2, 2, 2, 2),
        dec_channels=(16, 32, 64, 128),  # Halved to match encoder
        dec_num_head=(2, 2, 4, 8),  # Adjusted to divide channels evenly
        dec_patch_size=(256,256,256,256),
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
        flash_backend=flash_backend,        
        upcast_attention=False,
        upcast_softmax=False,
        traceable=False,
        enc_mode=False,    # Need decoder for segmentation
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
    # Class order: electron=0, muon=1, pion=2, proton=3, gamma=4, michel=5, delta=6, led=7, ghost=8, other=9
    # Approximate class frequencies from data (adjust based on actual statistics):
    #class_priors=[0.052,0.797,0.009,0.024,0.030,0.006,0.077,0.007],
    class_priors=None
)

# scheduler settings - lower LR for fine-tuning pretrained encoder
# Learning rates:
#   - Decoder + seg_head (randomly initialized): higher LR (base_lr)
#   - Encoder + embedding (pretrained): lower LR (base_lr / 10)
#pretrained_lr = base_lr / 10.0  # 0.0001
pretrained_lr = 0.0 # freeze the encoder

optimizer = dict(type="AdamW", lr=base_lr, weight_decay=0.005)

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
    pct_start=0.01,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)

# dataset settings
dataset_type = "LArTPCDataset"

data = dict(
    num_classes=8,
    ignore_index=-1,
    names=[
        "electron",
        "muon",
        "pion",
        "proton",
        "gamma",
        "michel",
        "delta",
        "led"
    ],
    train=dict(
        type=dataset_type,
        data_list_file=TRAIN_FILE_LIST,
        use_reco_coords=True,
        use_edep_as_strength=True,
        label_mode="ssnet",
        coord_scale=1.0,
        wire_scale=1.0/3456.0,
        include_ghosts=False,
        exclude_other=True,
        true_points_only=True,
        log_transform_edep=True,
        drop_cosmics=True,
        drop_cosmics_prob=0.9,
        transform=[
            # Voxelize to grid
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            # Limit points per sample for 
            # (1) memory management,
            # (2) near neutrino interactions
            dict(type="BiasedSphereCrop", 
                anchor_points_key="nu_vertices",
                anchor_pdf_key=None,
                radius=biased_spherecrop_radius,
                point_max=max_points_spherecrop, 
                point_min=min_points_spherecrop, 
                prob_random=0.25,
                max_retries=100,
                fallback_to_random=True),
            # Minimal augmentation for linear probe - just test feature quality
            # Can add augmentations matching pretraining if desired
            dict(type="RandomScale", scale=[0.95, 1.05]),
            dict(type="RandomFlipAxis", p=0.5, axis="y", center="mean",
                 wire_projections=wire_projections),
            dict(type="RandomFlipAxis", p=0.5, axis="z", center="mean",
                 wire_projections=wire_projections),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "segment_counts"),
                feat_keys=("strength",),  # Must match pretraining: strength only, no wire coords
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        data_list_file=VAL_FILE_LIST,
        use_reco_coords=True,
        use_edep_as_strength=True,
        label_mode="ssnet",
        include_ghosts=False,
        exclude_other=True,
        true_points_only=True,  # Must match train config to filter ghost points
        coord_scale=1.0,
        wire_scale=1.0/3456.0,
        log_transform_edep=True,
        drop_cosmics=True,
        drop_cosmics_prob=0.9,        
        transform=[
            # Note: removed inverse mapping since BiasedSphereCrop invalidates it
            # Evaluation happens on voxelized+cropped data (same as training)
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            # Limit points per sample for
            # (1) memory management,
            # (2) near neutrino interactions
            dict(type="BiasedSphereCrop",
                anchor_points_key="nu_vertices",
                anchor_pdf_key=None,
                radius=biased_spherecrop_radius,
                point_max=max_points_spherecrop,
                point_min=min_points_spherecrop,
                prob_random=0.25,
                max_retries=100,
                fallback_to_random=True),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "segment_counts"),
                feat_keys=("strength",),  # Must match pretraining: strength only, no wire coords
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        data_list_file=TEST_FILE_LIST,
        use_reco_coords=True,
        use_edep_as_strength=True,
        label_mode="ssnet",
        include_ghosts=False,
        exclude_other=True,
        true_points_only=True,  # Must match train config to filter ghost points
        coord_scale=1.0,
        wire_scale=1.0/3456.0,        
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
                    keys=("coord", "grid_coord", "index","segment","segment_counts"),
                    feat_keys=("strength"),
                ),
            ],
            aug_transform=[],
        ),
    ),
)

hooks = [
    # For loading SONATA pretrained weights (encoder only), use:
    #   dict(type="SonataFinetuneCheckpointLoader", use_teacher=True),
    # For resuming from a previous finetuned checkpoint, use:
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator", write_cls_iou=True),
    dict(type="CheckpointSaver", save_freq=1),
    dict(type="PreciseEvaluator", test_last=False),
]
