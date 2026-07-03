"""
Shower Origin Prediction V3 - Virtual Grid + Hungarian Matching (v6 Backbone)

Multi-shower origin prediction using query-conditioned Slot Attention with:
- Virtual grid points for predicting origins in empty space
- Per-slot predictions (each slot maps to one shower via Hungarian matching)
- Inside/outside TPC classification head per slot
- DETR-style no-object penalty for unmatched slots

Uses the v6 Sonata backbone:
- in_channels=6 (coord + strength)
- enc_channels=(48, 96, 192, 384, 512)
- up_cast_level=2 -> backbone_out_channels = 192+384+512 = 1088
- flash_attn backend, bfloat16

All coordinates are normalized via NormalizeShowerCoords:
  center=[125.0, 0.0, 518.0], scale=179.55

Usage:
    python tools/train.py \
        --config-file configs/lartpc/shower_origin/archive/shower-origin-sonata-v1m1-v3.py \
        --options weight=sonata/lartpc_v6_h200_noghosts_pretrain/model/epoch_50.pth
"""

_base_ = ["../../../_base_/default_runtime.py"]

# =============================================================================
# Training settings
# =============================================================================
batch_size = 120  # Smaller batch due to per-slot memory
batch_size_val = 96
num_worker = 20
mix_prob = 0.0
empty_cache = True  # Free CUDA cache between forward/backward to reduce peak memory
enable_amp = False
enable_wandb = True
wandb_project = "pointcept-shower-origin"
save_path = "shower_origin/sonata_v1m1_v3_v6backbone_pax_pi0filter"
clip_grad = 1.0

epoch = 300
eval_epoch = 300  # Evaluate every 10 epochs
evaluate = True
base_lr = 1.0e-4  # Lower LR since only training head

# Backend settings (must match v6 pretrained model)
flash_backend = 'flash_attn'
amp_dtype = "bfloat16"
#flash_backend='xformers'   # backend needed to run on P100
#amp_dtype = "float16"      # use this with xformer backend on P100

# Grid size must match pretraining
grid_size = 0.25  # cm (applied before NormalizeShowerCoords)

# Point limits (reduced from 10240 to reduce GPU memory usage)
max_points_spherecrop = 10240
min_points_spherecrop = 512
biased_spherecrop_radius = 0.005  # cm (applied before NormalizeShowerCoords)

# Coordinate normalization (must match v6 pretraining)
coord_center = [125.0, 0.0, 518.0]
coord_scale = 1036.0 * 3**0.5 / 2.0 / 5.0  # ~179.55

# =============================================================================
# Data files
# =============================================================================
#TRAIN_FILE_LIST = "/home/twongjirad/working/larbys/gen2/container_u22/Pointcept/shower_origin_single_event_test.txt"
#VAL_FILE_LIST = "/home/twongjirad/working/larbys/gen2/container_u22/Pointcept/shower_origin_single_event_val.txt"

#TRAIN_FILE_LIST = "/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/lartpc_data_prep/hdflist_showerfragment_bnb_nu_pi0filter_dev_train.txt"
#VAL_FILE_LIST = "/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/lartpc_data_prep/hdflist_showerfragment_bnb_nu_pi0filter_dev_val.txt"

TRAIN_FILE_LIST = "/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/lartpc_data_prep/hdflist_showerfragment_bnb_nu_pi0filter_train.txt"
VAL_FILE_LIST   = "/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/lartpc_data_prep/hdflist_showerfragment_bnb_nu_pi0filter_val.txt"

# ============================================================================
# Pretraining weights
# ============================================================================
weight = "/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/sonata/lartpc_v6_h200_noghosts_pretrain/lartpc_v6_h200_noghosts_pretrain_epoch50.pth"


# =============================================================================
# Model configuration
# =============================================================================
# Backbone feature dimension after upcast
# With enc_channels=(48, 96, 192, 384, 512) and up_cast_level=2:
# 192 + 384 + 512 = 1088
backbone_out_channels = 1088

# Normalized TPC bounds for virtual grid
tpc_bounds_normalized = dict(
    x_min=-0.94,
    x_max=0.92,
    y_min=-0.65,
    y_max=0.65,
    z_min=-2.88,
    z_max=2.88,
)

model = dict(
    type="ShowerOriginPredictorV3",
    backbone=dict(
        type="Sonata-v1m1",
        # PT-v3m2 backbone config - MUST match v6 pretrained model exactly
        backbone=dict(
            type="PT-v3m2",
            in_channels=6,  # coord(3) + strength(3)
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
            enable_flash=True,
            flash_backend=flash_backend,
            upcast_attention=False,
            upcast_softmax=False,
            traceable=True,
            enc_mode=True,
            mask_token=True,
        ),
        # SONATA head config - must match v6 pretraining
        head_in_channels=1088,
        head_hidden_channels=2048,
        head_embed_channels=256,
        head_num_prototypes=4096,
        num_global_view=2,
        num_local_view=6,
        up_cast_level=2,
    ),
    backbone_out_channels=backbone_out_channels,

    # Slot Attention config
    num_slots=1,
    slot_iterations=3,

    # Cross-attention config
    num_cross_attn_layers=5,
    num_heads=16,

    # Score head config
    hidden_channels=256,
    predict_distance=False,

    # Loss weights
    score_loss_weight=1.0,
    distance_loss_weight=0.1,
    classification_loss_weight=0.5,

    # Gaussian sigma for soft labels (normalized units: 4cm / 179.55 ≈ 0.022)
    gaussian_sigma=0.022,

    # Origin classification: 3 classes (INSIDE, OUTSIDE, ON_TRACK)
    num_origin_classes=3,

    # Freeze backbone (critical for linear probe style training)
    freeze_backbone=True,

    # Virtual grid config (all in normalized units)
    # Disabled by default: 91K virtual points × 1088 channels causes OOM on <=16GB GPUs
    virtual_grid_enabled=False,
    dense_spacing=0.011,       # ~2 cm / 179.55
    dense_radius=0.167,        # ~30 cm / 179.55
    sparse_spacing=0.056,      # ~10 cm / 179.55
    tpc_bounds=tpc_bounds_normalized,
    k_neighbors=8,
    pos_embed_dim=64,
)

# =============================================================================
# Optimizer and scheduler
# =============================================================================
optimizer = dict(
    type="AdamW",
    lr=base_lr,
    weight_decay=0.01,
)

scheduler = dict(
    type="OneCycleLR",
    max_lr=[base_lr],
    pct_start=0.1,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)

# =============================================================================
# Shared transform components
# =============================================================================
# Common Collect keys for all splits
_collect_keys = (
    "coord",
    "grid_coord",
    "shower_mask",
    "origin_coord",
    "origin_type",
    "origin_distance",
    "start_coord",
)

# Common dataset kwargs
_dataset_common = dict(
    type="ShowerOriginDataset",
    use_reco_coords=True,
    log_transform_edep=False,  # LogTransform handles this in pipeline
    coord_scale=1.0,
    wire_scale=1.0/3456.0,
    min_shower_points=20,
    include_ghosts=False,  # Match v6 backbone pretraining (no ghosts)
    max_showers_per_event=200,
)

# =============================================================================
# Dataset configuration
# =============================================================================
dataset_type = "ShowerOriginDataset"
data_root = "data/lartpc"

data = dict(
    num_classes=1,
    ignore_index=-1,
    names=["origin"],
    train=dict(
        **_dataset_common,
        data_list_file=TRAIN_FILE_LIST,
        transform=[
            # 1. Crop around fragment centroid (in cm, before normalization)
            dict(
                type="BiasedSphereCrop",
                anchor_points_key="fragment_centroid",
                anchor_pdf_key=None,
                radius=biased_spherecrop_radius,
                point_max=max_points_spherecrop,
                point_min=min_points_spherecrop,
                prob_random=0.0,
                max_retries=100,
                fallback_to_random=True,
            ),
            # 2. Voxelize (in cm, before normalization)
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            # 3. Normalize all coordinates (coord, origin_coord, start_coord)
            dict(
                type="NormalizeShowerCoords",
                center=coord_center,
                scale=coord_scale,
                extra_keys=("origin_coord", "start_coord"),
            ),
            # 4. Log-transform pixel values (strength)
            dict(
                type="LogTransform",
                min_val=0.01,
                max_val=1000.0,
                log=True,
                keys=("strength",),
            ),
            # 5. Augmentation (in normalized coords)
            # extra_coord_keys ensures origin_coord/start_coord are
            # transformed consistently with coord
            # dict(
            #     type="RandomScale",
            #     scale=[0.95, 1.05],
            #     extra_coord_keys=("origin_coord", "start_coord"),
            # ),
            # dict(
            #     type="RandomFlipAxis",
            #     p=0.5,
            #     axis="y",
            #     center="mean",
            #     coord_scale=1.0,
            #     swap_strength_columns=(0, 1),
            #     extra_coord_keys=("origin_coord", "start_coord"),
            # ),
            # dict(
            #     type="RandomFlipAxis",
            #     p=0.5,
            #     axis="z",
            #     center="mean",
            #     coord_scale=1.0,
            #     swap_strength_columns=(0, 1),
            #     extra_coord_keys=("origin_coord", "start_coord"),
            # ),
            # 6. Convert to tensors
            dict(type="ToTensor"),
            # 7. Collect with coord+strength = 6 input channels
            dict(
                type="Collect",
                keys=_collect_keys,
                feat_keys=("coord", "strength"),
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        **_dataset_common,
        data_list_file=VAL_FILE_LIST,
        transform=[
            dict(
                type="BiasedSphereCrop",
                anchor_points_key="fragment_centroid",
                anchor_pdf_key=None,
                radius=biased_spherecrop_radius,
                point_max=max_points_spherecrop,
                point_min=min_points_spherecrop,
                prob_random=0.0,
                max_retries=100,
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
                type="NormalizeShowerCoords",
                center=coord_center,
                scale=coord_scale,
                extra_keys=("origin_coord", "start_coord"),
            ),
            dict(
                type="LogTransform",
                min_val=0.01,
                max_val=1000.0,
                log=True,
                keys=("strength",),
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=_collect_keys,
                feat_keys=("coord", "strength"),
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        **_dataset_common,
        data_list_file=VAL_FILE_LIST,
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
                    type="NormalizeShowerCoords",
                    center=coord_center,
                    scale=coord_scale,
                ),
                dict(
                    type="LogTransform",
                    min_val=0.01,
                    max_val=1000.0,
                    log=True,
                    keys=("strength",),
                ),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=_collect_keys + ("index",),
                    feat_keys=("coord", "strength"),
                ),
            ],
            aug_transform=[],
        ),
    ),
)

# =============================================================================
# Hooks
# =============================================================================
hooks = [
    # Load pretrained Sonata v6 weights into backbone
    dict(type="SonataCheckpointLoader"),
#    dict(type="GradScalerMonitor", log_frequency=10), # AMP overflow tracking
    dict(type="AdamStateMonitor", log_frequency=10),  # gradient/param monitoring
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="ShowerOriginEvaluator", score_threshold=0.05),
    dict(type="CheckpointSaver", save_freq=None),
]
