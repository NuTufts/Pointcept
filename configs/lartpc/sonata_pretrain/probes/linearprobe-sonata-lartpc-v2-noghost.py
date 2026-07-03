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
wire_projections = None

_base_ = ["../../../_base_/default_runtime.py"]

# misc custom setting
batch_size = 64
num_worker = 4
mix_prob = 0.0
empty_cache = False
enable_amp = True
enable_wandb = True
wandb_project = "pointcept"
save_path = "sonata/linearprobe_v2_p100_noghost"
epoch=100
eval_epoch=100
base_lr=0.010

# Grid size must match pretraining
grid_size = 0.25

max_points_per_view=20480
max_points_spherecrop=20480
min_points_spherecrop=4096
biased_spherecrop_radius=20.0

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
    num_classes=5,  # electron, muon, pion, proton, gamma
    backbone_out_channels=496,  # 16 + 32 + 64 + 128 + 256 for up_cast_level=4
    backbone=dict(
        type="Sonata-v1m1",
        # PT-v3m2 backbone config - MUST match pretrained model exactly
        backbone=dict(
            type="PT-v3m2",
            in_channels=3,
            order=("z", "z-trans","hilbert", "hilbert-trans"),
            stride=(2, 2, 2, 2),
            enc_depths=(2, 2, 2, 6, 2),
            enc_channels=(16, 32, 64, 128, 256),  # Halved model
            enc_num_head=(2, 2, 4, 8, 16),
            enc_patch_size=(256, 256, 256, 256, 256),
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
            flash_backend='flash_attn', # default backend
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
    # Class order: electron=0, muon=1, pion=2, proton=3, gamma=4
    # Approximate class frequencies from data (renormalized without ghost):
    #   electron: ~11%, muon: ~87%, pion: ~0.2%, proton: ~0.8%, gamma: ~1%
    class_priors=[0.10923, 0.87100, 0.00181, 0.00759, 0.01041],
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
    num_classes=5,
    ignore_index=-1,
    names=[
        "electron",
        "muon",
        "pion",
        "proton",
        "gamma"
    ],
    train=dict(
        type=dataset_type,
        data_list_file=TRAIN_FILE_LIST,
        use_reco_coords=True,
        use_edep_as_strength=True,
        label_mode="pid",
        coord_scale=1.0,
        wire_scale=1.0/3456.0,  # Must match pretraining
        include_ghosts=False,
        exclude_other=True,
        true_points_only=True,
        log_transform_edep=True,
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
        split="val",
        data_list_file=VAL_FILE_LIST,
        use_reco_coords=True,
        use_edep_as_strength=True,
        label_mode="pid",
        include_ghosts=False,
        exclude_other=True,
        true_points_only=True,  # Must match train config to filter ghost points
        coord_scale=1.0,
        wire_scale=1.0/3456.0,  # Must match pretraining
        log_transform_edep=True,
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
        data_list_file=VAL_FILE_LIST,
        use_reco_coords=True,
        use_edep_as_strength=True,
        label_mode="pid",
        include_ghosts=False,
        exclude_other=True,
        true_points_only=True,  # Must match train config to filter ghost points
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
                    feat_keys=("strength",),  # Must match pretraining: strength only, no wire coords
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
