"""
LoRA Fine-Tuning for SONATA on LArTPC -- De-ghosting Task (binary ghost/real)
=============================================================================

Fine-tunes the SONATA pretrained backbone (V5) using LoRA for the de-ghosting
task: binary classification of each point as real (0) or ghost (1).

Loss
----
FocalLoss + LovaszLoss for stable training with imbalanced classes (80% ghost, 20% real)

Model head
----------
deghost_head: nn.Linear(backbone_out_channels, 2) -- outputs (N, 2) logits.
Fully compatible with SemSegEvaluator (output.max(1)[1]) and CheckpointSaver.

Two-stage inference
-------------------
Stage 1: Load this checkpoint -> deghost_head -> argmax -> ghost mask.
Stage 2: Load SSNet checkpoint -> seg_head on real points -> 8-class labels.
"""

wire_projections = None

_base_ = ["../_base_/default_runtime.py"]

# ============================================================================
# Hyper-parameters
# ============================================================================
find_unused_parameters = True
batch_size       = 96
batch_size_val   = 48
num_worker       = 22
num_worker_val   = 20
mix_prob         = 0.0
empty_cache      = False
enable_amp       = False
enable_wandb     = True
wandb_project    = "pointcept"
save_path        = "sonata/lora_v5_deghost_10_epochs_April"
epoch            = 10
eval_epoch       = 1
base_lr          = 5e-4
lora_lr          = 2e-4
head_lr          = base_lr

flash_backend = 'xformers'
amp_dtype     = "float16"

grid_size = 0.25

max_points_per_view      = 20480
max_points_spherecrop    = 10240
min_points_spherecrop    = 2048
biased_spherecrop_radius = 20.0

TRAIN_FILE_LIST = "/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/lartpc_data_prep/hdflist_combined_prod4_validated_shuffled_trainsplit.txt"
VAL_FILE_LIST   = "/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/lartpc_data_prep/hdflist_combined_prod4_validated_shuffled_valsplit.txt"
#TRAIN_FILE_LIST ="test_files_train.txt"
#VAL_FILE_LIST ="test_files_val.txt"

# ============================================================================
# Model
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
            in_channels=3,   
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
        num_local_view=4,       
        up_cast_level=4,
    ),
    # Focal + Lovász combinati
    criteria=[
        dict(
            type="FocalLoss",
            gamma=2.0,
            alpha=0.2,          # ~= fraction of real points (minority class)
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

scheduler = dict(
    type="OneCycleLR",
    max_lr=base_lr,
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)

# ============================================================================
# Dataset - V5 transform pipeline
# ============================================================================
dataset_type = "LArTPCDataset"
data_root    = "data/lartpc"

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
        wire_scale=1.0 / 3456.0,
        include_ghosts=True,
        true_points_only=False,
        exclude_other=True,
        log_transform_edep=True,  
        drop_cosmics=True,
        drop_cosmics_prob=0.9,
        transform=[
            dict(
                type="BiasedSphereCrop",
                anchor_points_key="nu_vertices",
                anchor_pdf_key=None,
                radius=biased_spherecrop_radius,
                point_max=max_points_spherecrop,
                point_min=min_points_spherecrop,
                prob_random=0.25,
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
                type="RemapGhostLabel",
                ghost_source_index=8,
                ghost_target_index=1,
                real_target_index=0,
                ignore_index=-1,
            ),
            dict(type="RandomScale", scale=[0.95, 1.05]),
            dict(
                type="RandomFlipAxis",
                p=0.5,
                axis="y",
                center="mean",
                coord_scale=1.0,
                swap_strength_columns=(0, 1),
                wire_projections=wire_projections,
            ),
            dict(
                type="RandomFlipAxis",
                p=0.5,
                axis="z",
                center="mean",
                coord_scale=1.0,
                swap_strength_columns=(0, 1),
                wire_projections=wire_projections,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "segment_counts"),
                feat_keys=("strength",),
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
        label_mode="ssnet",
        coord_scale=1.0,
        wire_scale=1.0 / 3456.0,
        include_ghosts=True,
        true_points_only=False,
        exclude_other=True,
        log_transform_edep=True,
        drop_cosmics=True,
        drop_cosmics_prob=0.9,
        transform=[
            dict(
                type="BiasedSphereCrop",
                anchor_points_key="nu_vertices",
                anchor_pdf_key=None,
                radius=biased_spherecrop_radius,
                point_max=max_points_spherecrop,
                point_min=min_points_spherecrop,
                prob_random=0.25,
                max_retries=100,
                fallback_to_random=True,
            ),
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
            ),
            dict(
                type="RemapGhostLabel",
                ghost_source_index=8,
                ghost_target_index=1,
                real_target_index=0,
                ignore_index=-1,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "segment_counts"),
                feat_keys=("strength",),
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        data_list_file=VAL_FILE_LIST,
        use_reco_coords=True,
        use_edep_as_strength=True,
        label_mode="ssnet",
        coord_scale=1.0,
        wire_scale=1.0 / 3456.0,
        include_ghosts=True,
        true_points_only=False,
        exclude_other=True,
        log_transform_edep=True,
        drop_cosmics=True,
        drop_cosmics_prob=0.9,
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
                    type="RemapGhostLabel",
                    ghost_source_index=8,
                    ghost_target_index=1,
                    real_target_index=0,
                    ignore_index=-1,
                ),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index"),
                    feat_keys=("strength",),
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
        pretrained_path="/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/sonata/lartpc_v5_h200_noghosts/model/model_last.pth",
    ),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator", write_cls_iou=True),
    dict(type="CheckpointSaver", save_freq=1),
]
