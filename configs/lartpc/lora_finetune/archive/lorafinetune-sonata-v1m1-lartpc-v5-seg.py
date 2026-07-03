"""
LoRA Fine-Tuning for SONATA on LArTPC (SSNet, 8-class)
Fine-tunes the SONATA pretrained backbone using Low-Rank Adaptation (LoRA).
  - Uses SonataLoRASegmentor instead of SonataSegmentor
  - Backbone attention layers get trainable LoRA A/B adapters
  - All other backbone weights remain frozen

Wire projections (must match pretraining):

Overall design followed closely the linear probe finetuning script
"""

wire_projections = None

_base_ = ["../../../_base_/default_runtime.py"]

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
save_path        = "sonata/lora_seg_v5_fixed"
epoch            = 10          
eval_epoch       = 1
base_lr          = 5e-4       
lora_lr          = 2e-4         # LoRA adapters – more conservative
head_lr          = base_lr      # Segmentation head – standard

flash_backend = 'xformers'
amp_dtype     = "float16"

# Grid size must match pretraining
grid_size = 0.25

max_points_per_view   = 20480
max_points_spherecrop = 10240
min_points_spherecrop = 2048
biased_spherecrop_radius = 20.0

TRAIN_FILE_LIST = "/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/lartpc_data_prep/hdflist_combined_prod4_validated_shuffled_trainsplit.txt"
VAL_FILE_LIST   = "/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/lartpc_data_prep/hdflist_combined_prod4_validated_shuffled_valsplit.txt"
true_points_only = True

# ============================================================================
# Model - Transforms were done to mirror the way pre-trained model was processed
# ============================================================================
model = dict(
    type="SonataLoRASegmentor",
    num_classes=8,
    backbone_out_channels=1232,
    lora_rank=16,
    lora_alpha=32.0,
    lora_dropout=0.05,
    lora_target_modules=["qkv", "proj"],  
    freeze_backbone_non_lora=True,
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
    criteria=[
        dict(
            type="FocalLoss",
            gamma=2.0,
            alpha=0.5,
            loss_weight=1.0,
            ignore_index=-1,
            reduction='mean',
        ),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=0.1, ignore_index=-1),
    ],
    # Class priors for head bias init (electron/muon/pion/proton/gamma/michel/delta/led) Taken from Pre Training Config
    class_priors=[0.052, 0.797, 0.009, 0.024, 0.030, 0.006, 0.077, 0.007],
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
    dict(keyword="seg_head", lr=head_lr, weight_decay=0.0),
]

scheduler = dict(
    type="OneCycleLR",
    max_lr=base_lr,        # ← single value instead of a list
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)

# ============================================================================
# Dataset (identical to linear probe)
# ============================================================================
dataset_type = "LArTPCDataset"
data_root    = "data/lartpc"

data = dict(
    num_classes=8,
    ignore_index=-1,
    names=["electron", "muon", "pion", "proton", "gamma", "michel", "delta", "led"],
    train=dict(
        type=dataset_type,
        data_list_file=TRAIN_FILE_LIST,
        use_reco_coords=True,
        use_edep_as_strength=True,
        label_mode="ssnet",
        coord_scale=1.0,
        wire_scale=1.0 / 3456.0,
        include_ghosts=False,
        exclude_other=True,
        true_points_only=True,
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
        include_ghosts=False,
        exclude_other=True,
        true_points_only=True,
        coord_scale=1.0,
        wire_scale=1.0 / 3456.0,
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
        include_ghosts=False,
        exclude_other=True,
        true_points_only=True,
        coord_scale=1.0,
        wire_scale=1.0 / 3456.0,
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
    # LoRA-aware checkpoint loader (same key remapping as SonataCheckpointLoader)
    dict(type="LoRASonataCheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    #dict(type="SemSegEvaluator", write_cls_iou=True, eval_freq=100),
    dict(type="SemSegEvaluator", write_cls_iou=True),
    dict(type="CheckpointSaver", save_freq=1),
]
