"""
LoRA Fine-Tuning for SONATA on LArTPC -- De-ghosting Task (binary ghost/real)
=============================================================================

Fine-tunes the SONATA pretrained backbone (v6) using LoRA for the de-ghosting
task: binary classification of each point as real (0) or ghost (1).

Corrections applied (after overfit-test debugging on a single dev event):

  - drop_cosmics=False on both train and val. The previous drop_cosmics=True/0.9
    left a Swiss-cheese ghost cloud in cosmic regions (cosmic real points were
    masked out but cosmic-region ghosts survived). Unphysical training target.
  - HasmatchAsGhost transform reads the producer's per-spacepoint hasmatch
    field directly: real (hasmatch=1) -> 0, ghost (hasmatch=0) -> 1.
    Replaces RemapGhostLabel + the dataset's SSNETLABEL_TO_CLASS chain, which
    was correct in principle but fragile (transform not in committed code).
  - FocalLoss alpha=0.5 (neutral). The previous alpha=0.2 was reweighting
    classes in the wrong direction given how Pointcept's FocalLoss applies
    alpha in the multi-class one-hot form; LovaszLoss handles class balance.
  - Model now defaults strip_student=True (in SonataLoRADeghostSegmentor) so
    the unused student subtree is removed before optimizer build, halving
    the backbone's GPU memory and parameter count.

The training-pipeline differences vs the single-event overfit config:

  - Batched: batch_size=96 events; BiasedSphereCrop crops each event to
    point_max=10240 points so they fit in a batch.
  - Augmentations restored: RandomRotate (x/y/z), RandomFlip, RandomJitter,
    MultiplicativeRandomJitter on strength.
  - OneCycleLR scheduler (anneals over the full training run).

Model head
----------
deghost_head: nn.Linear(backbone_out_channels, 2) -- outputs (N, 2) logits.
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
wandb_project    = "pointcept-lora-deghost"
save_path        = "sonata/lora_deghost_v6_hasmatch_extbnb_larmatch"
epoch            = 50
eval_epoch       = 50
base_lr          = 5e-4
lora_lr          = 2e-4
head_lr          = base_lr

#flash_backend = 'xformers'
flash_backend = 'flash_attn'
amp_dtype     = "float16"

# Grid size and coordinate normalization
grid_size    = 0.25
coord_scale  = 1036.0 * 3**0.5 / 2.0 / 5.0   

_scaled_grid_size = 0.0013932197615579773   # grid_size / coord_scale
_jitter_sigma     = 0.0003483049403894943   # _scaled_grid_size / 4

# Strength jitter -- matches global_shared_transform in v6 pretraining 
_strength_jitter_sigma = 0.05
_strength_jitter_clip  = 0.05

max_points_per_view      = 20480
max_points_spherecrop    = 10240
min_points_spherecrop    = 4096
biased_spherecrop_radius = 20.0

TRAIN_FILE_LIST = "/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/lartpc_data_prep/hdflist_combined_prod4_validated_shuffled_trainsplit.txt"
VAL_FILE_LIST   = "/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/lartpc_data_prep/hdflist_combined_prod4_validated_shuffled_valsplit.txt"

pretrain_model_path = "/cluster/tufts/wongjiradlab/twongj01/mphys/Pointcept/sonata/lartpc_v7_h200_extbnb_larmatch_run1/model/epoch_18.pth"

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
            in_channels=6,
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
        head_in_channels=1088,
        head_hidden_channels=2048,
        head_embed_channels=256,
        head_num_prototypes=4096,
        num_global_view=2,
        num_local_view=6,
        up_cast_level=4,
    ),
    # Focal + Lovász combination.
    criteria=[
        dict(
            type="FocalLoss",
            gamma=2.0,
            alpha=0.5,          # neutral: LovaszLoss does the class-balance work
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
# Dataset
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
        include_ghosts=True,
        true_points_only=False,
        exclude_other=True,
        drop_cosmics=False,
        drop_cosmics_prob=0.0,
        transform=[
            dict(
                type="BiasedSphereCrop",
                anchor_points_key="nu_vertices",
                anchor_pdf_key=None,
                radius=biased_spherecrop_radius,
                point_max=max_points_spherecrop,
                point_min=min_points_spherecrop,
                # Half the train crops are uniform-random (anywhere in the
                # TPC), half are nu-anchored. Gives the deghoster meaningful
                # exposure to cosmic-dominated regions of the TPC, not just
                # the dense nu-interaction region. Bumped from 0.25.
                prob_random=0.5,
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
                type="NormalizeCoord",
                center=[125.0, 0.0, 518.0],
                scale=coord_scale,
            ),
            # Log-transform ADC values. max_val=1000.0 matches v6 pretraining
            
            dict(
                type="LogTransform",
                min_val=0.01,
                max_val=1000.0,
                log=True,
                keys=("strength",),
            ),
            # Strength jitter in log space
            dict(
                type="MultiplicativeRandomJitter",
                sigma=_strength_jitter_sigma,
                clip=_strength_jitter_clip,
                keys="strength",
                p=0.8,
                log_space=True,
            ),
            dict(
                type="HasmatchAsGhost",
                real_target_index=0,
                ghost_target_index=1,
                ignore_index=-1,
            ),
            dict(
                type="CenterShift",
                apply_z=False,
                axes=("x", "y", "z"),
            ),
            dict(
                type="RandomRotate",
                angle=[-1, 1],
                axis="z",
                center=[0, 0, 0],
                p=0.8,
            ),
            dict(
                type="RandomRotate",
                angle=[-1, 1],
                axis="x",
                center=[0, 0, 0],
                p=0.8,
            ),
            dict(
                type="RandomRotate",
                angle=[-1, 1],
                axis="y",
                center=[0, 0, 0],
                p=0.8,
            ),
            dict(
                type="RandomFlip",
                p=0.5,
                axes=("x", "y", "z"),
            ),
            dict(
                type="RandomJitter",
                sigma=_jitter_sigma,
                clip=_scaled_grid_size,
                keys=("coord",),
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "segment_counts"),
                feat_keys=("coord", "strength"),
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
        include_ghosts=True,
        true_points_only=False,
        coord_scale=1.0,
        exclude_other=True,
        drop_cosmics=False,
        drop_cosmics_prob=0.0,
        transform=[
            dict(
                type="BiasedSphereCrop",
                anchor_points_key="nu_vertices",
                anchor_pdf_key=None,
                radius=biased_spherecrop_radius,
                point_max=max_points_spherecrop,
                point_min=min_points_spherecrop,
                # Val uses uniform-random crops (NOT nu-anchored) so the
                # reported val mIoU reflects deghoster performance on the
                # full TPC distribution, not just the easy nu-vertex region.
                # The previous 0.25 (75% nu-anchored) overstated val mIoU
                # by scoring only the densest part of every event.
                prob_random=1.0,
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
                type="NormalizeCoord",
                center=[125.0, 0.0, 518.0],
                scale=coord_scale,
            ),
           
            dict(
                type="LogTransform",
                min_val=0.01,
                max_val=1000.0,
                log=True,
                keys=("strength",),
            ),
            dict(
                type="HasmatchAsGhost",
                real_target_index=0,
                ghost_target_index=1,
                ignore_index=-1,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "segment_counts"),
                feat_keys=("coord", "strength"),
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
        include_ghosts=True,
        true_points_only=False,
        coord_scale=1.0,
        exclude_other=True,
        drop_cosmics=False,
        drop_cosmics_prob=0.0,
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
                    type="NormalizeCoord",
                    center=[125.0, 0.0, 518.0],
                    scale=coord_scale,
                ),
                
                dict(
                    type="LogTransform",
                    min_val=0.01,
                    max_val=1000.0,
                    log=True,
                    keys=("strength",),
                ),
                dict(
                    type="HasmatchAsGhost",
                    real_target_index=0,
                    ghost_target_index=1,
                    ignore_index=-1,
                ),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index"),
                    feat_keys=("coord", "strength"),
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
        pretrained_path=pretrain_model_path,
    ),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator", write_cls_iou=True),
    dict(type="CheckpointSaver", save_freq=1),
]
