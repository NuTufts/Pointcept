"""
SONATA Pretraining for LArTPC Point Clouds

Self-supervised pretraining using masked prediction, cross-view consistency,
and local-to-global matching. Inspired by SONATA and Panda (arXiv:2512.01324).

Adapted for LArTPC physics constraints:
  - No rotations (tomographic projection geometry must be preserved)
  - No x-flip (diffusion effects break x-symmetry along drift direction)
  - Y-flip and z-flip allowed (detector symmetries)
  - Local 3D crops are valid (spacepoints already reconstructed from wire planes)

Features (3 channels total):
  - strength (3): Pixel values from u, v, y wire plane images (pixval)
  - NOTE: Wire coordinates removed to avoid geometric trap during pretraining

Coordinates:
  - units in data are in cm. We do not need to normalize these values
  - detector spans in x=[0,256],y=[-117,117],z=[0,1040]
  - wire pitch is 0.3 cm and drift spacing is about the same.
  - use grid size of 0.25 cm
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
#    ((0.0,    0.0,  -338.6334821387676), (0.0, -0.866, 0.5)),
#    ((0.0,    0.0,  -333.0331845276306), (0.0,  0.866, 0.5)),
#    ((0.0,    0.0,  0.33), (0.0, 0.0, 1.0))
#]
# =============================================================================
# Wire projections disabled - not using wire coordinates to avoid geometric trap
wire_projections = None

_base_ = ["../../../_base_/default_runtime.py"]

# misc custom setting
batch_size = 32  # Adjust based on GPU memory; LArTPC events can be large
num_worker = 20
mix_prob = 0
clip_grad = 3.0
empty_cache = False
enable_amp = True
evaluate = False
find_unused_parameters = False

#flash_backend='flash_attn' # default backend
#amp_dtype = "bfloat16" # use this for default 'flash_attn' backend

flash_backend='xformers'   # backend needed to run on P100
amp_dtype = "float16"      # use this with xformer backend on P100

enable_wandb = True
wandb_project = "pointcept"
save_path = "sonata/lartpc_v3_p100_noghosts-extended"

TRAIN_FILE_LIST="/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/trainval_split_combined_prod3_validated.txt"
#TRAIN_FILE_LIST="pi0_test_files_100events.txt"
true_points_only=True

# max_points_per_view=98304
# max_points_spherecrop=98304
# min_points_spherecrop=20480
max_points_per_view=20480
max_points_spherecrop=20480
min_points_spherecrop=4096
biased_spherecrop_radius=20.0

# scheduler settings
# original run
#epoch = 24
#eval_epoch = 24
#base_lr = 0.003
# extended run
epoch = 50
eval_epoch = 50
base_lr = 0.0001

lr_final_div_factor=1000.0 # original
#lr_pct_start=0.05 # original
#lr_final_div_factor=100.0 # extended
lr_pct_start=0.0 # extended par

lr_decay = 0.9  # layer-wise lr decay
base_wd  = 0.04
final_wd = 0.2

# LArTPC scale parameters
# Detector: ~1036 cm largest dimension, 0.3 cm wire pitch / time sampling
grid_size = 0.25 # cm
jitter_sigma=0.05 # cm (reduced)
jitter_clip=0.25  # cm (reduced)
wire_scale=1.0/3456.0 # normalize the wire indices which range from 0-3456
# notes for next run:
# - in sonata, grid_size is 0.02 with jitter 0.005, so grid_size/4. the jitter_clip=grid_size.
#   i think my settings were much too big. the sequence was probably grid positions like crazy.
# - [DONE] remove wire coordinates, maybe this is keeping us in a geometric trap
# - the position encoding is actually the output of a sparse-submanifold convolution combined in a residual manner.
# - restore the hilbert curves
# - consider using a random filter: for some p, drop ghosts, for some p drop cosmics as well.
# - make a particle bomb + corsika sample
# lessons from first run
# - linear probing is decent at ghost removal and maybe recognizing muons
# - I think a two stage processing is unavoidable.
#   - first pass: ghost removal, cosmic keypoints, nu vertex candidates, cosmic clustering
#   - second pass: nu interaction pass, refinement
#   - maybe can course grain first pass with larger grid size, for speed.

# model settings
model = dict(
    type="Sonata-v1m1",
    # backbone - student & teacher
    backbone=dict(
        type="PT-v3m2",  # Must use v3m2 for SONATA (supports mask_token)
        in_channels=3,  # strength only (3: pixel values from u,v,y planes) - no wire coords to avoid geometric trap
        order=("z", "z-trans","hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(3, 3, 3, 9, 3), # voxel size at each encoder layer: (0.25, 0.5, 1.0, 2.0, 4.0); panda depths: (3, 3, 3, 9, 3)
        enc_channels=(16, 32, 64, 128, 256),  # Halved for 16GB GPU; panda channels: (48, 96, 192, 384, 512)
        enc_num_head=(2, 2, 4, 8, 16),  # Adjusted to divide channels evenly
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
        enc_mode=True,  # Encoder-only for pretraining
        mask_token=True,  # Enable learnable mask tokens
    ),
    teacher_custom=dict(
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,  # No stochastic depth in teacher
    ),
    # Head configuration
    # head_in_channels = concatenated features after up_cast
    # With enc_channels=(16, 32, 64, 128, 256) and up_cast_level=2:
    # Halved from original to match smaller backbone
    head_in_channels=448,  # Halved from 896
    head_hidden_channels=1024,  # Halved from 2048
    head_embed_channels=256,   # Sonata originally 256
    head_num_prototypes=4096,  # Sonata originally 2048

    # View configuration
    num_global_view=2,   # Two global views with different augmentations
    num_local_view=4,    # Four local crops (valid for 3D spacepoints)

    # Masking schedule - scaled for LArTPC
    # Patch sizes in coordinate units
    # 2cm -> 15cm in real detector coordinates, following Panda
    mask_size_start=2.0,    # Start with ~1cm patches
    mask_size_base=15.0,    # End with ~5cm patches
    mask_size_warmup_ratio=0.05,
    mask_ratio_start=0.3,   # Mask 30% initially
    mask_ratio_base=0.8,    # Mask 70% at end
    mask_ratio_warmup_ratio=0.05,
    mask_jitter=0.125,      # half of grid_size 

    # Temperature schedule
    teacher_temp_start=0.04,
    teacher_temp_base=0.07,
    teacher_temp_warmup_ratio=0.05,
    student_temp=0.1,

    # Loss weights (all three losses active)
    mask_loss_weight=2 / 8,       # Predict masked regions from context
    roll_mask_loss_weight=2 / 8,  # Cross-view consistency (global views)
    unmask_loss_weight=4 / 8,     # Local-to-global matching

    # EMA momentum for teacher
    momentum_base=0.994,
    momentum_final=1.0,

    # Matching parameters - scaled for LArTPC geometry
    match_max_k=8,
    match_max_r=5.0,  # ~5cm matching radius

    # Feature upsampling through decoder levels
    up_cast_level=2,
)

# Layer-wise learning rate decay
enc_depths = model["backbone"]["enc_depths"]
param_dicts = [
    dict(
        keyword=f"enc{e}.block{b}.",
        lr=base_lr * lr_decay ** (sum(enc_depths) - sum(enc_depths[:e]) - b - 1),
    )
    for e in range(len(enc_depths))
    for b in range(enc_depths[e])
]
del enc_depths

optimizer = dict(type="AdamW", lr=base_lr, weight_decay=base_wd)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[base_lr] + [g["lr"] for g in param_dicts],
    pct_start=lr_pct_start,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=lr_final_div_factor,
)

# dataset settings
dataset_type = "LArTPCDataset"
data_root = "data/lartpc"

# Transform pipeline for SONATA pretraining
transform = [
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
        prob_random=0.10,
        max_retries=100,
        fallback_to_random=True),
    # Keep original coordinates for cross-view matching
    dict(type="Copy", keys_dict={"coord": "origin_coord"}),
    # Generate multi-scale views
    dict(
        type="MultiViewGenerator",
        view_keys=("coord", "origin_coord", "strength"),  # No wire coords (color) to avoid geometric trap
        # Global views: see most/all of the event
        global_view_num=2,
        global_view_scale=(0.6, 1.0),
        # Local views: crops of 3D regions (valid for spacepoints)
        local_view_num=4,
        local_view_scale=(0.1, 0.4),
        # Shared transforms for all views
        global_shared_transform=[
            # No color augmentation - physics signals should not be modified
        ],
        # Per-view augmentations (physics-appropriate only)
        global_transform=[
            # Small scale variation
            dict(type="RandomScale", scale=[0.95, 1.05]),
            # Y-flip (vertical) - detector is symmetric, flip around mean
            # wire_projections recalculates wire coords after flip (set above)
            dict(type="RandomFlipAxis", p=0.5, axis="y", center="mean",
                 coord_scale=1.0,
                 swap_strength_columns=(0, 1),
                 wire_projections=wire_projections),
            # Z-flip (beam direction) - symmetric for uniform ionization
            # swap_strength_columns=(0,1) swaps u/v wire signals because their
            # wire directions are reflections through the z-plane
            dict(type="RandomFlipAxis", p=0.5, axis="z", center="mean",
                 coord_scale=1.0,
                 swap_strength_columns=(0, 1),
                 wire_projections=wire_projections),
            # Small position jitter
            dict(type="RandomJitter", sigma=jitter_sigma, clip=jitter_clip),
            # NO rotations - breaks tomographic projection physics
            # NO x-flip - diffusion effects create asymmetry in drift direction
        ],
        # Local view transforms (same physics constraints)
        local_transform=[
            dict(type="RandomScale", scale=[0.95, 1.05]),
            dict(type="RandomFlipAxis", p=0.5, axis="y", center="mean",
                 coord_scale=1.0,
                 swap_strength_columns=(0, 1),
                 wire_projections=wire_projections),
            dict(type="RandomFlipAxis", p=0.5, axis="z", center="mean",
                 coord_scale=1.0,
                 swap_strength_columns=(0, 1), 
                 wire_projections=wire_projections),
            dict(type="RandomJitter", sigma=jitter_sigma, clip=jitter_clip),
        ],
        max_size=max_points_per_view,  # Max points per view
    ),
    dict(type="ToTensor"),
    dict(type="Update", keys_dict={"grid_size": grid_size}),
    dict(
        type="Collect",
        keys=(
            "global_origin_coord",
            "global_coord",
            "global_offset",
            "local_origin_coord",
            "local_coord",
            "local_offset",
            "grid_size",
            "name",
        ),
        offset_keys_dict=dict(),
        # Features: strength only (no wire coords to avoid geometric trap)
        global_feat_keys=("global_strength",),
        local_feat_keys=("local_strength",),
    ),
]

data = dict(
    num_classes=6,  # Not used for pretraining
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
        #data_root=data_root,
        data_list_file=TRAIN_FILE_LIST,
        use_reco_coords=True,
        use_edep_as_strength=True,
        label_mode="pid",
        coord_scale=1.0,
        log_transform_edep=True,
        include_ghosts=True,
        exclude_other=True,
        transform=transform,
        test_mode=False,
        loop=1,
        true_points_only=true_points_only,
    ),
)

# Original training parameters (for extending scheduler calculations)
# These are needed to compute where the original schedule was at the resume point
original_epochs = 24
#original_steps_per_epoch = 8125  # From checkpoint (110500 total_steps / 100 epochs)
original_steps_per_epoch = 11718  # For expanded data set
original_total_steps = original_epochs * original_steps_per_epoch  # = 110500

# Hooks for pretraining
hooks = [
    # extend_scheduler=True: When resuming, creates a new scheduler with the updated
    # total_steps (based on eval_epoch=200) and advances it to the current step.
    # This allows extending training beyond the original epoch count while
    # preserving optimizer state (Adam momentum, etc.)
    # CRITICAL: This also restores the scheduler's LR values after loading optimizer state.
    dict(type="CheckpointLoader", extend_scheduler=True),
    dict(type="ModelHook"),  # Calls model.before_train(), before_step(), after_step()
    # extend_scheduler=True: Computes the WD that the original schedule had at the resume
    # point, then creates a new schedule from that WD to final_wd over the remaining steps.
    dict(type="WeightDecaySchedular", base_value=base_wd, final_value=final_wd,
         extend_scheduler=True, original_total_steps=original_total_steps),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="CheckpointSaver", save_freq=1),
]
