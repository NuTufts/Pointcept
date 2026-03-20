"""
SONATA Pretraining for LArTPC Point Clouds

Self-supervised pretraining using masked prediction, cross-view consistency,
and local-to-global matching. Inspired by SONATA and Panda (arXiv:2512.01324).

Adapted for LArTPC physics constraints:
  - No rotations (tomographic projection geometry must be preserved)
  - No x-flip (diffusion effects break x-symmetry along drift direction)
  - Y-flip and z-flip allowed (detector symmetries)
  - Local 3D crops are valid (spacepoints already reconstructed from wire planes)

Features (6 channels total):
  - strength (3): Pixel values from u, v, y wire plane images (pixval)
  - coords (3): spacepoint location

Coordinates:
  - units in data are in cm. We do not need to normalize these values
  - detector spans in x=[0,256],y=[-117,117],z=[0,1040]
  - wire pitch is 0.3 cm and drift spacing is about the same.
  - use grid size of 0.25 cm

v6 changes:
  - apply LogTransform to input ADC values
  - provide coordinates to features like Panda does
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

_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 64  # Adjust based on GPU memory; LArTPC events can be large
batch_size_val = 8   # Batch size for validation (used by PretrainEvaluator)
num_worker = 32
mix_prob = 0
clip_grad = 1.0
empty_cache = False
enable_amp = True
evaluate = False  # Disabled - PretrainEvaluator hook is commented out
find_unused_parameters = False

flash_backend='flash_attn' # default backend
amp_dtype = "bfloat16" # use this for default 'flash_attn' backend

#flash_backend='xformers'   # backend needed to run on P100
#amp_dtype = "float16"      # use this with xformer backend on P100

enable_wandb = True
wandb_project = "pointcept"
save_path = "sonata/lartpc_v6_h200_extbnb_run1"

TRAIN_FILE_LIST="/cluster/tufts/wongjiradlab/hmcgui01/mphys/Pointcept/lartpc_data_prep/combined_extbnb_run3_g1_g2_shuffled_training.txt"
VAL_FILE_LIST="/cluster/tufts/wongjiradlab/hmcgui01/mphys/Pointcept/lartpc_data_prep/combined_extbnb_run3_g1_g2_shuffled_validation.txt"
true_points_only=False

max_points_per_view=20480
max_points_spherecrop=20480
min_points_spherecrop=4096
biased_spherecrop_radius=20.0

# scheduler settings
# original run
epoch = 50
eval_epoch = 50
base_lr = 0.002
# extended run
#epoch = 50
#eval_epoch = 50
#base_lr = 0.0001

lr_final_div_factor=1000.0 # original
lr_pct_start=0.06 # original
#lr_final_div_factor=100.0 # extended
#lr_pct_start=0.0 # extended par

lr_decay = 0.9  # layer-wise lr decay
base_wd  = 0.04
final_wd = 0.2

# LArTPC scale parameters
# Detector: ~1036 cm largest dimension, 0.3 cm wire pitch / time sampling
grid_size = 0.25 # cm
jitter_sigma=0.05 # cm (reduced)
jitter_clip=0.25  # cm (reduced)
wire_scale=1.0/3456.0 # normalize the wire indices which range from 0-3456
coord_scale=1036.0 * 3**0.5 / 2.0 / 5.0
scaled_grid_size = grid_size/coord_scale
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
        in_channels=6,  # strength only (3: pixel values from u,v,y planes) - no wire coords to avoid geometric trap
        order=("z", "z-trans","hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(3, 3, 3, 9, 3), # voxel size at each encoder layer: (0.25, 0.5, 1.0, 2.0, 4.0);
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
    # With enc_channels=(48, 96, 192, 384, 512) and up_cast_level=2:
    # Sum of channels from levels 2, 3, 4: 192 + 384 + 512 = 1088
    head_in_channels=1088,
    head_hidden_channels=2048,
    head_embed_channels=256,   # Sonata originally 256
    head_num_prototypes=4096,  # Sonata originally 2048, increased because semantic imbalance (muons vs non-muons)

    # View configuration
    num_global_view=2,   # Two global views with different augmentations
    num_local_view=6,    # Must match local_view_num in MultiViewGenerator

    # Masking schedule - scaled for LArTPC
    # Patch sizes in coordinate units
    # 2cm -> 15cm in real detector coordinates, following Panda
    mask_size_start=2.0/coord_scale,    # Start with ~2cm patches
    mask_size_base=15.0/coord_scale,    # End with ~15cm patches
    mask_size_warmup_ratio=0.06,
    mask_ratio_start=0.2,   # Initial fraction of patches masked
    mask_ratio_base=0.7,    # End fraction of patches masked after current_epoch=epoch*mask_ratio_warmup_ratio
    mask_ratio_warmup_ratio=0.06,
    mask_jitter=0.125/coord_scale, # half of grid_size 

    # Temperature schedule
    teacher_temp_start=0.04,
    teacher_temp_base=0.07,
    teacher_temp_warmup_ratio=0.06,
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
    match_max_r=16.0*grid_size/coord_scale,  # ~5cm matching radius

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
    div_factor=100.0,
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
        anchor_points_key=None,
        anchor_pdf_key=None,
        radius=biased_spherecrop_radius, # in detector distance coordinates
        point_max=max_points_spherecrop, 
        point_min=min_points_spherecrop, 
        prob_random=1.0,
        max_retries=100,
        fallback_to_random=True),
    # Center and scale spacepoints in detector coordinates
    dict(type="NormalizeCoord", center=[125.0, 0.0, 1036.0/2.0], scale=coord_scale),
    # Log-transform pixel values
    dict(
        type="LogTransform",
        min_val=0.01,
        max_val=1000.0,
        log=True,
        keys=("strength",),
    ),
    # Keep original coordinates for cross-view matching
    dict(type="Copy", keys_dict={"coord": "origin_coord"}),
    # Generate multi-scale views
    dict(
        type="MultiViewGenerator",
        view_keys=("coord", "origin_coord", "strength"),  # No wire coords (color) to avoid geometric trap
        # Global views: see most/all of the event
        global_view_num=2,
        global_view_scale=(0.4, 1.0),
        # Local views: crops of 3D regions (valid for spacepoints)
        local_view_num=6,
        local_view_scale=(0.1, 0.4),
        # Shared transforms for all views
        global_shared_transform=[
            dict(
                type="MultiplicativeRandomJitter",
                sigma=0.05,
                clip=0.05,
                keys=("strength"),
                p=0.8,
            ),
        ],
        # Per-view augmentations
        # The transforms on the spacepoints do not respect the pixel values from the wires
        # So the network cannot learn the detector/readout physics.
        # Instead, the hope is that relative positions provide info on per-point distances,
        #   which does provide some info on relative wire plane dE/dx.  Though the
        #   absolute dE/dx for one wire will also be broken.
        global_transform=[
            dict(type="CenterShift", apply_z=False, axes=("x", "y", "z")),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.8),
            dict(type="RandomRotate", angle=[-1, 1], axis="x", center=[0, 0, 0], p=0.8),
            dict(type="RandomRotate", angle=[-1, 1], axis="y", center=[0, 0, 0], p=0.8),
            dict(type="RandomFlip", p=0.5, axes=("x", "y", "z")),
            dict(
                type="RandomJitter",
                sigma=scaled_grid_size / 4,
                clip=scaled_grid_size,
                keys=("coord",),
            ),
        ],
        # Local view transforms (same physics constraints)
        local_transform=[
            dict(type="CenterShift", apply_z=False, axes=("x", "y", "z")),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.8),
            dict(type="RandomRotate", angle=[-1, 1], axis="x", center=[0, 0, 0], p=0.8),
            dict(type="RandomRotate", angle=[-1, 1], axis="y", center=[0, 0, 0], p=0.8),
            dict(type="RandomFlip", p=0.5, axes=("x", "y", "z")),
            dict(
                type="RandomJitter",
                sigma=scaled_grid_size / 4,
                clip=scaled_grid_size,
                keys=("coord",),
            ),
        ],
        max_size=max_points_per_view,  # Max points per view
    ),
    dict(type="ToTensor"),
    dict(type="Update", keys_dict={"grid_size": scaled_grid_size}),
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
        global_feat_keys=("global_coord","global_strength",),
        local_feat_keys=("local_coord","local_strength",),
    ),
]

# Validation transform for PretrainEvaluator (standard point cloud, not multi-view)
val_transform = [
    # Voxelize to grid
    dict(
        type="GridSample",
        grid_size=grid_size,
        hash_type="fnv",
        mode="train",
        return_grid_coord=True,
        return_inverse=True,
    ),
    # Limit points per sample (no neutrino vertices in real data)
    dict(type="BiasedSphereCrop",
        anchor_points_key=None,
        anchor_pdf_key=None,
        radius=biased_spherecrop_radius,
        point_max=max_points_spherecrop,
        point_min=min_points_spherecrop,
        prob_random=1.0,
        max_retries=100,
        fallback_to_random=True),
    # Center and scale spacepoints in detector coordinates
    dict(type="NormalizeCoord", center=[125.0, 0.0, 1036.0/2.0], scale=coord_scale),
    # Log-transform pixel values
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
        keys=(
            "coord",
            "grid_coord",
            "segment",
            "inverse",
        ),
        feat_keys=("strength",),
    ),
]

# SSNet class names for linear probing evaluation
ssnet_class_names = [
    "electron",
    "muon",
    "pion",
    "proton",
    "gamma",
    "michel",
    "delta",
    "led",
]

data = dict(
    num_classes=8,  # SSNet classes for PretrainEvaluator linear probing
    ignore_index=-1,
    names=ssnet_class_names,
    train=dict(
        type=dataset_type,
        split="train",
        data_list_file=TRAIN_FILE_LIST,
        use_reco_coords=True,
        use_edep_as_strength=True,
        label_mode="ssnet",
        coord_scale=1.0,
        include_ghosts=True,
        exclude_other=False,
        transform=transform,
        test_mode=False,
        loop=1,
        true_points_only=true_points_only,
        data_only=True,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_list_file=VAL_FILE_LIST,
        use_reco_coords=True,
        use_edep_as_strength=True,
        label_mode="ssnet",  # Use SSNet labels for linear probing evaluation
        coord_scale=1.0,
        include_ghosts=True,  # SSNet doesn't include ghosts
        exclude_other=False,
        transform=val_transform,
        test_mode=False,
        loop=1,
        true_points_only=true_points_only,
        data_only=True,
    ),
)

# Original training parameters (for extending scheduler calculations)
# These are needed to compute where the original schedule was at the resume point
original_epochs=50
original_steps_per_epoch = 8125  # For expanded data set
original_total_steps = original_epochs * original_steps_per_epoch  # = 110500

# Hooks for pretraining
hooks = [
    # extend_scheduler=True: When resuming, creates a new scheduler with the updated
    # total_steps (based on eval_epoch=200) and advances it to the current step.
    # This allows extending training beyond the original epoch count while
    # preserving optimizer state (Adam momentum, etc.)
    # CRITICAL: This also restores the scheduler's LR values after loading optimizer state.
    dict(type="CheckpointLoader", extend_scheduler=False),
    dict(type="ModelHook"),  # Calls model.before_train(), before_step(), after_step()
    # extend_scheduler=True: Computes the WD that the original schedule had at the resume
    # point, then creates a new schedule from that WD to final_wd over the remaining steps.
    dict(type="WeightDecaySchedular", base_value=base_wd, final_value=final_wd,
         extend_scheduler=False, original_total_steps=original_total_steps),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="CheckpointSaver", save_freq=1),
    # === Stability Monitoring ===
    # GradScaler monitor - detect overflow events
    dict(
        type="GradScalerMonitor",
        log_frequency=10,
        prefix="grad_scaler",
        warn_on_overflow=True,
        warn_on_low_scale=1.0,
    ),
    # Adam state monitor - track momentum/variance growth
    dict(
        type="AdamStateMonitor",
        log_frequency=100,        # Every 100 steps (less frequent, more expensive)
        prefix="adam_state",
        track_layers=True,        # Group by layer type (attention, mlp, etc.)
        track_histograms=False,   # Set True for detailed debugging (expensive)
        histogram_frequency=500,  # If histograms enabled, log every 500 steps
    ),
    # =========================================================================
    # Monitoring Hooks (adapted from particle-imaging-models)
    # =========================================================================
    # PretrainEvaluator: Linear probing evaluation during pretraining
    # Trains a linear classifier on frozen backbone features to evaluate
    # representation quality during self-supervised pretraining
    # Uses SSNet labels: electron, muon, pion, proton, gamma, michel, delta, led
    # Note: does not work currently on the P100 machines
    #dict(
    #    type="PretrainEvaluator",
    #    write_cls_iou=True,
    #    every_n_steps=10,  # Run evaluation every 1000 steps
    #    max_train_events=48,  # Number of events for training linear probe
    #    max_test_events=48,   # Number of events for testing linear probe
    #    class_names=ssnet_class_names,  # SSNet classes for evaluation
    #    prefix="",
    #),
    # PrototypeUsageLogger: Monitor prototype utilization in Sonata
    # Tracks how many prototypes are being used, unused percentage,
    # and entropy of assignment distribution
    dict(
        type="PrototypeUsageLogger",
        log_frequency=10,  # Log every 10 steps
        prefix="prototypes",
    ),
    # FeatureStdMonitor: Monitor feature standard deviation
    # Tracks feature collapse by monitoring std of student/teacher features
    dict(
        type="FeatureStdMonitor",
        log_frequency=10,  # Log every 10 steps
        prefix="feature_std",
        monitor_student=True,
        monitor_teacher=True,
        track_channels=False,  # Set True for per-channel stats
    ),
]
