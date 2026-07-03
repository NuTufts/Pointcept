"""LArFormer keypoint — Phase 1: dense per-SP keypoint-score head.

Trains a LArFormerKeypoint (frozen Sonata backbone + KeypointScoreHead) to
regress the per-spacepoint `(N, 6)` keypoint proximity-score field
(`kpscores`) on the nu-candidate spacepoints from the stage-12 cache.

See lartpc_data_prep/larformer_keypoint/README.md (Phase 1). The overfit
milestone is validated by
lartpc_data_prep/larformer_keypoint/train_phase1_overfit.py; this config is
the scale-up training run.

Run:
    python tools/train.py --config-file configs/lartpc/larformer/stage4_keypoint/archive/larformer-keypoint-v1.py \
        --num-gpus 1 --options save_path=exp/larformer_keypoint_v1

NOTE: the cache must be augmented with the mckeypoints group first
(tools/augment_stage12_cache_keypoints.py) — the dev cache already is.
"""

_base_ = ["../../../../_base_/default_runtime.py"]

# Side-effect: register LArFormerTrainer (provides larformer_collate).
from pointcept.models.LArFormer import trainer as _larformer_trainer_module
del _larformer_trainer_module

# =============================================================================
# Paths
# =============================================================================
TRAIN_CACHE_ROOT = "/home/twongjirad/working/larbys/gen2/container_u22/Pointcept/exp/cache_stage12_devdata/train/"
VAL_CACHE_ROOT   = "/home/twongjirad/working/larbys/gen2/container_u22/Pointcept/exp/cache_stage12_devdata/val/"

# Sonata-v1m1 pretrain (frozen backbone). Local path active for laptop runs;
# swap to the cluster path when training there.
sonata_pretrain_weight = (
    "/home/twongjirad/working/larbys/gen2/container_u22/Pointcept/"
#   "/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/"
    "sonata/lartpc_v6_h200_noghosts_pretrain_logspace_resume/model/epoch_42.pth"
)

coord_center = (125.0, 0.0, 518.0)
coord_scale  = 179.55
flash_backend = "flash_attn"

N_KEYPOINT_TYPES = 6
BACKBONE_OUT_CH  = 1232      # Sonata-v1m1 + PT-v3m2 head (up_cast_level=4)

# =============================================================================
# Dataset — stage-12 cache with keypoints (emit_keypoints=True)
# =============================================================================
_dataset_common = dict(
    type="LArFormerStage12CacheDataset",
    coord_center=coord_center,
    coord_scale=coord_scale,
    emit_keypoints=True,
    keypoint_sigma_cm=3.0,
    keypoint_score_threshold=0.01,
    n_keypoint_types=N_KEYPOINT_TYPES,
    # Train on the inference-realistic nu-candidate set. Switch to
    # "stage2_random_tau" for τ augmentation, or "union" for max coverage.
    source_set_filter="stage2_pass",
    recenter_to_centroid=False,
    min_spacepoints=16,
)

data = dict(
    num_classes=N_KEYPOINT_TYPES,
    ignore_index=-1,
    names=["nu_vertex", "track_start", "track_end", "shower", "michel", "delta"],
    train=dict(split="train", data_root=TRAIN_CACHE_ROOT, loop=1, **_dataset_common),
    val=dict(split="val", data_root=VAL_CACHE_ROOT, loop=1, **_dataset_common),
    test=dict(split="test", data_root=VAL_CACHE_ROOT, loop=1, **_dataset_common),
)

# =============================================================================
# Model — LArFormerKeypoint (frozen Sonata backbone + dense score head)
# =============================================================================
model = dict(
    type="LArFormerKeypoint",
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
            mlp_ratio=4, qkv_bias=True, qk_scale=None,
            attn_drop=0.0, proj_drop=0.0, drop_path=0.0,
            shuffle_orders=False, pre_norm=True,
            enable_rpe=False, enable_flash=True, flash_backend=flash_backend,
            upcast_attention=False, upcast_softmax=False,
            traceable=True, enc_mode=True, mask_token=False,
        ),
        head_in_channels=1088, head_hidden_channels=2048,
        head_embed_channels=256, head_num_prototypes=4096,
        num_global_view=2, num_local_view=6,
        up_cast_level=4,
    ),
    backbone_out_channels=BACKBONE_OUT_CH,
    backbone_weight=sonata_pretrain_weight,
    n_keypoint_types=N_KEYPOINT_TYPES,
    freeze_backbone=True,
    head_hidden_dim=512,
    head_dropout=0.0,
    loss_kind="mse",          # MSE on sigmoid(logits); reaches 0 at perfect fit
    pos_weight=50.0,          # up-weight points within the Gaussian peaks
    pos_threshold=0.05,
    weight_keypoint=1.0,
    # PPN/VoteNet offset head: per-(SP,type) vote toward the nearest keypoint,
    # supervised only at points within the Gaussian peaks. Lets keypoints be
    # localized BETWEEN measured points. predicted_kp = coord_norm + offset.
    enable_offset_head=True,
    weight_offset=1.0,
    offset_supervision_threshold=0.05,
    coord_scale=coord_scale,
)

# =============================================================================
# Training loop
# =============================================================================
weight        = None
save_path     = "exp/larformer_keypoint_v1"
epoch         = 100
eval_epoch    = 100
batch_size    = 4
batch_size_val = 4
num_worker    = 8
num_worker_val = 4
# No keypoint evaluator yet (Phase 4 — per-type recall within distance). Train
# loss + the mse_pos diagnostic are the signal for now.
evaluate      = False
enable_amp    = False
empty_cache   = False
clip_grad     = 1.0
enable_wandb  = False
find_unused_parameters = True

# Trainer: use larformer_collate (flat per-SP concat + nested per-event lists).
train = dict(type="LArFormerTrainer")

# Minimal hooks — drop the SemSeg/Precise evaluators (they assume a
# segmentation model output, not this keypoint model).
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="CheckpointSaver", save_freq=None),
]

# Optimizer / scheduler (Adam on the head; backbone is frozen).
optimizer = dict(type="Adam", lr=1e-3, weight_decay=0.0)
scheduler = dict(
    type="OneCycleLR",
    max_lr=1e-3,
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)
