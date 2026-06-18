"""LArFormer Keypoint v2 — Phase 1: slice-level dense classifier.

A SEPARATE keypoint encoder–decoder (attempt 2; see
lartpc_data_prep/larformer_keypoint_v2/README.md). This config is the
slice-level half only:

  - own PTv3 backbone: encoder FROZEN (Sonata pretrain), decoder TRAINABLE
    (random-init, from scratch) — `freeze_backbone=True, unfreeze_decoder=True`.
  - tokenizer levels: voxel_4cm, ptv3_dec3 (~2cm), ptv3_dec2 (~1cm), spacepoint.
  - NO Mask2Former query decoder (`num_queries=0` → decoder-optional path).
  - two `level_keypoint_heads`:
      * ptv3_dec2 "object" : soft object/no-object (kpscores cols 1..5 unioned)
        — must become a reliable mixed-query source for Phase 2.
      * spacepoint "nu_vertex" : soft nu-vertex/no (kpscores col 0).

Targets are pooled per-token (max over member SPs) from the dataset's per-SP
`kpscores` Gaussian-proximity field (emit_keypoints=True). Trains STANDALONE on
the Stage-2 cache (no cascade yet). The cascade wrapper + per-particle query
decoder are Phases 2-3.
"""

_base_ = ["../_base_/default_runtime.py"]

from pointcept.models.LArFormer import trainer as _larformer_trainer_module
from pointcept.models.LArFormer import keypoint2_evaluator as _kp2_eval_module
del _larformer_trainer_module
del _kp2_eval_module

# =============================================================================
# Paths (laptop dev)
# =============================================================================
# 80k spacepoint capped cache
#CACHE_ROOT  = "/home/twongjirad/working/larbys/gen2/container_u22/Pointcept/exp/cache_stage12_devdata/"
CACHE_ROOT = "/mnt/ddrive/data/ub_on_tufts/devdata_cache_stage12_nocap"
TRAIN_ROOT  = f"{CACHE_ROOT}/train"
VAL_ROOT    = f"{CACHE_ROOT}/val"

sonata_pretrain_weight = (
    "/home/twongjirad/working/larbys/gen2/container_u22/Pointcept/"
    "sonata/lartpc_v6_h200_noghosts_pretrain_logspace_resume/model/epoch_42.pth"
)

# =============================================================================
# Constants
# =============================================================================
TOKEN_DIM            = 256
NUM_CLASSES          = 8          # unused (no query/cls heads) but plumbed
_PTV3_DEC_CHANNELS   = (64, 64, 128, 256)
BACKBONE_OUT_CH      = _PTV3_DEC_CHANNELS[0]   # 64 = dec0 width

coord_center = (125.0, 0.0, 518.0)
coord_scale  = 179.55
flash_backend = "flash_attn"

# =============================================================================
# Dataset
# =============================================================================
_kp_ds_common = dict(
    type="LArFormerStage12CacheDataset",
    coord_center=coord_center,
    coord_scale=coord_scale,
    emit_keypoints=True,          # surfaces per-SP kpscores (the head targets)
    keypoint_sigma_cm=3.0,
    recenter_to_centroid=True,
    source_set_filter="stage2_pass",
)

data = dict(
    num_classes=NUM_CLASSES,
    ignore_index=-1,
    names=["e", "gamma", "mu", "pi", "p", "other", "(unused)", "no_object"],
    train=dict(split="train", data_root=TRAIN_ROOT, min_spacepoints=20,
               loop=1, **_kp_ds_common),
    val=dict(split="val", data_root=VAL_ROOT, loop=1, **_kp_ds_common),
    test=dict(split="test", data_root=VAL_ROOT, min_spacepoints=20,
              loop=1, **_kp_ds_common),
)

# =============================================================================
# Levels — pure builders, NO mask/cls supervision (the dense keypoint heads
# carry the whole supervision load). scale_pattern is empty (no decoder).
# =============================================================================
kp_levels = [
    dict(name="voxel_4cm",
         builder="VoxelBuilder",
         builder_cfg=dict(voxel_size_cm=4.0, coord_scale=coord_scale)),
    dict(name="ptv3_dec3",
         builder="PTv3DecoderStageLevel",
         builder_cfg=dict(stage_key="dec3", in_dim=_PTV3_DEC_CHANNELS[3])),
    dict(name="ptv3_dec2",
         builder="PTv3DecoderStageLevel",
         builder_cfg=dict(stage_key="dec2", in_dim=_PTV3_DEC_CHANNELS[2])),
    dict(name="spacepoint",
         builder="SpacepointBuilder"),
]

model = dict(
    type="LArFormer",
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
            traceable=True,
            enc_mode=False,
            mask_token=False,
        ),
        head_in_channels=1088, head_hidden_channels=2048,
        head_embed_channels=256, head_num_prototypes=4096,
        num_global_view=2, num_local_view=6,
        up_cast_level=0,
    ),
    backbone_out_channels=BACKBONE_OUT_CH,
    backbone_weight=sonata_pretrain_weight,
    levels=kp_levels,
    scale_pattern=[],               # no decoder
    token_dim=TOKEN_DIM,
    num_queries=0,                  # decoder OFF — pure dense classifier
    num_classes=NUM_CLASSES,
    freeze_backbone=True,           # encoder frozen ...
    unfreeze_decoder=True,          # ... PTv3 decoder trainable (from scratch)
    ptv3_decoder_init_scale=0.01,
    enable_origin_head=False,
    capture_decoder_stages=True,    # ptv3_dec2/dec3 levels need the captures
    # ---- the delta: per-level keypoint classification heads --------------
    level_keypoint_heads=[
        dict(level="ptv3_dec2", name="object",
             kp_types=[1, 2, 3, 4, 5],     # track_start/end/shower/michel/delta
             hidden_dim=256, weight=1.0, pos_weight=20.0, pos_threshold=0.05),
        dict(level="spacepoint", name="nu_vertex",
             kp_types=[0],                  # nu_vertex
             hidden_dim=256, weight=1.0, pos_weight=50.0, pos_threshold=0.05),
    ],
)

# =============================================================================
# Trainer (evaluator TBD — Phase 1 watches train loss levelkp_object /
# levelkp_nu_vertex; a dense-head P/R + nu-vertex-resolution evaluator is the
# next step).
# =============================================================================
train = dict(type="LArFormerTrainer")

hooks = [
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="LArFormerLevelKeypointEvaluator",
         object_head="object", nu_vertex_head="nu_vertex",
         object_kp_types=[1, 2, 3, 4, 5],
         target_pos_thresh=0.5, score_thresh=0.5,
         nu_decode_thresh=0.3, nu_recall_cm=3.0,
         coord_scale=coord_scale,
         best_metric="val/object_ap"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="IterCheckpointSaver", save_iter_freq=5, keep_history=False),
    dict(type="SignalCheckpointHook", check_every_n_iter=30),
]

# =============================================================================
# Training loop knobs
# =============================================================================
weight = None
resume = False
save_path        = "exp/larformer_keypoint2"
epoch            = 50
eval_epoch       = 50
batch_size       = 10
batch_size_val   = 10
num_worker       = 4
num_worker_val   = 4
evaluate         = True
enable_amp       = False
amp_dtype        = "bfloat16"
empty_cache      = False
clip_grad        = 5.0
enable_wandb     = True
find_unused_parameters = True

skip_dataloader_on_resume = True
resume_seed_strategy = "per_resume"

# =============================================================================
# Optimizer / scheduler — PTv3 decoder + heads train from scratch.
# =============================================================================
base_lr = 1.0e-4
param_dicts = None
optimizer = dict(type="AdamW", lr=base_lr, weight_decay=0.01)
scheduler = dict(
    type="FlatWithDecayLR",
    mode="plateau",
    gamma=0.5,
    min_lr=5e-8,
    step_period_epochs=50,
    patience_epochs=3,
    min_delta=1e-4,
    cooldown_epochs=1,
    warmup_iters=200,
    warmup_start_lr=0.0,
    ema_alpha=0.3,
)
