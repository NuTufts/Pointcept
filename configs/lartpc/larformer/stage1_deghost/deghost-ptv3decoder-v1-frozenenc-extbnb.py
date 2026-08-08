"""Stage-1 deghoster V1 EXPERIMENT — full PTv3 DECODER on a frozen Sonata
encoder + per-point 2-class head, replacing LoRA + Linear(1232->2).

Motivation (label-ceiling study, 2026-08-03, job 2164256): the production
SonataLoRADeghostSegmentor sits 16-21 charge-percentage-points BELOW its own
hasmatch label ceiling (gamma 0.652 vs 0.815; e- 0.622 vs 0.813; e+ 0.589 vs
0.794 aggregate charge completeness on the pi0filter val/test set). A
higher-capacity head has real headroom to chase. (The labels themselves cap
at ~0.80-0.82 — LArMatch truth-matching misses soft shower charge — that is
a separate label-generation work item.)

Design (Option B of the deghoster plan; template =
configs/lartpc/semseg/archive/semseg-sonata-v1m1-lartpc-v5-decoder-finetune.py):
  - DefaultSegmentorV2 over a BARE PT-v3m2 with enc_mode=False: the learned
    decoder (dec_channels (64,64,128,256), dec0 = 64-ch per-point output)
    replaces the 1232-ch up_cast skip-concat the LoRA model classifies on.
  - Encoder + stem FROZEN via param_dicts lr=0.0 (DefaultSegmentorV2's
    freeze_backbone flag is all-or-nothing and would freeze the decoder
    too). NOTE: lr=0 params still receive gradients (compute cost) — this
    is the template's proven trade-off, accepted for the experiment.
  - Pretrain: v7 extbnb-larmatch Sonata (the GHOST-AWARE, real-data
    pretrain the production deghoster uses — NOT the no-ghost v6), loaded
    via SonataFinetuneCheckpointLoader(use_teacher=True) which remaps
    teacher.backbone.* -> backbone.*. Decoder + seg head start random
    (Sonata pretrains are enc_mode=True; no decoder weights exist).
  - EVERYTHING ELSE (dataset, HasmatchAsGhost labels real=0/ghost=1,
    Focal+Lovasz, BiasedSphereCrop + augmentation stack, batch size, file
    lists) is copied VERBATIM from the production LoRA config
    (lorafinetune-sonata-v1m1-lartpc-v6-deghost-extbnb-larmatch.py) so the
    comparison isolates exactly one variable: the classifier head path.

Downstream: emits `seg_logits`, so CascadedSlicer consumes it through the
same branch as the LoRA deghoster with deghoster_class_index_real=0
unchanged — swap deghoster_cfg + deghoster_weight only.

Decision metric: after training, swap into the cascade and re-run
particle_slice_completeness.py — target is the post-deghost column
(gamma 0.652 baseline -> toward the 0.815 ceiling).

RESUME NOTE: fresh start loads `weight` (the Sonata pretrain) via
SonataFinetuneCheckpointLoader. For mid-run resume, pass
`--options weight=<save_path>/model/model_last.pth resume=True` — the
Sonata loader will find no teacher.backbone.* keys in a training
checkpoint (harmless no-op) and the standard CheckpointLoader in the hook
list performs the actual resume.
"""

_base_ = ["../../../_base_/default_runtime.py"]

# ============================================================================
# Run knobs
# ============================================================================
batch_size       = 96
batch_size_val   = 48
num_worker       = 22
num_worker_val   = 20
mix_prob         = 0.0
empty_cache      = False
enable_amp       = False
amp_dtype        = "bfloat16"
enable_wandb     = True
wandb_project    = "pointcept-deghost-ptv3dec"
save_path        = "exp/deghost_ptv3decoder_v1_frozenenc_extbnb"
# 20-epoch OneCycle horizon (LoRA ran 50; the from-scratch decoder at
# 5e-4 should converge much faster — watch val per-class IoU and extend
# only if still improving; epoch/eval_epoch move together).
epoch            = 20
eval_epoch       = 20
base_lr          = 5e-4
clip_grad        = 1.0
find_unused_parameters = True

skip_dataloader_on_resume = True
resume_seed_strategy = "per_resume"

# xformers, NOT flash_attn: this model trains in fp32 (enable_amp=False) and
# flash_attn silently produces garbage/NaN in fp32 — the exact failure
# LArFormer._ensure_decoder_fp32_forward() exists to patch (it switches its
# trainable PT-v3m2 decoder blocks to xformers). DefaultSegmentorV2 has no
# such runtime patch, so the whole backbone runs xformers here. Production
# run 2169933 NaN'd on every batch with flash_attn before this fix.
flash_backend = "xformers"

# ============================================================================
# Geometry / normalization — identical to the LoRA deghost config
# ============================================================================
grid_size    = 0.25
coord_scale  = 1036.0 * 3**0.5 / 2.0 / 5.0
_scaled_grid_size = 0.0013932197615579773   # grid_size / coord_scale
_jitter_sigma     = 0.0003483049403894943   # _scaled_grid_size / 4
_strength_jitter_sigma = 0.05
_strength_jitter_clip  = 0.05

max_points_spherecrop    = 10240
min_points_spherecrop    = 4096
biased_spherecrop_radius = 20.0

# DATASET CHANGE vs the LoRA config (2026-08-03): the prod4
# v2_expandedclasses dataset the LoRA deghoster trained on has been DELETED
# from ub_on_tufts/hdf5 (smoke 2166864: 0 of 390k sampled files exist).
# Trains instead on the v3_larmatch LANTERN merged-h5 lists (the same files
# the slicer trains on; verified to carry entry_0/triplet_data/{pos,
# hasmatch, ssnet_label, pixval, lm_score, edep} + mc_particle_tree/
# nu_vertices — everything LArTPCDataset + HasmatchAsGhost + BiasedSphereCrop
# need). The LoRA-vs-decoder comparison is therefore NOT matched on training
# data (unavoidable); the decision metric — post-deghost per-particle
# completeness through the cascade on the pi0filter valtest set — remains
# apples-to-apples.
TRAIN_FILE_LIST = "/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/lartpc_data_prep/lantern_scripts/h5lists/h5list_mcall_lantern_train.txt"
VAL_FILE_LIST   = "/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/lartpc_data_prep/lantern_scripts/h5lists/h5list_mcall_lantern_val.txt"

# v7 extbnb-larmatch Sonata pretrain (ghost-aware; real off-beam data).
# Consumed by SonataFinetuneCheckpointLoader via the top-level `weight` key.
weight = "/cluster/tufts/wongjiradlab/twongj01/mphys/Pointcept/sonata/lartpc_v7_h200_extbnb_larmatch_run1/model/epoch_18.pth"

# ============================================================================
# Model — DefaultSegmentorV2 + bare PT-v3m2 with the decoder ON
# ============================================================================
model = dict(
    type="DefaultSegmentorV2",
    num_classes=2,
    # dec_channels[0] = 64 → per-point feature width at full resolution.
    backbone_out_channels=64,
    backbone=dict(
        type="PT-v3m2",
        in_channels=6,                       # coord(3) + strength(3), as pretrain
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(3, 3, 3, 9, 3),
        enc_channels=(48, 96, 192, 384, 512),
        enc_num_head=(3, 6, 12, 24, 32),
        enc_patch_size=(256, 256, 256, 256, 256),
        # Decoder: PT-v3m2 defaults, stated explicitly for the record.
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(48, 48, 48, 48),
        mlp_ratio=4, qkv_bias=True, qk_scale=None,
        attn_drop=0.0, proj_drop=0.0, drop_path=0.0,
        shuffle_orders=True, pre_norm=True,
        enable_rpe=False, enable_flash=True, flash_backend=flash_backend,
        upcast_attention=False, upcast_softmax=False,
        traceable=True,
        enc_mode=False,                      # the whole point: decoder ON
        mask_token=False,
    ),
    criteria=[
        dict(type="FocalLoss", gamma=2.0, alpha=0.5,
             loss_weight=1.0, ignore_index=-1, reduction="mean"),
        dict(type="LovaszLoss", mode="multiclass",
             loss_weight=1.0, ignore_index=-1),
    ],
    # Freezing handled by param_dicts lr=0.0 below, NOT this flag (which
    # would freeze the decoder too).
    freeze_backbone=False,
)

# ============================================================================
# Optimizer / scheduler — encoder frozen via lr=0.0 param groups
# ============================================================================
# Group 0 (default): backbone.dec.* + seg_head  → base_lr
# Group 1: backbone.enc.*                        → lr 0.0 (frozen)
# Group 2: backbone.embedding.*                  → lr 0.0 (frozen)
# (AdamW's decoupled weight decay scales by lr, so lr=0 groups also see no
# decay. no_decay_on_1d_and_embeddings does not compose with param_dicts —
# accepted for this experiment.)
param_dicts = [
    dict(keyword="backbone.enc", lr=0.0),
    dict(keyword="backbone.embedding", lr=0.0),
]
optimizer = dict(type="AdamW", lr=base_lr, weight_decay=0.01)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[base_lr, 0.0, 0.0],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=25.0,
    final_div_factor=100.0,
)

# ============================================================================
# Dataset — copied VERBATIM from the production LoRA deghost config
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
            dict(
                type="LogTransform",
                min_val=0.01,
                max_val=1000.0,
                log=True,
                keys=("strength",),
            ),
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
            dict(type="RandomRotate", angle=[-1, 1], axis="z",
                 center=[0, 0, 0], p=0.8),
            dict(type="RandomRotate", angle=[-1, 1], axis="x",
                 center=[0, 0, 0], p=0.8),
            dict(type="RandomRotate", angle=[-1, 1], axis="y",
                 center=[0, 0, 0], p=0.8),
            dict(type="RandomFlip", p=0.5, axes=("x", "y", "z")),
            dict(
                type="RandomJitter",
                sigma=_jitter_sigma,
                clip=_scaled_grid_size,
                keys=("coord",),
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                # NO segment_counts: the dataset builds it in the 9-class
                # ssnet space BEFORE HasmatchAsGhost rewrites segment to
                # 2-class, and FocalLoss's count-weighting asserts on the
                # mismatch (smoke 2169779). Without it FocalLoss uses its
                # plain alpha/gamma path; Lovasz handles IoU-level balance.
                keys=("coord", "grid_coord", "segment"),
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
                # Uniform crops in val — no nu-anchoring, so val IoU is not
                # inflated by dense-region bias (LoRA config convention).
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
                # NO segment_counts: the dataset builds it in the 9-class
                # ssnet space BEFORE HasmatchAsGhost rewrites segment to
                # 2-class, and FocalLoss's count-weighting asserts on the
                # mismatch (smoke 2169779). Without it FocalLoss uses its
                # plain alpha/gamma path; Lovasz handles IoU-level balance.
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("coord", "strength"),
            ),
        ],
        test_mode=False,
    ),
    test=dict(),  # SemSegTester unused for now; per-particle completeness
                  # via the cascade is the decision metric.
)

# ============================================================================
# Hooks
# ============================================================================
hooks = [
    # Fresh start: remap teacher.backbone.* -> backbone.* from `weight`
    # (harmless no-op on a resume checkpoint — see RESUME NOTE).
    dict(type="SonataFinetuneCheckpointLoader", use_teacher=True),
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator", write_cls_iou=True),
    dict(type="CheckpointSaver", save_freq=1),
    dict(type="IterCheckpointSaver", save_iter_freq=500, keep_history=False),
    dict(type="SignalCheckpointHook", check_every_n_iter=30),
]
