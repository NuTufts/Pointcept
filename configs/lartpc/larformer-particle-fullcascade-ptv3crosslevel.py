"""LArFormer full-cascade INFERENCE config — ptv3crosslevel stage-3.

Derived from `larformer-particle-v1-cached-ptv3crosslevel.py` (the trained
Stage-3 particle segmenter) wrapped into a `CascadedParticleSegmenter` so it
runs end-to-end on RAW per-event merged_h5 (the output of
`lartpc_data_prep/larformer_scripts/convert_dlmerged_to_larformer_h5.py`).

Used by:
    tools/run_larformer_stage3_inference.py --input-mode full-cascade

Cascade composition (all frozen at inference):
    Stage 1  SonataLoRADeghostSegmentor   (per-SP real/ghost)
    Stage 2  CascadedSlicer's slicer       (ptv3hybrid crosslevel)
    Stage 3  LArFormer particle segmenter  (ptv3crosslevel — THIS config's
             `particle_segmenter`, loads model_iter_98652.pth)

CHECKPOINTS are read from environment variables so the bash .conf can drive
them; each falls back to a documented default. Override in the dataset .conf:

    LARFORMER_DEGHOSTER_CKPT
    LARFORMER_SLICER_CKPT        <-- see NOTE below
    LARFORMER_PARTICLE_CKPT
    LARFORMER_SONATA_PRETRAIN

NOTE on the slicer checkpoint: the Stage-3 weights (model_iter_98652.pth)
were trained on a Stage-1+2 cache built with a ptv3crosslevel slicer at
iter 75750 (see exp/cache_stage12_ptv3crosslevelslicer_iter_75750/). For the
most faithful cascade, point LARFORMER_SLICER_CKPT at THAT slicer
checkpoint. The default below is the best ptv3crosslevel slicer present in
this tree; confirm it matches before trusting absolute numbers.

The trained Stage-3 checkpoint is a STANDALONE LArFormer (un-prefixed keys).
run_stepB_cascade_wconfig.sh re-prefixes it to `particle_segmenter.*` before
passing it as --weights (the cascaded-slicer half loads via this config's
`cascaded_slicer_weight` in CascadedParticleSegmenter.__init__).
"""

import os

_base_ = ["../_base_/default_runtime.py"]

# Side-effect registrations.
from pointcept.models.LArFormer import trainer as _t
from pointcept.models.LArFormer import particle_evaluator as _pe
del _t, _pe

_REPO = "/home/twongjirad/working/larbys/gen2/container_u22/Pointcept"


def _env(name, default):
    v = os.environ.get(name, "").strip()
    return v if v else default


# =============================================================================
# Checkpoints (env-overridable)
# =============================================================================
deghoster_weight = _env(
    "LARFORMER_DEGHOSTER_CKPT",
    f"{_REPO}/sonata/lora_deghost_v6_hasmatch/model/epoch_30.pth")

sonata_pretrain_weight = _env(
    "LARFORMER_SONATA_PRETRAIN",
    f"{_REPO}/sonata/lartpc_v6_h200_noghosts_pretrain_logspace_resume/"
    "model/epoch_42.pth")

# Stage-2 slicer ckpt — SEE NOTE in the module docstring.
cascaded_slicer_weight = _env(
    "LARFORMER_SLICER_CKPT",
    f"{_REPO}/exp/larformer_slicer_v1_cascaded_ptv3hybrid_crosslevel_"
    "nonzeroinit_maskdn_noamp/model/model_ptv3crosslevel_iter_75750.pth")

# Stage-3 particle segmenter trained ckpt (informational — the actual load
# happens via --weights after run_stepB re-prefixes it; recorded here so the
# config is self-documenting).
particle_segmenter_weight = _env(
    "LARFORMER_PARTICLE_CKPT",
    f"{_REPO}/exp/larformer_particle_v1_cached_ptv3crosslevel_smallbatch_"
    "lr1e4_bugfixed/model_iter_98652.pth")

# =============================================================================
# Geometry / shapes
# =============================================================================
coord_center = (125.0, 0.0, 518.0)
coord_scale  = 179.55
flash_backend = "flash_attn"

STAGE3_NUM_QUERIES = 32
STAGE3_NUM_CLASSES = 8           # e±, γ, μ±, π±, p, other, (unused), no_object
STAGE3_TOKEN_DIM   = 256
STAGE3_MASK_PROB_THRESHOLD        = 0.5
STAGE3_RECENTER_TO_SLICE_CENTROID = True

_PTV3_DEC_CHANNELS    = (64, 64, 128, 256)
STAGE3_BACKBONE_OUT_CH = _PTV3_DEC_CHANNELS[0]      # 64 = dec0 width

_SLICER_PTV3_DEC_CHANNELS    = (64, 64, 128, 256)
slicer_backbone_out_channels = _SLICER_PTV3_DEC_CHANNELS[0]
slicer_token_dim = 256

USE_SINUSOIDAL_POS_EMB = False

# =============================================================================
# Dataset — raw merged_h5 via LArFormerDataset. The inference tool forces
# gt_source="particle" (or disables GT with --no-gt for real data).
# data_list_file is overridden on the command line by run_stepB.
# =============================================================================
_dataset_common = dict(
    type="LArFormerDataset",
    coord_center=coord_center,
    coord_scale=coord_scale,
    # Real data (no MC truth) sets LARFORMER_GT_SOURCE=deghost in its .conf
    # and passes --no-gt; sim leaves the default "particle".
    gt_source=_env("LARFORMER_GT_SOURCE", "particle"),
    emit_fragments=False,
    merge_nu_slices=True,
    lm_score_aug_low=0.0,
    lm_score_aug_high=0.0,
    lm_score_val_threshold=0.0,
    wire_scale=1.0 / 3456.0,
    min_fragment_points_post_filter=50,
)
data = dict(
    num_classes=STAGE3_NUM_CLASSES,
    ignore_index=-1,
    names=["e", "gamma", "mu", "pi", "p", "other", "(unused)", "no_object"],
    train=dict(split="train", data_root="/", data_list_file="PLACEHOLDER",
               loop=1, max_spacepoints=100_000, **_dataset_common),
    val=dict(split="val", data_root="/", data_list_file="PLACEHOLDER",
             loop=1, max_spacepoints=150_000, **_dataset_common),
    test=dict(split="test", data_root="/", data_list_file="PLACEHOLDER",
              loop=1, max_spacepoints=None, **_dataset_common),
)

# =============================================================================
# Stage 1 — LoRA deghoster (verbatim from larformer-particle-v1.py)
# =============================================================================
deghoster_cfg = dict(
    type="SonataLoRADeghostSegmentor",
    backbone_out_channels=1232,
    lora_rank=16, lora_alpha=32.0, lora_dropout=0.05,
    lora_target_modules=["qkv", "proj"],
    freeze_backbone_non_lora=True,
    ghost_class_index=1,
    backbone=dict(
        type="Sonata-v1m1",
        backbone=dict(
            type="PT-v3m2", in_channels=6,
            order=("z", "z-trans", "hilbert", "hilbert-trans"),
            stride=(2, 2, 2, 2),
            enc_depths=(3, 3, 3, 9, 3),
            enc_channels=(48, 96, 192, 384, 512),
            enc_num_head=(3, 6, 12, 24, 32),
            enc_patch_size=(256, 256, 256, 256, 256),
            mlp_ratio=4, qkv_bias=True, qk_scale=None,
            attn_drop=0.0, proj_drop=0.0, drop_path=0.3,
            shuffle_orders=True, pre_norm=True,
            enable_rpe=False, enable_flash=False, flash_backend=flash_backend,
            upcast_attention=False, upcast_softmax=False,
            traceable=True, enc_mode=True, mask_token=True,
        ),
        head_in_channels=1088, head_hidden_channels=2048,
        head_embed_channels=256, head_num_prototypes=4096,
        num_global_view=2, num_local_view=6, up_cast_level=4,
    ),
    criteria=[
        dict(type="FocalLoss", gamma=2.0, alpha=0.5,
             loss_weight=1.0, ignore_index=-1, reduction="mean"),
        dict(type="LovaszLoss", mode="multiclass",
             loss_weight=1.0, ignore_index=-1),
    ],
)

# =============================================================================
# Stage 2 — ptv3hybrid crosslevel slicer (verbatim from larformer-particle-v1.py)
# =============================================================================
_token_refiner_cfg = dict(
    type="CrossLevelAttn", num_layers=2, num_heads=4, mlp_ratio=4.0,
    target_levels=["voxel_16cm", "voxel_8cm", "ptv3_dec3", "ptv3_dec2"],
    max_source_tokens_per_level=8192,
)
slicer_levels = [
    dict(name="voxel_16cm", builder="VoxelBuilder",
         builder_cfg=dict(voxel_size_cm=16.0, coord_scale=coord_scale),
         supervision=dict(mask=dict(weight=1.0, mode="aux"))),
    dict(name="voxel_8cm", builder="VoxelBuilder",
         builder_cfg=dict(voxel_size_cm=8.0, coord_scale=coord_scale),
         supervision=dict(
             mask=dict(weight=1.0, mode="aux"),
             cls=dict(num_classes=3, label_src="origin_label",
                      label_remap={0: 0, 1: 2, 2: 1}, reduce="amax",
                      weight=0.5, loss="ce", ignore_index=-1))),
    dict(name="ptv3_dec3", builder="PTv3DecoderStageLevel",
         builder_cfg=dict(stage_key="dec3", in_dim=_SLICER_PTV3_DEC_CHANNELS[3]),
         supervision=dict(mask=dict(weight=1.0, mode="aux"))),
    dict(name="ptv3_dec2", builder="PTv3DecoderStageLevel",
         builder_cfg=dict(stage_key="dec2", in_dim=_SLICER_PTV3_DEC_CHANNELS[2]),
         supervision=dict(mask=dict(weight=1.0, mode="aux"))),
    dict(name="spacepoint", builder="SpacepointBuilder",
         supervision=dict(mask=dict(weight=5.0, mode="primary"))),
]
slicer_scale_pattern = ["voxel_16cm", "voxel_8cm", "ptv3_dec3", "ptv3_dec2",
                        "spacepoint", "spacepoint"]
slicer_cfg = dict(
    type="LArFormer",
    backbone=dict(
        type="Sonata-v1m1",
        backbone=dict(
            type="PT-v3m2", in_channels=6,
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
            traceable=True, enc_mode=False, mask_token=False,
        ),
        head_in_channels=1088, head_hidden_channels=2048,
        head_embed_channels=256, head_num_prototypes=4096,
        num_global_view=2, num_local_view=6, up_cast_level=0,
    ),
    backbone_out_channels=slicer_backbone_out_channels,
    levels=slicer_levels, scale_pattern=slicer_scale_pattern,
    token_dim=slicer_token_dim, num_queries=128, num_classes=3,
    freeze_backbone=True, unfreeze_decoder=True, capture_decoder_stages=True,
    ptv3_decoder_init_scale=0.01, enable_origin_head=False,
    token_refiner=_token_refiner_cfg,
    decoder_kwargs=dict(num_heads=4, mlp_ratio=4.0, zero_init_output_proj=False,
        **(dict(pos_emb_kind="sinusoidal") if USE_SINUSOIDAL_POS_EMB else {})),
    loss_kwargs=dict(
        weight_class=2.0, weight_mask_primary=5.0, weight_dice_primary=5.0,
        weight_aux_mask=0.7, weight_per_level_cls=0.5, weight_origin=0.0,
        num_sample_points=16392, use_importance_sampling=True,
        importance_oversample_ratio=3.0, importance_ratio=0.375,
        importance_hard_neg_ratio=0.375, aux_max_tokens=20_000,
        no_object_weight=0.5),
    mixed_query_selection=dict(source_level="voxel_8cm", score_source="cls_head",
        selection_mode="top_m_then_fps", score_filter_multiplier=4),
    mask_denoising=dict(dn_groups=3, max_dn_per_event=96, anchor_jitter_std=0.05),
)
cascaded_slicer_cfg = dict(
    type="CascadedSlicer",
    deghoster=deghoster_cfg, deghoster_weight=deghoster_weight,
    slicer=slicer_cfg, slicer_backbone_weight=sonata_pretrain_weight,
    deghost_threshold_min=0.4, deghost_threshold_max=0.6,
    deghost_threshold_val=0.5, freeze_deghoster=True,
    deghoster_class_index_real=0, report_keep_frac=True,
)

# =============================================================================
# Stage 3 — particle segmenter (ptv3-decoder + crosslevel refiner)
# Grafted from larformer-particle-v1-cached-ptv3crosslevel.py's `model` so
# the trained model_iter_98652.pth keys match the particle_segmenter slot.
# =============================================================================
particle_levels = [
    dict(name="voxel_8cm", builder="VoxelBuilder",
         builder_cfg=dict(voxel_size_cm=8.0, coord_scale=coord_scale),
         supervision=dict(mask=dict(weight=1.0, mode="aux"))),
    dict(name="voxel_4cm", builder="VoxelBuilder",
         builder_cfg=dict(voxel_size_cm=4.0, coord_scale=coord_scale),
         supervision=dict(
             mask=dict(weight=1.0, mode="aux"),
             cls=dict(num_classes=STAGE3_NUM_CLASSES,
                      label_src="particle_class_id", reduce="soft_presence",
                      weight=0.3, loss="ce", ignore_index=-1))),
    dict(name="ptv3_dec3", builder="PTv3DecoderStageLevel",
         builder_cfg=dict(stage_key="dec3", in_dim=_PTV3_DEC_CHANNELS[3]),
         supervision=dict(mask=dict(weight=1.0, mode="aux"))),
    dict(name="ptv3_dec2", builder="PTv3DecoderStageLevel",
         builder_cfg=dict(stage_key="dec2", in_dim=_PTV3_DEC_CHANNELS[2]),
         supervision=dict(mask=dict(weight=1.0, mode="aux"))),
    dict(name="spacepoint", builder="SpacepointBuilder",
         supervision=dict(mask=dict(weight=5.0, mode="primary"))),
]
particle_scale_pattern = ["voxel_8cm", "voxel_4cm", "ptv3_dec3", "ptv3_dec2",
                          "spacepoint", "spacepoint"]
_particle_token_refiner_cfg = dict(
    type="CrossLevelAttn", num_layers=2, num_heads=4, mlp_ratio=4.0,
    target_levels=["voxel_8cm", "voxel_4cm", "ptv3_dec3", "ptv3_dec2"],
    max_source_tokens_per_level=8192,
)
particle_segmenter_cfg = dict(
    type="LArFormer",
    backbone=dict(
        type="Sonata-v1m1",
        backbone=dict(
            type="PT-v3m2", in_channels=6,
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
            traceable=True, enc_mode=False, mask_token=False,
        ),
        head_in_channels=1088, head_hidden_channels=2048,
        head_embed_channels=256, head_num_prototypes=4096,
        num_global_view=2, num_local_view=6, up_cast_level=0,
    ),
    backbone_out_channels=STAGE3_BACKBONE_OUT_CH,
    backbone_weight=sonata_pretrain_weight,
    levels=particle_levels, scale_pattern=particle_scale_pattern,
    token_dim=STAGE3_TOKEN_DIM, num_queries=STAGE3_NUM_QUERIES,
    num_classes=STAGE3_NUM_CLASSES,
    freeze_backbone=True, unfreeze_decoder=True, capture_decoder_stages=True,
    ptv3_decoder_init_scale=0.01, enable_origin_head=True,
    token_refiner=_particle_token_refiner_cfg,
    decoder_kwargs=dict(num_heads=4, mlp_ratio=4.0, zero_init_output_proj=False,
        **(dict(pos_emb_kind="sinusoidal") if USE_SINUSOIDAL_POS_EMB else {})),
    loss_kwargs=dict(
        weight_class=2.0, weight_mask_primary=5.0, weight_dice_primary=5.0,
        weight_aux_mask=0.5, weight_per_level_cls=0.3, weight_origin=0.5,
        num_sample_points=8192, use_importance_sampling=True,
        importance_oversample_ratio=3.0, importance_ratio=0.375,
        importance_hard_neg_ratio=0.375, aux_max_tokens=10_000,
        no_object_weight=0.1, weight_dn_loss=1.0),
    mixed_query_selection=dict(source_level="voxel_4cm", score_source="cls_head",
        selection_mode="top_m_then_fps", score_filter_multiplier=4),
    mask_denoising=dict(dn_groups=3, max_dn_per_event=64, anchor_jitter_std=0.05),
)

# =============================================================================
# Top-level CascadedParticleSegmenter
# =============================================================================
model = dict(
    type="CascadedParticleSegmenter",
    cascaded_slicer=cascaded_slicer_cfg,
    particle_segmenter=particle_segmenter_cfg,
    cascaded_slicer_weight=cascaded_slicer_weight,
    particle_segmenter_backbone_weight=sonata_pretrain_weight,
    freeze_cascaded_slicer=True,
    nu_class_id=0,
    mask_prob_threshold=STAGE3_MASK_PROB_THRESHOLD,
    spacepoint_level="spacepoint",
    recenter_to_slice_centroid=STAGE3_RECENTER_TO_SLICE_CENTROID,
    report_keep_frac=True,
)

# Drop the `os` module + helper from the config namespace — Pointcept's
# Config deep-copies all top-level names, and a module object can't be
# pickled/deepcopied (`cannot pickle 'module' object`). All _env() reads
# above already resolved to plain strings at load time.
del os, _env

# Inference-only — no training loop is used, but keep the keys present so
# Config.fromfile + any generic tooling don't trip.
weight = None
save_path = "exp/larformer_particle_fullcascade_ptv3crosslevel_infer"
batch_size = 1
batch_size_val = 1
num_worker = 2
num_worker_val = 2
evaluate = False
enable_amp = False
find_unused_parameters = True
