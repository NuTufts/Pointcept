"""LArFormer keypoint — Phase 2: query-based per-particle start/end head.

Fine-tunes the trained Stage-3 particle segmenter
(`larformer-particle-v1-cached-ptv3crosslevel.py` / its epoch_6 checkpoint)
with the Phase-2 keypoint head turned ON (`enable_keypoint_head=True`). Each
matched particle query additionally predicts its start (= the existing origin
head), end (track-like particles), an end-existence gate, and per-axis
log-variance; supervised via `keypoint_query_loss` (β-NLL) on the existing
query→GT Hungarian match. See lartpc_data_prep/larformer_keypoint/README.md §4.

Deltas vs the cached Stage-3 config this is copied from:
  - model.enable_keypoint_head = True  (+ keypoint loss weights, coord_scale)
  - datasets emit_keypoints = True     (gt_instances gain end_coord_norm /
    has_end; per-SP kpscores / kpoffsets surfaced — these supervise the
    integrated DENSE keypoint head below)
  - enable_keypoint_dense_head = True  (PPN-style per-SP score+offset head on
    the backbone, for the slice-level nu VERTEX — the evaluator decodes it and
    logs val/kp_*_nu_vertex + val/nu_vertex_res_cm_*)
  - weight = <epoch_6 ckpt>            (fine-tune init; CheckpointLoader is
    strict=False so the new keypoint heads stay random-init, reset_optimizer)
  - weight_origin = 0.0                (start is handled by kpq_start now)
  - new save_path; shorter warmup (fine-tune, not from-scratch)

The nu vertex is slice-level (NOT a query). It is produced by the integrated
DENSE per-SP keypoint head (enable_keypoint_dense_head) on the backbone
features — so this one model emits BOTH per-particle start/end (query head)
AND the dense keypoint field the vertex is decoded from. Dedicated vertex
queries remain deferred.

For inference, these weights load into the `particle_segmenter` slot of a
`CascadedParticleSegmenter` built with the same levels + enable_keypoint_head.
"""

_base_ = ["../_base_/default_runtime.py"]

# Side-effect: register LArFormerTrainer + the keypoint evaluator (which
# subclasses LArFormerParticleEvaluator, so the particle metrics log too).
from pointcept.models.LArFormer import trainer as _larformer_trainer_module
from pointcept.models.LArFormer import keypoint_particle_evaluator as _kp_eval_module
del _larformer_trainer_module
del _kp_eval_module

# =============================================================================
# Paths
# =============================================================================
CACHE_ROOT  = "/home/twongjirad/working/larbys/gen2/container_u22/Pointcept/exp/cache_stage12_devdata/"
#CACHE_ROOT  = "/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/exp/cache_stage12_devdata"
TRAIN_ROOT  = f"{CACHE_ROOT}/train"
VAL_ROOT    = f"{CACHE_ROOT}/val"

sonata_pretrain_weight = (
    "/home/twongjirad/working/larbys/gen2/container_u22/Pointcept/"
    #"/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/"
    "sonata/lartpc_v6_h200_noghosts_pretrain_logspace_resume/model/epoch_42.pth"
)

# Trained Stage-3 checkpoint to fine-tune the keypoint head from. Loaded by
# the CheckpointLoader hook (strict=False → new keypoint heads stay random).
stage3_init_weight = (
    "/home/twongjirad/working/larbys/gen2/container_u22/Pointcept/"
    "exp/larformer_particle_v1_cached_ptv3crosslevel_smallbatch_lr1e4_bugfixed/"
    "epoch_6.pth"
)

# =============================================================================
# Toggles
# =============================================================================
USE_SINUSOIDAL_POS_EMB      = False
PURE_RANDOM_NEGATIVES       = False
HARD_NEG_FRACTION_OF_IMPORT = 0.5

_IMPORTANCE_BUDGET = 0.0 if PURE_RANDOM_NEGATIVES else 0.75
_IMPORTANCE_RATIO  = _IMPORTANCE_BUDGET * (1.0 - HARD_NEG_FRACTION_OF_IMPORT)
_HARD_NEG_RATIO    = _IMPORTANCE_BUDGET * HARD_NEG_FRACTION_OF_IMPORT

STAGE3_NUM_QUERIES      = 32
STAGE3_NUM_CLASSES      = 8
STAGE3_TOKEN_DIM        = 256

_PTV3_DEC_CHANNELS    = (64, 64, 128, 256)
STAGE3_BACKBONE_OUT_CH = _PTV3_DEC_CHANNELS[0]   # 64 = dec0 width

coord_center = (125.0, 0.0, 518.0)
coord_scale  = 179.55
flash_backend = "flash_attn"

# Keypoint-regression warm-up: run the start/end regression with smooth_l1 for
# this many epochs before switching to β-NLL (lets the mean settle before the
# variance head + β-NLL noise kick in, so the LR doesn't have to be throttled
# for early-training stability). 0 disables. Scale this with the run length —
# ~20-25% of total epochs is a good rule of thumb (e.g. ~300-400 for a 2k run).
# Override per-run with: --options model.loss_kwargs.kp_reg_warmup_epochs=N
KP_REG_WARMUP_EPOCHS = 5

# =============================================================================
# Dataset (cache reader) — emit_keypoints=True is the delta.
# =============================================================================
_kp_ds_common = dict(
    type="LArFormerStage12CacheDataset",
    coord_center=coord_center,
    coord_scale=coord_scale,
    # Keypoint GT: gt_instances gain end_coord_norm / has_end; nu_vertex +
    # kpscores/kpoffsets also surfaced (unused by the query head, cheap).
    emit_keypoints=True,
    keypoint_sigma_cm=3.0,
    recenter_to_centroid=True,
    source_set_filter="stage2_pass",
)

data = dict(
    num_classes=STAGE3_NUM_CLASSES,
    ignore_index=-1,
    names=["e", "gamma", "mu", "pi", "p", "other", "(unused)", "no_object"],
    train=dict(split="train", data_root=TRAIN_ROOT, min_spacepoints=20,
               loop=1, **_kp_ds_common),
    val=dict(split="val", data_root=VAL_ROOT, loop=1, **_kp_ds_common),
    test=dict(split="test", data_root=VAL_ROOT, min_spacepoints=20,
              loop=1, **_kp_ds_common),
)

# =============================================================================
# Stage-3 particle segmenter — HYBRID levels (same as the source config).
# =============================================================================
particle_levels = [
    dict(name="voxel_8cm",
         builder="VoxelBuilder",
         builder_cfg=dict(voxel_size_cm=8.0, coord_scale=coord_scale),
         supervision=dict(mask=dict(weight=1.0, mode="aux"))),
    dict(name="voxel_4cm",
         builder="VoxelBuilder",
         builder_cfg=dict(voxel_size_cm=4.0, coord_scale=coord_scale),
         supervision=dict(
             mask=dict(weight=1.0, mode="aux"),
             cls=dict(num_classes=STAGE3_NUM_CLASSES,
                      label_src="particle_class_id",
                      reduce="soft_presence",
                      weight=0.3, loss="ce",
                      ignore_index=-1),
         )),
    dict(name="ptv3_dec3",
         builder="PTv3DecoderStageLevel",
         builder_cfg=dict(stage_key="dec3", in_dim=_PTV3_DEC_CHANNELS[3]),
         supervision=dict(mask=dict(weight=1.0, mode="aux"))),
    dict(name="ptv3_dec2",
         builder="PTv3DecoderStageLevel",
         builder_cfg=dict(stage_key="dec2", in_dim=_PTV3_DEC_CHANNELS[2]),
         supervision=dict(mask=dict(weight=1.0, mode="aux"))),
    dict(name="spacepoint",
         builder="SpacepointBuilder",
         supervision=dict(mask=dict(weight=5.0, mode="primary"))),
]
particle_scale_pattern = [
    "voxel_8cm", "voxel_4cm",
    "ptv3_dec3",  "ptv3_dec2",
    "spacepoint", "spacepoint",
]

_particle_token_refiner_cfg = dict(
    type="CrossLevelAttn",
    num_layers=2,
    num_heads=4,
    mlp_ratio=4.0,
    target_levels=["voxel_8cm", "voxel_4cm", "ptv3_dec3", "ptv3_dec2"],
    max_source_tokens_per_level=8192,
)

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
    backbone_out_channels=STAGE3_BACKBONE_OUT_CH,
    backbone_weight=sonata_pretrain_weight,
    levels=particle_levels,
    scale_pattern=particle_scale_pattern,
    token_dim=STAGE3_TOKEN_DIM,
    num_queries=STAGE3_NUM_QUERIES,
    num_classes=STAGE3_NUM_CLASSES,
    freeze_backbone=True,
    # ---- Keypoint-only training (freeze the particle network) ------------
    # freeze_non_keypoint=True freezes EVERYTHING (backbone, PT-v3 decoder,
    # token refiner, Mask2Former decoder layers + class/mask heads, per-level
    # cls heads, query selection, denoising) EXCEPT the keypoint params (the
    # per-query start/end/uncertainty/gate heads, the 2B refinement decoder,
    # and the dense head). This preserves the loaded Stage-3 segmentation
    # quality and directs ALL optimization to keypoints — fixes the "first 500
    # epochs improve segmentation, not keypoints" behaviour.
    #   - unfreeze_decoder=False so the PT-v3 decoder stays at the loaded
    #     weights and `_encode` runs the backbone under no_grad (no wasted
    #     graph). capture_decoder_stages stays True (the ptv3_dec levels still
    #     need the decoder forward; it runs frozen).
    # To do PARTIAL-JOINT instead (let the query embeddings adapt for
    # keypoints, at the cost of also moving segmentation): set
    # freeze_non_keypoint=False + unfreeze_decoder=True.
    freeze_non_keypoint=True,
    unfreeze_decoder=False,
    capture_decoder_stages=True,
    ptv3_decoder_init_scale=0.01,
    enable_origin_head=True,
    # ---- Phase-2 keypoint head (the delta) -------------------------------
    enable_keypoint_head=True,
    # ---- Phase-3 (2B) refinement decoder ---------------------------------
    # Each matched query cross-attends to its own spacepoints over 2 layers to
    # sharpen start/end (esp. track-end, which 2A regresses coarsely from the
    # global embedding). Identity at init, so it composes with a 2A ckpt.
    # Set to None to disable (pure 2A).
    keypoint_refine=dict(
        num_layers=2,
        num_heads=4,
        mlp_ratio=4.0,
        mask_threshold=0.0,
    ),
    # ---- Dense per-SP keypoint head (PPN-style) for the nu VERTEX ---------
    # Adds the Phase-1 score + offset heads on the backbone features so this
    # one model also produces the slice-level keypoint field. The evaluator
    # decodes the nu vertex from it and logs val/kp_{R,P}{1,3,10}_nu_vertex +
    # val/nu_vertex_res_cm_{median,mean}. Supervised by the per-SP kpscores /
    # kpoffsets the dataset already emits (emit_keypoints=True above).
    enable_keypoint_dense_head=True,
    keypoint_dense_cfg=dict(
        n_keypoint_types=6,
        head_hidden_dim=256,
        enable_offset_head=True,
        pos_weight=50.0,
        weight_score=1.0,
        weight_offset=1.0,
        coord_scale=coord_scale,
        # STAGED (off): concatenate a coordinate pos-emb onto the per-point
        # feature before the dense MLPs, giving them an explicit position/
        # distance handle (the backbone feature carries position only
        # implicitly; the offset head benefits most). Flip to "sinusoidal"
        # (or "mlp") to enable. pos_emb_dim defaults to the feature width.
        #pos_emb=None,
        pos_emb="sinusoidal", pos_emb_dim=64, pos_emb_max_freq=256.0,
    ),
    token_refiner=_particle_token_refiner_cfg,
    decoder_kwargs=dict(
        num_heads=4, mlp_ratio=4.0,
        zero_init_output_proj=False,
        **(dict(pos_emb_kind="sinusoidal") if USE_SINUSOIDAL_POS_EMB else {}),
        # STAGED (off): add an anchor pos-emb (prior layer's predicted origin /
        # the mixed-query anchor at init) to the query before the per-query
        # keypoint heads (start/end/uncertainty/gate) — DAB/DN-DETR style, so
        # the regression conditions on "where am I now". Class/mask heads are
        # untouched. Trains under the freeze (kp_pos_emb is a keypoint marker).
        #keypoint_pos_emb_kind=None,
        keypoint_pos_emb_kind="sinusoidal", keypoint_pos_emb_max_freq=256.0,
    ),
    loss_kwargs=dict(
        # ---- Particle (mask/cls) losses ZEROED under freeze_non_keypoint ----
        # The particle network is frozen, so these terms produce no gradient on
        # any trainable param anyway; zeroing them removes them from the logged
        # total (cleaner curves) and avoids any wasted backward through the
        # frozen branch. The matcher's cost_* terms (default cost_class/mask/
        # dice/origin) are SEPARATE knobs and stay on, so the Hungarian
        # query->GT assignment that the keypoint heads rely on is unaffected.
        # Restore these to (2,5,5,0.5,0.3) if you switch to partial-joint
        # training (freeze_non_keypoint=False).
        weight_class=0.0,
        weight_mask_primary=0.0,
        weight_dice_primary=0.0,
        weight_aux_mask=0.0,
        weight_per_level_cls=0.0,
        # Origin L1 is superseded by the keypoint start loss (kpq_start)
        # when enable_keypoint_head is on; set to 0 to be explicit.
        weight_origin=0.0,
        # ---- keypoint loss (β-NLL start/end + end-existence BCE) ----
        weight_kp_start=1.0,
        weight_kp_end=1.0,
        weight_kp_end_exist=0.5,
        kp_reg_kind="betanll",
        kp_beta=0.5,
        # Warm up the start/end regression with smooth_l1 before β-NLL (see
        # KP_REG_WARMUP_EPOCHS above + KeypointRegWarmupHook in `hooks`).
        kp_reg_warmup_epochs=KP_REG_WARMUP_EPOCHS,
        kp_reg_warmup_kind="smooth_l1",
        coord_scale=coord_scale,
        num_sample_points=8192,
        use_importance_sampling=(not PURE_RANDOM_NEGATIVES),
        importance_oversample_ratio=3.0,
        importance_ratio=_IMPORTANCE_RATIO,
        importance_hard_neg_ratio=_HARD_NEG_RATIO,
        aux_max_tokens=10_000,
        no_object_weight=0.1,
        weight_dn_loss=1.0,
    ),
    mixed_query_selection=dict(
        source_level="voxel_4cm",
        score_source="cls_head",
        selection_mode="top_m_then_fps",
        score_filter_multiplier=4,
    ),
    # Mask denoising OFF under freeze_non_keypoint: DN only supervises the
    # (now frozen) mask/cls path and never DN-supervises the keypoint heads,
    # so it would be pure wasted compute. Re-enable (the dict below) if you
    # switch to partial-joint training (freeze_non_keypoint=False).
    mask_denoising=None,
    # mask_denoising=dict(dn_groups=3, max_dn_per_event=64,
    #                     anchor_jitter_std=0.05),
)

# =============================================================================
# Trainer + evaluator. The particle evaluator scores masks/class/origin; it
# ignores the extra keypoint outputs (a keypoint evaluator is a Phase-4 item).
# =============================================================================
train = dict(type="LArFormerTrainer")

hooks = [
    # strict=False (default) so epoch_6 loads while the new keypoint heads
    # stay random-init; reset_optimizer since we changed the objective.
    dict(type="CheckpointLoader", reset_optimizer=True),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    # Drives the keypoint-regression warm-up (smooth_l1 -> β-NLL) by pushing
    # the current epoch into LArFormerLoss each epoch. No-op when
    # KP_REG_WARMUP_EPOCHS == 0.
    dict(type="KeypointRegWarmupHook"),
    dict(type="LArFormerKeypointEvaluator",
         best_metric="mask_iou_mean",
         class_names=["e", "gamma", "mu", "pi", "p", "other",
                      "(unused)", "no_object"],
         coord_scale=coord_scale,
         coord_center=coord_center,
         # 1 cm = precision target (watch val/kp_R1_*); 3/10 cm = context.
         thresholds_cm=(1.0, 3.0, 10.0),
         gt_acceptance_cm=5.0,
         # Dedup duplicate queries (mask-IoU NMS, SAME inference.dedup_queries
         # as the particle masks) before scoring keypoints, so co-extensive
         # duplicate queries don't double-count as keypoint FPs.
         dedup_iou_threshold=0.6),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="IterCheckpointSaver", save_iter_freq=5, keep_history=False),
    dict(type="SignalCheckpointHook", check_every_n_iter=30),
    dict(type="PreciseEvaluator", test_last=False),
]

# =============================================================================
# Training loop knobs
# =============================================================================
weight = stage3_init_weight     # fine-tune from the trained Stage 3
resume = False
save_path        = "exp/larformer_keypoint_query_v1"
epoch            = 20
eval_epoch       = 20
batch_size       = 16
batch_size_val   = 40
num_worker       = 12
num_worker_val   = 8
evaluate         = True
enable_amp       = False
amp_dtype        = "bfloat16"
empty_cache      = False
# With the particle network frozen, the grad norm is ENTIRELY keypoint-head
# gradient (dominated by the dense head's pos_weight=50 BCE + offset). A 1.0
# clip against a raw norm of 20-100 was scaling the keypoint gradient down
# 20-100x — i.e. the clip, not the LR, was the main throttle. Raise it so the
# heads actually move; lower again only if you see instability.
clip_grad        = 10.0
enable_wandb     = True
wandb_project    = "pointcept-larformer-keypoint"
find_unused_parameters = True

skip_dataloader_on_resume = True
resume_seed_strategy = "per_resume"

# =============================================================================
# Optimizer / scheduler — fine-tune (shorter warmup than from-scratch).
# =============================================================================
base_lr = 1.0e-5
# Per-head LR groups. The query/refine keypoint heads are smooth at base_lr
# (1e-5); the DENSE per-SP head (PPN-style nu-vertex score+offset) is a fresh
# random-init head on frozen features that needs to move much faster — give it
# its own ~10x group so it can start its precision upswing without forcing the
# whole run to a higher (noisier) LR. Tune `dense_lr` independently. Params are
# matched by substring: every tensor whose name contains "kp_dense" (the
# kp_dense_score_head / kp_dense_offset_head) goes in this group; everything
# else (incl. origin/end/refine heads) stays at base_lr.
dense_lr = 1.0e-4
param_dicts = [dict(keyword="kp_dense", lr=dense_lr)]
optimizer = dict(type="AdamW", lr=base_lr, weight_decay=0.01)
scheduler = dict(
    type="FlatWithDecayLR",
    mode="plateau",
    gamma=0.5,
    min_lr=5e-8,
    step_period_epochs=50,
    patience_epochs=2,
    min_delta=1e-4,
    cooldown_epochs=1,
    # Shorter warmup than the from-scratch source config: the decoder is
    # already trained; only the keypoint heads are fresh. Scale to your
    # dataset (iters/epoch = n_train / batch_size). For a real cluster run
    # over the full cache, bump this toward ~1 epoch of iters.
    warmup_iters=500,
    warmup_start_lr=0.0,
    ema_alpha=0.3,
)
