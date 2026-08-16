"""Full-cascade config for the V2-CHAIN stage-1+2 cache rebuild (tau=0.20).

Overlay on `larformer-particle-fullcascade-ptv3crosslevel.py` that installs
the production stage-1+2 pair chosen by the 2026-07/08 retraining campaign
(see lartpc/larformer_analysis/slicer_eval/stage1and2_retraining_plan_and_results.md):

  Stage 1 deghoster : DefaultSegmentorV2 + PT-v3m2 decoder (full-event
                      fine-tuned), replacing SonataLoRADeghostSegmentor.
                      xformers backend MANDATORY (flash_attn NaNs in fp32).
  Deghost threshold : 0.20 (was 0.5) — the tau-sweep showed near-ceiling
                      shower completeness here (gamma 0.807 vs 0.815
                      ceiling) with a flat slicer gap and no degradation of
                      flash-chi2 slice selection.
  Stage 2 slicer    : m2frecipe-v2 epoch_4 — 48 queries (was 128).

Used by tools/larformer/build_stage12_cache_shard.py, which builds the
model from `cfg.model`, runs ONLY `model.cascaded_slicer`, and caches the
post-deghost / slice-loose spacepoint set. The stage-3 particle-segmenter
block is inherited unchanged: it is constructed but never executed by the
cache builder, and its (stale, old-chain) weights are irrelevant to the
cache contents.

NOTE the two distinct thresholds:
  - deghost_threshold_val = 0.20   → the DEGHOST cut (this campaign's choice)
  - --tau-loose-* CLI args         → SLICE-level nu-probability floors for
                                     the cache's union source set; unchanged
"""

_base_ = ["./larformer-particle-fullcascade-ptv3crosslevel.py"]

# ---------------------------------------------------------------------------
# Stage 1 — new PTv3-decoder deghoster (full-event fine-tuned)
# ---------------------------------------------------------------------------
_DEGHOST_FLASH_BACKEND = "xformers"   # fp32 inference: flash_attn -> NaN

deghoster_cfg = dict(
    _delete_=True,
    type="DefaultSegmentorV2",
    num_classes=2,
    backbone_out_channels=64,
    backbone=dict(
        type="PT-v3m2",
        in_channels=6,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(3, 3, 3, 9, 3),
        enc_channels=(48, 96, 192, 384, 512),
        enc_num_head=(3, 6, 12, 24, 32),
        enc_patch_size=(256, 256, 256, 256, 256),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(48, 48, 48, 48),
        mlp_ratio=4, qkv_bias=True, qk_scale=None,
        attn_drop=0.0, proj_drop=0.0, drop_path=0.0,
        shuffle_orders=False, pre_norm=True,
        enable_rpe=False, enable_flash=True,
        flash_backend=_DEGHOST_FLASH_BACKEND,
        upcast_attention=False, upcast_softmax=False,
        traceable=True, enc_mode=False, mask_token=False,
    ),
    criteria=[
        dict(type="FocalLoss", gamma=2.0, alpha=0.5,
             loss_weight=1.0, ignore_index=-1, reduction="mean"),
        dict(type="LovaszLoss", mode="multiclass",
             loss_weight=1.0, ignore_index=-1),
    ],
    freeze_backbone=False,
)

_KPV2 = "/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept"
deghoster_weight = (
    f"{_KPV2}/exp/deghost_ptv3decoder_v2_fullevent_ft/model/model_best.pth"
)
cascaded_slicer_weight = (
    f"{_KPV2}/exp/larformer_slicer_v1_cascaded_ptv3hybrid_crosslevel_"
    f"cap300k_m2frecipe_v2/model/epoch_4.pth"
)

# ---------------------------------------------------------------------------
# Stage 2 — m2frecipe-v2 slicer: 48 queries (ckpt shape), tau 0.20
# ---------------------------------------------------------------------------
model = dict(
    cascaded_slicer=dict(
        deghoster=deghoster_cfg,
        deghoster_weight=deghoster_weight,
        # Fixed eval threshold used by the cache builder (model runs .eval()).
        deghost_threshold_val=0.20,
        # Kept for completeness; train-time randomization is unused here.
        deghost_threshold_min=0.4,
        deghost_threshold_max=0.6,
        deghoster_class_index_real=0,
        slicer=dict(num_queries=48),
    ),
    cascaded_slicer_weight=cascaded_slicer_weight,
)
