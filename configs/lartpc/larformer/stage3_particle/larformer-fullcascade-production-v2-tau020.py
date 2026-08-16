"""PRODUCTION stage-1+2 cascade, v2 (2026-08-08) — for cache building + reco.

Overlay on `larformer-particle-fullcascade-ptv3crosslevel.py` implementing
the retraining-campaign production choices (full record:
lartpc/larformer_analysis/slicer_eval/stage1and2_retraining_plan_and_results.md):

  1. DEGHOSTER: PTv3-decoder DefaultSegmentorV2 (full-event fine-tuned,
     `exp/deghost_ptv3decoder_v2_fullevent_ft/model/model_best.pth`)
     replaces the LoRA model. `_delete_=True` so no LoRA kwargs leak.
     flash_backend MUST be xformers: this model runs fp32 at inference and
     flash_attn produces garbage/NaN in fp32 (training job 2169933).
     Emits seg_logits -> existing cascade branch; class_index_real=0.
  2. DEGHOST THRESHOLD tau = 0.20 (user decision 2026-08-08, from the
     measured operating curve: gamma completeness 0.807 @ purity 0.659;
     flash-chi2 selection unaffected). min/max set to 0.2 as well for
     documentation — they only matter in train mode; the cache builder and
     reco run the cascade in eval mode, which uses deghost_threshold_val.
  3. SLICER: m2frecipe-v2 epoch_4
     (`exp/larformer_slicer_v1_cascaded_ptv3hybrid_crosslevel_cap300k_
     m2frecipe_v2/model/epoch_4.pth`) — num_queries 128 -> 48 to match its
     architecture; refiner max_source_tokens_per_level 8192 -> 16392 (the
     value it trained and was validated with); eos 0.1 (loss-only,
     provenance). NOTE the ep4 checkpoint contains the OLD LoRA deghoster's
     keys under `deghoster.*` — they don't overlap the new DefaultSegmentorV2
     key space, so the strict=False loads are order-independent (pattern
     proven by the valtest ftfull cascade config).
  4. DATA: max_spacepoints=None here; the cache-build scripts pass
     --max-spacepoints 300000 on the CLI (overrides this) — the cap-study
     value that bites only ~1.6% of events while guarding worst-case OOM
     (the original bug was the much tighter 100k/150k caps).

The particle_segmenter block is inherited unchanged (OLD stage-3 weights) —
it is loaded but unused by the stage-1+2 cache builder; stage-3 retrains
against the new cache next.
"""

_base_ = ["./larformer-particle-fullcascade-ptv3crosslevel.py"]

_KPV2 = "/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept"

model = dict(
    cascaded_slicer=dict(
        deghoster=dict(
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
                flash_backend="xformers",     # MANDATORY in fp32 (see docstring)
                upcast_attention=False, upcast_softmax=False,
                traceable=True,
                enc_mode=False,
                mask_token=False,
            ),
            criteria=[
                dict(type="FocalLoss", gamma=2.0, alpha=0.5,
                     loss_weight=1.0, ignore_index=-1, reduction="mean"),
                dict(type="LovaszLoss", mode="multiclass",
                     loss_weight=1.0, ignore_index=-1),
            ],
            freeze_backbone=False,
        ),
        deghoster_weight=(
            f"{_KPV2}/exp/deghost_ptv3decoder_v2_fullevent_ft/model/model_best.pth"
        ),
        slicer=dict(
            num_queries=48,
            token_refiner=dict(max_source_tokens_per_level=16392),
            loss_kwargs=dict(no_object_weight=0.1),
        ),
        deghost_threshold_min=0.2,
        deghost_threshold_max=0.2,
        deghost_threshold_val=0.2,
        deghoster_class_index_real=0,
    ),
    cascaded_slicer_weight=(
        f"{_KPV2}/exp/larformer_slicer_v1_cascaded_ptv3hybrid_crosslevel_"
        "cap300k_m2frecipe_v2/model/epoch_4.pth"
    ),
)

data = dict(
    train=dict(max_spacepoints=None),
    val=dict(max_spacepoints=None),
)
