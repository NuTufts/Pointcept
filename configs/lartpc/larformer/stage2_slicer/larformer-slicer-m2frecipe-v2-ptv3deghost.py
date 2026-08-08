"""m2frecipe-v2 slicer + the NEW PTv3-decoder deghoster — cascade eval config.

Overlay on the m2frecipe-v2 slicer config that swaps the stage-1 deghoster:
SonataLoRADeghostSegmentor (LoRA + Linear(1232,2)) -> the DefaultSegmentorV2
PTv3-decoder deghoster trained 2026-08-04 (exp/deghost_ptv3decoder_v1_
frozenenc_extbnb, model_best = epoch 19, crop-val mIoU 0.7813 vs LoRA's
0.7777 — tied at crop level; this config exists to measure the FULL-EVENT
difference via the per-particle completeness pipeline).

Intended for inference/valtest (run_slicer_inference.py --weights <v2 ep4
slicer ckpt>), not training. Notes:
  - The deghoster block uses _delete_=True so none of the LoRA kwargs leak
    into the DefaultSegmentorV2 ctor.
  - flash_backend MUST stay "xformers" for the deghoster: it runs fp32 at
    inference and flash_attn produces garbage in fp32 (the NaN failure of
    training job 2169933).
  - deghoster emits seg_logits -> CascadedSlicer's SonataLoRA-style branch;
    deghoster_class_index_real=0 (HasmatchAsGhost real=0) UNCHANGED.
  - CascadedSlicer._load_deghoster_weight strips a uniform "backbone."
    prefix only if ALL keys carry it; this ckpt has backbone.* + seg_head.*
    so no strip occurs (correct).
"""

_base_ = ["./larformer-slicer-v1-cascaded-ptv3hybrid-crosslevel-m2frecipe-v2.py"]

_DEGHOST_FLASH_BACKEND = "xformers"

model = dict(
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
            flash_backend=_DEGHOST_FLASH_BACKEND,
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
        "/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept/"
        "exp/deghost_ptv3decoder_v1_frozenenc_extbnb/model/model_best.pth"
    ),
    deghoster_class_index_real=0,
)

save_path = "exp/larformer_slicer_m2frecipe_v2_ptv3deghost_eval"
