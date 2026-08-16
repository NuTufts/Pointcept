"""HYBRID interim cascade (2026-08-14): OLD LoRA deghoster at tau=0.20
feeding the NEW m2frecipe-v2 slicer + (via the kp2 config) the new stage-3.

Rationale: the ft PTv3 deghoster collapses on the run3b data-overlay domain
(photon keep 0.531 @ tau=0.2) while the LoRA deghoster is robust there
(0.854 @ 0.2). The new slicer/segmenter were TRAINED on a tau=0.2 cache, so
running the LoRA at 0.2 matches their training operating point — though the
cache was built with the FT deghoster's tau=0.2 output, so the kept-set
purity/composition still differs; this config exists to MEASURE whether the
extra photon charge survives downstream, on the 174-event pi0 pilot.

Overlay on the v2 production config: ONLY the deghoster block (back to
SonataLoRADeghostSegmentor, old production weights) is swapped; thresholds
stay 0.2; slicer (m2frecipe-v2 ep4, 48 queries) inherited unchanged.
class_index_real stays 0 (both deghosters use HasmatchAsGhost real=0).
LoRA's Sonata backbone runs enable_flash=False -> fp32-safe as-is.
"""

_base_ = ["./larformer-fullcascade-production-v2-tau020.py"]

_OLD_REPO = "/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept"

model = dict(
    cascaded_slicer=dict(
        deghoster=dict(
            _delete_=True,
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
                    # MUST be False at inference (reproducibility; see the
                    # old cascade config's note).
                    shuffle_orders=False, pre_norm=True,
                    enable_rpe=False, enable_flash=False,
                    flash_backend="xformers",
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
        ),
        deghoster_weight=(
            f"{_OLD_REPO}/sonata/lora_deghost_v6_hasmatch/model/epoch_30.pth"
        ),
    ),
)
