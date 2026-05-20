"""
LArFormer Stage-1 deghoster — PT-v3 *decoder* backbone variant.

Same task and dataset as `larformer-deghost-v0.py` (per-token binary ghost
vs real cls), but the backbone uses the PT-v3 decoder branch
(`enc_mode=False`) instead of Sonata's `up_cast_level=4` skip-concat
upsampling. Output dim is `dec_channels[0]` (48) instead of 1232 — much
narrower, but the features are produced by a learned hierarchical decoder
rather than raw skip concatenation.

See `Pointcept/docs/LArFormer.md` §8 and the comparison table in
`docs/LArFormer.md` (decoder trade-off discussion).

How this differs from `larformer-deghost-v0.py`:

  backbone.backbone.enc_mode   :  True  →  False
  + dec_depths, dec_channels, dec_num_head, dec_patch_size (new)
  backbone.up_cast_level        :  4     →  0   (PT-v3 decoder already
                                                  emits full-res features)
  backbone_out_channels         :  1232  →  48  (= dec_channels[0])

The Sonata-v1m1 wrapper is kept (not switched to raw PT-v3m2) because:
  - it preserves the same `backbone(bb_in, return_point=True)["point"].feat`
    API that LArFormer's `_encode` expects (zero code changes to
    LArFormer itself)
  - its `up_cast` loop is a no-op when `up_cast_level=0`
  - its teacher/student structure is preserved, so the
    SonataCheckpointLoader still works for any pretrained-Sonata weights
    you might load on top
"""

_base_ = ["../_base_/default_runtime.py"]

# =============================================================================
# Paths
# =============================================================================
_DEFAULT_VAL_LIST = "/home/twongjirad/working/larbys/gen2/container_u22/Pointcept/devdata_mergedh5_pi0filter_10files.txt"

# =============================================================================
# Coords + backbone shape
# =============================================================================
coord_center = (125.0, 0.0, 518.0)
coord_scale = 179.55
flash_backend = "xformers"
token_dim = 256

# PT-v3 decoder output dim = dec_channels[0]. The cls head reads this dim
# through the SpacepointBuilder's projection (Linear(backbone_out → token_dim)).
backbone_out_channels = 48

# =============================================================================
# Dataset (identical to larformer-deghost-v0.py)
# =============================================================================
_dataset_common = dict(
    type="LArFormerDataset",
    coord_center=coord_center, coord_scale=coord_scale,
    gt_source="deghost", emit_fragments=False,
    lm_score_aug_low=0.05,
    lm_score_aug_high=0.30,
    lm_score_val_threshold=0.10,
    log_transform_strength=True, wire_scale=1.0 / 3456.0,
    min_fragment_points_post_filter=50,
)
data = dict(
    num_classes=2,
    ignore_index=-1,
    names=["ghost", "real"],
    train=dict(split="train", data_root="/",
               data_list_file=_DEFAULT_VAL_LIST, loop=1,
               max_spacepoints=100_000, **_dataset_common),
    val=dict(split="val", data_root="/",
             data_list_file=_DEFAULT_VAL_LIST, loop=1,
             max_spacepoints=150_000, **_dataset_common),
    test=dict(split="test", data_root="/",
              data_list_file=_DEFAULT_VAL_LIST, loop=1,
              max_spacepoints=None, **_dataset_common),
)

# =============================================================================
# Levels (identical structure to larformer-deghost-v0.py)
# =============================================================================
levels = [
    dict(name="spacepoint",
         builder="SpacepointBuilder",
         supervision=dict(
             cls=dict(num_classes=2, label_src="hasmatch",
                      reduce="amax", weight=1.0, loss="ce",
                      hidden_dim=token_dim, ignore_index=-1),
         )),
]
scale_pattern = []

# =============================================================================
# Model
# =============================================================================
model = dict(
    type="LArFormer",
    backbone=dict(
        type="Sonata-v1m1",
        backbone=dict(
            type="PT-v3m2",
            in_channels=6,
            order=("z", "z-trans", "hilbert", "hilbert-trans"),
            stride=(2, 2, 2, 2),

            # ---- encoder (same as v0; must match if you load pretrained Sonata)
            enc_depths=(3, 3, 3, 9, 3),
            enc_channels=(48, 96, 192, 384, 512),
            enc_num_head=(3, 6, 12, 24, 32),
            enc_patch_size=(256, 256, 256, 256, 256),

            # ---- decoder (NEW relative to v0) ------------------------------
            # Mirrors semseg-sonata-v1m1-lartpc-v5-decoder-finetune.py:79-82
            # which is the established LArTPC PTv3-decoder config.
            dec_depths=(2, 2, 2, 2),
            dec_channels=(48, 96, 192, 384),
            dec_num_head=(3, 6, 12, 24),
            dec_patch_size=(256, 256, 256, 256),

            mlp_ratio=4, qkv_bias=True, qk_scale=None,
            attn_drop=0.0, proj_drop=0.0, drop_path=0.0,
            shuffle_orders=False, pre_norm=True,
            enable_rpe=False, enable_flash=True, flash_backend=flash_backend,
            upcast_attention=False, upcast_softmax=False,
            traceable=True,

            # KEY DIFFERENCE FROM v0: full encoder + decoder, not encoder-only.
            enc_mode=False,
            mask_token=False,
        ),
        head_in_channels=1088, head_hidden_channels=2048,
        head_embed_channels=256, head_num_prototypes=4096,
        num_global_view=2, num_local_view=6,
        # Decoder branch already emits full-resolution features, so the
        # Sonata wrapper's extra skip-concat upsampling is a no-op (would
        # also crash because dec output has no `pooling_parent`).
        up_cast_level=0,
    ),
    backbone_out_channels=backbone_out_channels,
    levels=levels,
    scale_pattern=scale_pattern,
    token_dim=token_dim,
    num_queries=0,
    num_classes=2,
    freeze_backbone=True,
    enable_origin_head=False,
    loss_kwargs=dict(weight_per_level_cls=1.0),
)
