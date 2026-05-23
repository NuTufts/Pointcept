"""
LArFormer Stage-2 cascaded slicer — variant that uses the existing trained
SonataLoRADeghostSegmentor as the Stage-1 deghoster.

Same scaffold as `larformer-slicer-v1-cascaded.py`, but `deghoster=` points
at the LoRA-finetuned Sonata deghoster from
`configs/lartpc/lorafinetune-sonata-v1m1-lartpc-v6-deghost.py`. Two changes
make this work:

  1. CascadedSlicer._run_deghoster_p_real already handles both output
     conventions (LArFormer's per-event `predictions` dict and the LoRA
     deghoster's flat `seg_logits` dict).
  2. `deghoster_class_index_real=0` is set on the cascade. The LoRA
     deghoster uses class 0=real, class 1=ghost (via the HasmatchAsGhost
     transform: hasmatch=1→0, hasmatch=0→1). LArFormer's per-token cls
     deghoster uses the opposite convention (class 1=real). The cascade
     just needs to know which column of the softmax is the real-class
     probability.

Usage:
    1. Edit `deghoster_weight` below to point at your trained LoRA
       deghoster checkpoint (e.g. `exp/<run-name>/model/model_best.pth`).
    2. Run:
           python tools/train.py --config $THIS_FILE
"""

_base_ = ["../_base_/default_runtime.py"]

# Side-effect: register LArFormerTrainer + LArFormerSlicerEvaluator.
from pointcept.models.LArFormer import trainer as _larformer_trainer_module
from pointcept.models.LArFormer import evaluator as _larformer_evaluator_module
del _larformer_trainer_module
del _larformer_evaluator_module

# =============================================================================
# Paths
# =============================================================================
_DEFAULT_LIST = "/home/twongjirad/working/larbys/gen2/container_u22/Pointcept/devdata_mergedh5_pi0filter_10files.txt"
#_DEFAULT_LIST = "/home/twongjirad/working/larbys/gen2/container_u22/Pointcept/devdata_mergedh5_pi0filter_1event.txt"

# Path to the trained SonataLoRADeghostSegmentor checkpoint. Replace with
# your actual run path; defaults are placeholders.
deghoster_weight = "/home/twongjirad/working/larbys/gen2/container_u22/Pointcept/sonata/lora_deghost_v6_hasmatch/model/epoch_30.pth"

# =============================================================================
# Toggles to combat "mirror-symmetric" slice mergers
# =============================================================================
# Observed pathology: when the decoder's only spatial signal comes from a
# learnable 3-layer MLP pos_emb AND queries carry no spatial anchor, two
# content-similar tracks at very different positions can get bound to the same
# query (the model has no good way to break the tie). The two knobs below
# inject hard positional priors:
#
# 1) USE_SINUSOIDAL_POS_EMB:
#      Replace the learnable MLP pos_emb with a fixed NeRF-style sinusoidal
#      embedding (log-spaced frequencies, sin/cos per axis). Every coord
#      gets a unique structured signature out of the box, so positions can't
#      collapse to mirror-symmetric features during the early training.
#
# 2) ENABLE_ORIGIN_HEAD_WITH_CENTROID:
#      Turn the per-query origin head back on AND use slice-centroid as the
#      regression target (vs. primary_start_pos, which can sit outside the
#      slice for charged primaries). Centroid is well-bounded inside the
#      TPC and tightly correlated with the slice's spacepoints, so the
#      regression has a clean signal. The predicted origin feeds back into
#      query_pos_dyn each layer → each query develops a spatial bias.
USE_SINUSOIDAL_POS_EMB = False
ENABLE_ORIGIN_HEAD_WITH_CENTROID = False

# =============================================================================
# Toggles for the per-pair negative sampler (over-clustering failure mode)
# =============================================================================
# The default PointRend importance sampler picks "uncertain" negatives near
# the predicted boundary (smallest |sigm - 0.5|). That's the right
# intervention for ambiguous-halo failures, but for over-clustering — where
# a query confidently absorbs another slice (sigm≈1 on the wrong points) —
# those confidently-wrong negatives are explicitly DEPRIORITIZED by the halo
# criterion and barely get any gradient pressure. See docs/LArFormer.md §15.
#
# Two complementary knobs:
#
# 1) PURE_RANDOM_NEGATIVES:
#      Disable importance sampling entirely → all negatives are uniform
#      random over bg. Cheapest ablation: tells you whether the sampler is
#      the bottleneck. Costs the boundary-refinement benefit. When True,
#      HARD_NEG_FRACTION_OF_IMPORTANCE is ignored.
#
# 2) HARD_NEG_FRACTION_OF_IMPORTANCE:
#      Within the 75% importance budget (the non-random share of negatives),
#      what fraction goes to "confident false-positive" mining — topk(sigm,
#      largest=True). 0.0 = all halo (current behavior). 0.5 = half halo /
#      half hard-neg. 1.0 = all hard-neg. The remaining 25% stays uniform
#      random for coverage.
PURE_RANDOM_NEGATIVES = False
HARD_NEG_FRACTION_OF_IMPORTANCE = 0.5

_IMPORTANCE_BUDGET = 0.0 if PURE_RANDOM_NEGATIVES else 0.75
_IMPORTANCE_RATIO = _IMPORTANCE_BUDGET * (1.0 - HARD_NEG_FRACTION_OF_IMPORTANCE)
_HARD_NEG_RATIO = _IMPORTANCE_BUDGET * HARD_NEG_FRACTION_OF_IMPORTANCE

# =============================================================================
# Token refiner (PerLevelSelfAttn — Option 1)
# =============================================================================
# When enabled, applies N self-attention + FFN blocks INDEPENDENTLY to each
# non-spacepoint level's tokens BEFORE the Mask2Former decoder sees them.
# The voxel tokens get a chance to evolve their representations so the mask
# head can distinguish content-similar distant tracks via more than just
# pos_emb. See docs/LArFormer.md §15 for the failure-mode analysis.
#
# Design notes:
#   - Operates on voxel levels only (target_levels=None → heuristic = every
#     `voxel_*` level). Spacepoint level is intentionally excluded: full
#     O(N²) self-attention on ~50K SPs is infeasible and PTv3's windowed
#     attention already mixes SP context inside the (frozen) backbone.
#   - Refiner has its OWN pos_emb (not shared with the decoder's). Set
#     `pos_emb_kind="sinusoidal"` to use the fixed sinusoidal variant on
#     the refiner side independently of the decoder's choice.
#   - Adds ~100k params per layer per voxel level (small). The cascade's
#     existing slicer-backbone checkpoint loads cleanly; the refiner
#     weights initialize randomly and train from scratch.
#
# Three options available — pick one by setting TOKEN_REFINER_KIND:
#
#   "identity"   — no refiner (current pre-refiner behavior; A/B baseline).
#
#   "per_level"  — Option 1 (PerLevelSelfAttn): each voxel level gets its
#                  own self-attention stack. Voxels within a level mix
#                  context with each other but not across levels. Cheapest.
#                  ~4.94M params at the default num_layers=2.
#
#   "cross_level"— Option 2 (CrossLevelAttn): each voxel level cross-
#                  attends against the concatenated token pool of all
#                  source levels (default: all levels including spacepoint).
#                  Level-agnostic analog of Mask2Former's pixel decoder —
#                  voxel tokens can READ from per-SP features, so a coarse
#                  voxel can pull fine-scale context. The shared pos_emb
#                  bridges levels via coords (no hierarchical pool needed,
#                  preserving LArFormer's flexible-levels design).
#                  ~4.78M params at num_layers=2. Set
#                  `max_source_tokens_per_level` if SP-as-source dominates
#                  GPU memory; 8192 is a safe cap for ~50K-SP events.
#
TOKEN_REFINER_KIND = "cross_level"          # "identity" | "per_level" | "cross_level"
TOKEN_REFINER_LAYERS = 2

if TOKEN_REFINER_KIND == "identity":
    _token_refiner_cfg = None
elif TOKEN_REFINER_KIND == "per_level":
    _token_refiner_cfg = dict(
        type="PerLevelSelfAttn",
        num_layers=TOKEN_REFINER_LAYERS,
        num_heads=4,
        mlp_ratio=4.0,
        # target_levels=["voxel_20cm", "voxel_10cm", "voxel_5cm"],
        # pos_emb_kind="sinusoidal",
    )
elif TOKEN_REFINER_KIND == "cross_level":
    _token_refiner_cfg = dict(
        type="CrossLevelAttn",
        num_layers=TOKEN_REFINER_LAYERS,
        num_heads=4,
        mlp_ratio=4.0,
        # target_levels=["voxel_20cm", "voxel_10cm", "voxel_5cm"],
        # source_levels=None,                  # None = all levels (incl. SP)
        max_source_tokens_per_level=8192,      # cap SP's K/V contribution
        # pos_emb_kind="sinusoidal",
    )
else:
    raise ValueError(
        f"TOKEN_REFINER_KIND must be 'identity', 'per_level', or "
        f"'cross_level'; got {TOKEN_REFINER_KIND!r}"
    )

# Back-compat alias for any external code that read the old name.
USE_PERLEVEL_REFINER = (TOKEN_REFINER_KIND == "per_level")
PERLEVEL_REFINER_LAYERS = TOKEN_REFINER_LAYERS

# =============================================================================
# Coords + backbone shape
# =============================================================================
coord_center = (125.0, 0.0, 518.0)
coord_scale = 179.55
#flash_backend = "xformers"
flash_backend = "flash_attn"
token_dim = 256                       # bumped from 128 for slicer capacity

# =============================================================================
# PTv3 decoder levels (Option "PTv3 native pyramid") — Mask2Former-style
# learned multi-scale features via PT-v3m2's own self.dec module.
# =============================================================================
# When False (default): use the existing voxel + spacepoint level scheme.
#   The slicer's PT-v3m2 runs encoder-only (enc_mode=True) and Sonata-v1m1
#   upcasts to per-SP features at 1232 channels.
#
# When True: enable PT-v3m2's learned decoder (enc_mode=False), DISABLE
#   Sonata-v1m1's upcast (up_cast_level=0), capture each decoder stage's
#   output via forward hooks, and expose them as PTv3DecoderStageLevel
#   levels. The PTv3 decoder is then the "refiner" — its trainable
#   transformer blocks process features at each native stride, much like
#   Mask2Former's pixel decoder.
#
# Trade-offs:
#   - Tokens-per-level are determined by PTv3's pool clusters (encoder
#     stride), not by user-chosen voxel sizes. Less flexible.
#   - Decoder weights are NOT in the Sonata pretrain (pretrain ran with
#     enc_mode=True), so they initialize randomly and train from scratch.
#     `unfreeze_decoder=True` is set automatically.
#   - The Sonata pretrain ENCODER weights still load cleanly (frozen).
#   - This setup is independent of TOKEN_REFINER_KIND — typically you
#     pair it with TOKEN_REFINER_KIND="identity" so the PTv3 decoder is
#     doing the multi-scale work alone, but a refiner ON TOP of the PTv3
#     decoder is also a valid hybrid (e.g. CrossLevelAttn).
USE_PTV3_DECODER_LEVELS = False

if USE_PTV3_DECODER_LEVELS:
    # PT-v3m2 decoder defaults (mirror the model's own defaults):
    #   dec_depths   = (2, 2, 2, 2)
    #   dec_channels = (64, 64, 128, 256)
    #   dec_num_head = (4, 4, 8, 16)
    #   dec_patch_size = (48, 48, 48, 48)
    # Per-stage output widths: dec1→64, dec2→128, dec3→256.
    _PTV3_DEC_CHANNELS = (64, 64, 128, 256)
    _PTV3_ENC_MODE = False
    _PTV3_UP_CAST_LEVEL = 0
    backbone_out_channels = _PTV3_DEC_CHANNELS[0]   # 64 = dec0 = per-SP feature width
else:
    _PTV3_DEC_CHANNELS = (64, 64, 128, 256)         # unused, but kept for clarity
    _PTV3_ENC_MODE = True
    _PTV3_UP_CAST_LEVEL = 4
    backbone_out_channels = 1232                    # current behavior

# =============================================================================
# Dataset — LArFormerDataset(gt_source="slice")
# =============================================================================
# NOTE: the LoRA deghoster was trained on a different dataset class
# (LArTPCDataset) with its own augmentation pipeline. For the cascade we
# feed it LArFormerDataset output. The relevant inputs (coord, feat,
# offset, grid_coord) are identical in shape and meaning across the two
# datasets, and the deghoster never sees the dataset-specific augmentation
# transforms (it only reads from data_dict in eval mode). Coord scale is
# 179.55 here vs ~179.44 in the LoRA training — a ~0.06% mismatch that
# is well within the model's robustness margin.
_dataset_common = dict(
    type="LArFormerDataset",
    coord_center=coord_center,
    coord_scale=coord_scale,
    gt_source="slice",
    emit_fragments=False,
    slice_class_map={1: 0, 2: 1},
    slice_origin_kind=("centroid" if ENABLE_ORIGIN_HEAD_WITH_CENTROID
                       else "primary_start_pos"),
    merge_nu_slices=True,
    # No lm_score pre-filter: the cascade's deghoster (Stage 1) is the
    # sole ghost discriminator. The LArMatch-stage lm_score is left
    # available to the model as a per-SP feature (via `feat`) but is not
    # used to drop SPs at the dataset level — that would double-up ghost
    # rejection (LArMatch already trims obvious ghosts at its own
    # threshold) and would feed the deghoster a non-representative input
    # subset that depresses its measured recall.
    lm_score_aug_low=0.0,
    lm_score_aug_high=0.0,
    lm_score_val_threshold=0.0,
    wire_scale=1.0 / 3456.0,
    min_fragment_points_post_filter=50,
)

data = dict(
    num_classes=3,
    ignore_index=-1,
    names=["nu", "cosmic", "no_object"],
    train=dict(split="train", data_root="/", data_list_file=_DEFAULT_LIST,
               loop=1, max_spacepoints=100_000, **_dataset_common),
    val=dict(split="val", data_root="/", data_list_file=_DEFAULT_LIST,
             loop=1, max_spacepoints=150_000, **_dataset_common),
    test=dict(split="test", data_root="/", data_list_file=_DEFAULT_LIST,
              loop=1, max_spacepoints=None, **_dataset_common),
)

# =============================================================================
# Stage 1 — SonataLoRADeghostSegmentor subconfig
# =============================================================================
# Copy of the model block from
# `configs/lartpc/lorafinetune-sonata-v1m1-lartpc-v6-deghost.py`. Must match
# the trained model architecture exactly (lora_rank, lora_target_modules,
# backbone params including drop_path / enable_flash / mask_token) so the
# checkpoint loads with missing=0 / unexpected=0.
deghoster_cfg = dict(
    type="SonataLoRADeghostSegmentor",
    backbone_out_channels=1232,
    lora_rank=16,
    lora_alpha=32.0,
    lora_dropout=0.05,
    lora_target_modules=["qkv", "proj"],
    freeze_backbone_non_lora=True,
    ghost_class_index=1,             # LoRA model's own convention (unused
                                      # by cascade; cascade uses
                                      # deghoster_class_index_real below)
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
            attn_drop=0.0, proj_drop=0.0, drop_path=0.3,
            shuffle_orders=True, pre_norm=True,
            enable_rpe=False, enable_flash=False, flash_backend=flash_backend,
            upcast_attention=False, upcast_softmax=False,
            traceable=True, enc_mode=True, mask_token=True,
        ),
        head_in_channels=1088, head_hidden_channels=2048,
        head_embed_channels=256, head_num_prototypes=4096,
        num_global_view=2, num_local_view=6,
        up_cast_level=4,
    ),
    # criteria is required by SonataLoRADeghostSegmentor.__init__ but never
    # invoked in our cascade (we call the deghoster in eval mode + the
    # dataset doesn't emit `segment`, so the loss path is skipped).
    criteria=[
        dict(type="FocalLoss", gamma=2.0, alpha=0.5,
             loss_weight=1.0, ignore_index=-1, reduction="mean"),
        dict(type="LovaszLoss", mode="multiclass",
             loss_weight=1.0, ignore_index=-1),
    ],
)

# =============================================================================
# Stage 2 — Slicer subconfig
# =============================================================================
if USE_PTV3_DECODER_LEVELS:
    # PT-v3m2 decoder stages dec1/dec2/dec3 (strides 2/4/8). dec0 is per-SP
    # and is covered by the spacepoint level. Channel widths come from
    # dec_channels = (64, 64, 128, 256) — stage k uses dec_channels[k].
    # Note: per-level cls supervision sits on dec1 (the finest pyramid
    # stage) here, mirroring the role voxel_10cm played in the voxel
    # variant — at this stride, cls labels still tile the event densely.
    levels = [
        dict(name="ptv3_dec3",
             builder="PTv3DecoderStageLevel",
             builder_cfg=dict(stage_key="dec3", in_dim=_PTV3_DEC_CHANNELS[3]),
             supervision=dict(mask=dict(weight=1.0, mode="aux"))),
        dict(name="ptv3_dec2",
             builder="PTv3DecoderStageLevel",
             builder_cfg=dict(stage_key="dec2", in_dim=_PTV3_DEC_CHANNELS[2]),
             supervision=dict(mask=dict(weight=1.0, mode="aux"))),
        dict(name="ptv3_dec1",
             builder="PTv3DecoderStageLevel",
             builder_cfg=dict(stage_key="dec1", in_dim=_PTV3_DEC_CHANNELS[1]),
             supervision=dict(
                 mask=dict(weight=1.0, mode="aux"),
                 cls=dict(num_classes=3, label_src="origin_label",
                          label_remap={0: 0, 1: 2, 2: 1}, reduce="amax",
                          weight=0.5, loss="ce", ignore_index=-1),
             )),
        dict(name="spacepoint",
             builder="SpacepointBuilder",
             supervision=dict(mask=dict(weight=5.0, mode="primary"))),
    ]
    scale_pattern = [
        "ptv3_dec3", "ptv3_dec2", "ptv3_dec2",
        "ptv3_dec1", "spacepoint", "spacepoint",
    ]
else:
    levels = [
        dict(name="voxel_20cm",
             builder="VoxelBuilder",
             builder_cfg=dict(voxel_size_cm=20.0, coord_scale=coord_scale),
             supervision=dict(mask=dict(weight=1.0, mode="aux"))),
        dict(name="voxel_10cm",
             builder="VoxelBuilder",
             builder_cfg=dict(voxel_size_cm=10.0, coord_scale=coord_scale),
             supervision=dict(
                 mask=dict(weight=1.0, mode="aux"),
                 cls=dict(num_classes=3, label_src="origin_label",
                          label_remap={0: 0, 1: 2, 2: 1}, reduce="amax",
                          weight=0.5, loss="ce", ignore_index=-1),
             )),
        dict(name="voxel_5cm",
             builder="VoxelBuilder",
             builder_cfg=dict(voxel_size_cm=5.0, coord_scale=coord_scale),
             supervision=dict(mask=dict(weight=1.0, mode="aux"))),
        dict(name="spacepoint",
             builder="SpacepointBuilder",
             supervision=dict(mask=dict(weight=5.0, mode="primary"))),
    ]
    scale_pattern = [
        "voxel_20cm", "voxel_10cm", "voxel_10cm",
        "voxel_5cm",  "spacepoint", "spacepoint",
    ]

slicer_cfg = dict(
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
            traceable=True, enc_mode=_PTV3_ENC_MODE, mask_token=False,
        ),
        head_in_channels=1088, head_hidden_channels=2048,
        head_embed_channels=256, head_num_prototypes=4096,
        num_global_view=2, num_local_view=6,
        up_cast_level=_PTV3_UP_CAST_LEVEL,
    ),
    backbone_out_channels=backbone_out_channels,
    levels=levels,
    scale_pattern=scale_pattern,
    token_dim=token_dim,
    num_queries=64,                   # bumped from 32: with ~30 GT slices per
                                       # event, 32 had no Hungarian slack →
                                       # noisy per-query specialization. 64
                                       # gives ~half the queries as "extra"
                                       # so matched assignments stabilize.
    num_classes=3,
    freeze_backbone=True,
    unfreeze_decoder=USE_PTV3_DECODER_LEVELS,    # train PTv3 decoder from scratch
    capture_decoder_stages=USE_PTV3_DECODER_LEVELS,
    enable_origin_head=ENABLE_ORIGIN_HEAD_WITH_CENTROID,
    token_refiner=_token_refiner_cfg,
    decoder_kwargs=dict(
        num_heads=4, mlp_ratio=4.0,
        **(dict(pos_emb_kind="sinusoidal") if USE_SINUSOIDAL_POS_EMB else {}),
    ),
    loss_kwargs=dict(
        weight_class=2.0,
        weight_mask_primary=5.0,
        weight_dice_primary=5.0,
        weight_aux_mask=0.3,           # dropped from 1.0: aux losses at 3
                                        # voxel levels were swamping the
                                        # primary mask signal (3 × 1.0 = 3.0
                                        # vs primary 5+5=10; aux is computed
                                        # at every layer too, so effective
                                        # weight is even higher).
        weight_per_level_cls=0.5,
        weight_origin=(0.5 if ENABLE_ORIGIN_HEAD_WITH_CENTROID else 0.0),
        num_sample_points=8192,         # bumped from 4096 — after dropping
                                        # the lm_score pre-filter, the slicer
                                        # sees ~3x more SPs per event, so the
                                        # per-pair sampler needs more budget
                                        # to cover each query's positive set.
        # PointRend-style hard-negative mining for the per-pair mask BCE/Dice.
        # 75% of each pair's negative budget is drawn from the model's
        # currently-uncertain region (sigmoid(logit) ≈ 0.5 — the halo
        # around the predicted mask boundary). Ported from shower_clustering
        # where it gave a measurable uplift on the same set-prediction loss
        # shape. Defaults match shower_clustering's documented values.
        use_importance_sampling=(not PURE_RANDOM_NEGATIVES),
        importance_oversample_ratio=3.0,
        importance_ratio=_IMPORTANCE_RATIO,
        importance_hard_neg_ratio=_HARD_NEG_RATIO,
        aux_max_tokens=20_000,
        no_object_weight=0.1,
    ),
    # Phase A: DINO/Mask-DINO mixed query selection. Pick K query anchors
    # from voxel_10cm's tokens, scored by `1 - p(no_object)` from the
    # level's cls head (same head supervised via weight_per_level_cls).
    # voxel_10cm is the cls-supervised level here (the ptv3hybrid_perlevel
    # twin uses voxel_8cm); both share the role of intermediate-granularity
    # objectness scorer for the selector. FPS over the top-M filter keeps
    # anchors spatially diverse. See docs/LArFormer.md §17.
    mixed_query_selection=dict(
        source_level="voxel_10cm",
        score_source="cls_head",
        selection_mode="top_m_then_fps",
        score_filter_multiplier=4,
    ),
    #mixed_query_selection=None,
    # Phase B: Mask DINO-style mask denoising. Train-only auxiliary path:
    # appends `dn_groups × n_gt` denoising queries after the regular K_reg
    # queries, each anchored at a jittered GT centroid and content-init
    # to a class embedding. Loss supervises them directly to their GT
    # (no Hungarian). max_dn_per_event caps total DN queries per event
    # (most events have ≤20 GT slices → ≤60 DN queries; the cap kicks
    # in for outlier cosmic-heavy events). See docs/LArFormer.md §18.
    mask_denoising=dict(
        dn_groups=3,
        max_dn_per_event=96,
        anchor_jitter_std=0.05,
    ),
    # mask_denoising=None,
)

# =============================================================================
# CascadedSlicer
# =============================================================================
# Slicer backbone pretrain. Loaded by CascadedSlicer.__init__ into
# self.slicer.backbone only (NOT through SonataCheckpointLoader — that
# hook would prepend "backbone." and overwrite the deghoster too).
# Replace with your vanilla Sonata pretrain path; the keys should look
# like `{teacher,student}.backbone.*` (the standard Sonata-pretrain
# layout). Set to None to leave the slicer's backbone at random init.
slicer_backbone_weight = (
    #"/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/"
    "/home/twongjirad/working/larbys/gen2/container_u22/Pointcept/"
    "sonata/lartpc_v6_h200_noghosts_pretrain_logspace_resume/model/epoch_42.pth"
)

model = dict(
    type="CascadedSlicer",
    deghoster=deghoster_cfg,
    deghoster_weight=deghoster_weight,
    slicer=slicer_cfg,
    slicer_backbone_weight=slicer_backbone_weight,
    deghost_threshold_min=0.3,
    deghost_threshold_max=0.6,
    deghost_threshold_val=0.5,
    freeze_deghoster=True,
    # KEY: the LoRA deghoster has class 0 = real, class 1 = ghost (from
    # HasmatchAsGhost: hasmatch=1→0 = real, hasmatch=0→1 = ghost).
    # The LArFormer-flavored deghoster (larformer-slicer-v1-cascaded.py)
    # uses class 1 = real instead.
    deghoster_class_index_real=0,
    report_keep_frac=True,
)

# =============================================================================
# Training loop knobs
# =============================================================================
# Top-level `weight` is unused: both submodule checkpoints are loaded by
# CascadedSlicer.__init__ via `deghoster_weight` and `slicer_backbone_weight`
# above. For RESUME of a cascaded run, swap SonataCheckpointLoader for the
# generic `CheckpointLoader` (which does no prefix munging) and set
# `weight = "exp/.../model/model_last.pth"` + `resume = True`.
weight = None

save_path        = "exp/larformer_slicer_v1_cascaded_loradeghost_crosslevelrefiner_mixedq_maskdn_nonzeroinit_10event"
epoch            = 2000
eval_epoch       = 400
batch_size       = 2
batch_size_val   = 2
num_worker       = 2
num_worker_val   = 2
evaluate         = True          # LArFormerSlicerEvaluator runs after each epoch
enable_amp       = False
empty_cache      = True
enable_wandb     = True
clip_grad        = 1.0
wandb_project    = "pointcept-larformer"
find_unused_parameters = True

# =============================================================================
# Optimizer / scheduler
# =============================================================================

# Previous Simple, One-Cycle Cosine Annealing Schedule
#base_lr = 5e-5
#param_dicts = None
# optimizer = dict(type="AdamW", lr=base_lr, weight_decay=0.01)
# scheduler = dict(
#     type="CosineAnnealingLR",
#     eta_min=base_lr * 0.01,
# )

# Flat LR with two decay triggers (see pointcept/utils/scheduler.py
# FlatWithDecayLR docstring):
#   - epoch trigger:   cut LR by `gamma` every `step_period_epochs`.
#   - plateau trigger: cut LR by `gamma` after `patience_epochs` of no
#                      val/loss improvement > `min_delta`, then suppress
#                      further plateau triggers for `cooldown_epochs`.
#   - mode="both":     either trigger fires (whichever first).
# Driven by the LREpochScheduler hook (must sit between the evaluator
# and CheckpointSaver in `hooks=` below). When resuming a run trained
# with this scheduler, set `extend_scheduler=False` on the checkpoint
# loader — the OneCycleLR-aware extend path rewrites max_lr/initial_lr
# and would corrupt FlatWithDecayLR state.
base_lr = 5.0e-5
param_dicts = None
optimizer = dict(type="AdamW", lr=base_lr, weight_decay=0.01)
scheduler = dict(
    type="FlatWithDecayLR",
    mode="plateau",
    gamma=0.5,
    min_lr=1e-7,
    step_period_epochs=400,
    patience_epochs=500,
    min_delta=1e-4,
    cooldown_epochs=2,
    # Linear warmup over the first 500 training iters (~100 epochs on the
    # 10-event dev sample at batch_size=2). PTv3 decoder + refiner +
    # Mask2Former decoder all initialize randomly here — gradients in the
    # first ~50 iters are noisy enough to spike the loss curve, so we
    # don't want the plateau detector touching anything during that
    # phase. step_epoch is a no-op while in warmup (counters frozen).
    warmup_iters=500,
    warmup_start_lr=0.0,
    # EMA over the val/loss for plateau detection. A single lucky-low
    # raw val_loss (which fluctuates a few % epoch-to-epoch on this
    # small dev sample) was pinning best_val_loss too tight, collapsing
    # the LR prematurely. alpha=0.3 means each new val_loss contributes
    # 30% to the smoothed signal; one outlier moves the EMA by at most
    # 30% of the gap, not the full distance. Set None to disable
    # smoothing (raw val_loss tracked, current pre-EMA behavior).
    ema_alpha=0.3,
    # No reset_lr by default — set on resume only.
    reset_lr=None,
    reset_counters=False,
)

hooks = [
    # No CheckpointLoader hook: both deghoster + slicer-backbone weights
    # are loaded inside CascadedSlicer.__init__ (see deghoster_weight +
    # slicer_backbone_weight on the model config). Add `dict(type=
    # "CheckpointLoader")` here (NOT SonataCheckpointLoader) only when
    # resuming a previous cascaded-training run.
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="LArFormerSlicerEvaluator",
         eval_freq=0,                # after each epoch
         best_metric="nu_mIoU",       # matched-pair IoU on the nu class
         class_names=["nu", "cosmic"],
         nu_class_id=0,
         empty_cache=True,
         log_per_event=False),
    dict(type="LREpochScheduler"),
    dict(type="CheckpointSaver", save_freq=None),
]

train = dict(type="LArFormerTrainer")
