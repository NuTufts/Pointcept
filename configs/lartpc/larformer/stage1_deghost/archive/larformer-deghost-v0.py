"""
LArFormer Stage-1 deghoster config (v0 / pre-training).

Pattern: pure per-token semantic segmentation. No queries, no decoder, no
Hungarian matching — only a per-token cls head on the spacepoint level
predicting real (hasmatch=1) vs ghost (hasmatch=0). This is the Stage-1 of
the LArFormer cascade described in `Pointcept/docs/LArFormer.md` §6.

Things this config does NOT need (compared to the slicer):
  - num_queries (set to 0 → decoder is skipped at __init__)
  - scale_pattern (must be empty when num_queries=0)
  - gt_instances (LArFormerDataset(gt_source="deghost") emits no instances)
  - aux mask losses (no level declares supervision.mask)
  - origin head (no queries → no decoder → no head)

Per-token cls is on the spacepoint level. `label_src="hasmatch"` already
follows the natural priority (1 = real > 0 = ghost), so `reduce="amax"` is
equivalent to plurality here; we use amax for consistency with the slicer
config's priority-pool idiom.

Visualizer compatibility: this config is fully readable by
`tools/viz/visualize_larformer_gt.py` (the viz never touches the decoder).
"""

_base_ = ["../../../../_base_/default_runtime.py"]

# Side-effect import: triggers @TRAINERS.register_module() on
# LArFormerTrainer. Done here (not in pointcept.models.LArFormer.__init__)
# to avoid the circular import — pointcept.engines.train imports
# pointcept.models before TRAINERS is defined.
#
# We use `from X.Y import Z as _name` (not `import X.Y.Z as _name`) because
# the latter form interacts badly with Python's import-name resolution when
# X.Y has a camelCase name like `LArFormer` — produces a cryptic
# `ImportError: cannot import name 'Z' from 'X' (unknown location)`.
# The `from ... import` form is equivalent for side-effect purposes.
# Trailing `del` keeps the name out of Pointcept's config dumper (which
# captures every non-dunder name in the config namespace).
from pointcept.models.LArFormer import trainer as _larformer_trainer_module
del _larformer_trainer_module

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
backbone_out_channels = 1232          # Sonata-v1m1 up_cast_level=4

# =============================================================================
# Dataset
# =============================================================================
_dataset_common = dict(
    type="LArFormerDataset",
    coord_center=coord_center,
    coord_scale=coord_scale,
    gt_source="deghost",
    emit_fragments=False,
    lm_score_aug_low=0.05,            # MUCH lower so ghosts are present
    lm_score_aug_high=0.30,            # for training (else they're filtered
    lm_score_val_threshold=0.10,       # before the deghoster ever sees them)wire_scale=1.0 / 3456.0,
    min_fragment_points_post_filter=50,
)

data = dict(
    num_classes=2,                    # real, ghost
    ignore_index=-1,
    names=["ghost", "real"],          # post-priority order: 0=ghost, 1=real
    train=dict(
        split="train",
        data_root="/",
        data_list_file=_DEFAULT_VAL_LIST,
        loop=1,
        max_spacepoints=100_000,
        **_dataset_common,
    ),
    val=dict(
        split="val",
        data_root="/",
        data_list_file=_DEFAULT_VAL_LIST,
        loop=1,
        max_spacepoints=150_000,
        **_dataset_common,
    ),
    test=dict(
        split="test",
        data_root="/",
        data_list_file=_DEFAULT_VAL_LIST,
        loop=1,
        max_spacepoints=None,
        **_dataset_common,
    ),
)

# =============================================================================
# Levels — just the spacepoint level with a binary cls head
# =============================================================================
levels = [
    dict(name="spacepoint",
         builder="SpacepointBuilder",
         supervision=dict(
             # `hasmatch` is {0=ghost, 1=real} already in priority order;
             # amax acts as identity per-token (each SP maps 1:1 to its
             # own token, so the reduce is trivial). We declare the cls
             # head with hidden_dim>0 to give a small MLP capacity over
             # the projected backbone features.
             cls=dict(num_classes=2, label_src="hasmatch",
                      reduce="amax", weight=1.0, loss="ce",
                      hidden_dim=token_dim, ignore_index=-1),
         )),
]
# Empty scale_pattern is required when num_queries=0 (decoder is skipped).
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
            enc_depths=(3, 3, 3, 9, 3),
            enc_channels=(48, 96, 192, 384, 512),
            enc_num_head=(3, 6, 12, 24, 32),
            enc_patch_size=(256, 256, 256, 256, 256),
            mlp_ratio=4, qkv_bias=True, qk_scale=None,
            attn_drop=0.0, proj_drop=0.0, drop_path=0.0,
            shuffle_orders=False, pre_norm=True,
            enable_rpe=False, enable_flash=True, flash_backend=flash_backend,
            upcast_attention=False, upcast_softmax=False,
            traceable=True, enc_mode=True, mask_token=False,
        ),
        head_in_channels=1088, head_hidden_channels=2048,
        head_embed_channels=256, head_num_prototypes=4096,
        num_global_view=2, num_local_view=6,
        up_cast_level=4,
    ),
    backbone_out_channels=backbone_out_channels,
    levels=levels,
    scale_pattern=scale_pattern,
    token_dim=token_dim,
    num_queries=0,                    # ← disables the decoder
    num_classes=2,                    # unused when decoder is skipped
    freeze_backbone=True,
    enable_origin_head=False,
    loss_kwargs=dict(
        # Only `weight_per_level_cls` is consulted on the cls-only path; the
        # other weights are unused but kept here so the config is portable
        # if we later flip on a tiny decoder for joint training.
        weight_per_level_cls=1.0,
    ),
)

# ============================================================================
# Training loop knobs (override the defaults in _base_/default_runtime.py)
# ============================================================================
save_path        = "exp/larformer_deghost_v0"
epoch            = 500
eval_epoch       = 500            # rare; eval is currently disabled (see hooks)
batch_size       = 4             # bound by per-event N_sp via max_spacepoints
batch_size_val   = 2
num_worker       = 4
num_worker_val   = 2
evaluate         = False         # no LArFormer evaluator yet — disable for v0
enable_amp       = False
empty_cache      = True
enable_wandb     = True
wandb_project    = "pointcept-larformer"
find_unused_parameters = True    # frozen backbone has many unused params

# ============================================================================
# Optimizer & Scheduler
# ============================================================================
# Only 6 trainable params in the v0 deghoster (frozen Sonata backbone +
# Linear(1232→256) + 2-layer cls MLP). All similar scale, so single LR /
# single param group is enough. If you later wrap the backbone in LoRA at
# training time (instead of folding) or bolt on additional heads, switch
# back to `param_dicts = [dict(keyword="lora_", lr=...), ...]` style.
base_lr = 5e-4
param_dicts = None   # default: single param group, all params use `base_lr`

optimizer = dict(
    type="AdamW",
    lr=base_lr,
    weight_decay=0.01,
)

# `PolyLR` and `CosineAnnealingLR` are both forgiving when total_steps is
# small (which is the case on the 10-event dev dataset: ~2 steps/epoch).
# OneCycleLR divides by zero in `pct_start * total_steps - 1` when total_steps
# < ~20, so it's intentionally NOT used here for the dev config. For a real
# training run (~100k events) the user can swap in:
#     scheduler = dict(type="OneCycleLR", max_lr=base_lr,
#                      pct_start=0.05, anneal_strategy="cos",
#                      div_factor=10.0, final_div_factor=100.0)
scheduler = dict(
    type="CosineAnnealingLR",
    # `total_steps` is injected by the trainer's build_scheduler
    # (len(train_loader) * eval_epoch // gradient_accumulation_steps);
    # Pointcept's CosineAnnealingLR wrapper passes it to PyTorch as T_max.
    eta_min=base_lr * 0.01,
)

# ============================================================================
# Hooks — minimal v0 set. SonataCheckpointLoader loads `weight` (e.g. a
# folded LoRA backbone). SemSegEvaluator is intentionally OMITTED: it
# expects `seg_logits` in the model output, but LArFormer's deghoster
# returns `per_level_cls[spacepoint]`. A dedicated LArFormerEvaluator is
# a follow-up; for now, train + save checkpoints + inspect with the GT
# visualizer.
# ============================================================================
hooks = [
    dict(type="SonataCheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="CheckpointSaver", save_freq=None),
]

train = dict(type="LArFormerTrainer")