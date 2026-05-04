"""
Draft config for the shower-clustering Mask2Former model (Phase 2 dataset
plumbing only; model sections are TODO placeholders).

This config sets up the ShowerClusteringDataset the way it will be wired up
during training. It targets the new merged H5 schema (with
mc_particle_tree). See pointcept/docs/shower_clustering_design.md.

The visualizer at tools/visualize_shower_clustering.py reads this config so
the GT labels shown match what the model will see.
"""

# =============================================================================
# Paths (override in subclass configs / via CLI)
# =============================================================================
# Default: the 3 fixed-merge test events. Real training will replace these
# with the production train/val/test split lists once the full re-merge job
# completes (~1 day per pi0 sample, in flight as of 2026-05-04).
_DEFAULT_TRAIN_LIST = (
    "/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/"
    "lartpc_data_prep/lantern_scripts/tmp_workdir/"
    "lantern_bnb_nu_pi0filter_corsika_jobid0000_line00001/"
    "shower_clustering_filelist.txt"
)
_DEFAULT_VAL_LIST = _DEFAULT_TRAIN_LIST  # same for now; smoke-test only

# Backbone pretrain (frozen Sonata-v1m1 / PT-v3m2). Reused as-is from V3.
weight = (
    "/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/sonata/"
    "lartpc_v6_h200_noghosts_pretrain/"
    "lartpc_v6_h200_noghosts_pretrain_epoch50.pth"
)
backbone_out_channels = 1088  # 192 + 384 + 512 (up_cast_level=2)

# =============================================================================
# Coordinate normalization (must match Sonata pretraining)
# =============================================================================
coord_center = (125.0, 0.0, 518.0)
coord_scale = 179.55

# =============================================================================
# Voxel / fragment parameters (see design doc §3, §4)
# =============================================================================
voxel_size_cm = 5.0
min_fragment_points_post_filter = 20  # matches DBSCAN's min_fragment_points

# =============================================================================
# lm_score threshold augmentation (design doc §4e)
# =============================================================================
lm_score_aug_low = 0.15        # production deghoster floor
lm_score_aug_high = 0.40       # cap to avoid emptying events
lm_score_val_threshold = 0.15  # fixed on val for run-to-run comparability

# =============================================================================
# Common dataset kwargs
# =============================================================================
_dataset_common = dict(
    type="ShowerClusteringDataset",
    coord_center=coord_center,
    coord_scale=coord_scale,
    voxel_size_cm=voxel_size_cm,
    lm_score_aug_low=lm_score_aug_low,
    lm_score_aug_high=lm_score_aug_high,
    lm_score_val_threshold=lm_score_val_threshold,
    min_fragment_points_post_filter=min_fragment_points_post_filter,
    log_transform_strength=True,
    wire_scale=1.0 / 3456.0,
    transform=None,  # augmentation is handled inside the dataset itself
)

dataset_type = "ShowerClusteringDataset"
data_root = "/"  # absolute paths only

data = dict(
    num_classes=5,  # inside / outside / on_track / ghost / true_track
    ignore_index=-1,
    names=["inside", "outside", "on_track", "ghost", "true_track"],
    train=dict(
        split="train",
        data_root=data_root,
        data_list_file=_DEFAULT_TRAIN_LIST,
        loop=1,
        **_dataset_common,
    ),
    val=dict(
        split="val",
        data_root=data_root,
        data_list_file=_DEFAULT_VAL_LIST,
        loop=1,
        **_dataset_common,
    ),
    test=dict(
        split="test",
        data_root=data_root,
        data_list_file=_DEFAULT_VAL_LIST,
        loop=1,
        **_dataset_common,
    ),
)

# =============================================================================
# Model — TODO (Phases 3–7 of the design doc)
# =============================================================================
# The Mask2Former decoder, fragment tokenizer, mask heads, and Hungarian loss
# do not exist yet. This stub is here so configs that import from this base
# don't fail to parse.
model = dict(
    type="ShowerClusteringMask2Former",  # not implemented yet
    # Backbone (frozen Sonata)
    backbone=dict(
        type="Sonata-v1m1",
        backbone=dict(
            type="PT-v3m2",
            in_channels=6,  # coord(3) + log(strength)(3)
            order=("z", "z-trans", "hilbert", "hilbert-trans"),
            stride=(2, 2, 2, 2),
            enc_depths=(3, 3, 3, 9, 3),
            enc_channels=(48, 96, 192, 384, 512),
            enc_num_head=(3, 6, 12, 24, 32),
            enc_patch_size=(256, 256, 256, 256, 256),
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            drop_path=0.0,  # frozen → no drop
            shuffle_orders=False,  # deterministic on inference
            pre_norm=True,
            enable_rpe=False,
            enable_flash=True,
            upcast_attention=False,
            upcast_softmax=False,
            traceable=True,
            enc_mode=True,
            mask_token=False,  # not pretraining
        ),
        head_in_channels=1088,
        head_hidden_channels=2048,
        head_embed_channels=256,
        head_num_prototypes=4096,
        num_global_view=2,
        num_local_view=6,
        up_cast_level=2,
    ),
    backbone_out_channels=backbone_out_channels,
    freeze_backbone=True,
    # Decoder / heads / matcher — TODO Phases 4–6
    num_queries=64,
    num_decoder_layers=6,
    num_classes=5,           # 5 origin types (no_object handled by matcher)
    no_object_class_id=5,    # 6th slot for unmatched queries
    voxel_size_cm=voxel_size_cm,
)

# =============================================================================
# Optimizer / scheduler — TODO when training is wired up
# =============================================================================
base_lr = 1e-4
optimizer = dict(type="AdamW", lr=base_lr, weight_decay=0.01)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[base_lr],
    pct_start=0.1,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)

# =============================================================================
# Misc
# =============================================================================
flash_backend = "fa"  # match V3
save_path = "shower_clustering/v1_baseline"
