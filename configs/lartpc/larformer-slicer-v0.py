"""
LArFormer Stage-2 slicer config (v0 / pre-training).

Defines the dataset + model levels used by:
    - the GT visualizer at tools/visualize_larformer_gt.py
    - (future) a full training run, once a Stage-2 trainer + evaluator land

The model block here is the LArFormer config the trainer / evaluator will
consume; the visualizer only reads `data.val.*` and `model.levels` plus
`model.token_dim`, so the heavy backbone block doesn't get instantiated at
visualization time.

Use case: slice instance segmentation (one slice per cosmic primary, one
merged slice per nu_vertex). gt_source="slice" pulls slice GT via
lartpc_data_prep.slice_labels.

Edit `data.{train,val}.data_list_file` to point at your own H5 list.
"""

_base_ = ["../_base_/default_runtime.py"]

# =============================================================================
# Paths
# =============================================================================
_DEFAULT_VAL_LIST = "/home/twongjirad/working/larbys/gen2/container_u22/Pointcept/devdata_mergedh5_pi0filter_10files.txt"

# =============================================================================
# Coordinate normalization (must match Sonata pretraining)
# =============================================================================
coord_center = (125.0, 0.0, 518.0)
coord_scale = 179.55

# =============================================================================
# Dataset — LArFormerDataset(gt_source="slice")
# =============================================================================
_dataset_common = dict(
    type="LArFormerDataset",
    coord_center=coord_center,
    coord_scale=coord_scale,
    gt_source="slice",
    emit_fragments=False,            # slicer doesn't use the fragment level
    slice_class_map={1: 0, 2: 1},     # nu→0, cosmic→1; no_object=2
    merge_nu_slices=True,
    lm_score_aug_low=0.40,
    lm_score_aug_high=0.80,
    lm_score_val_threshold=0.60,
    log_transform_strength=True,
    wire_scale=1.0 / 3456.0,
    min_fragment_points_post_filter=50,
)

data = dict(
    num_classes=3,                   # nu, cosmic, no_object
    ignore_index=-1,
    names=["nu", "cosmic", "no_object"],
    train=dict(
        split="train",
        data_root="/",
        data_list_file=_DEFAULT_VAL_LIST,   # placeholder; replace for real training
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
# Model — LArFormer with 3-level voxel + spacepoint primary
# =============================================================================
flash_backend = "xformers"
token_dim = 128
backbone_out_channels = 1232          # Sonata-v1m1 up_cast_level=4

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
             # 3-class per-voxel cls. The dataset's raw `origin_label`
             # is {0=ghost, 1=nu, 2=cosmic}; we relabel to a priority
             # order so amax-reduction acts as "any nu wins, else any
             # cosmic, else ghost" — the right semantics for a Stage-2
             # region-of-interest signal. Post-remap:
             #     0 = ghost   1 = cosmic   2 = nu
             cls=dict(num_classes=3, label_src="origin_label",
                      label_remap={0: 0, 1: 2, 2: 1},
                      reduce="amax",
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
    num_queries=32,
    num_classes=3,                  # nu, cosmic, no_object
    freeze_backbone=True,
    # Slice origins live at primary_start_pos, which for cosmics is outside
    # the TPC and produces a noisy regression target. Disable for v0; the
    # slicer doesn't need a precise vertex prediction.
    enable_origin_head=False,
    decoder_kwargs=dict(num_heads=4, mlp_ratio=4.0),
    loss_kwargs=dict(
        weight_class=2.0,
        weight_mask_primary=5.0,
        weight_dice_primary=5.0,
        weight_aux_mask=1.0,
        weight_per_level_cls=0.5,
        weight_origin=0.0,
        num_sample_points=4096,
        aux_max_tokens=20_000,
        no_object_weight=0.1,
    ),
)
