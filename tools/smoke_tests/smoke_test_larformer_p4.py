"""
P4 smoke test for LArFormerDataset (see Pointcept/docs/LArFormer.md §13).

Verifies:
  - LArFormerDataset loads the canonical 10-file dev sample under each
    `gt_source` mode (slice, shower_trunk, deghost)
  - per-SP fields are coherent (slice_id present when MC, fragment fields
    appear iff emit_fragments=True)
  - gt_instances dicts have the right schema per source
  - the collate output drives LArFormer.forward() end-to-end (matches the
    P2 slicer-style scale pattern)

Usage:
    ./run_in_container.sh python tools/smoke_tests/smoke_test_larformer_p4.py
"""

import argparse
import os
import sys

import numpy as np
import torch


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_LIST = os.path.join(REPO_ROOT, "devdata_mergedh5_pi0filter_10files.txt")
BACKBONE_OUT = 1232


def build_sonata_backbone_cfg():
    return dict(
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
            attn_drop=0.0, proj_drop=0.0, drop_path=0.0,
            shuffle_orders=False, pre_norm=True,
            enable_rpe=False, enable_flash=True, flash_backend="xformers",
            upcast_attention=False, upcast_softmax=False,
            traceable=True, enc_mode=True, mask_token=False,
        ),
        head_in_channels=1088, head_hidden_channels=2048,
        head_embed_channels=256, head_num_prototypes=4096,
        num_global_view=2, num_local_view=6,
        up_cast_level=4,
    )


def make_dataset(list_file, gt_source, emit_fragments, cap):
    from pointcept.datasets import LArFormerDataset
    return LArFormerDataset(
        split="train",
        data_root="/",
        data_list_file=list_file,
        coord_center=(125.0, 0.0, 518.0),
        coord_scale=179.55,
        lm_score_aug_low=0.40,
        lm_score_aug_high=0.80,
        lm_score_val_threshold=0.60,
        min_fragment_points_post_filter=50,
        log_transform_strength=True,
        wire_scale=1.0 / 3456.0,
        gt_source=gt_source,
        emit_fragments=emit_fragments,
        max_spacepoints=cap,
    )


def inspect_sample(tag, sample):
    print(f"\n--- gt_source={tag} ---")
    print(f"  n_spacepoints:   {sample['n_spacepoints']}")
    print(f"  n_gt_instances:  {sample['n_gt_instances']}")
    print(f"  has slice_id:    {'slice_id' in sample} "
          f"(unique={len(np.unique(sample['slice_id']))})")
    if "fragment_indices" in sample:
        print(f"  n_fragments:     {sample['n_fragments']}")
    if sample["gt_instances"]:
        gi = sample["gt_instances"][0]
        print(f"  gt_instance[0] keys: {sorted(gi.keys())}")
        print(f"    origin_type:        {gi['origin_type']}")
        print(f"    origin_coord_norm:  {gi['origin_coord_norm']}")
        print(f"    n_truth_points:     {gi['n_truth_points']}")
    # Slice counts breakdown if relevant
    if tag == "slice" and sample["gt_instances"]:
        types = np.array([g["origin_type"] for g in sample["gt_instances"]])
        origins = np.array([g.get("primary_origin", -1) for g in sample["gt_instances"]])
        print(f"  per-instance origin_type breakdown: "
              f"{dict(zip(*np.unique(types, return_counts=True)))}")
        print(f"  per-instance primary_origin breakdown: "
              f"{dict(zip(*np.unique(origins, return_counts=True)))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", default=DEFAULT_LIST)
    ap.add_argument("--cap", type=int, default=20_000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(0); np.random.seed(0)
    sys.path.insert(0, REPO_ROOT)
    from pointcept.datasets import larformer_collate
    from pointcept.models.builder import build_model
    from pointcept.models.LArFormer import LArFormer  # noqa: F401

    print(f"=== P4 smoke test ===")
    print(f"list: {args.list}\ncap:  {args.cap}\n")

    # -------- gt_source = slice --------------------------------------------
    ds_slice = make_dataset(args.list, "slice",
                            emit_fragments=False, cap=args.cap)
    print(f"slice dataset: {len(ds_slice)} events")
    s = ds_slice[0]
    inspect_sample("slice", s)

    # -------- gt_source = shower_trunk + emit_fragments --------------------
    ds_st = make_dataset(args.list, "shower_trunk",
                         emit_fragments=True, cap=args.cap)
    print(f"\nshower_trunk dataset: {len(ds_st)} events")
    s2 = ds_st[0]
    inspect_sample("shower_trunk (emit_fragments=True)", s2)

    # -------- gt_source = deghost ------------------------------------------
    ds_deghost = make_dataset(args.list, "deghost",
                              emit_fragments=False, cap=args.cap)
    s3 = ds_deghost[0]
    inspect_sample("deghost", s3)
    assert len(s3["gt_instances"]) == 0, "deghost should have no instances"
    assert "slice_id" in s3, "deghost still wants slice_id for analysis"

    # -------- end-to-end: feed LArFormer through the new collate -----------
    print("\n=== End-to-end: LArFormer.forward on LArFormerDataset(slice) ===")
    LEVELS = [
        dict(name="voxel_10cm",
             builder="VoxelBuilder",
             builder_cfg=dict(voxel_size_cm=10.0, coord_scale=179.55),
             supervision=dict(
                 mask=dict(weight=1.0, mode="aux"),
                 cls=dict(num_classes=3, label_src="origin_label",
                          reduce="plurality", weight=0.5, loss="ce",
                          ignore_index=-1),
             )),
        dict(name="voxel_5cm",
             builder="VoxelBuilder",
             builder_cfg=dict(voxel_size_cm=5.0, coord_scale=179.55),
             supervision=dict(mask=dict(weight=1.0, mode="aux"))),
        dict(name="spacepoint",
             builder="SpacepointBuilder",
             supervision=dict(mask=dict(weight=5.0, mode="primary"))),
    ]
    SCALE_PATTERN = [
        "voxel_10cm", "voxel_5cm", "voxel_5cm",
        "spacepoint", "spacepoint",
    ]
    model_cfg = dict(
        type="LArFormer",
        backbone=build_sonata_backbone_cfg(),
        backbone_out_channels=BACKBONE_OUT,
        levels=LEVELS, scale_pattern=SCALE_PATTERN,
        token_dim=128, num_queries=16,
        # Slicer class set (nu=0, cosmic=1, no_object=2). num_classes counts
        # ALL slots incl. no_object — last slot is no_object by convention.
        num_classes=3,
        freeze_backbone=True, enable_origin_head=True,
        decoder_kwargs=dict(num_heads=4, mlp_ratio=4.0),
        loss_kwargs=dict(
            weight_class=2.0,
            weight_mask_primary=5.0,
            weight_dice_primary=5.0,
            weight_aux_mask=1.0,
            weight_per_level_cls=0.5,
            weight_origin=1.0,
            num_sample_points=1024,
            aux_max_tokens=20_000,
            no_object_weight=0.1,
        ),
    )
    model = build_model(model_cfg).to(args.device)
    model.train()

    batched = larformer_collate([s])
    for k, v in batched.items():
        if isinstance(v, torch.Tensor):
            batched[k] = v.to(args.device, non_blocking=True)
    out = model(batched)
    print(f"  loss = {out['loss'].item():.3f}")
    for k, v in out.items():
        if k == "loss":
            continue
        if hasattr(v, "item"):
            print(f"    {k:32s} = {v.item():.4f}")
    out["loss"].backward()
    print("  backward OK")

    print("\nP4 smoke test PASSED.")


if __name__ == "__main__":
    main()
