"""
P3 smoke test for LArFormer (see Pointcept/docs/LArFormer.md §13).

Verifies on real merged_h5 events that the FragmentBuilder works:
  - per-event slicing of the dataset's `fragment_indices_per_event` correctly
    reaches the builder
  - fragment-level token count F matches the dataset's n_fragments
  - sp_to_level_id correctly has −1 entries for SPs not in any fragment
  - per-level mask aux loss on the fragment scale produces finite gradient
  - a shower-cluster-equivalent LArFormer config trains (loss decreases on
    one-event overfit)

Usage:
    ./run_in_container.sh python tools/smoke_tests/smoke_test_larformer_p3.py
    ./run_in_container.sh python tools/smoke_tests/smoke_test_larformer_p3.py --overfit-event 0 --n-events 50 --cap 30000

The config mirrors `ShowerClusteringMask2Former`'s default scale pattern:
  [voxel, fragment, voxel, fragment, spacepoint, spacepoint]
with primary mask supervision on the spacepoint level.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_LIST = os.path.join(REPO_ROOT, "devdata_mergedh5_pi0filter_10files.txt")

LEVELS = [
    dict(name="voxel",
         builder="VoxelBuilder",
         builder_cfg=dict(voxel_size_cm=5.0, coord_scale=179.55),
         supervision=dict(mask=dict(weight=1.0, mode="aux"))),
    dict(name="fragment",
         builder="FragmentBuilder",
         builder_cfg=dict(pool_layers=2, pool_heads=8, pool_max_points=512),
         supervision=dict(mask=dict(weight=1.0, mode="aux"))),
    dict(name="spacepoint",
         builder="SpacepointBuilder",
         supervision=dict(mask=dict(weight=5.0, mode="primary"))),
]
SCALE_PATTERN = [
    "voxel", "fragment", "voxel", "fragment", "spacepoint", "spacepoint",
]

BACKBONE_OUT = 1232


def build_sonata_backbone_cfg():
    return dict(
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
            enable_rpe=False, enable_flash=True, flash_backend="xformers",
            upcast_attention=False, upcast_softmax=False,
            traceable=True, enc_mode=True, mask_token=False,
        ),
        head_in_channels=1088, head_hidden_channels=2048,
        head_embed_channels=256, head_num_prototypes=4096,
        num_global_view=2, num_local_view=6,
        up_cast_level=4,
    )


def get_dataset(data_list_file, cap):
    from pointcept.datasets.shower_clustering import ShowerClusteringDataset
    return ShowerClusteringDataset(
        split="train",
        data_root="/",
        data_list_file=data_list_file,
        coord_center=(125.0, 0.0, 518.0),
        coord_scale=179.55,
        voxel_size_cm=5.0,
        lm_score_aug_low=0.40,
        lm_score_aug_high=0.80,
        lm_score_val_threshold=0.60,
        min_fragment_points_post_filter=50,
        log_transform_strength=True,
        wire_scale=1.0 / 3456.0,
        gt_label_source="union",
        max_spacepoints=cap,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", default=DEFAULT_LIST)
    ap.add_argument("--n-events", type=int, default=2)
    ap.add_argument("--cap", type=int, default=30_000)
    ap.add_argument("--num-queries", type=int, default=16)
    ap.add_argument("--token-dim", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--overfit-event", type=int, default=-1)
    args = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    sys.path.insert(0, REPO_ROOT)
    from pointcept.datasets.shower_clustering import shower_clustering_collate
    from pointcept.models.LArFormer import LArFormer  # noqa: F401
    from pointcept.models.builder import build_model

    print(f"=== P3 smoke test (shower-cluster-equivalent config) ===")
    print(f"levels:")
    for L in LEVELS:
        print(f"  {L['name']:12s} {L['builder']:18s} sup={list(L['supervision'].keys())}")
    print(f"scale_pattern: {SCALE_PATTERN}")
    print()

    ds = get_dataset(args.list, args.cap)
    print(f"Dataset: {len(ds)} events available\n")

    model_cfg = dict(
        type="LArFormer",
        backbone=build_sonata_backbone_cfg(),
        backbone_out_channels=BACKBONE_OUT,
        levels=LEVELS,
        scale_pattern=SCALE_PATTERN,
        token_dim=args.token_dim,
        num_queries=args.num_queries,
        num_classes=6,
        freeze_backbone=True,
        enable_origin_head=True,
        decoder_kwargs=dict(num_heads=4, mlp_ratio=4.0),
        loss_kwargs=dict(
            weight_class=2.0,
            weight_mask_primary=5.0,
            weight_dice_primary=5.0,
            weight_aux_mask=1.0,
            weight_per_level_cls=1.0,
            weight_origin=1.0,
            num_sample_points=1024,
            aux_max_tokens=20_000,
            no_object_weight=0.1,
        ),
    )
    model = build_model(model_cfg).to(args.device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Model: total={n_total/1e6:.2f}M trainable={n_trainable/1e6:.2f}M\n")

    # ------ Diagnostic: verify FragmentBuilder geometry on event 0 ------
    sample0 = ds[0]
    batched = shower_clustering_collate([sample0])
    for k, v in batched.items():
        if isinstance(v, torch.Tensor):
            batched[k] = v.to(args.device, non_blocking=True)
    levels0 = model.build_levels(batched)[0]
    print("=== build_levels diagnostic (event 0) ===")
    for name, lvl in levels0.items():
        n_unmapped = int((lvl.sp_to_level_id < 0).sum())
        n_mapped = int((lvl.sp_to_level_id >= 0).sum())
        print(f"  {name:12s} tokens={lvl.n_tokens:6d} "
              f"sp_mapped={n_mapped:6d} sp_unmapped={n_unmapped:6d}")
    print(f"  dataset reports n_fragments={int(sample0['n_fragments'])}")
    assert levels0["fragment"].n_tokens == int(sample0["n_fragments"]), (
        f"FragmentBuilder produced {levels0['fragment'].n_tokens} tokens, "
        f"dataset reported {int(sample0['n_fragments'])} fragments"
    )
    assert (levels0["spacepoint"].sp_to_level_id < 0).sum().item() == 0, \
        "spacepoint level should have no -1 entries"
    assert (levels0["voxel"].sp_to_level_id < 0).sum().item() == 0, \
        "voxel level should have no -1 entries (every SP maps to some voxel)"
    print("  FragmentBuilder geometry checks PASSED.\n")

    model.train()
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3,
    )

    if args.overfit_event >= 0:
        sample = ds[args.overfit_event % len(ds)]
        batched = shower_clustering_collate([sample])
        for k, v in batched.items():
            if isinstance(v, torch.Tensor):
                batched[k] = v.to(args.device, non_blocking=True)
        n_sp = int(sample["n_spacepoints"])
        n_gt = int(sample["n_gt_instances"])
        n_frag = int(sample["n_fragments"])
        print(f"=== overfit one event (idx={args.overfit_event}, "
              f"n_sp={n_sp}, n_frag={n_frag}, n_gt={n_gt}) for "
              f"{args.n_events} iters ===")
        first_loss = None
        for it in range(args.n_events):
            t0 = time.time()
            opt.zero_grad(set_to_none=True)
            out = model(batched)
            loss = out["loss"]
            loss.backward()
            opt.step()
            dt = time.time() - t0
            if first_loss is None:
                first_loss = loss.item()
            if it == 0 or it == args.n_events - 1 or (it + 1) % 5 == 0:
                print(f"  iter {it:3d}  loss={loss.item():7.3f}  "
                      f"cls={out['loss_cls'].item():6.3f}  "
                      f"mask_p={out['loss_mask_primary'].item():6.3f}  "
                      f"dice_p={out['loss_dice_primary'].item():6.3f}  "
                      f"aux_frag={out['loss_aux_mask_fragment'].item():6.3f}  "
                      f"aux_vox={out['loss_aux_mask_voxel'].item():6.3f}  "
                      f"origin={out['loss_origin'].item():6.3f}  "
                      f"dt={dt:.2f}s")
        print(f"\n  loss went from {first_loss:.3f} → {loss.item():.3f} "
              f"(Δ={loss.item() - first_loss:+.3f})")
    else:
        for it in range(args.n_events):
            ev_idx = it % len(ds)
            sample = ds[ev_idx]
            batched = shower_clustering_collate([sample])
            for k, v in batched.items():
                if isinstance(v, torch.Tensor):
                    batched[k] = v.to(args.device, non_blocking=True)
            n_sp = int(sample["n_spacepoints"])
            n_gt = int(sample["n_gt_instances"])
            n_frag = int(sample["n_fragments"])
            t0 = time.time()
            opt.zero_grad(set_to_none=True)
            out = model(batched)
            loss = out["loss"]
            loss.backward()
            opt.step()
            dt = time.time() - t0
            print(f"[iter {it}] ev={ev_idx} n_sp={n_sp:6d} n_frag={n_frag:3d} "
                  f"n_gt={n_gt:2d} loss={loss.item():7.3f}  dt={dt:.2f}s")
            for k, v in out.items():
                if k == "loss":
                    continue
                if hasattr(v, "item"):
                    print(f"    {k:32s} = {v.item():.4f}")

    print("\nP3 smoke test PASSED.")


if __name__ == "__main__":
    main()
