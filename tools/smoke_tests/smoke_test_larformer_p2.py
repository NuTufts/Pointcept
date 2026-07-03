"""
P2 smoke test for LArFormer (see Pointcept/docs/LArFormer.md §13).

Verifies on real merged_h5 events that:
  - the per-event slicing path in LArFormer works
  - the multi-voxel scale_pattern dispatch behaves
  - per-level aux mask losses + a per-level cls head produce finite values
  - backward flows through a frozen Sonata backbone end-to-end

Usage:
    ./run_in_container.sh python tools/smoke_tests/smoke_test_larformer_p2.py
    ./run_in_container.sh python tools/smoke_tests/smoke_test_larformer_p2.py --n-events 3 --cap 50000

The script DOES NOT load pretrained Sonata weights — this is an architecture
smoke test, not a quality check. P3+ will plug in real weights via the
SonataCheckpointLoader hook.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_LIST = os.path.join(REPO_ROOT, "devdata_mergedh5_pi0filter_10files.txt")

# Slicer-style LArFormer config: 3 voxel levels + spacepoint primary. No
# fragment level, so this exercises the pure-pooling pyramid that the
# Event Slicer will want.
LEVELS = [
    dict(name="voxel_20cm",
         builder="VoxelBuilder",
         builder_cfg=dict(voxel_size_cm=20.0, coord_scale=179.55),
         supervision=dict(mask=dict(weight=1.0, mode="aux"))),
    dict(name="voxel_10cm",
         builder="VoxelBuilder",
         builder_cfg=dict(voxel_size_cm=10.0, coord_scale=179.55),
         supervision=dict(
             mask=dict(weight=1.0, mode="aux"),
             # 3-class per-token cls on the dataset's `origin_label` field
             # (0 ghost, 1 nu, 2 cosmic). Verifies the per-level cls path.
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
    "voxel_20cm", "voxel_10cm", "voxel_10cm",
    "voxel_5cm",  "spacepoint", "spacepoint",
]


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
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            drop_path=0.0,
            shuffle_orders=False,
            pre_norm=True,
            enable_rpe=False,
            enable_flash=True,
            flash_backend="xformers",
            upcast_attention=False,
            upcast_softmax=False,
            traceable=True,
            enc_mode=True,
            mask_token=False,
        ),
        head_in_channels=1088,
        head_hidden_channels=2048,
        head_embed_channels=256,
        head_num_prototypes=4096,
        num_global_view=2,
        num_local_view=6,
        up_cast_level=4,  # full spacepoint resolution; out dim = 1232
    )


BACKBONE_OUT = 1232


def get_dataset(data_list_file, cap):
    from pointcept.datasets.shower_clustering import ShowerClusteringDataset
    return ShowerClusteringDataset(
        split="train",
        data_root="/",  # absolute paths in the list
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
    ap.add_argument("--list", default=DEFAULT_LIST,
                    help="path to a text file of merged_h5 paths")
    ap.add_argument("--n-events", type=int, default=2,
                    help="how many events to push through fwd+bwd")
    ap.add_argument("--cap", type=int, default=30_000,
                    help="per-event spacepoint cap (memory bound)")
    ap.add_argument("--num-queries", type=int, default=16)
    ap.add_argument("--token-dim", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--overfit-event", type=int, default=-1,
                    help="if >= 0, loop forever on this event index; "
                         "useful to verify the loss decreases. Use with --n-events.")
    args = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    sys.path.insert(0, REPO_ROOT)
    from pointcept.datasets.shower_clustering import shower_clustering_collate
    from pointcept.models.LArFormer import LArFormer  # noqa: F401

    from pointcept.models.builder import build_model

    print(f"=== P2 smoke test ===")
    print(f"list:       {args.list}")
    print(f"n_events:   {args.n_events}")
    print(f"cap:        {args.cap}")
    print(f"device:     {args.device}")
    print(f"token_dim:  {args.token_dim}  num_queries: {args.num_queries}")
    print(f"levels:")
    for L in LEVELS:
        print(f"  {L['name']:14s} {L['builder']:18s} sup={list(L['supervision'].keys())}")
    print(f"scale_pattern: {SCALE_PATTERN}")
    print()

    ds = get_dataset(args.list, args.cap)
    print(f"Dataset: {len(ds)} events available")

    model_cfg = dict(
        type="LArFormer",
        backbone=build_sonata_backbone_cfg(),
        backbone_out_channels=BACKBONE_OUT,
        levels=LEVELS,
        scale_pattern=SCALE_PATTERN,
        token_dim=args.token_dim,
        num_queries=args.num_queries,
        num_classes=6,  # shower-cluster GT origin_type 0..4 + no_object
        freeze_backbone=True,
        enable_origin_head=True,
        decoder_kwargs=dict(num_heads=4, mlp_ratio=4.0),
        loss_kwargs=dict(
            weight_class=2.0,
            weight_mask_primary=5.0,
            weight_dice_primary=5.0,
            weight_aux_mask=1.0,
            weight_per_level_cls=0.5,
            weight_origin=1.0,
            num_sample_points=1024,  # bounded for smoke test
            aux_max_tokens=20_000,
            no_object_weight=0.1,
        ),
    )
    model = build_model(model_cfg).to(args.device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Model: total={n_total/1e6:.2f}M trainable={n_trainable/1e6:.2f}M "
          f"(backbone frozen)")

    model.train()
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3,
    )

    if args.overfit_event >= 0:
        # Load the fixed event ONCE, reuse across all iters (dataset's
        # lm_score augmentation would otherwise change tau per call, but
        # for an architecture-level overfit test, keeping the batched dict
        # pinned removes a confound).
        sample = ds[args.overfit_event % len(ds)]
        batched = shower_clustering_collate([sample])
        for k, v in batched.items():
            if isinstance(v, torch.Tensor):
                batched[k] = v.to(args.device, non_blocking=True)
        n_sp = int(sample["n_spacepoints"])
        n_gt = int(sample["n_gt_instances"])
        print(f"\n=== overfit one event (idx={args.overfit_event}, "
              f"n_sp={n_sp}, n_gt={n_gt}) for {args.n_events} iters ===")
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
                      f"cls={out.get('loss_cls', torch.tensor(0)).item():6.3f}  "
                      f"mask_p={out.get('loss_mask_primary', torch.tensor(0)).item():6.3f}  "
                      f"dice_p={out.get('loss_dice_primary', torch.tensor(0)).item():6.3f}  "
                      f"origin={out.get('loss_origin', torch.tensor(0)).item():6.3f}  "
                      f"dt={dt:.2f}s")
        last_loss = loss.item()
        print(f"\n  loss went from {first_loss:.3f} → {last_loss:.3f} "
              f"(Δ={last_loss - first_loss:+.3f})")
    else:
        for it in range(args.n_events):
            ev_idx = it % len(ds)
            sample = ds[ev_idx]
            # Single-event "batch" via the dataset's collate
            batched = shower_clustering_collate([sample])
            # Move flat tensors to device
            for k, v in batched.items():
                if isinstance(v, torch.Tensor):
                    batched[k] = v.to(args.device, non_blocking=True)

            n_sp = int(sample["n_spacepoints"])
            n_gt = int(sample["n_gt_instances"])
            t0 = time.time()
            opt.zero_grad(set_to_none=True)
            out = model(batched)
            loss = out["loss"]
            loss.backward()
            opt.step()
            dt = time.time() - t0

            print(f"[iter {it}] event_idx={ev_idx} n_sp={n_sp:6d} n_gt={n_gt:3d} "
                  f"loss={loss.item():.3f}  dt={dt:.2f}s")
            for k, v in out.items():
                if k == "loss":
                    continue
                if hasattr(v, "item"):
                    print(f"    {k:30s} = {v.item():.4f}")

    # Quick eval-mode sanity: predictions shape inspection
    model.eval()
    with torch.no_grad():
        sample = ds[0]
        batched = shower_clustering_collate([sample])
        for k, v in batched.items():
            if isinstance(v, torch.Tensor):
                batched[k] = v.to(args.device, non_blocking=True)
        out = model(batched)
    preds = out["predictions"]
    print(f"\nEval-mode predictions: {len(preds)} event(s)")
    p = preds[0]
    print(f"  class_logits: {tuple(p['class_logits'].shape)}")
    print(f"  origin:       {tuple(p['origin'].shape)}")
    for name, ml in p["mask_logits"].items():
        print(f"  mask_logits[{name}]: {tuple(ml.shape)}")
    for name, cl in p["per_level_cls"].items():
        print(f"  per_level_cls[{name}]: {tuple(cl.shape)}")
    for name, lvl in p["levels"].items():
        print(f"  levels[{name}].coords: {tuple(lvl['coords'].shape)} "
              f"sp_to_level_id: {tuple(lvl['sp_to_level_id'].shape)}")

    # Public viz helper smoke test
    print("\n=== build_levels (visualizer hook, no backbone forward) ===")
    per_event_levels = model.build_levels(batched)
    for name, lvl in per_event_levels[0].items():
        print(f"  {name}: tokens={tuple(lvl.tokens.shape)} "
              f"coords={tuple(lvl.coords.shape)} "
              f"sp_to_level_id range=({int(lvl.sp_to_level_id.min())},"
              f"{int(lvl.sp_to_level_id.max())})")

    print("\nP2 smoke test PASSED.")


if __name__ == "__main__":
    main()
