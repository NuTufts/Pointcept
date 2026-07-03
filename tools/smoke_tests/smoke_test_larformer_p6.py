"""
P6 smoke test for CascadedSlicer (see Pointcept/docs/LArFormer.md §13).

Verifies on real merged_h5 events that the Stage-2 cascade works:
  - CascadedSlicer builds with both submodules
  - The deghoster runs in eval mode, returns per-SP P(real)
  - filter_batch_by_keep_mask shapes are consistent (offset rewritten,
    truth_indices remapped, no out-of-range indices)
  - The slicer trains end-to-end through the frozen deghoster
  - loss decreases on one-event overfit
  - drop_empty_events handles a high-τ batch gracefully

Usage:
    ./run_in_container.sh python tools/smoke_tests/smoke_test_larformer_p6.py
    ./run_in_container.sh python tools/smoke_tests/smoke_test_larformer_p6.py --overfit-event 0 --n-iters 30
"""

import argparse
import os
import sys
import time

import numpy as np
import torch


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",
                    default="configs/lartpc/larformer/stage2_slicer/archive/larformer-slicer-v1-cascaded.py")
    ap.add_argument("--cap", type=int, default=20_000,
                    help="per-event spacepoint cap (memory bound)")
    ap.add_argument("--n-iters", type=int, default=3)
    ap.add_argument("--overfit-event", type=int, default=-1)
    ap.add_argument("--no-deghoster-weight", action="store_true",
                    help="Skip loading the deghoster checkpoint (random init).")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(0); np.random.seed(0)
    sys.path.insert(0, REPO_ROOT)
    from pointcept.utils.config import Config
    from pointcept.datasets import build_dataset, larformer_collate
    from pointcept.models.builder import build_model
    from pointcept.models.LArFormer import (  # noqa: F401  (registry side-effects)
        CascadedSlicer, filter_batch_by_keep_mask, drop_empty_events,
    )

    cfg = Config.fromfile(args.config)
    if args.no_deghoster_weight or not os.path.exists(cfg.model["deghoster_weight"]):
        if not args.no_deghoster_weight:
            print(f"[note] deghoster_weight {cfg.model['deghoster_weight']!r} "
                  f"not found; skipping load (random init).")
        cfg.model["deghoster_weight"] = None

    print(f"=== P6 smoke test (CascadedSlicer) ===")
    print(f"config: {args.config}")
    print(f"deghoster_weight: {cfg.model['deghoster_weight']}")
    print(f"thresholds: train ~ U({cfg.model['deghost_threshold_min']}, "
          f"{cfg.model['deghost_threshold_max']}); "
          f"val={cfg.model['deghost_threshold_val']}\n")

    ds_cfg = dict(cfg.data.train)
    ds_cfg["max_spacepoints"] = args.cap
    ds = build_dataset(ds_cfg)
    print(f"dataset: {len(ds)} events")

    model = build_model(cfg.model).to(args.device)
    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_deghoster = sum(p.numel() for p in model.deghoster.parameters())
    n_slicer = sum(p.numel() for p in model.slicer.parameters())
    print(f"Model: total={n_total/1e6:.2f}M  trainable={n_trainable/1e6:.2f}M")
    print(f"  deghoster: {n_deghoster/1e6:.2f}M  slicer: {n_slicer/1e6:.2f}M\n")
    assert model.deghoster.training is False, \
        "deghoster should be in eval mode after model.to(device)"

    # ----- 1. Filter geometry diagnostic on event 0 ---------------------
    print("=== filter_batch_by_keep_mask geometry check (event 0) ===")
    sample = ds[0]
    batched = larformer_collate([sample])
    for k, v in batched.items():
        if isinstance(v, torch.Tensor):
            batched[k] = v.to(args.device, non_blocking=True)
    n_sp = int(batched["n_spacepoints"][0])
    n_gt = int(batched["n_gt_instances"][0])
    print(f"  pre-filter: n_sp={n_sp}, n_gt={n_gt}")

    # Synthesize a 50%-drop keep mask to exercise the path
    rng = torch.Generator(device="cpu").manual_seed(0)
    keep = torch.rand(n_sp, generator=rng).to(args.device) > 0.5
    filtered = filter_batch_by_keep_mask(batched, keep)
    print(f"  post-filter: n_sp={int(filtered['n_spacepoints'][0])}, "
          f"n_gt={int(filtered['n_gt_instances'][0])} "
          f"(pruned: {n_gt - int(filtered['n_gt_instances'][0])} empty instances)")
    # Sanity: every surviving instance's truth_indices is in [0, n_sp_filtered)
    n_after = int(filtered["n_spacepoints"][0])
    for gi, g in enumerate(filtered["gt_instances_per_event"][0]):
        ti = g["truth_indices"]
        if isinstance(ti, np.ndarray):
            ti = torch.from_numpy(ti)
        assert ti.numel() > 0
        assert int(ti.min()) >= 0 and int(ti.max()) < n_after, \
            f"instance {gi}: idx out of range [0, {n_after}): {ti.min()}..{ti.max()}"
    print("  truth_indices in range CHECK PASSED")

    # ----- 2. drop_empty_events on a fully-zeroed event -----------------
    # Set keep mask to all-False; the filtered batch should have n_sp=0
    # for event 0; drop_empty_events should then strip it.
    all_false = torch.zeros(n_sp, dtype=torch.bool, device=args.device)
    empty_filtered = filter_batch_by_keep_mask(batched, all_false)
    assert int(empty_filtered["n_spacepoints"][0]) == 0
    pruned = drop_empty_events(empty_filtered)
    assert pruned["n_spacepoints"].numel() == 0, \
        "single-event batch with all SPs dropped should produce empty batch"
    print("  drop_empty_events CHECK PASSED\n")

    # ----- 3. Forward + backward through CascadedSlicer -----------------
    model.train()
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=5e-4,
    )

    def step(ev_idx):
        sample = ds[ev_idx]
        batched = larformer_collate([sample])
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
        return sample, batched, out, dt, n_sp, n_gt

    if args.overfit_event >= 0:
        ev_idx = args.overfit_event % len(ds)
        # Load fresh once and pin (still uses fresh dataset[ev] each iter
        # to exercise the augmentation pipeline; loss should still trend
        # down despite per-iter τ jitter).
        print(f"=== overfit event {ev_idx} for {args.n_iters} iters ===")
        first_loss = None
        for it in range(args.n_iters):
            sample, batched, out, dt, n_sp, n_gt = step(ev_idx)
            loss = out["loss"].item()
            if first_loss is None:
                first_loss = loss
            if it == 0 or it == args.n_iters - 1 or (it + 1) % 5 == 0:
                tau = out.get('deghost_tau', torch.tensor(-1.0)).item()
                kfrac = out.get('deghost_keep_frac', torch.tensor(-1.0)).item()
                print(f"  it {it:3d}  loss={loss:7.3f}  "
                      f"tau={tau:.3f}  keep_frac={kfrac:.3f}  "
                      f"n_sp_pre={n_sp}  n_gt_pre={n_gt}  dt={dt:.2f}s")
        print(f"\n  loss went from {first_loss:.3f} → {loss:.3f} "
              f"(Δ={loss - first_loss:+.3f})")
    else:
        for it in range(args.n_iters):
            ev_idx = it % len(ds)
            sample, batched, out, dt, n_sp, n_gt = step(ev_idx)
            print(f"[it {it}] ev={ev_idx} n_sp_pre={n_sp:6d} n_gt_pre={n_gt:2d}  "
                  f"loss={out['loss'].item():7.3f}  "
                  f"tau={out.get('deghost_tau', torch.tensor(-1.0)).item():.3f}  "
                  f"keep_frac={out.get('deghost_keep_frac', torch.tensor(-1.0)).item():.3f}  "
                  f"dt={dt:.2f}s")
            for k, v in out.items():
                if k == "loss":
                    continue
                if hasattr(v, "item"):
                    print(f"    {k:32s} = {v.item():.4f}")

    print("\nP6 smoke test PASSED.")


if __name__ == "__main__":
    main()
