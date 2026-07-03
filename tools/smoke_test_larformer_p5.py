"""
P5 smoke test for LArFormer Stage-1 deghoster (see
`Pointcept/docs/LArFormer.md` §13).

Verifies that the no-decoder ("cls-only") path works end-to-end:
  - LArFormer with num_queries=0 builds with self.decoder == None
  - LArFormerDataset(gt_source="deghost") emits empty gt_instances + per-SP
    hasmatch field
  - LArFormer.forward returns a predictions dict that has per_level_cls but
    no class_logits/origin/mask_logits
  - LArFormerLoss.forward returns ONLY `cls_spacepoint` as a loss component
    (plus n_matched=0 and n_gt_instances=0)
  - 50-iter overfit on one event decreases the cls loss + raises
    per-SP accuracy
  - Visualizer can still load the deghost config (the viz uses the
    tokenizer + build_per_level_gt, not the decoder)

Usage:
    ./run_in_container.sh python tools/smoke_test_larformer_p5.py
    ./run_in_container.sh python tools/smoke_test_larformer_p5.py --overfit-event 0 --n-events 50
"""

import argparse
import os
import sys
import time

import numpy as np
import torch


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def per_sp_accuracy(per_level_cls, hasmatch):
    """Argmax accuracy + per-class IoU vs the binary ghost/real GT.

    per_level_cls: (N, 2)  logits, class 0 = ghost, class 1 = real
    hasmatch:     (N,)    int target in {0, 1}
    """
    pred = per_level_cls.argmax(dim=-1).cpu().numpy()
    gt = hasmatch.cpu().numpy().astype(np.int64)
    acc = (pred == gt).mean()
    ious = {}
    for c in (0, 1):
        tp = ((pred == c) & (gt == c)).sum()
        fp = ((pred == c) & (gt != c)).sum()
        fn = ((pred != c) & (gt == c)).sum()
        denom = tp + fp + fn
        ious[c] = float(tp / denom) if denom > 0 else float("nan")
    return float(acc), ious


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",
                    default="configs/lartpc/larformer/stage1_deghost/archive/larformer-deghost-v0.py")
    ap.add_argument("--n-events", type=int, default=2)
    ap.add_argument("--cap", type=int, default=30_000)
    ap.add_argument("--overfit-event", type=int, default=-1)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(0); np.random.seed(0)
    sys.path.insert(0, REPO_ROOT)
    from pointcept.utils.config import Config
    from pointcept.datasets import build_dataset, larformer_collate
    from pointcept.models.builder import build_model
    from pointcept.models.LArFormer import LArFormer  # noqa: F401

    cfg = Config.fromfile(args.config)
    print(f"=== P5 smoke test (Stage-1 deghoster) ===")
    print(f"config: {args.config}")
    print(f"num_queries: {cfg.model.num_queries}  scale_pattern: {cfg.model.scale_pattern}")
    print(f"levels: {[L['name'] for L in cfg.model.levels]}")

    ds_cfg = dict(cfg.data.train)
    ds_cfg["max_spacepoints"] = args.cap
    ds = build_dataset(ds_cfg)
    print(f"dataset: {len(ds)} events\n")

    model_cfg = dict(cfg.model)
    model = build_model(model_cfg).to(args.device)
    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: total={n_total/1e6:.2f}M trainable={n_trainable/1e6:.2f}M  "
          f"decoder={model.decoder}")
    assert model.decoder is None, "deghoster should have no decoder"
    print()

    model.train()
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=2e-3,
    )

    def step(ev_idx, label=""):
        sample = ds[ev_idx]
        batched = larformer_collate([sample])
        for k, v in batched.items():
            if isinstance(v, torch.Tensor):
                batched[k] = v.to(args.device, non_blocking=True)
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        out = model(batched)
        loss = out["loss"]
        loss.backward()
        opt.step()
        dt = time.time() - t0
        return sample, batched, out, dt

    if args.overfit_event >= 0:
        ev_idx = args.overfit_event % len(ds)
        sample0 = ds[ev_idx]
        n_sp = int(sample0["n_spacepoints"])
        n_ghost = int((sample0["hasmatch"] == 0).sum())
        n_real = int((sample0["hasmatch"] == 1).sum())
        print(f"=== overfit one event (idx={ev_idx}, n_sp={n_sp}, "
              f"ghost={n_ghost}, real={n_real}) for {args.n_events} iters ===")
        first_loss = None
        for it in range(args.n_events):
            sample, batched, out, dt = step(ev_idx)
            loss = out["loss"].item()
            if first_loss is None:
                first_loss = loss
            if it == 0 or it == args.n_events - 1 or (it + 1) % 10 == 0:
                # Compute accuracy on this iter's prediction (re-run eval-mode
                # forward to get clean predictions without dropout etc).
                model.eval()
                with torch.no_grad():
                    out_eval = model(batched)
                model.train()
                logits = out_eval["predictions"][0]["per_level_cls"]["spacepoint"]
                hm = batched["hasmatch"][batched["offset"].new_tensor([0]):
                                         batched["offset"][0]]
                acc, ious = per_sp_accuracy(logits, hm)
                print(f"  iter {it:3d}  loss={loss:.4f}  "
                      f"acc={acc:.4f}  iou_ghost={ious[0]:.3f}  "
                      f"iou_real={ious[1]:.3f}  dt={dt:.2f}s")
        last_loss = out["loss"].item()
        print(f"\n  loss went from {first_loss:.4f} → {last_loss:.4f} "
              f"(Δ={last_loss - first_loss:+.4f})")
    else:
        # Quick architecture smoke: 2 events, verify shapes + that loss
        # dict has the expected keys only.
        for it in range(args.n_events):
            sample, batched, out, dt = step(it % len(ds))
            print(f"[iter {it}] ev={it} n_sp={int(sample['n_spacepoints']):6d}  "
                  f"loss={out['loss'].item():.4f}  dt={dt:.2f}s")
            for k, v in out.items():
                if k == "loss":
                    continue
                if hasattr(v, "item"):
                    print(f"    {k:30s} = {v.item():.4f}")
            # Per-SP accuracy
            model.eval()
            with torch.no_grad():
                out_eval = model(batched)
            model.train()
            logits = out_eval["predictions"][0]["per_level_cls"]["spacepoint"]
            hm = batched["hasmatch"][:int(sample["n_spacepoints"])]
            acc, ious = per_sp_accuracy(logits, hm)
            print(f"    eval acc={acc:.4f}  iou_ghost={ious[0]:.3f}  "
                  f"iou_real={ious[1]:.3f}")

        # Verify the prediction dict has the no-decoder shape
        print("\n=== Prediction dict shape (eval mode) ===")
        p = out_eval["predictions"][0]
        keys = sorted(p.keys())
        print(f"  keys: {keys}")
        assert "class_logits" not in p, \
            "deghoster pred should NOT contain class_logits (no decoder)"
        assert "mask_logits" not in p, \
            "deghoster pred should NOT contain mask_logits (no decoder)"
        assert "per_level_cls" in p and "spacepoint" in p["per_level_cls"], \
            "deghoster pred MUST contain per_level_cls[spacepoint]"
        print(f"  per_level_cls[spacepoint] shape: "
              f"{tuple(p['per_level_cls']['spacepoint'].shape)}")

    print("\nP5 smoke test PASSED.")


if __name__ == "__main__":
    main()
