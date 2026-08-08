"""Direct parity + speed test: _per_pair_sampled_mask_loss (Python loop)
vs _per_pair_sampled_mask_loss_vectorized, on identical synthetic inputs.

Both functions sample stochastically (and consume RNG differently), so
per-call outputs can't match; instead each is called n_rep times and the
MEANS are compared — they should agree in expectation, except the documented
small-instance corner (pairs with < n_target_pos positives: the loop tops up
with extra negatives, the vectorized version leaves slots empty), which the
small-mask cases below quantify deliberately.

Timing includes backward() (the loop's tiny-kernel cost hits backward too).

Run inside the pointcept container from the repo root:
    python3 slurm_scripts/larformer/test_pair_loss_parity.py
"""

import time

import numpy as np
import torch

from pointcept.models.LArFormer.losses import (
    _per_pair_sampled_mask_loss,
    _per_pair_sampled_mask_loss_vectorized,
)


def run_case(name, Q, M, K, P, density, n_sample=16392, n_rep=20, **kw):
    device = torch.device("cuda")
    torch.manual_seed(1234)
    gt = (torch.rand(K, M, device=device) < density).float()
    q_idx = (np.arange(P) % Q).astype(np.int64)
    k_idx = (np.arange(P) % K).astype(np.int64)
    base = torch.randn(Q, M, device=device)
    # Make each paired query's logits correlate with its GT mask (a
    # partially-trained-looking prediction, so BCE/Dice are off-plateau).
    for p in range(P):
        base[q_idx[p]] = base[q_idx[p]] + 4.0 * gt[k_idx[p]] - 2.0

    n_pos = int(gt[0].sum())
    print(f"\n[{name}] Q={Q} M={M} K={K} P={P} ~pos/mask={n_pos} "
          f"(n_target_pos={n_sample // 2}"
          f"{' — UNDER-FILL CORNER' if n_pos < n_sample // 2 else ''})")

    stats = {}
    for label, fn in (("loop", _per_pair_sampled_mask_loss),
                      ("vec", _per_pair_sampled_mask_loss_vectorized)):
        pred = base.clone().requires_grad_(True)
        torch.manual_seed(777)
        # warmup (also catches shape/dtype errors before timing)
        b, d = fn(pred, gt, q_idx, k_idx, n_sample=n_sample, **kw)
        (b + d).backward()
        pred.grad = None
        torch.cuda.synchronize()

        bces, dices = [], []
        t0 = time.time()
        for _ in range(n_rep):
            b, d = fn(pred, gt, q_idx, k_idx, n_sample=n_sample, **kw)
            (b + d).backward()
            pred.grad = None
            bces.append(float(b))
            dices.append(float(d))
        torch.cuda.synchronize()
        dt = (time.time() - t0) / n_rep
        stats[label] = (np.mean(bces), np.std(bces),
                        np.mean(dices), np.std(dices), dt)
        print(f"  {label:4s}: bce {np.mean(bces):.4f}±{np.std(bces):.4f}  "
              f"dice {np.mean(dices):.4f}±{np.std(dices):.4f}  "
              f"{dt * 1000:.1f} ms/call (fwd+bwd)")

    lb, _, ld, _, lt = stats["loop"]
    vb, _, vd, _, vt = stats["vec"]
    print(f"  delta: bce {abs(lb - vb):.4f} ({abs(lb - vb) / max(lb, 1e-9) * 100:.1f}%)  "
          f"dice {abs(ld - vd):.4f} ({abs(ld - vd) / max(ld, 1e-9) * 100:.1f}%)  "
          f"speedup x{lt / max(vt, 1e-9):.1f}")


if __name__ == "__main__":
    # Production sampler settings (m2frecipe: importance 0.375 + hard-neg 0.375)
    kw = dict(use_importance_sampling=True, importance_oversample_ratio=3.0,
              importance_ratio=0.375, importance_hard_neg_ratio=0.375)

    # Regular matched pairs, big masks (~10.5k pos > 8196 -> exact regime)
    run_case("P25_bigmask", Q=128, M=70_000, K=30, P=25, density=0.15, **kw)
    # Regular matched pairs, small masks (~2.1k pos -> under-fill corner)
    run_case("P25_smallmask", Q=128, M=70_000, K=30, P=25, density=0.03, **kw)
    # DN-sized pair count
    run_case("P96_DN", Q=224, M=70_000, K=30, P=96, density=0.05, **kw)
    # Pure-random negatives control
    run_case("P25_purerand", Q=128, M=70_000, K=30, P=25, density=0.05,
             use_importance_sampling=False)
