"""
M2F-recipe slicer + bf16 AMP — thin overlay on the m2frecipe config.

*** DO NOT USE — FAILED BENCHMARK (job 1808825, 2026-07-26) ***
Config-only bf16 autocast is BOTH slower and numerically broken here:
  - 34.96 s/iter vs 7.35 s/iter no-AMP (4.8x SLOWER), same GPU/batch/seed
  - peak mem 31.8 GB vs 38.9 GB (-18%, the only win)
  - forward collapses from iter ~3: mask logits identically 0
    (diag_mask_logit_p95 = 0.000, diag_mask_bce_rand = ln 2 exactly) — the
    decoder's NaN sanitizers are zeroing a bf16-poisoned tensor, so the
    mask/cls pathway never learns; loss pinned ~168-174 while no-AMP
    reached ~105 in the same 100 iters.
Making AMP work needs CODE changes (locate the bf16 NaN source — likely the
from-scratch PTv3 decoder / refiner path — and fp32-island it), plus a
diagnosis of the slowdown (suspect: autocast per-op overhead over the
thousands of tiny kernels in the per-pair loss loop and per-event decoding).
Kept for reference; see exp/_bench_amp/{noamp,bf16}/train.log.

Identical to `larformer-slicer-v1-cascaded-ptv3hybrid-crosslevel-m2frecipe.py`
in every respect except autocast: enable_amp=True with the base config's
amp_dtype="bfloat16". CONFIG-ONLY change — verified against the code paths:

  - Trainer (engines/train.py): with bfloat16, build_scaler returns None, so
    the step is plain backward + clip_grad + step (no GradScaler; bf16 shares
    fp32's exponent range so loss scaling isn't needed).
  - Loss precision under autocast: F.binary_cross_entropy_with_logits,
    F.cross_entropy and softmax are on autocast's fp32 policy list, so the
    BCE/CE terms compute in fp32 automatically. The manual Dice runs on bf16
    logits but its CUDA reductions accumulate in fp32; eps=1.0 keeps it
    well-conditioned. The diagnostics' torch.quantile input is explicitly
    .float()ed.
  - Matcher: the class-cost term comes from softmax (fp32 under autocast), so
    the summed cost matrix promotes to fp32 before .cpu().numpy() (numpy has
    no bf16 — a pure-bf16 cost would throw; the promotion covers it).
  - The slicer backbone's flash_attn path already ran bf16 in the no-AMP run;
    AMP extends bf16 to the linears/matmuls and the (frozen, enable_flash=
    False) deghoster's math attention.

Validation: benchmarked against the no-AMP base by
`slurm_scripts/larformer/bench_larformer_slicer_amp_a100.sh` — 100 iters each
at batch 4 / seed 42 on the 40-biggest-events list x10; compares sec/iter,
peak GPU memory, and last-15-iter mean loss. Bit-identical behavior is not
achievable (nondeterministic scatter/atomics + sampling), so "similar loss
level after ~100 iters with identical seed/data order" is the parity bar.
"""

_base_ = ["./larformer-slicer-v1-cascaded-ptv3hybrid-crosslevel-m2frecipe.py"]

enable_amp = True
# amp_dtype = "bfloat16" is inherited from the base config (required for the
# flash_attn backend; do NOT switch to float16 — the fp16 GradScaler path is
# untested with this model's sanitizer stack).

save_path = "exp/larformer_slicer_v1_cascaded_ptv3hybrid_crosslevel_cap300k_m2frecipe_bf16amp"
