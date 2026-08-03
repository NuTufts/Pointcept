"""
M2F-recipe slicer + vectorized per-pair mask loss — thin overlay.

Identical to `larformer-slicer-v1-cascaded-ptv3hybrid-crosslevel-m2frecipe.py`
except `loss_kwargs.use_vectorized_pair_loss=True`: routes the per-pair
sampled BCE/Dice (primary mask loss, aux-mask fallback, DN path, diagnostics)
through `_per_pair_sampled_mask_loss_vectorized` instead of the per-pair
Python loop. The loop launches thousands of tiny CUDA kernels per event
(pairs x 7 supervision layers x regular+DN); the vectorized version replaces
them with ~10 large batched ops. Same sampling semantics except the
documented corner: pairs with fewer than n_target_pos (=8196) positives get
an under-filled sample (the loop version tops up with extra negatives; the
vectorized version leaves the slots empty) — per-sample expectation is
unchanged, per-pair variance slightly higher for small instances.

*** BENCHMARK PASSED (job 1809369, 2026-07-26) ***
  - End-to-end: 3.78 vs 7.70 s/iter (2.04x FASTER); peak mem 39.9 vs 38.9 GB
    (+2.4%); healthy learning (final diag_mask_logit_p95=3.5, match_agreement
    0.113 vs loop's 0.120; last-15 mean loss 109.0 vs 105.6 — within noise +
    the corner's scale shift below).
  - Function level (fwd+bwd, identical inputs): 14-21x faster per call.
    Exact-regime (masks > 8196 pos): bce/dice deltas 0.1%/0.0% — equivalent.
    Under-fill corner (small masks): vec reads bce ~5% / dice ~21% LOWER at
    equal prediction quality (fewer sampled negatives in the dice/bce set) —
    the documented semantics deviation, per-sample expectation unchanged.
  - Caveat: diag_mask_bce_rand / diag_dice_rand values are NOT comparable
    against loop-based runs (same corner affects the measurement); use
    match_agreement / mask IoU / val metrics for cross-run comparison.
Adopt for the NEXT retrain (epochs ~45h -> ~22h); do not flip mid-run (loss
scale for small instances shifts slightly, breaking within-run comparability).
"""

_base_ = ["./larformer-slicer-v1-cascaded-ptv3hybrid-crosslevel-m2frecipe.py"]

model = dict(
    slicer=dict(
        loss_kwargs=dict(
            use_vectorized_pair_loss=True,
        ),
    ),
)

save_path = "exp/larformer_slicer_v1_cascaded_ptv3hybrid_crosslevel_cap300k_m2frecipe_vecloss"
