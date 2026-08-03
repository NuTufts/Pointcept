"""
M2F-recipe slicer v2 — vectorized loss + 48 queries + per-layer matching.

Overlay on `larformer-slicer-v1-cascaded-ptv3hybrid-crosslevel-m2frecipe.py`.
Motivated by the v1 run's matching-stability regression: diag_match_agreement
peaked ~0.3 then fell back to ~0.1 and stuck, with performance metrics
slipping alongside — the plan's rec-7 signature (assignment churn persisting
late). Three changes on top of the v1 recipe (user-selected, 2026-07-28):

  1. use_vectorized_pair_loss=True — 2.04x end-to-end speedup (bench job
     1809369: 3.78 vs 7.70 s/iter at batch 4; exact parity for masks >8196
     positives; small-mask corner reads dice ~21% lower at equal quality —
     documented sampling-composition shift, per-sample expectation unchanged).
     NOTE: diag_mask_bce_rand / diag_dice_rand are NOT comparable against
     loop-based runs; compare match_agreement / mask IoU / val metrics.

  2. num_queries 128 -> 48 — plan rec 6. Most events have <= 20 GT slices
     (cosmic-heavy outliers ~30); 128 COCO-scale queries mean ~85% dead
     no-object slots and near-duplicate queries competing for the same
     instance — a direct cause of assignment flapping. 48 keeps headroom
     over the outliers. Mixed query selection still draws from a 4x
     candidate pool (score_filter_multiplier=4 -> 192).

  3. loss_kwargs.match_per_layer=True — plan rec 7, standard DETR/M2F deep
     supervision: the Hungarian assignment is recomputed on EACH supervision
     layer's own predictions (shared sampled cost points), so early layers
     are trained toward the pairing their own outputs support instead of the
     final layer's. Implemented as an opt-in flag in LArFormerLoss (default
     False -> legacy single-match behavior; the v1 production chain is
     unaffected by the code change). ~7 extra no_grad scipy solves/event on
     (48 x K) costs — negligible.

Everything else (lr 1e-4 OneCycle over 5 epochs, wd 0.05 + no-decay groups,
eos 0.1, clip 0.1, diagnostics, DN, mixed query selection, 300k cap)
inherits from the v1 m2frecipe config. With the ~2x faster loss, epochs
should drop toward ~22 h (2/48h window); consider raising epoch/eval_epoch
if a longer cosine is wanted — both must move together (OneCycle horizon).
"""

_base_ = ["./larformer-slicer-v1-cascaded-ptv3hybrid-crosslevel-m2frecipe.py"]

model = dict(
    slicer=dict(
        num_queries=48,
        loss_kwargs=dict(
            use_vectorized_pair_loss=True,
            match_per_layer=True,
        ),
    ),
)

save_path = "exp/larformer_slicer_v1_cascaded_ptv3hybrid_crosslevel_cap300k_m2frecipe_v2"
