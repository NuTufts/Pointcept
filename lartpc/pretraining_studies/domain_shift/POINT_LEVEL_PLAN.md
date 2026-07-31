# Point-level domain-gap study plan (2026-07-30)

Extension of `../domain_shift_study_plan.md`: repeat the gap measurements on
per-spacepoint embeddings instead of event-pooled vectors. Approved by PI
2026-07-30 (chat); this file is the record of scope and statistical rules.

## Why (what the event level cannot answer)

1. **The CLT question.** Event level shows near-shared prototype vocabulary
   (JSD 0.011, no exclusive prototypes) yet perfect event separability
   (AUC 0.999). Hypothesis: pooling ~1e5 points averages away per-point
   noise, so a small coherent per-point shift becomes a separable
   event-level mean shift. The point-level AUC measures the LOCAL gap
   directly; if modest (guess 0.55-0.70), the event-level saturation is a
   pooling artifact of a small local gap -- a much better headline.
2. **Localization.** Score points with the domain classifier; overlay
   high-score points on event displays: WHERE is the mismodeling.
   Prototype-conditional shifts separate "vocabulary used at different
   rates" from "vocabulary internally shifted".
3. **A usable reweighting space.** Event-level DCTR collapses above d~8
   (support mismatch). Point distributions overlap far more; point-level
   density ratios should keep healthy ESS, and point weights propagate to
   segmentation-type downstream quantities (probe IoU per class).

## Statistical rules (non-negotiable)

Points within an event are correlated (shared noise realization, gain,
cosmic multiplicity). Therefore:

- Classifier CV: **event-grouped folds** (StratifiedGroupKFold; groups =
  event). Plain KFold inflates AUC via event memorization.
- MMD permutation null: **permute events between domains** (block
  permutation), never points.
- Bootstrap: **resample events**, carry their points.
- Interpretation caveat carried everywhere: even with grouped folds, a
  point can betray its EVENT context (e.g. a global gain shift imprints on
  every point), so point-level AUC = local structure + event-context
  leakage.

## Work items

1. `point_metrics.py`: grouped-fold point PAD (linear + HistGB on PCA-64),
   event-block-permutation MMD^2 on point subsamples, event-blocked
   bootstrap. [core]
2. Dense point extraction: `--points-per-event 2048` on the anchor
   (img6144000) tiers, written as `*_pts2048.npz` (event-level files stay
   untouched): P5B.1 mc_cosmic / data_raw / mc_cosmiclmclean /
   data_cosmicclean; P1A.2 + P1A.3 mc_cosmic / data_raw for the
   cross-model point comparison. [core]
3. `point_level_battery.py`: point AUC (grouped) vs event AUC from the same
   files (the CLT comparison), block-MMD with p-value, per-point vs
   event-level summary JSON + figure. [core]
4. Prototype-conditional gap: for prototypes with enough points both sides,
   standardized within-prototype mean shift + event-bootstrap CI; ranked
   table + gallery of the most-shifted prototypes (point_coord scatter).
   [follows 3]
5. Point-level DCTR: same dim-scan/ESS/closure protocol as the event demo;
   propagate to pixval spectrum and (later) probe-class composition.
   [follows 3]

Measurement definition: same extractor, same tiers, same frozen diag sets;
point features are the upcast (stride-4) teacher features, fixed-seed
subsample of 2048/event, float16.
