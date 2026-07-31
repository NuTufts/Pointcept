# Domain-shift study infrastructure

Week-1 infrastructure for `../domain_shift_study_plan.md` (section 6).
Everything here consumes the frozen preservation-archive snapshots; nothing
requires resumed pretraining.

## Pieces

| file | role |
|---|---|
| `extract_features.py` | GPU batch extractor: snapshot + filelist + tier -> `.npz` of per-event pooled embeddings, prototype histograms, optional per-point samples. The measurement definition (whole-event, val_transform minus crops, tier masks before voxelization) is documented in its docstring -- do not vary it mid-study. |
| `domain_metrics.py` | PAD/C2ST (linear + kNN), multi-kernel unbiased MMD^2 with permutation p-values, prototype-occupancy JSD, CKA. Pure numpy/sklearn. |
| `bootstrap.py` | P0.2: event-level bootstrap CIs + same-domain null splits. |
| `compute_metrics.py` | Full battery on a pair of feature files -> one JSON (input to the F2/F3 figures). |
| `test_domain_metrics.py` | Synthetic calibration tests (uniform null p-values, known shifts recovered). Run after any metrics change; doubles as methods-calibration evidence. |
| `submit_extract_tufts.sbatch` | Generic one-extraction GPU job (args pass through). |
| `acceptance_test.sbatch` + `verify_acceptance.py` | 20-event end-to-end check of the extraction path (shapes, tier-mask semantics, smoke metrics). |

## Workflow

```bash
cd lartpc/pretraining_studies/domain_shift
mkdir -p logs features results

# 1. once per (snapshot x diag-set x tier):
sbatch submit_extract_tufts.sbatch \
    --config ../../../configs/lartpc/p05/pretrain-sonata-p5b1-mix-raw-detsym.py \
    --checkpoint ../../../sonata/p5b/P5B.1-mix_raw-s0/snapshot/snapshot_iter0128000_img6144000.pth \
    --data-list ../../../lartpc/filelists/h5list_v3_mc_diag1k_tufts.txt \
    --tier cosmic \
    --out features/P5B.1_img6144000_mc_cosmic.npz
# (paths relative to the repo root also work; the job cd's to the repo)

# 2. CPU, minutes per pair:
python3 compute_metrics.py \
    --a features/P5B.1_img6144000_mc_cosmic.npz \
    --b features/P5B.1_img6144000_data_cosmic.npz \
    --label "P5B.1@6.1M cosmic" --out results/P5B.1_img6144000_cosmic.json
```

Feature-file naming convention: `<run>_img<imagesseen>_<sample>_<tier>.npz`.

## First results (2026-07-28, P5B.1 @ 6.14M images, diag1k sets)

| comparison | AUC(lin) | MMD^2 | proto JSD | excl protos |
|---|---|---|---|---|
| null (same-domain splits) | 0.50 +- 0.03 | ~0.0006 | 0.003 | -- |
| Tier 0 raw vs raw | 0.9991 | 0.107 | 0.011 | 0 / 0 |
| Tier 1 nu-masked MC vs data | 0.9990 | 0.109 | 0.011 | 0 / 0 |
| Tier 1-clean truth-MC vs lm-data | 1.0000 | 0.432 | 0.059 | 0 / 52(data) |
| Tier 1-lmclean SYMMETRIC lm cut | 0.9996 | 0.102 | 0.009 | 0 / 1 |

Readings: (1) nu-content composition explains ~none of the event-level gap
(Tier 0 == Tier 1 within CIs); the gap is cosmic/detector/ghost-level.
(2) Method-asymmetric cleaning (truth on MC, LArMatch on data)
QUADRUPLES the measured gap and creates 52 data-exclusive prototypes;
the SYMMETRIC lm cut returns the gap to (marginally below) the raw level
and the exclusive prototypes vanish -- the entire amplification was
cleaning-method asymmetry, not LArMatch response mismodeling. Direct
quantitative endorsement of the P5B.3 symmetric-filter design; warning
for truth-cleaned-MC deployment pipelines (P5B.2 pairing).
(3) AUC/PAD saturate; MMD^2 and proto JSD are the graded statistics for
F2/F3. Full numbers with CIs/nulls: `results/P5B.1_img6144000_tier*.json`.

Driver diagnostic (`results/*_drivers.json`, figures/): the best single
simple observable reaches AUC 0.672 (q90 charge: data ~8% higher; data
also ~20% fewer points/event) vs the embedding's 0.999 -- the separation
is structural, not a global scale; simple summary variables cannot see
most of what the embedding sees. LArMatch score distributions are closely
matched across domains (mean 0.598 MC / 0.586 data; figure). Prototype
frequency ratios: MC-enriched extremes up to 49x (proto 2421 -- candidate
sim-only pattern, inspectable via saved per-point samples), data-enriched
max ~3.6x.

## Cross-model + DCTR results (2026-07-29, matched anchor img6144000)

Normalized gap (MMD^2 / perm-null-95 -- the cross-model-comparable
statistic; raw MMD^2 is not comparable across embedding spaces):

| model | tier1 gap | kNN AUC | tier1clean | tier1lmclean |
|---|---|---|---|---|
| P1A.2 (MC-only) | 95x | 0.932 | 510x | (pair rerunning) |
| P1A.3 (data-only) | 125x | 0.951 | 491x | 155x |
| P5B.1 (mixture) | 111x | 0.971 | 492x | 135x |

Readings: (1) The tier ordering (tier0 ~= tier1 << asymmetric-clean;
symmetric lm-clean ~= raw) REPLICATES in all three models -- the
decomposition is a property of the data, not of one embedding.
(2) Joint pretraining does NOT shrink embedding-level separability; the
mixture model separates the domains slightly MORE than the MC-only model
(it has learned data-specific features; single-domain models project the
other domain onto their own axes). Separability also GROWS with training
(P5B.1 tier1: 53x @1.5M -> 111x @6.1M images). Mixture = the faithful
measurement instrument; curation alone does not close the representation
gap -> explicit alignment/conditioning is genuinely proposed work.
(3) CKA (aligned events): P5B.1 is closer to BOTH single-domain models
(0.981-0.989) than they are to each other (0.974-0.979) on both eval
sets -- the mixture bridges the two representations.

DCTR reweighting demo (`results/*_dctr.json`, figures/dctr_*.png),
method-symmetric clean pair: usable only in the leading PCA dims --
d=4: AUC 0.68, ESS 0.69, closure AUC 0.55; d>=16: ESS < 0.06 and closure
fails (support mismatch at near-separability, as predicted). At d=4 the
weights move points/event +43% of the way toward data but move charge
scale -34% (WRONG direction): low-d embedding reweighting captures only
the leading mismatch directions. Path forward for F5: 5k diag sets
(power), conditional/point-level reweighting, or transport maps.

## F3 ladder, COMPLETE (2026-07-31; all 10 snapshots)

`figures/f3_gap_vs_images.png` (P5B.1, tier1): MMD^2/null95 grows
monotonically 11x (24k images) -> 111x (6.1M), with a knee: slow growth
through ~400k images, then steeper power-law. kNN AUC is FLAT at ~0.63
across all five snapshots up to 384k images, then climbs steeply to 0.97
-- domain-specific representation learning "turns on" between ~0.4M and
1.5M images; the 0.63 plateau is the constant input-level domain signal
the embedding passes through before that. Prototype JSD is non-monotonic:
2.5e-3 -> peak 0.015 at ~0.8M -> consolidates to ~0.010. The JSD peak
coincides with the AUC takeoff: the model first splits vocabulary USAGE,
then re-merges the vocabulary while moving domain information into finer
structure -- which the point-level study then showed is a small coherent
per-point shift (AUC ~0.61) amplified by pooling.

## Point-level results (2026-07-31; POINT_LEVEL_PLAN.md items 1-4)

The CLT hypothesis is CONFIRMED. Per-point domain AUC (event-grouped
folds, PCA-64) vs event-level AUC on the same files:

| pair | point linear | point HistGB | event linear | point MMD/null95 |
|---|---|---|---|---|
| P5B.1 tier1 | 0.612 [0.603,0.628] | 0.632 | 0.999 | 18x |
| P5B.1 lmclean | 0.584 [0.577,0.589] | 0.612 | 1.000 | 10x |
| P1A.2 tier1 | 0.624 [0.613,0.639] | 0.638 | 0.998 | 20x |
| P1A.3 tier1 | 0.637 [0.625,0.650] | 0.656 | 0.999 | 21x |

Readings: (1) The LOCAL gap is modest (AUC ~0.6); event-level saturation
is the pooling (CLT) amplification of a small coherent per-point shift.
Headline sentence: "a per-point shift of AUC ~0.61 accumulates over ~1e5
points into event-level AUC ~0.999". (2) HistGB adds little over linear
-- the local gap is approximately linearly encoded too. (3) At point
level the MIXTURE model has the SMALLEST local gap (0.612 vs 0.624/0.637)
-- opposite ordering to the event-level normalized MMD: joint training
aligns local features while event-level context stays separable.
(4) Symmetric lm-cleaning lowers the local gap further (0.584): part of
the local shift lives in ghost-region points. (5) Prototype-conditional
shifts: ~3900 prototypes tested; ~89% show excess shift over the
split-half null (median 1.97 vs 1.28) -- the gap is DISTRIBUTED across
the vocabulary, not concentrated; top offenders (P5B.1: protos 519,
3827, 891, 1767, 2553; excess up to 20) overlap with the
frequency-ratio outliers (519 was 9x MC-enriched) -- strongest
candidates for identifiable sim-specific patterns. Figures:
point_vs_event_auc_*.png, proto_shift_ranking_*.png (note: pre-slug
proto figures were overwritten by the last-finishing job; regenerate per
pair if needed for the proposal).

## Point-level DCTR closure (2026-07-31; P5B.1 tier1 pair, 2M pts/side)

The support-overlap prediction is confirmed dramatically
(`results/P5B.1_img6144000_dctr_points.json`, figures/dctr_pts_*.png):

- ESS/N stays 0.75-0.97 across the ENTIRE dim scan d=8..128 (event level
  collapsed below 0.06 at d>=16), and the closure test succeeds at every
  dim (reweighted AUC 0.49-0.51). Report dim 128, ESS 0.755.
- Every physics observable now moves TOWARD data (event-level d=4 moved
  charge the wrong way): pixU/V/Y means close +40/+25/+48% of their gaps,
  plane ratios U/Y +39%, V/Y +12%, and the lm-score mean closes fully
  (+120%, overshoot 0.003).
- Profile closure (mean |resid|): pixY vs x 2.78->2.53, vs y 2.45->1.33,
  vs z 3.72->2.94; lm vs x 0.013->0.008, vs y ->0.003, vs z ->0.006.
- Physics readings: (1) MC lm scores sit systematically ABOVE data (MC
  points more confidently track-like) with a y-dependent slope;
  reweighting lands almost exactly on data -- the weights are largely
  repairing ghost/LArMatch-response composition, consistent with the
  symmetric-clean tier result. (2) The z~700 cm dead-wire dip appears
  identically in both domains and is untouched by reweighting -- the
  method does not distort well-modeled features. (3) pixY closes more
  strongly vs y (flux/charge-scale axis, -46%) than vs x (drift axis,
  -9%): either the drift-response mismatch is subdominant or the
  out-of-time t0 dilution hides it; the t0-taggable
  anode/cathode-piercing subsample is the designed follow-up.

## Conventions / gotchas

- Use the *run's own config* for extraction (asinh runs need their asinh
  config, P5E widths their width configs) -- preprocessing must match what
  the checkpoint was trained with.
- Diag lists only (`*_diag1k_tufts.txt`): they are held out of ALL training.
- The extractor turns OFF every loader-side filter and applies tier masks
  itself, so the loader's too-few-points retry can never silently swap
  events. Skipped events are recorded in the `.npz`, never substituted.
- a100 constraint (flash_attn needs sm80+; and matches the probe-sweep
  arch pin). `mkdir -p logs` before the first sbatch (SLURM exit-53 gotcha).
- `features/` and `results/` are gitignored artifacts; JSONs carry full
  provenance (checkpoint, list, tier, seeds) in their `meta`.
