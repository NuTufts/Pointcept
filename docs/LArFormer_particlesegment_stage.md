# LArFormer Stage 3 — Particle segmenter

**Status:** IMPLEMENTED and in production training (2026-06-11). §§1–11 are
the design plan (decisions there held up well); **§13 is the as-built
record** — start there for what actually exists. The active training
campaign, its loss-stability analysis, the per-batch diagnostics, and the
inference-side query dedup live in
[LArFormer_Stage3_TrainingStability.md](LArFormer_Stage3_TrainingStability.md).
Project hub: [LArFormer.md](LArFormer.md) §0.

**Scope:** instance segmentation of the spacepoints inside a (model-selected) neutrino slice into individual particles. Same architecture pattern as the Stage-2 event slicer; new decoder / refiner / head weights; GT comes from per-particle truth instead of per-slice truth.

This document lays out the implementation plan; design choices that are still open are flagged explicitly in §10.

---

## 1. Architecture (model)

### 1a. Re-use Stage 2's `LArFormer` class with new weights

The Stage-2 slicer's architecture (tokenizer → token-refiner → Mask2Former decoder + heads) is the right inductive bias for particle segmentation. The change is purely:

- **Frozen backbone**, same Sonata-v1m1 checkpoint as Stages 1 + 2. Same per-SP encoder features.
- **New `token_refiner` weights** (PerLevelSelfAttn or CrossLevelAttn — same code, separate weights).
- **New `Mask2FormerDecoder` weights** + new per-query class head + new mask head + per-query origin head.
- **`mixed_query_selection`** stays on, source level the same (`voxel_8cm` or `voxel_10cm`).
- **`mask_denoising`** stays on, with particle-aware noise (see §5).

### 1b. Cascade plumbing — start model-side, measure, then decide

The cascade integration mode (model-side vs dataset-side caching) is **decided empirically** rather than up-front:

- **Implement model-side cascade FIRST** (`CascadedParticleSegmenter` extending the existing `CascadedSlicer` pattern). This has to be built regardless — it's how the model runs on real data — so it's also the baseline against which a caching shortcut would be compared.
- **Measure training iteration time + GPU memory** under model-side cascade with frozen Stages 1+2.
- **Measure data-cascade-side cost** for comparison: per-event cache file size + the one-time bulk inference cost (already characterized from `run_slicer_inference.py` timing).
- **Decide caching only if model-side cascade is too slow** — it's a Phase-2 optimization, not a Phase-1 architecture decision.

This means S3.1 builds the production path; S3.2 is the benchmark; S3.3+ uses whichever winner emerges.

### 1c. Per-query class taxonomy

7 classes, between the minimal-physics and PDG-detailed extremes:

```
{ e+/-,  γ,  μ+/-,  π+/-,  p,  other_track,  no_object }
```

Rationale:
- Lepton identification matters (e vs μ separates CC νₑ from CC νμ).
- π0 reco needs to count γ's, so γ is its own class.
- Charged-pion identification matters for π0 + π± topology separation.
- Protons their own class (HIP signature is distinctive).
- All other tracks (K±, n, exotic primaries) get bucketed into `other_track`.
- `no_object` is the matcher's slot for unmatched queries (and absorbs cosmic-leakage SPs — see §2c).

Adjustable post-v1; the per-class IoU plots in S3.6 will tell us if any class is starved.

### 1d. Query count

K=32 queries. The nu slice typically has 1–15 particles (BNB events mostly ≤8); 32 gives ~4× headroom, mirroring Stage 2's 4× over-provisioning. Adjustable later.

### 1e. Backbone features at the nu-slice subset

**Re-use Stage 2's backbone features**, sliced by the nu-mask. The forward pattern: Stage 2 runs once and produces per-SP backbone features for the full event; Stage 3 reads those features at the SPs that pass the (loosened — see §3) nu-mask threshold.

This avoids re-running Sonata on a smaller input (sparse-conv neighborhoods would change → potential feature destabilization) and is the cheapest option. A "re-run encoder on nu-only set" variant is logged as a Phase-2 ablation in S3.9 if performance disappoints.

### 1f. Coordinate system — recenter to slice centroid

**Per-event coord recentering** before the decoder's pos_emb:

```python
coord_centered = coord_norm - centroid(nu_slice_SPs)
```

The Stage-3 decoder consumes `coord_centered` (not `coord_norm`) wherever pos_emb is computed. Query anchors from `mixed_query_selection` are likewise recentered.

Rationale:

1. **The frozen backbone features already encode absolute-position context.** Sonata's encoder + decoder ran over the full event with absolute coords; its hierarchical sparse-conv neighborhoods leave "where in the detector is this slice" baked into the per-SP feature vector that Stage 3 starts from. Pos_emb at Stage 3 should add NEW information; the natural orthogonal axis is "where in the slice am I."

2. **Translation invariance helps query specialization.** With recentered coords, a learned query like "find electrons upstream of the centroid" generalizes across all events. Absolute coords would force each query to relearn the same pattern at every (x, y, z) in the TPC — wasted parameters.

3. **Detector-region effects we DO care about can be added as aux features cheaply** (see §1g, ablation).

**Items unaffected by this choice** — they all happen post-Stage-3 in the analyzer-side pipeline and use absolute coords:
- Flash chi² (PhotonLib visibility lookup at detector cm).
- OOB rejection against the active-TPC bounds.
- Drift correction using absolute x.

These are independent of what coord system the model's pos_emb saw at training time, because the per-event H5 outputs carry absolute SP positions.

### 1g. Auxiliary position feature — deferred as S3.9 ablation

A small extra per-SP feature channel could carry detector-region info that the recentered pos_emb discards:

| Aux channel | Meaning | Cost |
|---|---|---|
| `x_absolute_norm` | Drift coordinate in coord_norm frame | 1 dim |
| `dist_to_x_min`, `dist_to_x_max` | Distance to cathode / anode | 2 dims |
| `dist_to_y_boundary`, `dist_to_z_boundary` | Distance to TPC walls | 2 dims |
| `centroid_xyz_abs` | One-shot per-event "where is this slice" (broadcast to every SP) | 3 dims (constant per event) |

**v1: do NOT include these** — recenter only, and let the backbone features carry whatever absolute-position context the frozen Sonata learned.

**S3.9 ablation**: turn on a 3–8-dim aux channel and measure whether per-particle IoU improves. This is also a clean **probe of what the backbone encodes**: if adding `x_absolute_norm` helps a lot, Sonata under-represented drift context; if it doesn't help, the frozen features already had it.

---

## 2. GT definition

### 2a. Particle granularity — primary + visible secondaries, particle-agnostic with energy thresholds

GT instances are constructed by walking the MC particle tree (`entry_0/mc_particle_tree`) for tracks with `origin == 1` (nu-origin) and keeping any track — primary OR secondary — whose initial kinetic energy passes a per-PDG threshold:

| PDG | KE threshold |
|---|---|
| e±  (11)  | 10 MeV |
| γ   (22)  | 10 MeV |
| μ±  (13)  | 30 MeV |
| π±  (211) | 30 MeV |
| p   (2212) | 60 MeV |
| other hadrons (n, K±, etc.) | 60 MeV |

A particle below threshold gets merged into its parent's instance (its SPs are assigned to the parent's slice_id). This avoids over-fragmenting events with many low-energy δ-rays / brems while still capturing the visible secondaries that matter for π0 reco (the γ's), high-energy bremsstrahlung off the μ, etc.

**Implementation:** new function in `lartpc/data_prep/labels/slice_labels.py`:

```python
def compute_particle_labels(mpt_group, sp_trackid, sp_hasmatch,
                             ke_thresholds=DEFAULT_KE_THRESH,
                             nu_origin=1):
    """Per-particle slice labels — particle-agnostic with KE thresholds."""
```

Returns the same dict shape `compute_slice_labels` returns (slice_id per SP, primary_trackid array, etc.). The merging-low-KE-into-parent logic walks the parent chain upward until it finds an above-threshold ancestor.

A new `gt_source="particle"` plug in `LArFormerDataset` calls this routine.

### 2b. SP → GT assignment

Per-SP truth has `trackid`. Each SP gets assigned to its above-threshold ancestor's slice id via the merging logic above. Existing `lartpc/data_prep/labels/slice_labels.py` machinery is the foundation; the new routine just plugs in a different walk termination.

### 2c. Cosmic-leakage SPs → `no_object`

If Stage 2 over-claimed cosmic SPs into the nu slice (the `sp_level_nu_precision=0.61` finding from §11 of `LArFormer.md`), those SPs have no particle GT inside the nu interaction.

**Treatment:** assign GT `slice_id = -1` for cosmic-leakage SPs. The Hungarian matcher's `no_object` slot absorbs them. Stage 3 *learns to leave cosmic leakage in no_object*, which is the right behavior at panoptic-segmentation time. No dedicated "background" class needed.

---

## 3. Input-set definition (the nu slice that Stage 3 sees)

**Loosened Stage-2 mask threshold**. For each Stage-2 nu query, the input SP set is `pred_mask_prob > τ_loose`, with `τ_loose < 0.5` (the default panoptic-argmax threshold). This recovers under-clustered SPs (long muon ends, shower tails) that Stage 2's panoptic argmax dropped.

**τ_loose sweep**: `{0.3, 0.5, 0.7}` for ablation (S3.7). Same τ in train and eval to avoid train-eval mismatch. Open call on whether to make τ per-event-adaptive (e.g., as a function of n_pred_nu slices) — answered after the static-τ ablation.

**KNN halo expansion** (`§3b option 2` in the prior draft) is deferred to Phase 2: it adds per-event neighbor-search cost and is a Phase-2 optimization to revisit only if the loosened-τ remedy doesn't fully close the under-clustering gap.

---

## 4. Cascade conditioning at training time

### 4a. Pure cascade is the primary training path

For training Stage 3:

- **Pure cascade.** Run Stages 1+2 in the forward pass, take their output, then run Stage 3. Slowest but most faithful to real-data inference. **This is the path to benchmark against** — any paper needs to show the model works with real Stage-2 output, not just truth-conditioned input.
- **Noised-truth approximation.** Drop the cascade in training; use noised GT nu mask as input. Faster training; evaluated against the pure-cascade benchmark.

Concretely:
- **S3.3 trains the pure-cascade version** first. This establishes the headline number.
- **S3.7 ablates noised-truth approximation** as a training-speed alternative. If it matches pure-cascade quality, it becomes the production training path for future iterations of Stage 3 (e.g., LArFormer retrains where we want to iterate fast).

### 4b. Vertex / flash conditioning

Per-particle "starting point" (origin) is predicted by Stage 3's per-query origin head — that's covered in §6.

**Per-event nu vertex prediction is OUT of scope** for Stage 3. A separate model (call it "Stage 2.5 — nu vertex finder") would run alongside Stage 3 to predict the nu vertex from the same nu-slice input. Stage 3 does not consume a vertex token.

This keeps Stage 3's responsibility scoped: instance-segment the nu slice into particles. Vertex finding is an independent reco task.

---

## 5. Mask denoising for Stage 3

Same machinery as Stage 2's `MaskDenoiser`, with adaptations.

### 5a. Noise model (shared between INPUT and DN paths)

**Single shared noise procedure** that defines both:
- The Stage 3 *regular path's* input SP set (when running noised-truth-approximation training in §4a's alternative path).
- The Stage 3 *DN path's* per-particle noised masks.

The procedure operates on the whole-event SP set (we have access via the frozen Stage-2 backbone forward), and can both:

- **Drop** SPs from a particle's GT mask (probability calibrated per-class — long μ ends drop more; well-formed showers drop less).
- **Add** SPs from OUTSIDE the GT mask, drawn from a spatial neighborhood around the particle (probability calibrated per-class; KE-shower particles attract more "halo" additions).

Because the procedure can both drop and add, it can mimic Stage 2's failure modes:
- *Under-clustering* — Stage 2 drops endpoints → simulated by drop noise on truth.
- *Over-claim* — Stage 2 adds cosmic-leakage SPs → simulated by add noise from whole-event SPs.

### 5b. Noise model calibration

A one-time calibration pass:
- Run Stage 2 over a portion of the training set.
- Per Stage-2-predicted nu query, compute per-SP `(p_drop, p_add)` against truth.
- Fit a parametric model — features available per SP: distance-from-trunk, distance-from-vertex, per-class indicators.

The fit becomes a frozen artifact loaded by the dataset at training time. Output: `noise_params_stage2_<weights_hash>.json` (or similar).

### 5c. DN groups + per-event cap

`dn_groups = 3` (same as Stage 2's setting). With K=32 and ≤15 GT particles per event, total DN queries ≤45 typical. `max_dn_per_event = 96` for safety (rarely hit).

### 5d. Anchor jitter

Per-particle anchor jitter σ. Per-class scaling: showers diffuse more (σ = 0.07 in coord_norm ≈ 12 cm); MIP tracks tighter (σ = 0.03 ≈ 5 cm). Adjustable per ablation.

---

## 6. Loss budget

Reuse Stage 2's `LArFormerLoss` skeleton. Per-component weight defaults:

| Component | Weight | Note |
|---|---|---|
| `weight_class` | 2.0 | Same as Stage 2 |
| `weight_mask_primary` | 5.0 | At per-SP level |
| `weight_dice_primary` | 5.0 | Same |
| `weight_aux_mask` | 0.3–0.7 | Per-voxel aux mask supervision |
| `weight_per_level_cls` | 0.5 | Per-voxel class supervision |
| `weight_origin` | **0.5 (ON)** | Per-particle starting-point regression — Stage 3 has the origin head active |
| `weight_dn_loss` | 1.0 | Same as Stage 2 |
| `no_object_weight` | 0.1–0.5 | Tuned against per-particle GT count; expect lower than Stage 2's because there are typically 5–15 particle GTs vs 30 slice GTs |

`enable_origin_head=True` is mandatory for Stage 3 (per §4b).

---

## 7. Data-prep — per-event load cost

Reduced point count is the big win: typical nu slice is ~5–15K SPs (down from ~50–150K full event). Cost depends on §1b's choice:

| Approach | Estimated per-event cost (training) |
|---|---|
| Model-side cascade (Stages 1+2 in fwd pass) | ~500ms — drops to ~200ms with batching |
| Dataset-side cache | ~50ms after one-time cache build |
| Noised-truth approximation (no Stages 1+2 in train) | ~50ms + ~5ms noise sample |

S3.2 measures these for the actual stack and decides whether caching is needed. The default is **try model-side cascade first** — it's the production path.

### 7a. Caching strategy (only if S3.2 shows we need it)

If model-side cascade is the bottleneck, the dataset-side cache stores per-event:

```
.../stage3_inputs/<TAG>/cache_<TAG>_fileno<NNNNN>_entry<NNNNNN>.h5
  attrs:
    stage1_weights_hash
    stage2_weights_hash
    stage2_config_hash
    tau_loose                    — the loosened mask threshold used to define input set
  entry_0/
    sp_keep_mask   (N_pre,) bool — SPs in the nu candidate set
    particle_gt_id (N_pre,) int64 — per-SP GT particle slice id (or -1 = cosmic-leakage)
    backbone_feat  (N_keep, D) f16 — sliced Stage-2 backbone features (or build at load time)
```

Cache invalidation: hash mismatch → rebuild.

---

## 8. Evaluation

### 8a. Per-particle IoU

Same Hungarian-matched per-pair IoU the Stage-2 evaluator computes. Reported per-class (e±, γ, μ±, π±, p, other_track).

### 8b. Topology recovery (the science target)

Per event:
- **Correct topology** = Hungarian-matched particle assignments AND per-class IoU > 0.5 → "the analyzer would have reconstructed this event correctly."
- **Topology categories**: 1-γ, 2-γ (π0), 1-μ + N protons, etc. — categorical recovery rate per category.

Plus the existing categories (CC νμ, CC νₑ, π0, 1-vis-γ) from Stage 2's analyzer.

### 8c. Stage-cascade evaluation

New tool: `tools/run_full_cascade_inference.py` runs Stages 1 → 2 → 3 end-to-end on val+test, emits per-event H5s of:
- Per-SP particle ID (panoptic)
- Per-event topology classification
- Per-particle metrics

Reuses the analyzer-side pipeline architecture from [`lartpc/larformer_analysis/slicer_eval/`](../lartpc/larformer_analysis/slicer_eval/) — Stage 3 outputs a per-event H5, an aggregator builds an event_summary.h5, plots produce per-category headlines.

---

## 9. Implementation phasing

Revised order, with the model-side-first benchmark approach:

| Phase | Deliverable | Gating |
|---|---|---|
| **S3.0 — Particle GT extraction** | `compute_particle_labels` in `slice_labels.py` with KE-threshold + merging logic. Smoke-tested on a few events. | Independent — can start any time. |
| **S3.1 — Model-side cascade scaffold** | `CascadedParticleSegmenter` (or extend `CascadedSlicer`) that wraps frozen Stages 1+2 + a fresh Stage 3 `LArFormer`. Stage 3 reads backbone features from Stage 2's forward (per §1e). New Stage-3 config `larformer-particle-v1.py`. Smoke test: build, forward, backward on synthetic input. | Independent of S3.0; can build alongside |
| **S3.2 — Benchmark** | Measure per-iter time + GPU memory for the model-side cascade. Compare against the data-cascade-side cost estimated from existing `run_slicer_inference.py` runs. Decide whether caching (per §7a) is needed for v1. | S3.1 |
| **S3.3 — Pure-cascade training** | Train Stage 3 with model-side cascade, loosened τ (per §3), pure-cascade GT (per §4a's primary path). No DN yet, no noise. Establish the benchmark number. | S3.0 + S3.2 |
| **S3.4 — Per-particle origin head** | Add per-query origin regression to Stage 3. Train with `weight_origin = 0.5`. | S3.3 |
| **S3.5 — Noise model calibration** | Run Stage 2 over training set, fit per-particle drop/add noise (per §5b). Save as JSON artifact. | Stage 2 trained (have it). S3.0. Independent of S3.3/S3.4. |
| **S3.6 — Mask denoising on Stage 3** | Add `MaskDenoiser` (Stage 2's class, with per-particle noise sourced from S3.5's fit). Both DN path and regular-input noise use the same shared procedure. | S3.4 + S3.5 |
| **S3.7 — Noised-truth approximation alternative** | Train Stage 3 with noised-GT input instead of pure cascade. Compare quality vs S3.3 + train-speed gain. If matches: this becomes the future-iteration training path. | S3.5 |
| **S3.8 — Full cascade eval tool** | `tools/run_full_cascade_inference.py` + per-event analysis (mirrors `larformer_analysis/`). Per-particle / per-category headline metrics. | S3.6 |
| **S3.9 — Ablations** | τ_loose sweep; class-taxonomy variants; KNN-halo input (§3 Phase 2); separate-vs-shared DN-and-input noise samples; re-run Sonata on nu-only set (§1e Phase 2); **aux absolute-position feature on/off (§1g) — also a probe of what the backbone encodes**. | S3.8 baseline |

S3.0, S3.1, S3.5 are independent and parallelizable.

**Estimated effort** (rough):

- S3.0: 1–2 days (per-particle labels with energy thresholds; MC tree walk)
- S3.1: 3–4 days (model-side cascade wiring; smoke test)
- S3.2: 1–2 days (benchmark + decision)
- S3.3: 1–2 days code + 1–3 days training run
- S3.4: 1 day code + 1 day training
- S3.5: 1–3 days (depends on how the noise fit converges)
- S3.6: 1–2 days code + 1–2 days training
- S3.7: 1 day code + 1–3 days training
- S3.8: 2–3 days
- S3.9: open-ended

Total to S3.8 baseline: ~3–4 weeks of work.

---

## 10. Open questions (remaining)

Most of the design choices are now decided. What's still open:

1. **τ_loose value** for the loosened Stage-2 mask threshold (§3). The S3.9 ablation sweeps `{0.3, 0.5, 0.7}`; pick the right starting point for S3.3.

2. **Per-class noise parameters** for the calibration in §5b. The fit's degrees of freedom + how to regularize. Worth pre-discussing what the noise model's parametric form should be (per-class additive vs. multiplicative; whether to also model spatial correlation).

3. **Train-eval τ consistency** (§3): same τ in both, OR allow eval-time τ tuning per event? The recommendation is "same in both" for v1, but we may want eval-time τ as a per-event knob for analysis use.

4. **Whether to require a nu vertex finder in the cascade** before Stage 3 trains. The current plan says no — vertex finding is a separate model. But if Stage 3's per-particle origin head benefits substantially from knowing the nu vertex (as an extra token / starting point), we'd need the vertex finder online by S3.4. Pending: ablation in S3.9 of "with nu-vertex truth conditioning" vs without.

5. **`no_object` weight for Stage 3** (§6, `no_object_weight`). Stage 2 used 0.1–0.5; Stage 3's much smaller GT count means the matcher has many more unmatched queries per event → expect a different optimum. Worth a sweep at S3.6.

---

## 11. References

- Stage 2 spec: [`LArFormer.md`](LArFormer.md) §6 (cascade), §7 (dataset)
- Stage 2 mask-denoising: [`LArFormer.md`](LArFormer.md) §18
- Stage 2 mixed query selection: [`LArFormer.md`](LArFormer.md) §17
- Existing `compute_slice_labels` (which `compute_particle_labels` will extend): [`lartpc/data_prep/labels/slice_labels.py`](../lartpc/data_prep/labels/slice_labels.py)
- Analyzer-side category definitions: [`lartpc/larformer_analysis/slicer_eval/lib/categorize.py`](../lartpc/larformer_analysis/slicer_eval/lib/categorize.py)
- Per-event analysis pattern (template for S3.8): [`lartpc/larformer_analysis/slicer_eval/`](../lartpc/larformer_analysis/slicer_eval/)
- Stage 2 cascade wrapper (template for `CascadedParticleSegmenter`): [`pointcept/models/LArFormer/cascaded.py`](../pointcept/models/LArFormer/cascaded.py)

---

## 12. Revision history

- **Rev 1 (2026-05-30)**: initial draft with 8 open questions; recommendations for cascade location (dataset-cache), class taxonomy (detector-signature), visible-secondary GT (γ from π0 only), and DN-as-additional-noise.
- **Rev 2 (2026-05-30)**: revised per user feedback:
  - Cascade location: model-side first, benchmark, then decide on caching (§1b).
  - Class taxonomy: 7-class middle option `{e±, γ, μ±, π±, p, other_track, no_object}` (§1c).
  - GT granularity: particle-agnostic with KE thresholds (10/30/60 MeV by species) (§2a).
  - Cascade training mode: pure-cascade primary, noised-truth-approximation as S3.7 ablation (§4a).
  - Vertex: per-particle origin head YES, per-event nu vertex as a separate model OUT of Stage 3 (§4b).
  - DN noise: shared sample for both input and DN paths (§5a). Both drop and add components.
  - Phasing: S3.0–S3.9 reflecting model-side-first plus the noise-calibration + DN sub-phases.
- **Rev 3 (2026-05-30)**: coordinate system decision:
  - Recenter pos_emb input to slice centroid for v1 (§1f); backbone features already carry absolute-position context, so pos_emb's job is to add slice-internal "where am I" information orthogonal to that.
- **Rev 4 (2026-06-03)**: implementation status — S3.0 through S3.3+S3.6 plumbing landed. See §13 for the as-built design.
- **Rev 5 (2026-06-08)**: S3.8 inference dump + visualization landed (§13.10). Shared per-event extractor module (`pointcept/models/LArFormer/inference.py`) ensures the slicer's output schema is byte-identical between standalone slicer inference and the slicer half of a full-cascade Stage-3 run. The Stage-3 visualizer (`tools/visualize_stage3_larformer_from_cached.py`) gained a prediction overlay panel with byte-identical color matching to the GT panel, camera sync, side-by-side layout toggle, particle-symbol legend labels, and rich hover text including per-SP query id + full per-class probability distribution.
- **Rev 6 (2026-06-11)**: status flipped to "implemented / in production training". Added §13.11 (training campaign + loss-stability tracker + inference query dedup — details in [LArFormer_Stage3_TrainingStability.md](LArFormer_Stage3_TrainingStability.md)), §13.12 (validation analysis pipeline), §13.13 (production dataprep workflow). Cross-links to the LArFormer.md §0 project hub.

---

## 13. Implementation status (as built through 2026-06-08)

This section documents what actually shipped for each phase of §9, the
files involved, and any deltas from the original plan.

### 13.1 S3.0 — Particle GT extraction ✅

- [`lartpc/data_prep/labels/slice_labels.py`](../lartpc/data_prep/labels/slice_labels.py):
  `compute_particle_labels(mpt_group, sp_trackid, sp_hasmatch, ...)`
  with constants `DEFAULT_PARTICLE_KE_THRESH_MeV = {11: 10, 22: 10,
  13: 30, 211: 30, 2112: 60, 2212: 60, 321: 60}`,
  `NEVER_VISIBLE_PARTICLE_PDGS = {111, 130, 310}` (π⁰, K⁰L, K⁰S),
  `PARTICLE_PDG_RELABEL = {2112: 2212, -2112: -2212}` (neutron → proton).
- The parent walk collapses each Geant4 track up to its nearest
  *visible* nu-origin ancestor; orphans / cosmic primaries → GHOST.
- Output `primary_pid` is post-relabel; `primary_pid_raw` carries the
  original PDG for analysis.
- **SCE-corrected origin** (rev 4): `compute_particle_labels` now also
  reads `mc_particle_tree/start_pos_sce` and returns
  `primary_start_pos_sce` alongside the raw `primary_start_pos`.
  Falls back to the raw start_pos when the H5 lacks the SCE field, so
  older files still load.
- Audit: [`lartpc/data_prep/validation/audit_particle_labels.py`](../lartpc/data_prep/validation/audit_particle_labels.py).
- Dataset plug: `gt_source="particle"` in
  [`pointcept/datasets/larformer.py`](../pointcept/datasets/larformer.py)
  (method `_gt_from_particles`). Knobs:
  `particle_class_map` (defaults to `{11: 0, 22: 1, 13: 2, 211: 3,
  2212: 4}` with everything else → class 5 = `other_track`),
  `particle_other_class_id`, `particle_ke_thresholds`,
  `particle_other_ke_threshold`, `particle_never_visible_pdgs`,
  `particle_pdg_relabel`, `particle_nu_origin`,
  `min_particle_points_post_filter`.
- Per-instance GT fields:
  ```
  pid, pid_raw, class_id,
  origin_type     # alias of class_id (legacy field name LArFormerLoss reads)
  primary_trackid, ke_mev,
  origin_coord_norm,  origin_cm,         # SCE-applied (what the
                                          #   origin head should predict)
  origin_cm_truth,                       # raw truth (analysis only)
  truth_indices, n_truth_points,
  ```

  **Bug fixed mid-implementation**: `LArFormerLoss` reads
  `g["origin_type"]` as the per-instance class slot (legacy slicer
  schema). An initial version set `origin_type=0` for every particle
  GT, which would have trained Stage 3 to predict everything as class
  0 (electron). Now `origin_type = class_id`, so the matcher sees the
  correct 7-class target.

### 13.2 S3.1 — Model-side cascade ✅

- [`pointcept/models/LArFormer/cascaded_particle.py`](../pointcept/models/LArFormer/cascaded_particle.py):
  `CascadedParticleSegmenter` wraps `CascadedSlicer` + a Stage-3
  `LArFormer`. Cascade composition (per §1b / §3):

  1. Run frozen `cascaded_slicer` in eval/no-grad → slicer per-event
     predictions + `filtered_batch` (post-deghoster).
  2. `build_nu_keep_mask(...)` (in
     [`cascade_particle_filter.py`](../pointcept/models/LArFormer/cascade_particle_filter.py))
     OR-reduces per-SP nu mask probabilities over the slicer's nu
     queries and thresholds at `τ_loose`.
  3. `filter_batch_for_particle_segmenter(...)` slices per-SP fields,
     remaps `truth_indices` for each GT instance, optionally recenters
     `coord_norm` to the per-event nu-SP centroid (§1f).
  4. Run `particle_segmenter` on the filtered batch (the only training-
     mode forward).

- Shape-aware weight loading for all three checkpoints (LoRA deghoster,
  slicer ckpt, Sonata pretrain for both backbones). Mismatched keys
  are reported and dropped, not raised.

- **Particle GT must be stripped before the slicer call**: the slicer's
  eval-with-GT path would try to compute a 3-class CE against a
  7-class target and CUDA-assert. Both
  `CascadedParticleSegmenter.forward` and the cache builder do this:
  ```python
  slicer_input = {k: v for k, v in data_dict.items()
                  if k not in ("gt_instances_per_event", "n_gt_instances")}
  ```

- Standalone `LArFormer` gained a `backbone_weight` kwarg with the
  same shape-aware loader, so the cached config (§13.4) can load the
  Sonata pretrain into the standalone particle segmenter without a
  wrapper.

### 13.3 S3.2 — Benchmark + caching decision ✅

- [`tools/benchmark_larformer_s3_cascade.py`](../tools/benchmark_larformer_s3_cascade.py):
  config-driven benchmark with two modes:
  - **full**: end-to-end `CascadedParticleSegmenter` per iter.
  - **cached**: precompute Stage 1+2 once per sample, time only
    `particle_segmenter(ps_batch)`.

  Reports forward / backward / total wall-clock + peak
  `torch.cuda.max_memory_allocated`.

- **Result on RTX 3080 16 GiB** (batch=1, 60K SPs, real checkpoints):
  | Mode | Forward | Backward | Peak alloc |
  |---|---|---|---|
  | Full cascade | 1745 ms | 25 ms | 2.93 GiB |
  | Cached Stage-3 only | 78 ms | 24 ms | 2.17 GiB |

  Caching speedup ≈ **17×**. Stage 1+2 dominates 94 % of full-mode
  forward. ⇒ Decision: build a precomputed Stage 1+2 cache (§7a is
  ON for v1).

### 13.4 S3.3 — Pure-cascade training (from cache) ✅

The **cache** is the Stage-3-only training format. Schema (per
[`tools/build_stage12_cache_event.py`](../tools/build_stage12_cache_event.py)
format_version=2):

```
event_<id>.h5
attrs:
  format_version=2, source_h5, run, subrun, event, nu_class_id,
  spacepoint_level, coord_center, coord_scale, deghost_tau (= 0.5),
  tau_loose_floor, tau_loose_nominal, tau_loose_delta,
  n_raw_spacepoints, n_after_dataset_filter, n_after_deghost,
  n_in_cache, n_passes_tau_loose, n_gt_nu_in_cache,
  n_particle_instances, n_queries, n_queries_with_per_sp_data,
  cascade_skipped, ...

entry_0/
  # Per-SP tensors (mirrors merged_h5 conventions)
  coord, coord_norm, feat, lm_score, wire,
  trackid, pid, origin_label, hasmatch, ssnet_label, slice_id,
  deghost_p_real, stage2_nu_mask_prob, stage2_nu_mask_prob_sum,
  source_mask                  # uint8 bitmask (see below)

  particle_instances/
    instance_<k>/
      truth_indices, n_truth_points, n_truth_points_orig, n_truth_points_in_cache,
      pid, pid_raw, class_id, origin_type (= class_id),
      origin_coord_norm, origin_cm, origin_cm_truth, ke_mev,
      primary_trackid

  slicer/
    query_class_argmax, query_class_max_prob, query_nu_prob,
    query_is_nu, query_ids_with_per_sp_data,
    per_query_mask_prob          # (Q_kept, N_cache), gzip-compressed
```

**Inclusion rule** for the cached SP set (per §3 + the dual-purpose
"matcher / mask denoising" requirement):

> deghost-kept AND (`stage2_nu_mask_prob > τ_loose_floor` OR
> SP belongs to a GT nu-origin particle)

So the cache stores the UNION of Stage 2's "plausibly nu" prediction
and the ground-truth nu SPs. The trainer picks the subset per
iteration.

**`source_mask` bits** (uint8):

| Bit | Meaning |
|---|---|
| 0 (=1) | SP passes Stage 2 nu-mask at the cache's *nominal* τ_loose (default 0.5). Inference-realistic predicted-pass set. |
| 1 (=2) | SP belongs to a GT nu-origin particle. Truth anchor for mask denoising. |
| 2 (=4) | SP passes at τ_loose − δ (default δ=0.2, ⇒ τ=0.3). Curriculum / lower-τ headroom. |

A SP with `source_mask = 1` is impossible (nominal-pass implies
delta-pass), so values in practice are {0, 2, 4, 5, 6, 7}.

**Build pipeline**:

- [`tools/build_stage12_cache_event.py`](../tools/build_stage12_cache_event.py):
  single-event entry. CLI takes config + inputlist + sample_idx +
  output path. The importable `build_cache_event(...)` is what shard
  drivers call.
- [`tools/build_stage12_cache_shard.py`](../tools/build_stage12_cache_shard.py):
  SLURM-array driver. Stride layout `indices = range(shard_id, N,
  n_shards)`. Output path
  `<cache_root>/<split>/<idx//1000>/<idx//100>/<basename>__event<idx>.h5`
  (3-level hash, same convention as the production driver). Idempotent
  — skips events that already have an `.h5` or a `.skipped` marker.
  Failed-empty events get a `.skipped` marker rather than no file, so
  re-runs don't keep retrying.
- [`tools/visualize_stage12_cache.py`](../tools/visualize_stage12_cache.py):
  3-panel Plotly HTML viewer (cached SPs by `source_mask`, by particle
  GT instance, by `stage2_nu_mask_prob` with false-negative rings).

**Cache-reader dataset**:

[`pointcept/datasets/larformer_stage12_cache.py`](../pointcept/datasets/larformer_stage12_cache.py):
`LArFormerStage12CacheDataset` reads the cache and emits dicts in the
same shape as `LArFormerDataset` (so `larformer_collate` and the
existing trainer plumbing work unchanged). Key knobs:

| Knob | Default | Purpose |
|---|---|---|
| `source_set_filter` | `"stage2_pass"` | Picks the SP subset. See table below. |
| `tau_loose_range` | `(0.3, 0.7)` | For `stage2_random_tau` mode. |
| `random_tau_include_gt` | `True` | OR with GT-nu SPs when augmenting τ. |
| `gt_keep_prob` | `0.5` | For `stage2_plus_gt_dropout` curriculum mode. |
| `recenter_to_centroid` | `False` | Subtract per-event coord_norm centroid (config-on for production). |
| `backbone_grid_size_cm` | `0.25` | For recomputing `grid_coord` (the cache doesn't store it). |

`source_set_filter` modes:

| Mode | Selection rule | Use case |
|---|---|---|
| `stage2_pass` | `source_mask & 1` | Inference-realistic (default) |
| `gt_nu` | `source_mask & 2` | Truth-anchored — denoising warm-up / ablation |
| `stage2_delta` | `source_mask & 4` | Lower-τ predicted-pass set |
| `union` | any bit set | Most signal — matcher sees both predicted and GT |
| `all` | every cached SP | Includes floor-only noise SPs |
| `stage2_random_tau` | sample τ ~ U(`tau_loose_range`), filter by mask_prob > τ, optionally OR with GT | τ-augmentation |
| `stage2_plus_gt_dropout` | stage2_pass ∪ random fraction of GT-only SPs | Curriculum from truth-anchor to pure-prediction |

**Trainer**:

- [`pointcept/models/LArFormer/trainer.py`](../pointcept/models/LArFormer/trainer.py):
  `LArFormerTrainer` already drove `larformer_collate`. Added
  `build_train_dn_loader()` + `_dual_run_step()` (§13.6).

- [`pointcept/models/LArFormer/particle_evaluator.py`](../pointcept/models/LArFormer/particle_evaluator.py):
  `LArFormerParticleEvaluator` subclasses
  `LArFormerSlicerEvaluator`. Defaults to the 7-class taxonomy and
  `best_metric = "mask_iou_mean"`. Drops `nu_recall` / `nu_purity` /
  `nu_mIoU` (no leading-class semantics for Stage 3). Reports
  per-class mask IoU and per-matched-pair origin Euclidean error.

- **Evaluator extension points** (added to
  `LArFormerSlicerEvaluator` so subclasses can plug in without
  duplicating the eval loop):
  ```
  _init_extra_state()                       # called at start of eval()
  _on_event_processed(ev_pred, eval_loss,   # called per val event
                      q_idx, k_idx,           after the standard
                      no_object_class_id)     metrics are computed
  ```
  The base implementations are no-ops; the particle evaluator overrides
  both to bucket origin L2 by class.

**Production config**:

[`configs/lartpc/larformer/stage3_particle/archive/larformer-particle-v1-cached.py`](../configs/lartpc/larformer/stage3_particle/archive/larformer-particle-v1-cached.py):

- `model = dict(type="LArFormer", backbone_weight=sonata_pretrain, ...)`
  — standalone particle segmenter (no cascade wrapper at train time).
- `data.train` / `data.val`: `LArFormerStage12CacheDataset` with
  `source_set_filter="stage2_pass"`, `recenter_to_centroid=True`.
- Commented `data.train_dn` block for the dual-forward path (§13.6).
- `hooks` include `LArFormerParticleEvaluator(best_metric=
  "mask_iou_mean", coord_scale=179.55)`.

### 13.5 S3.4 — Per-particle origin head ✅ (folded into S3.3 config)

The Stage-3 `LArFormer` config sets `enable_origin_head=True` and
`weight_origin=0.5`, so the origin head trains alongside the matcher
from the start. Targets come from the GT's `origin_coord_norm` field
(SCE-applied, per §13.1). The evaluator (§13.4) reports
`val/origin_l2_cm_mean` and per-class breakdowns
`val/origin_l2_cm_{e, gamma, mu, pi, p, other}` in detector cm.

### 13.6 S3.6 — Mask denoising decoupling ✅ (architecture only)

The §5 mask-denoising machinery itself is unchanged — it's the same
`MaskDenoiser` Stage 2 uses, configured via `mask_denoising=dict(
dn_groups=3, max_dn_per_event=64, anchor_jitter_std=0.05)` in the
Stage-3 config. What's new is the input-side decoupling so the
matcher and the DN path can see different SP subsets per iter.

**Optional dual-forward**: when `cfg.data.train_dn` is set (alongside
the standard `cfg.data.train`), `LArFormerTrainer._dual_run_step`
runs TWO forwards per iter:

| Forward | Input | Contribution to loss |
|---|---|---|
| A (matcher) | `cfg.data.train` (default `source_set_filter="stage2_pass"`) | All `loss_*` MINUS `loss_dn_*` |
| B (DN) | `cfg.data.train_dn` (e.g. `"union"`) | Only `loss_dn_*` |

Combined loss = (A − dn_in_A) + (dn_in_B). The DN path uses the full
GT-anchored SP set so mask perturbation has real truth to perturb;
the matcher path stays inference-realistic.

Cost: 2× per-iter wall-clock vs single-forward. The S3.2 cache
speedup (17×) absorbs it.

Cfg activation:
```python
data = dict(
    train    = dict(..., source_set_filter="stage2_pass"),
    train_dn = dict(..., source_set_filter="union"),    # uncomment to enable
    val      = dict(..., source_set_filter="stage2_pass"),
)
```

The `train_dn` stanza is commented out by default in
`larformer-particle-v1-cached.py`. Recommended order: first train
with the stanza disabled (single-forward, verify the model learns),
then enable for production quality runs.

### 13.7 As-built deltas from §9

- **Caching is in v1**, not deferred (§7a). S3.2 measured 17× speedup,
  so it earned its way in. Cache lives at
  `<cache_root>/<split>/<3-level-hash>/`, one event per `.h5`.
- **`origin_cm` is SCE-applied**, with `origin_cm_truth` kept separately
  for analysis. Drove by visualizer verification — without SCE, the GT
  origin diamonds didn't land on the reco SP clusters (~3 cm off,
  mostly in X = drift direction).
- **Standalone-`LArFormer` `backbone_weight` kwarg** added so the cached
  config doesn't need to wrap the model in a cascade just to load the
  Sonata pretrain.
- **Evaluator extension hooks** were added preemptively for §13.6's
  origin metric; future Stage-3 metrics (per-class topology recovery,
  vertex-distance error) plug in the same way.

### 13.8 Operator checklist

To run Stage 3 from a freshly-built training set:

```bash
# 1) Build cache shards (one SLURM array task per shard, GPU each).
#    Per-event cost ≈ 2 s (single forward of the cascade in eval).
#SBATCH --array=0-127
python tools/build_stage12_cache_shard.py \
    --config configs/lartpc/larformer/stage3_particle/larformer-particle-v1.py \
    --inputlist /path/to/h5list_train.txt \
    --cache-root /path/to/stage12_cache_v2 \
    --split train --shard-id $SLURM_ARRAY_TASK_ID --n-shards 128
#    Repeat for split=val with the val list.

# 2) Edit configs/lartpc/larformer/stage3_particle/archive/larformer-particle-v1-cached.py:
#    set CACHE_ROOT to the cluster path above.
#    Optionally uncomment the `train_dn` stanza (§13.6).

# 3) Train.
python tools/train.py --config configs/lartpc/larformer/stage3_particle/archive/larformer-particle-v1-cached.py
```

**Eyeball-test a cached event** (recommended before kicking off the
full training run):

```bash
python tools/visualize_stage12_cache.py \
    --cache /path/to/stage12_cache_v2/train/.../event_000000.h5 \
    --output /tmp/cache.html --browser
```

**Sanity metrics** to watch in the per-epoch log:

| Scalar | "Is it learning?" |
|---|---|
| `val/loss` | drops monotonically (~140 → <50 in ~5 epochs of devdata) |
| `val/mask_iou_mean` | rises from ~0 to >0.4 |
| `val/origin_l2_cm_mean` | drops from ~80 cm (random init) to <30 cm |
| `val/cls_accuracy` | rises from ~1/(num_classes) to >0.5 |

Per-class breakdowns (`val/mask_iou_{e, gamma, mu, pi, p, other}`,
`val/origin_l2_cm_{...}`) tell you which classes are easy / hard.

### 13.9 What's still open (vs §9 phasing)

- **S3.5 (noise calibration)** — not started. The mask-denoising in
  v1 uses fixed `anchor_jitter_std=0.05` (5 % of slice scale) per
  Stage 2's default. The per-class calibration from running Stage 2
  over the training set is still on the to-do list.
- **S3.7 (noised-truth alternative)** — depends on S3.5.
- **S3.8 (full cascade eval tool)** — **inference dump + visualization
  landed in §13.10.** The analyzer-side per-event metric harness
  (per-particle topology recovery, vertex distance, confusion
  breakdowns) is still TODO and best built on top of the
  `stage3pred_*.h5` schema that §13.10 defines.
- **S3.9 (ablations)** — most are parameterized cleanly:
  - τ_loose sweep: change `mask_prob_threshold` in
    `CascadedParticleSegmenter` config + rebuild cache OR sample
    via `source_set_filter="stage2_random_tau"`.
  - Class taxonomy: change `particle_class_map` in the dataset
    and the `num_classes` in the model config.
  - Aux absolute-position feature (§1g): add a per-SP "absolute coord
    before recentering" channel in the cache (one extra `feat`
    column) + extend the dataset reader. Not yet wired.

  - Auxiliary absolute-position feature (drift-x, boundary distances) deferred as S3.9 ablation (§1g) — doubles as a probe of what the frozen backbone encodes.

### 13.10 S3.8 — Inference dump + visualization ✅

The inference and visualization halves of S3.8 (§8c) landed together:
two CLI tools share a single helper module so the slicer's per-event
output schema is byte-identical between standalone slicer inference and
the slicer half of a full-cascade Stage-3 inference run.

#### Shared helpers — `pointcept/models/LArFormer/inference.py`

Single source of truth for per-event prediction extraction. All paths
that produce `slicerpred_*.h5` / `stage3pred_*.h5` files call into this
module. Exports:

- `slicer_predict_event(model, sample, batched, no_object_class_id)` —
  runs the cascade forward and returns the canonical slicerpred dict.
  Extracted from the pre-refactor `tools/run_slicer_inference.py`
  verbatim. Schema-regression-tested (46 canonical keys).
- `slicer_predict_event_from_out(out, sample, no_object_class_id)` —
  same body but accepts an already-computed cascade output dict, so the
  full-cascade Stage-3 CLI can reuse the slicer forward instead of
  re-running it.
- `stage3_predict_event(model, sample, batched, no_object_class_id, *,
  class_prob_threshold=0.0, coord_scale=179.55)` and
  `stage3_predict_event_from_out(...)` — Stage-3 analog. Uses the same
  panoptic-argmax SP-assignment rule as the slicer (so
  `pointcept/models/LArFormer/inference.py:per_sp_predicted_slice` is
  shared between both stages), plus particle-level extras: origin
  prediction, origin-error stats, per-SP `particle_class_id` GT
  lookup, per-SP `source_mask` and `stage2_nu_mask_prob` carried from
  the cache.
- `write_event_h5(path, event_data)` / `load_event_h5(path)` — atomic
  writes (`tmp + os.replace`) plus a reader keyed by `group/.../leaf`
  paths. Single-source loader: both visualizers go through the same
  loader so they can't disagree about the schema.

The Stage-3 confidence-floor (`class_prob_threshold`) is the
operationalization of question Q3 from the design discussion: queries
whose max softmax cls probability falls below the floor are demoted to
`no_object` for the per-SP panoptic assignment, while
`stage3_queries/class_argmax` still records their raw argmax for
diagnostics.

#### Slicer CLI — `tools/run_slicer_inference.py`

Refactored to be a thin CLI (~150 lines, down from ~650). Imports
`slicer_predict_event` from the helpers module; no behavioral change.
Output naming and schema are unchanged so existing slicer-side
analysis scripts continue to work.

#### Stage-3 CLI — `tools/run_larformer_stage3_inference.py`

Two input modes:

- **`--input-mode cached`** (default — the production training path).
  Reads cached events via `LArFormerStage12CacheDataset`, runs the
  Stage-3 `LArFormer` on each, writes `stage3pred_<cache_basename>.h5`
  with `stage3*/...` keys. Output filename derives from the
  cache file's `__event<idx>` basename for guaranteed uniqueness.
  Auto-detects when the config's `model` is a
  `CascadedParticleSegmenter` wrapper and uses its inner
  `particle_segmenter`.

- **`--input-mode full-cascade`**. Reads raw merged_h5 events via
  `LArFormerDataset`, builds the whole `CascadedParticleSegmenter`,
  runs the slicer (with GT if available) for slicerpred-shaped
  output, then applies the Stage-2 → Stage-3 boundary helpers
  (`build_nu_keep_mask`, `filter_batch_for_particle_segmenter`) and
  runs the particle segmenter, appending `stage3*/...` keys.
  Output is one combined `stage3pred_*.h5` per event whose top-level
  `pre/`, `post/`, `queries/`, `gt/`, `meta/`, `levels/` keys form a
  valid slicerpred file — so the existing slicer viz works on the
  Stage-2 half directly via its `--slicerpred-dir` flag.

- **`--no-gt`** disables `gt_source="particle"` on the dataset for
  the real-data inference use case where no GT is available. The
  schema's `gt/`, `stage3_gt/`, matching, and IoU fields populate
  with empty arrays / NaNs; `meta/has_gt = 0`. Predictions still
  populate.

Operator commands:

```bash
# Cached-mode (production training-loop iteration)
python tools/run_larformer_stage3_inference.py \
    --config configs/lartpc/larformer/stage3_particle/larformer-particle-v1-cached-ptv3crosslevel.py \
    --weights exp/.../model/model_last.pth \
    --cache-dir exp/cache_stage12_ptv3crosslevelslicer_iter_75750/val \
    --output-dir exp/.../inference \
    --class-prob-threshold 0.3 \
    --split val

# Full-cascade on real data with no GT (Q8 use case)
python tools/run_larformer_stage3_inference.py \
    --input-mode full-cascade \
    --config configs/lartpc/larformer/stage3_particle/larformer-particle-v1.py \
    --weights exp/.../model/model_last.pth \
    --input-list inputlists/realdata_run3b.txt \
    --output-dir exp/.../inference_realdata \
    --class-prob-threshold 0.3 \
    --no-gt
```

#### Output schema — `stage3pred_<basename>.h5`

Per-event HDF5. Top-level groups (slicer half + Stage-3 half are
disjoint namespaces so analysis scripts can pick whichever they need):

| group | populated in |
|---|---|
| `pre/...` `post/...` `queries/...` `gt/...` `meta/...` `levels/...` | full-cascade mode only — schema-identical to `slicerpred_*.h5` |
| `stage3/...` per-SP fields | always |
| `stage3_queries/...` per-query | always |
| `stage3_gt/...` per-particle | always (empty arrays when `--no-gt`) |
| `stage3_levels/<name>/...` per-voxel-level | always (one block per Stage-3 voxel level the model uses) |
| `stage3_meta/...` flat attrs | always |

Stage-3 per-SP fields (`stage3/`): `coord`, `coord_norm`,
`particle_class_id_gt`, `source_mask`, `stage2_nu_mask_prob`,
`pred_query` (the panoptic-argmax winner — directly addresses the
query→SP mapping the visualizer needs), `pred_class`,
`pred_particle_idx` (K-index into GT), `pred_particle_trackid`
(physical Geant4 trackid — used by the visualizer for color
matching), `pred_mask_prob`.

Stage-3 per-query fields (`stage3_queries/`): `class_logits`,
`class_probs`, `class_argmax`, `class_max_prob`, `is_active`,
`origin_coord_norm`, `matched_gt_idx`.

Stage-3 per-GT fields (`stage3_gt/`): `primary_trackid`, `class_id`,
`pid`, `ke_mev`, `n_truth_points`, `origin_cm`, `origin_coord_norm`,
`matched_query`, `pair_iou`, `pair_cls_correct`, `pair_origin_l2_cm`
(in cm; -1 where unmatched).

Stage-3 meta (`stage3_meta/`): identity + summary counters +
`class_prob_threshold` + `coord_scale` (so the visualizer can scale
predicted origins to cm without round-tripping the config) +
`has_gt`.

#### Visualization — `tools/visualize_stage3_larformer_from_cached.py`

Extended with two flags:

- `--stage3pred-dir <DIR>`: directory of `stage3pred_*.h5` produced
  by the inference CLI. When set, a second 3D Plotly scene appears
  alongside the GT scene. File matching: the viz looks for
  `stage3pred_<cache_basename>.h5` for each event index — the cache
  filename's `__event<idx>` suffix guarantees uniqueness.
- `--min-mask-prob <float>`: confidence-floor for the panoptic
  assignment in the pred panel (SPs whose assigned-query sigm < floor
  are demoted to no_object). Adjustable live via a number input in
  the top bar.

The prediction panel's figure builders live in
`pointcept/models/LArFormer/viz_inference.py` (color/threshold/symbol
helpers + `figure_for_stage3_prediction`) — the tool itself is a
thin Dash app.

**Color matching between GT and prediction panels.** The viz's
`pointcept/models/LArFormer/viz_inference.py:track_id_color` is
byte-identical to `tools/visualize_larformer_gt.py:_track_id_color`
when alpha = 1 (`abs(int(tid))` hash, saturation 0.80, value 0.95,
`{:g}` alpha format suppresses the `.0` so `rgba(...,1)` matches
exactly). In the pred panel's `pred_particle_idx` color mode, each
matched-query trace is colored by `track_id_color(pred_particle_trackid)`
of its assigned GT — so the predicted-particle SP cluster wears the
SAME COLOR as that physical track's instance in the GT panel.
Legend entries read `particle idx N (trackid TID)` to make the
cross-reference explicit.

**Talk-shot UX.** Three checkboxes (all on by default):

- `show pred↔GT origin lines` — connects each matched query's
  predicted-origin diamond to its GT origin (length = L2 error).
- `sync rotation/zoom` — server-side `Patch` callbacks in both
  directions mirror `scene.camera` between the two panels. Reads
  `relayoutData["scene.camera"]`; falls back to reassembling the dict
  from individual axis updates. Programmatic camera updates don't
  re-fire `relayoutData`, so no feedback loop.
- `side-by-side panels` — toggles the scene container between
  `display: flex; flexDirection: row` (78 vh each) and stacked
  (42 vh each). Panel widths auto-balance via `flex: 1 1 0`.

**Hover text** (per SP). 14-column mixed-dtype customdata holds:
mask_prob, pred class symbol, pred_query (matches the origin
diamonds' `q=N`), pred_particle_idx, pred_particle_trackid, GT class
symbol, then the full per-class softmax probability of the
assigned query (8 floats). Renders as:

```
x=125.4 y=10.2 z=400.1
mask_prob=0.534  pred=μ±  GT=p
query=12  pred_particle_idx=4  trackid=1595065
class probs (assigned query):
  e±=0.09  γ=0.06  μ±=0.12  π±=0.17
  p=0.05  other=0.19  ∅=0.13
```

Particle symbols (`PARTICLE_CLASS_SYMBOLS`): `("e±", "γ", "μ±", "π±",
"p", "other", "—", "∅")` — used in legend labels (`pred_class`,
`particle_class_id_gt` modes) AND in hover text. The Unicode `∅`
specifically signals the `no_object` slot.

**Origin diamond hover** (per active query). Each diamond hover shows
the query id (bold), predicted class symbol + max prob, origin in
cm, and the full per-class probability distribution. Matches the
per-SP hover schema so the visual cross-reference (SP → diamond)
also reads consistently in tooltips.

**Color-by modes** (8): `pred_particle_idx` (GT-color-matched —
default), `pred_class` (by particle taxonomy), `pred_query` (by query
id), `pred_mask_prob` (continuous Plasma colorbar), `particle_class_id_gt`
(per-SP GT class), `pred_class_correct` (correct/wrong/no-GT),
`stage2_nu_mask_prob` (cache telemetry — Viridis), `source_mask`
(cache provenance bitmask).

#### Slicer-viz reuse path

The full-cascade `stage3pred_*.h5` files carry the slicer half at the
top level (`pre/`, `post/`, `queries/`, `gt/`, `meta/`, `levels/`)
with the slicer's canonical schema. So
`tools/visualize_larformer_gt.py --slicerpred-dir <dir>` works on a
full-cascade output directory unchanged — the slicer viz's
prediction-panel callbacks find files matching its
`slicerpred_<basename>.h5` pattern, but because the schema is also
exposed under the `stage3pred_<basename>.h5` files, a small alias
(symlink the stage3pred to slicerpred names, or extend the slicer
viz's filename matcher) is all that's needed if you want
panoptic-slicer diagnostics on the same events.

### 13.11 Production training campaign + loss-stability work (2026-06)

Documented separately in
[LArFormer_Stage3_TrainingStability.md](LArFormer_Stage3_TrainingStability.md)
— kept there because it's an active tracker, not settled design. It
covers:

- the training-run lineage (`lr1e4_bugfixed` → `resume2` →
  `resume2B_resetoptim` → `resume3_cosinedecay`, ~22.6k iters/epoch on
  the iter-75750 Stage-1+2 cache) and the diagnosis of the
  rising/oscillating mask losses (adaptive hard-negative sampling +
  Hungarian churn — NOT model degradation; val IoU rises throughout);
- the mid-run LR-schedule swap machinery (`DelayedCosineLR` +
  `CheckpointLoader(extend_scheduler=True)` with an
  `iter_in_epoch`-aware fast-forward);
- log-only per-batch diagnostics in `LArFormerLoss`
  (`loss_kwargs.log_diagnostics=True` → `train_batch/loss_diag_*` and
  `val/loss_diag_*`) plus pre-clip `train_batch/grad_norm`;
- the dominant prediction failure mode — a μ-classifying + a
  π-classifying query co-covering one true track and fragmenting it
  under the panoptic argmax — and the **inference-side query dedup**
  that fixes it: `dedup_queries` in
  [`pointcept/models/LArFormer/inference.py`](../pointcept/models/LArFormer/inference.py),
  exposed as `--dedup-iou-threshold` (default 0.6) on
  `tools/run_larformer_stage3_inference.py`, with merge tracking under
  new `stage3_queries/dedup_*` H5 keys (new keys only — the §13.10
  schema is otherwise unchanged; `stage3/pred_query` becomes the
  post-dedup assignment and `stage3/pred_query_nodedup` preserves the
  raw one).

### 13.12 Validation analysis pipeline ✅

[`lartpc/larformer_analysis/particle_eval/`](../lartpc/larformer_analysis/particle_eval/README.md)
— split-level efficiency/purity validation on SLURM: per-event
distillation of `stage3pred_*.h5` into per-pair records
(`analyze_event.py`), aggregation into the
`LArFormerParticleEvaluator` scalars plus size-stratified stress
metrics (`aggregate_metrics.py` → JSON + parquet), and an
auto-sizing sbatch driver supporting both cached and full-cascade
input modes (`slurm/submit_valtest.sh` +
`slurm/run_valtest_per_task.py`). See the README there for the
output schema and usage.

### 13.13 Production data-prep / inference workflow ✅

[`lartpc/data_prep/uboone_official/LARFORMER_DATAPREP.md`](../lartpc/data_prep/uboone_official/LARFORMER_DATAPREP.md)
— the config-driven two-stage pipeline that takes raw
`merged_dlreco.root` (sim or data) to full-cascade `stage3pred_*.h5`
without LArMatch/SSNet/lantern: Stage A conversion via
`SimChTripletLabelMaker`, Stage B `CascadedParticleSegmenter`
inference, plus `tools/visualize_full_cascade.py` for camera-synced
prediction-vs-truth event displays.