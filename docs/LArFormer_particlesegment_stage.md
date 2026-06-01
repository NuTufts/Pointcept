# LArFormer Stage 3 — Particle segmenter

**Status:** design / planning. Not yet implemented. Revision 2 (2026-05-30).

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

**Implementation:** new function in `lartpc_data_prep/slice_labels.py`:

```python
def compute_particle_labels(mpt_group, sp_trackid, sp_hasmatch,
                             ke_thresholds=DEFAULT_KE_THRESH,
                             nu_origin=1):
    """Per-particle slice labels — particle-agnostic with KE thresholds."""
```

Returns the same dict shape `compute_slice_labels` returns (slice_id per SP, primary_trackid array, etc.). The merging-low-KE-into-parent logic walks the parent chain upward until it finds an above-threshold ancestor.

A new `gt_source="particle"` plug in `LArFormerDataset` calls this routine.

### 2b. SP → GT assignment

Per-SP truth has `trackid`. Each SP gets assigned to its above-threshold ancestor's slice id via the merging logic above. Existing `lartpc_data_prep/slice_labels.py` machinery is the foundation; the new routine just plugs in a different walk termination.

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

Reuses the analyzer-side pipeline architecture from [`lartpc_data_prep/larformer_analysis/`](../lartpc_data_prep/larformer_analysis/) — Stage 3 outputs a per-event H5, an aggregator builds an event_summary.h5, plots produce per-category headlines.

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
- Existing `compute_slice_labels` (which `compute_particle_labels` will extend): [`lartpc_data_prep/slice_labels.py`](../lartpc_data_prep/slice_labels.py)
- Analyzer-side category definitions: [`lartpc_data_prep/larformer_analysis/lib/categorize.py`](../lartpc_data_prep/larformer_analysis/lib/categorize.py)
- Per-event analysis pattern (template for S3.8): [`lartpc_data_prep/larformer_analysis/`](../lartpc_data_prep/larformer_analysis/)
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
  - Auxiliary absolute-position feature (drift-x, boundary distances) deferred as S3.9 ablation (§1g) — doubles as a probe of what the frozen backbone encodes.