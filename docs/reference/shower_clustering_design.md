# Shower Clustering Model — Design and Implementation Plan

> **Status: REFERENCE** — Mask2Former shower-clustering design (predecessor generalized by LArFormer).

**Status:** Phase 1 in progress. First diagnostics on 500 NC pi0 training events done 2026-05-04.
**Owner:** taritree.wongjirad@tufts.edu
**Replaces:** `ShowerOriginPredictorV3` (per-point regression and slot-attention origin prediction).

This document is the living design reference. Update as decisions change or phases complete.

---

## 1. Motivation

`ShowerOriginPredictorV3` (defined in [pointcept/models/shower_origin/shower_origin_model.py:769](../../pointcept/models/shower_origin/shower_origin_model.py)) classifies fragments well — inside / outside / cosmic accuracies are all >90%. But:

- The **regression head** (predicted origin coord per slot) is much noisier than the origin scores. It does not localize the start point reliably.
- **Start-point labeling is too inconsistent across fragments of the same shower** to drive the cone-based merger in [ub_showerorigin_reco/ubshowerorginreco/shower_fragment_merger.py](../../../ub_showerorigin_reco/ubshowerorginreco/shower_fragment_merger.py).
- This kills downstream reconstruction even though the per-fragment classification is strong.

V3's slot attention forces a single head to do classification *and* origin regression *and* implicit instance assignment. Mask2Former's bipartite-matching set-prediction framework decouples these and lets the strong classification signal carry without being dragged down by the regression target.

**Goal:** a new model that simultaneously (a) clusters shower fragments into showers and (b) classifies each shower (5 classes: `inside`, `outside`, `on_track`, `ghost`, `true_track`), replacing the cone merger entirely.

---

## 2. Architecture summary

**Hierarchical Mask2Former with three token scales.**

```
                                [ N learnable queries ]
                                          |
         +--------------------------------+--------------------------------+
         |                                |                                |
   cross-attn (rotates per layer):  voxel tokens   fragment tokens   spacepoint tokens
         |                                |                                |
   self-attn between queries (every layer)
         |
   final per-query heads:
       - class logits (6: 5 origin types + "no object")
       - mask logits (fragment-level + spacepoint refinement on assigned fragments)
       - aux origin coord regression
```

### Token scales and what each one is for

| Scale | Token count | What it carries | Why we need it |
|---|---|---|---|
| **Spacepoint** | ~100k | per-spacepoint Sonata feature | fine mask refinement on assigned fragments |
| **Voxel (~5 cm)** | ~5–10k | mean-pooled spacepoint features per voxel | **the only way the model sees non-shower content** (tracks, ghosts, vertex region). DBSCAN fragments cover only shower-tagged spacepoints; voxel tokens carry cosmic-track context, candidate neutrino vertex regions, and the broader event topology. |
| **Fragment** | ~50–200 | learnable pool over the fragment's spacepoints | the dominant clustering unit; carries the per-fragment geometric prior |

**This is the load-bearing architectural choice in this design.** Without voxel tokens, the model has no view of track or ghost spacepoints near a shower — and "is there a cosmic muon nearby" / "is there a neutrino vertex in this region" is exactly the kind of context that distinguishes inside/outside/on_track.

### Backbone

- **Frozen Sonata** (`PT-v3m2 + v1m1 SONATA pretrain head`), checkpoint `lartpc_v6_h200_noghosts_pretrain_logspace_resume/model/epoch_42.pth`.
- Run **once per event**, **tiled** at 20k-point overlapping crops, output features stitched by averaging overlapping spacepoints. Output: per-spacepoint 1088-dim features.
- Cached to disk so decoder iteration does not re-run the backbone.

### Decoder

- N=64 learnable queries, 4–6 decoder layers.
- Each layer: masked cross-attention (Mask2Former-style — restrict attention to entries with previous-layer mask logit > 0) + self-attention between queries + FFN.
- Cross-attention rotates over the three scales (e.g. layer 0: voxel; layer 1: fragment; layer 2: spacepoint; repeat).
- Query positional encoding: learnable + MLP on the **predicted origin coord from the auxiliary regression head**, refined each layer.

### Heads

1. **Per-query class** — 6-way (`inside`, `outside`, `on_track`, `ghost`, `true_track`, `no_object`)
2. **Fragment mask** — `sigmoid(query · fragment_emb)` per (query, fragment)
3. **Spacepoint refinement** — `sigmoid(query · spacepoint_emb)` over spacepoints in fragments assigned to this query (gated; bounded cost)
4. **Auxiliary origin regression** — per-query 3D coord; supervised on matched queries; also feeds positional encoding

### Loss

- **Hungarian matching** between queries and ground-truth instances. Cost = λ_cls · CE + λ_mask · BCE + λ_dice · Dice.
- Per-query training loss = same form as cost, plus λ_origin · L1 on origin coord (matched only).
- Class weights for rare instances.

---

## 3. Architectural decisions (what we settled on, and why)

| Decision | Choice | Why |
|---|---|---|
| Token granularity | hierarchical fragment + voxel + spacepoint | fragment-only loses track/ghost/vertex context (DBSCAN only fragments shower-tagged points); spacepoint-only is too expensive |
| Sonata backbone | **frozen** | matches V3; faster iteration; pretraining was on similar data |
| DBSCAN parameters | **unchanged for now** | visual inspection looks decent; under-clustering is the conservative choice. Phase 1 IoU diagnostic will revisit if needed. |
| Spacepoint→fragment pool | mini set-transformer (2–3 self-attn layers + learnable pool query) | richer than mean/max; handles variable fragment sizes naturally |
| Voxel size | 5 cm (initial) | balances token count vs spatial resolution; revisit in Phase 1 |
| Number of queries | N=64 | far more than typical true shower count (~1–10), gives matching headroom |
| Ghost / track-only GT | explicit GT instances with their own class | keeps the strong classification signal; better than mapping to "no object" |
| Class label set | 5 classes from V3 + 1 `no_object` slot | unchanged from V3 |
| GT mask source | `trackid` from `shower_fragments` group → all spacepoints with that trackid | already present in merged H5 from `merge_reco_truth_showerorigin.py` |
| Inference granularity | full event in one pass | feature cache makes it cheap; no chunked-and-stitched query merging needed |
| Replaces cone merger | yes — model output IS the merged shower | Steps 5–7 of the [pipeline](../../../ub_showerorigin_reco/CLAUDE.md) collapse to "run model, write TTree" |

### New code location

- New directory: `pointcept/models/shower_clustering/` (sibling to `shower_origin/`)
- New configs in: `pointcept/configs/lartpc/shower-clustering-*.py`
- New dataset class: `pointcept/datasets/shower_clustering.py`
- New data-prep / characterization scripts: `pointcept/lartpc_data_prep/`
- Pipeline integration in `ub_showerorigin_reco`: new `tools/run_shower_clustering_inference.py`, eventually replacing the Step 5–7 driver.

---

## 4. Open questions and things to revisit on data

### 4a. Phase 1 findings (2026-05-04, n=500 NC pi0 events)

Run via `tools/viz_archive/characterize_fragments.py`, output in `pointcept/exp/shower_clustering/phase1_smoke_500events/`.

Headline numbers from the merged H5 (training data: `bnb_nu_pi0filter_corsika`):

| Metric | Value |
|---|---|
| Spacepoints / event (median, p95) | 115k, 194k |
| Shower-pid spacepoints / event (median) | 8.1k (~7%) |
| Track-pid spacepoints / event (median) | 52k (~45%) |
| Unmatched spacepoints / event (median) | 53k (~45%) |
| Fragments / event (median, p95) | 52, 102 |
| **Shower-pid spacepoints covered by any fragment** | **~55%** (orphan rate ~45%) |
| Orphan→nearest-fragment distance (median, p95) | 4.3 cm, 108 cm |
| Fragment plurality purity (median) | 0.63 |
| True shower instances / event (significant ≥20 pts) | 37 |
| Avg shower instances / event (all sizes) | 41 |

Stratified by truth-instance point count:

| Bucket | count | frac with ≥1 own-tid frag | IoU median | IoU p95 |
|---|---|---|---|---|
| [1–19] (sub-DBSCAN) | 1988 | 7.0% | 0.000 | 0.500 |
| [20–99] (small) | 7706 | 46.4% | 0.000 | 0.714 |
| [100–499] (medium) | 6577 | 83.8% | 0.358 | 0.652 |
| [500+] (large) | 4381 | 37.6% | 0.000 | 0.587 |

### 4b. What these findings mean

**The per-trackid IoU metric is misleading.** Single-event spot-check confirmed that the largest shower in one event (tid=659979, 1602 truth points) has 1458 of its 1602 truth points (91%) in fragments — but the script credits its IoU as low because **plurality voting splits one observable shower across multiple MC trackids**. A single visible shower spans many Geant4 trackids (parent + bremsstrahlung + Compton + secondaries), DBSCAN groups them spatially, and the merge step assigns each fragment to one plurality trackid, leaving the other trackids with IoU=0 even though their points are well-covered.

**The real headline:** DBSCAN+SSNet recovers ~**55%** of shower-pid spacepoints into fragments. The other ~45% are orphans, dominated by **2D SSNet false negatives** (true shower pixels SSNet labeled as track, never entering the DBSCAN clustering). Single-event evidence: trackids exist with hundreds of shower-pid truth spacepoints and zero DBSCAN coverage.

### 4c. Implications for the architecture

1. **Voxel scale is mandatory and load-bearing.** It must include features from *all* spacepoints (track-pid, shower-pid, unmatched) so the model can recover SSNet false negatives — fragments cover only what SSNet+DBSCAN consented to cluster.
2. **5 cm voxel size is well-supported.** Median orphan→fragment distance is 4.3 cm — at 5 cm voxels, most orphans sit in voxels adjacent to fragment voxels, giving the decoder direct spatial access.
3. **Spacepoint refinement is critical, not optional.** With 45% of shower truth in orphan spacepoints, fragment-only masks systematically lose them.
4. **N=64 queries is right.** 37 significant instances/event median, p95 unknown but likely <60. 64 queries with "no_object" slack is comfortable.
5. **GT instance definition needs work** — see open questions below.

### 4d. GT instance definition — RESOLVED (2026-05-04)

The new merge-step output (test files in `pointcept/lartpc/data_prep/training_data/tmp_workdir/lantern_bnb_nu_pi0filter_corsika_jobid0000_line00001/`) preserves `mc_particle_tree` (`trackid`, `parent_trackid`, `pid`, `process_code`, `start_pos`, `energy_mev`, `origin`). This unblocks **option (a)** from the previous design discussion: walk the Geant4 tree to define GT instances.

**GT instance algorithm:**
1. Take the set of unique non-(-1) trackids appearing in `shower_fragments/trackid` — these are shower trunks per the merge step's plurality voting.
2. For each trunk trackid T, walk `mc_particle_tree.parent_trackid` to gather all descendant trackids.
3. GT instance for T = (per-spacepoint truth set: `triplet_data/trackid ∈ descendants(T)`, predicted set: union of fragments with `shower_fragments/trackid == T`).

**Validation on 3 test events** (run via `characterize_fragments.py`, output in `pointcept/exp/shower_clustering/phase1_newmerge_3events/`):

| Metric | Trunk-descendant GT | Old per-trackid GT |
|---|---|---|
| Instances / event | 23 | 41 |
| Points / instance (median) | 196 | 109 |
| **Median IoU** | **0.374** | 0.141 |
| **Median coverage** | **0.603** | 0.162 |
| Frac significant instances with ≥1 fragment | **100%** | varies |
| Stratified IoU [20–99 / 100–499 / 500+] | 0.42 / 0.37 / 0.32 | 0.00 / 0.36 / 0.00 |

The trunk-descendant numbers are honest: they reflect actual DBSCAN+SSNet recall, not artifacts of plurality voting.

**SSNet false-negative confirmation:** 22,939 orphan shower-pid spacepoints in 3 events; SSNet labels: 74% label=7, 21% label=3, 3% label=8 — virtually **none** labeled as shower. Confirms the voxel scale is mandatory for recovering these.

### 4e. Ghost handling — RESOLVED via larmatch-score augmentation (2026-05-04)

The new merge step preserves all 254k spacepoints/event including ~163k unmatched ghosts. The upstream pipeline does an initial deghosting pass that drops obvious ghosts at larmatch score < 0.15 (so the H5 floor is 0.15). The per-spacepoint **larmatch score** is preserved in the H5, allowing further threshold-driven ghost removal at training time.

**Decision: vary the larmatch-score threshold as a per-event training-time augmentation.** This:

- Trains the model on a range of ghost densities (matches production where the deghoster threshold may be tuned).
- Avoids committing to one ghost-handling regime.
- Is essentially free in compute (just a mask).

**Implementation rules:**

1. **Backbone feature cache** is built at the H5's floor threshold (0.15). All higher-threshold filtering happens at training time, downstream of the cache.
2. **At training step start**, sample threshold τ ~ Uniform(0.15, 0.40) per event. Mask out spacepoints with `lm_score < τ`.
3. **Validation**: fix τ = 0.15 (production deghoster setting) for comparability across runs.
4. **Fragments**: subset their `pointindices_flat` by the surviving spacepoint mask. Drop fragments whose surviving-point count falls below `min_fragment_points = 20` (matches original DBSCAN setting in `convert_larlite_to_showerorigin_h5.py`).
5. **No re-DBSCAN**. Fragments that *would have* internally split at the higher threshold remain single fragment-tokens. The Mask2Former design tolerates this: the spacepoint-refinement head + voxel scale can disambiguate at sub-fragment granularity, and per-query masks can independently fire on the same fragment.
6. **Voxel tokens**: recompute each step from surviving spacepoint features (mean pool). Drop voxels with zero surviving members.
7. **GT instance masks**: descendant trackid set is unchanged (defined from `mc_particle_tree`), but the per-spacepoint truth membership shrinks alongside surviving spacepoints. IoU computed on surviving set only.

**Limitations and what to watch:**

- At τ ≥ 0.4 many fragments drop below the 20-pt floor; events become voxel-context-dominated. Cap the augmentation at 0.40 for the first cut.
- The `ghost` class (class id 3) becomes rare at high τ. Class-weighted loss will keep gradients flowing; track per-class precision/recall on val.
- The "fragment should have split" pathology degrades smoothly with τ; it is not a sharp failure mode.

This decision supersedes the earlier "filter ghosts" / "keep ghosts" / "hybrid" framing.

### 4f. Other open questions

- **Voxel-scale mask granularity.** Given the 45% orphan rate, voxel-level masks (not just fragment-level) may need to be a primary mask head — see §3 architecture note.
- **DBSCAN parameter revision.** Diagnostic shows DBSCAN itself is fine — bottleneck is upstream SSNet. Out of scope.
- **End-to-end fine-tuning of backbone.** Defer.
- **Re-merge of training data.** The 52k-file production training set was generated with the old merge step (no `mc_particle_tree`). Need to re-run the merge for all training data before Phase 4. Estimated cost: SLURM array, a few hours.

---

## 5. Phased implementation plan

Each phase is a checkable milestone. Update checkboxes and add notes as you go.

### Phase 1 — Data characterization and GT construction *(no new model code yet)*

Goal: Quantify the fragment quality, build per-event GT instances from `trackid`, and confirm the design's assumptions before writing model code.

- [x] **`pointcept/tools/viz_archive/characterize_fragments.py`** — implemented, supports per-event diagnostics, stratified IoU by instance size, orphan→fragment distance, fragment purity, summary H5 + plots.
- [x] Run on 500 events from `bnb_nu_pi0filter_corsika` training H5s. Outputs in `pointcept/exp/shower_clustering/phase1_smoke_500events/`. See §4a above for results.
- [x] **Decision: keep voxel scale (mandatory, see §4c), keep spacepoint refinement (45% orphan rate makes it critical), 5 cm voxel size confirmed, DBSCAN params unchanged (bottleneck is upstream SSNet, not DBSCAN).**

### Phase 1.5 — GT instance definition *(RESOLVED 2026-05-04)*

- [x] **Inspect `mc_particle_tree` schema** in merged H5. Confirmed: `trackid`, `parent_trackid`, `pid`, `process_code`, `start_pos`, `energy_mev`, `origin`, `daughter_trackids`, `daughter_start_indices`, `num_daughters`, `nu_vertices`. New merge step preserves it.
- [x] **GT instance source decided: option (a)** — walk `mc_particle_tree.parent_trackid` from each unique non-(-1) `shower_fragments/trackid` to gather descendants. See §4d.
- [x] **`characterize_fragments.py` updated** to support both new (trunk-descendant) and old (per-trackid legacy) schemas; reports which mode was used per event.
- [x] **Validation on 3 test events** done, see §4d table.
- [ ] **Re-merge all training data** with the fixed merge step before Phase 4 begins.

### Phase 2 — Dataset class *(COMPLETE 2026-05-04)*

**Storage reality check**: Caching per-spacepoint backbone features at full precision is infeasible (254k SPs × 1088 dim × 2 bytes × 52k events = 28 TB at fp16, 56 TB at fp32). Voxel-level-only cache is ~1 TB and feasible but isn't needed until profiling demands it. Caching deferred entirely; the frozen backbone runs on-the-fly inside the model's forward pass.

- [x] **`pointcept/datasets/shower_clustering.py`** — `ShowerClusteringDataset` + `shower_clustering_collate`. Returns per-event dict with `coord`, `coord_norm`, `feat`, `lm_score`, `wire`, `trackid`, `pid`, `origin_label`, `hasmatch`, `ssnet_label`, `voxel_id`, `voxel_keys`, `fragment_indices`, `fragment_trackid/pid/type`, `gt_instances`, scalar counts, identity (`run`/`subrun`/`event`).
- [x] **lm_score augmentation hook**: train samples τ ~ U(0.15, 0.40) per event; val/test fixed at 0.15. Drops spacepoints below τ, drops fragments below 20 surviving pts, recomputes voxel ids on surviving spacepoints.
- [x] **GT-instance precomputation**: `mc_particle_tree.parent_trackid` walked at load time. Cheap (microseconds per event).
- [x] **Smoke test on 3 test events** + **collate function on 2-event batch**. Numbers in §4d.
- [x] **Registered with Pointcept's DATASETS builder** in `pointcept/datasets/__init__.py`.
- [x] **Draft config** at `pointcept/configs/lartpc/shower_origin/archive/shower-cluster-sonata-v1.py` (dataset params + Sonata backbone stub + Mask2Former model placeholder).
- [x] **Visualizer** at `pointcept/tools/viz_archive/visualize_shower_clustering.py` — 4-panel Dash app (GT instances / fragments / SSNet / lm_score) with detector outline, MicroBooNE-style figure layout matching `tools/viz_archive/visualize_larmatch_h5data.py`. Reads the same config so what's shown is exactly what the model will see.

### Phase 2b — Optional voxel-level feature cache *(deferred until profiling)*

If, during Phase 4 training, backbone inference becomes a measurable per-step bottleneck, add voxel-pool-only cache (~22 MB/event × 52k = ~1 TB, feasible). This is an optimization, not a correctness requirement.

### Phase 2b — Optional voxel-level feature cache *(deferred until profiling)*

If, during Phase 4 training, backbone inference time becomes a measurable bottleneck per training step, add a voxel-level-only cache:
- voxel-pooled features only (~10k voxels/event × 1088 dim × 2 bytes ≈ 22 MB/event)
- ~1 TB total — feasible
- Does not cache per-spacepoint features (still computed by the backbone on-the-fly for spacepoint refinement)

This is an optimization, not a correctness requirement.

### Phase 3 — Tokenizer module *(COMPLETE 2026-05-04)*

Trainable spacepoint→fragment pool + per-fragment / per-voxel positional encoders + voxel mean-pool. Voxel mean-pool is deterministic (no learnable params for the pool itself).

- [x] **`pointcept/models/shower_clustering/tokenizer.py`**:
  - `FragmentPool` — mini set-transformer (default 2 self-attn layers, 8 heads, MLP ratio 4) over each fragment's spacepoints. Learnable pool query is the per-fragment output token.
  - `VoxelPool` — `index_add_`-based mean pool + linear projection (no learnable pool query).
  - `FragmentPositionalEncoder` — 13-dim per-fragment geometric features (centroid + PCA axis + bbox extent + log point count + mean strength) → MLP → token_dim.
  - `VoxelPositionalEncoder` — 4-dim per-voxel features (voxel center + log point count) → MLP → token_dim.
  - `ShowerClusteringTokenizer` — top-level wrapper; emits `spacepoint_tokens (N, D)`, `fragment_tokens (F, D)`, `voxel_tokens (V, D)`.
- [x] **Memory cap**: `frag_pool_max_points` (default 512) randomly subsamples oversize fragments before pooling. Bounded MultiheadAttention's O(M²) memory; without it, events with thousand-point fragments would OOM. The pool token represents fragment identity, not exhaustive coverage, so subsampling preserves correctness.
- [x] **Smoke test on 3 fixed-merge events** (CPU, synthetic 128-dim per-spacepoint features). Shapes correct for all three token sets, no NaN, all 43 trainable params receive non-zero gradients on a dummy loss. Empty-fragment edge case handled. Total trainable params 1.8 M at token_dim=256, hidden=256, 2 fragment-pool layers.

### Phase 4 — Mask2Former decoder *(COMPLETE 2026-05-04)*

- [x] **`pointcept/models/shower_clustering/decoder.py`** with:
  - `_MaskedDecoderLayer` — pre-norm masked cross-attn → query self-attn → FFN.
  - `Mask2FormerDecoder` — N=64 learnable queries, L=6 stacked layers, init-layer predictions to gate the first cross-attn (Mask2Former-style), per-layer predictions for deep supervision (loss applied at init + every layer in Phase 6).
  - `QueryPositionalEncoder` — static learnable per-query + MLP on per-layer predicted origin coord (origin head feeds back as PE for next layer).
  - `_PerLayerHeads` — class logits, origin coord, and shared mask-embed; mask logits per scale = `mask_embed(q) @ key.T`.
- [x] **Scale rotation pattern**: `[voxel, fragment, voxel, fragment, spacepoint, spacepoint]` (default). Rationale: spacepoint-scale cross-attention has K up to 250k; the first cross-attn at any scale is unmasked (no prior mask to gate by), so unmasked spacepoint cross-attn would dominate memory. Putting both spacepoint layers at the end means they can always gate by a previously-computed spacepoint mask. Configurable via `scale_pattern` arg.
- [x] **Safety net for all-masked queries**: standard Mask2Former trick — if a query's attention mask covers all keys, drop the mask for that query (else softmax NaNs).
- [x] **Smoke test on 3 fixed-merge events** (CPU, synthetic 128-dim per-spacepoint features): all per-layer output shapes correct ((Q, V), (Q, F), (Q, N) for masks; (Q, C) for class; (Q, 3) for origin); no NaN/Inf in any output; all 239 params (tokenizer + decoder) receive non-zero gradients on a deep-supervision-style dummy loss; |Δclass| from init→final ≈ 0.45–0.54 confirms the decoder is actually updating predictions (not a no-op). Edge case (empty fragment list) handled.
- [x] **Param counts**: tokenizer 1.8 M + decoder 7.8 M = **9.6 M trainable params** at default sizes (token_dim=256, num_queries=64, 6 decoder layers, 8 heads). Backbone (Sonata, frozen) is separate.

**Open optimization (Phase 6/7 concern)**: spacepoint mask logits are computed at every layer ((Q, N) per layer × 7 layers). At N=250k, this is ~640 MB per forward pass at fp32. Acceptable on H200 (120 GB) but tight on P100 (16 GB) once backbone activations + gradients are added. If profiling during training shows memory pressure, two cheap optimizations:
1. Compute spacepoint mask logits **only at layers that do spacepoint cross-attn next** + final (i.e. at the 2 spacepoint layers and the final layer = 3 of 7 layers).
2. Use chunked computation for spacepoint mask logits.

Both preserve the deep-supervision signal at the cheap scales (voxel + fragment).

### Phase 5 — Heads

- [ ] **`pointcept/models/shower_clustering/heads.py`**:
  - `ClassHead` — 6-way per-query
  - `MaskHead` — produces fragment mask logits + spacepoint refinement logits (gated by predicted fragment mask)
  - `OriginHead` — 3D coord per query (auxiliary, also feeds next-layer query PE)

### Phase 6 — Loss and matcher *(COMPLETE 2026-05-04)*

- [x] **`pointcept/models/shower_clustering/matcher.py`** — `HungarianMatcher` using `scipy.optimize.linear_sum_assignment`. Cost = λ_cls·(-p[gt_class]) + λ_mask·BCE + λ_dice·Dice + λ_origin·L1, all evaluated on a sampled spacepoint subset (Mask2Former-style point sampling at S=4096).
- [x] **`pointcept/models/shower_clustering/losses.py`** — `ShowerClusteringLoss` with:
  - **Single Hungarian solve** on the model's final-layer prediction (Mask2Former-style — same matching reused for all layers' losses).
  - **Deep supervision**: loss summed across init layer + every decoder layer (7 supervision points at default 6-layer config).
  - **Per matched pair**: CE class loss, BCE + Dice on sampled-S spacepoint mask, L1 on origin coord, BCE aux losses on voxel + fragment scale masks.
  - **Per unmatched pair**: CE with target = no_object class, weighted at 0.1 (Mask2Former default) so unmatched queries don't dominate the gradient.
  - **GT instances** include `origin_type`, `origin_coord_norm`, `truth_indices`, `trunk_trackid` — the dataset was extended in Phase 6 to populate the first three (origin_type and originpt taken from the first surviving fragment with this trunk trackid).
- [x] **Toy matcher test** (3 queries, 2 GTs, planted class costs): assignment is `[(q=0, k=1), (q=1, k=0)]` as expected. Hungarian solve correct.
- [x] **End-to-end smoke on 3 fixed-merge events**: all 3 events (28, 20, 21 GT instances) match every GT (n_matched == n_gt since K << Q=64). Initial total loss ~125 across 7 supervised layers (cls ~14, mask ~7, dice ~7, origin ~17, aux_voxel ~7, aux_frag ~5). All 239 trainable params receive non-zero gradients on backward.

The five trainable modules now wire together end-to-end on the test events: ShowerClusteringDataset → ShowerClusteringTokenizer → Mask2FormerDecoder → ShowerClusteringLoss. Phase 7 (model assembly + training loop integration) is the next milestone.

### Phase 7 — Model assembly *(COMPLETE 2026-05-04)*

- [x] **`pointcept/models/shower_clustering/model.py`** — `ShowerClusteringMask2Former` registered with the Pointcept `MODELS` registry. Wires:
  - frozen Sonata backbone (built via Pointcept's MODELS registry from the config's `backbone` dict)
  - `ShowerClusteringTokenizer`
  - `Mask2FormerDecoder`
  - `ShowerClusteringLoss`
- [x] Per-event iteration: backbone runs once on the flat-batched (sum N_b, 6) input dict, then tokenizer/decoder/loss run per-event in a loop because Hungarian matching is per-event. Loss aggregated as mean across events.
- [x] **`pointcept/models/__init__.py`** updated to import `from .shower_clustering import *` so the `@MODELS.register_module()` decorator runs at package load.
- [x] **Dataset extended** to emit `grid_coord` (PT-v3 expects integer voxel indices for serialization) AND **deduplicate** spacepoints to one entry per 0.25 cm cell. Both are required by PT-v3's serialized attention; without dedup the encoder's stride pooling produces out-of-bounds indices on the GPU. Matches V3's `GridSample(grid_size=0.25)` transform.
- [x] **Backbone `up_cast_level=4`** (vs V3's default `up_cast_level=2`). The encoder has 4 GridPooling stages, so up-casting only twice leaves features at ~1/64 of the input resolution — fine for V3's tiny cropped-fragment inference, but our full-event Mask2Former needs per-spacepoint masks. Setting `up_cast_level=4` un-pools all the way back to the input level. Feature dim becomes 48+96+192+384+512 = **1232** (vs 1088 with up_cast_level=2). The pretrained checkpoint loads cleanly because up_cast is gather+concat with no learnable params.
- [x] **Config** `pointcept/configs/lartpc/shower_origin/archive/shower-cluster-sonata-v1.py` updated from a model-stub to a full buildable config. `flash_backend` defaults to `xformers` (works on P100 and H200; the V3 default `flash_attn` only works on Hopper / newer).
- [x] **CPU smoke test with mock backbone** at `pointcept/tools/test_shower_clustering_assembly.py`. Verified:
  - The model builds from the config end-to-end via `MODELS.build(cfg.model)`.
  - On a 2-event batch (479k spacepoints, 4 169 voxels, 162 fragments, 48 GT instances), forward returns finite loss and backward propagates gradients to all 239 trainable params.
  - Total trainable params at production config (in_dim=1088): **10.4 M**.
- [x] **GPU run with real Sonata** (P100, 2026-05-04). Random-init backbone forward + backward succeeded after the dedup + up_cast_level=4 fixes. Loss = 158.14.
- [x] **GPU run with pretrained Sonata weights**. Added `--load-weights` flag to the assembly test. Pretrained checkpoint `lartpc_v6_h200_noghosts_pretrain_logspace_resume/model/epoch_42.pth` loaded 196.7 M / 196.7 M backbone params (100%). Two unexpected keys (`*.embedding.mask_token`) ignored (correct — our config has `mask_token=False`). Loss dropped to 120.49, confirming pretrained features are flowing through.

### Phase 8 — Training entrypoint smoke test *(COMPLETE 2026-05-04)*

- [x] **`pointcept/models/shower_clustering/trainer.py`** — `ShowerClusteringTrainer` subclasses Pointcept's `Trainer` (registered as `DefaultTrainer`) and overrides `build_train_loader` / `build_val_loader` to use `shower_clustering_collate`. Pointcept's hard-coded `point_collate_fn` doesn't know how to batch our nested per-event lists.
- [x] **`pointcept/configs/lartpc/shower_origin/archive/shower-cluster-sonata-v1.py`** — extended from a model-only config to a full training config: `_base_ = ["../_base_/default_runtime.py"]`, smoke-test knobs (`epoch=1`, `batch_size=1`, `num_worker=0`, `evaluate=False`, `enable_wandb=False`), `SonataCheckpointLoader` hook, and `train = dict(type="ShowerClusteringTrainer")`.
- [x] **Side-effect import** in the config: `import pointcept.models.shower_clustering.trainer as _trainer_module; del _trainer_module`. Done in the config (not the package `__init__.py`) to avoid a circular import — `pointcept.engines.train` imports `pointcept.models` *before* defining `TRAINERS`. Underscore-aliased + del'd so `Config.dump`'s yapf round-trip doesn't see a `<class …>` repr.
- [x] **Model output contract**: `forward()` returns flat-scalar values only (`loss`, `loss_cls`, `loss_mask`, …) when training, because Pointcept's `InformationWriter.after_step` calls `.item()` on every value in the model output dict. Predictions are returned only on eval.
- [x] **GPU smoke test on 3 fixed-merge events** (1 epoch, batch=1, P100): loss decreased monotonically across iterations (116.7 → 92.4 → 91.1), checkpoint saved cleanly. SONATA weights load correctly (100% backbone match; the long "Missing keys" log is just our untrained tokenizer / decoder / loss params — strict=False handles it).

### Phase 8.b — Validation evaluator *(implemented 2026-05-04, untested at scale)*

- [x] **`pointcept/engines/hooks/shower_clustering_evaluator.py`** — `ShowerClusteringEvaluator` hook. Per epoch (or every N steps via `eval_freq`):
  - Runs val_loader through model in eval mode + the loss_fn with `return_matching=True` to recover the Hungarian assignment.
  - Computes per-matched-pair metrics:
    - **Class accuracy** (overall + per-class)
    - **Mask IoU** at the spacepoint scale, computed on the *full* spacepoint mask (binarized at logit > 0), not the loss's sampled subset — gives an unbiased per-shower IoU estimate. Also reported per class.
    - **Origin error in cm** (denormalized via `cfg.coord_scale`)
  - Set-prediction sanity: `matched_fraction = n_matched / n_gt_total`, `n_active_queries_mean` (queries whose argmax class isn't no_object).
  - Per-component val loss averages.
  - Writes to TensorBoard always; to wandb when `cfg.enable_wandb=True`.
  - Sets `current_metric_value` from `mask_iou_mean` for `CheckpointSaver`.
- [x] **`ShowerClusteringLoss.forward(return_matching=True)`** — extended the loss module to optionally return Hungarian assignment + GT tensors so the evaluator doesn't redo the matching.
- [x] **Hook registered** in `pointcept/engines/hooks/__init__.py` (matches the `ShowerOriginEvaluator` registration pattern).
- [x] **Wired into the config** (commented out for the smoke test). Real training: uncomment, set `evaluate=True`, set `enable_wandb=True`.
- [ ] **Field-test on a few-event val set with `evaluate=True`** — verify the evaluator runs cleanly end-to-end. To be done by the user on the P100 once they want eval metrics.

### Phase 8.c — Loss-formulation tuning on 3-event memorize set *(2026-05-04)*

Iterative debugging of the mask + origin losses on a 3-event overfit test. Each row is a 100-epoch run on the same 3 fixed-merge events; metrics are at peak `mask_iou_mean`.

| Run | Sampling | Origin loss | mask_iou_mean | mask_iou_p25 | origin_err_cm | cls_acc |
|---|---|---|---|---|---|---|
| v1 | uniform random S=4096 | sum-axes, w=1 | **0.003** | 0 | 100 | 0.80 |
| v2 | union-balanced 50% pos | sum-axes, w=1 | 0.032 | 0.013 | 240 | 0.97 |
| v3 | per-pair exact 50/50 | mean-axes, w=3 | 0.014 | 0.004 | 154 | 0.985 |
| v4 | per-pair pos-biased 50% (no equalize) | mean-axes, w=3 | 0.043 | 0.022 | 140 | 0.98 |
| **v5** | **+ importance-sampled negatives** | mean-axes, w=3 | **0.059** | 0.008 | 144 | **1.000** |

Lessons learned:

- **Uniform random S=4096 is unworkable for sparse-mask training**: only ~3 of 4096 sampled points are positive for a typical 110-pt instance, and the model trivially minimizes BCE by predicting all zeros (v1 → mask IoU ≈ 0).
- **Balanced sampling (union- or per-pair) bootstraps the masks** but must keep wide negative coverage. Equalizing pos:neg per pair (v3) starves the model of negative supervision and the over-prediction halo grows back. Pos-biased without equalize (v4) recovers.
- **Origin loss should be per-axis-mean** to make it batch-size and matched-pair-count invariant. Compensate with `weight_origin = 3.0` to keep the same effective magnitude (v3, v4, v5).
- **Importance sampling for negatives** (PointRend-style — pick top-k uncertain points where `|sigmoid - 0.5|` is small, plus random for diversity) lifts mean IoU 4× over v3 and is essentially free at run-time. Caveat: tail (p25) regresses slightly because small/scattered instances don't have a coherent halo to focus on.
- **Per-class IoU at v5 peak**: inside 0.156, outside 0.087, on_track 0.042 — the easier classes (pi0 EM showers — `inside`/`outside`) gain most; cosmic-derived `on_track` instances drag the tail.

Default config has `use_importance_sampling=False` to preserve the v4 baseline; flip to `True` for the v5 recipe.

### Phase 8.d — Real training run *(next)*

- [ ] Switch the config's data lists to the production training set (re-merged pi0filter once that finishes).
- [ ] Bump knobs to real values: `epoch=N`, `batch_size_per_gpu>1` if memory permits, `num_worker=K`, `evaluate=True`, `enable_wandb=True`.
- [ ] DDP on 6×P100 with `--num-gpus 6`. Monitor: matcher stability (`val/matched_fraction` ≈ 1.0), per-component loss curves, per-class accuracy / mask IoU, origin-error distribution.
- [ ] Ablation: `importance_ratio = 0.5` vs `0.75` — the lower value preserves more uniform negative coverage and may recover the v4 p25 while keeping most of the v5 mean gain.
- [ ] LR schedule revisit — the OneCycleLR decays to zero by epoch 100 and the model plateaus at ~60. A constant or step schedule should give more headroom on real training.
- [ ] **Decision point:** does mask IoU on val exceed V3's effective downstream merge quality?

### Phase 9 — Inference and pipeline integration

- [ ] **`ub_showerorigin_reco/tools/run_shower_clustering_inference.py`** — load trained model, run on merged H5, write per-event ROOT TTree (same schema as current Steps 5–7 output)
- [ ] **`ub_showerorigin_reco/scripts/run_step567_clustering_wconfig.sh`** — sibling of current step567 driver, swappable via config
- [ ] Validate: same input event produces sensible TTree row; per-shower mask matches expectations visually
- [ ] **Decision point:** ready to replace Step 5–7 entirely?

### Phase 10 — Production deploy and ablations

- [ ] Full reprocess of NC pi0 dataset
- [ ] Compare downstream selection metrics vs V3-based pipeline
- [ ] Ablations: drop voxel scale; drop spacepoint refinement; vary N_queries; vary voxel size
- [ ] Update this document with results

---

## 6. Hardware and compute strategy

| Machine | Role | Notes |
|---|---|---|
| **2×H200 (120 GB)** | Phase 2 bulk feature cache; later end-to-end retraining if backbone unfreezes | bf16 OK, lots of memory headroom |
| **6×P100 (16 GB)** | Phase 4–8 decoder iteration via DDP | matches V3 memory budget; cache makes decoder cheap |
| **Laptop RTX 3090 (16 GB)** | Code-correctness sanity runs, single-event debugging | not for real training |

**Why this works:** the backbone is the expensive part and it's frozen, so caching its output decouples backbone cost from decoder iteration. After Phase 2, decoder iteration runs comfortably on 16 GB cards.

---

## 7. Reference: data schema and labels

### Origin classes (5)
From [merge_reco_truth_showerorigin.py:34](../../../ub_showerorigin_reco/ubshowerorginreco/merge_reco_truth_showerorigin.py):

| ID | Label | Meaning |
|---|---|---|
| 0 | `inside` | shower originates from the in-time neutrino interaction |
| 1 | `outside` | shower originates from outside the TPC (e.g. dirt) |
| 2 | `on_track` | shower originates from a track (e.g. delta ray, bremsstrahlung) |
| 3 | `ghost` | spacepoint not real (3D reconstruction artifact) |
| 4 | `true_track` | spacepoint is a track-tagged spacepoint mis-routed into shower fragmentation |

A 6th class `no_object` is added at the model level for unmatched queries.

### Merged H5 keys used (from [pointcept/datasets/shower_origin.py:8](../../pointcept/datasets/shower_origin.py))

Per event under `entry_<i>/shower_fragments`:
- `trackid` — true particle ID per fragment (multiple fragments share trackid → same shower)
- `pid` — PDG code
- `type` — origin class (0–4)
- `istrunk` — boolean, primary fragment per shower
- `startpt` — fragment start point (cm)
- `originpt` — apparent particle origin (cm)
- `pret0shiftedoriginpt` — true MC origin (cm); not a learning target
- `pointindices_flat`, `pointindices_counts` — spacepoint membership per fragment

Per-spacepoint truth (in the `triplet_data` group): `trackid`, `pid`, `origin`. Used to construct GT instance masks.

### Coordinate frames
- H5 raw `pos` / `coord` and the merger run in **detector cm**
- Model normalized coords: `(cm − [125, 0, 518]) / 179.55`
- ROOT outputs: detector cm, except `pret0shiftedoriginpt` which is true MC coords

---

## 8. Glossary

- **Fragment** — DBSCAN cluster of shower-tagged spacepoints (only shower-tagged; tracks/ghosts are NOT fragmented).
- **Instance** — a true shower (one `trackid`); the GT target for Mask2Former.
- **Query** — a learnable embedding that the decoder iteratively refines into a (class, mask) prediction.
- **Voxel token** — coarse-scale token from mean-pooling spacepoint features in a fixed-size voxel; carries non-shower context (tracks, ghosts, vertex).
- **Hungarian matching** — bipartite assignment of N queries to M GT instances minimizing set-prediction cost; standard Mask2Former / DETR matcher.
- **Masked attention** — Mask2Former trick: each decoder layer's cross-attention only attends to keys where the previous layer's mask logit was positive. Speeds convergence.
