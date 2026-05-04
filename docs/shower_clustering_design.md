# Shower Clustering Model — Design and Implementation Plan

**Status:** Phase 1 in progress. First diagnostics on 500 NC pi0 training events done 2026-05-04.
**Owner:** taritree.wongjirad@tufts.edu
**Replaces:** `ShowerOriginPredictorV3` (per-point regression and slot-attention origin prediction).

This document is the living design reference. Update as decisions change or phases complete.

---

## 1. Motivation

`ShowerOriginPredictorV3` (defined in [pointcept/models/shower_origin/shower_origin_model.py:769](../pointcept/models/shower_origin/shower_origin_model.py)) classifies fragments well — inside / outside / cosmic accuracies are all >90%. But:

- The **regression head** (predicted origin coord per slot) is much noisier than the origin scores. It does not localize the start point reliably.
- **Start-point labeling is too inconsistent across fragments of the same shower** to drive the cone-based merger in [ub_showerorigin_reco/ubshowerorginreco/shower_fragment_merger.py](../../ub_showerorigin_reco/ubshowerorginreco/shower_fragment_merger.py).
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
| Replaces cone merger | yes — model output IS the merged shower | Steps 5–7 of the [pipeline](../../ub_showerorigin_reco/CLAUDE.md) collapse to "run model, write TTree" |

### New code location

- New directory: `pointcept/models/shower_clustering/` (sibling to `shower_origin/`)
- New configs in: `pointcept/configs/lartpc/shower-clustering-*.py`
- New dataset class: `pointcept/datasets/shower_clustering.py`
- New data-prep / characterization scripts: `pointcept/lartpc_data_prep/`
- Pipeline integration in `ub_showerorigin_reco`: new `tools/run_shower_clustering_inference.py`, eventually replacing the Step 5–7 driver.

---

## 4. Open questions and things to revisit on data

### 4a. Phase 1 findings (2026-05-04, n=500 NC pi0 events)

Run via `lartpc_data_prep/characterize_fragments.py`, output in `pointcept/exp/shower_clustering/phase1_smoke_500events/`.

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

The new merge-step output (test files in `pointcept/lartpc_data_prep/lantern_scripts/tmp_workdir/lantern_bnb_nu_pi0filter_corsika_jobid0000_line00001/`) preserves `mc_particle_tree` (`trackid`, `parent_trackid`, `pid`, `process_code`, `start_pos`, `energy_mev`, `origin`). This unblocks **option (a)** from the previous design discussion: walk the Geant4 tree to define GT instances.

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

- [x] **`pointcept/lartpc_data_prep/characterize_fragments.py`** — implemented, supports per-event diagnostics, stratified IoU by instance size, orphan→fragment distance, fragment purity, summary H5 + plots.
- [x] Run on 500 events from `bnb_nu_pi0filter_corsika` training H5s. Outputs in `pointcept/exp/shower_clustering/phase1_smoke_500events/`. See §4a above for results.
- [x] **Decision: keep voxel scale (mandatory, see §4c), keep spacepoint refinement (45% orphan rate makes it critical), 5 cm voxel size confirmed, DBSCAN params unchanged (bottleneck is upstream SSNet, not DBSCAN).**

### Phase 1.5 — GT instance definition *(RESOLVED 2026-05-04)*

- [x] **Inspect `mc_particle_tree` schema** in merged H5. Confirmed: `trackid`, `parent_trackid`, `pid`, `process_code`, `start_pos`, `energy_mev`, `origin`, `daughter_trackids`, `daughter_start_indices`, `num_daughters`, `nu_vertices`. New merge step preserves it.
- [x] **GT instance source decided: option (a)** — walk `mc_particle_tree.parent_trackid` from each unique non-(-1) `shower_fragments/trackid` to gather descendants. See §4d.
- [x] **`characterize_fragments.py` updated** to support both new (trunk-descendant) and old (per-trackid legacy) schemas; reports which mode was used per event.
- [x] **Validation on 3 test events** done, see §4d table.
- [ ] **Re-merge all training data** with the fixed merge step before Phase 4 begins.

### Phase 2 — Dataset class *(revised: no feature cache for first cut)*

**Storage reality check (2026-05-04)**: Caching per-spacepoint backbone features at full precision is infeasible. 254k spacepoints/event × 1088 dim × 2 bytes (fp16) = 540 MB/event; × 52k events = 28 TB. A voxel-level-only cache would be ~22 MB/event × 52k ≈ 1 TB and feasible, but isn't needed until profiling shows the backbone is the bottleneck. **Defer caching entirely; run the frozen backbone on-the-fly each training step**, inside the Mask2Former model's forward pass.

Phase 2 reduces to: build the dataset class that loads the merged H5 and emits everything the Mask2Former model needs.

- [ ] **`pointcept/datasets/shower_clustering.py`** — `ShowerClusteringDataset` class. Per `__getitem__(idx)` return:
  - `coord` (N, 3) detector cm; `coord_norm` (N, 3) normalized for backbone input
  - `feat` (N, 6) coord(3) + strength(3) for backbone input
  - `lm_score` (N,) — used for threshold augmentation
  - `hasmatch`, `trackid`, `pid`, `origin`, `ssnet_label` — per-spacepoint truth
  - `voxel_id` (N,) — voxel index per spacepoint at fixed 5 cm grid
  - `fragment_indices` — list of arrays, one per DBSCAN fragment (post-threshold filter)
  - `fragment_trackid` (F,) — per-fragment plurality trackid
  - `gt_instances` — list of dicts: `{trunk_trackid, descendants, n_truth_points, dom_pid}`
  - `n_spacepoints`, `n_voxels`, `n_fragments`, `n_gt_instances` — scalars
  - `name`, `run`, `subrun`, `event` — identity
- [ ] **lm_score augmentation hook**: at `__getitem__` time, sample τ ~ U(0.15, 0.40) on train, fixed 0.15 on val (per §4e). Apply mask: drop spacepoints below τ, drop fragments < 20 surviving pts, recompute voxel ids on surviving spacepoints.
- [ ] **GT-instance precomputation** in dataset: walk `mc_particle_tree.parent_trackid` to gather descendants for each unique non-(-1) `shower_fragments/trackid`. Cache the per-event descendants in memory after first load (small).
- [ ] **Smoke test on 3 test events**: confirm shapes, check that filtering at τ=0.15 returns identical results to no-filter (since H5 floor is 0.15), check τ=0.30 reduces spacepoint count and drops some fragments, check GT instance count matches `len(unique trunk_trackid)`.
- [ ] **Integration test**: register dataset with Pointcept's `DATASETS` builder, verify it builds from a config dict, verify `point_collate_fn` collates a batch correctly.

### Phase 2b — Optional voxel-level feature cache *(deferred until profiling)*

If, during Phase 4 training, backbone inference time becomes a measurable bottleneck per training step, add a voxel-level-only cache:
- voxel-pooled features only (~10k voxels/event × 1088 dim × 2 bytes ≈ 22 MB/event)
- ~1 TB total — feasible
- Does not cache per-spacepoint features (still computed by the backbone on-the-fly for spacepoint refinement)

This is an optimization, not a correctness requirement.

### Phase 3 — Tokenizer module

Goal: Trainable spacepoint→fragment pool. (Voxel pool is deterministic mean and lives in the cache.)

- [ ] **`pointcept/models/shower_clustering/tokenizer.py`**:
  - `FragmentPool` — mini set-transformer (2–3 self-attn layers over spacepoints in a fragment + learnable pool query)
  - Positional encoders: per-fragment (centroid, PCA axis, length, point count, mean energy) and per-voxel (centroid)
- [ ] Unit test on a single event: shapes, NaN-free output, gradient flows

### Phase 4 — Mask2Former decoder

- [ ] **`pointcept/models/shower_clustering/decoder.py`**:
  - `MultiScaleMaskedDecoderLayer` — masked cross-attn (rotating scale) + query self-attn + FFN
  - `Mask2FormerDecoder` — N learnable queries + L stacked layers + per-layer aux outputs (for deep supervision)
- [ ] Position encoding: learnable query embedding + MLP on per-layer predicted origin coord
- [ ] Unit test: forward pass on a synthetic event; check attention masks are applied correctly

### Phase 5 — Heads

- [ ] **`pointcept/models/shower_clustering/heads.py`**:
  - `ClassHead` — 6-way per-query
  - `MaskHead` — produces fragment mask logits + spacepoint refinement logits (gated by predicted fragment mask)
  - `OriginHead` — 3D coord per query (auxiliary, also feeds next-layer query PE)

### Phase 6 — Loss and matcher

- [ ] **`pointcept/models/shower_clustering/matcher.py`** — Hungarian matcher (`scipy.optimize.linear_sum_assignment`); cost = λ_cls·CE + λ_mask·BCE + λ_dice·Dice
- [ ] **`pointcept/models/shower_clustering/losses.py`** — set-prediction loss with deep supervision (loss applied at every decoder layer)
- [ ] Unit test: matcher returns sensible assignments on toy 2-instance / 3-query case

### Phase 7 — Model assembly

- [ ] **`pointcept/models/shower_clustering/shower_clustering_model.py`** — top-level model class registered to Pointcept's MODELS registry. Inputs: cached features + fragment indices. Outputs: per-query class, mask, origin.
- [ ] Add to `pointcept/models/__init__.py`

### Phase 8 — Configs and training

- [ ] **`pointcept/configs/lartpc/shower-clustering-v1-baseline.py`** — frozen backbone, fragment-only mask first (Option A), to verify training converges
- [ ] **`pointcept/configs/lartpc/shower-clustering-v1-multiscale.py`** — full design (Option D)
- [ ] Training on 6×P100 DDP, batch=1 event/GPU. Monitor matcher stability, class loss, mask AP.
- [ ] **Decision point:** does mask AP on val set exceed V3's effective downstream merge quality?

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
From [merge_reco_truth_showerorigin.py:34](../../ub_showerorigin_reco/ubshowerorginreco/merge_reco_truth_showerorigin.py):

| ID | Label | Meaning |
|---|---|---|
| 0 | `inside` | shower originates from the in-time neutrino interaction |
| 1 | `outside` | shower originates from outside the TPC (e.g. dirt) |
| 2 | `on_track` | shower originates from a track (e.g. delta ray, bremsstrahlung) |
| 3 | `ghost` | spacepoint not real (3D reconstruction artifact) |
| 4 | `true_track` | spacepoint is a track-tagged spacepoint mis-routed into shower fragmentation |

A 6th class `no_object` is added at the model level for unmatched queries.

### Merged H5 keys used (from [pointcept/datasets/shower_origin.py:8](../pointcept/datasets/shower_origin.py))

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
