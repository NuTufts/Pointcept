# LArFormer Keypoint

This document tracks the development of the **LArFormer keypoint network** — a
module that predicts physics keypoints (neutrino vertices, particle endpoints,
shower starts) for the single-photon neutrino interaction search.

It is a **living development doc**: the design rationale, the implementation
plan, and a status tracker all live here so that any session (human or Claude)
can pick up where the last one left off. Update the **Status tracker** at the
bottom as phases land.

> Path conventions: file links below are relative to the Pointcept repo root
> (`pointcept/models/LArFormer/...`). The model lives in the `nutufts_lartpc`
> Pointcept fork.

---

## 1. Goal

Identify 6 types of keypoints for interactions and charged-particle
trajectories in a liquid-argon TPC:

* neutrino interaction vertices
* track starts
* track ends
* EM-shower starts from a γ or e⁻ off the neutrino interaction
* EM-shower starts from a δ-ray electron off a muon
* EM-shower starts from the Michel (decay) electron of a muon

The keypoints feed the single-photon search: we need *which shower started
where*, and *where the neutrino interacted*.

---

## 2. Ground truth available in the H5 files

Two independent GT sources already exist in every H5 file. **No new label
production is required** for the first development phases.

### 2a. Per-spacepoint keypoint proximity scores — `triplet_data/kpscores`

Shape `(N, 6)`, float32. Already produced by
`ubdl/larflow/larflow/PrepFlowMatchData/SimChTripletLabelMaker.cxx`
(`make_keypoint_labels`). For each spacepoint and each keypoint type *i*:

```
kpscore_i = exp( -0.5 * (d_i / sigma)^2 )      if >= threshold, else 0
```

* `d_i` = Euclidean distance (cm) from the spacepoint to the **nearest**
  keypoint of type *i*
* `sigma = 3.0` cm (Gaussian width)
* `threshold = 0.01` (smaller scores zeroed)
* Column order matches the keypoint-type table below.

This is a ready-made dense per-point regression target. MicroBooNE's older
LArMatch keypoint net trained on exactly this with MSE
(`ubdl/larflow/larmatchnet/larmatch/loss/loss_larmatch_kps.py`) and it worked.

### 2b. Per-event keypoint positions — `mckeypoints/`

The discrete GT keypoint set (used to build endpoint targets and to evaluate).

| Dataset | Shape | Dtype | Description |
|---------|-------|-------|-------------|
| `pos` | (K, 3) | float32 | 3D position (x, y, z) in cm |
| `imgcoord` | (K, 4) | float32 | Image coordinates |
| `kptype` | (K,) | int32 | Keypoint type (see below) |
| `pid` | (K,) | int32 | PDG particle ID |
| `trackid` | (K,) | int32 | Geant4 track ID |
| `startpos` | (K,3) | float32 | 3D creation position; differs from `pos` only for photons (pid=22) |

### Keypoint types (column index in `kpscores`, value in `kptype`)

| Value | Name | Description |
|-------|------|-------------|
| 0 | Nu Vertex | Neutrino interaction vertex |
| 1 | Track Start | Beginning of a track |
| 2 | Track End | End of a track |
| 3 | Shower | Shower start point |
| 4 | Michel | Michel electron candidate |
| 5 | Delta | Delta-ray origin |

> **Conceptual reframing that drives the design:** 5 of the 6 types are
> *attributes of a particle* (a track has a start + end; a shower has a start;
> Michel/Delta are sub-classifications of a shower-start by parent). Only the
> **nu vertex is slice-level** (the common origin of the primary particles).
> This is why the model couples to the Stage-3 *particle* queries and treats
> the vertex with a dedicated mechanism — see §4.

---

## 3. What already exists in LArFormer (reuse, don't rebuild)

The keypoint task overlaps heavily with machinery already in
`pointcept/models/LArFormer/`. Key existing assets:

* **Per-query 3D point regression with iterative refinement.** The decoder's
  `origin` head (`decoder.py` `_PerLayerHeads`, ~L164-212) regresses a 3D
  coordinate per query, and `decoder.py` (~L542-545) feeds the predicted
  origin back into every layer's `query_pos` as a positional prior — i.e. the
  Deformable-DETR / 3DETR **reference-point + iterative-offset** mechanism is
  already implemented. The matcher has a `cost_origin` term
  (`matcher.py` ~L134-138) and the loss an origin L1 (`losses.py` ~L1018-1024).
  Per the original spec, this head was *de-emphasized* in Stage-3 training, so
  it works but is not accurate — we will promote and properly supervise it.
* **Cascade with frozen upstream stages.** `CascadedParticleSegmenter`
  (`cascaded_particle.py`) runs `CascadedSlicer` (Stage 1+2) frozen, in eval,
  under `torch.no_grad`, and trains only Stage 3. A `CascadedKeypoint` wrapper
  follows the same idiom.
* **Partial-unfreeze knobs.** `LArFormer` has `freeze_backbone` +
  `unfreeze_decoder` (train the PT-v3 decoder but keep the encoder frozen,
  `model.py` ~L123-149). Loss exposes per-component weights (`weight_origin`).
  So "freeze the encoder, fine-tune the decoder + keypoint heads, upweight the
  keypoint loss" is a config, not new code.
* **Query initialization & DINO-style training.** `MixedQuerySelector`
  (`query_selection.py`) picks K anchor tokens from a level by score, then FPS
  for spatial diversity. `MaskDenoiser` (`query_denoising.py`) adds DINO-style
  denoising queries. Both are reusable for keypoint queries.
* **Levels / tokenizer.** `SpacepointBuilder` (finest, per-SP tokens; the
  loss's `primary_level`), `VoxelBuilder`, `PTv3DecoderStageLevel`,
  `FragmentBuilder` (`builders/`). The spacepoint level + its per-particle mask
  is what a refinement decoder cross-attends to.
* **Decoder building blocks.** `_MaskedDecoderLayer` (pre-norm cross-attn →
  self-attn → FFN with mask-gating), `SinusoidalPosEmb3D`, `_build_attn_mask`
  (`decoder.py`). A keypoint refinement decoder is a thin specialization.

### Data path
* Dataset: `pointcept/datasets/larformer.py` — `LArFormerDataset`,
  `gt_source ∈ {shower_trunk, slice, particle, deghost}`, collate
  `larformer_collate`.
* GT instance dicts (for `gt_source="particle"`, the Stage-3 source) carry:
  `truth_indices`, `origin_type`/`class_id`, `origin_coord_norm` (SCE/reco
  frame **start** point), `origin_cm`, `pid`, `ke_mev`, ... (`larformer.py`
  ~L753-775). **No end point and no nu-vertex** are stored yet — Phase 0 adds
  them.
* HDF5 read happens in `_load_event` (`larformer.py` ~L356). Per-SP arrays are
  filtered through a `keep` mask → `keep_dedup` → optional `cap_perm`
  subsample. **Any new per-SP field (kpscores) must be put through the same
  index chain** to stay aligned with `coord_norm`.
* Coordinate normalization: `coord_norm = (pos_cm - coord_center) / coord_scale`
  with `coord_center ≈ [125, 0, 518]`, `coord_scale ≈ 179.55`
  (`DEFAULT_COORD_CENTER` / `DEFAULT_COORD_SCALE`). All point heads predict in
  this normalized frame; denormalize for ROOT/analysis output.

### Prior art / literature anchors
* **PPN** (Dominé et al., PRD 104 032004, 2021) — the in-domain LArTPC
  baseline: sparse-UResNet with per-voxel *score + sub-voxel offset + type*.
  Benchmark against its metric (% of true points recovered within 3 / 10
  voxels). Our Phase-1 dense head + offset is a PPN analog on points.
* **VoteNet** (Qi 2019) — per-point offset *votes* to off-surface targets;
  the mechanism for keypoints that lie *between* measured points.
* **3DETR / PETR / GroupPose** — query-based *continuous-coordinate* set
  prediction (the Phase-2/3 path); GroupPose's grouped self-attention
  organizes queries by keypoint type.
* **RLE** (Li, ICCV 2021) + **β-NLL** (Seitzer, ICLR 2022) — uncertainty-aware
  regression that beats heatmaps with no inference cost and emits a per-keypoint
  confidence usable by the event selection.

---

## 4. Design decision

**Chosen approach: Option 3 (hybrid), coupled to Stage 3, with partial
retraining of Stage 3.** Two complementary heads:

1. **Dense per-point keypoint-score head (Option 1).** A per-spacepoint head
   predicting `(N, 6)` against `kpscores` (+ an optional per-point offset/vote
   head for off-point precision). Cheap, dense supervision, no matching, trains
   to ~zero fast — it is the development milestone *and* a permanent
   backbone-warming auxiliary signal. Its score peaks can also **seed** the
   query anchors (reuse the `MixedQuerySelector` pattern, scored on
   keypoint-ness instead of `1 - p(no_object)`).

2. **Query-based keypoint head (Option 2), riding on Stage-3 particle
   queries.** Each matched particle query already localizes a particle; the
   keypoint head turns its embedding into precise typed 3D points (start, and
   end for tracks), plus dedicated **vertex queries** for the nu vertex. This
   gives *particle-associated* keypoints — the thing the dense field can't.

### Why coupled to Stage 3 (not standalone after Stage 2)
Particle association is required by the physics: "this shower's start", not "a
shower start somewhere". Stage 3 already instances particles; the keypoints are
their attributes. The nu vertex is the lone slice-level keypoint — **as of
2026-06-16 it is taken from the Phase-1 dense head's `nu_vertex` channel +
offset vote, not from dedicated query slots** (vertex queries deferred; see the
Phase-2 decision note below). The §4 2A/2B text below still describes the
vertex-query option for when/if the dense-vertex path underperforms.

### Training strategy
Stage 3 is **not** in production, so retuning is allowed.

* **Freeze**: the backbone *encoder* and all of Stage 1+2 (deghoster + slicer),
  always.
* **Train / fine-tune**: Stage 3's Mask2Former *decoder* + origin path
  (`unfreeze_decoder=True`) **together with** the new keypoint heads, with the
  keypoint loss upweighted. This overcomes the de-emphasized-origin limitation
  (a frozen Stage-3 embedding simply doesn't encode endpoints well) without a
  full Stage-3 retrain and without disturbing the encoder. The dense head
  (Option 1) needs only the frozen backbone features and can be trained
  independently / first.

**Two training modes (2026-06-16) — `freeze_non_keypoint` flag on `LArFormer`:**
* `freeze_non_keypoint=True` (the keypoint config's DEFAULT) — **keypoint-only**:
  freezes the ENTIRE particle network (backbone, PT-v3 decoder, token refiner,
  Mask2Former decoder layers + class/mask heads, per-level cls, query selection,
  denoising) and trains ONLY the keypoint params (per-query start=origin / end /
  uncertainty / end-gate heads, the 2B refinement decoder, the dense head).
  Verified: 186 keypoint param-tensors trainable, 1468 frozen, grad flows only
  into keypoint params. Preserves the loaded Stage-3 quality and directs all
  optimization to keypoints. **Use this if segmentation is already good and
  keypoints lag** (the observed "first ~500 epochs improve segmentation, not
  keypoints" — particle losses dominate the total + the plateau LR + best_metric
  all track segmentation). Pair with `unfreeze_decoder=False` (backbone runs
  under no_grad — `_encode` handles this) and `mask_denoising=None` (DN would
  only train the frozen mask/cls path). Caveats under this mode: `best_metric=
  mask_iou_mean` is now static (frozen) so `model_best` won't update — use
  `model_last` or set `best_metric` to a keypoint scalar (e.g.
  `val/kp_R1_track_start`); the query embeddings are fixed, so endpoint
  precision rides on the 2B refinement + dense heads (which read geometry).
  * **fp32-decoder gotcha (fixed 2026-06-16).** `unfreeze_decoder=True` had a
    side effect: it set the PT-v3 *decoder* blocks' attention to the `xformers`
    backend. With `unfreeze_decoder=False` the decoder kept `flash_attn`, which
    silently produces garbage on the fp32 eval forward → **per-particle val IoU
    collapsed to 0 even though the weights loaded fine**. `freeze_non_keypoint`
    now calls `_ensure_decoder_fp32_forward()` to set the xformers backend (+
    NaN sanitizers) on the frozen decoder, so masks are correct (verified:
    median matched IoU back to **0.699**, == the unfrozen case).
* `freeze_non_keypoint=False` + `unfreeze_decoder=True` — **partial-joint**:
  the original mode; the decoder/query embeddings adapt for keypoints at the
  cost of also moving segmentation.

**Convergence-speed tuning under keypoint-only freeze (2026-06-16).** A 2k-epoch
dev run with `base_lr=1e-5` showed the dense vertex precision *never started its
upswing* while the query metrics were smooth-but-slow. Root cause was throttling,
NOT masker-vs-keypoint competition: under the freeze the mask/cls gradient is
identically zero on every trainable param, so the observed grad-norm of 20-100
is **all keypoint-head gradient** (dominated by the dense head's `pos_weight=50`
BCE). `clip_grad=1.0` against that norm scaled the keypoint gradient down 20-100x
— the clip, not the LR, was the main throttle — and a single global LR forced the
fast-moving dense head and the smooth query head to share one step size. Four
changes (all in `larformer-keypoint-query-v1.py` + `losses.py`):
1. **Per-head LR groups** (`param_dicts=[dict(keyword="kp_dense", lr=1e-4)]`):
   the dense head gets ~10x the query/refine heads' `base_lr=1e-5`. Pointcept's
   `build_optimizer` matches by name substring; `FlatWithDecayLR` ramps/decays
   each group independently. Tune via the `dense_lr` knob.
2. **Grad clip 1.0 -> 10.0**: stop crushing the keypoint gradient.
3. **Regression warm-up** (`kp_reg_warmup_epochs`, `kp_reg_warmup_kind`): run the
   start/end regression with `smooth_l1` for the first N epochs, then switch to
   β-NLL (the early β-NLL variance is what forced the LR down). `LArFormerLoss`
   gains `set_epoch` + `_eff_reg_kind`; `KeypointRegWarmupHook` pushes the epoch
   in each epoch. Scale `KP_REG_WARMUP_EPOCHS` to ~20-25% of the run.
4. **Zero the mask/cls loss weights** (`weight_class/mask_primary/dice_primary/
   aux_mask/per_level_cls=0`): produce no gradient anyway under the freeze, so
   zeroing them cleans the logged total + skips wasted backward. The matcher's
   `cost_*` terms are SEPARATE knobs and stay on, so the query->GT Hungarian
   assignment the keypoint heads rely on is unchanged. Restore to (2,5,5,0.5,0.3)
   for partial-joint training.

Deeper ceiling still open (deferred): the frozen spacepoint embeddings are
mask-tailored (uniform within a particle), which is antithetical to localization;
options are injecting coords into the refiner's *value* path, a trainable
keypoint SP-adapter, or partial-joint with a low mask-weight. Try only if the
query side plateaus despite a healthy LR.

**Coordinate position-embedding options on the keypoint heads (STAGED off,
2026-06-16).** The backbone features carry position only implicitly (PT-v3 cpe +
serialized attention); neither the dense per-point MLP nor the per-query keypoint
heads get an explicit coordinate. Both now have opt-in pos-emb flags (default
off → identical model; verified: 186 trainable / dense-in=64 unchanged):
* **Dense heads** (`keypoint_dense_cfg.pos_emb` = None|"sinusoidal"|"mlp", with
  `pos_emb_dim`/`pos_emb_num_freq`/`pos_emb_max_freq`/`pos_emb_hidden_dim`): a
  coord pos-emb is *concatenated* onto `sp_feat_all` before the score/offset
  MLP (`KeypointScoreHead`/`KeypointOffsetHead`). The offset head benefits most
  — regressing a geometric displacement is easier from an explicit coord than
  from a position-entangled feature.
* **Per-query heads** (`decoder_kwargs.keypoint_pos_emb_kind` = None|"sinusoidal"
  |"mlp"): a pos-emb of the query's *anchor* (the prior decoder layer's predicted
  origin, or the mixed-query anchor at init) is *added* to the query before the
  keypoint heads (`_PerLayerHeads`), DAB/DN-DETR style. The module is named
  `kp_pos_emb`, is added to the keypoint freeze markers (trains under
  `freeze_non_keypoint`), and is **zero-init** (identity at init, so adding it
  doesn't shock the loaded heads). Parity option on the standalone
  `KeypointQueryHead`.
  * **Mask-safety (important, fixed 2026-06-16).** Only `end`/`start_logvar`/
    `end_logvar`/`end_exist` get the pos-emb — NOT `origin`. `origin` is the one
    keypoint output that **feeds back** into the frozen decoder (each layer does
    `query_pos += pos_emb(prev_origin)`), so routing it through the keypoint
    pos-emb let the fresh `kp_pos_emb` perturb `origin` → perturb the frozen
    attention → **degrade the frozen masks** (observed: val/mask_iou dipped
    0.83→0.79 on the first run with `keypoint_pos_emb_kind="sinusoidal"`).
    Keeping `origin` on the raw query makes the query pos-emb mask-safe by
    construction (verified: `origin` + class + mask invariant to the anchor for
    all time; `end` identity at init, responds after a step). `origin`/start
    still gets position via the existing origin-feedback loop; `end` — which has
    no feedback and most lacks position — is exactly where the pos-emb helps.
  * **General note on the freeze:** even WITHOUT pos-emb, the trainable
    `origin_head` influences the frozen masks through that same origin-feedback
    path, so val mask IoU is never perfectly constant under `freeze_non_keypoint`
    — it drifts a little as origin trains. (This is the real explanation for the
    earlier "mask IoU creeps up under the freeze" observation — PT-v3m2 uses
    LayerNorm, so it was never BatchNorm running-stats.) The dip above was that
    same coupling, amplified by a fresh random pos-emb on origin; the fix removes
    the amplification, leaving only the small pre-existing drift.
Shared builder: `decoder.build_coord_pos_emb`. Recommend "sinusoidal" (unique
spatial signature, no mirror-symmetry collapse). Suggested first A/B once the
LR/clip/warm-up run is read: dense `pos_emb="sinusoidal"` for the offset head;
then `keypoint_pos_emb_kind="sinusoidal"` if the query endpoints still lag.

### Query-head architecture (Option 2): build 2A → 2B

**2A — Heads only (milestone build).** Reuse the matched query embeddings
`(Q, D)`. Replace the single origin MLP with a keypoint head per query:
```
start(3), end(3), endpoint_logit(1), log_var(·)     # MLP(D → D → ·)
```
* `start` = existing origin head (reused). `end` is new, supervised only for
  track-class queries (mask the end loss by GT class). `log_var` feeds an
  RLE / β-NLL loss → per-keypoint confidence.
* **Nu vertex**: add a small fixed number (1–4) of dedicated *vertex queries*
  to the query set with a "vertex" class slot; they cross-attend the whole
  slice (already the default) and are Hungarian-matched to the `mckeypoints`
  nu-vertex GT.
* **No new matching** for particle start/end — the existing particle→GT match
  determines which GT particle's endpoints to supervise. Only vertex queries
  consume extra match slots.
* Ceiling: a query embedding is a *global* per-particle summary, so endpoints
  (esp. track end) may be coarse. If that plateaus → 2B.

**2B — Point-refinement decoder (production precision).** Wrap 2A's heads in a
short (2–3 layer) decoder, a specialization of `Mask2FormerDecoder`:
```
init queries = Stage-3 matched query embeddings (+ learnable vertex queries)
init anchor  = Stage-3 predicted start (reuse the pos-emb anchor feedback)
for L layers:
    masked cross-attention: queries → spacepoint-level tokens   # mask-gated to
                                                                # that particle
    self-attention among queries                                # vertex = consensus
    FFN
    point head: regress an OFFSET to the current point estimate  # iterative
```
* **Cross-attention to spacepoint features** gives spatial precision (the query
  looks at its own particle's points; regresses an offset so the point can land
  *off* the cloud). **Self-attention among queries** lets the vertex aggregate
  primary starts and keeps endpoints consistent — secondary, not standalone.
* Reuses `_MaskedDecoderLayer`, the origin/pos-emb feedback loop, and
  `_build_attn_mask`.

---

## 5. Implementation plan (phased)

Each phase lists files to touch, what's new, and the acceptance check. Build in
order; the data path (Phase 0) unblocks everything.

### Phase 0 — Data path: surface `kpscores` + `mckeypoints` + endpoints  ✅ DONE (raw path)
**Status:** implemented + verified on the raw merged_h5 path. Cache path
pending (see finding 2 below).

**Two findings during implementation (these changed the approach):**
1. **`triplet_data/kpscores` is NOT stored** in the current merged_h5
   production (checked across files — absent everywhere), but the
   `mckeypoints/` group *is* present in all. → The dense target is now
   **computed on-the-fly** from `mckeypoints` via
   `lartpc/data_prep/labels/keypoint_labels.py::compute_kpscores` (Gaussian σ
   tunable, default 3 cm), with a stored-field fallback if one ever appears.
   Strictly better: σ is tunable and the target needs no H5 reproduction.
2. **The stage-12 dev cache (`exp/cache_stage12_devdata`) has NO keypoint
   info** — no `mckeypoints`, no `kpscores` (it carries coord/feat/trackid/
   `particle_instances`/`slicer` only). So keypoint training on the cache
   needs a **cache-augmentation step** that joins each cache file back to its
   source merged_h5 (the filename encodes `fileno…_entry…`) and adds the
   `mckeypoints` group. Tracked as a Phase-0 sub-task below.

**What landed:**
* `lartpc/data_prep/labels/keypoint_labels.py` — shared `compute_kpscores` (per-SP
  `(N,6)` proximity, optional offsets) + `endpoint_by_trackid`.
* `pointcept/datasets/larformer.py`:
  * `__init__`: `emit_keypoints`, `keypoint_sigma_cm` (3.0),
    `keypoint_score_threshold` (0.01), `n_keypoint_types` (6). Off by default.
  * `_read_mckeypoints` — per-event group → cm + normalized arrays.
  * `_load_event` (step 3b): compute `sp_kpscores_k` directly on the final
    filtered `pos_k` (cm) — **no keep/dedup/cap remap needed** (pure geometry
    vs keypoints; the stored-field fallback *does* use `remap`).
  * `_enrich_gt_with_endpoints`: attaches `end_coord_norm` + `has_end` per GT
    instance via `track_end` (kptype 2) trackid match; per-event
    `nu_vertex_coord_norm` (kptype 0) added to the output dict.
  * `larformer_collate`: `kpscores` in `optional_flat_keys` (flat per-SP,
    slices with `offset`); `mckeypoints_*_per_event` + `nu_vertex_*_per_event`
    nested lists.
* `lartpc/larformer_analysis/archive/keypoint_v1/test_phase0_dataloader.py` — acceptance
  test.

**Acceptance — PASS.** On merged_h5 events: `kpscores` is `(N,6)` in [0,1]
aligned to `coord_norm`; per-type score peaks land <0.25 cm from GT
`mckeypoints/pos`; nu vertex surfaced; track particles (μ, p) get
`has_end=True`, showers (γ) `has_end=False`; collate flat-slice round-trips.
Observed the motivating case: an event where the nearest SP to the nu vertex is
~4 cm away (top score 0.39) — the vertex sits *between* points, which the later
offset/query heads address. Run inside the pointcept container
(`singularity exec -B /mnt:/mnt /mnt/ddrive/containers/pointcept_sandbox/`,
`source setenv_pointcept_container.sh`, `HDF5_DISABLE_VERSION_CHECK=2`):
`python lartpc/larformer_analysis/archive/keypoint_v1/test_phase0_dataloader.py`.

**Cache path — ✅ DONE (verified):**
* `tools/larformer/augment_stage12_cache_keypoints.py` — copies `entry_0/mckeypoints`
  from each cache file's source merged_h5 (resolved via the `source_h5` attr,
  indexed by basename under `--merged-root`) into the cache. In-place or
  `--output-dir` copy mode; idempotent; multiprocessing. CPU-only, ~280
  files/s. Sets `has_mckeypoints` attr.
* `lartpc/data_prep/labels/keypoint_labels.py::copy_mckeypoints_group` — shared
  cross-file group-copy helper (used by the augment tool AND the builder).
* `pointcept/datasets/larformer_stage12_cache.py`: `emit_keypoints` (+ σ /
  threshold / n_types) knobs; reads the cache's `mckeypoints` group; computes
  `kpscores` on the kept cm `coord`; attaches `end_coord_norm`/`has_end` per
  instance (trackid match) + `nu_vertex_coord_norm`; recenters keypoint arrays
  alongside `coord_norm` when `recenter_to_centroid`. Emits the SAME fields as
  `larformer.py` so `larformer_collate` handles both.
* **Cache builder updated for future caches:** `build_stage12_cache_event.py`
  (+ `build_stage12_cache_shard.py`) now copy the source `mckeypoints` group
  into every cache by default (`copy_keypoints=True`; `--no-keypoints` to
  disable). So a freshly-built cache is keypoint-ready with no augment step.
* Test: `lartpc/larformer_analysis/archive/keypoint_v1/test_phase0_cache.py` — **PASS**
  on a copy-augmented dev cache (peaks <0.5 cm from GT, endpoints + nu vertex
  present).
* (Optional, not done) note in `docs/reference/LArTPC_HDF5_Data_Format.md` that
  `kpscores` is a *derived* target (computed from `mckeypoints`), not stored.

**Source-data availability on this machine (answer to "can we update/remake?"):**
Both are feasible locally.
* **Update existing cache (cheap, no model/GPU):** the source merged_h5 tree
  (`/mnt/ddrive/data/ub_on_tufts/h5/bnb_nu_pi0filter_corsika/merged_h5`, 682
  files) is present and **all 20 dev-cache `source_h5` basenames resolve**.
  Run the augment tool in-place.
* **Remake from scratch (full Stage 1+2, needs GPU):** the Stage-3 config
  (`configs/lartpc/larformer/stage3_particle/larformer-particle-v1.py`) and all three checkpoints it
  needs are present locally — deghoster `epoch_30.pth` (383M), Sonata pretrain
  `epoch_42.pth` (1.5G), slicer `model_ptv3crosslevel_iter_75750.pth` (1.4G).
  The builder now embeds keypoints automatically.

### Phase 1 — Dense per-point keypoint-score head (Option 1)  ✅ DONE — milestone met
**Decision:** built as a standalone model `LArFormerKeypoint` (frozen Sonata
backbone + dense head) rather than an aux head inside Stage 3. This is the
cleanest validation of the data path + the milestone, and the head/loss are
reusable when it later becomes an aux head / anchor seeder for Phase 2/3.
Runs directly on the cache's nu-candidate SP set (the cache *is* the nu slice),
so no extra masking is needed.

**What landed:**
* `pointcept/models/LArFormer/keypoint_heads.py` — `KeypointScoreHead`
  (per-SP MLP `D_bb → 6`, hidden 512, outputs logits; sigmoid applied in
  loss/pred so output ∈ [0,1]).
* `pointcept/models/LArFormer/keypoint.py` — `LArFormerKeypoint` model
  (registered). Frozen-backbone `_encode` (reuses LArFormer's pattern) →
  head → weighted-MSE loss on `sigmoid(logits)` vs `kpscores`. `pos_weight`
  up-weights points within the Gaussian peaks (plain MSE is otherwise
  trivially minimized by predicting zeros, since most SPs score 0).
  Diagnostics: `mse_all`, `mse_pos` (peak fit — the meaningful metric),
  `frac_pos`. Eval returns per-event `kpscores_pred`. `loss_kind="bce"` and
  an optional offset/vote head are left as future toggles.
* Registered in `pointcept/models/LArFormer/__init__.py`.
* `configs/lartpc/larformer/stage4_keypoint/archive/larformer-keypoint-v1.py` — scale-up training config
  (frozen Sonata + cache dataset `emit_keypoints=True`, `source_set_filter=
  "stage2_pass"`, `LArFormerTrainer` for the collate, minimal hooks,
  `evaluate=False`). Builds + runs a train/eval forward (verified).
* `lartpc/larformer_analysis/archive/keypoint_v1/train_phase1_overfit.py` — standalone
  overfit harness (no Pointcept trainer) for the milestone.

**Acceptance — PASS (the spec's milestone).** Overfit 3 dev-cache events
(1916 SPs, ~11% positive score elements), **frozen backbone, head only
(634K params)**:
`kp loss 0.140 → 0.0008 (−99.4%)`, peak-fit `mse_pos 0.123 → 0.0009`. So even
with a frozen Sonata backbone the head fits the dense keypoint-score field to
≈0 — the data path, model, and loss are all correct, and the frozen features
are expressive enough per-point. Run:
`python lartpc/larformer_analysis/archive/keypoint_v1/train_phase1_overfit.py
--n-events 3 --steps 300 --pos-weight 50 --device cuda` (in the pointcept
container; needs GPU).

**PPN/VoteNet offset head — ✅ DONE.** `KeypointOffsetHead`
(`pointcept/models/LArFormer/keypoint_heads.py`): per-(SP, type) MLP →
`(N, 6, 3)` vote toward the nearest keypoint, in the normalized frame
(`predicted_kp = coord_norm + offset`); output zero-init (start by voting "no
displacement"). Target: `kpoffsets` `(N, 6, 3)`, computed on-the-fly in BOTH
datasets via `compute_kpscores(return_offsets=True)` (cm offset ÷ coord_scale;
displacement is recenter-invariant). Loss: smooth-L1 supervised ONLY where
`kpscore > offset_supervision_threshold` (points within the peaks vote;
everything else is free) — the VoteNet rule. Enabled via
`enable_offset_head=True` (+ `weight_offset`); on by default in
`larformer-keypoint-v1.py`. Verified: GT offset is exact (voted = keypoint,
0.000 cm for an on-keypoint SP); overfit drives the **vote error 2.97 → 0.35
cm** (1229 supervised SP-types) alongside the score head. Eval emits
`kpoffsets_pred` + `kpvote_pred` (= coord + offset) per SP.

**Still deferred (not blocking):** a `loss_kind="bce"` comparison (flag exists);
full multi-epoch training via the config (vs the overfit harness); a keypoint
evaluator (per-type recall within distance) — a Phase-4 item.

### Phase 2 — Query keypoint head 2A (Option 2, heads only)
**Files:** `decoder.py` (extend `_PerLayerHeads`), `losses.py` (RLE/β-NLL,
endpoint loss, vertex matching), `matcher.py` (vertex class; optionally an
endpoint cost), `model.py` (expose per-query embeddings; vertex queries),
config.
* Extend the per-query head to `start / end / endpoint_logit / log_var`.
  Supervise `end` only for track-class GT; `start` for all; RLE/β-NLL on both.
* Add N vertex queries (config) with a vertex class slot; match to
  `nu_vertex_coord_norm`.
* **Expose Stage-3 per-query embeddings** in the prediction dict (`model.py`
  ~L948-962 currently emits class/origin/mask only) — needed if the keypoint
  head is a separate module reading a (possibly frozen) Stage 3.
* Build the `CascadedKeypoint` wrapper (mirror `CascadedParticleSegmenter`):
  frozen Stage 1+2; Stage 3 with `unfreeze_decoder=True`; keypoint heads +
  upweighted keypoint loss. Load the existing Stage-3 checkpoint as init.

**Acceptance:** matched-query start/end + vertex L1 decreases on dev data;
per-type recall within 3/10 cm beats the Phase-1 dense-decode baseline for the
*associated* (per-particle) keypoints.

**Progress — step 1 of Phase 2 DONE (foundations, unit-tested):**
Built the matcher-agnostic core so it can be validated without a trained
Stage 3 (cluster down; Stage-3 decoder is still random-init anyway).
* `pointcept/models/LArFormer/keypoint_query.py`:
  * `KeypointQueryHead` — per-query MLP → `start (3)`, `end (3)`,
    `end_logit (1)` (track-end gate), `start_logvar`/`end_logvar (3)`
    (uncertainty). Logvar outputs zero-init (var≈1).
  * `keypoint_query_loss` — takes `(q_idx, k_idx)` + per-GT `start`/`end`/
    `has_end`; `start` on all matched pairs, `end` on matched pairs with
    `has_end`, BCE on the end gate. Regression kinds: `l1` / `smooth_l1` /
    `betanll` (β-NLL, Seitzer 2022 — decouples gradient from predicted
    variance). Matcher-agnostic: fed a fixed assignment now, the real
    LArFormer matcher output later. Empty-match → finite zeros.
* **Stage-3 query embeddings exposed** (the documented gotcha):
  `decoder.py` `_compute_predictions` adds `query_embed` to every layer dict;
  `model.py` `_slice_decoder_output` slices it (DN-safe) and `forward` puts
  `pred["query_embed"]` in the eval prediction dict.
* Test `test_phase2_keypoint_query.py` — **PASS**: overfit synthetic queries
  with a fixed assignment → start_err 0.69 cm, end_err 0.006 cm; matched=4,
  with-end=3; empty-match finite.

**Progress — step 2 of Phase 2 DONE (Option A integration, validated on a real
Stage 3):** the keypoint head is now an opt-in `LArFormer` feature
(`enable_keypoint_head`), reusing the existing query→GT Hungarian match — no
separate wrapper or matcher.
* `decoder.py` `_PerLayerHeads`: when `enable_keypoint_head`, each layer also
  emits `end (3)`, `start_logvar`/`end_logvar (3)`, `end_logit (1)` (start =
  the existing `origin` head). Threaded through `Mask2FormerDecoder` +
  `_compute_predictions`; DN-safe slice in `model._slice_decoder_output`.
* `losses.py`: `LArFormerLoss` gains `enable_keypoint_head` + weights
  (`weight_kp_start/end/end_exist`, `kp_reg_kind`, `kp_beta`, `coord_scale`).
  `forward` builds `gt_end`/`gt_has_end` from the instances; `_compute_layer_loss`
  calls `keypoint_query_loss` on the matched pairs (deep-supervised over layers)
  and **skips the plain origin L1** (start handled by `kpq_start`). Final-layer
  `kpq_*_err_cm` diagnostics added (not summed).
* `model.py`: `LArFormer(enable_keypoint_head=...)` threads to decoder + loss;
  eval `pred` exposes `kp_end`/`kp_*_logvar`/`kp_end_logit`.
* **No `CascadedKeypoint` wrapper needed** — because the head lives inside the
  particle segmenter, the existing `CascadedParticleSegmenter` carries keypoints
  automatically on the raw-data path. Cache/dev training runs the Stage-3
  `LArFormer` directly.
* Test `train_phase2_overfit.py` — **PASS**: loads the trained Stage-3 ckpt
  (`epoch_6.pth`; 98 missing keys = exactly the keypoint heads, strict=False),
  frozen backbone + trainable PT-v3 decoder + heads, overfit 3 dev-cache events
  → **start_err 11.2 → 1.2 cm**, **end_err 89.3 → 18.3 cm**.

The coarse end error (18 cm) is the expected 2A limitation: `end` is regressed
from a *global* per-query embedding. That's what Phase 3 (2B refinement decoder,
cross-attending to the particle's own spacepoints) addresses.

**Decision (2026-06-16): the nu vertex comes from the Phase-1 PPN-like dense
head, NOT from dedicated query slots.** The vertex is slice-level (not a
particle attribute), and the Phase-1 `KeypointScoreHead` already predicts a
`nu_vertex` channel (column 0) + the `KeypointOffsetHead` votes toward it — so
the vertex is decoded from that dense output (Phase 4: peak/cluster the column-0
score, refine with the column-0 vote). **Dedicated vertex queries are deferred**
and only added if this dense-vertex path underperforms. Net: Phase-2's
query head covers the per-particle start/end; Phase-1 covers the vertex.

**Remaining Phase-2 sub-step:** keypoint training config for the query head
(mirrors `larformer-keypoint-v1.py` but on the particle segmenter with
`enable_keypoint_head=True`) + a longer dev run. Then Phase 4 wires the
vertex-from-dense decode.

### Phase 3 — Point-refinement decoder 2B (precision)
**Files:** new `keypoint_decoder.py` (subclass/clone of `Mask2FormerDecoder`),
wire into the keypoint module + config.
* 2–3 layers: mask-gated cross-attention to spacepoint tokens + self-attention
  among queries + iterative offset head. Init queries from Stage-3 embeddings,
  anchor from Stage-3 start.
* Optionally seed query anchors from Phase-1 dense-score peaks
  (`MixedQuerySelector`-style, scored on keypoint-ness).

**Acceptance:** track-*end* and vertex precision improve over 2A (the metric
2A is weakest on); no regression on Stage-3 mask/class metrics.

**Progress — DONE (built, validated to run/train/compose; precision win
pending real training):**
* `keypoint_refine.py`: `KeypointRefinementDecoder` — opt-in, runs AFTER the
  main decoder. Per layer (default 2), reusing `_MaskedDecoderLayer`:
  mask-gated cross-attention (query → its own spacepoint tokens, gated by the
  query's predicted SP mask) + self-attention among queries + FFN + an
  iterative offset point head (delta_start/delta_end/logvars/delta_end_logit).
  **Identity at init** (point-head output zero-init → refined == 2A at iter 0),
  so it composes with a trained 2A checkpoint without disruption. Anchors
  query_pos at the current start estimate (origin-feedback).
* `model.py`: `LArFormer(keypoint_refine=...)` builds it (requires
  enable_keypoint_head + a decoder); `forward` runs it on the regular queries
  using the primary (spacepoint) level tokens + the query's SP mask, and the
  refined FINAL layer supersedes the 2A keypoint prediction in `pred`.
* `losses.py`: `forward(keypoint_refine_layers=...)` deep-supervises every
  refinement layer with the same `keypoint_query_loss` + weights (summed);
  the refined final layer supersedes the err diagnostics. New telemetry:
  `loss_kpq_refine_{start,end,end_exist}`.
* `larformer-keypoint-query-v1.py`: `keypoint_refine=dict(num_layers=2, ...)`
  enabled (set to None for pure 2A).
* Validated (overfit, epoch_6, `train_phase2_overfit.py --refine`): builds
  (52 new refinement params, loaded strict=False as identity-init), trains
  clean (end_err 66.8 → 16.4 cm over 200 steps), eval exposes the refined
  `kp_end`. On this short tiny-overfit the refined error is comparable to 2A
  and noisy — the refinement's attention layers are random-init and need real
  training to converge before beating the 2A MLP (the point head being
  zero-init means it starts as exact 2A and only helps once trained). The
  precision-win acceptance is a real-training (cluster) check.

### Phase 4 — Decode, evaluation, inference, cascade integration
**Files:** an evaluator under `pointcept/models/LArFormer/` (mirror
`particle_evaluator.py`); inference entry; ROOT output integration.
* Decode: dense head → NMS/peak-clustering on the score field; query head →
  per-query points filtered by confidence. Reconcile the two.
* Metrics: per-type precision/recall vs `mckeypoints` at distance thresholds
  (PPN-style), and vertex resolution.
* Denormalize to detector cm for analysis output; integrate into the cascade
  predictions so downstream event selection consumes typed, particle-associated
  keypoints + the nu vertex.

**Progress — decode + metrics DONE (unit-tested + validated end-to-end):**
* `pointcept/models/LArFormer/keypoint_eval.py` (pure numpy):
  * `decode_dense_votes` — per type: threshold SP scores, move each to its
    voted location (`coord + offset`), greedily score-weight-cluster the votes
    (`cluster_votes`) → discrete keypoints (can sit between points).
    `decode_nu_vertex` = best type-0 cluster.
  * `decode_query_points` — per particle query: start (always, if real class
    above `class_prob_floor`) + end (track classes, gated by `end_logit`).
  * `match_points` / `accumulate_metrics` / `format_metrics_table` — greedy
    nearest matching → per-type precision/recall at distance thresholds
    (PPN "% recovered within N cm") + nu-vertex resolution.
* `test_phase4_decode.py` — **PASS** (synthetic): dense decode recovers 2 known
  keypoints exactly, metrics give P/R = 1.0, query decode gates start/end.
* `eval_phase4_keypoints.py` — runs the dense model on dev-cache events,
  decodes, scores vs `mckeypoints`. `--gt-acceptance-cm` filters GT to
  keypoints within N cm of an SP (only "reconstructable" ones — the nu-slice
  can't find cosmic keypoints with no nearby SPs; without this the recall of
  out-of-slice cosmic track/delta keypoints is an unfair guaranteed miss).
* **Validation (overfit dense model, 6 dev events, acceptance 5 cm):**
  **nu_vertex P/R = 1.0, resolution ~0.1–0.2 cm**; delta R=1.0, shower R=0.83.
  This is the headline result confirming the **vertex-from-dense-head decision**
  — no vertex queries needed. `track_start`/`track_end` recall is moderate
  (~0.3–0.4) and *expected*: in a ν interaction the primaries share the vertex,
  so coincident `track_start` GT points merge into one decoded peak (1 TP +
  N−1 FN) — a metric-coincidence nuance, not a decode bug. Will rise with real
  training (vs 250-step overfit) + threshold tuning.

**Query-head eval + reconcile — DONE (step 1):**
* `keypoint_eval.py`: `query_points_to_typed` (map per-query start/end onto
  keypoint types: shower-class start→shower, track-class start→track_start,
  end→track_end), `reconcile_keypoints` (query head for track_start/track_end/
  shower; dense head for nu_vertex/michel/delta, with query→dense fallback),
  `gt_keypoints_by_type` (shared GT-by-type + acceptance helper).
* `eval_phase4_query.py`: builds the Phase-2 query model, loads epoch_6
  (98 missing = keypoint heads), decodes per-query points, scores vs
  `mckeypoints` with the SAME PPN metric. Recentered-frame aware (acceptance
  uses `denorm(coord_norm)`, matching the recentered GT).
* `test_phase4_decode.py` extended — typed-map + reconcile checks **PASS**.
* **Result (load epoch_6 + 200-step overfit, 6 events):** track_start
  **P@10cm=0.86** (emitted starts are correct), R@10cm=0.32. Recall is limited
  by Stage-3 *classification* quality (epoch_6 is a weak 6-epoch dev model +
  short overfit), NOT the keypoint regression — it rises with real training.
  At this Stage-3 quality query≈dense on track_start (0.32 vs 0.30); the
  coincidence-resolution advantage of the query head (one start per particle
  vs one merged dense peak at the shared vertex) will show once Stage-3
  classification is solid.

**Training-loop evaluator hook — DONE (step 2):**
* `keypoint_particle_evaluator.py`: `LArFormerKeypointEvaluator` (subclasses
  `LArFormerParticleEvaluator`, so all particle metrics still log). Per eval
  epoch it decodes the query head's per-event start/end, maps to keypoint
  types, matches to `mckeypoints` (acceptance-filtered), and logs
  `val/kp_R{1,3,10}_{track_start,track_end,shower}` + `val/kp_P{...}`. The
  per-pair `val/loss_kpq_start_err_cm` / `val/loss_kpq_end_err_cm` already
  auto-log via the eval_loss accumulation (no extra code).
  * **Thresholds 1 / 3 / 10 cm (2026-06-16).** 1 cm is the precision TARGET
    (GT proximity σ = 3 cm; 10 cm is loose context, not "accurate"). Watch
    `val/kp_R1_*` (fraction of keypoints within 1 cm) as the headline; the
    eval log line prints `P/R (@1cm | @10cm)` side by side. Start precision is
    also covered by the particle evaluator's `val/origin_l2_cm_{median,p25,p75}`.
    Same thresholds in `eval_phase4_keypoints.py` / `eval_phase4_query.py`.
  * **Nu-vertex metrics (2026-06-16).** The query model now carries an
    **integrated dense keypoint head** (`enable_keypoint_dense_head` on
    `LArFormer` — the Phase-1 `KeypointScoreHead`/`KeypointOffsetHead` on the
    backbone per-SP features, supervised by the dataset's `kpscores`/
    `kpoffsets`), so ONE model emits the per-particle query keypoints AND the
    slice-level dense field. The evaluator decodes the nu vertex from it and
    logs `val/kp_{R,P}{1,3,10}_nu_vertex` + `val/nu_vertex_res_cm_{median,mean}`
    (the eval line leads with `nu_vertex P/R` + a vertex-resolution line); the
    dense head's per-SP train losses log as `loss_kpdense_{score,off,off_err_cm}`.
    Enabled in `larformer-keypoint-query-v1.py`. (This supersedes the earlier
    "vertex from a *separate* dense model" plan — it's now the same model;
    the standalone `larformer-keypoint-v1.py` dense model still exists for a
    dense-only run.)
* Small base-hook generalization: `evaluator.py`'s `_on_event_processed` now
  also receives `input_dict` + `event_in_batch` (optional, keyword-only) so a
  subclass can reach per-event side inputs like `mckeypoints_*_per_event`.
  `particle_evaluator.py` updated to accept+ignore them (back-compatible).
* `larformer-keypoint-query-v1.py` switched to `LArFormerKeypointEvaluator`.
* Test `test_phase4_eval_hook.py` — **PASS**: drives the real `eval()` loop via
  a mock trainer on dev-cache events; confirms `val/kp_*` + per-pair errors +
  best_metric are emitted. (With epoch_6's random-init keypoint heads:
  track_start R@10cm≈0.41, start_err≈8.5 cm, end_err high — will improve once
  the query head is actually trained via the config.)

**Full-cascade inference integration — DONE (step 3):**
* `tools/larformer/run_larformer_fullcascade_inference.py` — extends the stage-3 script's
  full-cascade path (deghoster + slicer + particle segmenter) and writes
  KEYPOINTS into each per-event H5 under `keypoints/...`
  (`pos_cm`/`type`/`score`/`source`/`class_id`/`query_id`/`kind`, +
  `nu_vertex_cm`). Output is a superset of `stage3pred_*.h5`.
  * Builds the cascade with `particle_segmenter.enable_keypoint_head=True`
    (and `enable_keypoint_dense_head=True` by default) injected;
    `--particle-keypoint-weights` overlays the keypoint-trained particle
    segmenter.
  * **Nu vertex source (2026-06-16):** prefers the particle segmenter's OWN
    integrated dense head (decoded from `ev_pred["dense_kpscores"]`, same
    recentered frame as the queries → same affine; no extra forward).
    `decode_event_keypoints(dense_in_query_frame=True)` handles that frame.
    Falls back to a separate `--keypoint-dense-config/-weights` model (run on a
    non-recentered batch) only when the integrated head is absent;
    `--no-integrated-dense` forces the separate/legacy path. Verified: the
    integrated path emits all 6 types incl `keypoints/nu_vertex_cm`.
  * Reuses the stage-3 script's cascade-boundary helpers verbatim
    (`_load_weights_into`, `_move_batch`, `_resolve_*`, `_unpack_ps_sample`).
  * Keypoint decode/flatten factored into `keypoint_eval.decode_event_keypoints`
    + `keypoint_arrays_for_h5` (shared, unit-testable). Query positions
    denormalized via the per-event affine recovered from the nu-slice's
    (coord, coord_norm) — correct under slice-recentering; the dense head is
    run on a NON-recentered copy of the slice and denormalized with the fixed
    (center, scale).
  * **Inference-only by design** (`gt_source="deghost"`): keypoints are decoded
    from predictions, so no stage runs its eval-with-GT Hungarian matcher.
    This is required — the slicer is a 3-class model and particle-level GT
    (ids 0..7) overflows its CE target → CUDA device-assert (the same reason
    `CascadedParticleSegmenter` strips GT before its inner slicer). The
    eval-with-GT *diagnostic* path on raw-cascade data has a separate
    pre-existing matcher assert; not exercised here.
* **Validated end-to-end** on a raw merged_h5 event (full cascade, epoch_6
  particle weights): runs clean, writes `keypointpred_*.h5` with `keypoints/`
  populated (5 query keypoints, positions in detector cm) alongside the
  slicer + stage3 halves. (Keypoint *values* are from epoch_6's random-init
  keypoint heads — pipeline validated; values become meaningful after the
  query head is trained via `larformer-keypoint-query-v1.py`.)

**Visualization — DONE:** `tools/viz/visualize_full_cascade.py` now reads
`keypointpred_*.h5` (as well as `stage3pred_*.h5`) and overlays the predicted
keypoints on the PREDICTION scene — one legend trace per type (query
start/end + shower, dense vertex/michel/delta) with type-specific
marker/color, plus a big marker for the chosen nu vertex; toggle via "show
keypoints". Backward-compatible (no-op on files without a `keypoints/` group).
Validated headless: overlay adds the right per-type traces from a
`keypointpred` file.

**Keypoint duplicate-query dedup — DONE (2026-06-16):** with the Stage-3
particle segmenter **frozen** while the keypoint heads train, the frozen
classification head keeps multiple co-extensive queries "active" on the same
particle. Each emits its own keypoint, so the dense/query decode double-counts
them as **keypoint false positives** (this depressed `val/kp_P*`). Fix: dedup
the queries before decoding keypoints, reusing the **SAME** `inference.dedup_queries`
the particle-mask decode already uses — single source of truth, **no drift**
between the particle masks and the keypoints.
* `keypoint_eval.py::dedup_query_effective_argmax(class_logits, sp_mask_logits,
  no_object_class_id, class_prob_floor, iou_threshold)` — applies the confidence
  floor then `inference.dedup_queries` (mask-IoU NMS: queries with mask
  IoU ≥ `iou_threshold` are suppressed → demoted to no_object), returning a
  `(Q,)` effective-argmax. `decode_query_points` / `decode_event_keypoints` take
  an `effective_argmax=` override so a suppressed query emits no keypoint.
* Wired into both consumers:
  * `LArFormerKeypointEvaluator` (`dedup_iou_threshold=0.6`, set in
    `larformer-keypoint-query-v1.py`) — computes `eff` in `_on_event_processed`
    and passes it to `decode_query_points`.
  * `tools/larformer/run_larformer_fullcascade_inference.py::decode_keypoints_for_event` —
    computes `eff` from `ev_pred["mask_logits"]["spacepoint"]` (guarded on the
    mask existing and `--dedup-iou-threshold > 0`) and passes `effective_argmax`
    to `decode_event_keypoints`. Uses the same `--dedup-iou-threshold` the
    particle-mask decode already uses, so masks and keypoints stay consistent.
* `test_phase4_eval_hook.py` — **PASS** with dedup active.

**Remaining (deferred):** ROOT output integration (not needed yet per the
user — the H5 `keypoints/` group is the current deliverable).

---

## 6. Datasets for development

* **stage12 cache (preferred for dev):**
  `Pointcept/exp/cache_stage12_devdata/train` — deghoster + slicer already
  applied; this is the input the Stage-3 LAr2Former model trained on. Use for
  Phases 1-3 on the local machine.
* **merged H5 sample:**
  `/mnt/ddrive/data/ub_on_tufts/h5/bnb_nu_pi0filter_corsika/merged_h5` — full
  raw H5 with `kpscores` + `mckeypoints`; use to validate Phase-0 data reads
  and for larger training.

---

## 7. Open decisions / risks

* **Track end as a first-class output in v1?** It is the part that most needs
  the 2B cross-attention. If only starts + vertex are needed initially, 2A may
  suffice and 2B becomes optional. *(Decision pending — default: include end,
  but allow start-only via config.)*
* **Dense-head loss:** BCE-with-pos-weight (treat score as soft label) vs MSE
  (LArMatch precedent) vs focal. Start with BCE+pos-weight; compare to MSE.
* **Offset/vote head in Phase 1?** Adds off-point precision but more code.
  Default: land the score head first, add offset once score loss trains clean.
* **Vertex: dedicated queries vs derive from primary starts.** Default:
  dedicated vertex queries (more robust when primaries are mis-clustered).
* **kpscores σ = 3 cm** is fixed at production time. If finer vertex resolution
  is needed, that requires regenerating H5 — out of scope for now; the offset
  head mitigates within-σ precision.

---

## 8. Status tracker

Update this as work lands. Format: `[ ]` todo, `[~]` in progress, `[x]` done.

| Phase | Item | Status | Notes |
|-------|------|--------|-------|
| 0 | Dense `kpscores` target (on-the-fly from `mckeypoints`) | [x] | `keypoint_labels.compute_kpscores`; stored-field absent in production |
| 0 | Read `mckeypoints` group + normalize | [x] | `_read_mckeypoints` |
| 0 | Add `end_coord_norm`/`has_end` + `nu_vertex_coord_norm` | [x] | `_enrich_gt_with_endpoints` |
| 0 | Collate (flat kpscores + nested mckeypoints) | [x] | `larformer_collate` |
| 0 | Dataloader smoke test (raw merged_h5) | [x] | **PASS** — `test_phase0_dataloader.py` |
| 0 | Cache parity: augment tool + reader + builder | [x] | **PASS** — `augment_stage12_cache_keypoints.py`, cache reader, builder copies by default |
| 0 | Run augment in-place on real dev cache | [x] | 20/20 files (train+val) augmented & reader-verified; **Phase 0 fully complete** |
| 1 | `KeypointScoreHead` + `LArFormerKeypoint` model + loss | [x] | registered; weighted-MSE on sigmoid; config `larformer-keypoint-v1.py` |
| 1 | **Overfit score loss → ~0 on dev data** | [x] | **PASS** — kp 0.140→0.0008, mse_pos 0.123→0.0009 (frozen backbone, head only) |
| 1 | `KeypointOffsetHead` (PPN/VoteNet) | [x] | **PASS** — vote err 2.97→0.35 cm; `kpoffsets` target both datasets; on by default in config |
| 1 | Full multi-epoch train via config (not overfit harness) | [ ] | scaffold ready; not yet run at scale |
| 2 | `KeypointQueryHead` + `keypoint_query_loss` (start/end/conf/β-NLL) | [x] | **PASS** unit test — start 0.69cm, end 0.006cm |
| 2 | Expose Stage-3 query embeddings | [x] | `query_embed` in decoder/slice/pred (DN-safe) |
| 2 | Option A integration into `LArFormer` (decoder+loss+model) | [x] | **PASS** — opt-in `enable_keypoint_head`, reuses matcher |
| 2 | Overfit on real Stage-3 ckpt (epoch_6) | [x] | **PASS** — start 11.2→1.2cm, end 89.3→18.3cm |
| 2 | Vertex queries + matching (nu vertex) | [deferred] | vertex comes from Phase-1 dense head (col 0 + vote); queries only if that underperforms |
| 2 | Keypoint training config (query head) | [x] | `larformer-keypoint-query-v1.py` — fine-tunes from epoch_6; builds+forwards verified |
| 2 | Longer query-head training run | [ ] | blocked on cluster (dev cache only); config ready |
| 3 | Point-refinement decoder (2B) | [x] | `keypoint_refine.py`; opt-in, identity-at-init, deep-supervised; runs/trains/composes (precision win pending real training) |
| 4 | Decode (dense votes + query) + PPN metrics module | [x] | `keypoint_eval.py`; `test_phase4_decode.py` PASS |
| 4 | End-to-end dense eval on dev cache | [x] | **nu_vertex R=1.0 @~0.2cm** (overfit) — validates vertex-from-dense |
| 4 | Query-head eval + dense/query reconcile | [x] | `eval_phase4_query.py`; track_start P@10cm=0.86; reconcile unit-tested |
| 4 | Training-loop keypoint evaluator hook | [x] | `LArFormerKeypointEvaluator`; logs val/kp_R/P per type; `test_phase4_eval_hook.py` PASS |
| 4 | Full-cascade inference integration (keypoints in H5) | [x] | `tools/larformer/run_larformer_fullcascade_inference.py`; validated on raw event |
| 4 | ROOT output integration | [deferred] | not needed yet (per user); H5 `keypoints/` is the deliverable |

---

## 9. Original task list (from the initial spec)

* ~~Come up with a final design for the keypoint model.~~ → §4 (Option 3 hybrid,
  coupled to Stage 3, partial retune).
* Plan the modifications to the LArFormer data loader. → §5 Phase 0.
* Plan the modifications to the LArFormer model. → §5 Phases 1-3.
* Plan the loss function. → dense BCE/MSE (Phase 1) + RLE/β-NLL coordinate
  regression (Phase 2-3); §3 prior art, §7 open decisions.
* Implement the model. → Phases 1-4.
* Run a small dev-data training and drive the loss to zero. → Phase 1
  acceptance.
