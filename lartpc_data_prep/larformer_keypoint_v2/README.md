# LArFormer Keypoint — Attempt 2 (parallel encoder–decoder + per-particle query)

Status: **PLAN (2026-06-17)** — not yet implemented. Forks from attempt 1
(`../larformer_keypoint/`, checked in). Attempt 1 may be deprecated.

## 1. Why a redesign

Attempt 1 appended keypoint heads onto the **frozen** Stage-3 particle masker
(query embeddings + decoder shared with masking). That shared representation is
mask-tailored (uniform within a particle), which is antithetical to localization
— it never gave enough expressivity, and metrics plateaued well short of the
1 cm target across LR/clip/warm-up/pos-emb experiments.

Attempt 2 makes the keypoint model **separate**: its own PTv3 encoder+decoder
and its own query decoder, so it can build representations specialized for
finding points instead of borrowing the masker's. This first defines a
performance **ceiling**; compute trade-offs (e.g. sharing the encoder) can be
made later once we know what's achievable.

## 2. Inputs assumed

- Stage-2 output: the candidate **nu slice** points (this is what the Stage-1/2
  cache files hold — `LArFormerStage12CacheDataset`, `emit_keypoints=True`).
- Stage-3 output: the **deduped particle queries / masks** (from the frozen
  `CascadedParticleSegmenter`, which itself contains the Stage-2 slicer).

The new model **contains the Stage-3 model**, the same way Stage-3 contains the
Stage-2 slicer (`CascadedParticleSegmenter` wraps `CascadedSlicer`:
`build_model` inner → freeze + pin eval → `train()` override → run inner under
`no_grad` in `forward`, then filter/recenter to the nu slice).

## 3. Architecture (three modules)

```
CascadedKeypoint  (top-level, Phase 3 — mirrors Stage-3-wraps-Stage-2)
├── frozen CascadedParticleSegmenter   → nu-slice batch + per-event particle masks
├── KeypointSliceModel  (Phase 1)      — 2nd PTv3 enc(frozen)+dec(trainable)
│     tokenizer levels: voxel_4cm, ptv3_dec3, ptv3_dec2, spacepoint(, spacepoint)
│     dense head @ ptv3_dec2 :  object / no-object   (soft Gaussian σ=3cm)
│     dense head @ spacepoint :  nu-vertex / no       (= kpscores[:,0], in cache)
│     NO query decoder
└── KeypointParticleDecoder  (Phase 2) — per-particle query decoder
      per particle (predicted mask | noised-GT-instance denoising):
        subset its points → spacepoint+voxel levels from already-sliced sp_feat
        seed queries from slice-level dec2 object scores (subset to particle)
        query classes {start, end, no-object}; regress 3D pos; MSE on matched kp
```

Inference flow: frozen cascade → recentered nu-slice batch + particle mask logits
→ `KeypointSliceModel` once on the slice → loop predicted masks through
`KeypointParticleDecoder` → decode keypoints.

**Note on the 2nd PTv3:** for now reload a full second PTv3 enc+dec copy (encoder
frozen from the Sonata pretrain, decoder trainable). Later: share Stage-3's
encoder and run only a second decoder — deferred.

## 4. Does it fit the existing LArFormer setup?

- **Slice-level pass — YES.** It is a LArFormer forward with the Mask2Former
  query decoder **off** and two per-level dense heads **on**. Reuses the backbone
  wrapper + PTv3 dec-stage capture (`_encode`, `_dec_stage_capture`,
  `PTv3DecoderStageLevel`), `CompositeTokenizer` + builders, `PerTokenClsHead`.
- **Per-particle pass — NOT in one `forward`.** Tokenizer/decoder are per-*event*;
  we need per-*particle* batching. Solution: treat **each particle as a
  pseudo-event** and run a second tokenizer+decoder pass on a particle-batched
  sub-batch (sub-batch built by masking points + recomputing `offset`, the
  `filter_batch_by_keep_mask` pattern).
- **The one trap:** `ptv3_dec2/dec3` levels need the global PTv3 dec-stage
  captures, so re-slicing them per particle needs index remapping. **Avoided** by
  the locked decision below: the per-particle decoder uses only spacepoint+voxel
  levels built on the subset (those need only `sp_feat`+`coords`, already
  computed for the slice — no second backbone run, no dec-stage remap). dec2 is
  used only at the slice level; per-particle query **seeds** borrow the slice-level
  dec2 object scores subset to the particle's points.

## 5. Ground truth & matching (mostly already available)

Per-point (cache, `emit_keypoints=True`): `coord_norm`, `feat`, `trackid`,
`particle_class_id`, `kpscores` (N,6 soft Gaussian σ=3cm), `kpoffsets`,
`mckeypoints_pos_norm/_type/_trackid`.
Per-instance (`gt_instances_per_event[*]`): `truth_indices` (the point set),
`primary_trackid`, `class_id`, `origin_coord_norm` (start, all types incl.
shower), `end_coord_norm` + `has_end` (tracks).

- **dec2 "object" soft label**: per dec2 token at coord c,
  `target = max_k exp(−‖c−kp_k‖²/2σ²)` over relevant keypoint types
  (track_start/end/shower/…; nu_vertex handled by the spacepoint head). σ=3cm.
- **spacepoint nu-vertex soft label**: `kpscores[:,0]` (already emitted).
- **per-particle start/end** (Phase 2): straight off the GT instance
  (`origin_coord_norm`, `end_coord_norm`/`has_end`) — no trackid puzzle for the
  GT/denoising path. For the **predicted-mask** path, match a predicted mask to a
  GT instance by point-overlap IoU, then use that instance's start/end.
- Frame: cascade recenters `coord_norm` to slice centroid → recenter keypoint
  targets by the same per-event centroid; `keypoint_eval._recover_affine`
  inverts it for decode.

## 6. Locked decisions (2026-06-17)

1. **Build = extend LArFormer (reuse).** Add (a) a decoder-optional path
   (run encoder+tokenizer+dense heads with the query decoder skipped) and
   (b) per-level dense **soft-label** heads + a soft-target classification loss.
   Phase 1 is then a LArFormer instance with the decoder off.
2. **Phase 2 supervises BOTH** predicted masks (primary, Hungarian/IoU match)
   **and noised GT instances (denoising auxiliary)** — the noised-GT path is the
   equivalent of LArFormer's existing DN path (`MaskDenoiser` /
   `query_denoising.py` / `compute_dn_loss` / DN self-attn mask). Reuse/adapt
   that machinery rather than inventing a separate GT-only phase. (May still
   stage *within* Phase 2: stand up one path, then add the other.)
3. **Per-particle decoder levels = spacepoint + voxel only** (on the subset).
   No PTv3 dec-stage levels per particle (sidesteps the global-index remap).

## 7. Phased plan

- **Phase 0 — data/labels.** Verify dev cache has `mckeypoints` + `kpscores`.
  Soft-label helper: per-token Gaussian proximity at arbitrary level coords (dec2
  object); reuse `kpscores[:,0]` (spacepoint nu-vertex). Soft-target cls loss
  (soft-CE/BCE).
- **Phase 1 — slice-level dense model (START HERE).** LArFormer (decoder off) +
  dense head @ dec2 (object) + dense head @ spacepoint (nu-vertex). Train
  standalone on the Stage-2 cache (no cascade needed yet). **Milestone:** dec2
  object precision/recall good enough to be a reliable mixed-query source +
  sensible nu-vertex resolution. This validates the premise before any
  per-particle complexity.
  - **DONE (mechanism, 2026-06-17).** Implemented as a decoupled
    `level_keypoint_heads` arg on `LArFormer` (model.py): a list of
    `{level, name, kp_types, hidden_dim, weight, pos_weight, pos_threshold}`
    specs. Each attaches a `KeypointScoreHead(in_dim=token_dim, n_types=1)` to a
    named level's tokens; the soft target is `max` over the level's member
    spacepoints (via `sp_to_level_id`, helper `_pool_sp_to_level_max`) of
    `max(kpscores[:, kp_types])` — the dataset's Gaussian proximity field
    (`emit_keypoints=True`); loss is weighted-MSE on the sigmoid
    (`_level_keypoint_loss`, mirrors the per-SP dense head). Predictions land in
    `pred["level_kp"][name] = {score, coords, level, sp_to_level_id}` for the
    evaluator + Phase-2 query seeding. `kpscores`/`kpoffsets` added to
    `_per_sp_labels_for_event`. The decoder-optional path (`num_queries=0`)
    already existed — no new plumbing needed there.
  - Config: `configs/lartpc/larformer-keypoint2-slice-v1.py` — frozen encoder
    (`freeze_backbone=True`) + trainable PTv3 decoder (`unfreeze_decoder=True`,
    from scratch, `ptv3_decoder_init_scale=0.01`), levels
    (voxel_4cm, ptv3_dec3, ptv3_dec2, spacepoint), heads: dec2 `object`
    (kp_types 1-5), spacepoint `nu_vertex` (kp_types 0). `num_queries=0`,
    `evaluate=False` for now.
  - **Validated:** build + forward + backward clean (decoder None, enc frozen /
    dec+heads trainable, finite grads); 80-step overfit on 4 dev events drove
    `object` 0.13→0.0004 (pos-region MSE 0.12→0.0003) and `nu_vertex`
    0.18→0.0001 — the heads have capacity to fit the soft targets.
  - **Evaluator DONE (2026-06-17).** `LArFormerLevelKeypointEvaluator`
    (keypoint2_evaluator.py) — a STANDALONE hook (the particle/slicer eval loop
    assumes query class/mask/origin outputs this model lacks). Logs:
    `val/object_ap` (threshold-free headline; GT = pooled max(kpscores[:,types])
    per dec2 token, binarized) + `val/object_P|R` @ score_thresh +
    `val/object_pos_frac`; `val/nu_vertex_res_cm_{median,mean}` +
    `val/nu_vertex_recall` (decode = score-weighted centroid of SPs >
    nu_decode_thresh; GT type-0 mckeypoint in the SAME recentered coord_norm
    frame — the dataset recenters mckeypoints with coord_norm, so resolution is
    `‖Δnorm‖·coord_scale`). `best_metric=val/object_ap` drives CheckpointSaver.
    Wired into `larformer-keypoint2-slice-v1.py`; validated standalone (AP/PR +
    res compute correctly, best_metric published).
  - **Dev run DONE (2026-06-17).** Full `tools/train.py`, 50 epochs on the
    10-event dev cache (output `/mnt/ddrive/pointcept_exp/larformer_keypoint2_slice_v1`
    — root disk full). Final dev metrics:
      * dec2 **object AP 0.99**, P/R@0.5 ≈ 0.90/0.99 (trajectory
        0.16→0.62→0.82→0.97→0.99) — strong mixed-query-source candidate.
      * **nu-vertex res median ~1.5 cm**, recall@3cm 0.8 (mean ~28 cm: a couple
        of events with no nearby decoded vertex drag the mean; median is the
        number to watch).
    This is memorization (train≈val, 10 events) — it proves pipeline + head
    capacity + that the evaluator metrics behave, NOT generalization. NOTE:
    `warmup_iters=200` >> dev iters so the LR under-ramped yet still hit AP 0.99;
    correct for the full run, but for a dev-perf read lower it (~5-10). Real
    generalization needs the full cache (cluster).
- **Phase 2 — per-particle query decoder.** Pseudo-event sub-batching
  (particles→events); spacepoint+voxel levels on subsets; seed from slice dec2
  scores; classes {start,end,no-object}; MSE regression; predicted-mask matching
  + noised-GT denoising path (decision 2).
  - **Phase 2a DONE (core mechanism, 2026-06-17).** `keypoint2_particle.py`:
    `ParticleKeypointDecoder` — DETR-style, queries cross-attend (FULL attention,
    no mask gating) to a particle's spacepoint tokens + self-attend over N
    layers; per-query heads class∈{start,end,no_object} + 3D pos (zero-init →
    pos=seed at init). `particle_keypoint_loss` — Hungarian match queries→{start,
    end} GT (cost = class CE + L2), CE(class)+smooth_L1(pos), deep-supervised.
    Integrated into `LArFormer` behind `enable_particle_keypoint` /
    `particle_keypoint_cfg`: `_particle_keypoint_pass` loops the event's GT
    instances (truth_indices → subset), builds a spacepoint level via a small
    own tokenizer (no PTv3 dec-stage → no remap), seeds queries from the
    slice-level object head (`seed_level=ptv3_dec2`, `seed_head=object`,
    top-`n_seed_queries` dec2 tokens by object score among the particle's points),
    runs the decoder, and adds the loss. GT origin/end share the recentered
    coord_norm frame (dataset recenters both) → plain coord_norm distance.
  - **Capacity VALIDATED (the decisive check):** 400-step overfit on 4 dev
    events drove cls→0.000 and **start_err 24→0.35 cm, end_err 7.6→0.21 cm** —
    SUB-CM. The separate decoder with full attention to the particle's geometry
    has the capacity to localize below the 1 cm target — the expressivity
    attempt 1 lacked (its frozen mask-tailored embeddings plateaued ~10 cm).
  - **Config + eval DONE (2026-06-17).** `larformer-keypoint2-particle-v1.py`
    inherits the slice config via `_base_` and just sets
    `enable_particle_keypoint=True` + `particle_keypoint_cfg` (num_queries=8,
    3 layers, seed_level=ptv3_dec2, seed_head=object, n_seed_queries=4,
    weight_class=1, weight_pos=5). `LArFormerLevelKeypointEvaluator` extended:
    reads `pred["particle_kp"]` (each carries `inst_idx` to pair with the GT
    instance — the pass skips empty instances so positions aren't 1:1) and logs
    `val/pkp_{start,end}_err_cm_{median,mean}` + `val/pkp_{start,end}_R{1,3,10}`
    (best START/END query by class prob vs the instance origin/end).
  - **Dev run DONE (pipeline validated, 2026-06-17).** Full `tools/train.py`,
    50 epochs on the 10-event dev cache (`--options scheduler.warmup_iters=10`,
    output `/mnt/ddrive/pointcept_exp/larformer_keypoint2_particle_v1`). All loss
    components + all eval metrics log correctly (object AP 0.97; per-particle
    start err median ~6-11 cm, R@1cm ~0 on VAL). The big gap vs the overfit's
    0.35 cm is EXPECTED and not a bug: overfit = same 4 events, 400 steps @3e-4
    (capacity); dev = 10 HELD-OUT val events, ~50 iters @1e-4 on 10 train events
    (tiny-data generalization, badly under-trained — start-err even regressed
    6→11 cm late = small-data train-overfit + the dec2 seed source co-training
    and shifting seeds). Pipeline + capacity proven; real per-particle perf needs
    the full cache (cluster).
  - **Phase 2b denoising DONE (2026-06-17).** Mirrors `MaskDenoiser`. In
    `keypoint2_particle.py`: `ParticleKeypointDecoder` gained a `class_embedding`
    + a DN forward mode (`add_learnable_base=False` → queries = pure init
    content, no learnable base/query_pos, variable count). `noise_particle_indices`
    (drop `drop_frac` of a particle's points + add `add_frac` random non-member
    event points — imperfect-mask sim, resampled per step) and
    `particle_keypoint_dn_loss` (KNOWN assignment query↔target, CE + smooth-L1,
    no Hungarian). In `_particle_keypoint_pass`: per instance, build `n_groups`
    DN copies of its keypoints, jitter the anchors (`anchor_jitter_std`), seed
    with class embeddings, run the decoder over the noised point set, add the DN
    loss (`particle_dn_weight`). Config knob `particle_keypoint_cfg.denoise`
    (enabled in `larformer-keypoint2-particle-v1.py`).
  - **Validated:** DN overfit — `dn_cls` 1.32→0.000 (learns the type), `dn_total`
    1.23→0.002; the REGULAR path still reaches ~1.3 cm (DN doesn't interfere).
    DN position error plateaus ~8 cm BY DESIGN (noise resampled each step → not
    memorizable; reflects genuine localize-under-noise difficulty = the
    robustness DN instills for Phase-3 predicted masks).
  - **Predicted-mask matching path: deferred to Phase 3** (needs the Stage-3
    cascade to supply predicted masks + IoU→GT matching; same decoder, just fed
    predicted point sets instead of GT instances).

**Phase 2 status: COMPLETE** (decoder + loss + dec2 seeding + per-particle eval +
denoising; runnable end-to-end via `larformer-keypoint2-particle-v1.py`).

- **Phase 3 — cascade wrapper (code complete, 2026-06-17; end-to-end deferred).**
  - **3a:** `CascadedParticleSegmenter` gained `expose_filtered_batch` — its eval
    return now also carries `ps_batch` (the recentered nu-slice the keypoint
    model's own backbone runs on).
  - **3b:** `keypoint2_cascade.predicted_masks_to_instances` — converts a
    predicted particle prediction into keypoint-decoder instances: dedup via the
    SAME `inference.dedup_queries` as the masks (confidence floor + mask-IoU NMS),
    each active query → pseudo-instance (`truth_indices` = its spacepoints,
    min_points filter); optional IoU→GT match attaches the GT origin/end.
    **Validated** (synthetic): dedup drops the duplicate query (2→1), 3 instances
    recovered, IoU-match attaches origin (iou 1.0); inference (no GT) → 3
    pseudo-instances, no origin.
  - **3c:** `CascadedKeypoint` (registered MODEL) — builds + freezes the cascade
    (`expose_filtered_batch=True`, eval-pinned, `train()` override, no_grad
    forward), builds the attempt-2 keypoint model, and in forward runs cascade →
    `ps_batch` + predictions → builds particle instances (predicted | gt) →
    runs the keypoint model. Inference: pred carries `level_kp` + `particle_kp`.
  - **FRAME CAVEAT:** the cascade recenters ONLY `coord_norm`, not GT origin/end
    (Stage-3 trains on the cache — which recenters together — and only infers
    through the cascade). So the wrapper is INFERENCE-correct; training the
    keypoint model THROUGH the cascade (predicted-mask matching w/ IoU GT) needs
    the matched GT recentered by the same centroid first (follow-up). Train the
    keypoint model standalone on the cache (Phase 1/2); Phase-2b denoising
    already gives imperfect-mask robustness. IoU-match branch left in for later.
  - **END-TO-END VALIDATED ON REAL DATA (2026-06-17).** All weights exist on
    this machine: deghoster `lora_deghost_v6_hasmatch/epoch_30`, sonata backbone
    `lartpc_v6_..._epoch_42`, slicer `..._iter_75750`, particle `epoch_6`, +
    raw merged_h5 (`devdata_mergedh5_pi0filter*.txt`). New tool
    `tools/run_larformer_keypoint2_cascade_inference.py` builds `CascadedKeypoint`
    (cascade self-loads deghoster+slicer+sonata; `--particle-weights` →
    particle_segmenter; `--keypoint-weights` → keypoint_model), runs on raw
    events, decodes per-particle start/end + dense nu-vertex to DETECTOR CM
    (per-event affine via `keypoint_eval._recover_affine`), writes per-event H5
    (`keypoints/particle_start_cm|particle_end_cm|particle_class|nu_vertex_cm`).
    - Verified on 146928-point raw events: cascade → ~4.1k-SP nu slice → 5-8
      predicted particles → keypoint model → cm keypoints. `particle_class` came
      out `[gamma,p,gamma,gamma,gamma,gamma,p]` on a π0-filter event (photon-rich
      — physically sensible, from the real epoch_6 particle weights). Keypoint
      POSITIONS are under-trained (dev keypoint ckpt) but the full wiring +
      cm-decode is proven.
    - Two wiring fixes made: (1) `CascadedKeypoint` feeds PREDICTED instances via
      `particle_instances_per_event` (separate from `gt_instances_per_event`) so
      the keypoint model's eval-with-GT diagnostic (`loss_fn` reads `origin_type`)
      isn't handed pseudo-instances; (2) it carries `ps_coord/ps_coord_norm/
      ps_offset` in the eval output for the cm decode. `pred_class` propagated
      into the per-particle result so the H5 carries the cascade PID.
  - **Still deferred:** a viz overlay for the attempt-2 cm keypoints.

Remaining: full-cache training run (cluster) of Phase 1/2 — the real
generalization test — then re-run this inference tool with the trained weights
for physically-meaningful keypoints.
- **Phase 2 — per-particle query decoder.** Pseudo-event sub-batching
  (particles→events); spacepoint+voxel levels on subsets; seed from slice dec2
  scores; classes {start,end,no-object}; MSE regression; predicted-mask matching
  + noised-GT denoising path (decision 2).
- **Phase 3 — cascade wrapper.** `CascadedKeypoint` containing the frozen
  `CascadedParticleSegmenter`; expose the recentered nu-slice batch from the
  cascade (small change to `CascadedParticleSegmenter.forward` to return it);
  loop predicted masks at inference.
- **Phase 4 — decode/metrics/inference/viz.** Reuse most of attempt 1's
  `keypoint_eval.py`, evaluator, full-cascade inference script, visualizer.

## 8. Reuse map

Reuse as-is: PTv3 backbone wrapper + dec-stage capture, `CompositeTokenizer` +
builders (`VoxelBuilder`, `SpacepointBuilder`, `PTv3DecoderStageLevel`),
`PerTokenClsHead`, `Mask2FormerDecoder`, `MixedQuerySelector`, `MaskDenoiser`,
`HungarianMatcher`, Phase-4 decode/eval.
New (fork): `keypoint2_slice.py` (or LArFormer extension), `keypoint2_particle.py`,
`cascaded_keypoint.py`, soft-label loss, evaluator, this README.

## 9. Risks / open items

- dec stage cm scales: confirm `ptv3_dec2`≈1 cm / `ptv3_dec3`≈2 cm for this
  backbone grid (the existing particle config wired dec2=in_dim128, dec3=in_dim256
  at coarser use — verify the actual voxel size per stage).
- Decoder-optional path in LArFormer: needs the mask/primary-supervision
  asserts relaxed when the decoder is off.
- Per-particle query seeding from a *subset* of dec2 tokens: custom seeder (the
  stock `MixedQuerySelector` reads a whole level's cls logits).
- Denoising for keypoints: DN currently noises masks/boxes for the masker; adapt
  to noise GT keypoint positions / particle point-sets.
- Predicted-mask→GT IoU matching introduces sim-to-real noise; the denoising
  path is the intended stabilizer.
