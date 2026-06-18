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

**Inference eval script (`eval_keypoint2_inference.py`, 2026-06-18).** Runs on a
dir of inference-output H5s and reports the SAME per-particle metrics as the
evaluator — start/end median/mean + R@{1,3,10}cm, a per-predicted-class
breakdown (shower vs track), nu-vertex resolution + recall@3 — micro-averaged.
Accepts multiple dirs for side-by-side (e.g. `--particle-source predicted` vs
`gt`). Object-head AP isn't computable from the inference H5 (it stores decoded
keypoints, not per-dec2-token object scores) — watch that in the training log.
(In `--particle-source gt` mode the per-class line shows `[-1]`: the GT-trackid
instances carry no predicted class; use the `start (all)` line there.)

**Train→inference gap diagnostic (`--particle-source`, 2026-06-18).** The
per-particle decoder TRAINS on GT instance masks (truth_indices) on the CACHE
slice, but at inference runs on PREDICTED Stage-3 masks on the LIVE CASCADE slice
— two distinct input shifts (mask source + slice membership) on top of any
overfitting. `run_larformer_keypoint2_cascade_inference.py --particle-source
{predicted|gt}` isolates the mask-source effect: "gt" reconstructs GT masks in
the slice frame by grouping the surviving per-SP trackid (CascadedKeypoint) and
feeds them to the decoder. If start error with "gt" ≈ val but "predicted" is far
worse → the gap is GT-mask vs predicted-mask (structural; fix = train on
predicted masks / stronger denoising, NOT just more data).

**ROOT-CAUSE FOUND — val→inference gap is overfit-memorization, not a pipeline
bug (2026-06-18).** Diagnosed with the start-GT overfit checkpoint
(`larformer_keypoint2_overfit_startkp`, 2000 iters on the 10-event
`cache_stage12_devdata`). **Val (cache input): particle start median 0.45 cm,
mean 0.47, R@1 0.95** (R@3≈1.0). **Cascade GT-mask inference on the SAME 10
events: median 1.85–2.0, mean 12–16, R@3 ≈ 0.58–0.61.** Ruled out, in order:
  - *Units* — evaluator and inference-eval both report cm (`*coord_scale`). Not it.
  - *Mask source* — `--particle-source gt` (GT masks, = training input) still
    gives R@3 0.58; predicted masks only drop it to 0.49. So GT-vs-pred mask is
    the SMALLER effect, ~0.09.
  - *Slice density / `max_spacepoints`* — the cache was built with a **cap of
    80000** (raw 122k–275k → ~80000 per the `n_after_dataset_filter` attr; the
    cap uses `np.random.permutation`, so it is a RANDOM subsample), but
    `data.test` (inference) uses `max_spacepoints=None` → the live nu-slice is
    2-11× larger/denser than the cache's. CORRECTION to the prior note: this
    asymmetry IS real, but it is NOT the dominant cause. Matching it via the new
    `--max-spacepoints 80000` flag shrank the slices to cache size (e.g.
    1282→287, 3102→148) yet moved start R@3 only 0.58→0.61 and made nu-vertex
    recall WORSE (0.70→0.50). Not it.
  - **Actual cause:** the cascade RE-DERIVES the nu slice live (random 80k
    subsample → different deghost members → different `build_nu_keep_mask` set →
    different recenter centroid → shifted absolute pos_emb; plus trackid-grouped
    masks vs the cache's frozen `truth_indices`). The cache, by contrast, froze
    one specific slice. An overfit-on-10-events model memorized the exact cache
    tensors, so on the cascade's (necessarily different) reconstruction of the
    same event the per-particle start errors are a **bimodal scatter, not a
    systematic offset**: mean error VECTOR only ~4.5 cm but per-axis std 11-15 cm;
    ~60% of particles stay <3 cm (≈val) while ~25-40% blow up to 15-177 cm. A
    frame/recenter bug would shift the median too — it doesn't (median stays
    ~2 cm). So the wiring is correct; the gap is memorization breaking under the
    cascade's input perturbation.

  **Takeaways:**
  1. For an overfit model, VAL ON THE CACHE IS NOT A VALID PROXY for cascade
     performance — the cache is a frozen, bit-reproducible slice; the cascade is
     not (random subsample + live recompute). Trust the cascade-path eval.
  2. The fix is NOT a threshold/flag — it is to (a) train on real (non-overfit)
     data so the model learns generalizable geometry, and/or (b) train the
     keypoint model THROUGH the cascade on the live slice (the deferred Phase-3
     predicted-mask training), so training sees the same live-slice distribution
     (live subsample, live centroid, predicted masks) as inference.
  3. Minor hardening worth doing regardless: make the `max_spacepoints` cap a
     seeded/deterministic subsample (or set the cache cap = inference cap = None)
     so the cache and cascade at least agree on point budget — removes one of the
     perturbation axes, though it won't close the gap for an overfit model.
  `--max-spacepoints N` was added to
  `run_larformer_keypoint2_cascade_inference.py` to override the test split's cap
  for this kind of diagnosis.

  **CLARIFICATION on the "recenter centroid" axis (2026-06-18).** The "Actual
  cause" bullet above lists "different recenter centroid → shifted absolute
  pos_emb" as one perturbation axis. To be precise: the diagnostic ran with
  `source_set_filter="stage2_pass"` (= source_mask bit 0 = the PREDICTED nu slice
  only — NOT a union with GT-nu), and the cascade likewise recenters over its
  predict-only nu-keep slice. So the centroid DEFINITION matched on both sides;
  the centroids differed only via slightly different predict-only MEMBERSHIP
  (the old cache's 80k random subsample + ~3% set diff), which is a small effect.
  This is CONFIRMED by the measurement: the mean error VECTOR was only ~4.5 cm
  (no large systematic offset). A genuinely centroid-driven shift would be a
  uniform offset — measured separately at ~10 cm IFF the training slice is a
  UNION (predict ∪ GT-nu) recentered over the full union (see below). The
  diagnostic used no union, so that 10 cm shift was NOT present and is NOT the
  cause of the observed gap. Net: the centroid finding does not invalidate the
  root cause (overfit-memorization scatter) — it explains WHY "not a systematic
  offset" held. The union-centroid shift is a FUTURE concern only.

  **`recenter_centroid_source` (2026-06-18).** If a UNION slice is adopted
  (`source_set_filter="union"`/`"stage2_plus_gt_dropout"`/`"stage2_random_tau"` —
  to enrich particle masks + augment toward a more-accurate slicer), recentering
  over the full union shifts every LIVE point ~10 cm vs the cascade's predict-only
  recenter at inference — a real train↔inference divergence. New dataset arg
  `LArFormerStage12CacheDataset(recenter_centroid_source=...)`: `"kept"` (default,
  legacy) recenters over all kept points; `"stage2_pass"` recenters over ONLY the
  bit-0 (predicted/live) subset, so the live points land in the SAME frame as
  inference even with a union slice. VERIFIED: union + `recenter_centroid_source=
  "stage2_pass"` reproduces the live points' coord_norm to 0.0000 cm vs the
  predict-only baseline; union + `"kept"` is off by 10.0 cm. No-op under the
  current `stage2_pass` filter (kept == bit 0).

**Predicted-mask training path (`main_source`, 2026-06-18).** To attack the
val→inference gap at its source (train on the SAME imperfect masks seen at
inference), the per-particle decoder can now split its two paths by mask source:
- **MAIN (Hungarian-matched) path → predicted masks.** A FROZEN particle
  segmenter is built INTO the keypoint LArFormer (`particle_keypoint_cfg.
  segmenter` + `segmenter_weight`) and run under `no_grad` on each cache slice
  per step; its deduped predicted masks (via the SAME
  `predicted_masks_to_instances` as CascadedKeypoint inference — confidence
  floor + mask-IoU NMS, then IoU→GT match to attach start/end targets;
  unmatched predicted masks → all-`no_object`) drive the matching loss.
- **DN path → GT masks** (unchanged): jittered GT keypoints over a
  drop/add-noised GT point set, known assignment → clean-mask supervision.

So the model sees correct masks (GT via DN) AND predicted masks (MAIN). Config:
`configs/lartpc/larformer-keypoint2-particle-predmask-v1.py` (inherits the
particle config, sets `main_source="predicted"`, pulls the segmenter sub-cfg
from the fullcascade config — the one matching `model_iter_98652.pth`,
`backbone_out_channels=64`; NOT larformer-particle-v1.py whose segmenter uses
`in_dim=1232` and size-mismatches the ckpt). `main_source="gt"` (default) keeps
the legacy behavior (GT masks drive both paths). Implementation:
`model.py::_particle_keypoint_pass` now takes `main_instances` + `dn_instances`;
`_run_particle_segmenter` / `_particle_main_instances` build the predicted list;
`predicted_masks_to_instances` also copies `start_coord_norm`/`has_start` (the
visible-start target). VERIFIED end-to-end on a no-cap cache event: both paths
emit losses (`loss_pkp_kp_*` from predicted, `loss_pkp_dn_*` from GT), backward
runs, segmenter stays frozen (no grads). Costs one extra cheap (small-slice)
frozen forward + a second backbone in VRAM.

**Cached predicted masks — stage-1+2+3 cache (`predicted_cached`, 2026-06-18).**
Running the frozen segmenter LIVE (`main_source="predicted"`) ~2x'd batch time
(1.8→3.6 s). To avoid that, precompute the deduped predicted masks ONCE into the
cache and read them at train time (NO live segmenter, NO second backbone in
VRAM; batch time back to ~1.7 s).
- `tools/augment_stage12_cache_pred_masks.py` (GPU): for each event it loads the
  exact trainer input via `LArFormerStage12CacheDataset` (same filter+recenter),
  runs the frozen segmenter, dedups via the SAME `predicted_masks_to_instances`,
  IoU-matches GT, and writes `entry_0/pred_instances/instance_<k>/`
  (`truth_indices` in ORIGINAL cache-SP space + `pred_class`/`primary_trackid`/
  `match_iou` attrs). In-place (r+) or `--output-dir`; idempotent.
- `LArFormerStage12CacheDataset(load_pred_instances=True)` reads them, remaps
  `truth_indices` exactly like GT, re-attaches the matched GT's recentered
  start/end targets by trackid, and emits `particle_instances` → collate →
  `particle_instances_per_event`. The model's MAIN path then uses these (DN still
  uses GT). Config: `larformer-keypoint2-particle-predmask-cached-v1.py`.
- VERIFIED: cached forward 1.66 s (vs 3.6 s live), no segmenter built; cached
  matched-instance IoU-vs-GT median 0.72 (round-trip correct). NOTE the segmenter
  is non-deterministic run-to-run (TF32/cuBLAS → dedup-boundary flips), so the
  cached masks are a representative SAMPLE, not bit-identical to any one live run
  — fine for training (inference re-derives its own). Add a `set_deterministic`
  pass to the augment if you ever need a reproducible cache.

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
  - **Reproducibility (2026-06-17; see docs/LArFormer_Reproducibility.md).** The
    inference tool got a `--deterministic` flag that calls `set_deterministic()`
    BEFORE building the model (TF32 off, deterministic algorithms,
    CUBLAS_WORKSPACE_CONFIG, seeds) — this pins `torch.sort` in `SerializedPooling`
    for ALL FOUR backbones (deghoster+slicer+particle+keypoint). The dataset is
    RNG-free in the test split (fixed lm-score threshold, `max_spacepoints=None`),
    but the forward consumes per-event RNG, so a once-at-startup seed only gives
    SAME-ORDER run-to-run repeatability — an event's output still depended on its
    POSITION in the sequence (measured: same event processed alone vs 2nd in a
    trio differed by up to 985 cm, with an endpoint-existence flip). FIX: the tool
    RE-SEEDS before every event (`np`/`torch`/`cuda`), so each event's result
    depends only on the event. Verified (deterministic, A100):
      * single event, run-to-run: **bit-exact (Δ=0.0 cm)**.
      * a 3-event set processed in different order / an event processed alone vs
        in the set: keypoints agree to **≤2e-3 cm (21 µm)**, with **0 particle-
        count / class / endpoint-existence flips** (residual is float-level cuBLAS
        accumulation order, no deterministic-algorithm fallbacks — physically
        negligible vs the 1 cm target).
    NOTE: the dataset SORTS its file list (`get_data_list -> sorted`), so the
    processing order is canonical regardless of input-list order; the tool labels
    each output H5 with the actual processed `src_file` (+ run/subrun/event when
    available) so reorder/subset checks match the same physical event.
  - **Visualizer DONE (2026-06-17).** The inference tool now writes a RICH
    per-event H5 (`slice/coord_cm`; per `particle/{i}`: `point_idx` into the
    slice, `cls`, `start_cm`, `end_cm`, and the matched-GT `gt_point_idx` /
    `gt_start_cm` / `gt_end_cm` / `iou` / `gt_trackid`; `nu_vertex_cm` +
    `gt_nu_vertex_cm`). GT matching is by **majority per-SP `trackid` vote**, NOT
    IoU-vs-gt_instances: the cascade strips `gt_instances` before the slicer, but
    per-SP `trackid` + per-event `mckeypoints` survive into `ps_batch`, so the
    predicted particle's points vote a GT trackid and GT keypoints come from
    `mckeypoints` (start = type track_start/shower, end = track_end, nu-vertex =
    type 0). Enable with `--with-gt` (default; `--no-gt` for real data).
    `tools/visualize_keypoint2_cascade.py` renders an interactive Plotly HTML:
    LEFT = predicted particles + predicted start/end + nu-vertex with a dropdown
    (All / individual particle; the selected one is colour-highlighted, the rest
    drawn small+grey for context); RIGHT (when GT present) = the trackid-matched
    GT particle + GT keypoints (empty if no match). Validated end-to-end on real
    events: 6/6 and 5/5 particles matched, sensible PIDs (e/gamma/p), figure
    structure correct (dropdown buttons, two scenes, context greying).
    A **Dash** version `tools/visualize_keypoint2_cascade_dash.py` serves a whole
    directory of event H5s as a live app (Event + Particle dropdowns → callback
    rebuilds the 3D figure; camera preserved via `uirevision`). Both share the
    trace/visibility logic (`_assemble_traces` / `_visibility` /
    `figure_for_view` in the static tool). Smoke-tested: app builds (2 callbacks),
    server returns HTTP 200 and serves all layout components.
    `python tools/visualize_keypoint2_cascade_dash.py <dir> --port 8050`.
  - **START target = VISIBLE start, not origin (2026-06-18; CHANGES THE LOSS —
    requires retraining).** Originally the per-particle decoder's start target
    was the instance `origin_coord_norm` (the particle BIRTH point). For a SHOWER
    the photon is born at the nu/pi0 vertex and travels invisibly before
    converting, so the origin is OFF the particle's own spacepoints — after the
    decoder slices out the particle's points there is nothing to localize the
    origin from, and the model could only MEMORIZE it (which is what the 2k run
    did; viz error >> val makes sense). The VISIBLE start — the `track_start`
    (tracks) / `shower` (showers) keypoint POSITION — sits ON the points and is
    learnable. Measured: origin vs visible-start differ by **10.8 cm median, up
    to 54 cm** for e/gamma showers (0 for tracks). Change: the datasets
    (`larformer_stage12_cache.py` cache + `larformer.py` raw) now attach
    `start_coord_norm`/`has_start` per instance = `endpoint_by_trackid` over
    {track_start, shower} of the mckeypoint `pos_norm` (recentered alongside the
    others). The per-particle loss (`_particle_keypoint_pass`, + DN path via the
    shared `gp`), the evaluator (`pkp_start_err`), and the viz inference tool all
    now use `start_coord_norm` (fallback to origin if no start kp tagged; 38/42
    instances tagged). **The existing checkpoint was trained on origin — RETRAIN
    to pick up the visible-start target; then pred should land on the on-cloud
    start.** END target (`track_end`) and nu-vertex (dense head) unchanged.
    (Viz GT for the matched particle is read from the RAW sample's gt_instances,
    which the cascade strips internally but the held batch still carries, keyed
    by trackid — same source as the loss; mckeypoints used only for nu-vertex.)
  - **GT-keypoint frame fix (2026-06-17).** Two normalized frames coexist in
    `ps_batch`: `coord_norm` is RECENTERED to the slice centroid by the cascade,
    but `mckeypoints_pos_norm` is the dataset's FIXED normalization
    ((cm-coord_center)/coord_scale; the cascade recenters only coord_norm). So
    PREDICTED keypoints (model `pos`, recentered) denormalize via the per-event
    recovered affine, while GT keypoints (mckeypoints) MUST denormalize with the
    fixed (coord_center, coord_scale) — using the recovered affine offset the GT
    keypoints by the slice-centroid shift (~100 cm; visible as GT keypoints not
    landing on the GT spacepoints). After the fix: GT start→nearest GT point
    0.0-0.5 cm, GT nu-vertex inside the slice and ~2 cm from the predicted one.
  - **Shared-range fix (2026-06-17).** The Predicted and Matched-GT scenes were
    auto-ranging independently, so identical coordinates mapped to different
    screen positions (looked like a pred-vs-GT spacepoint offset). Confirmed NOT
    a real offset — matched pred/GT particles share many identical slice indices
    (same `slice_coord_cm`, same frame; centroids agree to a few cm = mask-
    coverage difference only). Fix: both scenes use ONE shared x/y/z range (over
    slice points + finite keypoints) + the same initial camera (`_ranges` /
    `_scene`, shared by both tools). Figure-only change — no inference re-run
    needed, just regenerate the HTML / restart the Dash app.
  - **Linked rotation (Dash, 2026-06-17).** Rotating/zooming either 3D scene
    drives both (so Predicted and Matched-GT stay aligned while inspecting).
    Loop-safe via a `cam` `dcc.Store`: a capture callback updates the Store only
    when the camera actually changes; an apply callback `Patch`es both
    `scene.camera`/`scene2.camera`; the programmatic update re-emits the same
    camera, which the capture callback sees as unchanged → no-op, breaking the
    cycle. (Static HTML keeps Plotly's built-in per-scene controls.)

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
