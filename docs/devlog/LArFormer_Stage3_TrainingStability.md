# LArFormer Stage 3 — Training-Loss Stability Analysis & Remediation Plan

> **Status: WORKLOG (2026-06)** — Stage-3 training-loss diagnosis and LR-schedule fixes; conclusions folded into the Stage-3 production recipe.

Status doc for the rising/oscillating training losses observed in the Stage-3
particle-segmenter runs (`wandb: pointcept-larformer-stage3`). Use the
checkboxes in §4 to track implementation.

- **Config under analysis:** [`configs/lartpc/larformer/stage3_particle/larformer-particle-v1-cached-ptv3crosslevel.py`](../../configs/lartpc/larformer/stage3_particle/larformer-particle-v1-cached-ptv3crosslevel.py)
- **Runs:** (brown) base_lr=1e-4 flat; (blue) resume with `reset_lr=5e-5`;
  (purple) same checkpoint, `reset_optimizer=True`.
- **Symptom:** `loss_mask_primary`, `loss_dice_primary`, `loss_cls`, and the
  aux-mask losses bottom out around iter ~60–70k (≈ epoch 3) and then drift
  upward with large oscillation. Neither the LR cut nor the optimizer reset
  changed the behavior.
- **Counter-evidence that the model is fine:** `val/mask_iou_*` rises
  monotonically for every class, `val/origin_l2_cm_*` falls, and visual scans
  show masks improving (longer complete tracks, fewer overlapping queries).

Analysis date: 2026-06-10 (Claude-assisted code review of the Stage-3 stack:
`losses.py`, `matcher.py`, `decoder.py`, `query_selection.py`,
`query_denoising.py`, `trainer.py`, `utils/scheduler.py`, `hooks/misc.py`).

---

## 1. Diagnosis

The rise is **not model degradation**. Four mechanisms, in order of likely
contribution:

### 1.1 The mask loss is a moving target (adaptive hard-negative sampling)

The config sets `_IMPORTANCE_BUDGET=0.75` with `HARD_NEG_FRACTION_OF_IMPORT=0.5`,
i.e. per matched pair, of ~4096 sampled negatives: ~1536 are *boundary-halo*
points (smallest |σ−0.5|), ~1536 are the **most-confident false positives**
(topk σ over a 3×-oversampled bg pool), ~1024 random
([`_importance_sample_negatives`](../../pointcept/models/LArFormer/losses.py)).

Early in training all logits are small, so even "hard" negatives cost ~0.7 nats.
As the model sharpens, the hard-neg branch reliably finds the residual handful
of confidently-wrong points, each costing 5–20+ nats (logits capped at ±50).
**The measured per-sample loss rises even as the count of wrong points shrinks
and full-mask IoU improves.** The sampled Dice term has the same artifact (its
denominator includes the σ of the hand-picked confident FPs).

Note `val/loss_*` uses the **same sampler** (the eval path calls the same
`loss_fn`), so rising val loss does not contradict rising val IoU.

### 1.2 Confidence sharpening raises even non-sampled terms

The aux-mask BCE at `voxel_8cm` / `voxel_4cm` is *full-mask* (token counts <
`aux_max_tokens`) and also drifts up. Coarse voxels carry irreducible label
ambiguity (a voxel touched by two particles is GT-positive for both; Stage-2 FP
spacepoints and vertex-adjacent points are genuinely unlearnable). As logits
saturate, per-token BCE on those points grows linearly in logit magnitude.
Same for the query-class CE: a single match flip on a now-confident class head
costs far more CE than at epoch 1. This explains the *timing*: nothing in the
schedule changes at epoch 3 — it's the crossover where loss growth on residual
/ ambiguous points outweighs the shrinking error count.

### 1.3 Hungarian-assignment churn, amplified by cost-sampling noise

The matcher cost is mask-dominated (`cost_mask=5, cost_dice=5` vs
`cost_class=2`) and is evaluated on a **freshly randomized** balanced sample of
`num_sample_points=8192` tokens every forward
([`LArFormerLoss.forward`](../../pointcept/models/LArFormer/losses.py), the
`_balanced_point_sample` call before `self.matcher(...)`). With 32 queries,
~3 GT instances, and queries still partially overlapping the same particle,
near-tied costs flip assignment iter-to-iter; every flip spikes cls/mask/origin
loss for the affected pairs. The sawtooth in
`adam_state/head.../exp_avg_sq_mean` is consistent with recurring spike
gradients of this kind.

**Smoking gun in the existing plots:** the denoising-path losses, which use
*direct GT assignment with no Hungarian matching*
([`compute_dn_loss`](../../pointcept/models/LArFormer/losses.py)), fall smoothly
the whole run (`loss_dn_cls` 1.0→~0.3, `loss_dn_origin` likewise) while the
Hungarian-matched `loss_cls` bottoms at ~3.4 and rises to ~3.9. Same heads,
same decoder weights, same data — only the assignment mechanism differs.

### 1.4 Flat LR at the gradient-noise floor — and the plateau decay never fired

Post-warmup the LR is held flat. **Additionally: the `FlatWithDecayLR` plateau
mode was inert in all three runs.** Its `step_epoch()` is only called by the
[`LREpochScheduler`](../../pointcept/engines/hooks/misc.py) hook, which is **not
in the config's `hooks` list** — nothing else calls `step_epoch` (verified by
grep over `engines/` and `models/LArFormer/`). So the configured
plateau/gamma/patience knobs did nothing; `params/lr` stayed exactly flat at
1e-4 (then 5e-5 via the `reset_lr` resume override). At a constant LR, once at
the noise floor the loss diffuses rather than decreases, and mechanisms 1–3
tilt the measured curve upward.

### 1.5 On the optimizer reset (run 3)

Resetting Adam state cannot address any of the above: the moments re-estimate
within ~1k iters (the `update_ratio` spike-and-settle at the resume point is
the bias-correction transient). The purple curve matching the blue curve
confirms the oscillation source is the *objective*, not stale moments.
**Do not keep `reset_optimizer=True` for future resumes.**

---

## 2. Is this concerning for Mask2Former-type models?

Mostly no. Noisy, non-monotone training loss is the norm for set-prediction
models — bipartite matching makes the loss piecewise and assignment flips
inject spikes that never fully vanish. Adaptive hard-example mining further
decouples the loss *value* from model quality by design.

Healthy signals currently present: val IoU rising for all classes, origin
error falling, DN-path losses falling, `matched_fraction = 1`, no growing
matcher-sanitization warnings.

**Genuine warning signs to watch for** (none present as of 2026-06-10):
- `val/mask_iou_mean` declining for ≥ 2 consecutive epochs
- DN-path losses (`loss_dn_*`) starting to rise
- Increasing NaN-hook / grad-clip / matcher-sanitization activity
- Probe losses (§6, once implemented) rising

---

## 3. Recommendations (from the review)

| # | Recommendation | Priority | Status |
|---|----------------|----------|--------|
| R1 | Decay the LR deterministically (cosine from 5e-5 → ~1e-6) instead of relying on the (inert) plateau detector | High | ☐ planned — §5 |
| R2 | Add log-only probe/diagnostic quantities per training batch (fixed-sampler mask loss, matched-pair full-mask IoU, churn + saturation metrics) | High | ☐ planned — §6 |
| R3 | Reduce matcher cost-sampling noise (full-mask cost or per-event deterministic sample); log a match-churn metric | Medium | ☐ |
| R4 | Soften the hard-negative branch (`HARD_NEG_FRACTION_OF_IMPORT` 0.5 → 0.25, or anneal); optionally mask/class label smoothing to bound saturation-driven loss growth | Medium | ☐ |
| R5 | Maintain a weight EMA for eval/checkpoint selection | Medium | ☐ |
| R6 | Stop resetting the optimizer on resume | High (policy) | ☐ adopt |
| R7 | Note: train total includes DN terms; eval skips DN — don't compare the two `loss` curves directly | Info | — |
| R8 | Inference-side query dedup (mask-IoU NMS with merge tracking) — fixes the μ/π duplicate-query fragmentation observed in hand scans | High | ☐ planned — §7 |

---

## 4. Tracking checklist

- [x] R1.1 Register `DelayedCosineLR` in `pointcept/utils/scheduler.py` (2026-06-10)
- [x] R1.2 Patch `CheckpointLoader.extend_scheduler` fast-forward to include `iter_in_epoch` (2026-06-10 — **required**, not optional: the resume point is a mid-epoch IterCheckpointSaver checkpoint)
- [x] R1.3 Create `configs/lartpc/larformer/stage3_particle/larformer-particle-v1-cached-ptv3crosslevel-decaylrsched.py` (2026-06-10; unit tests in `tests/test_delayed_cosine_lr.py` all pass, incl. validation of the fast-forward formula against a real checkpoint)
- [ ] R1.4 Launch on cluster + verify resume log lines and `params/lr` trajectory
  - First launch attempt (2026-06-10) crashed in `CheckpointLoader`'s
    extend-scheduler logging: it formatted `group.get('max_lr', 'N/A')`
    with `:.6f`, but `max_lr` only exists for OneCycleLR-family
    schedulers — absent for LambdaLR-family ones like DelayedCosineLR.
    Fixed in `hooks/misc.py` (defensive `_fmt_lr`). The crash happened
    BEFORE the scheduler counter was set, so nothing was corrupted;
    plain relaunch is safe.
  - The cluster checkpoint named `model_iter_131140.pth` actually
    contains the NEXT IterCheckpointSaver save (iter 18250 → step
    131190, one save_iter_freq=50 later than the log line it was named
    after). Harmless: 131190 > decay_start_step=131140, so the cosine
    starts immediately at resume, 50 steps into the window (LR there is
    still 5.0e-5 to 6 significant figures). No config change needed.
- [x] R2.1 Add `log_diagnostics` flag + probe losses to `LArFormerLoss` (2026-06-10)
- [x] R2.2 Add saturation / confident-FP metrics (2026-06-10)
- [x] R2.3 Add init↔final match-agreement metric (2026-06-10)
- [x] R2.4 Stamp pre-clip grad norm into the logged output dict (2026-06-10 — both base `Trainer.run_step` clip sites + the LArFormerTrainer dual-DN path)
- [ ] R2.5 After launch: confirm `train_batch/loss_diag_*`, `val/loss_diag_*`, and `train_batch/grad_norm` appear in wandb; per-iter walltime delta negligible
  - Train-side CONFIRMED (2026-06-10, `resume3_cosinedecay`, iters
    131k–134k). First-look readings, all sensible — these are the
    baseline values for the flat-5e-5 regime (cosine decay not yet
    perceptible; half-amplitude ~step 291k):
    - `diag_mask_iou_matched` ≈ 0.72 flat — consistent with
      `val/mask_iou_mean` ≈ 0.69 / median ≈ 0.74. Probe validated.
    - `diag_mask_bce_rand` ≈ 0.19, `diag_dice_rand` ≈ 0.20, both flat —
      vs trained per-layer ≈ 0.21 / ≈ 0.29 (sums/7, incl. weaker early
      layers). The stationary baselines to compare future loss drift
      against.
    - `diag_mask_logit_p95` ≈ 20 — each residual confident-wrong point
      costs ~20 nats; confirms the §1.2 sharpening mechanism. Not at
      the ±50 cap (healthy).
    - `diag_frac_confident_fp` ≈ 0.02 (spikes 0.05–0.09 on hard
      events) — small residual error pool; should shrink long-term.
    - `diag_match_agreement` ≈ 0.42 — the decoder stack changes the
      preferred assignment for >half the GT instances within one
      forward (init vs final). Consistent with §1.3 near-tie matching.
      Watch the trend: rising = matches locking in.
    - `grad_norm` ≈ 150–190 smoothed, spikes to ~1400 — `clip_grad=1.0`
      is saturated on EVERY iter: training runs on gradient direction
      with per-batch magnitude equalized, spike batches suppressed ~8×
      harder. Keep clip at 1.0 (with Adam, the absolute level is
      nearly moot; saturation here is protective). Watch spike
      frequency as the LR decays.
  - Val-side (`val/loss_diag_*`) pending first eval epoch.
- [ ] R3 / R4 / R5 — design after R1+R2 data is in
- [x] R8.1 `dedup_queries()` in `pointcept/models/LArFormer/inference.py` + unit tests (2026-06-10 — `tests/test_stage3_dedup.py`: merge semantics + runner-up record, Michel-containment non-merge, no-chain semantics, off-switch/edge cases, H5 roundtrip; all pass)
- [x] R8.2 Wired into `stage3_predict_event_from_out` (+ `stage3_predict_event` kwarg); §7.4 schema keys emitted; per-level assignments inherit the deduped query set (2026-06-10)
- [x] R8.3 `--dedup-iou-threshold` CLI knob (default 0.6; 0 disables), forwarded in BOTH cached and full-cascade modes (2026-06-10)
- [ ] R8.4 Validate on duplicate-rich sample: smoke-tested end-to-end on the laptop dev cache (epoch-4 ckpt, 3 events: runs clean, 0 merges — and an IoU probe over 8 events confirms max active-pair mask IoU ≈ 0.43, i.e. no co-extensive duplicates exist there to merge: correct negative control). REMAINING: cluster run on the chargedpiplus events with the current checkpoint — before/after on fileno00380 entry000000, `dedup_max_pair_iou` distribution to confirm threshold placement, and the same-class two-GT negative control.

---

## 5. Implementation plan — R1: mid-run switch to a cosine decay schedule

> **IMPLEMENTED 2026-06-10.** Actual numbers (from the cluster run's
> IterCheckpointSaver log and checkpoint metadata):
>
> - iters/epoch = **22 588** (actual `len(train_loader)`, not the 25 625
>   estimate in the old config comment)
> - Resume checkpoint: `exp/..._lr1e4_bugfixed_resume2/model_iter_131140.pth`,
>   saved at `epoch=5 iter=18200/22588` → global optimizer step **131 140**
> - `decay_start_step=131140`, `total_steps=451 760` (20 epochs),
>   `base_lr=5e-5`, `final_lr_scale=0.02` → cosine 5e-5 → 1e-6 over the
>   remaining ~14.2 epochs
> - New `save_path`: `exp/larformer_particle_v1_cached_ptv3crosslevel_smallbatch_resume3_cosinedecay`
> - The §5.4 patch was applied (required: the resume point is mid-epoch).
>   Validated against a real checkpoint: `epoch*22588 + iter_in_epoch`
>   reproduces the saved scheduler counter exactly (98 302 for the laptop
>   copy `..._bugfixed/model_iter_98652.pth`).
> - Tests: `./run_in_container.sh python tests/test_delayed_cosine_lr.py`
>   (hold/decay/floor trajectory, monotonicity, the initial_lr
>   save/restore dance, constructor guards, real-checkpoint formula check).

### 5.1 Mechanism available today

- `Trainer.build_scheduler` injects
  `total_steps = len(train_loader) × eval_epoch // gradient_accumulation_steps`
  (≈ 25 625 iters/epoch × 20 epochs ≈ 512.5k for this config).
- `CheckpointLoader(extend_scheduler=True)`
  ([hooks/misc.py](../../pointcept/engines/hooks/misc.py)) already supports a
  mid-run scheduler **swap**: on resume it does *not* load the saved scheduler
  state; it builds the scheduler from the *new* config, preserves the new
  scheduler's `initial_lr`/`max_lr` against the optimizer-state load, and
  fast-forwards `last_epoch` to
  `checkpoint["epoch"] × len(train_loader) // grad_accum`.

### 5.2 The gap

A plain `CosineAnnealingLR(total_steps)` fast-forwarded to the resume step
lands **mid-cosine of a schedule that began at step 0** — e.g. resuming at
epoch 5/20 from base 1e-4 gives lr ≈ 8.5e-5: an upward LR jump, not a fresh
decay from 5e-5. We want the cosine to *begin* at the checkpoint.

### 5.3 New scheduler: `DelayedCosineLR` (~20 lines)

Add to [`pointcept/utils/scheduler.py`](../../pointcept/utils/scheduler.py),
implemented as a `LambdaLR` (pure function of step → safe under the
fast-forward, clean `state_dict` round-trip):

```python
@SCHEDULERS.register_module()
class DelayedCosineLR(lr_scheduler.LambdaLR):
    """Hold base LR until `decay_start_step`, then cosine-decay to
    final_lr_scale * base_lr at total_steps. For mid-run schedule swaps
    via CheckpointLoader(extend_scheduler=True)."""
    def __init__(self, optimizer, total_steps, decay_start_step=0,
                 final_lr_scale=0.02, last_epoch=-1):
        T = max(1, int(total_steps) - int(decay_start_step))
        def factor(s):
            if s < decay_start_step:
                return 1.0
            t = min(s - decay_start_step, T) / T
            return final_lr_scale + (1.0 - final_lr_scale) * 0.5 * (1.0 + math.cos(math.pi * t))
        super().__init__(optimizer, lr_lambda=factor, last_epoch=last_epoch)
```

If the fast-forward lands slightly before `decay_start_step`, the LR just
holds at base for a few iters — benign.

**Zero-new-code fallback:** `MultiStepLR` (already registered, milestones as
fractions of `total_steps`) with `base_lr=5e-5`, milestones all *after* the
resume fraction, `gamma=0.5`. The fast-forwarded factor is automatically
correct (1.0 until the first milestone). Gives the "×0.5 every 2–3 epochs"
step decay with no scheduler code at all.

### 5.4 Optional robustness patch

`extend_scheduler` computes `steps_completed = epoch × len(train_loader)` —
**epoch granularity**. Resuming from an `IterCheckpointSaver` mid-epoch
checkpoint under-counts by `iter_in_epoch`. Either resume from an
epoch-boundary checkpoint, or patch the line to add
`checkpoint.get("iter_in_epoch", 0) // grad_accum`.

### 5.5 The new config (copy of the current one)

`configs/lartpc/larformer/stage3_particle/larformer-particle-v1-cached-ptv3crosslevel-decaylrsched.py`,
deltas only:

1. **Resume source:** `weight = "<...>/exp/larformer_particle_v1_cached_ptv3crosslevel_smallbatch_lr1e4_bugfixed_resume2B_resetoptim/model/model_last.pth"`,
   `resume = True`.
2. **New `save_path`** (e.g. `..._cosinedecay`).
3. **Hooks:** `CheckpointLoader(extend_scheduler=True)` — **drop
   `reset_optimizer=True`** (keep Adam moments; see §1.5).
4. **Optimizer:** `base_lr = 5.0e-5` (must equal the LR the run is currently
   at — the new scheduler's `initial_lr` is taken from this).
5. **Scheduler:**
   ```python
   scheduler = dict(
       type="DelayedCosineLR",
       decay_start_step=RESUME_EPOCH * 25625,   # match the loader's fast-forward
       final_lr_scale=0.02,                      # 5e-5 → 1e-6 at epoch 20
   )
   ```
   No warmup (long past it), no `reset_lr` (that knob was `FlatWithDecayLR`-
   specific and re-stomps the LR on *every* resume).
6. Everything else (model, data, loss_kwargs, batch size) unchanged so the
   only experimental delta is the schedule.

### 5.6 Verification after launch

- Log shows `Extending scheduler: setting step counter to <N>` with N ≈
  `decay_start_step`, and `Extended scheduler LR at resume: 0.000050`.
- `params/lr` in wandb starts at 5e-5 and decreases smoothly (no upward jump,
  no warmup ramp).
- No `reset_optimizer` log line; `adam_state/update_ratio_*` shows **no**
  resume spike (moments preserved).
- Subsequent SLURM auto-resumes within the run: scheduler state now loads
  normally (`extend_scheduler` only matters for the swap resume; it is safe to
  leave on since the config's scheduler no longer changes — but flipping it
  back to default after the first successful swap checkpoint is cleaner).

---

## 6. Implementation plan — R2: per-batch diagnostic logging

> **IMPLEMENTED 2026-06-10** (Phase 1 + Phase 2 A/C/D in one pass).
>
> - `LArFormerLoss(log_diagnostics=True)` — new ctor flag; diagnostics
>   computed in `_compute_diagnostics` (no_grad, final layer only, only
>   when ≥1 matched pair). Keys: `diag_mask_bce_rand`, `diag_dice_rand`,
>   `diag_mask_iou_matched`, `diag_mask_logit_p95`,
>   `diag_frac_confident_fp`, `diag_match_agreement`. They surface as
>   `train_batch/loss_diag_*` AND, because the evaluator accumulates every
>   0-d tensor in `eval_loss`, as `val/loss_diag_*` (a free bonus — val-
>   side stationary probes).
> - `grad_norm`: both clip sites in `Trainer.run_step`
>   (`pointcept/engines/train.py`) and the `LArFormerTrainer._dual_run_step`
>   clip site stamp the pre-clip total norm returned by `clip_grad_norm_`
>   into the logged output dict → `train_batch/grad_norm`. Only present
>   on optimizer-step iters when `clip_grad` is set.
> - Enabled in `larformer-particle-v1-cached-ptv3crosslevel-decaylrsched.py`
>   via `loss_kwargs.log_diagnostics=True`; other configs unaffected
>   (flag defaults to False).
> - NOT implemented (deferred): `diag_match_margin` (needs the matcher to
>   return its cost matrix) and the matcher-sanitized-counter delta (§6.3
>   B/E).
> - Tests: `./run_in_container.sh python tests/test_larformer_diagnostics.py`
>   — key gating, total-loss invariance, near-perfect / over-claiming
>   synthetic events, and an ordering check that reproduces the §1.1
>   artifact in miniature: on identical predictions with injected
>   confident FPs, the importance/hard-neg-sampled training BCE (0.45/layer)
>   exceeds the random-probe BCE (0.33).

### 6.1 Mechanism

`InformationWriter.after_step` logs **every 0-d tensor** in the model's
training output dict to wandb as `train_batch/<key>`; the model's per-event
aggregator prefixes loss-dict keys with `loss_`
([model.py](../../pointcept/models/LArFormer/model.py), training-branch
aggregation). So diagnostics are added purely by inserting extra scalar
entries into the dict `LArFormerLoss.forward` returns — no trainer or hook
changes needed (except the grad-norm item, §6.3-D).

All probe quantities computed under `torch.no_grad()` on the **final layer
only** (interpretable per-layer values, not deep-supervision sums), gated by a
new `loss_kwargs` flag `log_diagnostics=True` so other configs are unaffected.

### 6.2 Phase 1 quantities (the decisive ones)

| Key (wandb: `train_batch/loss_<key>`) | What | Why |
|---|---|---|
| `diag_mask_bce_rand`, `diag_dice_rand` | `_per_pair_sampled_mask_loss` on final-layer matched pairs with `use_importance_sampling=False` (pure random negatives), no_grad | Stationary-sampler counterpart of the training loss. **If this falls/holds while `loss_mask_primary` rises, mechanism 1.1 is confirmed and the rise can be ignored.** |
| `diag_mask_iou_matched` | Hard IoU (σ>0.5 over *all* primary tokens) per matched pair, averaged | Train-time counterpart of `val/mask_iou_mean`; direct "is the model actually getting worse on train data" signal. Cost trivial: (P≈3, M≈10⁴) boolean ops. |
| `diag_mask_logit_p95` | 95th-percentile |logit| over matched queries' primary mask logits | Tracks confidence sharpening (mechanism 1.2). |
| `diag_frac_confident_fp` | Fraction of bg tokens with σ>0.9, averaged over matched pairs | The pool the hard-neg sampler feeds on. Should *shrink* over training even while its per-point loss grows — separates "fewer but pricier errors" from "more errors". |

### 6.3 Phase 2 quantities

- **A. `diag_match_agreement`** — run the (no_grad) Hungarian match on the
  `init`-layer predictions as well and report the fraction of GT instances
  assigned to the same query as the final-layer match. Cheap within-iteration
  proxy for assignment stability (cross-iteration churn isn't measurable —
  different events per batch).
- **B. `diag_match_margin`** — from the final cost matrix, mean over GT of
  (second-best − chosen) assignment cost. Low margin = flip-prone matches.
  Needs `HungarianMatcher` to optionally return the cost matrix.
- **C. `diag_cls_maxprob_matched`** — mean max-softmax of matched queries'
  class predictions (class-head sharpening; pairs with the rising `loss_cls`).
- **D. `grad_norm`** — pre-clip total norm: capture the return value of
  `clip_grad_norm_` in the trainer step and stamp it into
  `comm_info["model_output_dict"]` before `after_step` hooks run. Tells us
  whether `clip_grad=1.0` is permanently saturated (effective normalized-
  gradient regime) and exposes spike frequency. Requires a small trainer
  change (base `Trainer.run_step` or an `after_step`-ordered hook).
- **E. `diag_matcher_sanitized`** — per-iter delta of
  `matcher._n_sanitized_forwards` (should stay 0; a cheap canary).

### 6.4 Validation

- One dev-cache batch run on the laptop: confirm new keys appear in the log
  line and wandb, confirm per-iter walltime delta is negligible (<2%).
- Confirm eval path is unaffected (the probe code runs inside the loss; eval
  calls it too — extra keys in `eval_loss` are harmless to
  `LArFormerParticleEvaluator`, verify once).

### 6.5 Interpretation guide (once live)

- `loss_mask_primary` ↑ while `diag_mask_bce_rand` ↓ and `diag_mask_iou_matched` ↑
  → sampling artifact; no action.
- `diag_mask_iou_matched` ↓ → genuine regression; revisit LR / R4.
- `diag_match_agreement` low & `loss_cls` spiky → prioritize R3 (full-mask
  matcher cost).
- `grad_norm` ≫ 1 constantly → clip saturation; consider raising clip or
  lowering LR rather than letting clipping silently set the step size.

---

## 7. Implementation plan — R8: inference-side query dedup

### 7.1 Failure mode (hand-scan, 2026-06-10)

The most common Stage-3 error is one track covered by TWO overlapping
queries — characteristically a μ-classifying query and a π-classifying
query over the same true μ or π. The training loss subsidizes this
configuration: the matcher's class cost always matches the
correct-class query (low matched CE regardless of which hypothesis is
right), while the unmatched duplicate pays only `no_object_weight=0.1`
CE and **zero mask penalty** (mask losses run on matched pairs only).
So "one query per class hypothesis" is the optimal hedge for the
irreducible μ/π ambiguity. The damage happens at inference: the
per-SP panoptic argmax lets the two hypotheses FRAGMENT the track
(interleaved per-point assignment) instead of one winning the whole
instance. FPS query seeding guarantees the duplicate candidates exist
(long tracks always receive several anchors).

### 7.2 Algorithm — greedy mask-IoU NMS over active queries

Runs inside `stage3_predict_event_from_out`, between the existing
confidence-floor demotion (`effective_argmax`) and the per-SP panoptic
assignment. Operates on binarized spacepoint-level masks
(`logit > 0`, same convention as `per_pair_iou`).

```
inputs:  sp_mask_logits (Q, N), effective_argmax (Q,), cls_max_prob (Q,),
         no_object_class_id, iou_threshold (default 0.6)

1. active   = effective_argmax != no_object
2. binmask  = sp_mask_logits > 0                       # (Q, N) bool
3. score[q] = cls_max_prob[q] * mean(sigmoid(logit) over binmask[q])
              (Mask2Former panoptic-style: class conf × mask conf;
               score = 0 for empty masks)
4. pairwise IoU among active queries via one matmul:
              inter = binmask_f @ binmask_f.T ;  union = a_i + a_j − inter
5. greedy, descending score:
       for q in order:
           if suppressed[q]: continue            # q survives
           for r in lower-score active, not yet suppressed:
               if IoU(q, r) >= iou_threshold:
                   suppressed[r] = True ; winner[r] = q
6. demote: effective_argmax[suppressed] = no_object
7. per-SP assignment re-runs unchanged → the winner inherits the
   loser's points automatically (their masks overlap by construction,
   so the winner's logit is the surviving argmax there). No explicit
   mask union needed.
```

Design choices and why:

- **IoU (symmetric), not intersection-over-min:** keeps genuinely
  distinct contained instances (Michel e at a μ endpoint, δ-rays)
  unmerged — small-in-big has high IoM but low IoU. The μ/π duplicate
  pair has near-identical full-track masks → IoU typically ≳ 0.8;
  default threshold 0.6 with a CLI knob.
- **Class-agnostic merging:** required — the observed duplicates are
  *different-class* (μ vs π) by construction of the hedge.
- **Suppress-then-reassign, not mask union:** reuses the existing
  panoptic argmax; one code path for per-SP and per-level outputs.
- **Greedy one-level winners (no chains):** losers can't absorb;
  standard NMS semantics; Q=32 makes cost trivial.
- **Hungarian/GT diagnostics untouched:** matching, `pair_iou`,
  `pair_origin_l2_cm` stay raw-model metrics. Dedup affects only the
  panoptic *assignment* outputs (+ new keys below). A per-query
  winner-redirect array lets consumers follow a matched-but-suppressed
  query to its absorber.

### 7.3 Merge tracking (the ambiguity record)

For each surviving query, the absorbed runner-up's class hypothesis is
preserved — a track tagged "μ 0.6 / π 0.4" is more honest than a
fragmented pair and is potentially useful in the downstream event
selection. Recorded per query (arrays length Q; −1/0 where n/a):
suppression flag, winner index, dedup score, number absorbed,
runner-up class + prob (highest-prob absorbed query with class ≠
winner's, else highest absolutely), and max pair IoU at merge (the
"strength" of the duplication).

### 7.4 Schema integration (new keys only — no existing key changes)

Existing consumers (`visualize_stage3_larformer_from_cached.py`,
analysis joins) read `stage3/pred_query` etc. and will simply see the
deduped assignment. Pre-dedup assignment is preserved for A/B.

| Key | Shape | Meaning |
|---|---|---|
| `stage3/pred_query_nodedup` | (N,) i64 | pre-dedup per-SP assignment (only written when dedup enabled) |
| `stage3_queries/dedup_suppressed` | (Q,) bool | query was absorbed |
| `stage3_queries/dedup_winner_idx` | (Q,) i64 | suppressed → absorber; kept → own idx; inactive → −1 |
| `stage3_queries/dedup_score` | (Q,) f32 | ranking score (cls conf × mask conf) |
| `stage3_queries/is_active_postdedup` | (Q,) bool | `is_active & ~suppressed` |
| `stage3_queries/dedup_n_absorbed` | (Q,) i64 | duplicates merged into this query |
| `stage3_queries/dedup_runnerup_class` | (Q,) i64 | absorbed runner-up class (−1 none) |
| `stage3_queries/dedup_runnerup_prob` | (Q,) f32 | its class confidence |
| `stage3_queries/dedup_max_pair_iou` | (Q,) f32 | max mask IoU among absorbed |
| `stage3_gt/matched_query_dedup` | (K,) i64 | `matched_query` redirected through winner map |
| `stage3_meta/dedup_iou_threshold` | attr | 0 = dedup disabled |
| `stage3_meta/n_dedup_suppressed` | attr | per-event merge count |
| `stage3_meta/n_active_queries_postdedup` | attr | |

`stage3_levels/*/pred_query` uses the post-dedup `effective_argmax`
(consistent across levels), no extra per-level keys. Full class
posteriors of all queries are already stored
(`stage3_queries/class_probs`), so any recombination of merged
posteriors can be done offline via `dedup_winner_idx`.

### 7.5 Implementation steps

1. `dedup_queries(...)` as a standalone function in `inference.py`
   (torch in, numpy-records out), so the visualizer / future ROOT
   converter can reuse it. Unit tests: two heavy-overlap queries +
   one distinct (loser suppressed, winner inherits points, records
   correct); threshold sweep incl. 0=off; Michel-in-muon containment
   case NOT merged; no-active and empty-mask edge cases.
2. Call it in `stage3_predict_event_from_out` after the confidence
   floor; thread `dedup_iou_threshold` kwarg through
   `stage3_predict_event` as well; emit §7.4 keys.
3. CLI: `--dedup-iou-threshold` (default 0.6; `0` disables) in
   `tools/larformer/run_larformer_stage3_inference.py`, forwarded in BOTH
   cached and full-cascade modes; stamp into `stage3_meta`.
4. Validate on the val cache with the current checkpoint:
   - before/after visual on the known duplicate events
     (e.g. `fileno00380 entry000000`, μ/π pairs);
   - distribution of `dedup_max_pair_iou` (expect a high-IoU peak —
     confirms threshold placement) and per-event suppression counts;
   - check no legitimate splits merged: events where two SAME-class
     GTs share a vertex must keep two queries.
5. Optional follow-ups: post-dedup assigned-IoU per GT
   (`stage3_gt/assigned_iou`) as the deliverable-quality metric; a
   `pred_query_nodedup` color mode in the Stage-3 visualizer; the
   training-side `diag_duplicate_rate` (count of active query pairs
   with mask IoU > 0.5) to watch the duplicate rate during training.

## 8. Revision history

- **2026-06-10** — Initial analysis + R1/R2 implementation plans (Claude-assisted review).
- **2026-06-10** — R1 + R2 implemented and launched (`resume3_cosinedecay`); first-look diagnostic readings recorded (§4 R2.5). Added R8: inference-side query dedup plan (§7) after hand scans showed the dominant error is μ/π duplicate-query pairs fragmenting single tracks.
