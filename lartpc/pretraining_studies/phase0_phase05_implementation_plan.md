# Phase 0 / Phase 0.5 Implementation Plan

**Companion to:** `microboone_sonata_experiment_plan.md`
**Environment:** Isambard (pretraining, 4×GH200 nodes) + Tufts (probe training/eval on transferred checkpoints)
**Written:** 2026-07-13, from a code audit of the current working tree (branch `nutufts_isambard`).

---

## 1. Current-state audit — what already exists vs what the plan assumes

The audit found the infrastructure is further along than the experiment plan assumes in
some places, and has gaps the plan doesn't list in others.

### 1.1 Mid-epoch checkpointing/eval: mostly EXISTS already

Contrary to the "checkpoint/eval only at end of epoch" assumption:

| Capability | Status | Where |
|---|---|---|
| Iteration-based resumable checkpointing | **EXISTS** | `IterCheckpointSaver` (`pointcept/engines/hooks/misc.py:376`), fires in `after_step` every `save_iter_freq` iters; active in the v8 config (`save_iter_freq=50`) |
| SLURM-signal mid-epoch save + resubmit | **EXISTS** | `SignalCheckpointHook` (`misc.py:430`), active in v8 (`check_every_n_iter=30`) |
| Mid-epoch resume (fast-forward sampler / islice) | **EXISTS** | `train.py:194-245`, `skip_dataloader_on_resume=True` in v8 |
| Step-based evaluation | **EXISTS** | `SemSegEvaluator(eval_freq=N)` (`evaluator.py:127`); already used by the v5 linear-probe config (`eval_freq=100`) |
| In-loop linear-probe eval during pretraining | EXISTS but disabled | `PretrainEvaluator` (`pretrain_evaluator.py`), commented out in pretrain configs ("does not work on P100s") — optional to revive on GH200 |
| **Retained** snapshots at a cadence separate from the resumable save | **MISSING** | `IterCheckpointSaver(keep_history=True)` retains `iter_{N}.pth`, but at the same (very frequent) `save_iter_freq`, and always with full training state |
| Weights-only snapshots (for transfer) | **MISSING** | all saves include optimizer/scheduler/scaler → 1.5 GB each in the v8 run |
| Val metrics on an iteration axis | **MISSING** | evaluators log wandb/tensorboard val metrics keyed by integer epoch (`evaluator.py:214`), so multiple mid-epoch evals overwrite one point |

**⇒ The mid-epoch work item is a small extension of `IterCheckpointSaver`, not new machinery.** See WP2.

### 1.2 Dataloader for Phase 0.5 (ghost-dropped MC from the combined Isambard sample)

The Isambard dataset is one squashfs (`/projects/u6jo/datasets/combined_pretrain-sonata-v7-extbnb-larmatch.sqsh`, mounted at `/data`) driven by one shuffled file list `h5list_v3_combined_extbnb_mc_shuffled.txt` (954,325 files).

- **There is no `is_mc` flag** in the HDF5 files or the dataset class. But the list is
  cleanly separable by path: **EXTBNB real data** = paths containing `extbnb`
  (532,645 files); **MC** = paths containing `bnb_*_corsika` (421,680 files).
  **⇒ MC-only selection needs no code change — only filtered file lists** (WP1).
- `LArTPCDataset` already has every load-time knob Phase 0.5 needs except one:
  - `data_only` (`lartpc.py:303-314`): v8 sets `data_only=True`, so **truth is never
    loaded, even for MC files**. Phase 0.5 MC runs must set `data_only=False`.
  - `true_points_only=True` drops ghosts by `hasmatch` truth (`lartpc.py:450-456`) —
    this is the "ghost-dropped MC" switch.
  - `filter_larmatch` / `larmatch_threshold_range` — must be **disabled** for
    ghost-dropped-MC runs (truth does the filtering).
  - `drop_cosmics` / `drop_cosmics_prob` — exists (used by P05C.5).
  - **`ghost_keep_frac` (P0.4) does not exist** — small dataset addition (WP4).
- With `data_only=False`, `mckeypoints` load and `nu_vertices` become available, so
  `BiasedSphereCrop(anchor_points_key="nu_vertices")` works (needed by P05C.4; the
  probe config already uses it with `prob_random=0.25`).

### 1.3 Landmines found by the audit (not in the experiment plan)

1. **Train and val point at the SAME file list** in the v8 config (both
   `h5list_v3_combined_extbnb_mc_shuffled.txt`). Any run where val numbers matter
   needs disjoint train/val lists. Fix in WP1.
2. **The file lists live in the OLD repo path** (`/home/u6jo/twongj01.u6jo/ubpointcept/pointcept/`),
   referenced by absolute path from configs. Regenerate them under the current repo
   (version-controlled) as part of WP1.
3. **P0.10 augmentation audit is effectively answered: v6, v7 AND v8 all apply free
   3-axis rotations** (6 `RandomRotate` transforms inside `MultiViewGenerator`,
   p=0.8 each, e.g. v8 `:312-338`) — despite the v7/v8 file headers claiming
   "no rotations". All existing checkpoints are rotation-augmented (P05B.1-style
   policy). Headers must be corrected; policy must be recorded in the registry.
   Note P05B.1 still needs a fresh run — no existing checkpoint is ghost-dropped-MC.
4. **Budget conflict:** experiment plan §0.4 says batch 48 / peak_lr 5e-4; the actual
   v8 run used batch 80 / lr 2e-4. P0.1 must resolve this before any P05 pretraining
   launches (see measured throughput below).
5. **Checkpoint disk/transfer cost:** full checkpoints are 1.5 GB (55 GB already in the
   v8 run dir). Weights-only snapshots (WP2) are needed to keep the
   checkpoint-curve × many-runs matrix transferable to Tufts.
6. `PID_TO_CLASS`/ssnet label maps and probe configs exist and match the plan's E3
   protocol; the v5 probe config the plan aliases as `BASE_LINPROBE` is actually
   `configs/lartpc/linearprobe-sonata-lartpc-v5-noghost.py` (plan's filename doesn't exist).

### 1.4 Measured throughput (P0.1 input — measured 2026-07, v8 run)

From `sonata/.../train.log`: **~2.9 s/iter at batch 80 on one 4×GH200 node ⇒ ~27.6
images/s ⇒ ~2.4M images/day/node.**

| Budget | Images | Wall time (1 node) |
|---|---|---|
| MATCHED_BUDGET placeholder | 15M | ~6.3 days |
| P05_BUDGET (=1/3) | 5M | ~2.1 days |

⇒ Wave A (~14–17 jobs, mostly at P05_BUDGET) ≈ 30 node-days plus the generous-budget
P05A supervised runs. Feasible; the binding constraint is queue policy, not compute.

**DECIDED (2026-07-13, PI):**
- **batch_size = 48.** The v8 run's batch 80 was stable (no Adam-state spikes) but
  intermittently OOM'd on batches with near-max point counts — 48 is the baseline.
  At batch 48 expect roughly 1.8–2.9 s/iter (~17–27 img/s), i.e. **5M images ≈
  2.2–3.5 days/node**; re-measure in the pilot run and update here.
- **P05 budget = 5M images**, chosen to fit the remaining **~10-day Isambard
  allocation** (ends ~2026-07-23). Consequence: Wave A (~14 runs × ~2.5 days ≈ 35
  node-days) requires ~4–6 concurrent 4-GPU jobs, and any Wave C combination run
  must start by ~day 7 — see priority ordering in §3.
- **Eval placement:** P05A supervised runs get **in-loop val on Isambard**
  (`SemSegEvaluator`, small dedicated val split, eval every ~1–2k iters — cheap since
  supervised steps are single-view). P05B/C SSL runs get **no in-loop eval**;
  snapshots ship to Tufts for probing.
- **Snapshot cadence approved**, with a **pilot run first** to verify the log-spaced
  points trace the metric rise before saturation (WP2.4, §3).

### 1.5 Review of the physics-consistent flip transforms (P05B input, reviewed 2026-07-13)

The older sonata configs (`pretrain-sonata-v1m1-lartpc-v2/-v3.py:262-295`) contain a
deliberate detector-symmetry augmentation block: `RandomFlipAxis` on **y and z only**
with `swap_strength_columns=(0,1)`, plus small scale/jitter, and explicit
"NO rotations / NO x-flip" comments. Verdict on the physics logic
(`transform.py:397-524`):

**Correct:**
- **u↔v charge-column swap under y-flip is right.** MicroBooNE's U/V wire directions
  (±60° from vertical) are mirror images under y→−y, so the ionization pattern's
  U-plane projection becomes (approximately) the V-plane projection. Column order
  (u, v, y) confirmed in the loader (`lartpc.py:397-410`), so `(0,1)` swaps the right
  columns.
- **The same swap under z-flip is also right** — under z→−z the U direction maps to
  the (mirrored) V direction, so plane identities swap.
- **Swap validity does not depend on `center="mean"`:** a reflection about any plane
  parallel to the detector midplane equals the detector reflection composed with a
  translation along that axis, and translation does not mix planes. Mean-centering
  also keeps the crop in place, which the multi-view matching needs.
- Excluding x-flip (drift-direction asymmetry: diffusion, attenuation) and all
  rotations is correct for a charge-consistency-preserving policy.

**Approximations to document in the paper:** U and V responses are not identical
(first vs second induction plane, different dead-channel maps), so the swap is
approximate even after calibration. `RandomScale(0.95–1.05)` in the same block mildly
breaks charge-per-unit-length consistency — decide whether the strict P05B.2 policy
keeps or drops it (default: keep, it is small and shared across variants).

**Confirmed bug (latent):** `_reproject_wire_coords` (`transform.py:454-479`) does not
match its own docstring — the `coord − origin` subtraction is commented out and
replaced by subtracting only `origin_z` from the projected scalar, and the function
ignores the `NormalizeCoord` centering applied earlier in the pipeline. Harmless today
(all configs set `wire_projections=None` and drop wire coords from features), but it
must be rewritten for P05B.3 → folded into WP7.

**New audit finding:** the v8 config's flips are plain `RandomFlip` on **x, y, and z
with no charge swap** (`pretrain-sonata-v8...py:317,331`). So the current/default
policy (P05B.1) is charge-inconsistent through *both* free rotations *and* swap-less
flips including the non-symmetric x-flip — which sharpens the P05B.1-vs-B.2 contrast.

---

## 2. Work packages

Ordered so that WP1–WP3 unblock everything; each WP lists concrete file touches.

### WP1 — File lists + dataset splits *(blocking, ~half day, no GPU)* — **DONE 2026-07-13**

Implemented in `lartpc/filelists/` (`make_filelists.py` + `check_truth_keys.py`;
generated lists gitignored, hashes in `filelist_stats.txt`). Findings from the
smoke check: MC files carry all truth keys and **no `larmatch_score`** (so
`filter_larmatch` silently no-oped on MC events in the v8 combined run — EXTBNB
events were filtered, MC events entered unfiltered with all ghosts); mean MC ghost
fraction ≈ 0.60 (range 0.50–0.75 over 20 files); 15/20 MC files have a nu-vertex
keypoint (BiasedSphereCrop's random fallback covers the rest). MC actually
comprises 5 corsika subsamples (nu, nue, chargedpiplus, pi0filter, prod2 + set2).

1. Regenerate lists from the combined list into the repo, e.g. `lartpc/filelists/`:
   - `h5list_v3_mc_only_train.txt`, `h5list_v3_mc_only_val.txt` — `grep corsika`,
     then split disjointly (e.g. last ~5k files → val; keep the shuffle).
   - `h5list_v3_extbnb_only_{train,val}.txt` — for later phases.
   - `h5list_v3_combined_{train,val}.txt` — disjoint replacement for the current
     shared list (fixes landmine 1).
   - `h5list_v3_mc_diag1k.txt` — fixed 1k-event MC diagnostic set for M1/M2/t-SNE
     (frozen forever; every diagnostic uses this file).
2. Smoke-check (script, runs on a login/CPU node): open ~20 MC files from the squashfs
   and assert the truth keys Phase 0.5 needs exist and are sane:
   `origin`, `ssnet_label`, `hasmatch`, `pid`, `mckeypoints` (nu vertex present).
3. Deliverable: lists checked in + a `make_filelists.py` script so they are reproducible.

### WP2 — Checkpoint-snapshot infrastructure for images-seen curves *(blocking for the curve study, ~1 day)* — **DONE 2026-07-13** (pilot validation pending)

Implemented: `IterCheckpointSaver` gained `snapshot_at_iters` (explicit global-step
list) and `snapshot_freq`; snapshots are weights-only
(`state_dict` + epoch/iter/global_step/images_seen/batch_size), written atomically to
`save_path/snapshot/snapshot_iter{N}_img{M}.pth` (rsync this directory).
`SemSegEvaluator` with `eval_freq > 0` now logs val metrics on the global-Iter axis
(wandb + tensorboard) instead of colliding on integer epochs. Unit test:
`lartpc_tests/test_iter_snapshot_saver.py` (passes in the container; covers
scheduling across epoch boundaries, weights-only content, images-seen naming, and
the SonataCheckpointLoader key-remap convention). Remaining: item 6's end-to-end
validation happens via the P05B.1 pilot.

Extend `IterCheckpointSaver` / `_save_resumable_checkpoint` (`misc.py:326-428`):

1. New args: `history_at_iters=None` (explicit list of global steps — enables
   log-spacing) and/or `history_freq=None` (retain every N iters), independent of the
   frequent `save_iter_freq` overwrite of `model_last.pth`.
2. `history_weights_only=True`: retained snapshots store only
   `{"state_dict", "epoch", "iter_in_epoch", "global_step", "images_seen", "config_hash"}`
   — no optimizer/scheduler/scaler → ~3× smaller (~0.5 GB). Keep the `state_dict` key
   name so `SonataCheckpointLoader` (`misc.py:715`) consumes them unchanged.
3. Filename encodes images seen: `snapshot_iter{N}_img{M}.pth` where
   `M = global_step × batch_size × world_size` (batch size differs across runs;
   curves are plotted vs images seen, so bake it in).
4. Cadence (APPROVED): for a P05_BUDGET run (~104k iters at batch 48), log-spaced
   `[500, 1k, 2k, 4k, 8k, 16k, 32k]` then every 16k → ~12 points, ~6 GB per run.
   **Pilot-run validation:** the first Wave A run doubles as a cadence pilot — probe
   its snapshots at Tufts as soon as the first few land, and confirm the probe-mIoU
   curve resolves the rise before saturation; densify (or thin) the schedule for the
   remaining runs accordingly.
5. Val-axis fix (small, for the Tufts probe side): make `SemSegEvaluator` log val
   metrics keyed by global iter when `eval_freq > 0` (mirror what `PretrainEvaluator`
   does) so intra-epoch eval points don't collide in wandb/tensorboard.
6. Unit-ish test: 20-iter toy run asserting snapshots appear, are loadable by
   `SonataCheckpointLoader`, and `model_last.pth` resume still works.

### WP3 — Probe-at-Tufts pipeline *(~1–2 days, Tufts side)*

Workflow: Isambard writes snapshots → rsync `snapshot_*.pth` + frozen config to Tufts
→ one short probe job per snapshot → curve.

1. `tools/probe_sweep.py` (or shell driver): given a snapshot directory + probe config
   (`linearprobe-sonata-lartpc-v5-noghost.py` as base) + fixed probe budget
   (iterations, LR — set once, record in registry), launch/queue one probe per
   snapshot via `--options weight=<snapshot> save_path=exp/probes/{run_id}/img{M}`.
2. Harvest final/best val mIoU + per-class IoU + confusion (M4) per snapshot into
   `probes/{run_id}/curve.csv`; plot metric vs images-seen.
3. Bootstrap CIs (WP6.1) applied at each curve point (event-level resampling).
4. Verify once by hand: a weights-only snapshot from WP2's toy run loads into the
   probe config at Tufts (key remap `student./teacher.` → `backbone.*` path).
5. **Probe-budget calibration study** (sets the fixed budget once, per plan §0.5;
   runs at Tufts NOW on existing v6/v7/v8 checkpoints — no new pretraining needed):
   - Pick 3 checkpoints spanning training (early / mid / final epoch of one run).
   - Sweep probe length {2k, 5k, 10k, 20k iters} × head LR {3e-4, 1e-3, 3e-3},
     fixed val split, `eval_freq` on so the probe's own convergence is visible.
   - Selection rule: the smallest budget where (a) each checkpoint's metric is within
     its bootstrap CI of its 20k-iter plateau value, and (b) the *ranking* of the
     three checkpoints is stable across the sweep — ranking stability matters more
     than absolute convergence, since the curves compare checkpoints.
   - Probe-noise sizing: rerun one (checkpoint, budget) cell with 3 seeds; if
     seed-to-seed spread exceeds the bootstrap CI, report probe points as
     mean-over-seeds (and budget the extra probes accordingly).
   - Record the chosen budget in the registry; it is then frozen for every probe in
     the program.

### WP4 — `ghost_keep_frac` dataset knob (P0.4) *(~half day)* — **DONE 2026-07-13**

Implemented in `LArTPCDataset` (mutually exclusive with `true_points_only`;
per-point Bernoulli keep on `hasmatch==0`). The LArMatch mask-reconstruction
chain now replays applied masks explicitly (also fixes a latent edge-case bug
when `drop_cosmics` doesn't trigger). Test: `lartpc_tests/test_ghost_keep_frac.py`
passes against real MC files (real points preserved; ghost fraction within
binomial tolerance for frac ∈ {0, 0.5, 1}; frac=0 ≡ `true_points_only`).

In `LArTPCDataset.get_data` (`lartpc.py`), after truth load, when
`ghost_keep_frac` is set (float in [0,1]) and truth is available: keep all
`hasmatch==1` points, keep each `hasmatch==0` point with prob `ghost_keep_frac`
(per-event RNG seeded from the sampler seed for reproducibility). Mutually exclusive
with `true_points_only` (`true_points_only=True` ≡ `ghost_keep_frac=0`). Not needed
until P1A.2/P2A, but it is 30 lines next to code WP1 already touches — do it now,
with a unit test on a real MC file.

### WP5 — Phase 0.5 config generation *(~2 days once WP1–WP2 land)* — **DONE 2026-07-13** (11 configs; probe config + A.4 remain)

Implemented as a generator (`lartpc/pretraining_studies/gen_p05_configs.py` →
`configs/lartpc/p05/`, 11 configs: 8 SSL + 3 supervised), all CPU-smoke-validated
against the squashfs data (`lartpc_tests/validate_p05_configs.py`: transform
pipeline runs, feature channels match `in_channels`, truth labels reach the views,
ZeroKey produces constant charge, snapshot schedules hit identical images-seen
anchors across batch sizes). Design decisions taken:
- **Supervised ceiling = SonataSegmentor trained end-to-end** (same encoder,
  up_cast_level=4, linear head, probe-protocol data distribution: nu-anchored
  crops + drop_cosmics=0.9) so M6 fraction-of-ceiling is apples-to-apples.
- **M5 labels ride through the views**: `view_keys += segment`,
  `Collect keys += global_segment`; new `BatchCompositionLogger` hook
  (labels are never seen by the SSL loss).
- New transforms: `ZeroKey` (P05A.2/M3) and `ChannelReduce` (P05B.4 plane-summed
  charge, `in_channels=4`, no u/v swap needed — the sum is swap-invariant).
- Snapshot schedule is specified in images seen (24k→768k log-doubling, then
  every 768k), converted to iters per config batch size.
- P05C.2 ≡ P05B.1 (prototype 4096 on default augs) — one run, two roles.

**Update (same day):** the v8mc-matched Tufts probe config is now generated too
(`linearprobe-sonata-p05-mc-noghost-tufts.py`, 12th config; per-snapshot launch
via `--options weight=<snapshot> save_path=<...>`, and
`model.backbone.head_num_prototypes` must be overridden when probing the
P05C.1/C.3 prototype-sweep snapshots). **Critical fix found before launch:**
Sonata sets `requires_grad=False` on its whole teacher branch, and
`SonataSegmentor` inference runs through the teacher — so the P05A "end-to-end"
supervised configs would have silently trained ONLY the linear head.
`SonataSegmentor.__init__` now re-enables `teacher.backbone` (and idles the
student) when `freeze_backbone=False`; verified 90.7M trainable encoder params
in supervised mode and 0 in probe mode. Submit tooling (WP8-lite):
`slurm_scripts/lartpc_sonata_pretraining/launch_p05_run.sh` (config snapshot +
hash + registry row + resubmit chaining via `submit_p05_isambard.sh`) and
`launch_p05_smoke.sh` (P0.7 short-run shakedown on the val/diag lists).
Still deferred: optional P05A.4 (class-balanced sampling).

Common base for all P05 pretraining configs ("`pretrain-sonata-v8mc-*`"):
v8 config with `data_list_file` → MC-only lists, `data_only=False`,
`true_points_only=True`, `filter_larmatch=False`, budget per P05_BUDGET
(pending the P0.1 batch/LR decision), snapshot hook from WP2, and `Collect` extended
to carry `segment`/`origin` so the M5 hook can see labels.

- **P05A supervised ceiling (4 configs):** new supervised semseg config —
  `DefaultSegmentorV2` + `PT-v3m1` (same encoder dims as the Sonata student),
  feat = coord + log-ADC strength (identical `GridSample`/`NormalizeCoord`/
  `LogTransform` pipeline, no MultiView), ssnet labels, ghost-dropped MC.
  - P05A.2: new trivial transform `ZeroKey(keys=["strength"])` in train+test
    pipelines (also reused by the M3 harness).
  - P05A.3: add the 6-`RandomRotate` block from v8 to the supervised augs.
  - P05A.4 (optional): class-balanced sampling needs a weighted sampler — defer
    unless pion scarcity shows up.
- **P05B augmentation policy (4 configs):**
  - B.1 free rotations = v8's existing MultiView transform block, verbatim
    (including its swap-less x/y/z `RandomFlip` — B.1 replicates current behavior).
  - B.2 detector symmetries only: use the **existing physics-consistent block from
    `pretrain-sonata-v1m1-lartpc-v3.py:262-295`** — `RandomFlipAxis` y and z with
    `swap_strength_columns=(0,1)` (`wire_projections=None`), jitter, small scale;
    no rotations, no x-flip. Reviewed correct in §1.5. Fix the misleading v7/v8
    header text while here.
  - B.3 rotations + recomputed wire features — depends on WP7.
  - B.4 plane-summed charge: new transform `SumStrength` (3 pixval channels → 1
    scalar, then log); model `in_channels` 6→4.
- **P05C prototypes × batch composition (6 configs):** all pure config deltas —
  `head_num_prototypes` ∈ {2048, 4096, 8192}; C.4 `anchor_points_key="nu_vertices"`
  + lower `prob_random`; C.5 `drop_cosmics=True, drop_cosmics_prob≈0.9`;
  C.6 double `batch_size`, halve SphereCrop `point_max` (fixed total points).
- **M5 batch-composition hook:** new ~40-line hook (`after_step` or inside
  `before_step`) logging per-batch truth-class point fractions to wandb on the Iter
  axis; add to all P05B/C configs.

### WP6 — Diagnostics & metrics tooling (P0.2, P0.3, P0.8, P05D) *(~2–3 days, parallelizable, GPU-light)*

1. **Bootstrap utility (P0.2):** `pointcept/utils/bootstrap.py` — event-level
   resampling (≥1000), returns mean ± 68% CI for any per-event-aggregable metric;
   unit test with known distribution.
2. **M1/M2 prototype diagnostics:** script that runs a frozen Sonata checkpoint on the
   1k-event diagnostic set, takes hard prototype assignments from the head, computes
   prototype–label MI (M1) and occupancy-weighted purity (M2) vs ssnet truth.
   Runs on **existing** checkpoints immediately (P05D.2) — the v8 run's per-epoch
   checkpoints give an M1-vs-training-time curve for free.
3. **M3 charge-ablation harness:** wrapper that runs probe eval twice (intact vs
   `ZeroKey(strength)` / shuffled) and reports the delta with CIs (P05D.1).
4. **Sliced evals (P0.3):** post-processing on saved predictions — slice by `origin`
   (nu/cosmic), distance to nu vertex (`mckeypoints`, kptype==0), and local ghost
   density (ghost count in r=5 cm ball / total, KD-tree on `hasmatch==0` points).
5. **t-SNE panel (P05D.3):** re-render embeddings at PANDA-comparable upcast/scale,
   unit-normalized, colored by class.

### WP7 — Wire-projection recomputation (P0.9 → P05B.3) *(~1–2 days, has a design decision)*

`RandomFlipAxis._reproject_wire_coords` (`transform.py:454-479`) already recomputes
wire **coordinates** for flips given a `wire_projections` geometry dict — currently a
TODO stub set to `None` in configs, and **confirmed buggy** (§1.5): origin handling
contradicts the docstring and `NormalizeCoord` centering is ignored. Rewrite it as
part of item 2 below rather than patching.

1. Fill MicroBooNE geometry: U/V planes at ±60° from vertical, collection plane
   vertical; wire pitch 0.3 cm; verify sign conventions against `uwire/vwire/ywire`
   stored values on a few events (cheap assertion script).
2. New transform `RecomputeWireProjections` placed after all spatial augs inside the
   MultiView global/local transform lists: recompute the 3 wire-coordinate channels
   from the final rotated coords (linear projection).
3. **Design decision to flag:** wire *coordinates* can be exactly recomputed under
   rotation; the per-plane *charge* (pixval) cannot (it would require re-projecting
   through the original 2D images). P05B.3 as implementable = "rotation-consistent
   wire coordinates as features (+ frozen pixvals)". If the intent was
   rotation-consistent charge, B.4 (plane-summed scalar) is the realizable variant.
4. Adding wire coords to features changes `Collect.feat_keys` and `in_channels` 6→9.

### WP8 — Registry + orchestration (P0.6, P0.1, P0.7, P0.10) *(~1 day)*

1. `exp/registry.csv` + `tools/submit_run.py`: copies the generated config to
   `exp/configs/{run_id}.py`, records hash, appends row on submit; a small
   post-job hook appends status/metrics on completion (§7 schema).
2. **P0.1:** adopt the measured numbers in §1.4. Batch = 48 (DECIDED, §1.4);
   peak LR still open — see §4.1. Whatever is chosen, update plan §0.4 and never
   vary it.
3. **P0.10:** record in the registry that all existing v6/v7/v8 checkpoints used free
   3-axis rotations (§1.3.3) — done as part of the first registry commit.
4. **P0.7 smoke tests:** 500-iter runs of (a) the MC-only ghost-dropped pretrain base,
   (b) the supervised P05A base, (c) one probe config — on Isambard, verifying WP1
   truth loading, WP2 snapshots, and M5 logging in one shot.

### Deferred (not needed for Phase 0.5)

- **P0.5 LArMatch threshold schedule** (only P2C needs it). Implementation note for
  later: the threshold lives in the dataset, which doesn't know training progress;
  plumb it per-epoch via a `set_epoch`-style call from the trainer loop
  (`train.py:177-186`) — per-epoch granularity is sufficient for a 25%-of-training
  ramp. Watch `persistent_workers`: worker copies of the dataset only refresh if
  workers restart each epoch.

---

## 3. Sequencing

```
WP1 filelists ─┬─► WP5 P05 configs ─► WP8.4 smoke tests ─► Wave A launch (Isambard)
WP2 snapshots ─┤                                             │ snapshots accumulate
WP4 ghost_keep_frac (cheap, do with WP1)                     ▼
WP6 diagnostics ─► P05D.1/.2/.3 on EXISTING checkpoints     rsync → Tufts
     (runs now, needs no new training)                       ▼
WP3 probe pipeline (Tufts, build while Wave A trains) ─► curves + M-metrics
WP7 wire projections ─► P05B.3 (can trail the rest of Wave A by days)
WP8 registry — before first Wave A submission
```

Day-one parallelism: WP1+WP4 (one session), WP2 (one session), WP6 P05D diagnostics on
existing checkpoints (immediate science output), WP8 registry. Wave A can launch with
everything except WP7/P05B.3, which follows.

**10-day allocation timeline** (allocation ends ~2026-07-23): pilots (P05A.1 +
P05B.1) must launch by day 2, the rest of Wave A by day 3–4, leaving ~2.5 days of
margin for a Wave C combination run starting ~day 7. The WP3.5 probe-budget
calibration at Tufts runs in parallel from day 1 (uses existing checkpoints). If
Wave C cannot fit, it runs after the allocation as a Tufts-probe-selected single run
whenever GPU time reappears — Wave A results are the deliverable at risk, so protect
those first (priority order in §4.2).

## 4. Open decisions

All resolved as of 2026-07-13 (PI):

1. **Batch 48, peak LR 2e-4** — the NaN-free validated LR; at batch 48 it is a
   slightly higher effective per-sample LR than the stable v8 run. Experiment plan
   §0.4 must be updated (it said 48 / 5e-4; keep batch, replace LR).
2. **P05 budget 5M images**, inside the ~10-day Isambard allocation.
3. **Concurrency: 5+ simultaneous 4-GPU jobs** are sustainable — full Wave A fits
   with margin for Wave C. The §4.2-style priority order is kept in submission
   scripts anyway as insurance: P05A.1, P05A.2, P05B.1, P05B.2, P05C.1–3 →
   P05C.4–6, P05A.3, P05B.4 → P05B.3 (needs WP7), P05A.4.
4. **Pilots: P05A.1 + P05B.1 launch first** — supervised pilot validates in-loop
   val + MC-only dataloader on wandb within hours; SSL pilot exercises the full
   snapshot → rsync → Tufts-probe loop and the snapshot cadence.
5. **Transfer: rsync over ssh**, incremental (snapshots picked up as written).
   WP3 includes a small sync script + drop-directory convention at Tufts.
6. **Snapshot cadence** approved (log-spaced ~12 points, WP2.4), validated on the
   P05B.1 pilot before the fleet relies on it.
7. **P05B.2 policy** = existing physics-consistent `RandomFlipAxis` y/z with u↔v
   charge swap (reviewed correct, §1.5); wire-coord reprojection rewrite deferred
   to WP7 (only P05B.3 needs it).
8. **Eval placement**: in-loop val on Isambard for P05A supervised runs; Tufts
   probes for all SSL runs. Probe budget fixed by the WP3.5 calibration study.
