# Slicer retrain: domain-hardening intervention plan

*Written 2026-08-18, after the domain campaign closed
(DOMAIN_STUDY_RESULTS.md sections 1–24). Companion to
DOMAIN_SHIFT_MEASUREMENT_PLAN.md (measurement tooling) and the m2f-recipe
review (plan file zany-whistling-pine). Status: DRAFT for user review.*

## What the measurements pin down (constraints on the plan)

| finding | consequence for the retrain |
|---|---|
| Slicer collapse is its own (old 0.467 vs v2 0.383 at identical LoRA deghosting; even old leaves ~0.38 of photon charge unclustered on overlay) | deghoster fixes can't rescue it; slicer-specific work required |
| Overlay ≈ data in embedding space; val-sim outlier = EVENT CONTEXT (sim corsika cosmics + sim noise vs real) | overlay files in training = training on the deployment context, with truth |
| Label-free nu-deposit clouds + charge spectra nearly identical val vs ovl | per-point charge augmentation is insurance, not the main lever |
| Thin simch labels: val labels 55% of local cloud vs ovl-2D 70%; misses = thin shell (95% within Δrow≤2, Δwire≤2); shared by ALL sim-trained stages | label completion is a first-class fix and also unifies the two label mechanisms |
| Old slicer robustness plausibly = accidental under-optimization (flat 1e-5 LR, match agreement ~0.1) + randomized 100k cap | keep the m2f recipe fixes but add DELIBERATE regularization; ep1-vs-ep4 probe (job 2513133) quantifies the fit-transfer trade |
| lm composition exonerated at slicer eval (±0.004) | lm-threshold work is deghoster-lane, low priority here |
| Deghoster lane settled: v6-noghosts + LoRA (v6-lantern) | train the slicer cascaded on the ROBUST upstream it will deploy with |
| Eval-side density probe: slicers density-insensitive at eval | density dropout is cheap insurance (training-side untested), not the headline |

## Interventions

### A. Training data (largest expected effect)
- **A1 — Official overlay files in the training mix.** The satfix
  production conversions exist (67k events, mcc9 2D-path truth = the
  inclusive label mechanism; real cosmics + real noise + dead channels =
  the deployment context). Build `h5list_mix_lantern_overlay_train.txt`.
  Starting mixture 50/50; ablate 25/75 later if the cell wins.
  **Hygiene: exclude the 174 pilot signal events (and their files) and
  anything in eval lists from training.**
  **Sample mix (user, 2026-08-18)**: bnbnu overlay + bnbnu-pi0-filtered
  overlay + bnb intrinsic nue overlay + low-E-enhanced nue CC overlay —
  tier2 lists to be provided by user, then transfer + stepA conversion.
  Maintain a checked-in TRAINING-DATA LEDGER: per sample, source list,
  conversion tag, explicit train/eval split; eval reservations excluded
  from every list-builder.
  **SHORTCUT RISK (user)**: in overlay events nu charge is SIM-response
  while cosmics are REAL — "sim-textured => nu" is a learnable shortcut
  that wins on overlay but fails on data (nu is real there). Defenses:
  (i) the sim+overlay MIXTURE itself penalizes the shortcut (pure-sim
  events have sim-textured cosmics); (ii) charge-jitter augs (B2)
  degrade the texture cue; (iii) the DATA-leg gate is incorruptible —
  promoted to REQUIRED for any cell that wins on overlay (C1 alone is
  gameable by this shortcut); (iv) triangle measured data ~ ovl at
  kNN-chance for reco objects => texture signal weak in embedding
  space (weak != absent for a supervised head — defenses stay).
- **A2 — Label completion (both sources).** Spacepoint-level: attach
  unlabeled lm>=0.15 (sim: all-cloud) triplets within ~0.75 cm of a
  particle's labeled points to that particle, at conversion or as a
  sidecar relabel pass. Targets: labeled-fraction >= 0.70 on BOTH
  sources (audit tool exists: `domain_shift/label_definition_audit.py`);
  spot-check purity via charge-weighted contamination. Unifies the
  simch and 2D label mechanisms => mixing A1 sources becomes consistent,
  and the train/eval label mismatch disappears.

### A1-loss — partial-truth handling for overlay events (required for A1)

Overlay GT = {one nu instance}; real cosmics + real ghosts are UNLABELED.
Term-by-term (user question, 2026-08-18):
- VALID and valuable: the nu-mask loss — positives = 2D-truth nu points,
  negatives = real cosmic charge => "nu vs real cosmics" is directly
  supervised on the deployment domain. Matching is trivially stable
  (1 GT).
- BREAKS: (1) no-object CE floods the ~47 unmatched queries, punishing
  correctly-formed cosmic masks => contradictory cosmic-clustering
  supervision vs sim events; (2) cosmic-class head trains sim-only;
  (3) per-event mean-over-pairs concentrates overlay gradient in the nu
  mask (conscious knob, arguably desirable); (4) mask-DN degenerates
  with 1 GT.
- FIXES: (a) **masked no-object**: unmatched queries whose predicted
  mass sits mostly (>0.5) on unlabeled points are EXCLUDED from the
  class loss (ignore, not no-object) — one flag in LArFormerLoss;
  (b) skip/reduce mask-DN on events with < N GT instances;
  (c) upgrade path if cosmic clustering sags: cosmic pseudo-instances
  from the official `image2d_thrumu` product (already in the files) or
  geometric clustering, entering with down-weighted mask loss;
  (d) monitoring: sim-val cosmic-slice metrics tracked separately so a
  cosmic-clustering regression is immediately visible.

### A2-integration — how completed labels enter training (user Q, 2026-08-18)

- **In place, with in-file revert**: the tool rewrites
  trackid/pid/origin/hasmatch in the h5, preserving originals as
  `*_precomplete` (~10% size overhead). => S0-vs-S2 (labels-alone
  ablation) runs off IDENTICAL files via a dataset switch that reads
  the `_precomplete` arrays (small loader option to add). Idempotency
  guard: files carrying `label_completed` are skipped (a second pass
  would cascade-grow labels through adopted donors).
- **Provenance: NOT hasmatch=2** — hasmatch stays binary (adopted -> 1)
  so all existing consumers (HasmatchAsGhost transform, eval batteries)
  keep correct semantics with zero audits; provenance = the
  `label_completed` dataset (0 original / 1 adopted / 255 unlabeled),
  optionally exposed by the loader for per-point loss down-weighting of
  adopted points (default weight 1.0; noise-control lever).
- **Where it runs**: new overlay conversions -> post-step in the
  conversion array (files born complete); already-converted overlay ->
  in-place array; LANTERN sim -> in-place array over the training list.
  DECISION POINT (user): in-place completion MUTATES the shared
  production LANTERN files (reversibly, via _precomplete). Fallbacks
  (copy the training split: ~4-12 TB) look prohibitive; recommend
  in-place after per-sample QA sign-off.

### B. Augmentations (cheap, all recipe-preserving)
- **B1 — Density dropout**: per-event random keep-fraction
  U(0.4, 1.0) applied pre-deghost (the deliberate version of the old
  100k cap). Fresh draw per __getitem__ (the old code's behavior).
- **B2 — Charge jitter**: per-plane scale U(0.9, 1.1) + global gain
  U(0.9, 1.1) on strength/pixval columns.
- **B3 — Dead-wire dropout**: mask k~Poisson wire intervals per plane
  per event (mimics real dead channels; data has 3x more Y-plane
  zero-pixval).
- (B4 noise injection: deferred — riskier, and A1 supplies real noise.)

### C. Optimization / regularization
- **C1 — Overlay-gated checkpoint selection** (mandatory, all cells):
  save every epoch; model_best = best OVERLAY battery in-slice photon
  charge, with val as tiebreak. Never select on val alone again.
- **C2 — Keep the m2f recipe fixes** (OneCycleLR over a realistic
  horizon, no_object_weight 0.1, wd hygiene, 48 queries, per-layer
  matching, vecloss) — the ep1 probe tells us whether to ALSO cap the
  schedule (fewer epochs / lower peak LR) as explicit early stopping.
- **C3 — Weight averaging (EMA or 2-3-checkpoint SWA)** for flatter
  minima; nearly free to add.

### D. Cascade coupling
- **D1 — Robust upstream in the training cascade**: v6-lantern (or
  deployed LoRA) at tau 0.2, matching deployment.
- **D2 — Upstream randomization**: sample the deghost tau per event
  U(0.1, 0.35) during training so the slicer tolerates upstream
  operating-point drift.

## Cell design (staged, not factorial)

Judged at a FIXED short budget (2 epochs ≈ 2 days A100 with vecloss),
epoch-1 checkpoints kept for fit-transfer curves. Battery per cell:
overlay-174 sliceid decomp (in-slice photon charge, the headline gate) +
val twin ceiling + decomposed-cutflow γ-slice step.

| cell | interventions | isolates |
|---|---|---|
| S0 | none (ep4 + ep1 probe rows, section 25) | baseline + fit-transfer trade |
| S1 | A1 (+A2 on both sources) + C1 + D1 | data/context + labels (expected winner) |
| S2 | A2 only (LANTERN relabel) + C1 + D1 | label completion alone |
| S3 | B1+B2+B3 + C1 + D1 | augmentation alone |
| S4 | S1 + S3 (+C3, D2) | full recipe |
| S5 (optional) | recipe-element ablation: 128 queries and/or final-layer matching restored, on S0 data | isolates recipe deltas vs old slicer |

**S0 probe RESULT (2026-08-18, results section 25)**: ep1 val 0.509 /
ovl 0.403 vs ep4 0.558/0.383 — fit-transfer trade real but MODEST
(+0.02 ovl for -0.05 val). Under-optimization does NOT explain the old
slicer's 0.467 transfer => C2 demoted to minor knob; S3 (training-side
density randomization = the old cap) and S5 (recipe elements) carry the
remaining old-advantage hypotheses. v6-lantern+ep4 ovl == hybrid 0.383
(no deghoster windfall, as predicted).

Decision rule: extend the best cell to the full schedule (5 epochs);
if S1 ≈ S4, drop the augmentations from the production recipe (keep
C1/C3 regardless). Stage-3 cache rebuild + segmenter retrain follow the
winning slicer only (same interventions inherited by the cache; stage-3
gets the same battery gate).

## Prerequisites / build order

1. **P1 — label-completion implementation** (blocks S1/S2/S4):
   converter flag or sidecar relabel tool + the A2 VALIDATION SUITE
   (see below) on ~20 files per source.
2. **P2 — overlay training data** (blocks S1/S4). STATUS 2026-08-18:
   - Tier2 lists provided (8 samples, ~36k dlmerged:
     `uboone_official/overlay_tier2_lists/`). LEDGER BUILT
     (`uboone_official/training_data_ledger/`): deterministic md5-parity
     50/50 TRAINPOOL/RESERVED split per sample (user rule: <=half of
     each sample trains; rest reserved for expectation vs beam data);
     767 run3b_bnb_nu trainpool files matching the satfix-1500 eval
     production moved to EVAL_LOCKED (never train).
   - SMOKE PASSED on all 8 samples: full product set present (incl.
     v40a CCpi0/NCpi0 dlana — truth + raw inputs intact, old-reco
     extras ignored); stepA --mcc9 conversion + truth content validated
     on the two riskiest samples. Zero-label entries = genuine dirt
     events (nu vertex outside TPC) — list-builder must require a
     minimum labeled-point count.
   - PIPELINE (disk-constrained, user): process ONE SAMPLE AT A TIME —
     stage tier2 -> mcc9_scratch batch, stepA convert (--adc wire -tb
     --mcc9; NO larmatch pass per the no-lm policy; attachable later),
     verify counts, DELETE staged copies, next batch. Batch size sized
     to scratch headroom.
3. **P3 — augmentation transforms** (blocks S3/S4): B1 exists in spirit
   (max_spacepoints machinery); B2/B3 are small new transforms; smoke on
   the dev sample.
4. **P4 — battery runner** consolidation: one sbatch that takes a
   checkpoint and emits the three gate numbers (mostly exists —
   assemble from valchain_ceiling.sh / oldslicer_control.sh pieces).
5. Pending input: ep1-vs-ep4 probe result (job 2513133) — calibrates C2
   and expectation for S0 row.

## A2 validation suite (label expansion QA; user-requested 2026-08-18)

Before/after, on ~20 val-sim files + ~20 overlay files:
1. **Event displays** (user request): 3D + per-plane crops colored by
   hasmatch and by trackid, three point classes rendered distinctly —
   originally-labeled / newly-adopted / still-unlabeled — for a sample
   of photons and muons.
2. **Per-particle dedup-charge completeness** (the user-remembered
   ~0.8-coverage style metric; cf. campaign stage-1 gamma completeness
   0.807): labeled charge / (labeled + unlabeled-local-cloud charge).
   Baseline val 0.55 (count-frac); TARGET >= overlay's 0.70; overshoot
   >> 0.70 with a fixed radius = suspicion of over-attachment.
3. **Overlay-as-calibration null**: run the SAME expansion on overlay
   (2D labels, already inclusive) — the added fraction there bounds the
   noise floor of the procedure; a radius that adds a lot on overlay is
   too big.
4. **Purity guards on adopted points**: (a) ssnet-class consistency
   with the donor particle (mismatch rate); (b) ambiguity rate = points
   claimed by >=2 particles' expansions (resolve nearest-donor; report);
   (c) offset-shell profile — adopted points should live in the
   measured thin shell (drow<=2, dwire<=2, section 23), tail beyond =
   suspicious; (d) pixval spectrum of adopted points should look like
   charge periphery (low-amplitude real charge), not like the ghost
   spectrum.
5. **Truth-parity density recheck** (`truth_density_parity.py` +
   `label_definition_audit.py` rerun with expanded labels): val labeled
   NN-spacing should converge to the local-cloud 0.358 cm; the
   labeled/unlabeled charge-bias ratio (1.43) should shrink toward 1;
   labeled_frac 0.547 -> ~0.70.
6. **Functional (optional, = cell S2)**: quick LoRA deghoster retrain
   on expanded labels -> two-domain battery.

## Policy decision: jettison larmatch thresholding (user-proposed;
## measurements support it)

Future training data drops the lm>=0.15 file-level cut (a bootstrap-era
dependency on an old network). Evidence: eval-side composition
exonerated at both domain-sensitive stages (slicer ±0.004, deghoster
+0.01-0.06); chain inference already runs uncut; training-side pre-cut
largely exonerated for the deghoster (v6-lantern 0.911 transfer) while
UNCUT-trained deployed LoRA is mildly better at tight operating points
(easy-ghost margin) — uncut helps if anything; labels never needed lm;
the deghoster is the in-house replacement for larmatch's filtering
role. Costs: 2-4x larger clouds (storage/compute; cap machinery
absorbs), label-completion attaches over the full cloud instead of the
lm-cut cloud. SSL lineage already lm-free (v6-noghosts used sim labels).
Consequence: all NEW conversions use the stepA path (validated section
9: v3 = strict subset of stepA cloud) with no lm cut; lm-related
dataset knobs retired from future configs.

## Results & decisions log (rolling; newest last)

- **2026-08-18 ledger + smoke**: 8 tier2 samples listed (~36k dlmerged);
  md5-parity 50/50 train/reserved split; 767 satfix-eval files
  EVAL_LOCKED. All samples carry full products (incl. v40a dlana);
  conversion + truth validated; dirt events (nu outside TPC) label zero
  points — list-builder must set a minimum labeled-point count.
- **2026-08-18 tier2 gotcha**: tier2 NOT mounted on compute nodes —
  pipeline split into login-side `stage_overlay_train_batch.sh` +
  compute array `submit_overlay_train_convert.sh` (tasks delete their
  own staged file).
- **2026-08-18 A2 operating point LOCKED** (jobs 2513926/2514043):
  r=0.5 cm, shell ±2 — adopted ghost-frac 5.1% (real-lm join; the
  floor: ±1 variants same purity, −5 pts completeness), val
  completeness 0.647→0.895, ovl null 0.759→0.968, net label noise
  ~1.3%. ssnet metric retired (false alarm — SSNet class 0 on faint
  periphery). Displays: `NTUP/label_expansion_qa/` (r0.75 set) +
  `displays_r0.5_s2/` (production point).
- **2026-08-18 LowE run1 shakedown**: conversion array 2514224 running;
  first 38 files → 1207 h5s (~32 entries/file vs the smoke file's 4 —
  per-file entry counts are highly variable).
- **2026-08-18 mixture DECIDED: 50/50** (user; storage-limited) —
  convert all physics-enriched samples in full + bnb_nu subsample to
  ~410k overlay events (~2 TB).
- **2026-08-18 scratch cleanup**: deleted mcc9_scratch copies of
  run3b_bnb_nu_overlay_nocrtremerge (3.8T) + bnb5e19 (765G) after
  tier2 verification (exact counts + 20/20 basename spot-checks);
  retransfer from tier2 when reprocessing is needed (e.g. bnb5e19
  stepA0 reruns; inputlists point at scratch paths until then).
  **extbnb_dlreco (1.4T) HELD**: tier2 only has extbnb_dlana
  (different stage, different files) — no verified dlreco backup;
  awaiting user confirmation before deletion.
- **2026-08-18 LowE run1 conversion VERIFIED**: 292/292 tasks
  completed, 292/292 filenos with output, **12,292 events** (~42
  entries/file), staging cleaned, schema validated. First sample DONE
  + label-completed in place (array 2518801, 60/60 spot-check).
- **2026-08-18 in-place completion APPROVED (user)** — conversion array
  now runs the A2 post-step (files born complete); generic catch-up
  array `submit_complete_labels_array.sh` for pre-existing files.
- **2026-08-19 run3b conversions + a DATA FINDING**: run3b nue LowE:
  287/287 filenos, **12,794 events** (2 preempted filenos rerun;
  catch-up completion array 2557675). run3b intrinsic nue: **~32% of
  the production's files (533/1662 trainpool) have NO MC-truth trees**
  (confirmed at tier2 source; mixed within subdirs; the one-file smoke
  missed it) -> ledgered as `*_NO_TRUTH.txt`, excluded; 6 interrupted +
  6 converter-crash (rc=129, partial output) filenos retried (job
  2557660; crashers kept-partial if deterministic). Usable yield ~1123
  files / **~40k events**. CAVEAT for the user: the RESERVED half
  presumably has the same ~32% truthless rate — matters if the
  expectation-building use needs truth; and earlier nue-CC inputlists
  were likely pre-filtered to truth-carrying files.
  Next three samples queued: v28 nueintrinsics (1986), CCpi0 (392),
  NCpi0 (840).
- **2026-08-18 hang + straggler round**: the user-flagged 2h42m job
  (2559648_1694) = converter finished all 50 entries then hit
  "double free or corruption" in teardown and wedged in ROOT's abort
  handler — killed; all 50 h5s verified readable; labels completed
  manually. Straggler classification across the 3 new arrays + nue
  retry: run3b nue fileno 278 = ALSO TRUTHLESS (interrupted first
  attempt hid it; ledgered — NO_TRUTH now 534); crashers
  39/563/708/1127/1421 failed identically on retry -> KEEP-PARTIAL
  (ledgered in *_CRASH_PARTIAL_filenos.txt). Retry arrays
  2602165/2602195/2602208/2602214 cover 22+9+6+2 preempted/crashed/
  unclassified filenos. NOTE: rc=129 teardown crashes produce VALID
  partial h5s (work done, exit dirty) — a converter teardown bug worth
  a hygiene fix eventually, harmless to outputs.

### Mixture arithmetic (rough, pre-conversion — entries/file variable)

LANTERN (new-sim) train list: **410k events**. Overlay TRAINPOOL
estimate from smoke/early entry counts (BEFORE dirt filtering, which
costs ~20-30% on the generic bnbnu samples):

| sample | files | ~entries/file | ~events |
|---|---|---|---|
| v28 run1 bnboverlay | 4740 | ~50 | ~237k |
| v28 run1 nueintrinsics | 1986 | ~9 | ~18k |
| run1 nue LowE | 292 | ~32 (measured) | ~9k |
| run3b intrinsic nue | 1662 | ~22 | ~37k |
| run3b nue LowE | 287 | ~1-30 (?) | ~3-9k |
| run3b bnb_nu | 7059 | ~60 | ~424k |
| v40a CCpi0 | 392 | ~61 | ~24k |
| v40a NCpi0 | 840 | ~63 | ~53k |
| **total** | | | **~800k** |

=> full overlay trainpool : new-sim ≈ **2 : 1** — comparable scale, not
dwarfing; any mixture from 25/75 to 65/35 is reachable by subsampling
overlay at list-build. DECISION POINT (storage): converting the FULL
trainpool ≈ 4 TB at ~5 MB/event (bnb_nu alone ~2 TB). Alternative:
convert only what the mixture needs — all physics-enriched samples
(nue + pi0 + LowE ≈ 140k events) + a bnb_nu subsample (~270k events,
~4.5k files) reaches 410k overlay = 50/50 vs LANTERN at ~2 TB. Exact
numbers per sample firm up as each conversion completes; ledger records
what converted.

- **2026-08-19 overlay corpus tally + DECISION POINT (storage)**:
  enriched samples DONE = **204,184 events / 3.4 TB** (v28
  nueintrinsics 81k, run3b nue 40k, NCpi0 41k, CCpi0 17k, LowE 2x
  ~12.5k) — events 1.5x the estimate but ~17 MB/event (vs 5 est.), and
  disk is at 93% (2.1 TB free): v28 bnboverlay (~4+ TB) and the bnb_nu
  top-up DO NOT FIT. **Proposed: stop overlay conversion here; reach
  50/50 by subsampling LANTERN to ~204k at list-build (zero disk).**
  Rationale: every enriched event already carries the full real-cosmic/
  noise context (the domain ingredient); enriched nu composition suits
  the physics targets; generic-bnb adds topology composition only —
  addable later as a ~20-30k-event slice (~400 GB) if the S1 battery
  suggests composition matters and space is freed (extbnb decision).
  AWAITING USER SIGN-OFF.
- **2026-08-19 conversion CLOSED-OUT**: retry round done; every
  remaining failure is a deterministic teardown/mid-file crash with
  VALID partial output (19 filenos, ledgered in
  `*_CRASH_PARTIAL_filenos.txt`) except v28 fileno 1081 = single-entry
  NOISE event vetoed at the 5M-triplet cap (legitimate; ledgered).
  Coverage: v28 nueintrinsics 1985/1986, CCpi0 392/392, NCpi0 840/840,
  run3b nue 1128/1128 usable, LowE 287+292 complete. **Corpus:
  ~205.3k events / 3.4 TB.** Final idempotent completion sweep running
  over all samples (jobs 2614117-2614257); staging dirs cleaned.

## Open questions for review

- Overlay mixture fraction (50/50 default) and whether to weight
  overlay events by POT/spill realism.
- ~~Label-completion radius~~ RESOLVED by ablation (2026-08-18, jobs
  2513926/2514043, `NTUP/label_expansion_qa/`): **r=0.5 cm, shell ±2**
  — reaches the ~5% adopted-ghost floor (real-lm join; ±1 variants same
  purity, −5 pts completeness; r=0.75 is 7.6% ghost) with val
  completeness 0.647→0.895, ovl 0.759→0.968. Net added label noise
  ~1.3% of final labeled set vs curing the 30-45% periphery exclusion.
  Guard rejects a 46%-ghost population (6-9x purifier). ssnet metric
  retired (SSNet leaves faint periphery at class 0 — false alarm);
  the real-lm join is the purity instrument. Displays:
  `NTUP/label_expansion_qa/display_*.png`. Tool default now 0.5.
- Whether S1 should also drop the corsika LANTERN files entirely
  (overlay-only training) as an S1b variant — cheap to add if S1 wins.
- ~~Data-leg gate: adopt now or wait~~ RESOLVED (2026-08-18): REQUIRED
  for any cell that wins on overlay, as the anti-shortcut guard (see
  A1). Form: embedding battery on the 261 bnb5e19 flash-cut candidates
  (manifests + tooling exist) and/or selection-rate consistency.

- **2026-08-19 P2 OVERLAY SIDE COMPLETE**: 205,341 events / 3.4 TB,
  six samples, all label-completed (sweep 150/150; 50/50 spot-checks
  everywhere). Remaining before S1: (a) user sign-off on stop-here +
  subsample-LANTERN-to-205k for 50/50; (b) LANTERN in-place completion
  pass over the chosen training subset; (c) list-builder (min-labeled
  filter, ledger exclusions, mixture weights); (d) masked-no-object
  loss flag + DN guard (A1-loss); (e) battery-runner consolidation.

- **2026-08-19 mixture EXECUTED (user-approved)**: LANTERN subsample =
  ALL non-generic files (nue 96,836 + pi0filter 54,808 + chargedpiplus
  68,152 = 219,796; the enriched samples alone exceed the 205,341
  overlay corpus, so ALL generic prod2/set2 files removed per the
  "remove from bnb nu" rule) -> 48/52 overlay/sim, both sides enriched.
  Ledger: lantern_train_subsample_enriched.txt / _removed_generic.txt.
  DEFERRED (user): process remaining overlay (bnboverlay + bnb_nu) for
  a max-stat 50/50 later, when storage allows. LANTERN in-place
  completion: TRAIN SUBSAMPLE ONLY (val/eval files untouched to
  preserve campaign baselines) — array 2618348. Overlay dirt-filter
  scan (labeled counts per file) — array 2618423.
- **2026-08-19 A1-loss IMPLEMENTED + unit-verified**: LArFormerLoss
  `masked_no_object` flag (losses.py) — unmatched queries whose mask
  mass is ENRICHED on unlabeled points beyond the event's base rate
  (bar = max(cfg, (1+base)/2); the naive fixed threshold would suppress
  no-object on ~95%-unlabeled overlay events — caught in unit test) are
  excluded from the class CE; exact weighted-CE normalization; engage
  gate skips fully-labeled events (bit-exact baseline). DN guard:
  MaskDenoiser `min_gt_instances` (query_denoising.py) skips DN groups
  on partial-truth events. Unit test passes (flag-off bit-exact;
  exclusion exact; gate exact). Set `masked_no_object=True` +
  `min_gt_instances=2` in the S1/S4 cell configs.
- **2026-08-19 list-builder + battery runner DONE**: overlay dirt scan
  complete (205,341 files; 91% pass the >=200-labeled-point floor) ->
  `overlay_train_filtered_minlab200.txt` (186,529). MIX v1 =
  `h5list_mix_enriched_train_v1.txt`: 186,529 overlay + 219,796 LANTERN
  enriched = **406,325 events, 46/54 overlay/sim**, seed-42 shuffle
  (epoch size ~= the original 410k). Battery runner:
  `NTUP/run_slicer_battery.sh` (env SLICER_CKPT+LABEL -> val + overlay
  in-slice photon charge vs ep4/old references) with env-driven cascade
  config `larformer-fullcascade-v6lantern-envslicer-tau020.py`.
  REMAINING before S1 launch: LANTERN completion array 2618348 drains;
  S1 training config (mix list + masked_no_object=True +
  min_gt_instances=2 + v6-lantern cascade upstream + C2 recipe).
- **2026-08-19 quota incident RESOLVED + prerequisites CLOSED**:
  extbnb_dlreco deleted after tier2 verification at the user-provided
  nested path (50,410/50,410) -> volume unwedged; damage = only 5
  quota-truncated LANTERN files (excluded; lists now 219,791 /
  406,320). LANTERN completion **100%** (resume array + 1 manual
  straggler; tool now fault-tolerant + pipefail + gzip writes).
  h5repack reclamation campaign running over the 195k first-pass files
  (job 2624271; preserves source filters, gzips labels, verifies +
  atomic-replaces; ~9 MB/file => ~1.7 TB projected).
  **CELL S1 CONFIG WRITTEN**: `stage2_slicer/
  larformer-slicer-s1-mixenriched-v1.py` (MIX v1 list + masked_no_object
  + DN min_gt=2 + v6-lantern cascade deghoster; m2f-v2 recipe
  inherited). Smoke job 2624746 on a 12-overlay+8-LANTERN dev list.
- **2026-08-19 CELL S1 LAUNCHED** (job 2625015, 4xA100 self-chaining
  48h windows, `submit_larformer_slicer_s1_a100.sh`; STOP via
  `exp/larformer_slicer_s1_mixenriched_v1/STOP_AUTORESUBMIT`). Smoke
  passed after a nesting fix (loss_kwargs/mask_denoising live under
  model.slicer): 5 mixed-batch iters, all terms finite, DN active,
  n_gt 9-12 across the overlay/LANTERN mix. NOTE: the base recipe
  already jitters the cascade deghost tau (~0.5±0.01 in logs) — a D2
  variant inherited for free, same for S0's ep4 (fair comparison).
  Judgment: run_slicer_battery.sh on epoch_1/epoch_2 as they land
  (~22h/epoch expected); refs val 0.558 / ovl 0.383 (ep4), old 0.467.
- **2026-08-19 repack campaign COMPLETE**: 200/200 chunks, 0 errors,
  **2.18 TB reclaimed**; volume at 3.4 TB free (89%) — more headroom
  than before the incident. Data-side work fully closed; S1 training
  (2625015) is the only campaign job running.
- **2026-08-19 S1 RESTARTED with corrected tau band (user catch)**: the
  inherited train-time deghost tau U(0.4,0.6)/val 0.5 centered the OLD
  chain's operating point, leaving the v2-era deployment tau=0.2 OUT of
  the training domain. S1 config now U(0.15,0.60)/val 0.20 (D2 done
  properly); config merge verified (all interventions + inherited
  recipe knobs). ~30% of epoch 1 discarded; fresh chain job 2638469.
  NOTE for S5/recipe ablations: S0/ep4 trained at U(0.4,0.6) — its
  battery-at-0.2 numbers stand (measured), but tau-band is now a
  DELIBERATE delta of S1 vs S0 alongside data/labels/loss.
- **2026-08-20 valprobe diagnostic gap EXPLAINED (user observation)**:
  S1's train-vs-valprobe gap in diag_mask_iou_matched /
  diag_mask_bce_rand / diag_dice_rand is CONFIGURATION, not
  overfitting — the rand diags are stationary as designed, but the
  REFERENCE truth differs: train GT = completed labels, valprobe GT =
  untouched thin labels (baseline-preservation choice) => IoU ceiling
  vs thin GT ~0.65-0.7 even for perfect completed-mask predictions;
  periphery counts as FP in val bce/dice. Secondary asymmetries: val
  tau fixed 0.20 vs train U(0.15,0.60) (deliberate), val cap 450k /
  pure-LANTERN composition. Nice inadvertent demo of the section-22/23
  mechanism (label-definition change masquerading as performance).
  FIX: completed-COPY val list (originals untouched):
  `_lantern_val_completed/` 1,500 files + ledger list
  `h5list_lantern_val_completed_1500.txt` (array 2647642) — use in
  future cell configs / offline per-checkpoint diags; the running S1's
  wandb valprobe keeps the thin reference (interpret accordingly).
- **2026-08-20 S1 EPOCH-1 BATTERY + PURITY FORENSICS** (jobs 2670319 /
  2670431 / 2670442; battery files unexpanded => same thin ruler as all
  reference numbers; config bug fixed: no module-level `import os` in a
  `_base_`-inherited config — Config deepcopy cannot pickle modules).
  Photon charge in nu slice: VAL 0.558 -> **0.749**, OVL 0.383 ->
  **0.813** (old-slicer ceiling 0.467; domain gap INVERTS, ovl > val).
  Recall-only metric, so purity measured too:
  | cell | nu-slice/all-event charge | purity(true-nu) | cosmic-origin | no-truth | pts/ev |
  |---|---|---|---|---|---|
  | VAL ep4 | 0.034 | 0.807 | 0.020 | 0.172 | 4029 |
  | VAL S1ep1 | 0.054 | 0.678 | 0.031 | 0.291 | 7162 |
  | OVL ep4 | 0.026 | 0.871 | 0.000 | 0.129 | 3510 |
  | OVL S1ep1 | 0.061 | 0.736 | 0.000 | 0.264 | 8906 |
  Slice ~2x bigger. FORENSICS on the ADDED charge (vs ep4 slice):
  VAL 85.3% periphery(<1cm of true nu) / 6.5% far / 8.3% TRUE COSMIC;
  OVL 75.1% periphery / 24.9% far / 0% "cosmic-origin".
  **CAVEAT (corrects an earlier reading): on OVERLAY the cosmics are
  REAL DATA and carry no truth, so origin==2 can never fire there — the
  data-cosmic population lives inside "no-truth". The 24.9% far-added
  charge on overlay is therefore most likely real cosmic contamination
  (~14% of the S1 overlay slice), not benign ghosts.**
  READING: dominant effect = periphery recovery (the intended A2 fix),
  with a real secondary over-inclusion term. The recall-only gate is
  now insufficient on its own — C1 needs a purity companion.
  NEXT: full-chain CC1pi0 cutflow on S1 ep1 (purity-sensitive steps:
  >=20%-charge instance, class, attachment) as the physics arbiter;
  epoch 2 lands ~22h.
- **2026-08-20 S1-ep1 FULL-CHAIN CUTFLOWS** (job 2670507; cascade =
  v6-lantern deghoster + S1-ep1 slicer + **v2 m2f stage-3** (unchanged,
  trained on a v2-SLICER cache => mismatched to S1's ~2x larger slices)
  + old attempt-2 keypoint).
  SELECTION (pre-flash eff): VAL 0.487 (v2 0.490 / old 0.398);
  **OVL 0.333** (v2 0.322 / old 0.402).
  TRUTH-MATCHED: VAL 0.450 (v2 0.440 / old 0.300);
  **OVL 0.310 — 2x the v2 chain (0.155) and ABOVE deployed-old (0.282)**.
  Overlay step survivals (S1 / old / v2):
    gdeghost 100% / 97% / 85%
    gslice    95% / 82% / 68%   <- photon delivery FIXED
    gfound    80% / 75% / 53%   <- best measured on overlay
    gID       98% / 98% / 98%
    gcut      79% / 80% / 88%
    mufound   96% / 99% / 91%
    **muID    75% / 99% / 88%   <- NEW REGRESSION**
    mucut     79% / 78% / 73%
  READING: the photon lane is fixed (every photon step now best-ever on
  overlay, domain gap 0.310/0.450 = 0.69 vs v2's 0.35), but a NEW MUON
  PID failure appeared — 24 events lost where the muon instance exists
  with >=20% charge but is not classed mu. Prime suspect = the stage-3
  input mismatch flagged above (fatter, periphery-inclusive slices are
  out-of-domain for a segmenter cached on v2 slices); it also explains
  why the SELECTION eff (0.333) lags old-chain (0.402) despite better
  photons. => the planned stage-3 cache rebuild + segmenter retrain on
  the winning slicer is now the critical path, not optional. Check
  first: muon-instance contamination / class-score shift under S1
  slices before assuming a slicer defect.
- **2026-08-21 S1-ep2 EVALS** (battery 2727605, full chain 2727606).
  Photon charge in nu slice: VAL 0.749->**0.787**, OVL 0.813->**0.826**
  (refs: ep4 0.558/0.383, old 0.470/0.467). Still improving, and
  SHARPENING not just inflating: val photons losing >50% charge to a
  cosmic slice halved 16/400 -> 8/400; val cosmic-slice frac 0.027 ->
  0.024; unclustered VAL 0.144->0.109, OVL 0.041->0.028.
  Cutflows (ep1 -> ep2): truth-matched VAL 0.450 -> **0.500** (new best;
  v2 0.440, old 0.300); OVL 0.310 -> 0.310 (flat; v2 0.155, old 0.282).
  Selection VAL 0.487 -> 0.520; OVL 0.333 -> 0.328.
  Overlay steps ep2: gslice 97%, gfound 80%, gID 95%, **gcut 71%**
  (ep1 79%), mufound 96%, **muID 83%** (ep1 75%), mucut 77%.
  **KEY NEW FACT: muID is 98% on VAL but 83% on OVERLAY with the SAME
  slicer + SAME segmenter** => the muon-PID failure is DOMAIN-SPECIFIC,
  not merely an input-distribution mismatch from fatter slices. The v2
  stage-3 already sat at 88% overlay muID before S1 (old stage-3: 99%),
  so stage-3 carries its own overlay domain weakness; S1's slices
  (which on overlay include ~14% far/real-cosmic material) add to it.
  => With the photon lane fixed, THE OVERLAY BOTTLENECK IS NOW STAGE-3.
  Remedy = the cache rebuild + segmenter retrain on S1 slices over the
  MIXED corpus with completed labels (same medicine that fixed the
  slicer). Diagnostic to run first: fraction of the reco MUON instance
  charge that is not true-nu-origin, overlay S1 vs ep4.

## S1 results: full cutflow tables vs the reference chains

*(mirrored from DOMAIN_STUDY_RESULTS.md section 26; jobs 2670507 (ep1)
and 2727606 (ep2))*

Cascade for the S1 columns: v6-lantern deghoster @tau0.2 + S1 slicer
(mix-enriched + completed labels + masked-no-object) + **unchanged v2
m2f stage-3** + old attempt-2 keypoint. All columns share the same thin
(unexpanded) truth ruler and the same gslice/gfound bar (0.2).

### Truth-matched (decomposed) cutflow — OVERLAY, 174 CC1pi0 signal

| step (survival vs prev) | old | v2 | S1 ep1 | S1 ep2 |
|---|---|---|---|---|
| reco'd | 169 | 173 | 173 | 173 |
| vtx in FV | 167 | 171 | 168 | 171 |
| γ deghost (>20% survives) | 162 (97%) | 146 (85%) | 168 (100%) | 171 (100%) |
| γ slice (>20% in nu slice) | 133 (82%) | 100 (68%) | 160 (95%) | **166 (97%)** |
| γ found (≥20% instance) | 100 (75%) | 53 (53%) | 128 (80%) | **133 (80%)** |
| γ ID | 98 (98%) | 52 (98%) | 126 (98%) | 127 (95%) |
| γ cut (≥2 conf attached) | 78 (80%) | 46 (88%) | 99 (79%) | 90 (71%) |
| μ found | 68 (87%) | 42 (91%) | 95 (96%) | 86 (96%) |
| **μ ID** | 67 (**99%**) | 37 (88%) | 71 (**75%**) | 71 (**83%**) |
| μ cut | 52 (78%) | 27 (73%) | 56 (79%) | 55 (77%) |
| + cπ veto + mγγ | 49 | 27 | 54 | 54 |
| **eff (truth-matched)** | **0.282** | **0.155** | **0.310** | **0.310** |

### Truth-matched cutflow — VAL twin, 200 CC1pi0 signal

| step | old | v2 | S1 ep1 | S1 ep2 |
|---|---|---|---|---|
| vtx in FV | 193 | 197 | 197 | 198 |
| γ deghost | 189 (98%) | 197 (100%) | 197 (100%) | 198 (100%) |
| γ slice | 148 (78%) | 179 (91%) | 179 (91%) | 189 (95%) |
| γ found | 122 (82%) | 142 (79%) | 146 (82%) | 155 (82%) |
| γ ID | 118 (97%) | 140 (99%) | 141 (97%) | 152 (98%) |
| γ cut | 96 (81%) | 117 (84%) | 117 (83%) | 129 (85%) |
| μ found | 83 (86%) | 114 (97%) | 113 (97%) | 124 (96%) |
| μ ID | 78 (94%) | 112 (98%) | 111 (98%) | 121 (98%) |
| μ cut | 62 (79%) | 94 (84%) | 95 (86%) | 103 (85%) |
| + cπ + mγγ | 60 | 88 | 90 | 100 |
| **eff** | **0.300** | **0.440** | **0.450** | **0.500** |

### Selection (light) cutflow — pre-flash efficiency

| cell | reco'd | vtxFV | ≥2γ | +μ | +0cπ | +mγγ | eff |
|---|---|---|---|---|---|---|---|
| VAL old | 196 | 193 | 118 | 81 | 78 | 78 | 0.398 |
| VAL v2 | 199 | 197 | 129 | 104 | 98 | 98 | 0.490 |
| VAL S1 ep1 | 199 | 197 | 126 | 103 | 97 | 97 | 0.487 |
| VAL S1 ep2 | 200 | 198 | 134 | 107 | 104 | 104 | **0.520** |
| OVL old | 169 | 167 | 101 | 73 | 70 | 70 | **0.402** |
| OVL v2 | 173 | 171 | 89 | 58 | 56 | 56 | 0.322 |
| OVL S1 ep1 | 173 | 168 | 107 | 60 | 58 | 58 | 0.333 |
| OVL S1 ep2 | 173 | 171 | 98 | 59 | 57 | 57 | 0.328 |

Readings:
- **Photon lane fixed on overlay**: γ-slice 68% (v2) -> 97% (S1 ep2),
  γ-found 53% -> 80% — both best measured, exceeding even the old chain
  (82% / 75%). Truth-matched eff doubles v2 and passes deployed-old.
- **Domain gap narrowed**: overlay/val truth-matched ratio 0.35 (v2) ->
  0.62 (S1 ep2); old chain was 0.94 but at a much lower ceiling.
- **The bottleneck moved to STAGE-3 (unchanged in these runs)**:
  μ-ID 98% on VAL vs 83% on OVERLAY with the SAME slicer and segmenter
  => domain-specific, not an artifact of larger slices. The v2 stage-3
  was already 88% overlay (old stage-3: 99%).
- ep1->ep2 on overlay: delivery improved (γ-slice 95->97%, γ-found
  128->133) but γ-cut fell 79->71% and γ-ID 98->95%, so eff stayed
  0.310 — losses migrated downstream into stage-3 / attachment.
- SELECTION eff on overlay (0.328) still trails old (0.402) because the
  plain selection can pass with wrong objects; the truth-matched metric
  (which does not credit those) is where S1 wins.
- **2026-08-21 STAGE-3 CACHE REBUILD LAUNCHED (ep2 pilot, user-approved
  plan)**: pinned cascade larformer-fullcascade-s1ep2-tau020.py
  (v6-lantern ep25 + S1 slicer epoch_2); corpus = MIX v1 (406,320);
  val = completed-copy 1500. Smoke 2742054 PASSED incl. the overlay
  GT-instance path (n_gt=3, 11.4k cached pts, tau=0.2 stamped).
  Cache root larformer_cache_stage12__s1ep2_v6lantern_tau020/;
  train array 2742177 (26 shards), val 2742178. Old v2 cache (189G) KEPT until
  first retrain results. Next after drain: particle_class_id augment ->
  verify -> stage-3 retrain config (m2frecipe + new CACHE_ROOT +
  masked_no_object + min_gt=2), gated by the full-chain battery with
  mu-ID as the success criterion (83% -> toward 99%).
- **2026-08-23 STAGE-3 SEGMENTER RETRAIN LAUNCHED** (job 2828977,
  submit_larformer_particle_s1cache_a100.sh, self-chaining): config
  larformer-particle-s1cache-m2frecipe.py = m2frecipe recipe + S1 cache
  (VERIFIED: 100% coverage both splits, class-id augmented, 315G) +
  masked_no_object + DN min_gt=2; val = completed-copy 1500. Cache
  smoke passed (mixed 200 overlay + 200 LANTERN batches; exit-1 was
  the evaluate=False model_best teardown artifact only, 9.2GB peak).
  8-epoch OneCycle. Judgment: full-chain battery per epoch with
  **overlay mu-ID (83% -> toward 99%)** as the pre-registered success
  criterion at preserved photon-lane performance; old v2 cache + v2
  stage-3 kept untouched for direct comparison.
- **2026-08-23 STAGE-3 ep1 PREVIEW (job 2830048; S1 slicer ep2 + NEW
  s1cache stage-3 ep1)**: truth-matched VAL 0.380 / **OVL 0.345** —
  overlay BEST EVER (old 0.282, v2 0.155, S1+v2stage3 0.310) and the
  chain is now nearly DOMAIN-FLAT: ovl/val ratio **0.91** (v2 0.35,
  S1+v2stage3 0.62, old 0.94-at-low-ceiling). Steps (ovl, vs v2-stage3
  run): mu-ID 83%->**90%** (val 99%; criterion on track at ep1/8);
  gamma-found 80%->**94%** (segmenter matched to S1 slices);
  gamma-ID dropped to ~80-82% on BOTH domains (was 95-98%) —
  domain-symmetric => undertrained cls head at ep1/8, not domain; this
  is why val fell 0.500->0.380. Projection if gamma-ID recovers with
  epochs at held mu-ID: ovl ~0.42-0.46 truth-matched, near
  domain-flat. Next preview at stage-3 ep3-4 (post-maintenance).

### 2026-08-25 — Cache muon-composition scan (job 2837740): muon dilution CONFIRMED

Question (user, 2026-08-25): stage-3 val muon class-accuracy is flat at ~85% while
everything else improves — is the nue/pi0-enriched mix diluting muon representation?

Method: 2,500 randomly sampled train cache events per cache, ground truth via
per-point PDG (`entry_0/pid`, |pid|=13) — no class-convention assumptions.
(First scan attempt 2837717 read a wrong layout and returned zeros; layout is
`entry_0/` per-point datasets + `entry_0/particle_instances/`.)

| metric | v2 LANTERN cache | S1 mix cache | change |
|---|---|---|---|
| events containing a muon | 59.6% | 34.8% | −42% rel |
| muon share of cached points | 25.0% | 10.8% | −57% rel |
| muon share of labeled instances (unique trackids) | 16.1% | 9.3% | −42% rel |

Cross-check: particle_class_id 2 is muon (1.39M/1.41M of |pid|=13 points map to
class 2 in the S1 cache; same in v2). The residual −1 on muon points is the
unlabeled bucket.

Verdict: the MIX v1 corpus (46% overlay, nue/pi0-enriched) roughly halves muon
exposure vs the v2 LANTERN cache on every measure. Consistent with the flat 85%
muon val accuracy and the ep1-preview overlay mu-ID at 90% (vs old-stage3 99%).
Remedies on the table for the next training round (user, 2026-08-25): augment
muon-containing events in the training mix, and refresh the cache with a later
slicer epoch. Await ep7 full-chain eval (2837731) for the mu-ID trajectory
before deciding.

### 2026-08-25 — Stage-3 s1cache epoch-7 full-chain eval (job 2837731, label s1ep2p7): BEST EVER BOTH DOMAINS

Chain: v6-lantern deghoster ep25 -> S1 slicer ep2 -> s1cache-m2frecipe segmenter ep7
-> old kp2. CC1pi0 pilot samples (200 val / 174 ovl truth-matched sig).

Truth-matched eff: VAL 0.585 / OVL 0.546 (ratio 0.93 — domain-flat).
Light selection: VAL 0.595 / OVL 0.552 pre-flash.
References: old 0.300/0.282; v2 0.440/0.155; S1+v2stage3 0.500/0.310; ep1 preview 0.380/0.345.

Per-stage conditional rates (VAL / OVL):
  g-found 96.8% / 98.2%; g-ID 94.5% / 96.3% (ep1's ~80-82% sag fully recovered,
  as projected — undertrained head); g-cut 83.2% / 80.4%;
  mu-found 97.9% / 96.9%; mu-ID 98.6% / 95.1% (ep1: 99/90; old-stage3 ovl ref 99%);
  mu-cut 88.5% / 82.1%; cpi-veto ~95-99%.

Read vs the muon-dilution finding (same day, job 2837740): despite halved muon
exposure in the MIX cache, mu-ID recovered to 98.6/95.1 by ep7 — dilution slowed
but did not cap muon learning. Overlay mu-ID (95.1%) is the one metric still
below the old-stage3 reference (99%), so muon augmentation remains a candidate
polish for the next round rather than a must-fix.

Largest remaining losses are now the kinematic cut stages (g-cut ~80-83%,
mu-cut 82-88%) and the >=2g requirement in the light selection — candidates for
WP retune / stage-4 keypoint retrain, not slicer/segmenter capability.

Status: segmenter chain still training toward ep8 (2829306 running, successor
2837663); slicer chain toward ep5 (2735481). C1 checkpoint selection next once
ep8 lands.

## S1+s1cache-stage-3 results: full cutflow tables (ep7) vs the reference chains

*(jobs 2830048 (stage-3 ep1) and 2837731 (stage-3 ep7); same thin truth
ruler and gslice/gfound bar (0.2) as the tables above)*

Cascade for the "+s3" columns: v6-lantern deghoster @tau0.2 + S1 slicer
ep2 + **NEW s1cache-m2frecipe stage-3** (epoch 1 / epoch 7) + old
attempt-2 keypoint. "S1 ep2" column = same chain with the OLD v2 m2f
stage-3 (mirrored from above for reference).

### Truth-matched (decomposed) cutflow — OVERLAY, 174 CC1pi0 signal

| step (survival vs prev) | old | v2 | S1 ep2 (+v2 s3) | +s3 ep1 | +s3 ep7 |
|---|---|---|---|---|---|
| reco'd | 169 | 173 | 173 | 173 | 173 |
| vtx in FV | 167 | 171 | 171 | 171 | 172 |
| γ deghost (>20% survives) | 162 (97%) | 146 (85%) | 171 (100%) | 171 (100%) | 172 (100%) |
| γ slice (>20% in nu slice) | 133 (82%) | 100 (68%) | 166 (97%) | 166 (97%) | 167 (97%) |
| γ found (≥20% instance) | 100 (75%) | 53 (53%) | 133 (80%) | 156 (94%) | **164 (98%)** |
| γ ID | 98 (98%) | 52 (98%) | 127 (95%) | 125 (80%) | **158 (96%)** |
| γ cut (≥2 conf attached) | 78 (80%) | 46 (88%) | 90 (71%) | 88 (70%) | 127 (80%) |
| μ found | 68 (87%) | 42 (91%) | 86 (96%) | 84 (95%) | 123 (97%) |
| **μ ID** | 67 (**99%**) | 37 (88%) | 71 (**83%**) | 76 (**90%**) | 117 (**95%**) |
| μ cut | 52 (78%) | 27 (73%) | 55 (77%) | 60 (79%) | 96 (82%) |
| + cπ veto + mγγ | 49 | 27 | 54 | 60 | 95 |
| **eff (truth-matched)** | **0.282** | **0.155** | **0.310** | **0.345** | **0.546** |

### Truth-matched cutflow — VAL twin, 200 CC1pi0 signal

| step | old | v2 | S1 ep2 (+v2 s3) | +s3 ep1 | +s3 ep7 |
|---|---|---|---|---|---|
| vtx in FV | 193 | 197 | 198 | 198 | 198 |
| γ deghost | 189 (98%) | 197 (100%) | 198 (100%) | 198 (100%) | 198 (100%) |
| γ slice | 148 (78%) | 179 (91%) | 189 (95%) | 189 (95%) | 189 (95%) |
| γ found | 122 (82%) | 142 (79%) | 155 (82%) | 148 (78%) | **183 (97%)** |
| γ ID | 118 (97%) | 140 (99%) | 152 (98%) | 122 (82%) | 173 (95%) |
| γ cut | 96 (81%) | 117 (84%) | 129 (85%) | 96 (79%) | 144 (83%) |
| μ found | 83 (86%) | 114 (97%) | 124 (96%) | 92 (96%) | 141 (98%) |
| μ ID | 78 (94%) | 112 (98%) | 121 (98%) | 91 (99%) | 139 (99%) |
| μ cut | 62 (79%) | 94 (84%) | 103 (85%) | 82 (90%) | 123 (88%) |
| + cπ + mγγ | 60 | 88 | 100 | 76 | 117 |
| **eff** | **0.300** | **0.440** | **0.500** | **0.380** | **0.585** |

### Selection (light) cutflow — pre-flash efficiency

| cell | reco'd | vtxFV | ≥2γ | +μ | +0cπ | +mγγ | eff |
|---|---|---|---|---|---|---|---|
| VAL old | 196 | 193 | 118 | 81 | 78 | 78 | 0.398 |
| VAL v2 | 199 | 197 | 129 | 104 | 98 | 98 | 0.490 |
| VAL S1 ep2 (+v2 s3) | 200 | 198 | 134 | 107 | 104 | 104 | 0.520 |
| VAL +s3 ep1 | 200 | 198 | 117 | 99 | 93 | 93 | 0.465 |
| VAL +s3 ep7 | 200 | 198 | 147 | 125 | 119 | 119 | **0.595** |
| OVL old | 169 | 167 | 101 | 73 | 70 | 70 | 0.402 |
| OVL v2 | 173 | 171 | 89 | 58 | 56 | 56 | 0.322 |
| OVL S1 ep2 (+v2 s3) | 173 | 171 | 98 | 59 | 57 | 57 | 0.328 |
| OVL +s3 ep1 | 173 | 171 | 90 | 64 | 62 | 62 | 0.356 |
| OVL +s3 ep7 | 173 | 172 | 128 | 97 | 96 | 96 | **0.552** |

Readings:
- **γ found is the biggest single gain from the segmenter retrain**: the
  s1cache stage-3 clusters the S1 slices it was trained on — overlay
  80% -> 98%, val 82% -> 97% (both best measured, above the old chain's
  75/82%). The photon charge the S1 slicer delivers (γ-slice 97/95%) is
  now actually converted into instances.
- **γ ID recovered exactly as projected**: ep1's domain-symmetric 80-82%
  sag (undertrained cls head) -> 96/95% at ep7, near the 95-98% of the
  mature reference heads.
- **μ ID overlay 83% (v2 s3) -> 90% (ep1) -> 95% (ep7)**; val holds 99%.
  The pre-registered success criterion (toward 99%) is nearly met
  despite the confirmed muon dilution of the MIX cache (see 2026-08-25
  composition scan) — the remaining 4-point val/ovl gap is the argument
  for muon augmentation next round.
- **Domain gap essentially closed at a high ceiling**: truth-matched
  ovl/val ratio 0.93 (old 0.94-at-0.28; v2 0.35; S1+v2s3 0.62). The
  light selection now also beats old on overlay (0.552 vs 0.402), which
  the S1+v2s3 chain never did — the selection is passing with the RIGHT
  objects now.
- Remaining per-step losses concentrate in γ cut (80-83%) and μ cut
  (82-88%) — attachment/kinematics working points and the stage-4
  keypoint model, not slicer/segmenter capability — plus the last γ-slice
  5% on val.

### 2026-08-25 — Stage-3 s1cache epoch-8 (FINAL) full-chain eval (job 2843824, label s1ep2p8)

Truth-matched: VAL 0.590 / OVL 0.552 (ep7: 0.585/0.546) — ratio 0.94.
Light selection: VAL 0.595 / OVL 0.557 (ep7: 0.595/0.552).

| step | VAL ep7 | VAL ep8 | OVL ep7 | OVL ep8 |
|---|---|---|---|---|
| vtx in FV | 198 | 198 | 172 | 173 |
| γ slice | 189 (95%) | 189 (95%) | 167 (97%) | 168 (97%) |
| γ found | 183 (97%) | 180 (95%) | 164 (98%) | 167 (99%) |
| γ ID | 173 (95%) | 169 (94%) | 158 (96%) | 160 (96%) |
| γ cut | 144 (83%) | 144 (85%) | 127 (80%) | 127 (79%) |
| μ found | 141 (98%) | 141 (98%) | 123 (97%) | 122 (96%) |
| μ ID | 139 (99%) | 137 (97%) | 117 (95%) | 117 (96%) |
| μ cut | 123 (88%) | 122 (89%) | 96 (82%) | 98 (84%) |
| + cπ + mγγ | 117 | 118 | 95 | 96 |
| **eff** | **0.585** | **0.590** | **0.546** | **0.552** |

ep7 vs ep8 differ at the 1-2 event level everywhere — same model quality;
ep8 is the OneCycle-annealed final checkpoint and model_best coincides
with it, so **ep8 = C1 segmenter pick** unless the user prefers ep7.
Next per approved plan: attachment-OP Phase 0 (offline threshold sweep on
the s1ep2p8 pilot attachment matrices).

### 2026-08-25 — Attachment-OP Phase 0: offline threshold sweep on s1ep2p8 pilots (job 2843858)

Method: shower_attachment_study.py on the ep8 pilot kp2+nu_reco outputs
(val 200 ev -> 3,670 pairs / 1,293 correct_origin; ovl 174 ev -> 3,491 /
1,091), every pair scored with the DEPLOYED July-11 LLR tables, union
rule (hard cuts OR llr>=thr) swept. Records kept at
output/attach_phase0_s1ep2p8/records_{val,ovl}.npz.

| | hard cuts alone | union +4.0 | union +5.0 (deployed) | union +6.0 |
|---|---|---|---|---|
| VAL correct / false | 0.800 / 0.099 | 0.848 / 0.132 | 0.834 / 0.117 | 0.829 / 0.109 |
| OVL correct / false | 0.765 / 0.103 | 0.821 / 0.144 | 0.797 / 0.129 | 0.780 / 0.116 |

Findings:
1. Hard cuts alone now attach 0.77-0.80 of correct pairs (July study on
   old-chain instances: 0.42) — the new segmenter's instances are far
   better trunk-conditioned, so the LLR union adds less than it used to.
2. The deployed +5.0 is still a sensible point: the union curve is flat
   over thr 4-8 (VAL 0.848->0.807), so a threshold-only retune buys at
   most ~1.5 pts correct for ~1.5 pts false (thr 4.0). No emergency.
3. The case for the FULL TABLE REFIT (Phases 1-3) is the recovery tails,
   which are domain-asymmetric with old tables on new instances:
   far>55cm correct attach VAL 0.775 vs OVL 0.474; small<60pt showers
   VAL 0.550 vs OVL 0.062 (N=16, thin). The old densities miscalibrate
   the new instance geometry exactly where the LLR is supposed to help.
Phase 1 (1500-file overlay rerun with the frozen chain, ~100-130 A100-h)
awaits the C1 checkpoint decision (segmenter ep8 recommended; slicer
still training toward ep5).

### 2026-08-26 — C1 FREEZE + attachment Phase 1 launched

USER DECISION: freeze the chain at S1 slicer epoch_2 + s1cache segmenter
epoch_8 "for now" — moving the slicer epoch would require a cache remake
and segmenter retune; optimal-checkpoint hunting deferred. Current
slicer/segmenter performance judged good (ep8: 0.590/0.552).

Phase 1 inference campaign (attachment-variable regeneration) LAUNCHED:
array 2845199, 16 A100 shards over the FULL 67,211-event
merged_sp_mcc9_v29e_dl_run3b_bnb_nu_overlay_1500files.txt (July study
precedent: "A/B on 67k"; ~36 GPU-h, ~6 GB output). Config
larformer-keypoint2-fullcascade-v6lantern-envslicer.py with
LARFORMER_BATTERY_SLICER_CKPT=slicer_s1_mixenriched_v1/epoch_2,
LARFORMER_KP_PARTICLE_CKPT=particle_s1cache_m2frecipe/epoch_8,
--no-flash --deterministic --save-score-maps ->
output/mcc9_bnbnu_overlay_1500_s1ep2p8/keypoint2/.
After drain: build kp2 list -> nu_reco CPU shards -> Phase 2
shower_attachment_study shards -> merge -> Phase 3 LLR refit
(fit_attachment_likelihood --label correct_origin --save-tables), with
special attention to the far>55cm / small-shower overlay tails Phase 0
flagged (0.474 / 0.062 attach with old tables).

### 2026-08-27 — Attachment Phases 1-3 COMPLETE (frozen chain: slicer ep2 + segmenter ep8)

Phase 1 (array 2845199, 16 A100 shards, ~31 h wall): kp2 cascade on the FULL
67,211-event mcc9 BNB-nu overlay sample; 67,210 files (1 legit "no nu slice"
skip, event 34534); 0 failures; checkpoint provenance verified in logs; 60-file
random readback clean. 8.1 GB.
nu_reco (array 2900242, 20 CPU shards, ~40 min each): 55,048 reco / 12,162
no-interaction skips = 67,210 exact; 0 errors; production LLR union (+5.0).

Phase 2+3 (2908359 + 2908360): shower_attachment_study over all events ->
695,720 pairs (5.2x July's 133k; 66,829 correct[GT-vtx] / 33k correct_origin)
-> results_shower_attachment_mcc9_bnbnu_overlay_1500_s1ep2p8.npz.
Refit (even/odd honest split, label correct_origin) ->
trajfit/data/attachment_llr_tables_s1ep2p8.npz (production tables untouched);
plots in plots/shower_attachment_s1ep2p8_{study,fit}/.

KEY NUMBERS (test half):
  current hard cuts        : correct 0.799 | false 0.088  (July old-chain: 0.42!)
  new-LLR @ matched false  : correct 0.747 (thr +3.33)  <- LLR ALONE now LOSES
  far>55cm recovery        : cuts 0.165 -> LLR 0.787 (the LLR's remaining job)
  small<60pt               : cuts 0.537 -> LLR 0.453 (LLR no longer helps here)
  origin precision         : cuts median 1.01 cm (70% <3cm) vs LLR 2.18 cm (53%)

READING: with the retrained segmenter the hard cuts are already strong and
PRECISE; the LLR's value has narrowed to far-converting-photon recovery. The
union rule remains the right structure. Phase 4 A/B launched (job 2914325):
new tables at thr +3.33 and +5.0 vs the deployed old-tables/+5.0 baseline on
the CC1pi0 pilots; arbiter = decomposed gamma-cut/mu-cut + light selection.

### 2026-08-27 — Attachment Phase 4 A/B (job 2914325): NEW TABLES + thr +3.33 WIN

| truth-matched eff | VAL | OVL | ovl/val | light VAL/OVL |
|---|---|---|---|---|
| old tables +5.0 (deployed baseline) | 0.590 | 0.552 | 0.94 | 0.595/0.557 |
| new tables +5.0 (newtabB) | 0.600 | 0.586 | 0.98 | 0.610/0.598 |
| new tables +3.33 (newtabA, fitted) | **0.615** | **0.609** | **0.99** | 0.625/0.621 |

Gain concentrated in gamma-cut as predicted (VAL 85->89%, OVL 79->86%);
overlay gains most (+5.7 pts) — the old tables mis-scored new instance
geometry worst on overlay (Phase-0 far-photon finding). Chain now
domain-flat at ~0.61, >2x the old chain on both domains.
Pair-level false-attach at +3.33 = 0.089 (test half) ~= hard cuts' 0.088.

RECOMMENDATION: deploy attachment_llr_tables_s1ep2p8.npz with
--attach-llr-thr 3.33. REMAINING GATE before production: selection-level
background/purity check on a background-rich sample (EXT-BNB leg or
non-signal overlay light-cutflow) — pilots are signal-only.
A/B outputs kept: output/nu_reco_{val,ovl}_s1ep2p8_newtab{A,B}/.

### 2026-08-27 — EXT-BNB background gate Stage 1 LAUNCHED (200k events)

Purpose: selection-level background/purity check gating the attachment
deployment (new tables @ +3.33). Design: kp2 ONCE with the frozen chain,
nu_reco TWICE (old +5.0 / new +3.33), truth-free light cutflow, paired
comparison; spill normalization 0.17682554549 / 0.2992 (200,000/668,388
head of the stably-sorted full list).
Old-chain full-sample reference rates (ext_satfix_table, 668,388 ev):
sel_ge2 0.575% (3,846), sel_ge2&reco_cc 0.210% (1,406) -> ~1,150 / ~420
expected passing in 200k at old-chain rates (new chain likely higher).
Data-mode smoke 2917440 PASSED (20/20 EXT events, matched_gt=0, clean).
Inference array 2917469 (16 A100 shards, 8h limit, --no-flash
--output-tree) -> /cluster/tufts/wongjiradlab/larbys/data/ub_on_tufts/
larformer_extbnb200k_s1ep2p8/keypoint2/ (3.0T free on wongjiradlab;
~25 GB expected). Full 668k pass deferred until the chain is final.

### 2026-08-28 — EXT-BNB 200k background gate RESULTS (jobs 2942360/2942361/2946234)

Paired truth-free light cutflow on identical frozen-chain kp2 (200k EXT):

| step | old tables +5.0 | new tables +3.33 | rel |
|---|---|---|---|
| reco'd | 131,940 | 134,748 | +2.1% |
| vtxFV | 49,635 | 51,028 | +2.8% |
| >=2 gamma | 1,473 | 1,685 | +14.4% |
| +mu | 141 | 167 | +18.4% |
| +0cpi | 112 | 137 | +22% |
| full sel (+mgg) | 109 | 133 | **+22.0%** |

Spill-normalized to the beam sample (0.17682554549/0.2992 = 0.591):
cosmic bkg 64.4 -> 78.6 events per beam-sample exposure.
Paired-sample note: +24 events on identical inputs; even with the
conservative McNemar bound the increase is real (>~4 sigma).

TRADE vs signal (pilot A/B): overlay truth-matched +10.3% rel
(0.552->0.609), val +4.2%. Against cosmics ALONE S/sqrt(B):
1.10/sqrt(1.22) ~ 1.00 — FOM-NEUTRAL on overlay signal, slightly
negative on val. If nu-induced backgrounds (not measured here) dominate
total B, the effective FOM turns net-positive; if cosmics dominate, the
+3.33 point buys efficiency at ~matched significance.
Also noted: new-chain >=2g cosmic rate (0.74% oldtab) is higher than the
old chain's 0.575% — better photon-finding also finds more cosmic fakes.

OPEN: middle threshold. Pilots' newtab@+5.0 keeps ~60% of the val signal
gain (0.600/0.586); its EXT background not yet measured — a third CPU-only
nu_reco variant on the same kp2 would complete the threshold curve.

### 2026-08-28 — Attachment threshold curve COMPLETE (jobs 2946345-47, 2950414): DEPLOY +4.0

Full (signal, cosmic-bkg) curve, new tables on the frozen chain; baseline =
deployed old tables @ +5.0. Signal = pilot truth-matched; bkg = EXT-200k
full light selection (paired, identical kp2).

| operating point | sig VAL | sig OVL | EXT bkg (rel) | S/sqrtB ovl (cosmic-only) |
|---|---|---|---|---|
| old tables +5.0 (deployed) | 0.590 | 0.552 | 109 (ref) | ref |
| new tables +5.0 | 0.600 | 0.586 | 115 (+5.5%) | +3.4% |
| **new tables +4.0** | **0.610** | **0.603** | **119 (+9.2%)** | **+4.6%** |
| new tables +3.33 (fitted) | 0.615 | 0.609 | 133 (+22.0%) | -0.1% |

Reading: +4.0 is the knee — it keeps ~93% of the overlay signal gain
(+9.2% rel) at +9.2% rel cosmic background, the best cosmic-only FOM
(+4.6% on overlay; val FOM ~-1%, dominated by val's smaller signal gain).
+3.33 buys the last 1% of signal at 13 more points of background — FOM-
neutral at best. Any nu-induced-background contribution (unmeasured)
shifts all new-table points further positive vs baseline.

RECOMMENDATION: deploy attachment_llr_tables_s1ep2p8.npz @ thr +4.0
(run_nu_reco: --attach-llr-tables <s1ep2p8 npz> --attach-llr-thr 4.0).
Spill-normalized cosmic bkg 64.4 -> 70.3 events per beam exposure.
Chain at this OP: truth-matched CC1pi0 eff 0.610 (val) / 0.603 (ovl),
domain ratio 0.99, vs old chain 0.300/0.282.
Optional follow-ups: nu-induced background leg (non-signal overlay,
CPU-only on the existing 67k kp2); full-668k EXT pass once the chain is
final (stage-4 retrain + WP retunes pending).

### 2026-08-28 — Flash-match tuning check (jobs 2953412/2953414): GO, no gamma retune

Pre-campaign gate (user request): 1,500 overlay events, frozen chain, flash
ON + --save-slice-ids; new script flashmodel_calib/rank1_gamma_check.py
(truth-matches EVERY slice via sidecar slice ids x merged_sp GT).
Usable 769 (527 <20 GT-nu pts, 198 no IoU>=0.2 slice, 6 no flash).
- gamma: median obs/pred on true-nu slice = 0.903 (baked gamma good to 10%).
- true-nu nu-labeled slice rank-1 in chi2: 92.1% (old-chain satfix ref 72.1%).
- slicer nu-labeled the true-nu slice 769/769 -> no cosmic-tagged-true-nu
  population to rescue (fm stream stays the net for the 198 fragmented evts).
- gamma sweep 0.5-1.5x: rank-1 flat at 92.1% -> ranking insensitive to gamma.
FOLLOW-UP (analysis-level, later): re-optimize the flash-chi2 CUT values
(1e4/1778) on new-chain chi2 tables — absolute chi2 scale shifts with the
bigger S1 slices; flashchi2_from_tables.py once ntuples exist.
VERDICT: pi0 ntuple campaign cleared to launch (Phase A next).

### 2026-08-28 — MC-only pi0 analysis, NEW vs OLD chain (job 2971654, same WP, same 67,211 events)

| metric | OLD (satfix) | NEW (s1ep2p8 + attach@4.0) |
|---|---|---|
| reco vtx (nu-stream, in FV) | 26,011 | 39,771 (+53%) |
| >=2 reco photons >20 MeV | 3,112 | 7,089 (2.3x) |
| signal CC eff (sel + right tag) | 0.549 | **0.636** (any-sel 0.685 -> **0.832**) |
| signal NC eff (sel + right tag) | 0.542 | **0.728** (any-sel 0.602 -> 0.750) |
| reco-CC composition: signal CC | 0.64 | 0.43 |
| reco-CC composition: out-of-FV bkg | 0.06 | **0.29** |
| reco-NC composition: signal NC | 0.42 | 0.18 |
| reco-NC composition: out-of-FV bkg | 0.15 | **0.53** |

READING: the retrain delivered exactly the intended efficiency recovery
(any-sel CC 0.69->0.83, NC 0.60->0.75 — the soft-photon fix), but the
new chain also vertexes FAR more events (+53%), and the added selected
population is dominated by OUT-OF-FV true-nu interactions (dirt/edge)
whose reco vertex leaks into the FV. IMPORTANT: these are in-time real
neutrinos — the flash-chi2 cut will NOT remove them, and out-of-FV pi0s
peak in m_gg. Purity recovery must come from vertex/FV handling (tighter
FV, vtx-quality or dwall cut, or an out-of-FV-aware selection), not the
flash cut. Composition shown is pre-flash/cpi (table carries flash_chi2
for the downstream cuts). NC-purity WP re-derivation (chi2 cuts + FV)
should be done on the new tables once data+EXT ntuples land.

### 2026-08-28 — bnb5e19 data leg + data/MC first look (jobs 2988029-34, 3000414)

Data ntuple VERIFIED: 176,302/176,302 entries, all selection branches,
foundVertex 71.1% (matches nu_reco accounting exactly), 0 errors,
attach @ +4.00 confirmed in logs.

Data cutflow: in-FV vtx 79,789 -> >=2g 16,759 -> sel 16,747
(reco-CC 5,180 / reco-NC 11,567).
Pre-flash-cut, pre-EXT data/MC ratios: CC 3.35, NC 4.81 — cosmic-
dominated as expected before the EXT component and chi2 cuts.

Flash-chi2 (NEW chain): separation healthy — data median 16k-29k
(cosmic contamination) vs MC 300 (CC) / 1.3-3.1k (NC); data high-tail
frac(chi2>1e4) 0.57-0.63 vs MC 0.24-0.41.
WP FLAG CONFIRMED: MC reco-NC median chi2 3,112 EXCEEDS the July NC cut
(1778) — the chi2 scale moved (bigger S1 slices + the new out-of-FV
population, whose missing out-of-TPC charge degrades the flash pred).
NC chi2 cut must be re-derived on the new tables; silver lining: a chi2
cut may preferentially remove the out-of-FV background.
Next: EXT (Phase C) -> 3-sample overlay + flashchi2_from_tables cut
re-tune + SBND benchmark.

### 2026-08-29 — pi0 data/MC campaign CLOSING RESULTS (jobs 3016802-05; tables rebuilt with --cascade-dir chi2)

**SBND-SPINE CC1pi0 benchmark (after flash cut — the July-comparable line):**
| | purity | efficiency |
|---|---|---|
| July (old chain) | 0.69 | 0.39 |
| **NEW chain (s1ep2p8 + attach@4.0)** | **0.777** | **0.566** |
| SBND-SPINE reference | 0.86 | 0.65 |
Purity +9 pts AND efficiency +18 pts; the chain now sits within ~0.08 of
SPINE on both axes (was -0.17/-0.26). Data (365) vs MC pred (408) at
this selection: ratio 0.89.

**NC near-peak (100-170 MeV) data vs prediction (July WP, chi2<1778):**
eq2: data 265 | MC+EXT 248.8 -> ratio **1.07** (July ref 1.043+-0.082);
>=2: 1.12. Agreement holds — no unexplained excess with the new chain.

**NC pi0 purity at the July WP: 0.374 (0-cpi)** vs July 0.555 — the WP
is now mis-tuned for the new chain: EXT cosmic (66.2) is the largest
single component and out-of-FV adds 16.5 (per 213 weighted near-peak
events). Known causes: NC chi2 cut (1778) sits BELOW the new-chain MC
NC median (3112), and the out-of-FV population needs FV/dwall handling.
WP re-derivation on the new tables is the next (cheap, analysis-level)
step, with plots_s1ep2p8_flashchi2_tables/ as the starting point.

All artifacts: {mc,data,ext}_s1ep2p8_table.npz (chi2-filled),
plots_s1ep2p8_{mc,data,ext,datamc,ext_overlay,flashchi2,flashchi2_tables,sbnd}/.
Campaign chain: MC 67,211 / bnb5e19 176,302 / EXT 200,000 ntuples, all
verified, frozen chain + refit attachment @ +4.0 throughout.

### 2026-08-29 — Shower charge-to-energy RECALIBRATION for the new chain (jobs 3022713-3027686)

Problem (user): new-chain m_gg peak far off 135 (measured: 174.3 MeV median,
+29%) — expected, since the retrained segmenter's clusters capture more charge.

Method trail (all analysis-level after the valdata inference; NO reco reruns —
recal flags added to pi0_mass_analysis.py / sbnd_cc1pi0.py invert the deployed
calib and re-apply a candidate curve to showerRecoE in place):
1. Overlay-originals collection: BROKEN — new chain recovers UNLABELED
   peripheral charge, GT majority-trackid lands on the pid=-1 GeV aggregate
   row (81% of instances). Lesson: truth-matched studies on overlay need the
   label-completed copies. Quarantined: calo_calib_s1ep2p8_BROKEN_overlayfit.npz.
2. Valdata (10k, complete labels) collection via particle_momentum.py:
   through-origin fit overshoots (peak 156->221); plain affine contaminated.
   Profile fit (per-E-bin median Q, July's inverse-construction) vs
   mc_particle_tree KE: gamma 0.01791*Q-13.40 -> peak 156.4 (+16% residual).
3. m_gg DECOMPOSITION (truth-matched pairs, E vs angle swapped): angle
   contribution only ~3%; and fully-true m_gg = ~300 on BOTH chains =>
   **A_GAMMA x trueSimPartPixelSumQ 'visible energy' is ~2.2x the ACTUAL
   photon energy** (longstanding convention; the 20-MeV detectability
   threshold has always been ~9 MeV actual — flagged for user sign-off).
   Old-chain native peak measured at 136.7 ✓ (reco scale = actual energy).
4. FINAL FIT (in-situ, actual-E target): per-E-bin median cluster charge vs
   |p_true| of truth-matched photons, new-chain MC ntuple, 8,286 photons:
   **gamma: E = 0.01556*Q - 11.47** (bins close within +-5%, 47-389 MeV).
   VALIDATION: pi0 pairs m_gg median **132.3** / windowed mean 137.7 (135 ✓).
   (July analogue: 0.0201*Q-15.49; slope -23% = the completeness gain.)
   e-channel left at deployed calib (profile non-monotonic; negligible here).

Rollout (jobs 3027773-76): all three sample tables rebuilt with
--recal-gamma-a 0.01556 --recal-gamma-b -11.47 (+chi2 refill) -> 3-sample
overlay + flashchi2 tables + SBND benchmark, all at the July WP.
calo_calib.npz deployment for FUTURE reco runs awaits user sign-off
(gamma_a=0.01556, gamma_b=-11.47); deployed file currently restored to the
old constants.

### 2026-08-29 — RECALIBRATED 3-sample results (jobs 3027773-76): data/MC agreement + purity RESTORED

| metric (July WP) | pre-recal | RECAL (0.01556*Q-11.47) | July old-chain ref |
|---|---|---|---|
| NC near-peak data/(MC+EXT), eq2 | 1.07 | **1.05** | 1.043+-0.082 |
| NC near-peak data/(MC+EXT), >=2 | 1.12 | **1.02** | — |
| NC pi0 purity (0-cpi, near-peak) | 0.374 | **0.541** | 0.555 |
| SBND purity / eff (after flash cut) | 0.777 / 0.566 | **0.763 / 0.555** | 0.69 / 0.39 |

The recal pulled true pi0s into the mass window and mis-scaled bkg out:
near-peak sigNC 79.7 -> 142.6, EXT 66.2 -> 40.8. NC purity back at the
July benchmark WITHOUT yet retuning the chi2/FV working point (upside
remains there). SBND benchmark essentially unchanged (threshold bites at
the corrected scale) — still purity +7pts / eff +17pts over July.
PENDING USER SIGN-OFF: (1) bake gamma (a=0.01556, b=-11.47) into
calo_calib.npz for future reco runs; (2) A_GAMMA visible-energy
convention (currently ~2.2x actual photon energy, threshold ~9 MeV
actual — longstanding, affects denominators if changed).

### 2026-08-29 — EXT-background diagnostics (job 3049646, datamc_diagnostics.py, 21 vars)

Post-WP selection totals: MC 1821 (w) + EXT 964 (w) = 2785 vs data 3252
(+17% data excess). EXT = ~35% of prediction. Where EXT (and the data
excess) lives — every signature is classic cosmic:
- leading-photon conv distance ~0 (fragments starting AT the vertex);
- sub-leading KE piled at the 20-40 MeV threshold;
- 0-1 non-shower primaries (shower-only / single-track events);
- cos(theta_Y) horns at +-1 (vertical showers); vtx y at TPC top;
- low dwall; dim in-time flashes (<1.5k PE).
Data exceeds MC+EXT in the SAME corners -> residual under-predicted
cosmics (EXT 200k subset stats scaled x0.59 amplifies fluctuations; full
668k EXT would firm this up), not nu-MC mismodeling.
CANDIDATE CUTS (each attacks EXT with modest signal cost, to be tuned
together with the chi2/FV WP): leading conv-dist > ~2-3 cm; sub-leading
KE > ~35-50 MeV; n non-shower primaries >= 1 (for CC) / topology-aware;
|cos theta_Y| < ~0.9; dwall > ~15-20 cm; flash PE floor.
NOTE: vtxFracHitsOnCosmic branch is unfilled in these ntuples (empty
plot) — would be a natural discriminant if the exporter populated it.
Plots: plots_s1ep2p8_diag/ + gallery artifact (ext-diagnostics version).

### 2026-08-30 — EXT-rejection BDT (ext_bdt.py, job 3053359): AUC 0.988

Signal(MC cat<2) vs EXT, post-WP sample, even/odd split, 18 topological/
geometry/flash features (m_gg EXCLUDED; no sculpting seen). sklearn
HistGradientBoosting (no xgboost in container; user OK'd sklearn).
ROC (test): eff 0.99 -> EXT rej 0.83 | 0.97 -> 0.91 | 0.90 -> 0.98.
Importances: logchi2 dominates (the BDT subsumes the pending NC chi2
retune), then dist2/dist1/vtxScore. Data score distribution = cosmic+nu
cocktail as expected (closure ok).
Near-peak (100-170) impact: eff-0.99 cut -> EXT 103->41, sig-frac
0.732->0.801, data/pred 1.03->0.97 at sig cost 0.2%.
RECOMMENDED WP: eff 0.99 (thr 0.062). Tighter points push data/pred
below 1 (0.93/0.87) — chi2 input data/MC shape systematic; validate
inputs in a signal-depleted sideband before tightening.
Artifacts: plots_s1ep2p8_diag/bdt_{roc,score,sculpt}.png + bdt_scores.npz.

### 2026-08-30 — BDT robustness variants (job 3053537): FLASH-BLIND WINS

User concern: flash mismodeling drives the baseline BDT's data/pred dive
(1.03->0.87 with cut tightness). Variants (ext_bdt.py --drop-feats/--smear):
| eff | base | B no-chi2 | C flash-blind | D chi2-smeared 0.5dex |
|-----|------|-----------|---------------|------------------------|
| 0.99| 0.97 | 0.97 | 0.98 | 0.96 |
| 0.97| 0.93 | 0.95 | 0.96 | 0.95 |
| 0.90| 0.87 | 0.93 | 0.93 | 0.90 |
B/C plateau at ~0.92-0.95 (the nu-only data/pred level) — stable; baseline
digs below (chi2-shape over-rejection of data nu); smearing only halves
the drift. Flash-blind cost is small: C AUC 0.972, EXT rej 0.86@eff0.97
(vs 0.91 with chi2); importances now dist1/dist2/vtxScore (pure topology).
RECOMMENDED OFFICIAL CUT: variant C @ eff 0.97 (near-peak EXT 103->39,
sig-frac 0.732->0.806, d/p 0.96, sig cost 3%); C @ 0.95 if more purity
wanted. Awaiting user sign-off to wire into the selection.
Artifacts: bdt_{roc,score,sculpt}_{nochi2,noflash,smear}.png + npz.

### 2026-08-30 — ADOPTED: recalibration + flash-blind EXT BDT (job 3053911)

USER SIGN-OFF: (1) calo_calib.npz now carries gamma a=0.01556 b=-11.47
(e unchanged; old constants in calo_calib_oldchain_backup.npz) — future
reco runs produce corrected energies natively. (2) EXT-rejection BDT
adopted, flash-blind variant, OFFICIAL WP eff-0.97 (score >= 0.280).

Training hygiene (user-directed): BDT trained on even-event signal-MC +
EXT halves ONLY; all plots/tables use the held-out odd halves (w x2) —
no BDT-training events in any plotted distribution. Audit: the chain-
training corpus (MIX v1 = overlay_train nue/pi0 productions + LANTERN)
has ZERO overlap with the analysis run3b generic overlay, so the
analysis MC is also network-clean. Upgrade path if tighter cuts wanted:
process CC/NCpi0 overlay_train slices through the chain as dedicated
BDT training stats.

Official numbers (held-out only): AUC 0.972; at WP: EXT rej 0.86,
near-peak EXT 103->33, sig-frac 0.738->0.817, d/p 0.95, sig cost ~1%.
Importances: dist1/dist2/vtxScore/E1 (pure topology).
Model: pi0mass_peak/ext_bdt_model_flashblind.joblib (clf + feats + wps).

### 2026-08-30 — FINAL post-BDT pi0 selection (job 3054686): plots_s1ep2p8_ext_overlay_recalF_bdt/

BDT-filtered leakage-free tables (*_recalF_bdt_table.npz; sel flags
zeroed for bdt-fail + training halves, heldout w x2) -> standard overlay
suite with recal + flash cut + BDT. NC 0-cpi near-peak purity
0.541 -> **0.635** (July benchmark 0.555), EXT 40.8 -> 7.1, data/pred
1.03 (eq2) / 0.99 (>=2). Data bdt-fail 50% (its cosmic half), EXT 87%,
MC 12%. The v2-chain pi0 analysis now BEATS the July working point on
purity at higher efficiency with data/MC agreement intact.

### 2026-08-30 — Flash-chi2 with BDT applied FIRST (job 3055881): chi2 RETUNE RESOLVED

Chi2-uncut selection, official flash-blind BDT applied (scores for all
FV+2gamma events; BDT removes 84% of chi2-uncut EXT, 75% of chi2-uncut
data). plots_s1ep2p8_flashchi2_bdtfirst/. Readings:
- nu peak (log10 chi2 ~2-2.5) well described by MC in data — NO flash
  mismodeling for genuine neutrinos; the earlier chi2-shape tension was
  cosmic contamination.
- Post-BDT data excess concentrates at HIGH chi2 (the EXT-underpredicted
  residual cosmics) — the chi2 cut removes exactly that; BDT and chi2
  are complementary, both stay.
- The pre-BDT worry (MC NC median 3112 > 1778 cut) is GONE: that median
  was driven by the out-of-FV/cosmic-like population the BDT now removes;
  the surviving nu peak sits ~10^2.2, far below both cut lines. July cut
  values (1778/1e4) stand as-is.

### 2026-08-30 — CC flash-chi2 mismodeling breakdown (job 3060767): EXITING MUONS, not Cherenkov

User hypotheses: (1) broken mu reco, (2) mu exits TPC (out-of-TPC light
unmodeled), (3) Cherenkov. Study: cc_flash_breakdown.py — reco-CC,
chi2-uncut, BDT applied, mu = highest-KE primary PID13 track; splits on
trackStartDirX sign and SCE-corrected endpoint dwall. SCE via NEW numpy
module lartpc/flashmatch/sce_microboone.py (exact MCC9-backward map
replication; larlite pyROOT binding broken in current build — stale
dictionary, only copy/default ctors exposed; rebuild would fix).
| split | peak d/p | tail d/p |
| dirX>0 / dirX<0 | 0.80 / 0.83 | 1.39 / 1.37 |  <- NULL: no Cherenkov signal
| contained (dwall>15) | **0.98** | 1.51 |
| exiting (dwall<5) | **0.66** | 1.30 |
VERDICT: (2) confirmed — contained-mu events agree nearly perfectly in
the chi2 peak; exiting-mu data migrates to high chi2 (out-of-TPC light
in the real flash, absent from the charge-based prediction; MC is
self-consistent so it doesn't see the effect). (1) entangled (broken end
mimics exit). (3) no directional asymmetry. Tail d/p 1.3-1.5 everywhere
= the separate EXT-normalization issue.
Possible remedy: containment-aware CC chi2 cut.
Plot: plots_s1ep2p8_diag/cc_chi2_splits.png.

### 2026-08-31 — CC chi2 shape quantified (job 3067426): VERDICT REVISED — global scale+resolution, not containment

User observation (correct): data chi2 peak shifted high AND broadened in
ALL splits. cc_chi2_shape.py measures: peak shift +0.31 dex globally
(+0.34 contained / +0.26 exiting — containment-INDEPENDENT; the earlier
contained-vs-exiting window-ratio contrast was a windowing artifact).
Flash amplitude ratio r=obs/pred (nu slice, live PMTs):
  MC   median 0.809, IQR 0.27
  DATA median 0.736, IQR 0.50
=> (a) prediction too bright in BOTH legs on the new chain (gamma fit on
old-chain slices; fuller S1 slices predict more light), data ~9% further
off than MC -> peak SHIFT; (b) data obs/pred spread 1.85x MC ->
broadening (optical-sim fluctuations underestimated; f_sys=0.10 too
small for data). Remedies (analysis-level, no reprocessing): per-leg
gamma refit on new-chain slices (~x0.81 MC, x0.74 data) + f_sys retune.
Low urgency: chi2 only enters via loose WP cuts; BDT is flash-blind.
Plot: plots_s1ep2p8_diag/cc_chi2_shape.png.

### 2026-08-31 — Electron-lifetime correction: GATE FIRED, correction dropped (job 3071891)

Plan step 2 (elifetime_fit.py): in-situ tau from reco-CC muons (4k tracks
/leg, per-track-normalized point charge vs drift, median-binned):
  MC   slope -9.7e-5/cm -> full-drift attenuation 2.45% (tau ~ 94 ms)
  DATA slope -7.3e-5/cm -> full-drift attenuation 1.87% (tau ~124 ms)
Under the 2-3% gate; per-shower effect <<1% vs ~25% resolution; data/MC
slopes consistent (no differential bias either); slopes are upper bounds
on pure lifetime (residual SCE/wire x-dependence included). LIFETIME
CORRECTION NOT PURSUED. The drift-correction machinery (SCE module +
per-point charge extraction) stays available for the more promising
resolution levers: YZ wire-response uniformity map on SCE-corrected
positions, and clustering-completeness-dependent energy corrections.
Plots: plots_s1ep2p8_diag/elifetime_{mc,data}.{png,npz}.

### 2026-08-31 — Reco & detector performance studies (job 3078124)

A. SHOWER COMPLETENESS (shower_completeness_study.py, MC, 7,698
truth-matched photons, 2,299 pi0 pairs): completeness median 0.955
(67% >0.9, 13% <0.7) but PURITY median 0.79 (only 13% >0.9) — clusters
absorb neighbour charge (other photon / vertex activity).
E_reco/E_true scale tracks completeness monotonically: comp<0.5 0.26,
0.5-0.7 0.68, 0.7-0.8 0.86, 0.8-0.9 1.00, 0.9-0.95 1.13, >0.95 1.17 —
the calibration is right on AVERAGE but complete+impure photons read
~16% high, incomplete ones low. Within-bin IQR/med 0.33-0.38 (comp>0.8)
vs 0.49 overall; headroom at comp&pur>0.9 only 0.41 (N=710, 9%).
pi0 peak by min-completeness: <0.7 median 94 MeV; 0.7-0.9 134.5 (on
target); >0.9 150 (impurity-inflated) — the 135 peak is a SUPERPOSITION
of shifted sub-populations; its width is largely clustering-mix driven.
Frac in 100-170 only 0.55-0.59 even for well-clustered pairs.
IMPLICATION: resolution gains must come from clustering (purity —
splitting neighbour charge; completeness — soft periphery), or an
observable purity proxy; no simple charge-scale correction exists.

B. YZ UNIFORMITY (yz_uniformity_map.py, SCE-corrected muon points,
~4k tracks/leg, 312 bins): RMS non-uniformity MC 4.6%, data 4.7%
(structure: low-response band z~200-400 at y>50; high band z~700-750
esp. y<0 — data), data/MC ratio RMS 2.6% (0.92-1.08; residual
structure z<150 (+), z~650-700 (-)). GATE: a YZ correction adds <~3-5%
in quadrature vs ~25% shower resolution -> negligible gain; NOT pursued
as a shower correction, kept as a detector-performance plot set.
Plots: plots_s1ep2p8_diag/{shower_completeness,pi0_vs_completeness,
yz_map_mc,yz_map_data,yz_map_ratio}.png (+npz).

### 2026-08-31 — Label expansion of the pi0-analysis overlay sample (user-directed)

FINDINGS: (1) the 67k analysis overlay files are NOT label-completed
(sampled files lack the label_completion attr; training corpus copies
have it). (2) complete_labels.py IS idempotent — a `label_completed`
dataset guards re-runs ("ALREADY COMPLETED, skipped"; comment: a second
pass would cascade-grow labels via adopted donors). In-place mode
preserves originals as *_precomplete datasets (reversible; no 335 GB
copy needed). (3) trueSimPartPixelSumQ is EXPORTER-computed per-trackid
dedup charge from merged_sp labels (msp_qvis) -> updates automatically
on re-export; the ROOT-based truth sidecar (TID/PDG/E/GENIE) is
label-independent -> NO sidecar rerun. (4) kp2 gt_point_idx were built
from ORIGINAL labels at inference -> TruePurity/TrueComp (isin vs
gtpidx) would stay stale unless kp2 is re-inferred; PLAN: redefine
purity/comp in the exporter as merged_sp-TRACKID-based (charge with
trackid==matched TID / cluster charge; comp vs the TID's total event
charge) so a CPU re-export alone suffices post-expansion.

SEQUENCED (background handler): wait for the label-READING jobs
(contamination 3088227, charge-purity export 3088729/30 — expansion
must not race them) -> launch in-place completion array over the 67k
list (100 chunks, 40 concurrent; ~few hours). THEN: patch exporter to
TID-based purity/comp, re-export (qpurity2), rerun contamination study
on expanded labels (its ghost/unlabeled fraction before-vs-after is the
direct measure of what expansion reclaims).

### 2026-08-31 — Photon charge-contamination breakdown (job 3088227): it's the UNLABELED charge

7,698 truth-matched photons, charge fractions by true owner (per-point
dedup comb charge, original labels):
| type | mean frac | ==0 | >10% | >30% |
| other photon | 0.017 | 84% | 5.7% | 1.9% |
| electron | 0.001 | 98% | 0.4% | — |
| muon/pion/proton/other | ~0.000-0.001 | 95-98% | <=0.2% | — |
| **ghost/unlabeled** | **0.222** | 0.1% | **86.4%** | 23.7% |
Charge purity (this photon): median 0.777, >0.9 only 12.4%.
=> The "contamination" is ~ALL unlabeled charge (photon's own periphery
not claimed by the original labels + overlay real-cosmic charge), NOT
physics cross-clustering — real particle contamination is at the
percent level (other-photon 5.7% > 10%, everything else <0.5%).
E_reco vs E_true by charge purity: >0.9 median E/Etrue 0.91 (clean
clusters slightly LOW — the calibration absorbed typical unlabeled
charge); [0.7,0.9) 1.06 IQR 0.48; <0.7 1.09 IQR 0.58 (widest; low
E_reco<<E_true outlier branch = split clusters).
=> VALIDATES the label-expansion decision directly: after expansion the
ghost fraction should collapse and purity metrics become physical.
Plots: photon_contam_fractions.png, photon_contam_over10pct.png,
ereco_vs_etrue_by_purity.png (+npz).

### 2026-08-31 — Charge-purity ntuple PROMOTED; TID-based purity patched; label completion RUNNING

- Charge-based-purity MC export (3088729/30) verified on truth-matched
  nu photons: TruePurity(charge) 0.82 med / TrueUnlabeledPurity 0.17 med
  / TrueComp 0.94 — consistent with the contamination study. PROMOTED to
  the canonical ntuple name; point-based original kept as
  *_pointpurity.root. (Cosmic-stream prongs legitimately read unlabeled
  ~1.0 — filter primaryVtxStream==0 when using these branches.)
- Exporter purity/comp now TID-BASED (cluster charge with merged_sp
  trackid == matched TID; comp vs qv[tid], capped 1.5) — fully live off
  merged_sp labels; kp2 gt_point_idx no longer used, so post-expansion
  a CPU re-export alone suffices.
- Label-completion array 3089802 launched by the sequencer (in-place,
  67,211 files, 100 chunks x 40 concurrent) after the last label-reader
  drained. Post-drain plan: re-export (qpurity2, TID basis, expanded
  labels) -> promote -> rerun contamination study for the before/after
  ghost-fraction measurement -> revisit calibration/E_vis (PixelSumQ
  grows with expansion).

### 2026-08-31 — Label-completion array 3089802: PARTIAL (39/100 chunks silently empty) — diagnosed + makeup sequenced

All 100 chunks reported COMPLETED, but only 41,053/67,211 files were
processed. ROOT CAUSE: node-local /tmp full on some nodes — `sed >
/tmp/chunk` wrote 0 lines ("No space left on device" in .err), xargs got
no input, complete_labels errored on empty --h5, and the chunk still
printed "done" (no guard). FIXES: submit_complete_labels_array.sh now
writes chunk lists to shared storage (logs/overlay_train/chunks/) with a
line-count guard that aborts the task on a short chunk file.
Sequenced: full h5 scan (job 3089912) builds
merged_sp_overlay_incomplete_labels.txt -> auto-handler submits the
makeup array over the remainder with the patched submitter (idempotency
guard makes any overlap safe). Then: TID-basis re-export (qpurity2) ->
promote -> contamination-study rerun (before/after ghost fraction).

### 2026-08-31 — Label expansion COMPLETE (0/67,211 incomplete); expanded-label reruns launched

Makeup array 3089950 (39 chunks, patched submitter) drained clean;
verification scan 3090026: TOTAL incomplete 0/67,211. (Handler quirk
fixed along the way: grep -c on an empty file exits 1, polluting the
count variable — the abort was spurious; coverage was complete.)
Launched in parallel: (a) TID-basis re-export on expanded labels
(8 shards + hadd -> *_qpurity2.root) with in-handler verification of
the post-expansion purity/unlabeled/comp medians; (b) contamination-
study rerun (reads merged_sp labels directly) -> plots_s1ep2p8_diag_
expanded/ for the before/after ghost-fraction measurement
(before: ghost mean 0.222, >10% in 86.4% of photons).

### 2026-08-31 — EXPANDED-LABEL contamination rerun (job 3090061): expansion reclaims ~92% of the "contamination"

Same 7,698 truth-matched photons, before -> after label expansion:
| metric | before | after |
| charge purity median | 0.777 | **0.991** |
| purity >0.9 | 12.4% | **88.8%** |
| purity <0.7 | 29.4% | 3.8% |
| ghost/unlabeled mean | 0.222 | **0.018** |
| ghost >10% of photons | 86.4% | 3.0% |
Real particle contamination unchanged (other-photon >10%: 5.7 -> 6.1%
— some formerly-unlabeled charge belongs to the OTHER photon; e/mu/pi/p
still <0.5%). CONCLUSION: the clusters were physically pure all along —
the "purity problem" was label incompleteness. The segmenter's
clustering is validated; the calibration/E_vis chain can now be
re-examined against physical purity (post-expansion PixelSumQ also
changes the E_vis denominator once the qpurity2 export lands).
Plots: plots_s1ep2p8_diag_expanded/.

### 2026-08-31 — Expanded-label ntuple PROMOTED; denominator remake running

qpurity2 export verified (67,211 entries; TruePurity median 0.991 /
unlabeled 0.005 / TID-comp 0.860 — matches the contamination rerun) and
PROMOTED to canonical; archives: *_pointpurity.root (point-based,
orig labels), *_qpurity_origlabels.root (charge-based, orig labels).
Remake chain (job 3090318, user-directed): expanded-label MC table at
the WP (signal-denominator change vs 2002 CC / 1104 NC as PixelSumQ
grows past the 20-MeV E_vis threshold) -> BDT filter -> CC/NC pi0
overlay suite (plots_s1ep2p8_ext_overlay_expanded/) -> SBND benchmark
(plots_s1ep2p8_sbnd_expanded/, denominator vs 561). Data/EXT tables
truth-free -> reused unchanged.

### 2026-08-31 — Denominator remake on EXPANDED labels (job 3090318): NULL — denominators unchanged

Signal counts: CC 2002 -> 2003, NC 1104 -> 1105 (+1 each); SBND true
signal 561 -> 561. EXPLANATION: the E_vis convention (A_GAMMA x
PixelSumQ ~ 2.2x actual energy) made the 20-MeV detectability cut an
effective ~9-MeV actual-energy cut; expanding labels (+~20% PixelSumQ)
moves that to ~7.5 MeV actual, where essentially no pi0 photons live.
All headline numbers stand: near-peak data/(MC+EXT) 0.99/1.03, SBND
0.763/0.554 vs 561. (Reco-side counts differ from the older recal-CHECK
printouts only because those used interim constants 0.0230/0.01791; the
final 0.01556 curve gives >=2g 6,335 consistently.)
IMPLICATION for the pending A_GAMMA decision: if "detectable" should
mean 20 MeV ACTUAL energy, A_GAMMA must be re-derived (approx /2.2);
that WOULD shrink the denominators and raise reported efficiencies —
a convention choice, now cleanly separated from the label question.
Tables/plots: mc_s1ep2p8_exp{,_bdt}_table.npz,
plots_s1ep2p8_{mc_exp,ext_overlay_expanded,sbnd_expanded}/.

### 2026-08-31 — SCE module fully self-contained (user-directed)

TH3F backward-offset maps converted to a compressed npz checked into the
repo: lartpc/flashmatch/data/sce_offsets_mcc9_bkwd.npz (536 KB; float32
grids + float64 bin-center axes + provenance note). sce_microboone.py
now defaults to the bundled npz (numpy+scipy only — no ROOT, uproot, or
ubdl checkout); passing a .root path still works via uproot for
alternate maps. Equivalence verified: max |diff| = 0 over 2,000 random
in-TPC points vs the original ROOT-file loader.

### 2026-08-31 — SCE forward mode added (user-directed)

SCEForward (sce_microboone.py): true deposit position -> expected reco
position, replicating kMCC9_Forward (x_reco = x_true - off_x + 0.7
anode hack; y/z + off; no-op outside map bounds). Forward map bundled as
data/sce_offsets_mcc9_fwd.npz (checked in, ~600 KB). npz-vs-ROOT
equivalence exact; forward/backward round-trip residual reported by the
test (the two data-driven maps are approximate inverses).

### 2026-08-31 — Photon calibration refinement v3 (job 3108486, user-directed)

User observation: expanded-label ereco_vs_etrue shows purity>0.9 median
E/Etrue 1.05 (+5% residual) and the mass peak sits high. Chain launched:
1. Refit on truth-matched photons (expanded ntuple): per-E-bin median-Q
   profile (11 bins, E<450) for the SHAPE + global normalization so the
   per-photon median E/Etrue == 1 exactly; validation m_gg printed.
2. Rebuild ALL THREE tables (mc/data/ext) with the refined constants
   (recal shifts data/EXT reco energies too) + chi2 refill.
3. BDT-filter (official model; note: model trained at 0.01556-constants
   features, E1/E2 shift ~5% — small perturbation, E1 importance 0.03).
4. Overlay -> plots_s1ep2p8_ext_overlay_recal3; SBND -> ..._sbnd_recal3.
On success the refined constants supersede 0.01556/-11.47 (calo_calib
update pending user sign-off as usual).

### 2026-08-31 — Recal3 constants DERIVED: gamma a=0.01553 b=-12.80

Refit (11-bin profile shape + per-photon median normalization /0.9966):
per-photon median E/Etrue = 1.0000 (E<450); validation pi0 m_gg median
130.2 / windowed mean 136.3 (target 135). NOTE: the shape refit moved b
(-11.47 -> -12.80) more than the scale — the +5% seen in the by-purity
panel was partly the E>450 over-capture region pulling the old panel
median. Chain job 3108486 FAILED after the derivation on tee->/tmp
(compute-node /tmp full — same plague as the label-completion chunks);
steps 2-4 resubmitted with constants hardcoded and no /tmp usage.

### 2026-08-31 — Recal3 remake COMPLETE (job 3109500): peak centered, selections stable

Constants gamma a=0.01553 b=-12.80 deployed to calo_calib.npz (per-photon
median E/Etrue = 1.0000; validation m_gg 130.2 med / 136.3 mean).
Remade headline numbers (vs recalF): NC 0-cpi near-peak purity 0.637
(0.635); data/(MC+EXT) 0.96 (>=2) / 0.99 (eq2); SBND after-flash
purity 0.762 / eff 0.551 (0.763/0.554); denominators unchanged
(2003/1105, SBND 561). The few-% energy shift centers the peak without
moving the selections — analysis re-baselined on recal3 tables
({mc,data,ext}_s1ep2p8_recal3{,_bdt}_table.npz) and plots
(plots_s1ep2p8_ext_overlay_recal3/, plots_s1ep2p8_sbnd_recal3/,
plots_{mc,data,ext}_recal3/).

### 2026-08-31 — ereco_vs_etrue_by_purity remade at recal3 (job 3114710)

plots_s1ep2p8_diag_recal3/: purity>0.9 (89% of photons) median E/Etrue
1.05 -> 1.03 (IQR 0.49); all 1.04. The panel median sits slightly above
the recal3 normalization (1.000, defined on E<450) because the panel
includes the E>450 over-capture region — visible as the above-diagonal
band at E_true >~ 300 (the containment-breakdown zone from the profile
fit). Core population rides the diagonal; low-purity bins (11%) remain
the wide/mismeasured tail (1.11-1.14, IQR 0.68-1.21).

### 2026-09-01 — Per-shower cosmic BDT for the exporter (user-directed; training-hygiene plan enforced)

User: embed a cosmic-vs-nu photon score per shower in the ntuple
(electrons autopass 1.0), no pruning; AND keep to the plan: train on the
SEGMENTER-TRAINING corpus + a dedicated HALF of EXT (analysis half
reserved for plots). Correction recorded: the event-BDT (ext_bdt.py)
is pair-topology-bound (cos12, E2, dist2, nPhot) — a per-shower model
is required; shower_cosmic_bdt.py written (22 exporter-computable,
flash-blind features). Job 3118640 = PROTOTYPE only (analysis MC +
EXT-even) for feature validation.
OFFICIAL training inputs being produced: (1) overlay_train v40a run3
CC/NC pi0 corpus (40,589 events, label-completed, in the segmenter
training set) through the frozen chain -> ntuple (handler bs75kfou0:
inference NINF=12 + race-safe manual downstream; TAG
overlaytrain_pi0_bdt_run3; data-mode export — TrueTID + charge purities
still available for truth-matching); (2) EXT split by list halves:
merged_sp_extbnb_200k_bdttrain_half.txt (first 100k) for training,
merged_sp_extbnb_200k_analysis_half.txt for plots (EXT scale 1.1818).
Exporter: schema branch showerCosmicScore + score_showers() hook
(model via env LARFORMER_SHOWER_BDT; -9 when absent; electrons 1.0).

### 2026-09-01 — Per-shower BDT PROTOTYPE (job 3118640): feature set validated

Analysis-MC photons (7,650) vs EXT photons (60,886), even/odd split,
held-out: AUC 0.984; photon eff 0.985 -> EXT-photon rej 0.818; 0.97 ->
0.894; 0.95 -> 0.928. Importances: LArFormer photon score, nShower,
vtxScore, start dwall, pi/mu scores, attachment — all flash-blind and
exporter-computable. Data photons pass(eff0.985 thr) 24.6% vs EXT 18.2%
/ signal 98.5% — consistent cosmic+nu mix. Per-shower performance ~ the
event-level model => a per-shower branch loses nothing.
Model saved as export/data/shower_cosmic_bdt_PROTOTYPE_analysisMC.joblib
(NOT for deployment — trained on analysis MC). The official model will
be trained on the segmenter-training pi0 corpus ntuple (chain in
flight) + the EXT training half, then deployed via LARFORMER_SHOWER_BDT.

### 2026-09-01 — Per-shower BDT: exporter smoke + plot-side EXT convention

- Exporter hook smoke (prototype model via LARFORMER_SHOWER_BDT) retried
  with a valid TAG list (first attempt failed only because the shard
  script derives merged_sp_<TAG>.txt from TAG).
- EXT plot convention going forward: list-half split. Analysis-half
  table ext_s1ep2p8_recal3_analysishalf_table.npz (rows >=100,000; sel
  zeroed for the training half) with --ext-scale 1.1818 replaces the
  even/odd x2 convention for all plots once the official per-shower
  model is in use.
- .gitignore exception added for export/data/shower_cosmic_bdt.joblib
  (the official model, once trained) so the exporter is self-contained.
- Exporter smoke PASSED (job 3118702, 169 events): model loads (22
  feats), electrons all 1.000, photons scored in [0,1] (-9 only where
  RecoE<=0 / no vertex). Branch showerCosmicScore live in the schema.

### 2026-09-01 — Training-corpus (overlay_train pi0) inference COMPLETE; downstream launched

Inference array 3118647: 12/12 COMPLETED, 40,589/40,589 nu-stream kp2
files (2 fm). The session-bound handler was lost mid-wait; downstream
(nu_reco 12 + fm 1 -> larpid -> data-mode export 4 shards -> hadd)
submitted manually with verified lists ->
larformer_overlaytrain_pi0_bdt/dlgen2_larformer_ntuple_overlaytrain_pi0_bdt_run3.root.
Next: official per-shower BDT = this corpus (--no-mc-split) vs EXT
training half (--ext-row-max 100000) -> export/data/shower_cosmic_bdt.joblib.
- First official-training attempt found 0 signal photons: the corpus
  ntuple is a DATA-MODE export (no truth sidecar) so trueSimPart*/
  trueVtxInWCFV are absent. Trainer patched: sidecar-free MC signal
  definition = showerTrueTID>0 & showerTruePhPurity>0.5 (charge-based,
  filled by the data-mode export); retraining.

### 2026-09-01 — OFFICIAL per-shower cosmic BDT trained (job 3126294)

Per the user's training-hygiene plan: signal = 40,095 photons from the
segmenter-training pi0 corpus (charge-based truth match; content-based
sidecar detection fixed the empty-branch trap), background = 30,632
photons from the EXT TRAINING half (rows <100k). Held-out: AUC 0.978;
photon eff 0.985 -> EXT-photon rej 0.716; 0.97 -> 0.830; 0.95 -> 0.891.
(Prototype on analysis MC was slightly higher — 0.818/0.894 — expected:
the corpus includes harder multi-shower pi0 topologies and no
vertex-truth gate; provenance-clean per plan.) Importances now
nhits/phScore/E/dist/att. Model DEPLOYED:
export/data/shower_cosmic_bdt.joblib (git-tracked; exporter picks it up
via LARFORMER_SHOWER_BDT). Next: re-export MC/data/EXT with
showerCosmicScore -> plots on the EXT analysis half.
- showerCosmicScore DEPLOYED to all three analysis ntuples: re-exported
  with LARFORMER_SHOWER_BDT=official model (jobs 3126759/61/63), verified
  (row-aligned with old canonicals so all recal3 tables stay valid;
  electrons 1.0; sentinel only for RecoE<=0; on analysis MC — never seen
  in training — truth-matched nu photons median score 0.929 vs 0.014
  unmatched), then promoted to canonical names (*_prescore.root archives).
  pi0_mass_analysis.py + sbnd_cc1pi0.py gained opt-in --shower-bdt-min
  (folded into the photon-candidate definition; default None = July WP
  untouched). Score-cut campaign submitted: thr 0.090 (eff-0.985 WP),
  EXT analysis half only (rows>=100k, scale 1.1818), overlay ->
  plots_s1ep2p8_ext_analysishalf_sbdt, SBND -> plots_s1ep2p8_sbnd_sbdt
  (EXT training half gated via chi2=inf variant table).
- Score-cut campaign results (job 3134031, thr 0.090 = eff-0.985 WP, EXT
  analysis half x1.1818): signal any-sel eff CC 0.794 / NC 0.709 (vs
  0.803/0.726 uncut — ~1-2% cost). EXT events in selection 14,167->3,017
  (79% event-level rejection from the per-shower cut alone). CC-stream
  composition transformed: sigCC fraction 0.48->0.65, out-of-FV 0.24->0.08.
  NC near-peak (eq2): NCpi0 purity 0.567, data/pred 0.89, EXT alone 33.1.
  SBND-style: purity 0.762->0.788 at identical eff 0.551.
  COMPARISON vs event-BDT baseline (recal3_bdt, both-halves 0.5909): the
  event BDT is stronger near-peak (EXT 6.5, purity 0.637, data/pred 0.99)
  — expected: it sees pair/event topology the per-shower score cannot.
  The two are complementary; whether showerCosmicScore REPLACES the event
  BDT as the official cut is an open user decision. The score's job —
  a chain-level, selection-agnostic cosmic tag in the ntuple — is done.
- Staged-cut sweep (staged_bdt_sweep.py, job 3195597): shower-score gate
  FIRST then event BDT, thresholds swept, hygiene = odd halves + EXT
  analysis half. VERDICT: yes, staging beats the event BDT alone at every
  matched efficiency <=0.97, by ~+0.7 to +1.5 points near-peak purity,
  and the gain GROWS with the shower threshold (ts=0.192 best tested:
  eff 0.97 purity 0.841 vs 0.827; eff 0.90 0.857 vs 0.844) with a
  correspondingly WEAKER event cut (t_e 0.21 vs 0.27 at eff 0.97).
  Above eff ~0.98 staging can't compete (shower cut alone costs ~1.5%
  signal). Mechanism: the shower gate removes MC non-signal (mcbkg
  1146->830) and EXT (363->49) the event BDT ranks poorly. Data/pred
  near-peak closure stable 0.91-0.94 across all ts. Plot:
  plots_s1ep2p8_diag/staged_bdt_sweep.png/.npz.
- STAGED 97% WP adopted for plots (job 3196902): shower>=0.192 then event
  BDT>=0.21 (hygiene: MC even-signal + EXT rows<100k/even excluded, held
  x2, EXT scale 1.1818). MC any-sel eff CC 0.789 / NC 0.701. EXT events
  in selection 14,167 -> 1,761 from the shower stage alone (87.6%), then
  47/195 more from the event stage. CC composition: sigCC 0.69 / outFV
  0.05 (uncut 0.48/0.24). NC near-peak eq2: NCpi0 purity 0.631, EXT 4.7,
  data/pred 0.94 (event-only baseline: 0.637/6.5/0.99 with double EXT
  stats). SBND (shower stage only, event BDT n/a to that pair): purity
  0.793 @ eff 0.551 — best yet. Plots: plots_s1ep2p8_ext_staged097/,
  plots_s1ep2p8_sbnd_staged097/.
- Flash-chi2 spectra at the staged WP, chi2 UNCUT (job 3199199; tables
  rebuilt with chi2 gates open so all WP+shower events get event-BDT
  scores): plots_s1ep2p8_flashchi2_staged097/. Data/pred closes on both
  sides of the cut (CC 0.93/0.93 at 1e4; NC 0.91/0.99 at 1778). The
  flash-blind BDTs leave a large above-cut EXT population (NC: 610 of
  777 predicted above-cut) -> the chi2 cut remains essential and fully
  complementary to the BDT stages.
- datamc_ext_overlay.py gained --combined-chi2-cut: combined CC+NC m_gg
  panels (eq2/ge2, no stream split, one common chi2 cut, signal = true
  CC+NC pi0) -> mgg_ext_all_{eq2,ge2}.png. At the staged WP with
  log10chi2<3.5 (3162): eq2 near-peak purity 0.882 (data/pred 0.96),
  ge2 0.851 (0.93). EXT: 34 (eq2) / 44 (ge2) weighted events total.
- flashchi2_from_tables.py gained --chi2-cut-combined: combined CC+NC
  spectra (flashchi2_all_{eq2,ge2}.png) with the single 3162 line at the
  staged WP, chi2 uncut. Region closure eq2: 0.93/0.94/0.93 (below /
  3.5-6 / >6), ge2: 0.92/0.96/1.01. Above-cut is EXT-dominated (eq2:
  586 of 789 pred).
- MC-only plot suite remade at the full latest setup (job 3204567):
  recal3 + shower>=0.192 + event BDT>=0.21 (via new pi0_mass_analysis
  --restrict-rows hook fed by an externally scored pass-row npz; all MC
  events scored, no hygiene split — truth plots) + per-stream chi2 WP ->
  plots_s1ep2p8_mc_staged097/ (same 15-plot set as plots_s1ep2p8_mc_exp).
  Signal median mass 137 MeV (target 135). Any-sel eff CC 0.786 /
  NC 0.684; CC-stream signal fraction 0.71, out-of-FV 0.04.
