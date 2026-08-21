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
