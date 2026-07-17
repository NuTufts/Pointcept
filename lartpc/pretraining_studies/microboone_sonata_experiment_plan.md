# MicroBooNE Spacepoint-Sonata Pretraining: Experiment Plan for Paper

**Purpose of this document:** Itemized study list for orchestrating pretraining runs,
fine-tuning/probing runs, and analyses toward a paper on self-distilled spacepoint
pretraining on real MicroBooNE data with ghost contamination.

**Intended use:** A Claude Code session should use this to (1) generate configs,
(2) submit and track SLURM jobs, (3) collect metrics into the run registry, and
(4) produce the analysis artifacts listed per study.

**Repository:** Nutufts/Pointcept. Base configs live in `configs/lartpc/`. SLURM
templates in `slurm_scripts/lartpc_sonata_pretraining/`.

---

## STATUS (2026-07-17) — what has been launched / completed

Infrastructure and per-decision detail live in
`phase0_phase05_implementation_plan.md` (Phase 0/0.5) and
`probe_orchestration/RESULTS_WAVE_A_DECISION.md` (Wave A→C probe verdict).
Registry: `exp/registry.csv` on Isambard. Allocation ends **2026-07-22**.

**Phase 0.5 — COMPLETE except deferred items.** All at P05_BUDGET = 5M images
(1/3 matched), MC-only ghost-dropped lists, batch 48 / lr 2e-4:

- **P05A supervised ceilings (done):** A.1 mIoU **0.8576** (pion 0.773,
  proton 0.933); A.2 charge-zeroed 0.8152 → charge info = **+0.114 pion,
  +0.087 proton IoU**; A.3 free rotations 0.8596 (null — rotations don't hurt
  *supervised* training). A.4 (class-balanced) deferred. A.5 (asinh input
  scaling, from the P05F study below) launched 07-17.
- **Wave A SSL (8 runs, done/finishing 07-17/18):** B.1, B.2, B.4, C.1, C.3,
  C.4, C.5, C.6 (C.2≡B.1 by reuse; B.3 blocked on P0.9 wire projections).
- **Wave B probes (Tufts, decision-level done):** verdict at matched 768k/1.5M
  images — composition dominates (C.5 drop_cosmics=0.9: +7.7 mIoU, +10 pion,
  +20 proton over B.1); detsym augs +2.4 (rotation/charge hypothesis confirmed,
  second order); prototype count dead; 2×crops hurts; B.4 sum-charge a wash.
  Curve-filling probes resume after the fleet finishes.
- **P05F input-scaling study (NEW, done):** log ranks last on every μ/π/p
  pair; clip at 1000 ADC is a non-issue → **P05B.5** (asinh(50,1000), matched
  jitter σ=0.125) and **P05B.6** (σ=0.05) launched 07-17.
- **Wave C:** E.1 config generated (detsym × drop_cosmics=0.9); training
  deferred by PI decision (its cosmic-drop knob cannot transfer to Phase 1
  anyway — see below); can run later at Tufts or in spare Isambard capacity.

**Phase 1 — P1A LAUNCHED 2026-07-17** (jobs 5691894–97) at MATCHED_BUDGET.
**Base = "v9-provisional" restricted to TRUTH-FREE knobs** (detsym augs, 4096
prototypes, log scaling, NO drop_cosmics): the Wave A composition winner
requires MC truth, cannot run on the EXTBNB cells, and its keep-mask
(origin==1) would strip P1A.2's ghosts — so it is excluded from the 2×2 base
for internal validity and remains an MC-side result (C.5 / optional E.1).
P1A.5 (25th percentile) cut: list unavailable, plan already marked it a
candidate to cut. Runs reach ~60% of budget by allocation end; log-spaced
snapshots (15/run) + `model_last.pth` support resuming at Tufts (A100s, same
container) or a future allocation. Configs:
`configs/lartpc/p05/pretrain-sonata-p1a{1,2,3,4}-*.py`.

**P5B mixture pretraining — LAUNCHED 2026-07-17** (promoted from §6; jobs
5692138/5692140), same base/budget as P1A: **P5B.1** raw 1:1 MC+EXTBNB
mixture (both domains with natural ghosts; pairs with P1A.2/P1A.3) and
**P5B.2** cleaned mixture (MC ghost-dropped, EXTBNB LArMatch-filtered; pairs
with P1A.1/P1A.4). 18 epochs × the line-interleaved 831,360-file list =
MATCHED_BUDGET exactly, half per domain. Motivation: one-foundation-model
question for the reconstruction cascade + embedding-space domain-shift study
(data-vs-MC axis, cosmic prototype sharing, nu-origin representation) —
analysis runs on frozen features at Tufts against the frozen MC/EXTBNB
diag1k sets. **E.1 remains on hold** (PI, 2026-07-17).

**LArMatch naming-bug fix + symmetric-filter runs — LAUNCHED 2026-07-17**
(jobs 5692768/5692769): the MC production stores the LArMatch score as
`lm_score` (writer mismatch vs EXTBNB's `larmatch_score`), so every prior
"filtered" run silently left MC unfiltered — including v8, now understood as
an asymmetric mixture. Fixed via an opt-in `larmatch_score_keys` dataset
parameter (default preserves legacy behavior bit-identically; gate-tested).
New runs: **P1A.4b** (MC + stochastic LArMatch filter — vs P1A.4 isolates
pure domain at matched preprocessing; vs P1A.1 isolates reco-vs-truth
cleaning within MC) and **P5B.3** (symmetric LArMatch-filtered 1:1 mixture,
no truth anywhere — the method-symmetric alternative to P5B.2).

---

## 0. Conventions

### 0.1 Run ID scheme

```
{phase}{study}.{variant}-{dataset}-s{seed}
e.g.  P1A.1-extbnb_full-s0
      P2B.3-mc_ghosts-s0
```

### 0.2 Directory layout (adapt to cluster paths)

```
exp/
  registry.csv                  # master run registry (see §7)
  configs/                      # generated configs, one per run, named by run ID
  logs/                         # slurm stdout/err per run
  checkpoints/{run_id}/         # model_last.pth, model_best.pth
  probes/{run_id}/{task}/       # probe/fine-tune outputs per backbone
  analysis/{study_id}/          # figures + tables per study
```

### 0.3 Base configs (starting points for deltas)

| Alias | File | Notes |
|---|---|---|
| `BASE_PRETRAIN` | `configs/lartpc/pretrain-sonata-v7-extbnb-larmatch.py` | EXTBNB + stochastic LArMatch filtering |
| `BASE_PRETRAIN_MIX` | `configs/lartpc/pretrain-sonata-v8-extbnb-mc-combined-larmatch.py` | EXTBNB+MC mixture |
| `BASE_PRETRAIN_MC` | v6-family MC config (ghosts dropped at load time) | |
| `BASE_LINPROBE` | `configs/lartpc/linearprobe-sonata-v1m1-segmentation-v5.py` | frozen backbone + MLP head |
| `BASE_LORA_DEGHOST` | Vinicius's LoRA deghosting config | frozen backbone + LoRA |
| `BASE_LORA_SEMSEG` | Vinicius's LoRA semseg config | true points only |

### 0.4 Canonical matched pretraining budget (**MUST be identical across all Phase-1+ runs**)

Set once, then never vary within a comparison:

```python
MATCHED_BUDGET = dict(            # FROZEN 2026-07-17 (P0.1 done; used by P1A)
    images_seen   = 14_964_480,   # = 36 epochs x 415,680 files (both MC and the
                                  # trimmed EXTBNB list are exactly 415,680, so
                                  # every cell sees identical images/epoch)
    batch_size    = 48,           # locked 2026-07-13 (batch 80 OOM'd on big events)
    peak_lr       = 2e-4,         # REVISED from 5e-4: 2e-4 is the only value
                                  # validated free of Adam-state spikes (v8 run);
                                  # locked 2026-07-13, used by all P05/P1A runs
    lr_schedule   = "OneCycleLR, pct_start=0.06, div_factor=100, final_div=1000",
    seed          = 0,            # flagship configs also get s1, s2 (see §6)
)
```

Iterations = `images_seen / batch_size`. All schedulers (mask size/ratio, teacher temp,
momentum, wd) are fractions of total iters, so they rescale automatically.

### 0.5 Standard evaluation suite ("EVALSUITE")

Every pretrained backbone is evaluated with the same suite. Each item is a separate
job with the backbone frozen unless stated:

| Eval | Protocol | Dataset | Metrics |
|---|---|---|---|
| E1 deghost | LoRA, frozen backbone | MC (all points incl. ghosts) | mIoU, real IoU, ghost IoU |
| E2 semseg | LoRA, frozen backbone | MC (true points only) | mIoU + per-class (muon, e, p, gamma, LED, delta, michel, pion) |
| E3 semseg linprobe | linear/MLP probe, frozen | MC (true points) | mIoU + per-class |
| E4 deghost linprobe | linear probe, frozen | MC (all points) | AUC, mIoU |
| E5 sliced evals | reuse E1/E2 predictions | MC | metrics split by: nu vs cosmic origin; distance to nu vertex; local ghost density bins |

**Uncertainties:** every reported metric gets a bootstrap over evaluation events
(≥1000 resamples, report mean ± 68% CI). This is required — headline differences are 1–3%.

**Fixed probe budget:** define once (e.g., N fine-tune iterations, LoRA rank, probe LR)
and reuse for every backbone. Record in registry.

---

## 1. Phase 0 — Infrastructure (blocking; no GPUs needed beyond smoke tests)

| ID | Task | Deliverable |
|---|---|---|
| P0.1 | Set `MATCHED_BUDGET` from measured throughput; document in this file | updated §0.4 |
| P0.2 | Bootstrap error-bar utility for all eval metrics | script + unit test |
| P0.3 | Sliced-eval machinery (E5): nu/cosmic tags, vertex distance, ghost-density estimator (e.g., ghost count in r=5cm ball / total) | script |
| P0.4 | MC-with-ghosts dataloader path: keep ghosts at pretraining load time (currently dropped); ghost-retention fraction as a config knob `ghost_keep_frac` ∈ [0,1] using MC ghost truth | dataloader flag + config knob |
| P0.5 | Scheduled LArMatch threshold: threshold as a function of training progress (for P2C); implement `larmatch_threshold_schedule = dict(start=0.75, end=0.15, warmup_ratio=...)` alongside existing stochastic mode | transform option |
| P0.6 | Run registry (`registry.csv`) with columns per §7; auto-append on submit and on completion | script |
| P0.7 | Smoke test: 500-iteration run of each base config on current cluster env | pass/fail log |
| P0.8 | Diagnostic metric tooling for Phase 0.5: prototype–label MI (M1), prototype purity (M2), charge-ablation probe harness (M3), batch-composition logging hook (M5) | scripts |
| P0.9 | Fill in MicroBooNE `wire_projections` plane geometry (currently TODO stub in configs) and implement post-transform recomputation of wire-projected features (needed by P05B.3) | transform option |
| P0.10 | Audit augmentation policy of existing checkpoints (v6 free rotations vs v7 no-rotations) and record per checkpoint in registry | registry entries |

---

## 1.5 Phase 0.5 — Clean-MC reference: supervised ceiling + representation-quality diagnostics

**Question:** How much of the gap to PANDA-style representation structure (p/π/μ isolation)
is intrinsic to tomographic spacepoint inputs vs fixable in the SSL configuration?
Establish a supervised ceiling on identical inputs, identify the root cause of weak PID
structure (leading hypothesis: rotation augmentations with frozen wire-plane charge
features teach the model to discount charge), and freeze a "v9 reference config" that
all Phase-1+ studies build on.

**All Phase 0.5 runs use ghost-dropped MC** (ghost truth applied at load time) unless noted.
**Compute posture: GPU-rich (Isambard).** The design below is embarrassingly parallel:
Wave A is ~17 independent GPU jobs submitted simultaneously; Wave B is auto-queued probes;
Wave C is 1–2 combination runs. Do NOT serialize Wave A.

### Budget for Phase 0.5 pretraining runs

```python
P05_BUDGET = dict(
    images_seen = MATCHED_BUDGET["images_seen"] // 3,   # reduced; identical across all P05 runs
    batch_size  = MATCHED_BUDGET["batch_size"],
    peak_lr     = MATCHED_BUDGET["peak_lr"],
    seed        = 0,
)
```

### New metrics for this phase (add to EVALSUITE tooling; P0 tasks)

| ID | Metric | Definition |
|---|---|---|
| M1 | Prototype–label MI | mutual information between hard prototype assignment and truth particle label, on fixed 1k-event MC diagnostic set |
| M2 | Prototype purity | per-prototype majority-class fraction, occupancy-weighted mean |
| M3 | Charge-ablation delta | probe metric with `strength` channels intact minus zeroed/shuffled at inference (measures charge reliance) |
| M4 | PID confusion structure | per-class probe IoU + confusion matrix, with p/π and p/μ off-diagonals reported explicitly |
| M5 | Batch composition log | per-batch truth-class point fractions, logged during pretraining (one-line hook) |
| M6 | Fraction-of-ceiling | probe metric / supervised-ceiling metric (P05A), per class |

### Wave A — submit ALL simultaneously (independent jobs)

**P05A — Supervised ceiling (no pretraining; anchors everything)**

| Run | Description | Config |
|---|---|---|
| P05A.1 | Supervised PTv3 point-level semseg/PID, ghost-dropped MC, exact v6/v7 feature pipeline (log-ADC strength + coords) | supervised semseg config, no pretrained weights, generous budget |
| P05A.2 | P05A.1 with strength channels zeroed at train+test ("geometry-only ceiling") | delta: zero `strength` |
| P05A.3 | P05A.1 with PANDA-style augmentations (free rotations, charge frozen) instead of detector-symmetry-only | isolates whether rotation+frozen-charge hurts even supervised training |
| P05A.4 | (optional, cheap) P05A.1 trained per-class-balanced sampling | checks whether pion ceiling is label-scarcity limited |

P05A.1 minus P05A.2 = total calorimetric information available in your features.
This difference is the reviewer-facing number: if it is small, weak p/π isolation is
intrinsic to the dataset, not the SSL method.

**P05B — Augmentation policy (root-cause test for missing PID structure)**

Pretraining runs at `P05_BUDGET`, one delta each from the v7-style base:

| Run | Spatial augmentation policy | Charge features |
|---|---|---|
| P05B.1 | Free 3-axis rotations + flips (v6-style, replicate current behavior) | frozen pixvals (as trained) |
| P05B.2 | **Detector symmetries only: y-flip, z-flip, small jitter, NO rotations** (v7 header policy) | frozen pixvals |
| P05B.3 | Free rotations | wire-projected features **recomputed after transform** (use `wire_projections` machinery; fill in MicroBooNE plane geometry — currently a TODO stub in configs) |
| P05B.4 | Detector symmetries only | plane-summed / de-duplicated charge scalar instead of 3 raw pixvals (more nearly rotation-compatible feature) |

First action for the orchestrator: **audit which augmentation policy the existing
trained checkpoints actually used** (v6 configs rotate; v7 header says no rotations)
and record it in the registry — this determines whether P05B.1 can reuse an existing
checkpoint.

**P05C — Prototype count × batch composition (Sinkhorn/imbalance interaction)**

All at `P05_BUDGET` on the current-default augmentation (do not wait for P05B verdict;
GPU-rich, and the winner is combined in Wave C):

| Run | `head_num_prototypes` | Batch composition |
|---|---|---|
| P05C.1 | 2048 | default `BiasedSphereCrop` |
| P05C.2 | **4096** (reuse a base run if identical) | default |
| P05C.3 | 8192 | default |
| P05C.4 | 4096 | nu-anchored crop probability raised (lower `prob_random`, e.g. 0.10 → keep, sweep anchor prob up) |
| P05C.5 | 4096 | high cosmic-drop fraction (replicates user's prior experiment, now with M1/M2 diagnostics) |
| P05C.6 | 4096 | 2× crops per batch at ~half points each (fixed total points; higher per-batch semantic diversity) |

All P05C runs log M5 (batch composition) so the "Sinkhorn sees mostly muon points"
hypothesis becomes a measured number.

**P05D — Diagnostics on EXISTING checkpoints (GPU-light, submit immediately)**

| Run | Description |
|---|---|
| P05D.1 | Charge-ablation probe (M3) on current best frozen backbone(s) — no retraining |
| P05D.2 | M1/M2 prototype–label MI + purity on all existing checkpoints |
| P05D.3 | Recompute t-SNE at PANDA-comparable settings (upcast level / feature scale ~12 mm equivalent, unit-normalized) colored by class — rule out visualization mismatch before concluding structure is absent |

### Wave B — auto-queued on each Wave A completion

For every P05B/P05C backbone: E3 (semseg MLP probe), M1–M4, and a t-SNE panel.
Selection metric = **prototype–label MI (M1) + per-class probe IoU (M4)**, with special
weight on p/π/μ off-diagonals — not t-SNE aesthetics, not mIoU alone.

### Wave C — combination (1–2 runs)

| Run | Description |
|---|---|
| P05E.1 | Best augmentation policy (P05B) × best prototype/batch setting (P05C), at `P05_BUDGET` → if it confirms, promote to **v9 reference config** |
| P05E.2 | (only if P05B and P05C winners interact ambiguously) second combination |

### Acceptance gate for exiting Phase 0.5 (reviewer-facing framing)

The gate is **NOT** "match PANDA's absolute PID separation" — part of that gap is
intrinsic to tomographic charge sharing. Exit criteria:

1. Supervised ceiling (P05A.1) measured with CIs, including geometry-only ceiling (P05A.2).
2. v9 reference config reaches ≥ X% fraction-of-ceiling (M6) on per-class probe IoU
   (set X after seeing P05A; PANDA-analog SSL/supervised ratios suggest ~90–95% for
   probe+FT, lower for pure linear probe).
3. Charge-ablation delta (M3) for v9 is a substantial fraction of the supervised
   charge delta (P05A.1 − P05A.2) — i.e., the SSL model demonstrably *uses* charge.
4. Root cause of prior missing PID structure identified and documented (P05B verdict).

**Paper framing unlocked by this phase:** "SSL performance as fraction of supervised
ceiling on identical realistic tomographic inputs" — quantifies what detector realism
costs SSL, independent of the ghost study, and preempts the reviewer critique that
ghost-study effects ride on a weak baseline.

### Downstream impact

- `v9 reference config` replaces the v7-style base for ALL Phase 1+ runs
  (`BASE_PRETRAIN`, `BASE_PRETRAIN_MC` aliases repoint to v9 variants).
- P05A.1/P05A.2 rows enter the paper's main table as supervised ceilings.
- If P05B shows the rotation/charge effect, it becomes its own methods-section result
  ("charge-consistent augmentation for tomographic point clouds").

---

## 2. Phase 1 — Core matched-budget comparison (the paper's main table)

**Question:** separate domain (MC vs data) from contamination (ghosts vs removed).
All runs use `MATCHED_BUDGET`. All get full EVALSUITE.

### Study P1A — 2×2 domain × contamination + prelim replication

| Run | Dataset / prep | Config delta from base | Status |
|---|---|---|---|
| P1A.1 | MC, ghosts dropped ("MC-clean") | v9-provisional base, matched budget | **LAUNCHED 07-17** (job 5691894) |
| P1A.2 | **MC, ghosts kept ("MC-ghosts")** | + `ghost_keep_frac=1.0` (P0.4), `include_ghosts=True` | **LAUNCHED 07-17** (job 5691895) — validated: crops ~50% ghost points |
| P1A.3 | EXTBNB full, ghosts kept | `data_only=True`, no LArMatch filter, trimmed 415,680-file list | **LAUNCHED 07-17** (job 5691896) — at 2e-4 LR per frozen §0.4 |
| P1A.4 | EXTBNB + LArMatch stochastic filter | + `filter_larmatch=True`, threshold U[0.15,0.75] (v7 range) | **LAUNCHED 07-17** (job 5691897) |
| P1A.5 | EXTBNB 25th percentile | prelim config at matched budget | **CUT 07-17** — percentile list unavailable; plan already marked candidate to cut |

**Analysis (analysis/P1A/):**
- Main results table: 5 backbones × EVALSUITE metrics with CIs (paper Table 1).
- Loss curves overlay, plateau value vs dataset (paper "pretraining dynamics" figure).
- Explicit 2×2 interpretation: `(P1A.1 vs P1A.2)` isolates ghosts in MC; `(P1A.2 vs P1A.3)` isolates domain at fixed contamination; `(P1A.3 vs P1A.4)` isolates preprocessing on data.

**Dependencies:** P0.1, P0.4, P0.7, and the **v9 reference config from Phase 0.5** (all
P1 pretraining runs use v9 as base — do not launch P1 with the v7-style base).

### Study P1B — Baselines

| Run | Description | Config |
|---|---|---|
| P1B.1 | From-scratch PTv3, deghosting task, matched fine-tune budget | supervised config, no pretrained weights |
| P1B.2 | From-scratch PTv3, semseg task | same |
| P1B.3 | Random-init frozen backbone + E3/E4 probes | no pretraining; probe configs pointed at random weights |
| P1B.4 | LANTERN deghosting evaluated with identical E1 metrics/splits | analysis-only if predictions available |

**Analysis:** transfer-efficiency figure (validation metric vs images-seen: pretrained
fine-tune vs from-scratch — PANDA Fig. 10 analog); random-baseline row in main table.

---

## 3. Phase 2 — Ghost dose-response and curriculum (the paper's novel science)

### Study P2A — Ghost-fraction dose-response (MC, ghost truth available)

Pretrain at controlled contamination; everything else = P1A.2 config.

| Run | `ghost_keep_frac` |
|---|---|
| P2A.1 | 0.0 (≡ P1A.1, reuse) |
| P2A.2 | 0.25 |
| P2A.3 | 0.50 |
| P2A.4 | 0.75 |
| P2A.5 | 1.0 (≡ P1A.2, reuse) |
| P2A.6 | fraction matched to measured EXTBNB ghost fraction (compute and record) |

**Analysis:** downstream metric vs ghost fraction curves (E1–E4), plus loss-plateau vs
fraction. This is the flagship figure: "how much structured noise can self-distillation
tolerate."

**Dependencies:** P0.4, P1A complete enough to fix probe budgets. 4 new pretraining runs.

### Study P2B — Mask curriculum under ghosts

Base: P1A.4 config (EXTBNB + LArMatch). Sweep on the **25th-percentile subset** at a
**reduced budget** (e.g., 1/3 of matched, fixed for all P2B/P2C/P2D) with E3/E4 linear
probes + E1 as selection metrics. One variant changed at a time.

| Run | Parameter | Values (base in **bold**) |
|---|---|---|
| P2B.1–3 | `mask_size_base` | 7.5, **15**, 30 cm (note ~14 cm radiation length in LAr) |
| P2B.4–5 | `mask_ratio_base` | 0.5, **0.7**, 0.9 |
| P2B.6–7 | `mask_{size,ratio}_warmup_ratio` | **0.06** vs 0.0 (no curriculum) vs 0.2 |
| P2B.8–10 | `mask_jitter` | 0, 0.5×, **1×** (=0.125/coord_scale), 2× grid size |

P2B.6 (no-curriculum) directly answers "do we need the curriculum at all under ghosts" —
run it on BOTH a ghost-heavy dataset and MC-clean (2 runs) so the ghost×curriculum
interaction is measurable.

### Study P2C — Ghost curriculum (LArMatch threshold policy)

Base and budget as P2B. Requires P0.5.

| Run | Policy |
|---|---|
| P2C.1 | **stochastic U[0.25,0.75]** (≡ base, reuse P1A.4-style run at reduced budget) |
| P2C.2 | fixed 0.15 (permissive) |
| P2C.3 | fixed 0.50 |
| P2C.4 | fixed 0.75 (aggressive) |
| P2C.5 | scheduled: 0.75 ⇒ 0.15 over first 25% of training ("clean-to-noisy" curriculum) |
| P2C.6 | scheduled: 0.15 ⇒ 0.75 ("noisy-to-clean", control for direction) |

**Analysis:** table + short narrative; if P2C.5 wins, it's a headline methodological
finding and gets promoted to a full-budget P1-style run.

### Study P2D — View generation and matching under ghosts

| Run | Parameter | Values (base **bold**) |
|---|---|---|
| P2D.1–2 | `match_max_r` | 2.5, **~4–5 cm** (16×grid_size), 10 cm |
| P2D.3 | `local_view_num` | 4 vs **6** |
| P2D.4 | `local_view_scale` | (0.05,0.4) vs **(0.1,0.4)** |

**Phase-2 decision gate:** winning settings from P2B/P2C/P2D are combined into one
"tuned" config and rerun at full `MATCHED_BUDGET` on EXTBNB+LArMatch → becomes P1A.6
("tuned") row in the main table.

---

## 4. Phase 3 — Protocol ladder and label efficiency

### Study P3A — Adaptation-protocol ladder (parameter efficiency)

For the 3–4 most important backbones (P1A.1, P1A.2, P1A.4, tuned P1A.6), evaluate
semseg and deghosting under:

| Protocol | Trainable params |
|---|---|
| linear probe | ~0.01M |
| MLP probe (existing BASE_LINPROBE) | ~1.2M |
| LoRA frozen backbone (existing) | LoRA rank params |
| full fine-tune | all |

**Analysis:** parameter-efficiency table (Sonata Tab. 2 / PANDA Tab. 2 analog).
Key claim to test: ghost-contaminated pretraining hurts *linear* separability more than
it hurts full fine-tuning.

### Study P3B — Label efficiency

Backbones: P1A.2 (MC-ghosts), P1A.4 (data+filter), P1A.1 (MC-clean), from-scratch.
Fine-tune (LoRA + full FT) on K ∈ {0.1%, 1%, 10%, 100%} of labeled MC. Fixed
fine-tune step count across K (oversample small K; monitor probe for collapse per
PANDA §3.5).

**Analysis:** metric-vs-K curves per backbone (PANDA Fig. 6 analog). Watch for ranking
crossovers at low K — a publishable outcome on its own.

Runs: 4 backbones × 4 K × 2 tasks (batchable; small jobs).

---

## 5. Phase 4 — Representation analysis (analysis jobs, minimal GPU)

Applied to P1A backbones + tuned config. Inputs: frozen features on a fixed 1k-event
MC eval set (with ghost truth) + 1k EXTBNB events.

| ID | Analysis | Deliverable |
|---|---|---|
| P4.1 | t-SNE/PCA of point embeddings colored by (a) semantic class, (b) true/ghost | figure grid across backbones |
| P4.2 | Prototype occupancy: histogram of prototype assignments split true vs ghost; count "ghost-dominated" prototypes | figure + summary stat |
| P4.3 | Linear decodability of ghost/true from frozen features (≡ E4, aggregated across backbones) | AUC bar chart |
| P4.4 | k-means on frozen features, clustering vs semantic labels (unsupervised segmentation, Sonata-style) | qualitative event displays |
| P4.5 | Real-data qualitative: deghosting predictions on EXTBNB events, MC-pretrained vs data-pretrained backbone, side-by-side event displays; if available, LArMatch-agreement proxy metric on data | figure + proxy table |

---

## 6. Phase 5 — Optional / stretch (Isambard)

| ID | Study | Runs |
|---|---|---|
| P5A | Data scaling: EXTBNB+LArMatch pretraining at 10k / 100k / 857k events, fixed iterations | 2 new (857k = reuse) |
| P5B | Mixture pretraining (BASE_PRETRAIN_MIX) at matched budget, EVALSUITE | 1; interpret against P1A 2×2 |
| P5C | Seeds: s1, s2 for flagship configs (P1A.2, P1A.4, tuned) | 6 pretraining runs |
| P5D | One instance-level downstream task (particle clustering or keypoints) on top 2 backbones | scope-limited; only if schedule allows |

---

## 7. Run registry schema (`registry.csv`)

```
run_id, study, dataset, config_path, config_hash, base_config, deltas,
budget_images_seen, batch, peak_lr, seed,
slurm_jobid, status(queued|running|done|failed), submit_time, done_time,
checkpoint_path, wall_hours, gpu_type,
eval_E1_mIoU, eval_E1_realIoU, eval_E1_ghostIoU,
eval_E2_mIoU, eval_E3_mIoU, eval_E4_AUC,
ci_lo/ci_hi per metric, notes
```

Rules for the orchestrator:
1. Never submit a Phase-1+ pretraining run whose budget/batch/LR differ from §0.4 unless the study explicitly sweeps that knob; refuse and flag instead.
2. Every submitted run gets its exact generated config committed/copied to `exp/configs/` and its hash recorded before submission.
3. On completion, automatically queue the EVALSUITE jobs for that backbone (E1–E5), then the analysis scripts for the parent study.
4. Reuse runs where marked (≡) — do not resubmit duplicates.
5. Alert (do not silently continue) if loss plateaus >2× above sibling runs or NaNs — likely collapse; record in notes.

---

## 8. Priority order and dependency summary

```
P0.* (infra) ──► P0.5 Wave A (parallel: P05A ceiling ‖ P05B augmentation ‖ P05C proto/batch ‖ P05D diagnostics)
                     │                     └─► Wave B probes (auto) ─► Wave C combo ─► v9 reference config
                     ▼
                 P1A (2×2 on v9, incl. NEW MC-ghosts run) ──► P1B baselines
                     │
                     ├──► P2A dose-response  ──┐
                     ├──► P2B mask curriculum ─┼──► tuned config (P1A.6, full budget)
                     ├──► P2C ghost curriculum ┘
                     ├──► P2D views/matching
                     │
                     ├──► P3A protocol ladder ──► P3B label efficiency
                     ├──► P4.* representation analysis
                     └──► P5.* stretch
```

Single most important new runs: **P05A.1/P05A.2 (supervised ceiling ± charge)** — they
anchor every claim in the paper — and **P1A.2 (MC with ghosts kept)** — it completes the
2×2 that separates the paper's central claim (noise obscures structure) from the
sim-to-real domain gap.

Parallelism note for Isambard: Phase 0.5 Wave A (~17 jobs) and the P1B from-scratch
baselines (which do not depend on v9) can all be in the queue on day one. P2A–P2D within
Phase 2 are also mutually independent once v9 and P1A land.

## 9. Mapping to paper sections

| Paper element | Source studies |
|---|---|
| Supervised ceiling rows (± charge) in main table; "fraction of ceiling" framing | P05A |
| Charge-consistent augmentation result (if confirmed) | P05B, P05D.1 |
| Prototype/batch-composition ablation; prototype–label MI figure | P05C, P05D.2 |
| Table 1: main matched-budget comparison | P1A (+P1B, P05A rows) |
| Fig: dose-response (metric vs ghost fraction) | P2A |
| Fig: pretraining loss plateau vs contamination | P1A + P2A logs |
| Table: parameter efficiency (probe ladder) | P3A |
| Fig: label efficiency curves | P3B |
| Fig: t-SNE / prototype occupancy for ghosts | P4.1–P4.3 |
| Table/Fig: curriculum ablations | P2B, P2C, P2D |
| Fig: transfer efficiency vs from-scratch | P1B logs |
| Real-data evaluation / LANTERN comparison | P4.5, P1B.4 |
| (Optional) scaling | P5A |
