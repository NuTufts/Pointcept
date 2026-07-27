# Program status & Tufts hand-off (2026-07-23, post-Isambard)

**Read this first.** The Isambard allocation ended 2026-07-22 (~98% of 2,400
node-hours used; queue drained). All orchestration moves to Tufts. The
preservation archive (575 files / 273 GB: every snapshot, every run's
`model_last.pth` resume state, train logs, tfevents, configs, registry) is
at Tufts — verify with
`probe_orchestration/check_transfer.py` → must print `TRANSFER COMPLETE`.

Document map (all under `lartpc/pretraining_studies/`):

| doc | what it holds |
|---|---|
| `microboone_sonata_experiment_plan.md` | master study list + dated STATUS log + P5E grid + P6 future work |
| `phase0_phase05_implementation_plan.md` | Phase 0/0.5 infrastructure + decisions + results log |
| `probe_orchestration/ORCHESTRATION.md` | probe sweep protocol (frozen budget, mappings, gotchas) |
| `probe_orchestration/RESULTS_WAVE_A_DECISION.md` | Wave A probe verdict + frozen probe budget tokens |
| `p5a_scaling_study_plan.md` | data-scaling design (dual-row) |
| `input_dist_study/README.md` + `P05B5_IMPLEMENTATION_HANDOFF.md` | P05F verdict + asinh implementation record |
| this file | run-by-run state + take-over sequence |

## 1. Headline results already in hand

- Supervised ceilings (val, matched split): **A.1 log 0.8576** (pion 0.773,
  proton 0.933); A.2 charge-zeroed 0.8152 → charge = **+0.114 pion / +0.087
  proton IoU**; A.3 free-rotations 0.8596 (null); **A.5 asinh 0.8549 (null —
  as predicted; makes B.5/B.6 probes the decisive asinh test).**
- Wave A probe verdict (mid-training): composition dominates (C.5 +7.7 mIoU,
  +10 pion, +20 proton over B.1); detsym +2.4; prototype count dead; 2×crops
  hurts; sum-charge a wash.
- P05F: log input scaling ranks last on every μ/π/p pair; 1000 ADC clip is a
  non-issue; asinh(50,1000) adopted for B.5/B.6/P5E.
- lm_score naming bug found+fixed (opt-in `larmatch_score_keys`); v8 is
  retroactively an asymmetric mixture.

## 2. Run-by-run state (extracted from train logs 2026-07-23)

**COMPLETE (full budget; ready for full probe curves + analysis):**

| run | note |
|---|---|
| P05A.1/.2/.3/.5 | supervised ceilings (results above) |
| P05B.1/.2/.4, P05C.1/.3/.4/.5/.6 | Wave A SSL, 11 snapshots each |
| P05B.5/.6 | **asinh SSL pair — probe these first** (vs B.1: the asinh verdict; B.6 vs B.5: jitter strength) |
| P5E.S5/.S6/.S7, M3, L1 | complete C/4 compute tier incl. S-width LR sweep → first width-scaling fit is possible NOW |

**PAUSED at allocation end (resume state verified; % of budget):**

| run | progress | snaps |
|---|---|---|
| P1A.1-mc_clean | 40% (ep 15/36) | 9 |
| P1A.2-mc_ghosts | 52% (ep 19/36) | 11 |
| P1A.3-extbnb_full | 45% (ep 17/36) | 10 |
| P1A.4-extbnb_larmatch | 43% (ep 16/36) | 10 |
| P1A.4b-mc_larmatch | 47% (ep 18/36) | 10 |
| P5B.1/.2/.3 mixtures | 43–47% (ep 8–9/18) | 10 each |
| P5E.S1/S2/S3/S4 (S width, C) | 66–69% | 12 each |
| P5E.M1/M2 (M width, C) | 60–61% | 11 each |
| P5E.L0 (grid anchor, C) | 46% (ep 17/36) | 10 |
| P5A row-M ×3 + row-E ×3 | 28–29% | 8 each |
| (v8 legacy combined) | 73% (ep 37/50) | — |

## 3. Take-over sequence (priority order)

**P0 — verify the archive** (once): `python3 check_transfer.py` → TRANSFER
COMPLETE. The per-epoch `epoch_*.pth` piles were deliberately NOT archived
(redundant; 2.6 TB); everything needed to resume/probe/analyze is present.

**P1 — probe backlog (GPU, biggest science per hour).** Frozen budget
(WP3.5-calibrated, do not change):
`epoch=1 eval_epoch=1 optimizer.lr=0.001 "scheduler.max_lr=[0.001]"`,
batch 64 on `--constraint=a100`. Run
`launch_probe_sweep.py --budget-opts "<tokens above>"` — the PROBE_MAP
handles all families automatically (width-matched configs for P5E.S/M,
asinh config for B.5/B.6, sum-charge for B.4, prototype overrides for
C.1/C.3, standard for the rest). Probe-first order:
1. **P05B.5/B.6 full curves** → the asinh SSL verdict (compare to B.1;
   supervised was null, so any probe gain is SSL-specific).
2. Wave A curve completion (the ~46 cancelled curve-filling probes; the
   `.submitted` markers were already cleared).
3. Complete C/4 P5E tier (5 runs × 9 snapshots) → fit loss & probe vs
   width at fixed C/4 (first scaling-law readout).
4. Partial curves for the paused families (their snapshots are valid
   curve points regardless of resumption).

**P2 — analyses needing no training:** the P5B embedding/domain study
(diag1k sets are frozen and archived; mixtures have matched ~45% snapshots
vs P1A.2/P1A.3 at the same images-seen anchors); the P0.2 bootstrap-CI
utility (still pending — required before any result enters the paper);
M1/M2 prototype tooling adaptation; write-ups mirroring
`RESULTS_WAVE_A_DECISION.md` for each finished comparison.

**P3 — resume paused pretraining on Tufts A100s** (as GPU time allows).
No pretraining sbatch exists for Tufts yet — build one from
`submit_probe_tufts.sbatch` (same container `pointcept_cuml.sif`,
`--bind /cluster:/cluster`) with N-GPU srun and
`tools/train.py --num-gpus N --options resume=True weight=<run>/model/model_last.pth`.
Notes: batch 48 total = 12/GPU fit in 96 GB on GH200; A100-80 should hold
it (calibrate before committing; A100-40 likely needs num-gpus 8 or grad
accumulation). Mid-epoch resume + snapshot machinery is config-driven and
works unchanged. Resume priority: **P1A five cells** (the paper's core
table) → P5B mixtures → P5E.L0 + M1/M2 + S1–S4 (S cells are ~2/3 done and
cheap to finish) → P5A rows. Expect the ~50-iteration replay seam in wandb
at each resume (known artifact, documented 2026-07-19).

**P4 — deferred backlog:** E.1 (config exists, on hold by PI); P05B.3
wire-projections (needs WP7); P05A.4 class-balanced ceiling; P5E XL column
(next allocation); P6 emergence suite (§6.5 of the experiment plan — pure
analysis over the growing checkpoint array).

## 4. Gotchas (each cost real time once — don't repeat them)

1. Isambard ssh (if the archive needs re-pulling): clifton cert ~12 h;
   `./clifton auth` (human-interactive) on failure.
2. SLURM opens `--output` before the script runs: `mkdir -p logs` before
   first sbatch or every job dies instantly with exit code 53.
3. `base_lr` is parse-time-resolved: LR overrides must set BOTH
   `optimizer.lr` and `scheduler.max_lr=[...]`.
4. Checkpoint head sizes must match probe configs: C.1/C.3 need
   `model.backbone.head_num_prototypes` overrides; P5E.S/M need the
   width-matched probe configs; B.5/B.6 need the asinh probe config;
   B.4 the sum-charge config (all wired in PROBE_MAP).
5. Probe numbers are ceiling-comparable ONLY on the Tufts-remapped matched
   split (`*_tufts.txt` lists) — never the old prod4 lists.
6. Do not "fix" the legacy log_space jitter (±13% effective — intentional
   for pre-asinh comparability) or the loader's double +0.01 offset.
7. P5E absolute numbers are cross-recipe vs P1A (nu-focused composition);
   scaling fits use within-grid comparisons only.
8. The registry syncs to `exp/registry_isambard.csv` at Tufts (alias known
   to `check_transfer.py`); keep appending new Tufts runs to the Tufts
   registry rather than editing the Isambard snapshot.
