# P5A — Data-scaling study: detailed plan

**REVISED 2026-07-19 (PI concern):** EXTBNB contains no neutrino
interactions, so an EXTBNB-only scaling curve read out by the nu-centric
probe would measure cosmic-feature transfer saturation, not nu-physics
learning. P5A therefore runs TWO single-domain rows at 1x width — **MC +
LArMatch (P1A.4b base, contains nu interactions)** and **EXTBNB + LArMatch
(P1A.4 base)** — same nested subset sizes, same probe. The row pair adds the
cross-domain data-value readout (how many cosmic-only events equal one
nu-containing event at fixed compute). The P5E width grid (experiment plan
§6) uses the **MC + LArMatch base** for the same reason.

**Expands the one-row entry in `microboone_sonata_experiment_plan.md` §6.**
Drafted 2026-07-18, incorporating everything Phase 0.5 / Phase 1 established
(locked budget, snapshot machinery, Tufts probe protocol, lm_score fix).

## 1. Question

At **fixed compute** (fixed images seen = MATCHED_BUDGET = 14,964,480, fixed
batch/LR/schedule), how does representation quality depend on the number of
**unique events**? Equivalently: what does repeating a small dataset for many
epochs cost relative to fresh data — and where is the knee?

Why it matters:
- **Practical (the headline):** real-data pretraining is limited by event
  *processing* (LArMatch etc.), not collection. If ~100k EXTBNB events reach
  parity with 415k at this compute, the production burden for a LArTPC
  foundation model drops ~4×; if 26k suffices, it is negligible.
- **Methodological:** separates "more unique data" from "more gradient
  steps" — the data-constrained scaling question (cf. Muennighoff et al.
  2023 for LMs: ~4 epochs of repetition ≈ fresh data, decaying after).
  Whether self-distillation on detector data shows the same tolerance is
  a publishable curve on its own (PANDA-style Fig. analog).
- **Program-level:** informs whether future allocations should buy
  processing (more unique events) or GPU time (more steps).

## 2. Design

**Base config: exactly P1A.4** (EXTBNB + stochastic LArMatch filter,
detsym augs, 4096 prototypes, log scaling, batch 48, lr 2e-4, 311,760
iterations) — **the ONLY delta per run is the training file list.** P1A.4
itself is the full-data anchor (reuse; no new run).

**Dataset grid — nested ×4 prefix subsets** of the trimmed 415,680-file
EXTBNB list (prefixes of the same shuffled list, so each subset ⊂ the next;
nesting removes subset-selection variance from adjacent comparisons, and
every size gives an INTEGER epoch count against the fixed budget):

| run | unique events | epochs at fixed budget | repetitions |
|---|---|---|---|
| P5A.0 ≡ P1A.4 (reuse) | 415,680 | 36 | 36 |
| P5A.1 | 103,920 (=415,680/4) | 144 | 144 |
| P5A.2 | 25,980 (=/16) | 576 | 576 |
| P5A.3 | 6,495 (=/64) | 2,304 | 2,304 |
| P5A.3b (optional) | 6,495 (disjoint 2nd slice) | 2,304 | seed-variance check |

This brackets the original plan's 10k/100k intent with exact arithmetic.
P5A.3b (a second, disjoint 6,495-event slice) is the cheap insurance against
subset-content luck at the smallest size — only worth running if capacity
allows; the nested design already protects the larger sizes.

**Everything else identical:** same snapshot schedule (the P1A 15-anchor
images-seen grid — so all curves share x-axis points), same val list, same
M5 logging, same SLURM machinery (timeout-wrapped resubmit chains).

Two intentional properties to document, not "fix":
- The stochastic LArMatch threshold redraws per visit and the crops/augs
  redraw per epoch, so a "repeated" event is never bit-identical input —
  this measures repetition *under augmentation*, which is the regime any
  real training runs in.
- Small-subset epochs are tiny (6,495 files / 48 ≈ 135 iters/epoch), so
  epoch-boundary machinery churns more; `skip_dataloader_on_resume` and the
  per-epoch sampler reshuffle already handle this. OneCycle and the Sonata
  schedulers are iteration-based — unaffected.

## 3. Measurements and analysis

All via the existing Tufts probe pipeline (frozen budget, A100/batch-64
protocol, matched MC train/val split); `launch_probe_sweep.py` needs one
`("P5A", <standard probe config>)` mapping line.

1. **Primary figure:** probe mIoU (and pion/proton IoU) vs images seen —
   one curve per dataset size, P05A.1 ceiling line (0.8576) for reference.
   The readouts: (a) final-point metric vs unique events (log-x — the
   scaling curve); (b) the **departure point** — images-seen at which each
   small-data curve peels off the full-data curve ≈ the useful-repetition
   horizon in epochs; (c) whether small-data curves *plateau* or *degrade*
   (overfitting of the SSL objective to a small event set is possible and
   would show as a rollover).
2. **Loss plateau vs size** from wandb (free; the experiment plan's
   "pretraining dynamics" analog).
3. **Repetition-efficiency number** for the paper: images-to-match — how
   many repeated-data images the 26k run needs to match the 415k run at,
   e.g., its 3.07M-image point.
4. Registry rows + bootstrap CIs when the P0.2 tooling lands (same caveat
   as all Phase-1 numbers: single seed except the optional P5A.3b pair).

## 4. Cost, venue, timeline

Per run ≈ P1A.4: ~1.9 s/iter → ~165 h wall ≈ 165 node-hours on one 4×GH200
node; 3 new runs ≈ **500 node-hours** (+165 for optional P5A.3b).

Venue reality (allocation ends 2026-07-22): started now, each run reaches
~50–60% of budget on Isambard, then resumes at Tufts or a future allocation
via the standard snapshot/`model_last` machinery — same posture as the
P1A/P5B fleet, which also completes post-allocation. Two points make partial
runs unusually valuable here: the scientifically decisive behavior of the
SMALL-data runs (plateau/rollover) happens **early** (a 6,495-event run has
completed 500+ repetitions by 3M images), and the nodes freed by tonight's
Wave A completions would otherwise idle — this study is the natural consumer
of the remaining burn.

## 5. Mechanics checklist (Isambard session)

1. Sub-lists: `head -103920 / -25980 / -6495` of
   `h5list_v3_extbnb_only_train_415680.txt` (+ `sed -n '6496,12990p'` for
   the optional disjoint P5A.3b slice); record sha256 in `filelist_stats.txt`.
2. Generator: `P5A_RUNS` entries = P1A.4 overrides with `TRAIN_LIST=<subset>`,
   `SAVE_ROOT="p5a"`, run IDs `P5A.{1,2,3}-extbnb_{104k,26k,6k5}-s0`.
   Regenerate; confirm existing configs byte-identical.
3. Validate: dataset build per config; assert `len(ds.data_list)` matches
   the subset size and epochs × list = 14,964,480.
4. Launch via `launch_p05_run.sh` (registry + hash + timeout-wrapped chain).
5. Tufts: add the `P5A` probe mapping; probes flow through the standard sweep.

## 6. Non-goals / later extensions

- **Model-size scaling** (width/depth at fixed data) — a different study;
  out of scope here.
- **MC-side mirror** (same grid on the MC list) — cheap to add later if the
  data-side curve is interesting; would separate "unique events" from
  "unique physics" (MC subsets share the same generator statistics).
- Mixture scaling (vary the MC:data ratio at fixed budget) — natural
  follow-on to P5B once its verdict is in.
