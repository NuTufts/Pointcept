# P05F — Input charge-distribution study (Tufts CPU farm)

**STATUS: COMPLETE.** Run on the full MC train list at Tufts on 2026-07-16
(415,680 files, 29.5B true points, 0 bad files). Results are in `results/`
and summarized below. **If you are the Isambard session preparing training
configs, read "Verdict" and "Handoff" — you do not need to rerun anything.**

**Question:** does the current input scaling (clip at 1000 ADC → +0.01 →
log10 → [-1,1]) squash the calorimetric differences that separate
muon / pion / proton, and would a bounded-linear, sqrt, asinh, or quantile
transform present that information better to the network?

**Motivation:** the P05A supervised ceiling study measured the charge
information as +0.114 pion IoU / +0.087 proton IoU (A.1 − A.2). The SSL
models must capture this; if the input scaling wastes resolution where the
μ/π/p dE/dx differences live, both supervised and SSL models pay for it.

**Methodological note (why these metrics):** any monotone transform leaves
single-feature AUC and density overlap unchanged — a network with enough
capacity can in principle undo any monotone rescaling. What a transform
changes is the *geometry*: where class modes land in the bounded range, and
how wide class distributions are relative to the training-time strength
jitter (MultiplicativeRandomJitter σ=0.05). So stage 2 reports both the
transform-invariant ceiling (AUC, overlap) and geometry/noise-aware metrics
(Fisher d′, d′ with augmentation noise, median separation, dynamic-range
usage). The clip fraction above 1000 ADC is measured separately — clipping
is the one step that genuinely destroys information (expected to hit
protons and overlapping tracks hardest).

---

## Verdict (2026-07-16)

The two decision rules resolve in **opposite** directions: clipping is a
non-issue, but the log transform is costing real separation.

### 1. Clipping — RULE NOT TRIGGERED. Do not touch the 1000 ADC clip.

The rule was "raise/remove the clip if the proton clip fraction is ≳ a few %".
Measured proton clip fraction is **0.16–0.19%** depending on plane; the
maximum over *all* classes and planes is 0.19% (gamma, y-plane). The prior
expectation that clipping hits protons hardest is not borne out at this
threshold — protons are unremarkable. Leave `np.clip(edep, 0, 1000.0)` at
`pointcept/datasets/lartpc.py:412` alone. Full table: `results/clip_fractions.csv`.

### 2. Transform — RULE TRIGGERED. Log is the worst of the five tested.

Augmentation-noise-aware d′, **y-plane** (higher is better; full table with
u/v/sum planes, AUC, overlap, and width metrics in `results/metrics.csv`):

| pair | log (current) | linear1000 | sqrt | asinh25 | quantile |
|---|---|---|---|---|---|
| muon-vs-pion | 0.131 | 0.209 | 0.260 | 0.288 | **0.336** |
| pion-vs-proton | 0.188 | 0.324 | 0.385 | 0.413 | **0.467** |
| muon-vs-proton | 0.058 | 0.120 | 0.131 | 0.131 | **0.135** |
| electron-vs-gamma | 0.008 | **0.047** | 0.009 | 0.014 | 0.043 |

Quantile yields **2.3–2.6×** the current log's d′ on exactly the μ/π/p pairs
this study was built to probe, and is best on every μ/π/p pair. The current
log transform ranks **last on every pair**. This triggers the P05B.5 rule
below.

⚠️ **The `asinh25` column above is untuned and should not be implemented as
shown.** Stage 2 fixes each transform's parameters at a guess; tuning asinh
(`sweep_asinh_scale.py` → `results/asinh_scale_sweep.csv`) yields
`asinh(scale=50, xmax=1000)`, which is better than `asinh25` on every pair
*and* beats quantile on μ-vs-proton. See Handoff.

Because log is not near the best d′, the "attribute the pion ceiling deficit
to topology" branch does **not** apply. That said, this only rules out input
scaling as *the* explanation, not as *a partial* one — the M4 π→p confusion
row is still worth checking independently.

### 3. Two caveats that should shape the config choice

- **The gains are concentrated in the per-plane channels.** On the
  plane-summed channel the transforms nearly tie (pion-vs-proton: 0.515 log
  vs 0.568 quantile), and `sum` already beats *every* per-plane variant for
  muon-vs-proton (0.744 vs 0.135). **P05B.4 already switches to a
  plane-summed scalar** — it captures most of this benefit without any
  transform change. Do not assume P05B.5 and P05B.4 gains add.
- **The absolute numbers stay small.** AUC 0.594 (μ/π) and 0.626 (π/p) are
  transform-invariant ceilings on *single-point scalar* charge. The network
  aggregates over neighborhoods, so this bounds one pixel's information, not
  the model's. The result says the log **wastes** what is there; it does not
  promise the +0.114 pion IoU comes back.

---

## Handoff → Isambard session (config preparation)

**Recommended: add `P05B.5-mc_noghost-s0` with `asinh(scale=50, xmax=1000)`,
not quantile and not the stage-2 default asinh25.**

Use these numbers, from `sweep_asinh_scale.py` (`results/asinh_scale_sweep.csv`),
**not** the `asinh25` column in the table above — stage 2's asinh parameters
were an untuned guess. y-plane noise-aware d′:

| transform | μ-vs-π | π-vs-p | μ-vs-p | mean |
|---|---|---|---|---|
| log (current) | 0.131 | 0.188 | 0.058 | 0.126 |
| asinh25, xmax=2e4 (stage-2 default) | 0.288 | 0.413 | 0.131 | 0.277 |
| **asinh50, xmax=1000 (recommended)** | **0.297** | **0.435** | **0.145** | **0.292** |
| quantile | 0.336 | 0.467 | 0.135 | 0.313 |

Why asinh50 over quantile, despite quantile's higher mean:

- asinh50 captures ~88% of quantile's μ/π gain and ~93% of its π/p gain, and
  **beats quantile outright on μ-vs-proton** (0.145 vs 0.135), in closed form
  with no fitted state.
- `quantile` requires a global empirical CDF fitted to the training
  distribution, which must then be frozen, versioned, and shipped with the
  checkpoint, and which silently invalidates if the input sample changes.
  That is a real reproducibility burden for a ~7% mean-d′ increment. Only pay
  it if asinh underdelivers in practice.
- The asinh optimum is broad (scale 50–75 are within 1% of each other), so
  the choice is not knife-edge sensitive.

**Implementation notes (verified against the code on 2026-07-16):**

1. The pipeline this study models is exactly:
   `np.clip(pixval, 0, 1000.0) / adc_scale` → `+ add_min_pixval` →
   `LogTransform(min_val=0.01, max_val=1000.0)`. `adc_scale` defaults to 1.0
   and **no P05 config overrides it**, so the study's `t_log` is a faithful
   model of production. Do not "fix" a scale factor that is not there.
2. `LogTransform` (`pointcept/datasets/transform.py:1893`) already has a
   `log=True/False` switch selecting log vs linear. Add an **asinh branch**
   there (e.g. `mode="log"|"linear"|"asinh"` with `asinh_scale=50.0`) rather
   than writing a new transform class — it keeps all six `LogTransform` slots
   in `gen_p05_configs.py` interchangeable.
3. Reference implementation (note `xmax = max_val = 1000`, matching the
   loader clip — **no loader change is needed**):
   `2 * arcsinh(clip(x, 0, 1000) / 50.0) / arcsinh(1000 / 50.0) - 1`
4. **Do not copy `make_t_asinh`'s `xmax=2e4` default from
   `merge_and_analyze_pixval_hists.py`.** Only 0.19% of pixels exceed 1000
   ADC, so xmax=2e4 spends ~40% of the bounded output range on that 0.19%
   and squeezes the bulk into [-1, 0.19]. Since the noise-aware d′ measures
   class width against a *fixed* augmentation σ=0.05, wasted output range is
   lost separation — that alone accounts for most of the asinh25→asinh50 gap.
5. `gen_p05_configs.py` has six `LogTransform` slots (lines ~258, 334, 583,
   617, 840, 874). P05B.5 must change **all** slots used by its pipeline —
   train and val paths both — or train/eval scalings will silently disagree.
6. P05B.5 is a pretraining variant at `P05_BUDGET`, one delta from the
   P05B base, per the experiment plan's decision rule. The optional
   supervised `P05A.5` (~10 h) is worth it only if P05B.5 shows an effect.

---

## Run record

| | |
|---|---|
| Date | 2026-07-16 |
| File list | `lartpc/filelists/h5list_v3_mc_only_train_tufts.txt` (415,680 files) |
| SLURM | job 1630842 (`--array=0-199`), partition `batch` |
| Resubmit | job 1631124 (`--array=162,163,167,168,170,171`) |
| Shards | 200/200 complete |
| Statistics | 29,517,498,592 true points, 0 bad files, 0 read failures |
| Throughput | ~13–19 files/s/shard; ~2–5 min/shard wall |

Per-class true-point counts (note the 165:1 muon:pion imbalance, relevant to
P05A.4's label-scarcity question): muon 24.99B, delta 2.72B, electron 586M,
gamma 398M, proton 335M, led 175M, michel 159M, pion 152M.

**Two deviations from the original checklist, already folded into the sbatch:**

1. **The container is required.** Tufts login and batch nodes have no h5py
   outside it. Both stages run under
   `/cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif` via
   `apptainer exec --bind /cluster:/cluster`. `module load apptainer/1.2.4-suid`
   fails unless `modtree/deprecated` is loaded first — harmless here (apptainer
   is already on PATH on the batch nodes) but fixed in the sbatch.
2. **Node-failure resubmit.** Shards 162,163,167,168,170,171 were CANCELLED
   together on node pax060 at an identical 2:58 elapsed — a node event, not a
   script fault. Resubmitting the six per step 6 completed them. Expect this
   class of failure and check `sacct` states, not just shard count.

### Figures

`dist_{u,y,sum}.png` are **not committed** — the repo's blanket `*.png`
gitignore excludes them and they are not worth force-adding. They live at
`exp/p05f_input_dist/analysis/` on Tufts and regenerate in ~1 min from the
shards via stage 2 (below). The y-plane figure shows the log panel packing
the μ/π/p modes into the top of the range while quantile spreads them across it.

---

## Reproducing / rerunning

Stage-1 shards are preserved at `exp/p05f_input_dist/` on Tufts, so stages 2
and 2b alone rerun in ~1 min:

```bash
SIF=/cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif
# stage 2 — transform families, clip fractions, figures
apptainer exec --bind /cluster:/cluster $SIF \
  python3 merge_and_analyze_pixval_hists.py \
      --indir <REPO>/exp/p05f_input_dist \
      --outdir <REPO>/exp/p05f_input_dist/analysis
# stage 2b — tune the asinh scale (this is what P05B.5 should use)
apptainer exec --bind /cluster:/cluster $SIF \
  python3 sweep_asinh_scale.py \
      --indir <REPO>/exp/p05f_input_dist --outdir results
```

To rerun stage 1 from scratch (only needed for a different file list):

1. **Smoke one shard first** (~2 s for 20 files):
   `apptainer exec --bind /cluster:/cluster <SIF> python3 accumulate_pixval_hists.py
   --filelist <LIST> --num-shards 200 --shard 0 --outdir /tmp/p05f_smoke --max-files 20`
   Confirm the npz exists and reports nonzero true points.
2. Edit the ADJUST paths in `submit_pixval_hists_tufts.sbatch` (REPO,
   FILELIST, OUTDIR, CONTAINER). 1 CPU + 4 GB per task is sufficient; the
   `batch` partition is correct and is the Tufts default.
3. `mkdir -p logs` **before** `sbatch --array=0-199 submit_pixval_hists_tufts.sbatch`.
   (SLURM opens the --output file before the script runs; a missing logs/
   dir kills every task instantly with exit code 53.)
4. **Completion check:** `ls <OUTDIR>/pixval_hists_shard*.npz | wc -l` must
   equal 200, **and** `sacct -j <jobid> -X -n -o State | sort | uniq -c` must
   show no CANCELLED/FAILED — a cancelled task leaves no npz, so the count
   alone tells you *which* are missing but not *why*. Resubmit missing ids:
   `sbatch --array=<id1>,<id2> submit_pixval_hists_tufts.sbatch`.
   (Stage 2 also runs on a partial set — shards are interleaved, so any
   subset is an unbiased sample; note reduced statistics in the report.)

To run on a fraction of the list quickly, submit a subset of array IDs
(e.g. `--array=0-19`) with NUM_SHARDS left at 200 — shard interleaving makes
any ID subset an unbiased sample.

## Decision rules (feeding back into Phase 0.5) — retained for reference

- If the **clip fraction** for protons is substantial (≳ a few %), raise or
  remove the 1000 ADC clip in the loader regardless of transform choice.
  → **Not triggered** (0.16–0.19%); clip stays.
- If a transform beats log on **noise-aware d′** for μ-vs-π / π-vs-p by a
  meaningful margin, add a `P05B.5` pretraining variant (and optionally a
  supervised `P05A.5` at ~10 h cost) with that scaling — it composes with
  the LogTransform slot in the generated configs.
  → **Triggered** (2.3–2.6×); see Handoff above.
- If log is already near the best d′ and clipping is negligible, the pion
  ceiling deficit is attributed to topology (secondary interactions
  producing proton-like segments), not input scaling — check the π→p row
  of the confusion matrix (M4) to confirm.
  → **Does not apply** (log is worst), but check M4 π→p independently.
