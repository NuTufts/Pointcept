# P05F — Input charge-distribution study (Tufts CPU farm)

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

## Running (Tufts)

1. Prereq: the Tufts-remapped MC train list (see the file-list remap task);
   or point FILELIST at any Tufts MC list with truth.
2. Edit the three ADJUST paths in `submit_pixval_hists_tufts.sbatch`.
3. `sbatch --array=0-199 submit_pixval_hists_tufts.sbatch`
   (pure h5py+numpy; 1 CPU + 4 GB per task; ~5–15 min/task).
4. When the array finishes:
   `python3 merge_and_analyze_pixval_hists.py --indir <OUTDIR> --outdir <OUTDIR>/analysis`
5. Deliverables in `analysis/`: `metrics.csv`, `clip_fractions.csv`,
   `dist_{u,y,sum}.png`, `summary.md`.

A subsample is statistically plenty (per-point stats are enormous); to run
on e.g. 1/10th of the list quickly, submit `--array=0-19` with
`NUM_SHARDS=200` unchanged — shards are interleaved, so any subset of task
IDs is an unbiased sample of the list.

## Decision rules (feeding back into Phase 0.5)

- If the **clip fraction** for protons is substantial (≳ a few %), raise or
  remove the 1000 ADC clip in the loader regardless of transform choice.
- If a transform beats log on **noise-aware d′** for μ-vs-π / π-vs-p by a
  meaningful margin, add a `P05B.5` pretraining variant (and optionally a
  supervised `P05A.5` at ~10 h cost) with that scaling — it composes with
  the LogTransform slot in the generated configs.
- If log is already near the best d′ and clipping is negligible, the pion
  ceiling deficit is attributed to topology (secondary interactions
  producing proton-like segments), not input scaling — check the π→p row
  of the confusion matrix (M4) to confirm.
