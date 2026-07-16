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

## Orchestration checklist (for a Claude Code session at Tufts)

The pipeline was validated end-to-end on Isambard MC files on 2026-07-16
(80 files, 5.8M points) — the scripts themselves are known-good; the steps
below localize them to Tufts.

1. **Pick the file list.** Preferred: the Tufts-remapped MC train list
   (`lartpc/filelists/h5list_v3_mc_only_train_tufts.txt` if the remap task
   has been done). Fallback: any Tufts MC list with truth — e.g. the prod4
   lists referenced in `configs/lartpc/p05/linearprobe-sonata-p05-mc-noghost-tufts.py`
   (`TRAIN_FILE_LIST`). The worker needs
   `/entry_0/triplet_data/{pixval,ssnet_label,hasmatch}` in each file.
2. **Verify the environment.** Any python3 with h5py+numpy works for stage 1
   (no torch, no GPU); stage 2 additionally needs matplotlib. Use whichever
   env the probe jobs used. Quick check:
   `python3 -c "import h5py, numpy, matplotlib"`.
3. **Smoke one shard locally** before submitting 200 tasks (~1 min):
   `python3 accumulate_pixval_hists.py --filelist <LIST> --num-shards 200
   --shard 0 --outdir /tmp/p05f_smoke --max-files 20`
   then confirm the npz exists and reports nonzero true points.
4. **Edit the sbatch**: the three ADJUST paths (REPO, FILELIST, OUTDIR) and
   the partition name (discover with `sinfo -s`; pick the general CPU
   partition). 1 CPU + 4 GB per task is sufficient.
5. **`mkdir -p logs` in the submission directory, then**
   `sbatch --array=0-199 submit_pixval_hists_tufts.sbatch`.
   (SLURM opens the --output file before the script runs; a missing logs/
   dir kills every task instantly with exit code 53.)
   Expect ~5–15 min/task over the full 415k-file list.
6. **Completion check:** `ls <OUTDIR>/pixval_hists_shard*.npz | wc -l` must
   equal 200. Resubmit any missing shards individually:
   `sbatch --array=<id1>,<id2> submit_pixval_hists_tufts.sbatch`.
   (Stage 2 will also run on a partial set — shards are interleaved, so any
   subset is an unbiased sample; note the reduced statistics in the report.)
7. **Stage 2 (single node / login, ~1 min):**
   `python3 merge_and_analyze_pixval_hists.py --indir <OUTDIR> --outdir <OUTDIR>/analysis`
8. **Deliverables** in `analysis/`: `metrics.csv`, `clip_fractions.csv`,
   `dist_{u,y,sum}.png`, `summary.md`. Report: the clip-fraction table, and
   per class-pair (esp. muon-pion, pion-proton) the noise-aware d′ of each
   transform vs the current log — then apply the decision rules below.

To run on a fraction of the list quickly, submit a subset of array IDs
(e.g. `--array=0-19`) with NUM_SHARDS left at 200 — shard interleaving makes
any ID subset an unbiased sample.

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
