# Single Photon Selection

The goal of this analysis is to look at the performance of the LArFormer Full
Cascade model on single photon events.

The way we intend to find these events:

1. First select events with a neutrino slice, using the stage 2 event slicer output.
2. For events with a neutrino slice, use the stage 3 particle segmenter to look for photon showers.
3. Remove candidate photon showers that are too small. Ideally we would be able to cut by ionization energy. But using a rough proxy initially is OK: the numebr of spacepoints.

We want to define a selection based on the LArFormer outputs and estimate the true positive, false positive, and false negative rates.

## References

- LArFormer model information: [`docs/LArFormer.md`](../../../docs/LArFormer.md)
- MicroBooNE data set information: [`docs/MicroBooNE_Datasets_on_Tufts.md`](../../../docs/MicroBooNE_Datasets_on_Tufts.md)
- Flat-ntuple parsing spec: [`docs/Gen2_Flat_Ntuple_Spec.md`](../../../docs/Gen2_Flat_Ntuple_Spec.md)
- Cluster job submission: [`docs/Tufts_SLURM_Job_Guide.md`](../../../docs/Tufts_SLURM_Job_Guide.md)
- LArFormer cascade data pipeline: [`larformer_scripts/LARFORMER_DATAPREP.md`](../../larformer_scripts/LARFORMER_DATAPREP.md)

## Steps

1. Define the signal definition and determine the number of events in the dataset that satisfy this definition.
2. Define additional side band regions to estimate contributions from background events.
3. Isolate the files and events in the dataset that contain signal events.
4. Process the signal and side band events through the LArFormer Full Cascade model.
5. Estimate the efficiency (true positive) selection rate and the background contamination in the signal region.

---

## Why single photons are hard to define

Photons mostly come from π⁰s (from the ν interaction or secondary hadronic
interactions) and become *single-photon* topologies when only one photon is
"visible." We define visible as **≥20 MeV of ionization deposited in a single
cluster**. That quantity is not in the flat ntuples, so the study runs in two
passes over two data tiers (see [`MicroBooNE_Datasets_on_Tufts.md`](../../../docs/MicroBooNE_Datasets_on_Tufts.md)):

- **Pass 1 (flat ntuples)** — a loose truth pre-selection to *count* candidate
  signal and *isolate* the official files that contain it. The ntuple can't
  measure single-cluster ionization, so the cut here is a necessary-but-not-
  sufficient proxy.
- **Pass 2 (official sim files)** — the real ≥20 MeV-in-a-cluster detectability
  cut + the LArFormer cascade, on just the isolated files.

This directory currently implements **Pass 1** on the **BNB ν overlay** sample
(`mcc9_v29e_dl_run3b_bnb_nu_overlay`).

---

## Pass 1 — ntuple signal definition (implemented)

For each `EventTree` MC entry, signal if **both**:

1. **ν vertex inside the TPC box** (cm, SCE-corrected):
   `0 < trueVtxX < 255`, `-116.5 < trueVtxY < 116.5`, `0 < trueVtxZ < 1036`.
2. **≥1 neutrino-origin photon plausibly detectable:** a `trueSimPart` with
   `PDG == 22`, `trueSimPartE > 20 MeV` (initial energy — below this it cannot
   make a 20 MeV cluster), and first-deposit point `trueSimPartEDep{X,Y,Z}`
   inside the TPC box.

In an overlay sample every `trueSimPart` is neutrino-induced, so any PDG-22 entry
is nu-origin (verified — see the spec doc). The 20 MeV cut is a *proxy*; the real
single-cluster ionization cut happens in Pass 2.

> Note: the ntuple is pre-filtered to the Wire-Cell fiducial volume (a subset of
> the TPC box), so the vertex cut passes ~100% and these counts are a lower bound
> for the full TPC.

### Results (full run-3b BNB ν overlay ntuple, ΣgoodPOT = 8.98×10²⁰)

| Quantity | Value |
|---|---|
| EventTree entries scanned | 290,538 |
| LOOSE (vtx + ≥1 photon, any E) | 50,792 *(context only)* |
| **SIGNAL (vtx + 20 MeV + EDep-in-TPC)** | **47,503** (CC 30,805 / NC 16,698) |
| **POT-scaled @ 6.67×10²⁰** | **≈ 35,365 events** |
| unique source files with signal | 13,920 / 15,513 |

The loose photon cut barely isolates files (≈90% contain a candidate) because of
abundant neutron-capture γ's — real signal-richness only appears after the Pass-2
ionization cut.

---

## Scripts & workdir

All outputs land in `workdir/` (git-ignored intermediate area).

| Script | Purpose | Output |
|--------|---------|--------|
| `verify_ntuple.py` | Step-0 sanity checks (branches, POT, nu-origin of photons) | stdout |
| `select_single_photon_signal.py` | apply the Pass-1 signal definition | `signal_events.csv`, `signal_fileids.txt` |
| `map_signal_to_files.py` | map selected events → official `merged_dlreco` files via `(run,subrun,event)` / `larlite_id_tree` | `signal_files_subsample.txt`, `signal_file_map.csv` |

`signal_events.csv` columns:
`run, subrun, event, fileid, trueNuPDG, trueNuCCNC, trueVtxX/Y/Z, nPhotonsLoose,
nPhotonsSig, maxPhotonE, xsecWeight`.

### Run recipe (pointcept container)

```bash
cd Pointcept/lartpc_data_prep/larformer_physics/single_photon
SIF=/cluster/tufts/wongjiradlabnu/larbys/larbys-container/pointcept_cuml.sif
ENV=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/ubdl/setenv_pointcept_container.sh

apptainer exec --bind /cluster:/cluster $SIF bash -c "source $ENV >/dev/null 2>&1; \
    python3 verify_ntuple.py -n 3000"                          # Step 0
apptainer exec --bind /cluster:/cluster $SIF bash -c "source $ENV >/dev/null 2>&1; \
    python3 select_single_photon_signal.py --outdir workdir"   # Pass-1 counts + lists
apptainer exec --bind /cluster:/cluster $SIF bash -c "source $ENV >/dev/null 2>&1; \
    python3 map_signal_to_files.py --nfiles 20 --spread --outdir workdir"  # capped file map
```

> `source setenv_pointcept_container.sh` only puts ROOT on the path; the flat
> ntuple itself needs no serialized-class libraries. `--nfiles` caps how many
> unique `(run,subrun)` files are resolved. **`--spread`** spreads that cap evenly
> across the whole sample (auto-picks `--stride`) instead of clustering in the
> first ntuple region — use it for a representative first pass (e.g. 20 files →
> 20 distinct runs). Omit it (or `--stride 1`) to take the consecutive head.

---

## Pass 2 — convert + capped cascade (implemented)

The truth pre-selection does NOT meaningfully reduce the file count (~90% of files
contain a candidate), so Pass 2 is split so the expensive GPU step stays small:

1. **Stage A — convert (cheap, CPU, fan out wide).** Convert every event of each
   subsample file → per-event `merged_*.h5`, on the `batch` partition, one array
   task per file. Scales to the full sample by widening the array / `stride`.
2. **Select capped events (CPU, quick).** Stage A converts *all* events in a file,
   but we only want the signal events — and the GPU cascade is limited. Filter the
   Stage-A H5 to the signal `(run,subrun,event)` set and cap the count.
3. **Stage B — cascade (expensive, GPU, capped).** Run the LArFormer full cascade
   only on the capped event list, on a single GPU (wongjiradlab P100 / old cluster,
   or new-cluster A100).

| Piece | File |
|-------|------|
| Stage-A config (older sim: `--adc wire -tb --mcc9`, `RUN_CASCADE=0`) | `larformer_scripts/larformer_configs/single_photon_subsample.conf` |
| Stage-A SLURM array (CPU batch) | `slurm/submit_stageA_convert.sh` |
| event capping (signal `(run,subrun,event)` → cascade input list) | `select_cascade_events.py` |
| Stage-B runner (in-container) | `run_stageB_capped.sh` |
| Stage-B SLURM (GPU, new-cluster A100) | `slurm/submit_stageB_cascade.sh` |
| Stage-B SLURM (GPU, old-cluster P100) | `slurm/submit_stageB_cascade_p100_oldcluster.sh` |
| cascade-output sanity check (per-event predicted classes, photon flag) | `verify_stage3pred.py` |

> **Attention backend (GPU-dependent).** The cascade config's `flash_backend` is
> env-driven: default `flash_attn` (Ampere+: A100/H100/H200), but **P100 (Pascal)
> has no flash-attn kernel** — the P100 submit script sets
> `LARFORMER_FLASH_BACKEND=xformers`. `run_stageB_capped.sh` exports it.

```bash
# 1. Stage A (CPU array; --array sized to the subsample list)
sbatch slurm/submit_stageA_convert.sh
# ... wait for it to finish; outputs land in workdir/larformer_h5/<hash>/

# 2. cap signal events for the GPU step (in the pointcept container)
apptainer exec --bind /cluster:/cluster $SIF bash -c "source $ENV >/dev/null 2>&1; \
    python3 select_cascade_events.py --h5-dir workdir/larformer_h5 \
        --signal-csv workdir/signal_events.csv --nevents 200 --out workdir/cascade_inputs.txt"

# 3. Stage B (GPU; one job over the capped list)
sbatch slurm/submit_stageB_cascade.sh              # new-cluster A100, OR:
sbatch slurm/submit_stageB_cascade_p100_oldcluster.sh   # old-cluster P100 (xformers)
#    -> stage3pred_<input>.h5 (slicer half: pre/post/levels/gt ; particle half: stage3*/)

# 4. sanity-check the cascade output
apptainer exec --bind /cluster:/cluster $SIF bash -c "source $ENV >/dev/null 2>&1; \
    python3 verify_stage3pred.py --dir workdir/larformer_h5"
```

**First subsample run (36 capped events, checkpoint `model_iter_182304.pth`):**
29/36 events had ≥1 predicted photon (class-1) query; active-query class tally
`{gamma:65, p:78, e:21, mu:21, pi:15}`. Sanity-consistent — every input event has
a true ≥20 MeV nu-origin photon, and the cascade recovers a photon shower in most.

> Checkpoint note: the trained Stage-3 weights `model_iter_98652.pth` live in the
> `..._bugfixed_resume2/` exp dir (the `run_stepB` default points at the stale
> `_bugfixed/` dir); the config sets the correct path explicitly.

## Truth detectability (implemented)

`compute_photon_detectability.py` defines the *detectable* true-photon set from the
Stage-A H5, which is the denominator for the efficiency. Key facts established:

- Stage A **does** write GT shower fragments (`entry_0/shower_fragments`: per-fragment
  `pid`, `trackid`, `istrunk`, `pointindices_counts/_flat`). Fragments group by
  `trackid` = per photon; `istrunk==1` is the trunk ("single cluster").
- The older mcc9 sim has **no true per-spacepoint energy** — the H5 `edep` field just
  mirrors the wire-plane ADC `pixval` (`edep≈pixval`, ADC-scale not MeV). So the
  ≥20 MeV detectability cut uses **proxies**: `nSP` (spacepoint count) and
  `pixval_sum` (wire-plane ADC sum). `mc_particle_tree/energy_mev` gives true energy
  for calibration.
- Calibration (36-event subsample, 71 nu-origin photons): `corr(nSP,E)=0.82`,
  `corr(pixval,E)=0.79`; a ~20 MeV photon → **nSP_trunk ≈ 80, pixval_sum ≈ 25k**.
- Caveat: the ntuple pre-cut (`trueSimPartE>20 MeV` + EDep-in-TPC) already
  preselects, and GT fragments only exist for depositing photons, so on this set
  nearly **all** true photons are detectable (min nSP 36, median 1202). The cut
  mostly defines the denominator precisely; it bites harder if the ntuple pre-cut is
  loosened.

Output `workdir/photon_detectability.csv`:
`run,subrun,event,trackid,n_frag,nSP_total,nSP_trunk,pixval_sum,trueE_mev,origin`.

## Efficiency / matching (implemented)

`match_predictions_to_truth.py` matches cascade-predicted photon queries to true
nu-origin photons and computes efficiency + purity. Truth denominator = ALL
nu-origin photons in `mc_particle_tree` (each annotated with GT-fragment `nSP_trunk`;
detectable if `nSP_trunk >= --nsp-threshold`). Predicted photons = active post-dedup
queries with `class_argmax==1`. Match = occupied-voxel IoU (`--voxel` cm, greedy,
`>= --iou-min`) between the true photon's `triplet_data/pos` and the query's
`stage3/coord` (both detector cm).

```bash
python3 match_predictions_to_truth.py --nsp-threshold 80 --voxel 3 --iou-min 0.1 \
    --pred-dir workdir/larformer_h5 --merged-dir workdir/larformer_h5 \
    --out workdir/photon_match.csv
```

**Tight subsample (36 events, 80 nu photons, ckpt 182304):** detectable=69 →
TP 57 / FN 12 / FP 8 → **efficiency 0.83, purity 0.88**. Turn-on (efficiency over
*all* true photons): 0.33 (20–40 MeV) → ~0.78 (80–400 MeV), driven by the
detectability fraction (0.33 → 0.95). Per-photon detail in `workdir/photon_match.csv`.

## Loose pass — efficiency turn-on (results)

The **loose pass** reruns the pipeline with the 20 MeV cut removed
(`select_single_photon_signal.py --photon-emin 0`) over a 100-file spread subsample,
to get low-energy photon statistics for the turn-on. Everything lives under
`workdir_loose/` with TAG `sp_bnb_nu_overlay_loose`. Stage A converted 2207 events
(46/100 files; the rest hit the converter `out_of_range` crash); 204 capped → P100
cascade → 173 stage3pred.

**Result (204 events, 447 nu-origin photons):** detectable=387 → TP 298 / FN 89 /
FP 54 → **efficiency 0.77, purity 0.85**. Clear turn-on (efficiency over *all* true
photons):

| E (MeV) | N | detect-frac | eff‖all | eff‖detectable |
|---|---|---|---|---|
| 0–20    |  19 | 0.63 | **0.32** | 0.50 |
| 20–40   |  37 | 0.81 | 0.57 | 0.67 |
| 40–80   |  93 | 0.84 | 0.62 | 0.72 |
| 80–150  | 123 | 0.89 | 0.71 | 0.80 |
| 150–400 | 144 | 0.90 | 0.74 | 0.81 |
| 400+    |  31 | 0.94 | **0.77** | 0.83 |

Sensitivity scan over voxel{2,3,5}×iou{0.05–0.30}×nSP{50–200}: efficiency
**0.70–0.83**, purity **0.78–0.87** — stable. Per-photon detail in
`workdir_loose/photon_match.csv`. (The OOM-fix rerun recovered all 204/204 events,
`OOM-skipped 0`; numbers were unchanged from the 173-event partial → no size bias.)

Reproduce:

```bash
# (selection + 100-file map already done -> workdir_loose/signal_files_subsample.txt)
sbatch slurm/submit_stageA_convert_loose.sh                 # CPU array (running)
# after Stage A:
python3 select_cascade_events.py --tag sp_bnb_nu_overlay_loose \
    --h5-dir workdir_loose/larformer_h5 --signal-csv workdir_loose/signal_events.csv \
    --nevents 400 --out workdir_loose/cascade_inputs.txt
sbatch slurm/submit_stageB_cascade_loose_p100_oldcluster.sh # P100 (old cluster), or
sbatch slurm/submit_stageB_cascade_loose.sh                 # A100 (gpu partition)
python3 match_predictions_to_truth.py --pred-dir workdir_loose/larformer_h5 \
    --merged-dir workdir_loose/larformer_h5 --out workdir_loose/photon_match.csv
```

## Analysis scripts

| Script | Purpose |
|--------|---------|
| `compute_photon_detectability.py` | per-photon nSP/pixval/trueE truth table |
| `match_predictions_to_truth.py` | predicted-photon ↔ true-photon match → efficiency, purity, turn-on |
| `scan_match_params.py` | sweep voxel / iou_min / nSP_threshold → efficiency & purity grids |

**Robustness (tight subsample):** sweeping voxel∈{2,3,5}cm × iou_min∈{0.05–0.30} ×
nSP∈{50–200} keeps efficiency in **0.76–0.84** and purity **0.82–0.88** — the
headline numbers are stable, not artifacts of the chosen thresholds. `iou_min=0.30`
is slightly strict; voxel size barely matters in this range.

## Robustness fixes (done)

- **Stage-A converter** (`convert_dlmerged_to_larformer_h5.py`): per-event try/except —
  a single `SimChTripletLabelMaker` `out_of_range` event is logged + skipped (partial
  output removed) instead of crashing the whole file. Verified on a known-bad file:
  13/14 events kept (was 0). This recovers the ~50% of files previously lost.
- **Stage-B cascade** (`run_larformer_stage3_inference.py`): per-event OOM guard around
  both forward passes (skip + `empty_cache` + continue) + proactive per-event
  `empty_cache`; `run_stageB_capped.sh` exports
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. One OOM event no longer kills the
  run. Rerun with `--overwrite` off to fill gaps.

## Scale-up run (1500 files, ~10% sample)

Data on the larbys area:
`/cluster/tufts/wongjiradlab/larbys/data/larformer/mcc9_v29e_dl_run3b_bnb_nu_overlay/{merged_sp,stage3pred}`.

- Stage A ×1500 (`single_photon_scale1500.conf`, `submit_stageA_convert_scale1500.sh`):
  **67,211 events**, 334 GB, 97.8% file coverage (C++ fix held; the ~5k loss was
  cgroup-OOM — now `--mem-per-cpu=16000`).
- `select_visible_photon_events.py` → **6,743 visible-photon events, 829 (12.3%) 1γ+0X**;
  capped spread **3,000** → Stage B array (`submit_stageB_cascade_array.sh`, A100, 30×100).

### Results (3000-event sample)

**Photon finding** (re-baselined on the fixed reproducible base, `sp_compare_3k_fixed/base`,
5441 detectable photons): **efficiency 0.724, purity 0.825** (was 0.71/0.83 — essentially
unchanged; photon-finding is a coarse per-photon IoU match, robust to the per-SP perturbations
that flipped the finer 1γ+0X selection). Detail in `workdir_scale/photon_match_fixed.csv`.
- Energy turn-on: eff(all) 0.16 (0–20 MeV) → 0.65 (150–400 MeV).
- Vertex-distance falloff: 0.82 (0–5 cm) → 0.44 (120+ cm).

**1γ+0X scale-up confirmation** (base vs rescue on the full **829 truth-1γ+0X**, both fixes,
nominal τ=0.5): base eff **0.151** / pur 0.424; **rescue eff 0.169 / pur 0.406 (+15 TP, +12%)** —
confirms the small-sample rescue win (378: 0.156→0.167) at 2× statistics (~1.3σ). Outputs:
`sp_scale829_{base,rescue}`.

**1γ+0X selection — REPRODUCIBLE deghost/rescue sweep** (`analyze_1g0X.py`, full 3000-event
mixed sample, pinned to one node, `--deterministic` **+ both membership fixes**;
`base`≡`base_dup` bit-identical across all 2999 events confirms the A/B is exact). Outputs:
`sp_compare_3k_fixed/`. **These supersede the earlier buggy sweep — see the OLD column.**

| Arm | deghost τ | flash rescue | efficiency | purity | (old, buggy) |
|-----|-----------|--------------|------------|--------|--------------|
| **base** | 0.5 (default) | no | 0.156 | 0.437 | (0.161 / 0.436) |
| dg0p4 | 0.4 | no | 0.151 | 0.442 | (0.138 / 0.374) |
| dg0p3 | 0.3 | no | 0.148 | 0.412 | (0.153 / 0.379) |
| **rescue** | 0.5 | K=1 | **0.167** | **0.420** | (0.135 / 0.309) |
| dg0p3+rescue | 0.3 | K=1 | 0.156 | 0.383 | (0.146 / 0.346) |

- **Flash rescue HELPS — and the old "net-negative" was a reproducibility-bug artifact.** With
  the fixes, rescue is eff **0.156 → 0.167 (+4 TP)** with purity essentially held (0.437 → 0.420).
  The old buggy result (rescue eff 0.135 / purity 0.309) was inverted by the shuffle-RNG breaking
  additivity (the "rescue loses TPs it never touched" effect). Rescue is the best arm.
- **Lowering the deghoster threshold still does NOT help — even with the 8192 token cap removed.**
  dg0p4 ~neutral, dg0p3 slightly worse. The `max_source_tokens_per_level=8192` cap *was* randomly
  crowding out real points (now fixed: eval uses all tokens), but the extra low-confidence shower
  points still don't improve segmentation — they're not the limiting factor. Keep τ=0.5.
- **Significance:** deterministic now (no run-to-run noise), so +4 TP is a real effect on these 378
  truth events, but ~0.5σ on ~60 TP — direction is solid, scale up to pin the magnitude (now safe:
  small-sample predicts full-dataset).
- (Superseded) earlier comparison caveat: the order-dependence below is only fully common-mode
  for *identical* computations (base vs `base_dup` = bit-exact). Any arm that changes the
  per-event computation — a different deghost τ, or rescue's extra flash work — also shifts the
  history-dependent numerics of *unrelated* events, adding ±few-% noise. (Demonstrated: rescue,
  which is rescue-only by design, still flipped 86 events it never touched and lost 47 TPs all on
  not-rescued events.) So the sweep's **direction is robust** (the levers clearly don't help — the
  perturbation is symmetric, not a systematic gain) but the precise deltas and the **absolute**
  numbers are not citable until the order-dependence is fixed.
- Prior analysis reference: ~0.10 eff / ~0.40 purity.

**Where 1γ+0X photons are lost** (`analyze_photon_slice.py`): of 378, the slicer puts
only **49% in a ν-slice** (selected at 0.28), **34% in their OWN slice mislabeled
cosmic**, 16% merged into cosmic slices, 2% lost. FN breakdown: 71% no photon query
(slicer/ν-slice drop), 15% photon-split, 14% false-X.

Analysis scripts: `select_visible_photon_events.py`, `match_predictions_to_truth.py`
(energy + vertex-distance turn-on), `analyze_1g0X.py`, `analyze_photon_slice.py`,
`run_stageB_capped.sh` env knobs `DETERMINISTIC=1` / `DEGHOST_THRESHOLD_VAL` / `FLASH_RECOVER_K`,
sweep `slurm/submit_deghost_compare_3k_pin.sh`.

### Reproducibility & determinism (`docs/LArFormer_Reproducibility.md`)

Inference is non-deterministic by default; `run_larformer_stage3_inference.py --deterministic`
(env `DETERMINISTIC=1`) fixes it. Established facts:

- **Full-run reproducibility is bit-exact** with `--deterministic` (same input list, same node):
  validated 0 / 1.08M spacepoints differ across two runs.
- **Cross-hardware:** A100 (80 & 40 GB) and L40S are **mutually bit-identical** — determinism is
  a property of the **driver + library stack**, not the GPU model. Hopper (H100/H200) is a
  **separate conformance family** that diverges ~1.9% at the event level (source = architecture-
  specialized cuBLAS GEMMs, *not* attention — tested by swapping flash↔xformers). Deploy with a
  **driver allowlist + per-node conformance test** (`tools/capture_cascade_tensors.py` +
  `tools/cross_gpu_diff.py`); pin Hopper jobs separately.
- **✅ Membership/list dependence — ROOT CAUSE FOUND + FIXED.** A given event's output depended on
  which *other* events were in the run (→ ~6.6% of 1γ+0X labels flipped between a 378- and a
  3000-event list, same node). **Cause: `shuffle_orders=True` on the deghoster's PTv3** (config
  `...ptv3crosslevel.py:160`; slicer/segmenter were already `False`). `shuffle_orders` randomizes the
  serialization order via `torch.randperm` and is a **train-time augmentation not gated by eval** —
  at inference it consumed the global RNG per event (so an event's output depended on its
  predecessors) and randomized the attention patch order. That's why it was *precision-independent*
  and FP64 / dead-band / allocator-toggle all failed (those were investigated and ruled out).
  **Fix:** config → `False`, plus `run_full_cascade_mode` defensively disables `shuffle_orders`
  model-wide at inference (logs "disabled serialization order-shuffle on N modules"). **Result:**
  Stage-1 keep-flips 1.47% → **0.0000%**; end-to-end coord-mismatch 25/30 → 0/30, **1γ+0X label
  flips → 0/30**. The selection is now membership-independent, so **small-sample A/B predicts
  full-dataset behavior**. **There were TWO bugs of the same class** (train-time RNG augmentations not
  gated by eval, each consuming the global RNG per event): (1) `shuffle_orders=True` on the deghoster,
  and (2) `CrossLevelAttn.max_source_tokens_per_level` random token subsample
  ([`refiners/cross_level.py` `_maybe_subsample`](../../../pointcept/models/LArFormer/refiners/cross_level.py),
  now gated by `not self.training`). The slicer Tier-B (`tools/capture_deghost_layers.py --target
  slicer`) localized the 2nd: slicer backbone + all tokenizer builders bit-clean, divergence enters at
  the `token_refiner` (3e-5) and the `query_selector` amplifies it 100× (a tiny score shift flips which
  tokens topk/FPS picks). **With BOTH fixed: 0/1,457,270 spacepoints differ across membership — truly
  per-event reproducible.** Re-baseline once (prior absolute results had both augmentations ON). Note:
  `--deterministic` is still required for *same-run* reproducibility; these fixes are for
  *cross-membership* (subsample/list/batch) reproducibility. (FP64/dead-band/allocator/argsort-ties
  were investigated and ruled out.)

### Flash recovery (prototype)

`flash_recovery_prototype.py` tests the lever: for own-cosmic-slice photons, predict
each slice's PMT pattern via PhotonLib (`larformer_analysis/lib/flash_predict`,
γ_beam=5.25 from the slicer `gamma_tune/summary.txt`) and Neyman-χ²-match to the
in-time beam flash. Result (117 of 127 1γ+0X own-cosmic photons): the photon's slice
is the **best** flash match in **31.6%** (rank-1) and top-3 in 55.6%. So flash matching
recovers ~1/3 of the mislabeled single photons → 1γ+0X efficiency 0.14 → ~0.17 (seg at
0.28) to ~0.24 (clean). Headroom via `chi2_with_oob` (TPC-OOB rejection) + per-point
charge + a χ² threshold instead of strict global rank-1.

### Visualization (`view_1g0X_flash.py`)

Multi-event Dash viewer that reads `stage3pred`+`merged_sp` directly and predicts each
slice's flash on the fly (PhotonLib, γ=5.25). Per event: observed in-time beam flash vs
predicted PE per PMT (photon slice in gold, min-χ² slice in orange), a 3D slice view
(gold=photon slice, blue=selected, red=true photon points), and a χ²-sorted slice table.
Scan the event list with prev/next or the dropdown (`inspect_1g0X.csv`, sorted with the
flash-recoverable own-cosmic cases first).

```bash
SIF=/cluster/tufts/wongjiradlabnu/larbys/larbys-container/pointcept_cuml.sif
ENV=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/ubdl/setenv_pointcept_container.sh
P=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept
DATADIR=/cluster/tufts/wongjiradlab/larbys/data/larformer/mcc9_v29e_dl_run3b_bnb_nu_overlay
apptainer exec --nv --bind /cluster:/cluster $SIF bash -c \
  "source $ENV >/dev/null 2>&1; export PYTHONPATH=$P:\$PYTHONPATH; \
   python3 view_1g0X_flash.py --event-list workdir_scale/inspect_1g0X.csv \
       --merged-dir $DATADIR/merged_sp --gamma 5.25 --port 8053"
# open http://<node>:8053   (GPU node recommended for fast on-the-fly prediction)
```

`inspect_1g0X.csv` columns: run,subrun,event,lead_photon_E,slice_category,reco_1g0X,
flash_rank,flash_recovered,stage3pred_path (378 events; 127 own-cosmic of which 37
flash-rank-1).

### Flash recovery end-to-end (implemented + validated)

`run_larformer_stage3_inference.py --flash-recover-k K [--flash-recover-chi2-max ... -oob-max ...
-gamma 5.25] [--flash-recover-augment-all]` adds the top-K flash-matched slices to the particle-
segmenter keep mask (helper `flash_recovery_keep`). Default = **rescue-only** (apply only when the
ν-keep is empty, so already-segmented events aren't contaminated). `run_stageB_capped.sh` honors
`FLASH_RECOVER_K`/`_CHI2_MAX`/`_OOB_MAX`/`_AUGMENT_ALL`. Cut calibration: `calibrate_flash_path.py`.

Result, **now re-confirmed deterministically** (full 3000, pinned single node, vs `base` eff
0.161 / purity 0.436):
- **K=1 rescue-only is net-negative: efficiency 0.135, purity 0.309** (loses TP *and* adds fakes).
  It un-drops events, but they are mostly fakes, not clean single photons.
- The earlier 378-only run *looked* like a gain — that was the run-to-run / order noise (now
  understood, see Reproducibility), not a real effect.

## Still to do

- **Fix the absolute-number order-dependence** (Reproducibility): try removing per-event
  `empty_cache` / `expandable_segments`, or a dead-band/hysteresis at the deghoster keep-cut
  (also mitigates the cross-GPU Hopper divergence). Then re-quote absolute 1γ+0X eff/purity.
- **Re-measure photon-finding** (0.71 / 0.83) on the deterministic `sp_compare_3k_pin/base`
  outputs for a clean citable number.
- The deghost-threshold and flash-rescue levers are **net-negative** — drop them; the gains were
  artifacts of non-determinism. Photon-finding losses to attack instead: photon-split (15%) and
  false-X (14%) on correctly-sliced ν events.
- POT-normalize for absolute yields (fold in `xsecWeight`); side-band / fake-rate study.
- Recover the ~18 OOM Stage-A tasks (16 GB) for the full 72k; scale Stage B beyond 3000.
- (If deploying on Hopper) add it to the conformance allowlist or quote a measured systematic.
