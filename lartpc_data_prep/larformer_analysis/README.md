# LArFormer Analysis

The scripts in this folder are meant to evaluate the potential usefulness of the LArFormer model for MicroBooNE analyses.

One of the key steps in the analysis chain is the selection of the neutrino interaction slice.

We can do this by 3D point topology, which is what LArFormer does.

We can also select events by the consistent of the 3D energy deposit pattern with the intime scintillation pattern.

We want to know:

1. the IOU for neutrino interactions
2. comparison of the predicted flash to the observed flash when we use (a) the ground truth spacepoints and (b) the predicted spacepoints from LArFormer.
3. We also want to know the relative chi-2 between the predicted flash and the observed flash when we use (a) the ground truth spacepoints and (b) the predicted spacepoints from LArFormer. This tells us if the quality of the larformer mask to find the nu spacepoints is good enough to match to the in-time scintillation flash.
4. Finally, we compare the chi-2 between all the predicted slices with the in-time flash. Is the predicted slice that best matches to the ground truth neutrino slice the one that has the lowest chi-2? 

We want these quantities for differnt types of interactions:
1. CC numu inclusive events
2. CC nue inclusive events
3. events with a pi0
4. events with one single visible photon. This is the class we hopw to improve selection efficiency and purity.

## Workflow

1. We need to run the flash info maker for all events in the validation and test set.
2. We run the larformer model in inference mode for all events in the validation and test set.
3. We run the analysis scripts to compare the predicted flash to the observed flash when we use (a) the ground truth spacepoints and (b) the predicted spacepoints from LArFormer.
4. Make plots from the analysis scripts.

## Script Info

The validation and test file list is on the Tufts cluster at:

  /cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/lartpc_data_prep/lantern_scripts/h5lists/h5list_mcall_lantern_valtest.txt


### Flash Info Maker

We already have the flash info maker script. It is located at `lartpc_data_prep/lantern_scripts/run_step5_flashinfo_pointcept_wconfig.sh`. 

The problem is that the configs are based on input lists which are lists of the source ROOT files, partitioned by interaction type.  Our val+test file list are h5 files. We could make a new inputlist of the input ROOT files based on the h5 files in the val+test h5 list. 

However, the flashinfo script requires both the input root file and the h5 file. 

The way to do this is probably to find the subset of the files within each of the dataset types (bnbnu_corsika.conf, bnb_nu_coriska_set2.confg, etc.) that has a file with the val+test h5list.

We then make a truncated input list for each dataset config.

### LArMatch Inference

The script for running the inference is `tools/run_slicer_inference.py`.  This takes in an input list of h5 files and outputs h5 files per event into an output directory.  This step is all set. We will not really have multiple GPU nodes to work with, so we probably need to run serially one 2xL40s node.

### Analysis Scripts

For each event, this script will require the following inputs:

1. the h5 training file which has ground truth information
2. the flashinfo file which has the observed flash information and ground truth flash-to-slice matching

In order to make the flash prediction in this script, we have a UB Photon library interface class in

  pointcept/models/event_slicer/photonlib.py

An example of making the flash prediction using ground truth spacepoint slices is in

  tools/visualize_slice_flash_match.py

When comparing the predicted and observed flash agreement between the different predicted masks, we will use the chi-2 between the predicted and observed flash. For many of the spacepoints, they will exist outside of the TPC due to time mismatch of when the particles that made the slice went through the detector and the time of the in-time flash.  We need to either (1) automaticaly reject spacepoint slices with some fraction of points outside the TPC, or (2) add some kind of penalty to the chi-2 for spacepoints outside the TPC. 

## Models trained

We have trained two models. Their configs are:

1. crosslevel model: `configs/lartpc/larformer-slicer-v1-cascaded-crosslevel.py`
2. PTv3+crosslevel hybrid: `configs/lartpc/larformer-slicer-v1-cascaded-ptv3hybrid-crosslevel.py`

### Checkpoints available locally for testing

1. crosslevel: `exp/larformer_slicer_v1_cascaded_crosslevelrefiner_mixedq_maskdn/model/model_iter_18619.pth`
2. PTv3+crosslevel: `exp/larformer_slicer_v1_cascaded_ptv3hybrid_crosslevel_nonzeroinit_maskdn_noamp/model/model_iter_30351.pth`

### Example of inference outputs

1. crosslevel: `exp/larformer_slicer_v1_cascaded_crosslevelrefiner_mixedq_maskdn/inference_iter_18619/`
2. PTv3+crosslevel: `exp/larformer_slicer_v1_cascaded_ptv3hybrid_crosslevel_nonzeroinit_maskdn_noamp/inference_iter_30351/`

The input list of the h5 files for the above can be found: `devdata_mergedh5_pi0filter_10files.txt`.

Corresponding flashinfo files for the h5 files in `devdata_mergedh5_pi0filter_10files.txt` can be found in `/mnt/ddrive/data/ub_on_tufts/h5/bnb_nu_pi0filter_corsika/flashinfo_h5/`.


## Implementation status (last updated 2026-05-30)

End-to-end pipeline complete and running at scale. First headline
results from the **pi0 (`bnb_nu_pi0filter_corsika`) val+test set →
crosslevelrefiner model, 1581 events** are in the
[Headline results](#headline-results--pi0-crosslevel-1581-events)
section below.

| Component | Status |
|---|---|
| Stage 0 — `build_valtest_rootlists.py` (rerun_lines + manifest) | ✅ |
| Stage 1 — extended `prepare_flashinfo_h5.py` (event_truth + nu_showers) | ✅ |
| Stage 2 — `tools/run_slicer_inference.py` (with 0-GT crash fix) | ✅ |
| Stage 3 — `analyze_event.py` (M1/M3/M4 + over-claim) | ✅ |
| Stage 4 — `aggregate_metrics.py` (event_summary.h5 + post-hoc γ) | ✅ |
| Stage 5 — `plot_metrics.py` (per-cat histograms + headline table) | ✅ |
| SLURM driver — `slurm/submit_valtest.sh` (STRIDE-chunked array) | ✅ |
| Tools — `inspect_perevent.py`, `inspect_gt.py`, `tune_gamma.py` | ✅ |
| Pi0 val+test on `crosslevelrefiner_mixedq_maskdn` (1581 events) | ✅ |
| Pi0 val+test on `ptv3hybrid_crosslevel_nonzeroinit_maskdn_noamp` | TODO |
| Other datasets (numu, nue, etc.) val+test | TODO |
| Calibrated γ from a non-placeholder source | TODO |

### Design choices (locked in 2026-05-23)

| Choice | Decision |
|---|---|
| **OOB chi-2 policy** | Reject slice if OOB fraction > threshold. Report a threshold sweep `[0.0, 0.05, 0.10, 0.20, 0.50]` so plots show sensitivity. |
| **Neutrino-slice IoU (M1)** | "Best-predicted-nu-slice IoU vs GT nu slice" — of all predicted slices the model labels nu (`class_argmax==nu`), pick max IoU vs GT-nu mask. Also carry `m1_iou_intent` (Hungarian-matched-query `pair_iou`) as the model-intent companion view. |
| **M4 chi-2 candidate pool** | Carry both: `m4_rank_all` (all predicted slices) and `m4_rank_nu` (model-nu pool only). |
| **Event categorization** | Per-event mctruth from the source dlmerged ROOT, baked into the flashinfo H5 at prep time. Categorizer reads from flashinfo only. |
| **Chi-2 model** | Neyman-style with systematic floor: `Σ (PE_obs − γ·PE_pred)² / (PE_obs + (f_sys·PE_obs)² + ε)`. Defaults `f_sys=0.10`, `ε=1.0`. |
| **GT-spacepoint baseline** | Per-GT-instance: GT slice whose `slices/matched_flash_idx` points at the in-time flash; PE prediction from its post-SPs. |
| **Empty-nu-prediction events** | `has_nu_prediction=False`, `m1=0.0`, `m3=NaN`; M4-all is still computed. |
| **Compute target** | Standalone Python — every per-event script takes `(merged_h5, flashinfo_h5, inference_h5)` argv and produces `perevent_<run>_<sub>_<evt>.h5`. Runs locally OR under SLURM array. |
| **Aggregate output** | H5 (not parquet). `event_summary.h5` (one row per event) + `category_summary.h5` (per-cat aggregates). |
| **PhotonLib cache** | `lartpc_data_prep/dat/photonlib_v6_70kV.npz` (overridable). |
| **Categories** | Independent bitmask over `{ccnumu, ccnue, pi0, single_vis_gamma, other}` — events can land in multiple (e.g., CC νμ + pi0). |
| **γ override mechanism** | `aggregate_metrics.py` accepts `--gamma-beam`/`--gamma-cosmic` to rescale χ² post-hoc from stored `pe_pred` arrays. No analyzer / inference rerun. |

### Directory layout

```
larformer_analysis/
├── README.md                          (this file)
├── __init__.py
│
├── build_valtest_rootlists.py         Stage 0
├── analyze_event.py                   Stage 3
├── aggregate_metrics.py               Stage 4 (with --gamma override)
├── plot_metrics.py                    Stage 5
│
├── tune_gamma.py                      γ tuning from existing perevent_*.h5
├── inspect_perevent.py                Diagnostic: dump perevent + raw inference
├── inspect_gt.py                      Diagnostic: per-merged-H5 GT instance count
├── smoke_test_lib.py                  Library smoke test
│
├── lib/
│   ├── __init__.py
│   ├── categorize.py                  mctruth → category bitmask
│   ├── flash_chi2.py                  Neyman + sys floor + OOB rejection
│   └── flash_predict.py               PhotonLib wrapper: slice → PE
│
└── slurm/
    ├── submit_valtest.sh              SBATCH wrapper (auto-sizes array)
    ├── run_valtest_per_fileno.py      Per-task driver (chunked by STRIDE)
    ├── valtest_pi0_crosslevel.conf    Per-(TAG × model) config
    └── valtest_pi0_ptv3crosslevel.conf
```

---

## Pipeline stages

### Stage 0 — Build val+test rerun-line files + manifest

**Why not just truncate the inputlist?** The production pipeline writes
the merged-H5's fileno from the **lineno** at job time, not from the
fileno embedded in the ROOT filename. A truncated inputlist re-numbers
everything → the flashinfo prep grabs the wrong merged H5 to pair with
each ROOT entry. Use the existing `RERUN_LINES_FILE` mechanism in the
wconfig: keep the original inputlist, point at a list of which
original linenos to process.

`build_valtest_rootlists.py` parses the val+test merged-h5 list,
extracts unique filenos per TAG (each fileno = original inputlist
lineno), and emits per TAG:

- **`rerun_lines/<TAG>.txt`** — one original lineno per line. Plug
  directly into the wconfig's `RERUN_LINES_FILE`.
- `manifest/<TAG>.csv` — one row per `(fileno, entry)` with full paths
  (root + merged_h5 + flashinfo_h5). Drives Stage 3 directly.
- `rootlists/<TAG>.txt` *(optional)* — deduped ROOT paths. Sanity-check
  artifact only.
- `summary.txt` — per-TAG counts.

`--conf-dir` is optional. Without it you still get `rerun_lines/` +
`manifest/` (with empty `root_path`).

**Usage**:
```bash
python build_valtest_rootlists.py \
    --h5-list  /cluster/.../h5list_mcall_lantern_valtest.txt \
    --output   $POINTCEPT/lartpc_data_prep/larformer_analysis/valtest \
    [--conf-dir $POINTCEPT/lartpc_data_prep/lantern_scripts/lantern_configs]
```

### Stage 1 — Flashinfo regen (extended schema)

`prepare_flashinfo_h5.py` was extended to write two new per-event groups:

- `entry_0/event_truth/`
  - attrs: `has_neutrino`, `nu_pdg`, `nu_energy_MeV`, `ccnc`, `mode`,
    `interaction_type`
  - datasets: `nu_vertex_xyz_cm`, `primary_pdg`, `primary_KE_MeV`,
    `primary_status`
- `entry_0/nu_showers/`
  - datasets: `pdg`, `start_xyz_cm`, `start_t_ns`, `detprofile_E_MeV`,
    `true_E_MeV`, `trackid`

Legacy flashinfo files load too (categorizer falls through to `other`).

**Regen needs the lantern container** (CVMFS-based). Drive via the
existing `run_step5_flashinfo_pointcept_wconfig.sh` with Stage 0's
rerun_lines as the `RERUN_LINES_FILE`:

```bash
RERUN_LINES_FILE=$POINTCEPT/.../valtest/rerun_lines/<TAG>.txt
stride=1
OFFSET=0
N=$(wc -l < $RERUN_LINES_FILE)
sbatch --array=0-$((N-1)) scripts/submit_flashinfo.sh configs/<TAG>.conf
```

### Stage 2 + 3 — SLURM array driver (inference + per-event analysis)

`slurm/submit_valtest.sh` + `slurm/run_valtest_per_fileno.py` package
Stage 2 (inference) + Stage 3 (analysis) into a SLURM array. Each task
processes a chunk of **`STRIDE` filenos** (default `STRIDE=50` —
tunable in the per-TAG conf).

Total array size = `ceil(N_filenos / STRIDE)`. For pi0 (1480 filenos)
with `STRIDE=50` that's **30 tasks**, well under typical
`AssocMaxSubmitJobLimit` caps.

Per task:
1. Reads `linenos[task_id*STRIDE : (task_id+1)*STRIDE]` from
   `rerun_lines/<TAG>.txt`.
2. Selects manifest rows for those filenos.
3. Runs `run_slicer_inference.py` **once** for the whole chunk (model
   loads once, amortized across all events).
4. Loops `analyze_event.py` per event.

**Setup**:
1. Copy `slurm/valtest_pi0_crosslevel.conf` to a per-(TAG × model) conf
   and edit `WORKDIR`, `MODEL_CONFIG`, `MODEL_WEIGHTS`, `OUTPUT_DIR`,
   `STRIDE`, `GAMMA_BEAM` / `GAMMA_COSMIC`, SBATCH knobs.
2. From a Tufts head node:
   ```bash
   sbatch lartpc_data_prep/larformer_analysis/slurm/submit_valtest.sh \
          lartpc_data_prep/larformer_analysis/slurm/<your>.conf
   ```
   The script auto-sizes the SLURM array and re-execs itself with the
   right `--array=0-N`, partition, time, etc.

**Per-task outputs** (under `${OUTPUT_DIR}`):
```
inference/<TAG>/slicerpred_merged_<TAG>_fileno<N>_entry<M>.h5
analysis/<TAG>/perevent_<run>_<sub>_<evt>.h5
_inputlists/<TAG>/task<NNNNNN>.txt
```

**Driver flags**:
- `--skip-inference` — analyzer only (slicerpred_*.h5 must already
  exist). Lets you re-run analysis with different knobs without
  re-doing inference.
- `--skip-analysis` — inference only.

### Stage 3 — Per-event analysis (`analyze_event.py`)

Takes `(merged_h5, flashinfo_h5, inference_h5)` → emits one
`perevent_<run>_<sub>_<evt>.h5` with:
- M1: panoptic best-nu-pred IoU + intent (matched-query pair_iou)
- M3: chi-2 of GT-baseline and best-nu slice; delta_chi2 sweep
- M4: rank of GT-best-IoU slice by χ² ascending (all-pool + nu-pool)
- Over-claim metrics (sp_level_nu_recall/precision, matched-query
  survival, per-nu-GT diagnostics)
- OOB threshold sweep across `[0.0, 0.05, 0.10, 0.20, 0.50]`

CLI:
```bash
python analyze_event.py \
    --merged-h5    .../merged_*.h5 \
    --flashinfo-h5 .../flashinfo_*.h5 \
    --inference-h5 exp/<model>/inference_*/slicerpred_*.h5 \
    --output-h5    out/<model>/perevent_<r>_<s>_<e>.h5 \
    --model-tag    <model_tag> \
    [--gamma-beam G_BEAM --gamma-cosmic G_COSMIC]
```

γ defaults are 1.0 — for absolute χ² magnitudes pass calibrated values,
OR keep γ=1.0 here and use the post-hoc γ override in Stage 4 (preferred
when iterating, since Stage 4 is fast).

### Stage 4 — Aggregate (`aggregate_metrics.py`)

Reads N per-event H5s → emits `event_summary.h5` (one row per event)
and `category_summary.h5` (per-category aggregates).

Default mode (γ from analyzer):
```bash
python aggregate_metrics.py \
    --perevent-dir ${OUTPUT_DIR}/analysis/${TAG} \
    --output-dir   ${OUTPUT_DIR}/summary
```

Post-hoc γ override mode — see [γ tuning](#post-hoc-γ-tuning-workflow).

### Stage 5 — Plot (`plot_metrics.py`)

Reads `event_summary.h5` → emits PDFs + PNGs + ASCII headline table:
- `m1_iou_hist.{pdf,png}` — per-category M1 IoU histogram
- `m1_panoptic_vs_intent.{pdf,png}` — over-claim diagnostic bars
- `m3_delta_chi2_box.{pdf,png}` — Δχ² per (cat × OOB)
- `m4_rank1_frac.{pdf,png}` — rank-1 fraction per (cat × OOB), all + nu
- `sp_level_nu_recall_hist.{pdf,png}` — per-category SP-recall histogram
- `overclaim_gap_hist.{pdf,png}` — per-pair pair_iou-argmax_iou
- `headline_table.txt` — ASCII summary of all per-cat headline numbers

```bash
python plot_metrics.py \
    --event-summary    ${OUTPUT_DIR}/summary/event_summary.h5 \
    --category-summary ${OUTPUT_DIR}/summary/category_summary.h5 \
    --output-dir       ${OUTPUT_DIR}/plots
```

---

## Per-event H5 schema (output of `analyze_event.py`)

```
attrs:
  run, subrun, event, model_tag, has_nu_prediction
  oob_thresholds (T,), default_oob_idx
  category_names (S32 array), nu_class_id, no_object_class_id

truth/                       (attrs only)
  category_mask (uint8), n_visible_nu_gammas, n_primary_pi0
  has_neutrino, nu_pdg, nu_energy_MeV, ccnc, mode, interaction_type

in_time_flash/
  attrs: flash_idx, t0_us, producer_id, paired_slice_id
  pe_obs (32,) float32          — observed in-time flash PE

gt_baseline/
  attrs: n_sp, oob_frac
  pe_pred (32,) float32         — PhotonLib prediction from GT-nu mask
  chi2    (T,)  float32         — per-threshold (NaN where oob-rejected)

pred_slices/
  attrs: n_pred_slices, n_pred_nu_slices
  query_id      (Q,)   int32
  class_argmax  (Q,)   int32    — nu_class_id=0, cosmic=1, no_object=2
  n_sp          (Q,)   int32
  iou_vs_gt_nu  (Q,)   float32
  oob_frac      (Q,)   float32
  chi2          (Q,T)  float32
  pe_pred       (Q,32) float32

metrics/
  attrs: has_nu_prediction
         m1_iou, m1_slice_id
         m1_iou_intent, m1_slice_id_intent
         m3_chi2_gt, m3_chi2_nu
  m3_delta_chi2 (T,) float32    — chi2_nu - chi2_gt per threshold
  m4_rank_all   (T,) int32      — rank in all-pool (-1 = none)
  m4_rank_nu    (T,) int32      — rank in nu-only pool

overclaim/
  attrs:
    sp_level_nu_recall, sp_level_nu_precision     float
    n_post_sp_total, n_post_sp_gt_nu              int
    n_post_sp_pred_nu, n_post_sp_true_and_pred_nu int
    n_nu_gt_instances                             int
    n_nu_gt_matched_to_nu_query                   int
    n_nu_match_survived_panoptic                  int
  per-nu-GT arrays (K_nu rows):
    gt_nu_idx                       (K_nu,) int32
    matched_query_id                (K_nu,) int32
    matched_query_class             (K_nu,) int32
    matched_query_pair_iou          (K_nu,) float32
    matched_query_argmax_iou        (K_nu,) float32
    matched_query_panoptic_n_sp     (K_nu,) int32
```

---

## Over-claim metrics

`tools/measure_overclaim.py` already instruments the "matched query
loses panoptic SPs" failure mode at the inference-output level. The
per-event analysis now carries the same diagnostic alongside the
panoptic-view headline metrics, so we can distinguish two related
failure modes:

| Metric | View | Definition |
|---|---|---|
| `m1_iou` | panoptic (analyzer) | Best IoU over slices with `post/pred_class == nu` after the SP argmax. What an analyzer that consumes the panoptic output sees. |
| `m1_iou_intent` | per-pair (model) | Among Hungarian-matched nu queries, the highest `gt/pair_iou`. What the model intends before per-SP argmax competition. |
| `sp_level_nu_recall` | SP confusion | Fraction of GT-nu post-SPs that panoptic argmax labels nu. The downstream signal-quality number. |
| `sp_level_nu_precision` | SP confusion | Fraction of panoptic-nu SPs that are truly nu. |
| `frac_nu_match_survived_panoptic` | per-event | Of nu-GT instances whose matched query is class-correct nu, fraction whose matched query won ≥1 SP in panoptic. |
| `overclaim_gap = pair_iou - argmax_iou` | per-nu-GT | Same value `measure_overclaim.py` reports, but per-pair-per-event for joint analysis with categories + flash-match outcomes. |

**Two failure modes the pair distinguishes**:
- **Class-head failure**: `n_pred_nu == 0` AND `n_nu_match_survived_panoptic == 0` — the matched query has the wrong class.
- **Competitor over-claim**: `n_pred_nu > 0` AND `m1_iou_intent > m1_iou_panoptic` — class-correct matched query lost SPs to over-claiming competitors.

The aggregator surfaces these as `(N,)` arrays under `events/` plus a
flat `nu_pairs/` group (CSR-style cross-event pair pool). The plot
script renders three over-claim plots: `m1_panoptic_vs_intent`,
`sp_level_nu_recall_hist`, `overclaim_gap_hist` (per category).

---

## Post-hoc γ tuning workflow

The analyzer runs with γ=1.0 by default. Because χ² is linear in γ
(`pe_pred` enters the residual via `γ·pe_pred` and the variance depends
only on `pe_obs`), we can recompute M3 + M4 under any new γ straight
from the recorded `pe_pred` arrays. **M1 IoU and `sp_level_*` are
γ-independent and unchanged.** **No analyzer or inference rerun.**

### Step 1 — estimate γ from existing `perevent_*.h5`

```bash
python tune_gamma.py \
    --perevent-dir ${OUTPUT_DIR}/analysis/${TAG} \
    --output-dir   ${OUTPUT_DIR}/gamma_tune \
    --per-pmt-fit
```

Produces scatter + ratio histograms for three view-pairs, with γ
estimators on each:

- **(A) GT-nu baseline** — truth-side γ. Can't be done on real data.
- **(B) M1 panoptic-nu slice** — analyzer-faithful γ (what's available
  on real data). **Use this for the analyzer.**
- **(C) Σ all model-nu slices** — multi-slice fallback (matches "if you
  treat every model-nu slice as the candidate"). Equivalent to B when
  there's only one nu pred per event.

Four estimators per view:
- `γ_median` — `median(pe_obs / pe_pred)` per event; robust.
- `γ_mean` — mean of the same ratios.
- `γ_ratio_of_sums` — `Σ pe_obs / Σ pe_pred`. De-weights small events;
  usually the best choice for "the value to bake into χ²."
- `γ_lsq` — closed-form least-squares slope through origin using every
  per-PMT pair.

### Step 2 — re-aggregate with the chosen γ

```bash
python aggregate_metrics.py \
    --perevent-dir ${OUTPUT_DIR}/analysis/${TAG} \
    --output-dir   ${OUTPUT_DIR}/summary_g225 \
    --gamma-beam 225  --gamma-cosmic 225
```

The aggregator reads `pe_obs`, `gt_baseline/pe_pred`, and
`pred_slices/pe_pred` from each `perevent_*.h5`, scales `pe_pred` by
the producer's γ, and recomputes M3 + M4 from the stored OOB-fractions
(OOB rejection is geometry-only and γ-independent — reused as-is).

γ used + Neyman-floor knobs are stamped on `event_summary.h5.attrs`
(`gamma_override_applied`, `gamma_beam`, `gamma_cosmic`, `f_sys`,
`eps`). Setting only one of `--gamma-beam` / `--gamma-cosmic` defaults
the other to 1.0 and warns.

### Step 3 — re-plot

```bash
python plot_metrics.py \
    --event-summary    ${OUTPUT_DIR}/summary_g225/event_summary.h5 \
    --category-summary ${OUTPUT_DIR}/summary_g225/category_summary.h5 \
    --output-dir       ${OUTPUT_DIR}/plots_g225
```

A/B several γ values cheaply: keep one
`${OUTPUT_DIR}/analysis/<TAG>/` tree of `perevent_*.h5` files and
spin off `summary_g100/`, `summary_g225/`, `summary_g350/`, … with
re-plotted output for each.

---

## Diagnostic tools

| Tool | What it answers |
|---|---|
| `inspect_perevent.py` | Dump one `perevent_*.h5`'s metrics, pred-slice table, over-claim diagnostic. With `--inference-h5`, also pulls the source `queries/class_argmax`, `queries/class_probs`, `gt/*` for a side-by-side comparison of "what the perevent recorded" vs "what the inference actually said." |
| `inspect_gt.py` | For each merged H5 in a list (positional args or `--list`), report `(n_sp, n_gt, n_nu_gt, n_cosmic_gt)`. Default mode flags only `n_gt==0` rows; `--all` prints every file; `--csv` dumps a flat CSV. Use to identify zero-GT events (cosmic-only or fully-filtered events) in a task's inputlist. |
| `tune_gamma.py` | Scatter + ratio histograms + four γ estimators per view-pair, from existing `perevent_*.h5`. See [Post-hoc γ tuning workflow](#post-hoc-γ-tuning-workflow). |
| `smoke_test_lib.py` | Library smoke test for `lib.categorize`, `lib.flash_chi2`, `lib.flash_predict` against a real event triple. Used during development. |

---

## Headline results — pi0 crosslevel, 1581 events

`crosslevelrefiner_mixedq_maskdn`, `bnb_nu_pi0filter_corsika` val+test,
~15% of the dataset processed so far. **γ=1.0 (analyzer default)**:

```
has_nu_prediction frac: 98.36%
m1_iou (panoptic): mean=0.568, med=0.604
m1_iou (intent):   mean=0.559, med=0.643
sp_level_nu_recall:    mean=0.906
sp_level_nu_precision: mean=0.610
frac_nu_match_survived_panoptic: 99.94% (1546/1547)
oob<=0.00:  Δχ² mean=+0.55    rank1_all=13.69%  rank1_nu=96.70%
oob<=0.05:  Δχ² mean=-0.05    rank1_all= 5.21%  rank1_nu=96.93%
oob<=0.10:  Δχ² mean=-0.07    rank1_all= 4.59%  rank1_nu=96.96%
oob<=0.20:  Δχ² mean=-0.08    rank1_all= 3.62%  rank1_nu=96.78%
oob<=0.50:  Δχ² mean=-0.08    rank1_all= 2.35%  rank1_nu=96.78%
```

**Same events at γ=225** (estimated from view B's `γ_ratio_of_sums`
on this sample):

```
has_nu_prediction frac: 98.36%       (γ-independent)
m1_iou (panoptic): mean=0.568        (γ-independent)
sp_level_nu_recall:    mean=0.906    (γ-independent)
frac_nu_match_survived_panoptic: 99.94%
oob<=0.00:  Δχ² mean=+118500.03  rank1_all=81.87%  rank1_nu=97.28%
oob<=0.05:  Δχ² mean=+107143.68  rank1_all=75.71%  rank1_nu=97.52%
oob<=0.10:  Δχ² mean=+106103.69  rank1_all=75.38%  rank1_nu=97.48%
oob<=0.20:  Δχ² mean=+105506.23  rank1_all=74.60%  rank1_nu=97.43%
oob<=0.50:  Δχ² mean=+105438.37  rank1_all=73.18%  rank1_nu=97.36%
```

### What's working (γ-independent — robust signal)

- **`has_nu_prediction = 98.4%`** — class head fires on essentially
  every event with a nu GT.
- **`frac_nu_match_survived_panoptic = 99.94%`** — the matched nu query
  wins panoptic argmax. Competitor over-claim is essentially zero.
- **`sp_level_nu_recall = 0.91`** — 91% of true-nu SPs end up labeled
  nu by panoptic argmax. The model finds the nu cluster.
- **`m1 panoptic ≈ m1 intent`** (0.568 vs 0.559) — the two
  over-claim-diagnostic views agree in aggregate; the model's matched
  query is the model's panoptic-winning nu slice in ~all events.

### What's noteworthy

- **`sp_level_nu_precision = 0.61`** — only 61% of panoptic-nu SPs are
  truly nu. The matched query is winning the right SPs **plus extras**.
  Intra-query over-claim, not competitor over-claim.
- **`rank1_nu ≈ 97%` across all OOB thresholds** — when restricted to
  model-labeled-nu slices, the model's choice is the lowest-χ² candidate
  ~97% of the time. **Class call + flash match are mutually consistent.**
- **`rank1_all`**: at γ=1.0 just 2–14% (chi-2 alone identifies the nu
  rarely); at γ=225 jumps to **73–82%**. **Flash matching becomes a
  useful discriminator only at properly-calibrated γ.**
- **Sign-flip of `Δχ²` between γ=1 and γ=225**: at γ=1 the predicted-nu
  slice has fractionally better χ² than the GT baseline (Δχ²≈−0.07).
  At γ=225 the over-claim's cost in flash-matching becomes visible:
  the predicted slice's χ² is much worse than the GT baseline (Δχ² ~
  +10⁵), because the cosmic SPs in the predicted nu mask add light at
  PMTs the in-time flash isn't lit on.

### Interpretation for the science target

For an analyzer:
- **With class filter**: ~97% rank-1 within the nu pool at any OOB
  threshold. Use the model's class call to pre-filter, then χ² as the
  in-pool tiebreaker.
- **Without class filter** (at proper γ): ~75% rank-1 across all
  slices. χ² alone is meaningfully discriminating but not sufficient.

Worth checking next:
1. γ tuned from view (A) GT-nu baseline → likely 300–370 → Δχ² should
   flip sign in the other direction. The magnitude of the γ_A vs γ_B
   gap tells you how much the over-claim costs in flash-matching terms.
2. Per-category breakdown in `headline_table.txt` — especially
   `single_vis_gamma` category (the science target), if any events
   land in it.
3. The 1 event in 1547 where the class-correct matched nu query failed
   to survive panoptic argmax — likely an outlier worth inspecting via
   `inspect_perevent.py`.

---

## Operational notes

### `STRIDE` (SLURM array sizing)

`STRIDE` in the per-(TAG × model) conf controls how many filenos one
SLURM task handles. Total array tasks = `ceil(N_filenos / STRIDE)`.
Tune to your cluster's `AssocMaxSubmitJobLimit`: for Tufts with the
default ~100-task cap and pi0's 1480 filenos, `STRIDE=50` (→ 30 tasks)
is comfortable. `STRIDE=100` → 15 tasks if you want to stay further
under cap; `STRIDE=25` → 60 tasks if you want to finish faster.

### 0-GT inference crash fix

`tools/run_slicer_inference.py:_per_sp_predicted_slice` crashed when
processing events with 0 GT instances (cosmic-only / fully-filtered
events). `np.where(matched_k >= 0, primary_trackid[matched_k.clip(min=0)],
-1)` evaluates both branches even when the predicate is False
everywhere → `IndexError` on the empty `primary_trackid` array.

Fixed (2026-05-29) by short-circuiting to all-`-1` `pred_slice` when
`primary_trackid.size == 0`. Use `inspect_gt.py --list` against a
failing task's `_inputlists/<TAG>/task*.txt` to identify the offending
event(s).

### Bug fix history: `pred_slice_id` vs `pred_query`

While building the over-claim metric, an indexing bug surfaced (2026-05-23):
`analyze_event.py` keyed predicted slices off `post/pred_slice_id`,
which for matched queries is the **GT primary_trackid** (a huge integer
like 1172344) — NOT the query id. Class lookups via
`q_class_argmax[1172344]` silently fell out-of-bounds and defaulted
to `no_object`, spuriously reporting `n_pred_nu_slices=0` for events
where the model had clear nu predictions. Fixed in `analyze_event.py`
(`_compute_pred_slices`, `_compute_overclaim`) and `inspect_perevent.py`
to key off `post/pred_query` (the actual query id).

---

## Outstanding work

1. Pi0 val+test on the `ptv3hybrid_crosslevel_nonzeroinit_maskdn_noamp`
   model — second SLURM submission with `valtest_pi0_ptv3crosslevel.conf`.
2. Other datasets (numu, nue, ccpi+, etc.) val+test on both models —
   per-TAG flashinfo regen on Tufts + per-TAG SLURM array.
3. Calibrated γ from a non-placeholder source. Currently the
   analyzer-side γ tuning (view B `γ_ratio_of_sums ≈ 225` on pi0
   crosslevel) is data-driven; replace with the calibrated photon-
   library-aware value once available.
4. γ from view A vs view B comparison plot (currently the user must
   eyeball summary.txt's three blocks).
5. The 1-in-1547 matched-nu panoptic failure event — drill in with
   `inspect_perevent.py`, categorize.
6. `single_vis_gamma` category coverage in the val+test sample —
   verify the heuristic (`vis_gamma_E_thresh_MeV=35`,
   FV-inset 10 cm) is producing the expected fraction.
7. Side-by-side comparison plots across (model, γ) combinations
   in `plot_metrics.py` — currently each summary is plotted on its
   own.