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


## Implementation Plan

### Design choices (locked in 2026-05-23)

| Choice | Decision |
|---|---|
| **OOB chi-2 policy** | Reject slice if OOB fraction > threshold. Report a threshold sweep `[0.0, 0.05, 0.10, 0.20, 0.50]` so plots can show sensitivity. |
| **Neutrino-slice IoU (M1)** | "Best-predicted-nu-slice IoU vs GT nu slice" — analyzer-oriented: of all predicted slices the model labels nu (queries with `class_argmax==nu`), pick the one with max IoU vs the GT nu mask. |
| **M4 chi-2 candidate pool** | Carry both: `m4_rank_all` (all predicted slices, regardless of model class label) and `m4_rank_nu` (restricted to model-nu pool). |
| **Event categorization source** | Per-event mctruth from the source dlmerged ROOT, baked into the flashinfo H5 at prep time. Categorizer reads from flashinfo only — no second pass through ROOT. |
| **Chi-2 model** | Neyman-style with constant systematic floor: `Σ (PE_obs − PE_pred)² / (PE_obs + (f_sys·PE_obs)² + ε)`. Defaults: `f_sys=0.10`, `ε=1.0`. |
| **GT-spacepoint baseline** | Per-GT-instance: pick the GT slice that `flashinfo['slices']/matched_flash_idx` points at the in-time flash. |
| **Empty-nu-prediction events** | Record as `has_nu_prediction=False`, `m1=0.0`, `m3=NaN`; M4-all is still computed from the unrestricted pool. |
| **Compute target** | Standalone Python — every per-event script takes `(merged_h5, flashinfo_h5, inference_h5)` argv and produces `perevent_<run>_<sub>_<evt>.h5`. Same script runs locally in a for-loop or under a SLURM array (`$SLURM_ARRAY_TASK_ID`-th line of a manifest). |
| **Aggregate output** | H5 (not parquet). `event_summary.h5` carries one row per event under `events/`; `category_summary.h5` carries per-category aggregates. |
| **PhotonLib cache** | `lartpc_data_prep/dat/photonlib_v6_70kV.npz` (overridable). |
| **Categories** | Independent bitmask over `{ccnumu, ccnue, pi0, single_vis_gamma, other}` — events can land in multiple (e.g., CC νμ + pi0). |

### Directory layout

```
larformer_analysis/
├── README.md                          (this file)
├── build_valtest_rootlists.py         (Stage 0)
├── analyze_event.py                   (Stage 3 — TODO)
├── aggregate_metrics.py               (Stage 4 — TODO)
├── plot_metrics.py                    (Stage 5 — TODO)
├── smoke_test_lib.py                  (Pass-1 verification)
├── lib/
│   ├── categorize.py                  (mctruth → category bitmask)
│   ├── flash_chi2.py                  (Neyman + sys floor + OOB rejection)
│   └── flash_predict.py               (PhotonLib wrapper: slice → PE)
└── slurm/                             (SLURM array drivers — TODO)
```

### Stage 0 — Build val+test rerun-line files + manifest

**Why not just truncate the inputlist?** The production pipeline writes
the merged-H5's fileno from the **lineno** at job time, not from the
fileno embedded in the ROOT filename. A truncated inputlist re-numbers
everything → the flashinfo prep grabs the wrong merged H5 to pair with
each ROOT entry. Use the existing `RERUN_LINES_FILE` mechanism in the
wconfig: keep the original inputlist, point at a list of which
original linenos to process.

`build_valtest_rootlists.py` parses the val+test merged-h5 list,
extracts the unique filenos per TAG (each fileno IS the original
inputlist lineno), and emits per TAG:

- **`rerun_lines/<TAG>.txt`** — one original lineno per line. Plug
  directly into the wconfig's `RERUN_LINES_FILE`.
- `manifest/<TAG>.csv` — one row per `(fileno, entry)` with full paths
  (root + merged_h5 + flashinfo_h5). Drives Stage 3 directly. The
  `root_path` column is filled only when `--conf-dir` is provided AND
  the ROOT inputlist has a file matching that fileno.
- `rootlists/<TAG>.txt` *(optional)* — deduped ROOT paths for the
  val+test filenos. **Sanity-check artifact only** — NOT a flashinfo
  input (truncating breaks the fileno numbering as explained above).
- `summary.txt` — per-TAG counts.

`--conf-dir` is **optional**. Without it you still get
`rerun_lines/` + `manifest/` (with empty `root_path`); with it you
also get `rootlists/` + the `manifest.root_path` filled in.

**Usage**:
```bash
python build_valtest_rootlists.py \
    --h5-list  /cluster/.../h5list_mcall_lantern_valtest.txt \
    --output   $POINTCEPT/lartpc_data_prep/larformer_analysis/valtest \
    [--conf-dir $POINTCEPT/lartpc_data_prep/lantern_scripts/lantern_configs]
```

**Then drive Stage 1 (flashinfo regen) per TAG via rerun mode**:
```bash
# In the .conf file (or as env overrides):
RERUN_LINES_FILE=$POINTCEPT/lartpc_data_prep/larformer_analysis/valtest/rerun_lines/<TAG>.txt
stride=1
OFFSET=0

N_FILENOS=$(wc -l < $RERUN_LINES_FILE)
sbatch --array=0-$((N_FILENOS-1)) scripts/submit_flashinfo.sh \
       configs/<TAG>.conf
```

The wconfig's existing rerun branch uses `$SLURM_ARRAY_TASK_ID`-th
line of the rerun file as the lineno → reads ROOT entry `lineno-1` of
the ORIGINAL inputlist → preserves the production fileno numbering →
pairs correctly with `merged_<TAG>_fileno<lineno>_entry*.h5`.

### Stage 1 — Flashinfo (extended)

`prepare_flashinfo_h5.py` was extended (2026-05-23) with two new
per-event groups:

- `entry_0/event_truth/`
  - attrs: `has_neutrino`, `nu_pdg`, `nu_energy_MeV`, `ccnc`, `mode`,
    `interaction_type`
  - datasets: `nu_vertex_xyz_cm`, `primary_pdg`, `primary_KE_MeV`,
    `primary_status`
- `entry_0/nu_showers/` (per nu-origin mcshower)
  - datasets: `pdg`, `start_xyz_cm`, `start_t_ns`, `detprofile_E_MeV`,
    `true_E_MeV`, `trackid`

Legacy flashinfo files (without these groups) still load via the
existing reader code — `lib.categorize` falls through to the `other`
category when fields are absent.

**Regen needs the lantern container** (CVMFS-based). Drive via the
existing `run_step5_flashinfo_pointcept_wconfig.sh` with the val+test
ROOT lists from Stage 0.

### Stage 2 — LArFormer inference (existing tool)

Use `tools/run_slicer_inference.py` with the val+test merged-h5 list
and each model checkpoint. No script changes needed.

### Stage 2+3 — SLURM array driver (inference + per-event analysis)

`slurm/submit_valtest.sh` + `slurm/run_valtest_per_fileno.py` package
Stage 2 (inference) + Stage 3 (analysis) into a SLURM array. Each task
processes a **chunk of `STRIDE` filenos** (default `STRIDE=50` — set
in the per-TAG conf, tune to your cluster's `AssocMaxSubmitJobLimit`).

Total array size = `ceil(N_filenos / STRIDE)`. For pi0 (1480 filenos)
with `STRIDE=50` that's **30 tasks**, well under typical QOS caps.

Per task: reads `linenos[task_id*STRIDE : (task_id+1)*STRIDE]` from
`rerun_lines/<TAG>.txt`, selects the matching manifest rows,
runs `run_slicer_inference.py` **once** for the whole chunk (model
loads once, amortized across all events in the chunk), then loops
`analyze_event.py` per event.

**Setup**:
1. Copy `slurm/valtest_pi0_crosslevel.conf` to a per-TAG×model config
   and edit `WORKDIR`, `MODEL_CONFIG`, `MODEL_WEIGHTS`, `OUTPUT_DIR`,
   `GAMMA_BEAM` / `GAMMA_COSMIC` (and the SBATCH knobs at the bottom).
2. From the repo root on a Tufts head node:
   ```bash
   sbatch lartpc_data_prep/larformer_analysis/slurm/submit_valtest.sh \
          lartpc_data_prep/larformer_analysis/slurm/<your>.conf
   ```
   The script auto-sizes the SLURM array from the rerun-lines file and
   re-exec's itself with the right `--array=0-N`, partition, time, etc.

**Outputs per task** (under `${OUTPUT_DIR}`):
```
inference/<TAG>/slicerpred_merged_<TAG>_fileno<N>_entry<M>.h5
analysis/<TAG>/perevent_<run>_<sub>_<evt>.h5
_inputlists/<TAG>/fileno<N>.txt        # the tiny inputlist per task
```

**Useful flags on the per-task driver**:
- `--skip-inference`: run analyzer only (slicerpred_*.h5 must already
  exist). Lets you re-run analysis with different γ / OOB knobs without
  re-doing inference.
- `--skip-analysis`: inference only — useful for one model checkpoint
  → many analyzer reruns.

**After the array completes**:
```bash
# Stage 4: aggregate
python lartpc_data_prep/larformer_analysis/aggregate_metrics.py \
    --perevent-dir ${OUTPUT_DIR}/analysis/${TAG} \
    --output-dir   ${OUTPUT_DIR}/summary

# Stage 5: plot
python lartpc_data_prep/larformer_analysis/plot_metrics.py \
    --event-summary    ${OUTPUT_DIR}/summary/event_summary.h5 \
    --category-summary ${OUTPUT_DIR}/summary/category_summary.h5 \
    --output-dir       ${OUTPUT_DIR}/plots
```

### Post-hoc γ tuning (no rerun of inference / analyzer)

The analyzer runs with γ=1.0 by default. Because χ² is linear in γ
(`pe_pred` enters the residual via `pe_pred · γ` and the variance
depends only on `pe_obs`), you can recompute M3 + M4 under any new γ
straight from the recorded `pe_pred` arrays. M1 IoU and `sp_level_*`
are γ-independent and unchanged.

Two-step workflow:

```bash
# 1. Estimate γ from the analyzer's already-written perevent_*.h5
python lartpc_data_prep/larformer_analysis/tune_gamma.py \
    --perevent-dir ${OUTPUT_DIR}/analysis/${TAG} \
    --output-dir   ${OUTPUT_DIR}/gamma_tune

# Pick a value from the (B) M1-panoptic-nu view → γ_ratio_of_sums.
# That's the analyzer-faithful γ (uses what the model predicts, not
# the truth-side baseline).

# 2. Re-aggregate with the new γ; plots then reflect the rescaled M3/M4.
python lartpc_data_prep/larformer_analysis/aggregate_metrics.py \
    --perevent-dir ${OUTPUT_DIR}/analysis/${TAG} \
    --output-dir   ${OUTPUT_DIR}/summary_gamma237 \
    --gamma-beam   237  --gamma-cosmic 237

python lartpc_data_prep/larformer_analysis/plot_metrics.py \
    --event-summary    ${OUTPUT_DIR}/summary_gamma237/event_summary.h5 \
    --category-summary ${OUTPUT_DIR}/summary_gamma237/category_summary.h5 \
    --output-dir       ${OUTPUT_DIR}/plots_gamma237
```

The aggregator stamps `gamma_beam`, `gamma_cosmic`, `f_sys`, `eps`, and
`gamma_override_applied` on `event_summary.h5.attrs` so downstream
readers (and a future side-by-side comparison plot) can tell which γ
the summary was built with.

Per-producer γ: in BNB nu-on-cosmic samples the in-time flash is always
beam (`producer_id==0`), so `--gamma-cosmic` only matters if you point
`tune_gamma` / the analyzer at cosmic-only or out-of-time event sets.
Setting only one of `--gamma-beam` / `--gamma-cosmic` defaults the
other to 1.0 and warns.

If you want to A/B several γ values, run the aggregator multiple times
into separate output dirs (`summary_g100/`, `summary_g237/`, etc.) and
re-plot each. The analyzer + inference outputs (the slow part) are
written once and reused.

### Stage 3 — Per-event analysis (TODO — Pass 2)

`analyze_event.py` takes `(merged_h5, flashinfo_h5, inference_h5)` →
emits one `perevent_<run>_<sub>_<evt>.h5` with M1/M3/M4 + threshold
sweep, per-slice arrays, and per-PMT residuals.

### Stage 4 + 5 — Aggregate + plot (TODO — Pass 2)

`aggregate_metrics.py` reads all per-event H5s → emits
`event_summary.h5` + `category_summary.h5`. `plot_metrics.py` consumes
the summary H5s.

---

## Pass 1 status (2026-05-23) — COMPLETE

| Component | Status | Tested on |
|---|---|---|
| `prepare_flashinfo_h5.py` (extended) | ✅ extension committed | needs lantern container for regen |
| `lib/categorize.py` | ✅ smoke-tested | legacy flashinfo → `other` (graceful fallback) |
| `lib/flash_chi2.py` | ✅ smoke-tested | synthetic SP set; Neyman + OOB sweep behavior verified |
| `lib/flash_predict.py` | ✅ smoke-tested | end-to-end on real event: PhotonLib load + PE computation for GT-nu slice |
| `build_valtest_rootlists.py` | ✅ smoke-tested | synthetic h5 list against local data; emitted manifest + rootlist + summary |
| `smoke_test_lib.py` | ✅ passing | runs against `fileno00013_entry000004` triple in the pointcept container |

**Pending user-side actions before Pass 2**:

1. **Regenerate flashinfo on Tufts** for the val+test set using the
   extended `prepare_flashinfo_h5.py` (needs lantern container).
2. **Run Stage 0** on Tufts to generate the per-TAG manifests + rootlists.
3. **Run Stage 2 inference** for both models on the val+test set.

Once those are done, Pass 2 (`analyze_event.py`, `aggregate_metrics.py`,
`plot_metrics.py`) is ready to start.

**Local Pass-1 smoke test**:
```bash
cd $POINTCEPT
./run_in_container.sh python lartpc_data_prep/larformer_analysis/smoke_test_lib.py
```

Last run output:
- categorize: legacy flashinfo → `other` ✅
- flash_chi2: uniform-offset chi-2 = 8.86 (expected ~8.87); OOB sweep ✅
- flash_predict: PhotonLib load + 1962-SP GT-nu PE prediction = 18.9 PE
  (γ=1.0 placeholder; real γ needed for physically meaningful χ²) ✅

---

## Pass 2 status (2026-05-23) — COMPLETE end-to-end

All three Pass-2 scripts wired and validated on 3 events × 2 models
(`bnb_nu_pi0filter_corsika` fileno00001 entries 0/1/2):

| Component | Status | Notes |
|---|---|---|
| `analyze_event.py`     | ✅ | Per-event H5 with M1/M3/M4 + threshold sweep + per-slice arrays + per-PMT residuals |
| `aggregate_metrics.py` | ✅ | `event_summary.h5` (N rows) + `category_summary.h5` (per-category aggregates) |
| `plot_metrics.py`      | ✅ | M1 IoU histos, M3 Δχ² box, M4 rank-1 bars, headline_table.txt |

**Stage-3 → 4 → 5 invocation pattern** (per model):

```bash
# Stage 3: one perevent_*.h5 per event
for ENTRY in 000000 000001 000002; do
  ./run_in_container.sh python lartpc_data_prep/larformer_analysis/analyze_event.py \
      --merged-h5    .../merged_..._entry${ENTRY}.h5 \
      --flashinfo-h5 .../flashinfo_..._entry${ENTRY}.h5 \
      --inference-h5 exp/<model_run>/inference_*/slicerpred_..._entry${ENTRY}.h5 \
      --output-h5    out/<model_tag>/perevent_entry${ENTRY}.h5 \
      --model-tag    <model_tag> \
      [--gamma-beam G_BEAM --gamma-cosmic G_COSMIC]    # default 1.0 placeholder
done

# Stage 4: aggregate
./run_in_container.sh python lartpc_data_prep/larformer_analysis/aggregate_metrics.py \
    --perevent-dir out/<model_tag>/ \
    --output-dir   out/<model_tag>/summary/

# Stage 5: plot
./run_in_container.sh python lartpc_data_prep/larformer_analysis/plot_metrics.py \
    --event-summary    out/<model_tag>/summary/event_summary.h5 \
    --category-summary out/<model_tag>/summary/category_summary.h5 \
    --output-dir       out/<model_tag>/plots/
```

For SLURM array deployment, wrap each Stage-3 invocation in a one-line
driver that takes `$SLURM_ARRAY_TASK_ID` and pulls the N-th row from
the Stage-0 manifest CSV. Stages 4 + 5 run once after the array is done.

### Per-event H5 schema (output of `analyze_event.py`)

```
attrs:
  run, subrun, event, model_tag, has_nu_prediction
  oob_thresholds (T,), default_oob_idx, category_names (S32 array)
  nu_class_id, no_object_class_id

truth/                       (attrs only)
  category_mask (uint8), n_visible_nu_gammas, n_primary_pi0
  has_neutrino, nu_pdg, nu_energy_MeV, ccnc, mode, interaction_type

in_time_flash/
  attrs: flash_idx, t0_us, producer_id, paired_slice_id
  pe_obs (32,) float32     — observed in-time flash PE

gt_baseline/
  attrs: n_sp, oob_frac
  pe_pred (32,) float32    — PhotonLib prediction from GT-nu mask
  chi2    (T,)  float32    — per-threshold (NaN where oob_frac > thresh)

pred_slices/
  attrs: n_pred_slices, n_pred_nu_slices
  query_id      (Q,)   int32
  class_argmax  (Q,)   int32  — nu_class_id=0, cosmic=1, no_object=2
  n_sp          (Q,)   int32
  iou_vs_gt_nu  (Q,)   float32
  oob_frac      (Q,)   float32
  chi2          (Q,T)  float32  (NaN per-pair where oob_frac > thresh)
  pe_pred       (Q,32) float32

metrics/
  attrs: has_nu_prediction, m1_iou, m1_slice_id, m3_chi2_gt, m3_chi2_nu
  m3_delta_chi2 (T,) float32  — chi2_nu - chi2_gt per threshold
  m4_rank_all   (T,) int32    — rank of GT-best slice in ALL pool (-1=none)
  m4_rank_nu    (T,) int32    — same, restricted to nu pool
```

## Over-claim metrics (added 2026-05-23)

`tools/measure_overclaim.py` already instruments the "matched query
loses panoptic SPs" failure mode at the inference-output level. The
per-event analysis now carries the same diagnostic alongside the
panoptic-view headline metrics, so we can distinguish the model from
the analyzer:

| Metric | View | Definition |
|---|---|---|
| `m1_iou` | panoptic (analyzer)   | Best IoU over slices with `post/pred_class == nu` after the SP argmax. What an analyzer that consumes the panoptic output sees. |
| `m1_iou_intent` | per-pair (model)      | Among Hungarian-matched nu queries, the highest `gt/pair_iou` (above-threshold mask). What the model *intends* to predict before the per-SP argmax competition. |
| `sp_level_nu_recall` | SP confusion         | Fraction of GT-nu post-SPs that the panoptic argmax labels nu. The direct "downstream signal quality" number. |
| `sp_level_nu_precision` | SP confusion         | Fraction of panoptic-nu SPs that are truly nu. |
| `frac_nu_match_survived_panoptic` | per-event diagnostic | Of nu-GT instances whose Hungarian-matched query is class-correct nu, how many of those matched queries won ≥1 SP in panoptic. |
| `overclaim_gap = pair_iou - argmax_iou` | per-nu-GT           | Same value `tools/measure_overclaim.py` reports, but now per-pair-per-event for joint analysis with categories + flash-match outcomes. |

Two failure modes the pair tells apart:
- **Class-head failure**: `n_pred_nu == 0` AND `n_nu_match_survived_panoptic == 0` — the matched query has the wrong class.
- **Over-claim failure**: `n_pred_nu > 0` AND `m1_iou_intent > m1_iou_panoptic` — the matched query is class-correct but lost SPs to over-claiming competitors.

### Storage

`overclaim/` group inside `perevent_*.h5`:
```
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

The aggregator surfaces these as `(N,)` arrays under `events/` and a
flat `nu_pairs/` group (CSR-style cross-event-pair pool). The plot
script adds three plots: `m1_panoptic_vs_intent.{pdf,png}` (per-cat
bars), `sp_level_nu_recall_hist.{pdf,png}` (per-cat histogram), and
`overclaim_gap_hist.{pdf,png}` (the per-pair distribution split by
category — equivalent of `measure_overclaim.py` per category).

### Bug fix (2026-05-23): `pred_slice_id` vs `pred_query`

While building the over-claim metric an indexing bug surfaced in
`analyze_event.py`: predicted slices were being keyed by
`post/pred_slice_id`, which is the **GT primary_trackid** for matched
queries (e.g. 1172344) — NOT the query id (0..63). Looking up the
class via `q_class_argmax[1172344]` fell out-of-bounds and silently
defaulted to no_object, making `n_pred_nu_slices = 0` for events
where the model had clear nu predictions. Fixed in both
`analyze_event.py` (`_compute_pred_slices`, `_compute_overclaim`) and
`inspect_perevent.py` (the diagnostic's "won_sp_panoptic" count) to
key off `post/pred_query` (the actual query id).

### Pass-2 headline on the 3 test events (post-fix)

```
crosslevelrefiner_mixedq_maskdn  (3 events)
  category               n  nu_pred%  iou_pan iou_int sp_rec  surv%  rank1_nu
  all                    3   100.0%    0.666   0.627  1.000  100.0%    100.0%
  ccnumu                 2   100.0%    0.699   0.611  1.000  100.0%    100.0%
  pi0                    3   100.0%    0.666   0.627  1.000  100.0%    100.0%

ptv3hybrid_crosslevel_nonzeroinit_maskdn_noamp  (3 events)
  category               n  nu_pred%  iou_pan iou_int sp_rec  surv%  rank1_nu
  all                    3    66.7%    0.449   0.179  0.666  100.0%    100.0%
  ccnumu                 2   100.0%    0.674   0.269  1.000  100.0%    100.0%
  pi0                    3    66.7%    0.449   0.179  0.666  100.0%    100.0%
```

Notable on this tiny sample (don't draw conclusions yet):
- Crosslevel: 100% of events have a nu prediction; mean panoptic IoU
  ~0.67; nu prediction ranks #1 by chi-2 in the nu-only pool every time
  (above OOB=0).
- Entry-0 crosslevel: `m1_intent = 0.488 < m1_panoptic = 0.656` — the
  panoptic-winning nu slice (qid=21) has higher IoU than the Hungarian-
  matched-to-nu query. Hungarian picked a different query as the nu
  match. Matcher-instability signal.
- Entry-0 ptv3hybrid: `m1_intent = 0.000` but `m1_panoptic = 0.658` —
  Hungarian's nu match was NOT class-correct, yet a different query is
  class-correct and dominates panoptic. Same matcher-instability story.
- All non-degenerate events have `Δχ² ≈ −1.5` — the predicted nu slice's
  flash match is slightly better than the GT baseline (likely the GT
  mask carries some out-of-FV halo SPs that hurt its chi-2 marginally).
- Entry-1 ptv3hybrid: the only true "model failed" event in the sample
  — sp_recall=0, no matched-to-nu query. Use as a stress case while
  developing.

### Caveats / things to tune

- **γ defaults are 1.0** in `flash_predict.py`. Pass calibrated values
  via `--gamma-beam` / `--gamma-cosmic` for physically meaningful chi-2
  magnitudes. The χ² shape (and M4 ranking) is γ-invariant; only the
  absolute scale and the `--f-sys` choice care.
- **OOB threshold sweep**: default `[0.0, 0.05, 0.10, 0.20, 0.50]` with
  the headline at 0.20. Override via `--oob-thresholds` /
  `--default-oob-idx`.
- **Category overlap is intentional**: a CC νμ event with a π0 sits
  in BOTH `ccnumu` AND `pi0`. The aggregator preserves this by emitting
  per-category groups with overlapping membership.
- **`single_vis_gamma` rule**: exactly 1 visible nu-γ AND no primary π0.
  Tunable via `categorize.count_visible_nu_gammas(..., vis_gamma_E_thresh_MeV, fv)`.

### Pending for production scale-up

1. Run extended `prepare_flashinfo_h5.py` on the Tufts val+test set
   (uses the lantern container).
2. Run Stage 0 (`build_valtest_rootlists.py`) against the production
   val+test h5 list to get per-TAG manifests.
3. Run Stage 2 inference on val+test for both models.
4. Calibrate γ values before drawing absolute-chi-2 conclusions.
5. Submit Stages 3 → 4 → 5 as SLURM arrays driven off the Stage-0
   manifest CSVs.