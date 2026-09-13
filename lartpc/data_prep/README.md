# Data Preparation

Two pipelines with **different scopes and different truth information**:

## training_data/ — training & validation data (generation 2, current)

Processes newer sim files generated with MicroBooNE code; richer truth for building
spacepoint ground-truth labels. Generation 2 incorporates **LANTERN truepoint scores**
for data augmentation and curriculum focus.

Steps (each a `*_wconfig.sh` driven by a per-sample config in `lantern_configs/`):
1. `run_step1_lantern_wconfig.sh` — LANTERN processing (lantern container; uses the
   local bug-fixed `inference_sparse_ssnet_uboone.py` / `recreate_ubspurn.py`)
2–4. `run_step234_pointcept_wconfig.sh` — `convert_larlite_to_pointcept_h5.py`,
   `process_dlmerged_to_hdf5_event_files.py`, `merge_reco_truth_showerorigin.py`
5. `run_step5_flashinfo_pointcept_wconfig.sh` — flash-info H5
   (`lartpc/flashmatch/prepare_flashinfo_h5.py`)

## uboone_official/ — official MicroBooNE sim/data

Processes official datasets (different truth handling than the training data).
Feeds the physics analyses (e.g. `lartpc/larformer_analysis/physics/single_photon`).
See `uboone_official/LARFORMER_DATAPREP.md`: Stage A converts `merged_dlreco.root`
to H5 (`convert_dlmerged_to_larformer_h5.py`), Stage B runs full-cascade inference
(`run_stepB_cascade_wconfig.sh`).

## labels/

Shared ground-truth label makers: `slice_labels.py` (particle/slice labels — imported
by scripts as `lartpc.data_prep.labels.slice_labels`), `keypoint_labels.py`,
`shower_fragment_merger.py`. Flash-info and photon-library tools moved to
`lartpc/flashmatch/`.

## validation/

Data QA: `validate_hdf5_files.py`, `audit_particle_labels.py`, `test_particle_labels.py`,
`dump_h5_keypoints.py`, `checkjobs.py`.

## archive/

- `gen1/` — generation-1 training prep (direct dlmerged→H5 corsika drivers, no
  LANTERN step). Superseded by `training_data/`. The python scripts here are older
  copies of the ones now living in `training_data/`.
- `shower_origin/` — shower-origin dataset production (exploratory project; see
  `docs/reference/shower_origin_spec.md`).

## Normalization

POT and number of spills of the different datasets we have in use. 
Used for normalization when making plots that combine simulation and EXTBNB to estimate the number of expected events
which can then be compared with data.

Beam data are events recorded in sync with the beam trigger.

EXTBNB data are events recorded using a fixed rate trigger during windows where no beam trigger occurs.
The events are further filtered such that an above threshold amount of light occured within a time window emulating a beam window.
The EXTBNB is intended to provide an estimate of the non-neutrino background we reconstruct.

Simulation should be scaled by POT to match the beam data. EXTBNB is scaled by the number of spills.

Info for 'uboone official' samples

* mcc9_v29e_dl_run1_C1_extbnb: Run 1 C1 EXTBNB. Full sample number of spills: 23090946
  Subsets: the exact spill count for a subset of files is hard to recompute, so a
  subset is normalized by its file fraction x the full-sample spills (user decision
  2026-09-12). The stride-2 list (`mcc9_v29e_dl_run1_C1_extbnb_stride2.txt`, 13,801 of
  27,602 files) is every other file; the run-1 EXT half-stride-2 production (tranches
  A+B, filenos 1-6900 of that list) is 6,900 / 27,602 = 25% of the full sample ->
  5,772,737 spills. Tranche A alone (filenos 1-1000, `extbnb_run1_A`) is 3.62% -> 836,629 spills.
* mcc9_v28_wctagger_bnb5e19: Run 1 open beam data. POT: 4.4e19; number of spills: 94414115
* mcc9_v28_run1_bnboverlay: Run 1 BNB nu overlay MC, 9,538 tier2 files. The run-1
  overlay HALF production (`run1_bnboverlay_half/`, 2026-09-13) converts the
  TRAINPOOL list (4,740 files = unbiased 50% by md5 parity, see
  training_data_ledger/LEDGER.md) in MC mode with truth sidecars
  (`uboone_official/tranche_ovl_run1_half.spec` via `sbatch_tier2_tranche.sh`);
  POT comes from the sidecars' potTree (no spill count). The 12.8k-event pilot in
  overlay_train/ has no sidecars and is calibration-only. Chain launcher:
  `larformer_reco/slurm/launch_run1ovl_half_chain.sh` (MC knobs baked in).