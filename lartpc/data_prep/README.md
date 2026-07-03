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
