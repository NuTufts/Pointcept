# Shower Origin Reco Training Data Production

Produces shower origin training data with truth labels from reco fragments.
Processes ~8,313 input ROOT files (from MicroBooNE simulation) through a 4-step pipeline.

Input file list: `Pointcept/lartpc_data_prep/inputlists/bnb_nu_pi0filter_corsika.txt`

## Pipeline Overview

For each input ROOT file:

1. **Step 1 (lantern container):** Run SSNet inference and LArMatch deploy.
   - Produces `larmatchme_larlite.root` (1A) and `merged_dlreco_with_ssnet.root` (1B).
2. **Step 2 (pointcept container):** Run `convert_larlite_to_showerorigin_h5.py` on (1A) and (1B).
   - Produces per-event `showerorigin_*_entry*.h5` reco fragment files (2A).
   - Contains spacepoints passing the larmatch ghost filter and reco shower fragments from 2D SSNet.
3. **Step 3 (pointcept container):** Run `process_dlmerged_to_hdf5_event_files.py` on (1B).
   - Produces per-event `pointceptdata_*_entry*.h5` truth fragment files (2B).
   - Contains all spacepoints with truth labels and truth shower fragments with ground truth origin targets.
4. **Step 4 (pointcept container):** Run `merge_reco_truth_showerorigin.py` on each (2A)+(2B) pair.
   - Produces per-event `merged_showerorigin_entry*.h5` final output files (3).
   - These are (2A) files updated with ground truth labels transferred from matching (2B) truth fragments.

Only the final merged H5 files (3) are saved to the output directory.
All intermediate ROOT and H5 files are cleaned up per input file.

## Scripts

All scripts live in `Pointcept/lartpc/larformer_analysis/archive/shower_origin_reco_scripts/`.

### `submit_showerorigin_reco.sh` — SLURM submission script

Submit with: `sbatch submit_showerorigin_reco.sh`

- `#SBATCH --array=0-831` (8,313 files / stride of 10 = 832 jobs)
- 8 GB memory, 1 CPU, 3-day wall time, batch partition
- Loads the apptainer module, then calls `run_showerorigin_reco.sh` on the bare node

For testing, submit a single job: `sbatch --array=0-0 submit_showerorigin_reco.sh`

### `run_showerorigin_reco.sh` — Main per-job run script

Runs on the bare node (NOT inside a container). Processes `stride` input files
determined by `SLURM_ARRAY_TASK_ID * stride + OFFSET`. For each input file:

1. Creates a per-file working directory in `/tmp/shower_origin_reco_<TAG>_jobidNNNN_lineNNNNN/`
2. Copies the input ROOT file to the workdir
3. Calls `apptainer exec` into the **lantern container** to run `run_step1_lantern.sh`
4. Calls `apptainer exec` into the **pointcept container** to run `run_step234_pointcept.sh`
5. Copies final `merged_*.h5` files to `OUTPUT_DIR/<fileno/1000>/<fileno/100>/`
6. Cleans up the workdir

Configuration variables are at the top of the script:
```bash
WORKDIR=...              # Path to shower_origin_reco_scripts/
UBDL_DIR=...             # Path to ubdl
POINTCEPT_DIR=...        # Path to Pointcept
INPUTLIST=...            # inputlists/bnb_nu_pi0filter_corsika.txt
OUTPUT_DIR=...           # Final output for merged H5 files
LANTERN_CONTAINER=/cvmfs/uboone.opensciencegrid.org/containers/lantern_v2_me_06_03_prod
POINTCEPT_CONTAINER=/cluster/tufts/wongjiradlabnu/larbys/larbys-container/pointcept_cuml.sif
TAG=showerorigin_reco
stride=10                # files per array job
OFFSET=0
```

### `run_step1_lantern.sh` — Step 1 inner script (runs inside lantern container)

Takes `<workdir_path>` as argument. Sets up the lantern environment, then:

1. Copies **local bug-fixed** versions of `inference_sparse_ssnet_uboone.py` and `recreate_ubspurn.py`
   from `ubdl/lantern_scripts/` into the workdir. These local copies use the `sparseuresnetout`
   producer name instead of `sparsessnet`, fixing a tree-naming mismatch bug in the container's
   bundled versions.
2. Runs SSNet inference using the local script
3. Merges SSNet output into the input ROOT file (using `rootcp` with the corrected
   `sparseimg_sparseuresnetout_tree` tree name)
4. Runs the `recreate_ubspurn.py` local script to recreate UB sparse images
5. Runs LArMatch deploy (`deploy_larmatchme.py` from the container)
6. Cleans up intermediate files, keeping `larmatchme_larlite.root` and `merged_dlreco_with_ssnet.root`

Note: This does NOT run the full lantern workflow (`run_lantern_workflow_mc.sh`).
It runs only the SSNet + LArMatch subset needed for this pipeline, since the full
workflow would delete the intermediate files we need for Steps 2-4.

### `run_step234_pointcept.sh` — Steps 2-4 inner script (runs inside pointcept container)

Takes `<workdir_path>` as argument. Sources the pointcept environment via
`ubdl/setenv_pointcept_container.sh`, then runs Steps 2-4 sequentially:

- **Step 2:** `convert_larlite_to_showerorigin_h5.py -i larmatchme_larlite.root --input-larcv merged_dlreco_with_ssnet.root -o reco_h5/`
- **Step 3:** `process_dlmerged_to_hdf5_event_files.py -i merged_dlreco_with_ssnet.root`
- **Step 4:** For each event, matches reco and truth H5 files by entry number and runs
  `merge_reco_truth_showerorigin.py --reco-h5 <reco> --truth-h5 <truth> --output merged_showerorigin_entry*.h5`

After merging, cleans up all intermediate files (reco H5, truth H5, ROOT files).

### `check_status.sh` — Job completion checker

Usage: `bash check_status.sh [--rerun rerun_list.txt]`

Iterates through the input list and checks if final merged H5 files exist in the output directory.
Reports total files, completed, and missing/failed counts.
With `--rerun`, writes a file of line numbers for failed/missing jobs that can be used for resubmission.

## File Flow Per Input File

All intermediate files live in `/tmp/` workdir and are cleaned up after processing:

```
input: dlmerged_*.root (copied from cluster storage)
  -> Step 1 (lantern) -> larmatchme_larlite.root + merged_dlreco_with_ssnet.root
  -> Step 2 (pointcept) -> showerorigin_*_entry*.h5 (reco fragments)
  -> Step 3 (pointcept) -> pointceptdata_*_entry*.h5 (truth fragments)
  -> Step 4 (pointcept) -> merged_showerorigin_entry*.h5 (final output)
  -> copy merged_*.h5 to OUTPUT_DIR/<subdir1>/<subdir2>/
  -> rm all intermediate files
```

## Critical Source Files

- `ubdl/lantern_scripts/inference_sparse_ssnet_uboone.py` — Local bug-fixed SSNet inference (Step 1)
- `ubdl/lantern_scripts/recreate_ubspurn.py` — Local bug-fixed UB sparse image recreation (Step 1)
- `ubdl/lantern_scripts/setup_lantern_container.sh` — Lantern env setup reference
- `ubdl/setenv_pointcept_container.sh` — Pointcept container env setup (Steps 2-4)
- `Pointcept/lartpc/data_prep/archive/shower_origin/convert_larlite_to_showerorigin_h5.py` — Step 2
- `Pointcept/lartpc/data_prep/archive/gen1/process_dlmerged_to_hdf5_event_files.py` — Step 3
- `Pointcept/lartpc/data_prep/archive/gen1/merge_reco_truth_showerorigin.py` — Step 4

## Prior Art

The existing truth-only shower fragment production scripts served as the pattern for this pipeline:
- `Pointcept/lartpc/data_prep/archive/gen1/run_corsika_bnb_nu_pi0filter.sh` (example run script)
- `Pointcept/lartpc/data_prep/archive/gen1/submit_bnbnu_pi0filter_corsika.sh` (example SLURM submission)

## Verification

1. Submit test with `sbatch --array=0-0 submit_showerorigin_reco.sh` (processes first 10 files)
2. Check logs in `workdir/` for errors
3. Verify final merged H5 files exist in `OUTPUT_DIR`
4. Run `bash check_status.sh` to confirm
5. Spot-check a merged H5 with `Pointcept/lartpc_data_prep/test_reco_h5_with_dataloader.py`
