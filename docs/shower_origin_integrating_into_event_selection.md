# Integrating Inferfence Output of the Shower Origin Model into a MicroBooNE Photon Selection

For background information about the overall Shower Origin project, see the specfication file in `docs/shower_origin_spec.md.

The shower origin model when trained on reco clusters seems to perform decently well, with 80% classification accuracy and a median distance between the predicted origin and ground truth origin of ~1.5 cm.

The next step is to try to integrate the output of the shower origin model into a MicroBooNE photon selection.
To do this, we need to produce a ROOT TTree that contains the following information:
  - Run ID, Subrun ID, Event ID in order to match up the shower origin model output with the LANTERN reco ntuple
  - Reco cluster information for inside and outside showers. 
    - For each reco cluster, we need to store:
      - Cluster ID
      - Number of points in the cluster
      - Predicted Origin (x, y, z) in apparent (reco) coordinates
      - Ground truth origin (x, y, z, t) in true coordinates (pret0shiftedoriginpt)
      - Ground truth shower origin classification (0: inside, 1: outside, 2: on_track, 3: ghost, 4: true-track-only)
      - Geant Shower Track ID (mc_nu_origin_particle_id) for the fragment
      - Ground truth information about its 'trunk' status (is_trunk)
    - We also want to store info on true shower clusters we missed, in order to understand the efficiency of the shower origin model
      - For each true shower cluster, we need to store:
        - Cluster ID
        - Number of points in the cluster
        - Ground truth origin (x, y, z, t) in true coordinates (pret0shiftedoriginpt)
        - Ground truth shower origin classification (0: inside, 1: outside, 2: on_track, 3: ghost, 4: true-track-only)
        - Geant Shower Track ID (mc_nu_origin_particle_id) for the fragment
        - Ground truth information about its 'trunk' status (is_trunk)
      
It is not enough to just save the inference output. We need to do a bit of shower reconstruction to merge
fragments into one 'particle shower' so that we can classify events based on the number of showers and the number of photons. Shower fragment merging would proceed roughly as:
  1. Group showers by proximity of the predicted origin.
  2. Get the closest shower to the predicted origin and set it as the reco trunk cluster.
  3. Cluster fragments that are in the same direction as the trunk cluster and are within a certain distance of the trunk cluster. The direction is set using the first principle component axis of the trunk. We merge fragments that are in 
  a 20 degree cone made by the trunk cluster's first principle component axis, starting with the trunk cluster's start point, defined as the closest trunk point to the predicted origin.
     - Many showers might have an origin point that is not connected to any of the fragments and is not visible as an ionization deposit in the detector. We use the highest origin score point which will sit on the trunk fragment as the origin/start point of the shower. 

## The data processing workflow

We start with input files that contain the following information:
  - The input images
  - Truth information

The pipeline must then
  1. Run LArMatch in order to make reco spacepoints
  2. Run the shower origin truth labeler on the reco spacepoints to get the ground truth shower origin labels
  3. Run the shower truth and reco merger (the above steps are similar to the reco shower fragment pipeline for training data)
  4. Run the shower origin model on the reco spacepoints to get the predicted origin for each shower fragment
  5. Run the shower fragment merger to merge fragments into one 'particle shower'
  6. Save the output to a ROOT TTree


## Implementation (completed 2026-04-01)

The pipeline above has been implemented as Steps 5-7, extending the existing 4-step SLURM pipeline
(LArMatch → reco H5 → truth H5 → merge). Steps 5-7 run inside the pointcept container after Steps 1-4.

### New files

#### `lartpc_data_prep/shower_fragment_merger.py` — Core merging algorithm

Pure numpy module (no PyTorch dependency). Implements the fragment-to-shower merging algorithm:

1. **Group by predicted class**: Fragments are only merged within the same predicted class (inside, outside, on_track, ghost, true_track).
2. **Cluster origins by proximity**: Within each class, predicted origins are clustered using a fixed radius (default 5 cm) via `scipy.spatial.cKDTree`.
3. **Identify trunk**: For each origin cluster, the trunk fragment is the one whose closest point to the cluster's mean predicted origin is smallest.
4. **PCA shower axis**: First principal component of the trunk fragment's spacepoints, oriented to point away from the predicted origin.
5. **Start point**: Trunk point closest to the predicted origin. If the origin is disconnected (>30 cm from any trunk point), falls back to the highest origin_score point across all fragments in the cluster.
6. **Cone-based merging**: Non-trunk fragments are merged if >=50% of their points fall within a 20-degree half-angle cone along the shower axis, capped at 300 cm length.

Key data structures:
- `FragmentInfo` — per-fragment data after inference (predicted origin, class, scores, raw coords, truth labels)
- `MergedShower` — reconstructed shower (trunk index, fragment list, axis, start point)
- `MergerConfig` — tunable parameters: `origin_proximity_radius` (5 cm), `cone_half_angle` (20 deg), `cone_max_length` (300 cm), `min_points_fraction_in_cone` (0.5)

#### `tools/run_shower_origin_pipeline_step567.py` — Combined per-file pipeline script

Chains Steps 5-7 for each input merged H5 file:
- **Step 5**: Runs `run_event_inference()` (reused from `tools/run_shower_origin_inference.py`) to get per-fragment predictions.
- **Step 6**: Denormalizes model output coordinates back to detector cm, reads `pret0shiftedoriginpt` and `istrunk` directly from the H5 file (these are not returned by `get_all_fragments()`), builds `FragmentInfo` list, and calls `merge_fragments()`.
- **Step 7**: Writes results to a ROOT TTree via `uproot`. Computes missed true showers as truth fragment trackids not present in any reco fragment.

CLI arguments:
```
python tools/run_shower_origin_pipeline_step567.py \
    -c configs/lartpc/shower-origin-sonata-v1m1-v3-reco-fragments-p1cmp075.py \
    --checkpoint shower_origin/.../model_epoch165.pth \
    --input-h5 <merged_h5_file>        # single file
    --input-list <filelist.txt>        # or multiple files
    --output-dir <output_directory> \
    --device cuda \
    --origin-proximity-radius 5.0 \
    --cone-half-angle 20.0 \
    --cone-max-length 300.0 \
    --min-points-fraction-in-cone 0.5
```

ROOT TTree schema (`shower_reco` tree, one entry per event):

| Category | Branches |
|----------|----------|
| Event ID | `run`, `subrun`, `event` (int32) |
| Counts | `n_reco_fragments`, `n_merged_showers`, `n_missed_true` (int32) |
| Per reco fragment (var-length) | `frag_cluster_id`, `frag_n_points`, `frag_pred_origin_x/y/z`, `frag_gt_origin_x/y/z`, `frag_gt_origin_t`, `frag_gt_class`, `frag_pred_class`, `frag_geant_trackid`, `frag_is_trunk`, `frag_merged_shower_id` |
| Per merged shower (var-length) | `shower_id`, `shower_n_fragments`, `shower_n_total_points`, `shower_pred_origin_x/y/z`, `shower_start_x/y/z`, `shower_axis_x/y/z`, `shower_pred_class` |
| Per missed true shower (var-length) | `missed_trackid`, `missed_n_points`, `missed_gt_origin_x/y/z/t`, `missed_gt_class`, `missed_is_trunk` |

One ROOT file per input H5 file. Files can be merged downstream via `hadd`.

#### `lartpc_data_prep/shower_origin_reco_scripts/run_step567_pointcept.sh` — Container shell script

Runs inside the pointcept container (analogous to `run_step234_pointcept.sh`). Configuration via environment variables:
- `SHOWER_ORIGIN_CONFIG`, `SHOWER_ORIGIN_CKPT` — model config and checkpoint paths
- `DEVICE` — `cuda` or `cpu`
- `ORIGIN_PROXIMITY_RADIUS`, `CONE_HALF_ANGLE`, `CONE_MAX_LENGTH`, `MIN_POINTS_FRACTION_IN_CONE` — merger parameters

Separate from step234 so merger parameters can be iterated independently without re-running LArMatch or truth labeling.

### Modified files

#### `lartpc_data_prep/convert_larlite_to_showerorigin_h5.py`

Added (run, subrun, event) extraction from the larlite `storage_manager` after `io.go_to(ientry)` via `io.run_id()`, `io.subrun_id()`, `io.event_id()`. These are written as attributes on the `entry_0` group in each H5 file. Requires rerunning Steps 2-4 to populate existing H5 files with RSE attributes.

#### `lartpc_data_prep/merge_reco_truth_showerorigin.py`

Propagates `run`, `subrun`, `event` attributes from the reco H5 `entry_0` group into the merged output H5.

#### `lartpc_data_prep/shower_origin_reco_scripts/run_showerorigin_reco.sh`

Added:
- Configuration variables at top: `SHOWER_ORIGIN_CONFIG`, `SHOWER_ORIGIN_CKPT`, `DEVICE`, `ROOT_OUTPUT_DIR`
- Step 5-7 `apptainer exec` block after the existing Step 2-4 block
- ROOT file copy to `ROOT_OUTPUT_DIR/<subdir1>/<subdir2>/`
- Step 5-7 failure is non-fatal (H5 output from Steps 2-4 is still preserved)

### Coordinate frame handling

| Data source | Coordinate frame |
|-------------|-----------------|
| H5 raw `pos` / `coord` from `get_all_fragments()` | Detector cm |
| Model output `pred_coords` | Normalized: `(cm - [125, 0, 518]) / 179.55` |
| Fragment merger operations | All in detector cm |
| ROOT output | All in detector cm (except `pret0shiftedoriginpt` = true MC coords) |

The pipeline script denormalizes model outputs before passing to the merger.

### Data flow

```
merged_*.h5 (from Steps 1-4)
  |
  v  Step 5: run_event_inference() [reused from run_shower_origin_inference.py]
  Per-fragment: pred_scores(N,), pred_coords(N,3) [normalized], pred_class
  |
  v  Denormalize + read pret0shiftedoriginpt/istrunk from H5
  Build FragmentInfo list (all in detector cm)
  |
  v  Step 6: merge_fragments()
  Group by pred_class -> cluster origins (5cm) -> trunk ID -> PCA -> cone merge
  Output: list[MergedShower] + unmerged indices
  |
  v  Step 7: write_root_output() via uproot
  showerreco_<basename>.root -- one TTree entry per event
```

### Verification steps

1. Verify `uproot` is available in the pointcept container: `python -c "import uproot"`
2. Test on a single merged H5 file:
   ```bash
   python tools/run_shower_origin_pipeline_step567.py \
       -c configs/lartpc/shower-origin-sonata-v1m1-v3-reco-fragments-p1cmp075.py \
       --checkpoint shower_origin/sonata_v1m1_v3_v6backbone_pax_pi0filter_recofragments/model/model_epoch165.pth \
       --input-h5 <any_merged_h5_file> \
       --output-dir /tmp/test_showerreco \
       --device cpu
   ```
3. Open ROOT output and verify TTree structure and variable-length arrays
4. Rerun Steps 2-4 on a dev subset to get H5 files with RSE attributes
5. Submit small SLURM job: `sbatch --array=0-0 submit_showerorigin_reco.sh`
6. Verify `hadd` works on multiple ROOT output files

### Dev subset workflow

For iterating on model improvements or merger parameter optimization, use a small input list
with specific event types of interest (e.g., events with only 1 detectable photon, or events
with neutrino interactions outside the TPC). Run `run_step567_pointcept.sh` independently of
Steps 1-4 on pre-existing merged H5 files to avoid rerunning LArMatch.
