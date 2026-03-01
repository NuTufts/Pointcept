# Shower Origin Prediction

Predicting the 3D origin point and inside/outside-TPC classification of electromagnetic (photon) showers in MicroBooNE LArTPC data.

## Motivation

Photon showers in LArTPC detectors originate from a point that may be displaced from the visible ionization. A photon travels invisibly from its production vertex (e.g., a pi0 decay) to its conversion point where it pair-produces and creates detectable charge. Reconstructing this origin is critical for associating showers with their parent interaction vertices.

Key challenges:

1. **Context-dependent ground truth** -- the "correct" origin depends on how many sibling showers are visible and whether the origin is inside or outside the detector.
2. **Empty-space prediction** -- the PointTransformer backbone only produces features at ionization locations, but origins may be in empty space.
3. **Multi-shower events** -- the model must predict origins for all showers simultaneously.

## Ground Truth Logic

Ground truth is produced by the C++ `ShowerFragmentOriginMaker`, which clusters each shower's spacepoints with DBSCAN and assigns per-fragment origin labels. For each shower fragment, the labeler determines an origin point, start point, and classification.

### Origin Type Classification

The inside/outside check uses `pret0shiftedoriginpt` (raw Geant4 truth position without t0 shift or SCE correction), NOT the apparent position which is always inside the TPC by construction.

| Type | Condition | Description |
|------|-----------|-------------|
| `0` (INSIDE) | Origin inside TPC + neutrino-origin (MCParticleGraph origin==1) | Neutrino-induced, inside detector |
| `1` (OUTSIDE) | Origin outside TPC (regardless of origin flag) | Particle created outside detector |
| `2` (ON_TRACK) | Origin inside TPC + cosmic-origin (MCParticleGraph origin==2) | Cosmic-induced, inside detector |

TPC active volume bounds: X: [0, 255.6], Y: [-116.5, 116.5], Z: [0.5, 1035.5] cm.

### Origin Point Assignment

| Condition | Origin Point | Notes |
|---|---|---|
| Inside TPC (type 0 or 2), origin within image bounds, keypoint matched, vertex inferable | True physics vertex (SCE-corrected apparent position) | e.g., pi0 decay point with both photons visible |
| Inside TPC (type 0), pi0 vertex but sibling photon not visible | Trunk fragment start point | Pi0 vertex cannot be inferred from single photon (data loader override) |
| Outside TPC (type 1) | Trunk fragment start point | True origin unreachable by model |
| Origin tick outside image bounds (tick < 2410 or > 8438) | Trunk fragment start point | Origin not visible in detector readout |
| No keypoint matched for this particle | Trunk fragment start point | Particle not tracked by MCKeypointMaker |

The raw Geant4 truth position is always preserved in `pret0shiftedoriginpt` (4D: x,y,z,t).

### Definitions

- **Fragment**: A DBSCAN cluster of spacepoints belonging to a single MC particle (electron, positron, or photon shower).
- **Trunk fragment**: The fragment whose start point is closest to the shower's first energy deposition (preferring fragments with >20 points).
- **Start point**: Most upstream point in the fragment along the shower axis direction.
- **Inside TPC**: True origin (from `pret0shiftedoriginpt`) within MicroBooNE active volume.
- **Conversion point**: Where the photon first creates detectable ionization (`first_edep_pos` from MCParticleGraph).

## Architecture

### Model: ShowerOriginPredictorV3

Built on top of a frozen Sonata-pretrained PointTransformer V3 backbone (`PT-v3m2`, v6 config with `enc_channels=(48, 96, 192, 384, 512)`, `up_cast_level=2`, `backbone_out_channels=1088`).

```
Input point cloud (coord + strength = 6 channels)
    |
    v
[Frozen PT-v3m2 Backbone] --> per-point features (N x 1088)
    |
    v
[Virtual Grid Generator] --> augment with empty-space points (optional, disabled by default)
    |                         - Dense grid: 2cm spacing within 30cm of each shower
    |                         - Sparse grid: 10cm spacing over full TPC
    |                         - Features via kNN interpolation + learned position encoding
    v
[Slot Attention] (K=1 slot currently) --> object-centric representations
    |
    v
[Per-Slot Cross-Attention + Score Head] --> origin heatmap (Gaussian score) over all points
[Per-Slot Classification Head]          --> inside/outside/on_track logits (3 classes)
    |
    v
Loss = Gaussian score BCE + classification cross-entropy (+ optional distance L1)
```

**Current training mode (K=1):** Single-shower prediction per crop. Each training sample is a BiasedSphereCrop around one fragment centroid. The model predicts a Gaussian origin heatmap over all input points (not just shower points) and a 3-class origin type.

### Key Components

- **VirtualGridGenerator** (`virtual_grid.py`): Generates virtual grid points in empty space and computes features via hybrid kNN interpolation + learned 3D positional encoding. Currently disabled by default (OOM on <=16GB GPUs with ~91K virtual points × 1088 channels).
- **SlotAttention** (`slot_attention.py`): Competitive attention mechanism producing K object-centric slot representations. Currently K=1 for single-shower-per-crop training.
- **OriginClassificationHead**: 3-class MLP classifier per slot (inside/outside/on_track).
- **Hungarian Matching**: Optimal bipartite assignment of K slots to J ground-truth showers during training (scipy `linear_sum_assignment`). Not used when K=1.
- **ShowerOriginEvaluator** (`engines/hooks/shower_origin_evaluator.py`): Validation hook computing origin recall, specificity, peak distance, and classification accuracy.

### Previous Versions

- **V1** (`ShowerOriginPredictor`): Single shower per event, direct score prediction.
- **V2** (`ShowerOriginPredictorV2`): Added slot attention + cross-attention, but single combined origin heatmap.
- **V3** (`ShowerOriginPredictorV3`): Multi-shower with virtual grid, per-slot predictions, Hungarian matching, inside/outside classification.

## Data Pipeline

### Step 1: ROOT to HDF5 Conversion

Convert `dlmerged` ROOT files to per-event HDF5 using `SimChTripletLabelMaker`:

```bash
cd lartpc_data_prep
python process_dlmerged_to_hdf5_event_files.py \
    --input /path/to/dlmerged_file.root \
    --output-dir ./output/
```

The C++ `SimChTripletLabelMaker` (modified in `ubdl/larflow`) now exports an `mc_particle_tree` group alongside the standard `triplet_data` and `mckeypoints`.

**Note:** Shower fragment labels (`shower_fragments/` group) are now produced directly by the C++ `ShowerFragmentOriginMaker` during Step 1 — no separate post-processing step is needed.

### Step 3: Training

```bash
# Single GPU
python tools/train.py \
    --config-file configs/lartpc/shower-origin-sonata-v1m1-v3.py \
    --num-gpus 1 \
    --options save_path=exp/shower_origin/v3

# Multi-GPU via SLURM
sbatch scripts/slurm/train_shower_origin.sh
```

### Visualization

Inspect labeled events interactively with the Dash/Plotly 3D viewer:

```bash
cd lartpc_data_prep

# Show only shower fragments + origin markers
python vis_shower_fragments.py -i event_file.h5 -e 0

# With non-shower context points
python vis_shower_fragments.py -i event_file.h5 -e 0 --show-non-shower

# Color by origin type (green=INSIDE, red=OUTSIDE)
python vis_shower_fragments.py -i event_file.h5 -e 0 -c origin --show-non-shower

# With MC keypoints overlaid
python vis_shower_fragments.py -i event_file.h5 -e 0 --show-keypoints

# Include ghost points
python vis_shower_fragments.py -i event_file.h5 -e 0 --show-non-shower --include-ghosts
```

The viewer displays:
- Shower spacepoints colored by fragment index, origin type, or trackid
- Origin markers: green spheres (INSIDE), red spheres (OUTSIDE), labeled O0, O1, ...
- Conversion point markers: cyan diamonds (when different from origin), labeled C0, C1, ...
- Dashed lines from origin to conversion point (photon flight path)
- Summary table with per-fragment metadata

## HDF5 Data Schema

### mc_particle_tree (from SimChTripletLabelMaker)

```
entry_0/mc_particle_tree/
    trackid:              (M,) int     -- unique MC track IDs
    pid:                  (M,) int     -- PDG particle code
    parent_trackid:       (M,) int     -- parent's trackid (-1 for primaries)
    origin:               (M,) int     -- 1=neutrino, 2=cosmic
    start_pos:            (M, 3) float -- true start position [cm]
    start_pos_sce:        (M, 3) float -- SCE-corrected start position [cm]
    energy_mev:           (M,) float   -- kinetic energy [MeV]
    process_code:         (M,) int     -- creation process (0=primary, 1=Decay, 2=compt, 3=conv, ...)
    num_daughters:        (M,) int     -- number of daughters per particle
    daughter_start_indices: (M,) int   -- index into flattened daughter list
    daughter_trackids:    (D,) int     -- flattened daughter trackid list
    nu_vertices:          (V, 3) float -- neutrino interaction vertices [cm]
```

### shower_fragments (from C++ ShowerFragmentOriginMaker)

```
entry_0/shower_fragments/
    @num_fragments: int              -- attribute: total number of DBSCAN fragments
    trackid:              (F,) int   -- Geant4 track ID per fragment
    pid:                  (F,) int   -- PDG code (22, 11, -11)
    istrunk:              (F,) int   -- 1=trunk, 2=secondary fragment
    type:                 (F,) int   -- 0=nu-inside, 1=outside, 2=cosmic-inside
    startpt:              (F, 3) float -- most upstream point per fragment [cm]
    originpt:             (F, 3) float -- prediction target origin [cm] (apparent; for outside/unreachable, equals trunk startpt)
    pret0shiftedoriginpt: (F, 4) float -- raw Geant4 truth origin (x,y,z,t) [cm, ns]
    pointindices_flat:    (T,) long  -- concatenated point indices into triplet_data
    pointindices_counts:  (F,) int   -- number of points per fragment
```

Where F = number of fragments, T = total points across all fragments.

### Dataset Output Keys (ShowerOriginDataset)

The dataset returns a dict with these keys for the model:

| Key | Shape | Description |
|---|---|---|
| `coord` | (N, 3) | Point coordinates |
| `feat` | (N, C) | Point features (from Collect transform) |
| `shower_masks` | (max_showers, N) | Per-shower boolean masks, padded |
| `origin_coords` | (max_showers, 3) | Ground truth origin coordinates |
| `origin_types` | (max_showers,) | 0=INSIDE, 1=OUTSIDE, 2=ON_TRACK |
| `valid_showers_mask` | (max_showers,) | Which shower slots are real vs padding |
| `num_showers` | int | Actual number of showers in event |

## Input data sets: ROOT files containing different type of simulated data

List with ROOT files are in `/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/lartpc_data_prep/inputlists/`.

| Sample Name | file list | Num files | Description |
|---|---|---|---|
| bnb_nu_corsika           | bnb_nu_corsika_prod2.txt     | 5000 | BNB neutrino flux (all flavors). Nu interactions created within Liquid Argon volume. Includes cosmic simulation. |
| bnb_nu_pi0_corsika       | bnb_nu_pi0_corsika_prod2.txt | 4995 | Same as bnb_nu_corsika because intended pi0 filter did not work properly. But sample name kept. |
| bnb_nu_pi0filter_corsika | bnb_nu_pi0filter_corsika.txt | 8313 | BNB flux with all flavors. Only nu interactions in the LAr volume where at least 1 pi0 created. Includes cosmics. |
| bnb_nue_corsika          | bnb_nue_corsika_prod2.txt    | 4999 | BNB flux but only nue flavor. Nu interactions created inside TPC. Includes cosmics. |
| bnb_nu_chargedpiplus_corsika | bnb_nu_chargedpiplus_corsika_prod2.txt | 8050 | BNB flux with all flavors. Only nu interactions in the LAr volume where at least 1 charged pion created. Includes cosmics. |

### Production scripts

Located on the Tufts cluster at `/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/lartpc_data_prep/`.

Each data set has a slurm submission script. It calls a bash script which runs `process_dlmerged_to_hdf5_event_files.py` on N files from the input lists. 
We refer to N as the 'stride' in the bash scripts.

Need to make sure that the right output folder is being used for the created hdf5 files: 

Our new files with the shower fragment data will be stored in the folder: `/cluster/tufts/wongjiradlab/larbys/data/ub_on_tufts/hdf5/v3_showerfragments/[sample name]`.

| Sample Name | Number of jobs required | stride | slurm script | bash script |
|---|---|---|---|---|
| bnb_nu_corsika | 100 | 50 | submit_bnbnu_corsika.sh | run_corsika_bnb_nu.sh | 
| bnb_nu_pi0_corsika | 100 | 50 | submit_bnbnu_pi0_corsika.sh | run_corsika_bnb_nu_pi0.sh |
| bnb_nu_pi0filter_corsika | 83 | 100 | submit_bnbnu_pi0filter_corsika.sh | run_corsika_bnb_nu_pi0filter.sh | 
| bnb_nue_corsika | 100 | 50 | submit_bnbnue_corsika.sh | run_corsika_bnb_nue.sh |
| bnb_nu_chargedpiplus_corsika | 81 | 100 | submit_bnbnu_chargedpiplus_corsika.sh | run_corsika_bnb_nu_chargedpiplus.sh | 


## File Inventory

### Workstream 1: C++ Ground Truth (ubdl/larflow)

| File | Description |
|---|---|
| `ubdl/larflow/.../ShowerFragmentOriginMaker.h` | Header: fragment data struct forward decl, method declarations |
| `ubdl/larflow/.../ShowerFragmentOriginMaker.cxx` | DBSCAN clustering, trunk/type/origin logic, HDF5 export |
| `ubdl/larflow/.../ShowerFragmentOrigin.h` | Data struct for fragment arrays (trackid, pid, istrunk, type, startpt, originpt, pret0shiftedstart, pointindices) |
| `ubdl/larflow/.../SimChTripletLabelMaker.h` | Added `save_mc_particle_tree()` declaration |
| `ubdl/larflow/.../SimChTripletLabelMaker.cxx` | Calls `_shower_fragment_maker.save_entry_to_hdf()` + `save_mc_particle_tree()` |

### Workstream 2: V3 Model Architecture (Pointcept)

| File | Description |
|---|---|
| `pointcept/models/shower_origin/shower_origin_model.py` | V1, V2, and V3 model definitions |
| `pointcept/models/shower_origin/virtual_grid.py` | `VirtualGridGenerator`, `PositionalEncoding3D` |
| `pointcept/models/shower_origin/slot_attention.py` | `SlotAttention` module |
| `pointcept/models/shower_origin/__init__.py` | Exports V3, classification head, virtual grid |
| `configs/lartpc/shower-origin-sonata-v1m1-v3.py` | V3 training config |

### Workstream 3: Data Pipeline (Pointcept)

| File | Description |
|---|---|
| `pointcept/datasets/shower_origin.py` | `ShowerOriginDataset` with multi-shower support, dual format loading, min_fragment_points |
| `scripts/slurm/prep_shower_origin_data.sh` | SLURM array job for data prep |
| `scripts/slurm/train_shower_origin.sh` | SLURM training job |

### Training Infrastructure (Pointcept)

| File | Description |
|---|---|
| `pointcept/engines/hooks/shower_origin_evaluator.py` | `ShowerOriginEvaluator` hook — validation metrics (recall, specificity, peak distance, cls accuracy) |
| `pointcept/datasets/transform.py` | `RandomScale` and `RandomFlipAxis` extended with `extra_coord_keys` for consistent augmentation of origin/start coords |

### Visualization & Documentation (Pointcept)

| File | Description |
|---|---|
| `tools/visualize_shower_origin.py` | Dash 3-panel viewer: shower mask, origin_distance (from data loader), Gaussian target (as computed by loss). Loads data through full transform pipeline. |
| `lartpc_data_prep/vis_shower_fragments.py` | Interactive Dash/Plotly 3D viewer for raw HDF5 data (dual format, pret0 hover text, --min-frag-pts) |
| `docs/LArTPC_HDF5_Data_Format.md` | HDF5 schema documentation (shower_fragments, mc_particle_tree) |
| `docs/shower_fragment_origin_spec.md` | Shower fragment origin implementation specification |
| `docs/Shower_Origin_Prediction.md` | This document — overall project spec and status |

## Completion Status

### Done

- [x] **Phase 1a**: All three workstreams implemented in parallel
  - [x] C++ `save_mc_particle_tree()` added to `SimChTripletLabelMaker` and compiled
  - [x] C++ `ShowerFragmentOriginMaker` — DBSCAN clustering, per-fragment origin/start/type, HDF5 export
  - [x] `pret0shiftedoriginpt` field added — true Geant4 origin preserved for diagnostics
  - [x] Origin reassignment for outside-TPC and unreachable origins (moved to trunk startpt)
  - [x] `ShowerOriginPredictorV3` with virtual grid, Hungarian matching, classification head
  - [x] `ShowerOriginDataset` rewritten for multi-shower with padded batching (supports new flat format + legacy)
  - [x] `vis_shower_fragments.py` interactive 3D viewer with dual format support, pret0 hover text
  - [x] SLURM scripts for data prep and training
  - [x] HDF5 data format documentation (`LArTPC_HDF5_Data_Format.md`)
  - [x] Shower fragment origin specification (`shower_fragment_origin_spec.md`)
- [x] **Phase 1a-data**: ROOT files reprocessed with C++ ShowerFragmentOriginMaker
  - [x] 10 events visually inspected and validated



### In Progress

- [x] **Phase 1b**: Validate training data labels (redo with new C++ fragments)
  - [x] Verified origin labels for type-0 showers where pi0 vertex is unreachable
  - [x] Added Python data loader override: origin → trunk startpt when sibling photon has no visible fragments
  - [x] Validation on 8 events (17 files total, 9 without new format): 572 fragments (60 type-0, 81 type-1, 431 type-2), 6 pi0 pairs (4 both-visible, 2 single-visible), 2 single-photon cases correctly overridden
  - [x] Validation script: `lartpc_data_prep/validate_shower_origins.py`

- [ ] **Phase 3**: Data production
  - [x] Submit test job and look at file size increase. Not very large.
  - [x] Submit for bnb_nu_pi0filter_corsika sample ( `lartpc_data_prep/submit_bnbnu_pi0filter_corsika.sh` )
  - [ ] Submit for bnb_nu_corsika sample ( `lartpc_data_prep/submit_bnbnu_corsika.sh` )
  - [ ] Submit for bnb_nu_pi0_corsika sample ( `lartpc_data_prep/submit_bnbnu_pi0_corsika.sh` )
  - [ ] Submit for bnb_nue_corsika sample ( `lartpc_data_prep/submit_bnbnue_corsika.sh` )
  - [ ] Submit for bnb_nu_chargedpiplus_corsika ( `lartpc_data_prep/submit_bnbnu_chargedpiplus_corsika.sh` )
### Done

- [x] **Phase 2**: Integration testing
  - [x] End-to-end test: dataset -> model forward pass (batch support added, tested with batch_size=16)
  - [x] ShowerOriginEvaluator hook with validation metrics (recall, specificity, peak distance, classification accuracy)
  - [x] Data augmentation fix: `RandomScale` and `RandomFlipAxis` now transform `origin_coord`/`start_coord` consistently with `coord` via `extra_coord_keys` (was root cause of model not learning)
  - [x] BiasedSphereCrop `prob_random` set to 0.0 to prevent crops that miss the target fragment
  - [x] Overfitting test on single event (43 fragments, 10 epochs) — model learns successfully



#### Single-Event Overfit Results (43 fragments, 10 epochs, batch_size=16)

| Metric | Value |
|---|---|
| Train loss (final epoch avg) | 0.0495 |
| Val loss | 0.0486 |
| Origin Recall | 0.8925 |
| Non-origin Specificity | 0.9672 |
| Mean Peak Distance (normalized) | 0.0146 (~2.6 cm) |
| Median Peak Distance (normalized) | 0.0097 (~1.7 cm) |
| Classification Accuracy | 1.0000 |
| cls_inside_acc | 1.0000 |
| cls_outside_acc | 1.0000 |
| cls_on_track_acc | 1.0000 |

Peak distance in cm: value × coord_scale (179.55).

### Remaining

- [ ] **Phase 3**: Production
  - [ ] Full-scale data prep via SLURM (all available ROOT files -> HDF5 -> shower_fragments)
  - [ ] Multi-GPU training run on full dataset
  - [ ] Consider if some kind of validation check is needed for the new shower fragment data. For past version of data, used `Pointcept/lartpc_data_prep/validate_hdf5_files.py`.
- [ ] **Phase 4**: Evaluation
  - [ ] Analyze prediction quality on held-out data (origin localization error, inside/outside accuracy)
  - [ ] Ablation: with/without virtual grid
  - [ ] Hyperparameter tuning (grid spacing, min_visible_points threshold, num_slots for multi-shower)

## Known Issues and Fixes

### Cosmic Photon Labeling (Fixed)

The initial Python labeler only found photons via `mckeypoints` with `kptype==3` (shower start). Since `MCKeypointMaker` only creates keypoints for neutrino-origin particles, all cosmic photons were missed. This is now handled by the C++ `ShowerFragmentOriginMaker` which scans `MCParticleGraph` directly for all shower particles (electrons, positrons, photons).

### Pi0 Single-Photon Origin (Fixed in Python Data Loader)

When a pi0 decays inside the TPC into two photons but only one produces visible fragments (the other escapes or has too few points), the C++ code assigns the pi0 decay vertex as the origin. However, with only one shower direction, the model cannot geometrically infer the pi0 vertex. The Python data loader (`shower_origin.py`) now overrides the origin to the trunk startpt for these cases by checking `mc_particle_tree` for visible sibling photons from the same parent pi0. Validated on 8 events: found 2 single-photon pi0 cases, both correctly overridden.

### Data Augmentation Inconsistency (Fixed)

`RandomScale` and `RandomFlipAxis` transforms only modified `coord`, leaving `origin_coord` and `start_coord` in the original frame. After augmentation, the Gaussian loss target (computed from `origin_coord` and `coord`) was garbage — the model could not learn. Fixed by adding `extra_coord_keys=("origin_coord", "start_coord")` to both transforms in the config, and extending the transform implementations to support this parameter.

### BiasedSphereCrop Missing Fragment (Fixed)

With `prob_random=0.25`, 25% of crops used a random point as center, often far from the target shower fragment. Combined with `point_max=5120`, dense non-shower regions could fill the point budget before reaching the fragment. Fixed by setting `prob_random=0.0`.

### Fragment Sampling Bias (Fixed)

A cap on max fragments per event caused the same 4 fragments to be selected repeatedly. Fixed to sample from all available fragments.

### Increase in file size when including shower fragments and mc particle tree

Looked at the first few files made with new shower fragments and mc particle tree inside, comparing with the last version.

```
[twongj01@p1cmp028 mlreco_dlmerged2hdf5_coriska_bnb_nu_pi0filter_jobid000]$ ls -lh /cluster/tufts/wongjiradlab/larbys/data/ub_on_tufts/hdf5/v2_expandedclasses/bnb_nu_pi0filter_corsika/000/000/ | head -n 5
-rw-rw---- 1 twongj01 wongjiradlab 9.6M Jan 13 10:48 pointceptdata_dlmerged_coriska_bnb_nu_pi0filter_fileno000001_entry000000.h5
-rw-rw---- 1 twongj01 wongjiradlab  12M Jan 13 10:48 pointceptdata_dlmerged_coriska_bnb_nu_pi0filter_fileno000001_entry000001.h5
-rw-rw---- 1 twongj01 wongjiradlab 4.0M Jan 13 10:48 pointceptdata_dlmerged_coriska_bnb_nu_pi0filter_fileno000001_entry000002.h5
[twongj01@p1cmp028 mlreco_dlmerged2hdf5_coriska_bnb_nu_pi0filter_jobid000]$ ls -lh /cluster/tufts/wongjiradlab/larbys/data/ub_on_tufts/hdf5/v3_showerfragments/bnb_nu_pi0filter_corsika/000/000/ | head -n 5
-rw-rw---- 1 twongj01 wongjiradlab 9.8M Mar  1 09:14 pointceptdata_dlmerged_coriska_bnb_nu_pi0filter_fileno000001_entry000000.h5
-rw-rw---- 1 twongj01 wongjiradlab  12M Mar  1 09:14 pointceptdata_dlmerged_coriska_bnb_nu_pi0filter_fileno000001_entry000001.h5
-rw-rw---- 1 twongj01 wongjiradlab 4.1M Mar  1 09:14 pointceptdata_dlmerged_coriska_bnb_nu_pi0filter_fileno000001_entry000002.h5
```

The increase in size is tolerable. Can go ahead with making new files. When done v2 can be removed.

## Open Questions

1. **No-object loss weight**: DETR-style penalty for unmatched slots. Current weight may need adjustment to balance false positive suppression vs recall.
2. **Minimum fragment points**: Currently 20 in data loader. May need tuning.
