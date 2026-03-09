# Making a MicroBooNE Data Pipeline to test the Shower Origin Model

We want to test the effectiveness of the shower origin model in reconstructing showers
and aiding in identifying neutrino interactions with a signel photon. To do this we must
apply the shower origin model within a reconstruction pipeline for data coming
from the MicroBooNE detector. We describe in this document, the steps needed to do this,
and provide a detailed plan for implementation.

## Overview of Steps to integrate into LANTERN reco

  1. Make spacepoints from wire-plane image data, remove ghost points, and apply SSNet labels.
     This takes in ROOT data files and produces another ROOT data file.
     Both the input and output files will contain multiple events -- with the same number of events in
     each file. (Completed.)
  2. Convert the ROOT data file format into an h5 format that can be loaded by Pointcept/datasets/shower_origin/shower_origin.py.
     This will produce single h5 files with one event in each file. Each event will contain information on approximately 20-40
     shower fragents for which we need to make predictions.
  3. Apply the shower origin model to each of the event files. We list the paths to the input h5 event files in a textfile.
     The inference script will produce a single result h5 file for all of the events.
  4. Integrate the output into the lantern reco chain as an option module.
     The module reads in the output of step 1 and the output of step 3.
     The information from the shower fragment origin model is used to select neutrino-candidate fragments.
     We also use it to build the shower.

## Step 1: Making spacepoints with SSNet labels from data

The input that will go into step one is made by taking (real or simulated) data from the experiment
and processing it through `ubdl/lantern_scripts.sh` which runs `ubdl/larflow/larmatchnet/larmatch/deploy_larmatchme.py`

Example of running this command on real data:

```
python3 deploy_larmatchme.py --config-file config_larmatchme_deploycpu.yaml --supera merged_dlana_d5cd7f5c-67e6-4bee-8c3a-dcefb42a63c0.root --weights /cluster/home/ubdl/larflow/larmatchnet/larmatch//larmatch_ckpt78k.pt --output output_test.root --min-score 0.5 --adc-name wire --chstatus-name wire --device-name cpu --use-skip-limit -tb
```

Note that we run this in the production microboone "lantern" container that can be if one has access to MicroBooNE's CVMFS. 

The script `deploy_larmatchme.py` makes two ROOT files. They are the "larcv" and "larlite" ROOT files 
with names derived from the given `--output` command line argument but withthe terminal `.root` changes to 
`_larcv.root` and `_larlite.root`, respectively. 

The key outputs we need to make shower clusters for the shower origin model
are in the ROOT TTree `larflow3dhit_larmatch_tree`. The tree has a single branch which only one
element which is a container class called `larlite::data::event_larflow3dhit`. This class is 
essentially a wrapper of `std::vector<larlite::data::larflow3dhit>` and is a vector container of 
the class representing spacepoint information called `larflow::data::larflow3dhit`.
The header and source for this class can be found in `ubdl/larlite/larlite/DataFormat/larflow3dhit.h/.cxx`.
The hit itself inherits from `std::vector<float>` and stores a list of numbers to associate
with 3D spacepoints inside the MicroBooNE LArTPC.
The class is a bit abused as what is stored in the float is dependent on the class that makes it.

The class that makes the `larflow::data::larflow3dhit` instances we will use is 
the class `larflow::prep::FlowMatchHitMaker`.
Its source is in `ubdl/larflow/larflow/PrepFlowMatchData/FlowMatchHitMaker.h/.cxx`. Specifically,
the class method that makes and stores hits is `FlowMatchHitMaker::make_hits`.
From the documentation in that class the vector of floats stored for each `larflow3dhit` is

```
   * larflow3dhit inherits from vector<float>. The values in the vector are as follows:
   * [0-2]:   x,y,z
   * [3-9]:   7 flow direction scores + 1 max score (deprecated based on 2-flow paradigm. for triplet, [9] is the only score stored 
   * [10-16]: 7 ssnet scores, (bg,track,shower), from larmatch (not 2D sparse ssnet)
   * [17-22]: 6 keypoint label score [nu,track-start,track-end,nu-shower,delta,michel]
   * [23-25]: reserved for plane charge
   * [26-28]: 3D flow direction
```

It is this data that we will want to convert into the hdf5 format compatible with the shower origin dataset class.

In order to gain access to the information in the ROOT file containing the `larflow3dhit` objects, it is best to use
the larlite IO manager. Here is an example of the python commands needed to get the data for an event:

```
import ROOT as rt
from larlite import larlite

inputfile = "output_larlite.root"

io = larlite.storage_manager( larlite.storage_manager.kREAD )
io.add_in_filename( inputfile )
io.open() # initialize the IO manager

nentries = io.get_entries()

for ientry in range(nentries):
  event_hits = io.get_data( larlite.data.kLArFlow3DHit, "larmatch" )
  nhits = event_hits.size()
  for ihit in range(nhits):
    hit = event_hits.at(ihit)
    ...

```

The header and source code for the `larlite::storage_manager` class is in `ubdl/larlite/larlite/DataFormat/storage_manager.h/.cxx`

### Additional larflow3dhit member variables used

Beyond the float vector, the `larflow3dhit` class has member variables relevant to Step 2:

- `hit.tick` (int): Image row (tick) — used to sample pixel values from wire-plane images.
- `hit.targetwire` (std::vector\<int\>): Wire indices per plane. `targetwire[0]`=U, `targetwire[1]`=V, `targetwire[2]`=Y.
- `hit.renormed_shower_score` (float): Combined SSNet shower probability. Computed in `FlowMatchHitMaker::store_2dssnet_score` as the sum of the renormalized shower + delta + michel scores (indices 2, 3, 4 of the 5-class SSNet output: hip, mip, shower, delta, michel). This is the score used to classify hits as shower-like for DBSCAN clustering.

## Step 2: Convert larlite ROOT to ShowerOriginDataset HDF5

### Overview

Step 2 converts the larlite ROOT output from Step 1 into per-event HDF5 files
compatible with `ShowerOriginDataset` (`pointcept/datasets/shower_origin.py`).
Since this is reco data (no MC truth), shower fragments are created by:

1. Selecting shower-like hits using `hit.renormed_shower_score >= threshold`
2. Clustering with DBSCAN (eps=3.0 cm, min_samples=4, matching the C++ `ShowerFragmentOriginMaker`)
3. Computing a PCA-based start point per fragment (most upstream point along principal axis)

Truth-only fields (`originpt`, `type`, `pret0shiftedoriginpt`) are filled with placeholders
since the model will predict these values at inference time.

### Script: `lartpc_data_prep/convert_larlite_to_showerorigin_h5.py`

Converts a larlite ROOT file into one HDF5 file per event.

**Usage:**

```bash
python lartpc_data_prep/convert_larlite_to_showerorigin_h5.py \
    --input-larlite output_larlite.root \
    --output-dir ./showerorigin_h5/ \
    --input-larcv output_larcv.root \
    --shower-threshold 0.5
```

**Command line arguments:**

| Argument | Default | Description |
|---|---|---|
| `--input-larlite` | (required) | Larlite ROOT file from `deploy_larmatchme.py` |
| `--input-larcv` | None | Larcv ROOT file for wire-plane pixel values. If not provided, `pixval` is filled with ones. |
| `--output-dir` | (required) | Output directory for per-event HDF5 files |
| `--min-score` | None | Optional stricter larmatch score filter (ghosts already removed by deploy) |
| `--shower-threshold` | 0.5 | Threshold on `renormed_shower_score` for DBSCAN input |
| `--dbscan-eps` | 3.0 | DBSCAN neighborhood radius (cm) |
| `--dbscan-min-samples` | 4 | DBSCAN minimum cluster size |
| `--min-fragment-points` | 20 | Minimum points per fragment to keep |
| `--hit-producer` | "larmatch" | larlite producer name for larflow3dhit |
| `-n` / `--nentries` | -1 | Max entries to process (-1 = all) |
| `--start-entry` | 0 | First entry to process |

**Output HDF5 schema (per event):**

```
entry_0/
  triplet_data/
    pos:          (N, 3) float32  — x, y, z coordinates (cm)
    pixval:       (N, 3) float32  — wire-plane ADC values [U, V, Y]
    uwire:        (N,)   float32  — U wire index
    vwire:        (N,)   float32  — V wire index
    ywire:        (N,)   float32  — Y wire index
    hasmatch:     (N,)   int64    — all ones (ghosts already removed)
    shower_score: (N,)   float32  — renormed_shower_score per hit
    lm_score:     (N,)   float32  — larmatch score per hit
  shower_fragments/
    @num_fragments: int
    pointindices_flat:   (T,) int64   — concatenated point indices
    pointindices_counts: (F,) int64   — points per fragment
    startpt:             (F, 3) float32 — PCA-based start point
    trackid:             (F,) int64   — sequential dummy IDs (0, 1, 2, ...)
    pid:                 (F,) int64   — all 22 (photon, generic shower)
    istrunk:             (F,) int64   — all 1 (each fragment independent)
    type:                (F,) int64   — all -1 (unknown, model predicts this)
    originpt:            (F, 3) float32 — placeholder zeros (model predicts)
    pret0shiftedoriginpt:(F, 4) float32 — placeholder zeros (no MC truth)
    nu_vertex_is_visible: int64        — 0 (unknown for reco data)
```

**Output file naming:** `showerorigin_<input_basename>_entry<NNNNNN>.h5`

**Key implementation details:**

- `extract_hits()`: Reads `larflow3dhit` objects via `larlite.storage_manager`, extracting
  `pos`, `tick`, `targetwire`, `renormed_shower_score`, and larmatch score.
- `LArCVPixelReader`: Manages a `larcv.IOManager` for sampling wire-plane pixel values.
  Uses `hit.tick` → `meta.row(tick)` and `hit.targetwire[plane]` → `meta.col(wire)` to
  index into each plane's `Image2D`. Opened once and reused across entries.
- `cluster_shower_fragments()`: Selects shower hits by threshold, runs `sklearn.cluster.DBSCAN`,
  filters clusters below `min_fragment_points`.
- `compute_start_point()`: PCA via SVD on cluster points; picks the point with smallest
  projection along the principal axis (most upstream).

### Visualization: `tools/visualize_shower_origin_reco.py`

Interactive Dash/Plotly 3D viewer for inspecting the reco HDF5 output.
Reads HDF5 files directly without requiring the Pointcept dataset class or config system.

**Usage:**

```bash
# Single file
python tools/visualize_shower_origin_reco.py --input /path/to/event.h5

# Directory of files
python tools/visualize_shower_origin_reco.py --input-dir ./showerorigin_h5/

# File list
python tools/visualize_shower_origin_reco.py --data-list /path/to/filelist.txt
```

**Panels:**

- **Row 1, Left — Selected Fragment**: The currently selected fragment highlighted in red
  with its PCA start point as a cyan cross marker. Non-fragment points in gray.
  MicroBooNE detector outline shown.
- **Row 1, Right — Shower Score**: All points colored by `renormed_shower_score` (0–1 colormap).
  Validates the SSNet classification that drives DBSCAN clustering.
- **Row 2, Full Width — All Fragments**: Each DBSCAN cluster in a distinct color.
  All start points labeled S0, S1, etc. Shows the full fragment inventory for the event.

**Navigation:**

- Event: Next Event / Random / Go-to-index buttons
- Fragment: Dropdown selector (avoids reloading HDF5 on fragment switch via `dcc.Store` caching)

### Completion Status

- [x] Conversion script (`convert_larlite_to_showerorigin_h5.py`)
- [x] Reco visualization script (`visualize_shower_origin_reco.py`)
- [x] Validate on a set of events: check fragment counts, sizes, and start points
- [x] Implement `load_pixval_from_larcv()` — currently uses `hit.tick` + `hit.targetwire` to sample
  from `image2d` "wire" product. Needs testing with actual larcv files to verify
  coordinate mapping is correct.
- [ ] Test that `ShowerOriginDataset` can load the reco HDF5 files end-to-end
  (the `type=-1` placeholder passes through without filtering issues)

## Analyzing the output before full integration

Before performing Step 4, we can make data up to step 3. We then make a tree to merge with lantern ana.
We use it to help "select" an event. We ask if an inside shower event exists.
We also ask if the predicted origin satisfies certain requirements. (We define the selection criterion later.)