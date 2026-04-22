# ExtBNB LArMatch Ghost Filtering Pipeline

Runs LArMatch on ExtBNB (real detector) data to produce ghost scores,
then converts the output to HDF5 files with per-point larmatch scores
for use in Pointcept training.

## Pipeline Overview

For each input `merged_dlreco_*.root` file:

1. **Step 1 (lantern container):** Run SSNet + LArMatch.
   - Produces `larmatchme_larlite.root` and `merged_dlreco_with_ssnet.root`.
   - LArMatch assigns a ghost score to each spacepoint.
2. **Step 2 (pointcept container):** Convert to HDF5 with larmatch scores.
   - Runs `convert_larmatch_to_h5.py` to produce per-event HDF5 files
     with all the standard fields (pos, pixval, wire coords) plus
     `larmatch_score` (float32, range [0,1]).

## Containers

- **Lantern** (Step 1): `/cvmfs/uboone.opensciencegrid.org/containers/lantern_v2_me_06_03_prod`
- **Pointcept** (Step 2): `/cluster/tufts/wongjiradlabnu/larbys/larbys-container/pointcept_cuml.sif`

## Input Data

ROOT files from:
- `inputlists/inputlist_run3_G1_extbnb_dlreco.txt` (50,410 files)
- `inputlists/inputlist_run3_G2_extbnb_dlreco.txt` (29,052 files)

(Input lists are in `/cluster/tufts/wongjiradlab/hmcgui01/mphys/Pointcept/lartpc_data_prep/inputlists/`)

For additional simulated datasets made for Pointcept training (i.e. not official MicroBooNE production files), 
see the folder:

```
/cluster/tufts/wongjiradlab/larbys/data/ub_on_tufts/
```

which contains dlmerged ROOT files with simch-based truth.

## Usage

```bash
# Test with a single job (processes 1 file)
sbatch --array=0-0 submit_extbnb_larmatch.sh

# Check logs
cat workdir/extbnb_larmatch_jobid_0000/log_extbnb_larmatch_jobid0.txt

# Full production (adjust array range based on input list size / stride)
# G1: 50410 files, stride=1 -> array=0-50409
sbatch --array=0-50409 submit_extbnb_larmatch.sh
```

## Output

HDF5 files in `OUTPUT_DIR/<fileno/1000>/<fileno/100>/` with structure:
```
/entry_0/triplet_data/
    pos          (N, 3) float32  - 3D spacepoint coordinates
    pixval       (N, 3) float32  - ADC values from u, v, y wire planes
    uwire        (N,)   float32  - U wire index
    vwire        (N,)   float32  - V wire index
    ywire        (N,)   float32  - Y wire index
    tick         (N,)   float32  - drift tick
    edep         (N, 3) float32  - same as pixval (for compatibility)
    larmatch_score (N,) float32  - ghost score from LArMatch [0=ghost, 1=true]
```

## Downstream Use

In the Pointcept LArTPCDataset, load `larmatch_score` and filter points:
```python
# In dataset config:
larmatch_score_threshold=0.5  # cut points below this score
```
