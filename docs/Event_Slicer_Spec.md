# Event Slicer Spec

This discribes background information, model design, and plan for developing an event slicer for MicroBooNE events based
around the Sonata pre-trained backbone.

At its core, the event slicer is essentially an instance segmentation problem. 
We are trying to cluster spacepoints that belong to the particle tracks deriving from the same parent cosmic or neutrino event.
These are the particles that occur within an order tens of nanoseconds of each other.
For example, a cosmic muon might pass through the detector. All of the charge it or its daughter particles deposit
should be clustered together into a single event slice. If there is enough deposited energy, there should also be an observable scintillation flash that we can associate to this slice. 
This is the same for particles produced by a neutrino interaction. 
The scintillation light should be in time with the charge deposition, 
and the scintillation flash occurs in coincident with the beam trigger.
Associating charge deposits to the in-time beam flash is a key way we separate neutrino interactions from cosmic ray backgrounds. 

Our approach is to use the Sonata pre-trained backbone to extract features from the spacepoints. We then define a mask2former-type architecture on top of the extracted features to perform the clustering. 
At this level, we only need to do clustering for points merged into 5 cm voxels -- we are not trying to separate individual particles yet. 

Because of the charge-light relationship, we want to incorporate the flash information into the clustering.
We have a model that can predict the contribution of each spacepoint, given an amount of charge in the voxel and its position, to the amount of photoelectrons seen in each PMT.
However, the challenge is that this model cannot be used until we assign a charge deposit to a flash.
So during training, once the mask2former layer produces a cluster, we will need to assign each charge cluster to the possible flashes. We imagine this can be done using hungarian matching or probably easier, 
a soft assignment using the Sinkhorn divergence.

Finally, as a pre-processing step, we need to apply deghosting to the spacepoints.  
We have the output of the existing LANTERN network.
But we now have the deghosting model fine-tuned from the sonata backbone.

The steps the training pipeline should take are as follows:

For each training iteration,
  1. load the spacepoints and flashes for an event
  2. apply deghosting to the spacepointsw by filtering events above a true-point score threshold which we vary event-by-event as a means of data augmentation. We will preprocess data to get a true-point score for each point and remove low-score points.
  3. apply the sonata backbone, getting features for each spacepoint
  4. apply voxelization to the spacepoints, merging points within 5 cm of each other into a single voxel.
  5. apply the mask2former architecture to the voxelized spacepoints, producing clusters.
  6. for each cluster, we need to find all compatible flashes. For each flash, this defines a time-offset that we translate using a constant drift velocity to get a x-offset. We apply this offset and get a t0-corrected position for each voxel in the cluster for that flash. This then allows us to use the light model to produce a flash prediction. We can then compare this prediction to the paired flash using the sinkhorn divergence. 
  7. the result of the previous step is a set of (cluster,flash) pairs that form a cost matrix. We can then use hungarian matching for hard assignments or sinkhorn divergence to get a transport plan that we use as soft assignments.
  8. If we made hard-assignments,then it is possible to get a final predicted vs. observed flash loss. This flash match loss is added to the mask2former loss.  If we made soft-assignments, I guess we add up all the cluster,flash  
  pair losses weighted by the transport plan weights from the matching sinkhorn divergence. Do we stop the gradients from either of the hard or soft assignment weights?
  9. Like the spacepoint mask, we subsample points to calculate the loss. Do we also subsample the (cluster,flash) pairs for the flash-match loss?

## Component information

There are a number of components to be able to implement this loss. Here is information on each.

### Flashes in the data

The flashes are the summed charge collected in the PMTs for a given time window. 
For MicroBooNE there are 32 PMTs. So the flash is a vector of 32 float values. 
Each flash vector is associated to a time in microseconds relative to the event trigger.
We have 32 positions for each of the PMTs.

### Flash model

We have a model for calculating the predicted flash vector for a given set of voxels with their charge.
The model is in the `ublightmodelnet` of the workspace. 
There are two models. One is a simple feed-forward neural network that takes the charge in each voxel 
and its position as input and outputs a predicted flash vector. 
We can then sum the predicted vectors for all voxels in a cluster 
to get a predicted flash vector for the cluster.   

We also want to serve up a model that uses a library lookup table of values. 
The values are the fraction of photons that land in each PMT for a given position and time offset. 
This is a precomputed table. 

The script pointcept/tools/dump_flash_info.py provides an example of how to access the flash information in the data files.

Past code for building the charge-light training set for the light model Siren MLP is in `ublightmodelnet/data_prep/include`.

The code for the lookup table model is in `ubdl/ublarcvapp/ublarcvapp/UBPhotonLib`. The class that reads in the look up tablel information is in `ubdl/ublarcvapp/ublarcvapp/UBPhotonLib/PhotonLib.h/.cxx`.

### Spacepoint Ground Truth Labels

**Status (2026-05-13): the slice ground truth is recoverable from data already
in the H5 files, no regeneration required for v1.**

Per-spacepoint fields in `entry_*/triplet_data/` (see "Data schema" below for the
complete list):
- `trackid` (N,)   Geant4 trackid of the first non-(-1) contributing track
  (not necessarily highest-edep — see V1 caveat below)
- `origin` (N,)    0 = unmatched/ghost, 1 = neutrino, 2 = cosmic
- `hasmatch` (N,)  0 = ghost spacepoint, 1 = real
- `pid` (N,)       PDG of the contributing track
- A per-spacepoint ancestor trackid (`aid`) is also written by the C++ producer
  (`SimChTripletLabelMaker`) when `_is_mc` is true, but the example H5 files
  predate the Nov-2025 commit that added it and therefore lack the field.

The full Geant4 particle graph for the event is stored in
`entry_*/mc_particle_tree/` (see schema below). From `trackid` +
`mc_particle_tree` we walk `parent_trackid` to a primary, producing the slice
ground truth without rerunning data production.

**Slice definition for MicroBooNE v1:**
- One slice per cosmic primary (ancestor with `origin==2`). Daughters of a
  cosmic neutron / shower stay in the parent's slice.
- One merged slice per neutrino vertex containing every `origin==1` primary
  whose `start_pos` is nearest that nu_vertex. With MicroBooNE's usual ≤1
  nu interaction per event this collapses the genie final-state mu / p / n
  / pi block into a single slice.
- Spacepoints with `hasmatch==0` are forced to the "no-slice" target. The
  deghoster will drop most of these before the slicer sees them; the few that
  remain are an explicit "do-not-cluster" class.

**V1 caveats / known follow-ups:**
- `trackid` on a multi-contributor spacepoint is the *first* non-(-1) entry in
  `triplet.trackids`, not the highest-edep contributor. For shower overlap
  this gives a noisy slice assignment. Acceptable for v1; switch to the
  highest-edep contributor at H5-production time when this becomes the
  dominant error.
- The `mc_particle_tree` carries no per-particle time (`start_pos` is
  `(N,3)`; `MCPGNode` has no time field). Slice→flash matching (item #4 of
  the pre-implementation plan) therefore requires either (a) adding
  `start_t` to `MCPGNode` and re-running H5 production, or (b) inferring t0
  from the spacepoint `tick` distribution within the slice.
- Multi-nu_vertex events (rare for MicroBooNE, expected for SBND / DUNE-ND)
  are handled by assigning each `origin==1` primary to its nearest entry of
  `nu_vertices`, so the same code generalizes.

**Implementation:**
[`Pointcept/lartpc_data_prep/slice_labels.py`](../lartpc_data_prep/slice_labels.py)
walks the graph and returns `slice_id (N,)` + per-slice metadata
(`primary_trackid`, `primary_origin`, `primary_pid`, `primary_start_pos`,
`primary_n_spacepoints`, `slice_member_trackids`). Called on-the-fly from the
data loader / visualizer — overhead is negligible (~hundreds of nodes per
event).

**Visualization:**
[`Pointcept/tools/visualize_lartpc_h5data.py`](../tools/visualize_lartpc_h5data.py)
exposes a `Slice (Truth Instance)` color mode in the 3D dropdown. The nu
slice is rendered red; cosmic slices are spread across HSV; ghosts /
orphans are gray. The nu_vertex (gold diamond) and each cosmic primary's
`start_pos` (small marker matching its slice color) are overlaid.

### Deghosting

We have a deghosting model that is fine-tuned from the sonata backbone. 

We also have LArMATCH network scores saved in the data files for each spacepoint.

First thing to do is to understand if one is better than the other.
We need to establish an inference pipeline for the ghost removal with LoRA adaptation.
We also need to decide if we want to use the pure backbone features, 
or the features from the deghosting network (i.e. features from the original backbone + LoRA weights).

We want to look at mIOU and accuracy for true points. 
We also are interested the accuracy and mIOU for true neutrino spacepoints.
There should be an origin flag in the spacepoint truth labels that tells us if a point is from a neutrino interaction or not.

## Data schema (per-entry H5)

Each H5 file produced by
[`Pointcept/lartpc_data_prep/process_dlmerged_to_hdf5_event_files.py`](../lartpc_data_prep/process_dlmerged_to_hdf5_event_files.py)
contains one `entry_0` group per event with the following structure (sizes
are from the entry-1 example used throughout this doc, N=358 510 spacepoints,
M=333 MC nodes):

```
entry_0/                                            attrs: run, subrun, event
├── triplet_data/                                   per-spacepoint, length N
│   ├── pos               (N, 3) float32             detector cm (x,y,z)
│   ├── uwire, vwire, ywire  (N,) float32            per-plane wire index
│   ├── tick              (N,)   int32               readout tick
│   ├── pixval            (N, 3) float32             ADC value, 3 planes
│   ├── edep              (N, 3) float32             energy dep, 3 planes
│   ├── lm_score          (N,)   float32             LArMatch true-point score
│   ├── shower_score      (N,)   float32             LArMatch shower score
│   ├── larmatch_feats    (N, 48) float32            LArMatch backbone feats
│   ├── truth_match       (N,)   int64               internal pair index
│   ├── trackid           (N,)   int64    (MC only)  first non-(-1) contributing Geant4 tid
│   ├── pid               (N,)   int64    (MC only)  PDG of `trackid`
│   ├── origin            (N,)   int64    (MC only)  0=ghost, 1=nu, 2=cosmic
│   ├── hasmatch          (N,)   int32    (MC only)  0=ghost, 1=real
│   ├── ssnet_label       (N,)   int32    (MC only)  SimChTripletLabelMaker class (0..9)
│   └── aid               (N,)   int32    (MC only, post-Nov-2025 only)
│                                                    ancestor Geant4 trackid for `trackid`
├── mc_particle_tree/                               per-Geant4-node, length M
│   ├── trackid           (M,)   int32              node trackid
│   ├── parent_trackid    (M,)   int32              parent tid (-1 = root sentinel)
│   ├── pid               (M,)   int32              PDG
│   ├── origin            (M,)   int32              -1, 1=nu, 2=cosmic
│   ├── process_code      (M,)   int32              Geant4 process code
│   ├── energy_mev        (M,)   float32
│   ├── start_pos         (M, 3) float32            cm, true Geant4
│   ├── start_pos_sce     (M, 3) float32            cm, SCE-shifted
│   ├── num_daughters     (M,)   int32              CSR children counts
│   ├── daughter_start_indices (M,) int32           CSR offsets
│   ├── daughter_trackids (D,)   int32              CSR child trackids, D = sum(num_daughters)
│   └── nu_vertices       (V, 3) float32            nu interaction points, V usually 1
│       TODO: no `start_t` on nodes; needed for slice↔flash matching
├── mckeypoints/                                    truth keypoints for KP loss
│   ├── pos, startpos     (K, 3) float32
│   ├── imgcoord          (K, 4) int32
│   ├── kptype, pid, trackid (K,) int32             0..5 (NuVtx/Start/End/Shower/Michel/Delta)
├── shower_fragments/                               DBSCAN shower clusters (truth)
│   ├── originpt          (F, 3) float32            cluster origin point
│   ├── startpt           (F, 3) float32
│   ├── pret0shiftedoriginpt (F, 4) float32         (x,y,z,t) in true MC coords
│   ├── pid, trackid, type, istrunk (F,) int64
│   ├── pointindices_flat (P,)   int64              spacepoint indices into triplet_data
│   └── pointindices_counts (F,) int64              CSR offsets into pointindices_flat
└── image_data/                                     2D wireplane sparse images
    ├── plane{0,1,2}/
    │   ├── coord  (P_p, 2) int32                   (col=wire, row=tick/6 − 400)
    │   ├── feat   (P_p,)   float32                 ADC
    │   ├── dims, origin, pixsize
    └── triplet_imgpix_index (N, 4) int32            triplet ↔ 3-plane pixel mapping
```

Derived (computed on-the-fly by
[`lartpc_data_prep/slice_labels.py`](../lartpc_data_prep/slice_labels.py)):

```
slice_id            (N,) int64    primary_trackid per spacepoint (−1 ghost/orphan)
primary_trackid     (S,) int64    sorted unique slice keys
primary_origin      (S,) int64    1=nu, 2=cosmic
primary_pid         (S,) int64    PDG of slice's lead primary (0 for merged-nu)
primary_start_pos   (S, 3) float32
primary_n_spacepoints (S,) int64
slice_member_trackids list[list[int]]  the primaries collapsed into each slice
```

## Code flow

### Per-event data production (offline)

```
dlmerged_*.root  (one ROOT file, many entries — wire images, simch,
                  SSNet, MCTrack/MCShower, optical flashes)
  │
  ▼
process_dlmerged_to_hdf5_event_files.py        (Pointcept container)
  │   sets up larlite + larcv readers; iterates entries
  │
  ▼
larflow.prep.SimChTripletLabelMaker             (C++; ubdl/larflow)
  │   build MCPixelPGraph from MCTrack/MCShower
  │   form 3-plane triplets, propose 3D spacepoints
  │   per spacepoint: trackid/pid/origin from simch + MCPixelPGraph
  │   per spacepoint: lm_score / shower_score / larmatch_feats from LArMatch
  │   per spacepoint: hasmatch from comparison to simch truth voxels
  │   write entry_N/triplet_data, /mc_particle_tree, /mckeypoints,
  │         /shower_fragments, /image_data
  │
  ▼
one merged_*_entry*.h5 per (file, entry)
```

`merge_reco_truth_showerorigin.py` (only relevant for the shower-origin
pipeline) re-merges reco fragments with these truth labels — it preserves
`mc_particle_tree` so the slice walker works on its output too.

Note that **flashes are not currently propagated into the H5**. They live in
the source ROOT file's `opdigit_simpleFlashBeam` and friends (see
[`Pointcept/tools/dump_flash_info.py`](../tools/dump_flash_info.py) for the
access pattern) and need their own data-prep step — item #1 in the plan below.

### At training / visualization time

```
H5 file
  │
  ▼
data loader (e.g. ShowerClusteringDataset → EventSlicerDataset, TBD)
  │   pull triplet_data (pos, lm_score, larmatch_feats, ...)
  │   pull mc_particle_tree                                 ── if MC
  │   compute_slice_labels(mpt, trackid, hasmatch)          ── if MC
  │     - walk parent_trackid → primary
  │     - merge origin==1 primaries per nu_vertex
  │     - mask hasmatch==0 to slice_id = -1
  │
  ▼
training batch with per-spacepoint slice_id + per-slice metadata
```

The visualizer (`Pointcept/tools/visualize_lartpc_h5data.py`) follows the
same path and exposes `Slice (Truth Instance)` as a 3D color mode.

## Plan for pre-implementation

We want to establish the computation pipeline and input data formats for the event slicer.

This will be important information needed for implementation of the data loader and the model pipeline.

1. Flash information data prep pipeline and integration of the data loader.
   - Do we process the input files to make separate h5 files for the flash information 
   and load it in the data loader during training?  
   - Or do we make a special processing pipeline that appends the flash information to the training file h5 files?
2. ~~What is the status of the ground truth slice information for the spacepoints?~~ **Resolved
   2026-05-13.** A slice ID is *not* stored in the H5 directly, but the
   `mc_particle_tree` group lets us derive one cheaply: walk
   `parent_trackid` to a primary, then merge `origin==1` primaries that
   share a nu_vertex. See the "Spacepoint Ground Truth Labels" section
   above. Implementation:
   [`lartpc_data_prep/slice_labels.py`](../lartpc_data_prep/slice_labels.py).
   Visualization: `Slice (Truth Instance)` color mode in
   [`tools/visualize_lartpc_h5data.py`](../tools/visualize_lartpc_h5data.py).
   Open follow-ups noted there: highest-edep trackid resolution (currently
   first non-(-1)), and adding `start_t` to `MCPGNode` for slice↔flash
   matching.
3. Compare the accuracy of the two deghoster options: LANTERN versus LoRA-adapted backbone.
4. How do we efficiently calculate the flash match prediction from a cluster of voxels? 
   The UB light model lookup table file is huge, but once on GPU memory can be decomposed into MatMul operations for fast calculation. Maybe we can get away with sampling look-up table sparsely, and interpolating -- again a fairly fast operation I think for a GPU. This is the hardest part of the problem.

## Example data

Source ROOT file: `/mnt/ddrive/data/ub_on_tufts/bnb_nu_pi0filter_corsika/000/000/dlmerged_coriska_bnb_nu_pi0filter_fileno000001.root`
h5 training data file (for Sonata/shower clustering/deghost+semantic seg):
  - /mnt/ddrive/data/ub_on_tufts/h5/bnb_nu_pi0filter_corsika/000/000/merged_bnb_nu_pi0filter_corsika_fileno00001_entry000000.h5
  - /mnt/ddrive/data/ub_on_tufts/h5/bnb_nu_pi0filter_corsika/000/000/merged_bnb_nu_pi0filter_corsika_fileno00001_entry000001.h5
  - /mnt/ddrive/data/ub_on_tufts/h5/bnb_nu_pi0filter_corsika/000/000/merged_bnb_nu_pi0filter_corsika_fileno00001_entry000002.h5


## Detailed Specification of the Model

TODO with the help of CLAUDE

## Model Implementation Plan

TODO with the help of CLAUDE.




