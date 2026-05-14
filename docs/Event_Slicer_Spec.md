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

## Flash auxiliary H5 schema

Produced by
[`prepare_flashinfo_h5.py`](../lartpc_data_prep/prepare_flashinfo_h5.py) from
the source dlmerged ROOT file plus the paired merged H5. One file per entry,
in a parallel hashed-dirs tree alongside the merged H5 (separate location so
the flash side can be reprocessed without touching the merged data). All
times in ns or μs as labeled; tick conversions use the same constants as
`CrossingPointsAnaMethods::getTrueTick(...)`:

```
true_tpc_tick = TRIGGER_TICK + t_ns / NS_PER_TICK
flash_tpc_tick = TRIGGER_TICK + time_us / USEC_PER_TICK
```

with `TRIGGER_TICK=3200`, `USEC_PER_TICK=0.5`. (Note: the C++ source passes
`trig_time=4050.0` as a calibration knob, but for MicroBooNE that value zeros
out the subtractive offset, so the effective formula is just the two lines
above. Empirically verified on the canonical example: nu vertex → beam flash
matches at `Δtick = 0.19`.)

```
entry_0/                                    attrs:
                                              run, subrun, event,
                                              dtick_threshold,
                                              trigger_offset_ns (=4050, label only),
                                              usec_per_tick, trigger_tick,
                                              drift_velocity_cm_per_us,
                                              image_tick_min, image_tick_max,
                                              n_pmts (=32)
├── flashes/                                 length F (beam + cosmic, sorted by stream)
│   ├── pe                  (F, 32) float32   per-PMT photoelectrons (raw), remapped
│   │                                          onto physical PMT index. See note below.
│   ├── total_pe            (F,)    float32
│   ├── time_us             (F,)    float32   relative to TPC trigger (tick 3200)
│   ├── tpc_tick            (F,)    float32   = time_us/0.5 + 3200
│   ├── producer_id         (F,)    int32     0=simpleFlashBeam, 1=simpleFlashCosmic
│   ├── flash_index         (F,)    int32     index within producer stream
│   ├── y_center, z_center  (F,)    float32   from larlite::opflash::YCenter/ZCenter
│   ├── matched_slice_id    (F,)    int64     -1 if no slice within threshold
│   └── match_dtick         (F,)    float32   |Δtick| for the recorded slice match
├── pmt_positions           (32, 3) float32   from larutil::Geometry::GetOpDetPosition
├── mc_particle_start_times/                  parallel to merged H5 mc_particle_tree/trackid
│   ├── trackid             (M,)    int32     parallel-aligned to mc_particle_tree
│   ├── start_t_ns          (M,)    float64   from MCTrack/MCShower Start().T()
│   ├── start_tpc_tick_nodrift (M,) float32
│   └── source              (M,)    int8      0=mctrack, 1=mcshower, -1=Geant4 secondary
└── slice_flash_matches/                      one row per slice (S = n_slices)
    ├── slice_id            (S,)    int64
    ├── primary_origin      (S,)    int32     1=nu, 2=cosmic
    ├── primary_tpc_tick    (S,)    float32   lead primary's true tick
    ├── matched_flash_idx   (S,)    int32     -1 if no match within dtick_threshold
    ├── match_dtick         (S,)    float32
    ├── is_null_flash       (S,)    int8      1 = slice with charge but no matched flash
    ├── crosses_image_boundary (S,) int8      1 = primary trajectory leaves
    │                                          [image_tick_min, image_tick_max];
    │                                          downstream loss should down-weight
    │                                          (light is real but charge is incomplete)
    └── total_pe_matched    (S,)    float32   convenience: flashes/total_pe[matched_flash_idx]
```

**PMT channel remap** — MicroBooNE has two readout electronics chains on the
same 32 physical PMTs, with different trigger windows. larlite's
`opflash::PE(k)` indexes into a 336-long channel array where the beam stream
puts its 32 PMTs on OpChannels `[0, 31]` and the cosmic stream on OpChannels
`[200, 231]`. **OpChannel != OpDet:** the channel↔opdet mapping is a
non-trivial permutation (e.g., OpChannel 0 → OpDet 3, ch 4 → OpDet 0, …).
`Geometry::GetOpDetPosition(i)` is indexed by **OpDet**. The script reads
`OpDetFromOpChannel(ch)` once per entry, then assigns
`pe[opdet] = opflash::PE(ch)` so that `flashes/pe[i]` and
`pmt_positions[i]` always refer to the **same physical PMT (OpDet i)**.
Verified on the canonical example: the brightest cosmic slice's spacepoints
sit in y∈[35,100], z∈[63,136], and its matched flash's brightest PMT
(OpDet 29) is at (y=55, z=88) — geometric proximity matches the brightness
gradient. (Two earlier passes failed this check: first pass had
`total_pe=0` for every cosmic flash because [0, 31] was empty for them;
second pass had non-zero PE but wrong indexing because beam-channel-k and
position-opdet-k are not the same physical PMT.)

**Matching algorithm** (single-pass, greedy, mirrors FlashMatcherV2 logic):
for each slice, find the flash with smallest `|tick_flash − tick_primary_lead|`.
If `≤ dtick_threshold` (default 3 ticks = 1.5 μs, stored as an entry attr),
record it. Slices with no matched flash get `matched_flash_idx=-1,
is_null_flash=1`. A flash can be the best match for multiple slices; the
reverse table `flashes/matched_slice_id` keeps the closest one.

**Image-boundary tagging** (`crosses_image_boundary`): per slice, walk every
member trackid's mctrack/mcshower trajectory. If any step's reco tick lies
outside `[2400, 8448]`, flag the slice. Per user note: these slices have
*light from real charge, but the charge is incomplete in the image*, so the
loss should down-weight rather than drop them.

**Visualization:**
[`Pointcept/tools/visualize_slice_flash_match.py`](../tools/visualize_slice_flash_match.py)
takes a paired merged H5 + flashinfo H5, exposes a dropdown of all slices
(each labeled with origin, PID, point count, matched flash, ΣPE, and a
boundary-crossing tag), and shows the selected slice's spacepoints in 3D
plus the matched flash's PE pattern on a 2D y-z PMT map.

**Open follow-ups:**
- The `mc_particle_tree` group in the merged H5 still drops the t component
  of `MCPGNode.start`. A TODO comment is in place at
  [`SimChTripletLabelMaker.cxx:1081`](../../ubdl/larflow/larflow/PrepFlowMatchData/SimChTripletLabelMaker.cxx#L1081)
  to add `mc_particle_tree/start_t_ns` next time the merged H5 is
  reprocessed. Once that lands, the flash-prep script will no longer need to
  open the ROOT file just to read the time.
- The single-entry script needs a batch wrapper that iterates a dlmerged
  ROOT and writes the parallel hashed-dirs tree.
- `FlashMatcherV2` (C++) still uses `MCPixelPGraph`. Its logic is now
  ported into Python; the C++ class is not on the critical path for the
  event-slicer training and can be left as-is for now.

## Charge-to-flash predictor

The MicroBooNE photon library is a Geant4-simulated lookup: for each of
75×75×400 = 2.25M cryostat voxels, the fraction of isotropically-emitted
photons reaching each of the 32 OpDets is tabulated. The source ROOT TTree
has 33.6M (Voxel, OpChannel, Visibility) sparse entries; we densify once
and consume it as a `(75, 75, 400, 32) float32` tensor on the GPU (288 MB).

Forward pass for a single cluster of M spacepoints at TPC positions
`r_i ∈ R^3` with per-spacepoint charge `q_i`:

```
n_emitted_i = γ · q_i                          # γ = photons / charge
g_i         = (r_i - cryo_origin_tpc) / voxel_len_cm    # continuous voxel coords
vis_i       = trilinear_interp(LUT, g_i)       # (32,) per spacepoint, fraction
PE_j        = Σ_i  n_emitted_i · vis_i[j]      # (32,) predicted flash
```

Drift correction (per candidate flash with t0 = `t_flash` μs vs trigger):

```
x_true = x_apparent - v_drift · t_flash
```

For a `(cluster, flash)` cost matrix at training time, the same forward pass
runs once per pair, each pair shifting that cluster's spacepoints by the
flash-specific `dx`.

**Inputs the caller is responsible for:**

- **Charge selection** — `q_i` comes from `triplet_data/pixval`. Y plane
  (`pixval[:, 2]`) is the most reliable calorimetric proxy; when Y is zero
  (dead channel / non-readout pixel), fall back to `0.5 · (U + V)`.
  Implemented as `select_charge_y_with_uv_fallback(pixval)` in the same
  module.
- **γ (photons-per-charge factor)** — a fixed scalar to be calibrated
  empirically over a large set of (slice, matched flash) pairs by minimizing
  ΣPE residuals. On the canonical example's nu slice, γ ≈ 1.6 photons/ADC
  reproduces ΣPE; we'll redo this over a real sample.
- **PMT quantum efficiency** — the LUT visibility is purely geometric /
  scattering. QE is absorbed into γ (the predicted "PE" is really
  γ · charge · vis, with γ tuned so that quantity matches observed PE).

**Implementation files:**

- [`Pointcept/lartpc_data_prep/build_photonlib_cache.py`](../lartpc_data_prep/build_photonlib_cache.py)
  — one-time converter: reads the ROOT TTree via PyROOT's `RDataFrame`,
  densifies, saves
  `Pointcept/lartpc_data_prep/dat/photonlib_v6_70kV.npz` with grid metadata.
- [`Pointcept/pointcept/models/event_slicer/photonlib.py`](../pointcept/models/event_slicer/photonlib.py)
  — `PhotonLibLookup` torch module + the Y-with-UV-fallback helper.
  Buffers (`vis_table`, `cryo_origin_tpc`, `voxel_len_cm`, `nvoxels_dim`) are
  registered so `.to(device)` / `.half()` Just Work.
- [`Pointcept/lartpc_data_prep/test_photonlib_torch.py`](../lartpc_data_prep/test_photonlib_torch.py)
  — three sanity tests:
  1. **Trilinear parity vs C++ UBPhotonLib**: 300 random TPC points × 32
     OpDets = 9600 comparisons, **100% match** at machine precision
     (max abs err 5.22 × 10⁻⁸).
  2. **Nu slice prediction**: for the canonical entry-0 nu slice (3373
     spacepoints), drift-corrected predicted PE has **cosine similarity
     0.979** with the observed beam flash, top-OpDet ratios all 0.7–1.3.
  3. **Throughput** (GPU): `predict_flash` ≈ 3.5 M points/sec; the
     pair-wise variant `predict_flash_pairs` ≈ 2.8 ms/pair (Python-loop
     bound, vectorizable if it becomes a bottleneck).

**Module API summary:**

```python
pl = PhotonLibLookup("dat/photonlib_v6_70kV.npz",
                     fp16=False, use_trilinear=True).cuda()
# Single-cluster forward
pe = pl.predict_flash(pos_xyz, q_emitted, cluster_id, n_clusters)
# (cluster, flash) cost-matrix forward
pe_pairs = pl.predict_flash_pairs(pos_xyz, q_emitted, cluster_id,
                                  pair_cluster_idx, pair_flash_t0_us,
                                  v_drift_cm_per_us=0.1098)
```

**Running the viewer with predictions:**
[`tools/visualize_slice_flash_match.py`](../tools/visualize_slice_flash_match.py)
accepts a `--photonlib-cache` flag which enables a third panel (right of
the observed PMT view) showing the predicted PE pattern for the selected
slice. Both panels share their `log10(PE+1)` color scale so the patterns
are directly comparable, and the predicted panel's title shows the cosine
similarity with the observed flash plus the γ that was applied. γ is
per-producer: `--gamma-beam` is used when the matched flash is
`simpleFlashBeam` (producer_id=0), `--gamma-cosmic` when it's
`simpleFlashCosmic` (producer_id=1). `--gamma` is a default both fall back
to. Drift correction uses the matched flash's `time_us`.

```bash
python tools/visualize_slice_flash_match.py \
    --merged-h5    .../merged_..._entry000000.h5 \
    --flashinfo-h5 .../flashinfo_..._entry000000.h5 \
    --photonlib-cache lartpc_data_prep/dat/photonlib_v6_70kV.npz \
    --gamma-beam 1.6 --gamma-cosmic 0.16
```

**Open follow-ups:**

- See pre-implementation item #5 below for the γ + readout-window
  calibration.
- Vectorize `predict_flash_pairs` (process all pairs in one batched
  voxelization + index_select) if the Python loop becomes a bottleneck.
- The downstream loss objective (Sinkhorn / Hungarian assignment of slices
  to flashes using the predicted PE vectors as the cost) is a separate
  piece of work.

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

Flash side (auxiliary, paired by entry):

```
dlmerged_*.root  +  merged_<basename>_entry<NNNN>.h5
  │                                  │
  ▼                                  │
prepare_flashinfo_h5.py              │
  │   read simpleFlashBeam + simpleFlashCosmic from larlite::event_opflash
  │   read MCTrack/MCShower Start().T() for all trackids
  │   pull mc_particle_tree + triplet_data from paired merged H5
  │   compute_slice_labels → primary trackid per slice
  │   greedy match slices ↔ flashes by |Δtick| ≤ dtick_threshold (3 ticks)
  │   tag slices whose member trajectories leave the image tick window
  │
  ▼
flashinfo_<basename>_entry<NNNN>.h5  (parallel directory tree)
```

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

1. ~~Flash information data prep pipeline~~ **Resolved 2026-05-14 (v1).**
   Separate auxiliary H5 files paired by entry, in a parallel directory tree:
   `flashinfo_<basename>_entry<NNNN>.h5` ↔ `merged_<basename>_entry<NNNN>.h5`.
   Each aux file holds the per-entry flashes, MC particle start times (read
   from larlite MCTrack/MCShower since `mc_particle_tree` doesn't carry the
   time component), and a greedy slice↔flash match table. Producer:
   [`Pointcept/lartpc_data_prep/prepare_flashinfo_h5.py`](../lartpc_data_prep/prepare_flashinfo_h5.py)
   (single-entry interactive script; batch driver is a follow-up). See the
   "Flash auxiliary H5 schema" section below.
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
4. ~~How do we efficiently calculate the flash match prediction from a
   cluster of voxels?~~ **v1 done 2026-05-14: GPU lookup-table predictor.**
   The MicroBooNE photon library
   (`ubdl/ublarcvapp/ublarcvapp/UBPhotonLib/dat/uboone_photon_library_v6_70kV_EnhancedExtraTPCVis.root`,
   33.6M sparse entries) is densified once into a `(75, 75, 400, 32)` float32
   tensor (288 MB on GPU) and queried via vectorized advanced indexing
   (trilinear interpolation). See "Charge-to-flash predictor" section below.

5. **Calibrate the predictor: per-producer γ + readout-window transfer
   function.** The lookup-table predictor produces `PE_pred ∝ γ · Σ(q · vis)`
   where γ is a free photons-per-ADC scalar. Two distinct effects mean γ is
   not a single number across the dataset:
   - **Per-producer γ.** The beam stream (`simpleFlashBeam`) and cosmic
     stream (`simpleFlashCosmic`) use the same physical PMTs but different
     electronics. The cosmic stream's readout is triggered (a waveform
     sample is only kept once a pulse crosses threshold) and writes out a
     fixed-length window that is shorter than the LAr scintillation late
     component (τ ≈ 1 μs). So a fixed fraction of every cosmic flash's PE
     is missing from the recording, and `γ_cosmic ≪ γ_beam`. Empirically
     on the canonical example, `γ_beam ≈ 1.6` and `γ_cosmic ≈ 0.16` give
     ΣPE matched to ~5% per slice (cosine ≈ 0.98). Need to fit both
     constants over a real sample of (slice, matched flash) pairs.
   - **Readout-window transfer function (cosmic only).** The fraction of
     PE actually integrated by the cosmic-stream's fixed window probably
     depends on the absolute pulse amplitude (saturation, threshold
     timing, etc.). A scalar γ_cosmic captures the average but per-PMT
     residuals will likely show an amplitude-dependent shape. Plan: fit
     a per-OpDet (or pooled-across-OpDets) `PE_obs = f(PE_pred)` —
     linear (`a + b·PE_pred`) first, polynomial / sigmoid-saturation if
     residuals demand it — from a calibration sample of matched
     (slice, cosmic-flash) pairs.

   Calibration workflow (not yet implemented): scan a directory of paired
   merged + flashinfo H5 files, run `predict_flash` for each non-null
   matched slice, accumulate `(PE_pred[j], PE_obs[j])` separately for
   beam and cosmic producers, then (a) least-squares fit a single γ per
   producer, (b) least-squares fit `PE_obs = a + b·PE_pred` per OpDet for
   the cosmic stream. Store the fitted constants alongside the photonlib
   cache (e.g., `dat/photonlib_v6_70kV_calibration.npz`) so the predictor
   module can load them automatically. Boundary-crossing slices
   (`flashinfo/slice_flash_matches/crosses_image_boundary == 1`) should
   be excluded from the fit — their predicted PE is missing the
   out-of-image charge contribution.

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




