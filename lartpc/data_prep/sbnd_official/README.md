# SBND (larcv2) → LArFormer HDF5: input analysis

Status (2026-10-14): **analysis only, no converter yet.** This document records
what the SBND larcv2 files contain, how they map onto the per-event HDF5 schema
produced for MicroBooNE by
[`../uboone_official/convert_dlmerged_to_larformer_h5.py`](../uboone_official/convert_dlmerged_to_larformer_h5.py),
and what has to be decided before writing a converter.

The converter is deliberately on hold. The LArFormer chain hard-codes MicroBooNE
constants (§6), so a **detector-agnostic constants utility** (geometry, space
charge, LAr properties, electronics clocks for SBND / MicroBooNE / ICARUS) comes
first. Both the converter and the LArFormer code should then take their
constants from that utility.

Test file used throughout:
`test/larcv_mc_20250522_144925_596433-41b171a8-3369-4bbb-8a40-87a7f83caaca.root`
(21 entries, BNB neutrinos + CORSIKA cosmics, SBND MC).

---

## 1. Building larcv2 in the pointcept container

larcv2 (the SBND / DeepLearnPhysics fork, **not** the larcv1 MicroBooNE uses)
lives at `test/larcv2` (its own git repo, upstream commit `0d6ea74`).

```bash
sh test/start_tufts_container.sh          # apptainer shell into pointcept_cuml.sif
cd test
source setenv_larcv2.sh                   # ROOT, CUDA, OpenCV, conda libs
source larcv2/configure.sh                # -q for quiet
make -C larcv2 -j8                        # only needed after source changes
python3 -c "from larcv import larcv"
```

**Fix applied to `larcv2/configure.sh`** (uncommitted in the larcv2 repo;
original kept as `configure.sh.orig`):

- On the Tufts RHEL hosts `which` is an *exported bash function*
  (`/usr/bin/which --tty-only --read-alias --read-functions ...`). The container
  inherits the function but its Ubuntu `/usr/bin/which` rejects those options. So
  `LARCV_PYTHON` / `LARCV_PYTHON_CONFIG` were set to the usage message, which
  broke the numpy check and the include/link flags. All `which` calls are now the
  `command -v` builtin.
- `[[ $@ -eq -q ]]` (an integer comparison) is now `[[ "$1" == "-q" ]]`.

The build produces about 450 harmless `-Wdeprecated-copy` warnings and links
`build/lib/liblarcv.so`. It was verified by reading every product in the test
file from Python.

---

## 2. What is in the larcv2 file

### 2.1 Space points: `sparse3d_reco` is ghost-included

The voxels are 0.3 cm on a 2048³ grid, x,y ∈ [-307.2, 307.2], z ∈ [-57.4, 557].
`sparse3d_reco` is the raw 3D reconstruction (SBND Cluster3D), **not** deghosted.

| entry | `reco` voxels | ghosts (label 5) | non-ghost |
|---|---|---|---|
| 0 | 236,385 | 216,475 (91.6%) | 19,910 |
| 1 | 86,472 | 75,489 (87.3%) | 10,983 |
| 2 | 74,056 | 58,126 (78.5%) | 15,930 |

| tree | voxel set | content |
|---|---|---|
| `sparse3d_reco` | all reco points (ghost-included) | reco charge = **mean** of the contributing hit charges (`reco/Σhit_charge` = 1/2 for 2-plane points, 1/3 for 3-plane points) |
| `sparse3d_reco_hit_charge{0,1,2}` | == reco | per-plane hit integral; -1 = no hit on that plane. A hit's full charge is repeated on every voxel built from it |
| `sparse3d_reco_hit_key{0,1,2}` | == reco | hit index per plane (-1 = none). **The hit table itself (wire, tick) is not in the file** |
| `sparse3d_reco_nhits` | == reco | 2 or 3 planes. **71-88% of points are 2-plane, and 95-97% of those are ghosts** (3-plane points: 17-69% ghost) |
| `sparse3d_reco_chi2`, `_occupancy` | == reco | reco quality (chi2 0-6; occupancy 1-4) |
| `sparse3d_reco_tpc`, `_cryo` | == reco | TPC 0/1; cryostat always 0 |
| `sparse3d_pcluster_semantics_ghost` | == reco | SPINE semantic per reco voxel: 0 shower, 1 track, 2 Michel, 3 delta, 4 low-energy, **5 ghost** |
| `sparse3d_reco_rescaled` | non-ghost subset | reco charge after truth-based ghost removal, with each hit's charge re-shared among the remaining points |
| `sparse3d_pcluster_semantics` | non-ghost subset | semantic label (0-4) on the non-ghost reco voxels |
| `sparse3d_pcluster_index` | non-ghost subset | index into `particle_corrected` (overlap-resolved owner of the voxel) |
| `sparse3d_pcluster` | non-ghost subset | true deposited energy [MeV] per reco voxel |
| `sparse3d_pcluster_leftover` | non-ghost subset (slightly fewer voxels) | not investigated |
| `sparse3d_sed`, `sparse3d_masked_true` | true positions (≈2% overlap with reco) | true energy deposits; `cluster3d_masked_true2reco` maps true → reco |

### 2.2 Truth

- `particle_corrected` (one entry per cluster in `cluster3d_pcluster`, same as
  `particle_pcluster`): Geant4 particles with track_id, pdg, parent/ancestor
  ids, shape, energy_init / energy_deposit [MeV], position, first_step /
  last_step / end_position, creation process, group_id, interaction_id,
  children_id. `pcluster_index` → `particle_corrected` was checked: voxel
  semantics agree with particle `shape`. `num_voxels()` counts the
  overlap-inclusive cluster size, so it differs from the `pcluster_index` count.
- **Origin is encoded in the track ID**: `track_id // 1e7` = 1 for neutrino
  generator particles, 2 for CORSIKA cosmics (consistent over all 3,700
  particles). Cosmics are simulated and fully labelled. (In the MicroBooNE
  overlay, real-data cosmics appear as `hasmatch=0, pid=0`.)
- `neutrino_mpv`: GENIE interactions (pdg, current_type 0 = CC / 1 = NC, mode,
  interaction_type, energy [**GeV**; particles are in MeV], vertex, time [ns]).
  `lepton_pdg` is **unfilled** (0) and `interaction_id` is **unset** (65535).
- **Several neutrinos per event, mostly outside the TPC:** 41 neutrinos in 21
  events, only 15 with the vertex inside the TPC (|x|,|y|<200, 0<z<500). Every
  event has ≥1 neutrino. Neutrino times are 0.37-1.9 µs. Entry 0's only neutrino
  is dirt, and only its muon enters the TPC.
- `particle_mpv`: GENIE-level particles (track_id 0, includes nuclear
  fragments). `particle_mpr`: CORSIKA primaries (several hundred per event).
- Neutrino-induced fraction of the non-ghost voxels per event: 0.1%-48% (median ≈7%).

### 2.3 Optical and CRT

- `opflash_tpc0` / `opflash_tpc1`: PMT flashes, **one producer per TPC**.
  `opflash_tpc0xa` / `opflash_tpc1xa`: X-ARAPUCA flashes. `PEPerOpDet` is
  **cut short** at 305-312 entries (SBND has 312 photon detectors), so it must be
  padded by channel index. `volume_id`, `onBeamTime`, `inBeamFrame` are **unset**.
  Flash times span ±1.4 ms; a beam flash (0-2 µs) is present in nearly every event.
  `xCenter` / `yCenter` / `zCenter`, widths and `fastToTotal` are available.
- `crthit_crthit`: 100-150 hits per event over ±1.7 ms. Each hit has position
  ± uncertainty, `ts0_ns` == `ts1_ns` (time relative to trigger, ns), and PE.
  **`plane` is constant within an event (0 or 7) and `tagger` is empty**, so the
  CRT wall must be derived from position: y = -381 (bottom), 625 / 763-764 (top
  layers), x = ±403, z = -195 / 779. 0-2 hits per event fall in [-0.5, 2.5] µs.

### 2.4 Event ID

`io.event_id()` run/subrun/event are **unfilled (2⁶⁴-1)** for every entry.

---

## 3. Field-by-field mapping to the LArFormer HDF5 schema

The MicroBooNE schema is in
[`../uboone_official/LARFORMER_DATAPREP.md`](../uboone_official/LARFORMER_DATAPREP.md).
"Reader" is the consumer in this repo; `pointcept/datasets/larformer.py` is
abbreviated `larformer.py`.

### 3.1 Direct one-to-one sources

| HDF5 field | larcv2 source | reader / notes |
|---|---|---|
| `triplet_data/pos` | `sparse3d_reco` voxel id → xyz (voxel centre, via meta) | larformer.py, required |
| `triplet_data/hasmatch` | `pcluster_semantics_ghost != 5` | deghost target; slice GT sets ghosts to -1 |
| `triplet_data/trackid` | `particle_corrected[pcluster_index].track_id()`, 0 for ghosts | required |
| `triplet_data/pid` | same particle's `pdg_code()` (raw PDG), 0 for ghosts | required |
| `triplet_data/origin` | `track_id // 1e7` (1 = ν, 2 = cosmic), 0 for ghosts | matches the larflow convention (1 = ν, 2 = cosmic, 0 = unmatched) |
| `triplet_data/pixval` (N,3) | `sparse3d_reco_hit_charge{0,1,2}`, with -1 → 0 | different quantity: see decision D2 |
| `triplet_data/edep` | `sparse3d_pcluster` (true MeV per voxel) | not read by anything today; an improvement over MicroBooNE |
| `mc_particle_tree/{trackid,parent_trackid,pid,energy_mev,start_pos}` | `particle_corrected` | `process_code` needs a string → code table (not read) |
| `mc_particle_tree/origin` | `track_id // 1e7` | slice GT uses this |
| `mc_particle_tree/nu_vertices` | `neutrino_mpv` positions | may hold several (D3) |
| `flashes/{pe,time_us,total_pe,flash_index}` | `opflash_*` | see D7 |

### 3.2 Missing from the file

| field | situation |
|---|---|
| `entry_0` attrs run/subrun/event | unfilled. The sidecar and ntuple exporter key on (run, subrun, event), so IDs must be synthesized uniquely (e.g. file hash + entry) |
| `uwire/vwire/ywire/tick` | no wire/tick values; `hit_key` points to a hit table that is absent. Must be computed from SBND geometry (2 TPCs with mirrored drift, U/V at ±60°, Y vertical, 0.3 cm pitch, drift velocity, tick period). Used for the wire feature, de-duplicated charge, LArPID |
| `image_data/*`, `triplet_imgpix_index` | no `image2d` trees. Only LArPID reads these |
| `pmt_positions` | not in file; needs an SBND 312-detector table (PMT coated/uncoated, X-ARAPUCA VUV/visible) |
| POT | not in file; needed for normalization |
| `lm_score` | write dummy 1.0 (read unconditionally) |
| `shower_fragments` | write an empty group (opened unconditionally) |
| `triplet_truth`, `aid`, `ssnet_boundary` | nothing reads them; can be skipped or left empty |

### 3.3 Decisions required

| # | decision | options / suggested default |
|---|---|---|
| D1 | **2-plane points** (71-88% of points, ≈96% ghost; MicroBooNE triplets are always 3-plane) | keep and store `nhits` (default), or drop (3-8× fewer points) |
| D2 | **`pixval` semantics.** MicroBooNE: ADC of a pixel covering 6 ticks. SBND: whole-hit integral shared by every voxel of the hit, up to ~5,000. The dataset's clip [0,1000] + log scale would saturate | per-detector rescale constant. `hit_key` gives an exact de-duplication key, better than `tick*1e5+wire` |
| D3 | **multiple / dirt neutrinos** (15 of 41 vertices in the TPC) | neutrino class = in-TPC-vertex interactions only, plus a dirt flag (default); alternatively all neutrinos, or a separate class. Link via `particle.interaction_id()` grouping + vertex-position match, because `neutrino.interaction_id()` is unset |
| D4 | **`ssnet_label`** (MicroBooNE 0-9: bg, e, γ, μ, p, π/K, Michel, delta, low-energy, other) | rebuild from SPINE shape + PDG (shower → e/γ by PDG; Michel/delta/low-energy from shape) |
| D5 | **keypoints** (`kpscores`, `mckeypoints`) | build from particles: ν vertex, track start (`first_step` vs `position`), track end (`last_step`), shower start, Michel, delta. Dirt vertices outside the TPC need a rule |
| D6 | **`start_pos_sce`** | `= start_pos` (default; labels already sit on reco voxels), or apply an SBND space-charge map |
| D7 | **flashes**: per-TPC PMT + X-ARAPUCA producers, beam flags unset | `producer_id`: tpc0 = 0, tpc1 = 1, xa = 2/3; beam flash = largest PMT flash in a 0-2 µs window; pad `PEPerOpDet` to 312 |
| D8 | **truth sidecar** | `trueNu*` from `neutrino_mpv` (GeV); `trueLepPDG` / `trueLepE` from `particle_mpv` primaries; `trueVtxInWCFV` needs an SBND fiducial volume; `truePrimPart` / `trueSimPart` from `particle_mpv` / `particle_corrected` |

---

## 4. Information worth adding (not in the MicroBooNE schema)

- **CRT hits** → new `entry_0/crthits` group (x, y, z, errors, `ts1_ns`, PE,
  wall derived from position). Uses: in-time CRT muon veto (0-2 hits per event in
  the beam window), and cosmic tagging / t0 by matching track endpoints to CRT hits.
- **Containment** → per particle (`first_step` / `last_step` / `end_position`
  inside the active volume) and per interaction (contained / partially contained),
  plus a dirt flag and the neutrino time.
- **Per-point reco-quality features** (`reco_chi2`, `reco_occupancy`,
  `reco_nhits`, `reco_tpc`): strong ghost discriminants, and the TPC id gives the
  drift direction across the cathode.
- **`reco_rescaled`** charge for calorimetry studies.
- **Grouping truth**: `interaction_id` (direct slice ground truth) and
  `group_id` / `parent_id` (shower grouping).
- **Flash extras**: X-ARAPUCA flashes, centre / width, `fastToTotal`.

---

## 5. Which merged_sp fields the LArFormer chain actually reads

- **Required by `LArFormerDataset`:** `triplet_data/{pos, pixval, trackid, pid, lm_score}`;
  the `shower_fragments` group must exist.
- **Optional:** `uwire/vwire/ywire` (wire feature, zeros if missing),
  `origin`, `hasmatch` (all 1 if missing), `ssnet_label`, `kpscores`,
  `mc_particle_tree` (an empty `trackid` means "data"), `mckeypoints`.
- **Reco chain:**
  - `tick` + wires (charge de-duplication, `trajfit/calo.py`)
  - `flashes/{pe, producer_id, total_pe, time_us}` (beam flash = `producer_id==0`, largest PE)
  - `run` (dead PMTs / gamma calibration)
  - `image_data` (LArPID only)
- **Never read:** `triplet_truth`, `edep`, `aid`, `ssnet_boundary`,
  `label_completed`, `pmt_positions` (hard-coded instead), flash `tpc_tick` /
  `y_center` / `z_center`.
- `larformer_stage12_cache.py` reads a derived cache built by
  `tools/larformer/build_stage12_cache_event.py`, not merged_sp directly.

PDG → class (larformer.py:86-93): e 0, γ 1, μ 2, π 3, p 4, other 5. Neutrons
are folded into protons; π0 / K0 are never instances. The KE thresholds are in
`slice_labels.py:228-236`.

---

## 6. MicroBooNE constants hard-coded in the chain (input to the detector-agnostic utility)

These have to move behind a per-detector constants interface before SBND
(or ICARUS) files can run through the chain. Paths are relative to the repo root.

| category | constant | locations |
|---|---|---|
| coordinate normalization | `coord_center=(125,0,518)`, `coord_scale=179.55` | `pointcept/datasets/larformer.py:78-79`, `larformer_stage12_cache.py:56-57`; 179.55 default in `pointcept/models/LArFormer/{inference.py, keypoint.py, losses.py, keypoint_query.py, model.py}` |
| wire count | feature scale `1/3456` | `larformer.py:136`, `transform.py:429`, `shower_clustering.py:103`, `shower_origin.py:124,739`, `lartpc.py:162` |
| planes | `pixval` assumed (N,3); Y-else-mean(U,V) charge | `flash_predict.py:120-127`, `trajfit/calo.py:38` |
| TPC active volume | x[0,256.35], y[±116.5], z[0,1036.8] | `flash_chi2.py:40-42`, `eval_reco_performance.py:195-196`, `flashmatch_quality.py:56-57`, `export_gen2ntuple.py:349`, stage3 inference `:92` |
| fiducial volume | WireCell FV via ctypes | `export_gen2ntuple.py:99-108` |
| image geometry | 1008 rows × 3456 columns, row = (tick-2400)/6 | `larpid/crop.py:18,30,81`, `viz/scan_event_html.py:47` |
| electronics clock | 0.5 µs/tick, trigger tick 3200, image start tick 2400 | `prepare_flashinfo_h5.py:72-79`, converter |
| LAr / drift | drift velocity 0.1098 cm/µs | `flash_predict.py:47` |
| light model | fast fraction 0.3, τ_slow 1.6 µs, cosmic window 0.3125 µs | `flash_predict.py:43-46` |
| photon detectors | 32 PMTs, (Nf,32) arrays, positions from larutil | `flash_chi2.py:35`, `saturation.py:31,34-49`, `prepare_flashinfo_h5.py:82`, keypoint2 cascade `:471` |
| photon library | TPC origin `(-1.55, …, 0.1)` | `build_photonlib_cache.py:32` |
| flash producers | 0 = simpleFlashBeam, 1 = simpleFlashCosmic | `flash_predict.py:21-22` |
| run periods / calibration | run boundaries 7771/13697/18961/22270, opdet 15 dead runs 3-5, gamma 5.25 + per-period table | `dead_channels.py:32-57`, `flash_calib.py:24-31` |
| space charge | `SpaceChargeMicroBooNE`, map range x[0,256], y[±116.5] | `sce_microboone.py:10-11`, larflow C++ |
| larflow C++ truth maker | keypoint σ 3 cm (threshold 0.01), MCC9 edep thresholds in ADC, low-energy cut 15 points | `ubdl/larflow/larflow/PrepFlowMatchData/SimChTripletLabelMaker.cxx:175-187,681` |
| charge de-duplication | pixel key base `_WBASE=100000` | `trajfit/calo.py:20` |

---

## 7. Next steps

1. **Detector-agnostic constants utility** (next; design input from the user's
   earlier detector-agnostic larlite work). Covers geometry (TPCs, planes, wire
   angles / pitch / counts, active and fiducial volumes, photon-detector
   positions and types, CRT walls), space charge, LAr properties (drift velocity
   vs field, light yield / timing), and electronics clocks (tick period, trigger
   offsets). Values are extracted once from the official repositories (possibly
   through larsoft / gallery in a cvmfs container) and cached for SBND,
   MicroBooNE and ICARUS.
2. Port the §6 constants in the LArFormer chain to that utility.
3. Write `convert_larcv2_to_larformer_h5.py` using the utility, with the
   decisions D1-D8 settled.
