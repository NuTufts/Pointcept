# detinfo HDF5 schema (v1.4)

One file per (experiment, code version):
`$DETINFO_ROOT/<det>/<product>_<version>/detinfo_<det>_<version>.h5`
(default `DETINFO_ROOT=/cluster/tufts/wongjiradlab/larbys/detinfo`). It is written by
[`extract/extract_larsoft_detinfo.py`](extract/extract_larsoft_detinfo.py), which queries
the experiment's own LArSoft service providers. Inspect it with
[`dump_tables.py`](dump_tables.py).

Next to the file:
- `spacecharge/<map>.root`: verbatim copy of the SCE map. Its contents are also stored as arrays in `/spacecharge/histograms`.
- `aux/`: verbatim copies of experiment files that LArSoft geometry does not expose (e.g. `sbnd_pds_mapping.json`, the photon-detector type map).
- `photonlib/<library>.root` + `photonlib_<det>_<version>.h5`: verbatim copy of the photon
  visibility library and its dense conversion (see "Photon library file" below). These are large
  (~130 MB + the converted file for SBND) and are not meant for git.
- `extract.log`: the extraction log.

## Conventions

- **Units:** lengths cm, times µs, E-field kV/cm, temperature K, density g/cm³,
  angles rad, frequencies MHz. Coordinates are the experiment's LArSoft world frame.
- **IDs:** zero-based LArSoft indices `cryo`, `tpc`, `plane`, `wire`. `channel` is the
  readout channel. `opdet` is the optical-detector index and `opchannel` the optical
  readout channel.
- **Tables:** a *table* is a group whose datasets all have the same first dimension
  `n_rows` (a group attribute). A column of shape `(N,3)` is an xyz vector, unless it is
  per-plane (`(N, n_planes)`). A `bbox` column is
  `(N,6) = [xmin, ymin, zmin, xmax, ymax, zmax]`.
- **Scalars** are stored as group attributes (`@name`).
- **Enums** are stored as ints, with the mapping in an attribute:
  - `view`: `geo::View_t` 0 = kU, 1 = kV, 2 = kZ (= kW), 3 = kY, 4 = kX, 5 = k3D, 6 = kUnknown
  - `signal_type`: `geo::SigType_t` 0 = kInduction, 1 = kCollection

## Root attributes

| attr | meaning |
|---|---|
| `schema_version` | this document's version |
| `detector` | `sbnd` / `icarus` / `uboone` (our key) |
| `geometry_name` | LArSoft `GeometryCore::DetectorName()`, e.g. `sbndv2` |

## `/provenance`

| item | meaning |
|---|---|
| `@experiment_product`, `@experiment_version`, `@qualifiers` | e.g. `sbndcode`, `v10_14_02`, `e26:prof` |
| `@product_versions_json` | JSON of every relevant UPS product version (larsoft, larcorealg, lardataalg, sbnd_data, root, ...) |
| `@services_fcl`, `@service_table` | the FHiCL configuration the providers were built from |
| `@gdml_file` | GDML path used by the geometry |
| `@extracted_utc`, `@host`, `@container`, `@python` | where and when the extraction ran |
| `@extractor`, `@extractor_md5`, `@extractor_git_commit` | extractor script identity. The commit gets `+dirty` or `+untracked` if the script doesn't match HEAD |
| `@aux_files_json` | `[{source, copied_to, md5}]` for files in `aux/` |
| `@failed_queries_json` | `{query_key: error}`; any provider call that raised. Empty = complete extraction |
| `services_fhicl` | (string dataset) the full `fhicl-dump` of the services table |

## `/geometry`

### `/geometry/detector` (attributes)
`name`, `gdml_file`, `n_cryostats`, `n_tpcs_total`, `n_opdets`, `n_channels`,
`n_opchannels`, `world_bbox`, `enclosure_bbox`, `units_length`.

### `/geometry/cryostats` (table, one row per cryostat)
`cryo`, `bbox`, `center`, `n_tpcs`, `n_opdets`.

### `/geometry/tpcs` (table, one row per (cryo, tpc))
| column | meaning |
|---|---|
| `cryo`, `tpc` | ID |
| `bbox` | full TPC volume |
| `active_bbox` | active (sensitive) volume |
| `center`, `active_center` | volume centres |
| `drift_dir` | unit drift direction (electrons drift towards the anode). SBND TPC 0 = -x, TPC 1 = +x |
| `drift_distance` | cathode-to-anode distance |
| `cathode_center` | centre of the cathode face |
| `n_planes` | readout planes in this TPC |

### `/geometry/planes` (table, one row per (cryo, tpc, plane))
| column | meaning |
|---|---|
| `cryo`, `tpc`, `plane` | ID. `plane` 0 is the outermost plane (first crossed by drifting electrons) |
| `view`, `signal_type` | enums (see Conventions) |
| `n_wires`, `wire_pitch` | wire count and perpendicular wire spacing |
| `theta_z` | wire angle w.r.t. +z (rad). **U/V angles differ between TPCs** in SBND (TPC 0: U = 30°, V = 150°; TPC 1: U = 150°, V = 30°) |
| `center`, `normal_dir` | plane centre and normal (pointing into the TPC) |
| `wire_dir` | unit vector along the wires |
| `increasing_wire_dir` | unit vector in the plane, perpendicular to the wires, pointing towards increasing wire number |
| `first_wire_center`, `last_wire_center` | centres of wire 0 and wire n−1 |
| `first_channel` | channel of wire 0 |
| `width`, `depth`, `plane_bbox` | plane extent |

The continuous wire coordinate of a point P on this plane is
`(P − first_wire_center) · increasing_wire_dir / wire_pitch`. This reproduces
`PlaneGeo::WireCoordinate`; check it against `/validation`.

### `/geometry/wires` (table, one row per wire; 11,276 for SBND)
`cryo`, `tpc`, `plane`, `wire`, `channel` (int32); `start`, `end`, `center` (float32 xyz).
Rows are ordered by (cryo, tpc, plane, wire). **A channel may appear on several rows**
(ICARUS induction-1 wires are split; also detectors whose wrapped wires share a channel).

### `/geometry/opdets` (table, one row per optical detector)
| column | meaning |
|---|---|
| `opdet`, `cryo`, `center` | ID and position |
| `shape` | ROOT TGeo shape class: `TGeoSphere` (SBND PMT) or `TGeoBBox` (SBND X-ARAPUCA) |
| `dx`, `dy`, `dz` | half-sizes (bounding box of the shape) |
| `rmax` | radius for spheres and tubes (NaN otherwise) |
| `volume_name` | GDML volume, e.g. `volOpDetSensitive`, `volOpDetSensitiveXArapucaFilter` |

The PMT coated / uncoated and X-ARAPUCA VUV / visible distinction is **not** in the LArSoft
geometry. For SBND it is in `aux/sbnd_pds_mapping.json`.

### `/geometry/opchannels` (table)
`opchannel` → `opdet` (`WireReadoutGeom::OpDetFromOpChannel`).

### `/geometry/auxdets`, `/geometry/auxdet_sensitive` (tables; CRT)
- `auxdets`: one row per module: `auxdet`, `name` (GDML volume, e.g. `volAuxDetCRTStripArray92`), `center`, `half_width1`, `half_width2`, `half_height`, `length`, `normal`, `n_sensitive`, and `axis_x`, `axis_y`, `axis_z` (the module's local axes in world coordinates).
- `auxdet_sensitive`: one row per strip: `auxdet`, `sensitive`, and the same geometry columns.
- SBND: 146 modules, 2,360 strips. The wall each module belongs to is not labelled; derive it from position.

## `/clocks` (attributes; `DetectorClocksData::DataForJob()`)
- Per clock C ∈ {`TPCClock`, `OpticalClock`, `TriggerClock`, `ExternalClock`}:
  `C_TickPeriod_us`, `C_Frequency_MHz`, `C_FramePeriod_us`, `C_FrameTicks_ticks`.
- Plus `TriggerTime_us`, `BeamGateTime_us`, `TriggerOffsetTPC_us`,
  `G4ToElecTime_at_g4t0_us`, `TPCG4Time2Tick_at_g4t0`, `TPCTick2TrigTime_at_tick0_us`,
  `TPCTick2Time_at_tick0_us`.
- These are the job-default values (MC configuration). Data runs may override trigger times per event.

## `/detector_properties` (attributes; `DetectorPropertiesData`)
`Temperature_K`, `ElectronLifetime_us`, `Density_g_cm3`, `DriftVelocity_cm_us`,
`Efield_kV_cm` (gaps 0, 1, 2: drift region, then inter-plane gaps), `NumberTimeSamples`,
`ReadOutWindowSize`, `ElectronsToADC`.

### `/detector_properties/x_to_ticks` (table, one row per plane)
`tick = ticks_per_cm · x + tick_at_x0` (`ConvertXToTicks`). `x_at_tick0` =
`ConvertTicksToX(0)`; `tick_at_plane` is the tick of a charge arriving at the plane.
The sign of `ticks_per_cm` follows the drift direction.

## `/lar_properties` (attributes; `LArPropertiesStandard`)
- Scalars: `RadiationLength`, `AtomicNumber`, `AtomicMass`, `ExcitationEnergy`,
  `Argon39DecayRate`, `ScintYield`, `ScintPreScale`, `ScintYieldRatio`,
  `ScintFastTimeConst`, `ScintSlowTimeConst` (ns), `ScintBirksConstant`,
  `ScintResolutionScale`, `ScintByParticleType`, per-particle `*ScintYield` / `*ScintYieldRatio`,
  `CerenkovLightEnabled`, `ExtraMatProperties`, `TpbTimeConstant`.
- `/lar_properties/spectra/<name>/{energy_eV, value}` for `FastScintSpectrum`,
  `SlowScintSpectrum`, `RIndexSpectrum`, `AbsLengthSpectrum`, `RayleighSpectrum`, `TpbAbs`, `TpbEm`.

## `/spacecharge`
- **Attributes:** the flat values of the services SCE table (`InputFilename`,
  `RepresentationType`, `EnableSim/Cal{Spatial,Efield}SCE`, `EnableCorrSCE`,
  `is2DdriftSimHack`, `service_provider`, ...), plus:
  - `fhicl_key`: which services key held them (SBND: `SpaceCharge`)
  - `fhicl`: block text
  - `source_path`, `copied_to`, `md5`: of the map file
- **`/spacecharge/histograms/<name>`:** every TH1/2/3 in the map file, as float32
  arrays indexed `[ix, iy, iz]` (bin 1 → index 0; under/overflow dropped). Attributes:
  `class`, `title`, `axis_titles`, `edges_x`, `edges_y`, `edges_z` (n+1 bin edges each).
- **SBND map** (`Voxelized_TH3`, 49×49×61): `True{Fwd,Bkwd}_Displacement_{X,Y,Z}_{E,W}`
  and `True_ElecField_{X,Y,Z,Mag}_{E,W}`. `E`/`W` are the east/west drift volumes
  (separate maps per TPC). `Fwd` maps true → reco position (simulation); `Bkwd` maps
  reco → true (correction). The coordinate convention of the map axes (e.g. mirroring
  per TPC) is defined in `SpaceChargeSBND.cxx`; the python utility must reproduce it.

## `/photon_visibility` (attributes)
- Contents: the scalar values of the services `PhotonVisibilityService` table:
  - `LibraryFile`
  - the grid: `NX`, `NY`, `NZ`, `XMin`/`XMax`, `YMin`/`YMax`, `ZMin`/`ZMax`
  - flags: `StoreReflected`, `IncludePropTime`, `DoNotLoadLibrary`, `UseCryoBoundary`
  - semi-analytic model scalars
- The array parameters of the semi-analytic model (`Cut_off`, `Distances_*`, `Mpv`, ...)
  are only in `/provenance/services_fhicl`.
- Library copy and conversion: `source_path`, `copied_to`, `md5`, `tree_cycle` (key
  cycle read, the highest, as LArSoft does), `tree_entries`, `tree_branches`,
  `n_opchannels_in_library`, `visibility_nonzero`, `refl_visibility_nonzero`, and
  `converted_h5` (file name of the photon library file).

## Photon library file `photonlib_<det>_<version>.h5`
- Attributes: as in `/photon_visibility`, plus:
  - `grid_min` / `grid_max` (xyz, cm): the **effective** voxel grid. With `UseCryoBoundary` (MicroBooNE)
    it is the cryostat box, as LArSoft's PhotonVisibilityService uses, and X/Y/ZMin/Max are ignored;
    `grid_source` says which
  - `grid_n` (NX, NY, NZ)
  - `layout`
- `visibility`: float32 `[ix, iy, iz, opchannel]`, the fraction of isotropically emitted
  photons from voxel (ix, iy, iz) that reach the optical channel.
- `refl_visibility`: the same for reflected (visible) light, when the library stores it.
- Voxel indices follow LArSoft `sim::PhotonVoxelDef`:
  `ix = floor((x - XMin) / ((XMax - XMin) / NX))` (same for y, z); points outside the grid
  have no voxel (`-1`); voxel id = `ix + iy*NX + iz*NX*NY`.
- **SBND (`SBND_OpLibOUT_v2.00.root`) is an "outside" library: it has no entries inside the
  TPC active volumes** (verified: 0 of 2,000 active-region voxels populated in an x-slice through
  the TPC). SBND simulates light from inside the active volume with larsim's semi-analytic model,
  so in-TPC flash prediction needs that model (not yet ported); the library only covers light
  produced outside the active volume.
- Zero entries of the sparse LArSoft tree are zeros here. The layout matches the MicroBooNE
  `lartpc/flashmatch` cache (`(Nx, Ny, Nz, NOpDets)`); `PhotonLibrary.export_flashmatch_npz()`
  writes the npz read by `PhotonLibLookup`.

## `/optical_model` (light model)
Inputs of the semi-analytic light model and the experiment's flash matcher. These come from FHiCL
tables **outside** the services configuration.
- **`/optical_model/<name>/`**, one per entry in the extractor's `optical_fcl_tables`. SBND has:

  | name | fcl : table |
  |---|---|
  | `vuv_hits` | `opticalsimparameterisations_sbnd.fcl : sbnd_vuv_RS100cm_hits_parameterization` |
  | `vis_hits` | `… : sbnd_vis_RS100cm_hits_parameterization` |
  | `pdfastsim_par` | `PDFastSim_sbnd.fcl : sbnd_pdfastsim_par` (semi-analytic simulation, in the active volume) |
  | `pdfastsim_pvs` | `… : sbnd_pdfastsim_pvs` (library simulation, outside the active volume) |
  | `flashmatch` | `opt0finder_sbnd.fcl : sbnd_opt0_finder_one_to_many` (efficiencies, skip list, charge calibration, QLL settings) |

  Each has:
  - `@fcl`, `@table`
  - `fhicl`: the resolved table text (`fhicl-dump`)
  - `params/`: the same table as HDF5. Sub-tables are groups, rectangular numeric sequences are
    datasets (e.g. `vuv_hits/params/GH_PARS_flat` (4, 9), `vis_hits/params/VIS_correction_flat`
    (k, r, x)), scalars and strings are attrs, ragged or mixed sequences are `<key>__json` attrs.
    Examples: `flashmatch/params/FlashMatchConfig/PhotonLibHypothesis/VUVEfficiency` (312,),
    `flashmatch/params/OpChannelsToSkip`, `flashmatch/params/CalAreaConstants`.
- **`/optical_model/semianalytic`**: geometry as built by larsim's `SemiAnalyticalModel` itself.
  - attrs:
    - `DoReflectedLight`, `IncludeAnodeReflections`, `UseXeAbsorption` (from `pdfastsim_par`)
    - `optical_path_class` (e.g. `phot::SBNDOpticalPath`: a detector sees only its own side of the cathode)
    - `vuv_absorption_length` (cm; AbsLengthSpectrum at 9.7 eV)
    - `cathode_centre`, `anode_centre`, `drift_distance`, `reference_source`
  - `active_volumes` (n, 6): the `ISTPC` active volumes.
  - `opdets` (table): `opdet`, `type` (0 rectangle = X-ARAPUCA/bar, 1 dome PMT, 2 disk PMT),
    `orientation` (0 faces x, 1 faces y, 2 faces z), `h`, `w` (cm; rectangles only, -1 otherwise),
    exactly as `SemiAnalyticalModel::opticalDetectors()` derives them from `OpDetGeo`.

## `/validation_optical` (table)
- Contents: LArSoft reference values of the semi-analytic model for the python port.
- Columns:
  - `point` (N, 3)
  - `direct` (N, n_opdets): `detectedDirectVisibilities`
  - `reflected` (N, n_opdets): `detectedReflectedVisibilities(AnodeMode=false)`, NaN if not configured
- Points:
  - 150 random per TPC active volume
  - line scans: x through the centre, and z and y at x = ±half drift
  - 40 points 5–60 cm inside the active volume in front of randomly chosen detectors
- Edge case kept on purpose: the x-scan point exactly on the cathode (x = 0) gets **NaN** reflected
  visibilities from LArSoft (distance to the cathode is 0, so 0/0). The port must reproduce it,
  and comparisons use NaN == NaN.
- The values come from larsim's own `SemiAnalyticalModel.cxx`, compiled with cling from a patched
  copy (`extract/larsoft_ref/semianalytic_ref.py`). Only the constructor's service lookups are replaced
  (geometry, wire readout, LAr absorption spectrum passed in); the physics code is unmodified.

## `/validation` (table)
- Contents: LArSoft's own answers for random points (fixed seed), 100 per TPC inside the
  active volume plus 20 anywhere in the detector enclosure.
- Columns:
  - `point`: the sampled xyz position
  - `cryo`, `tpc`: the containing TPC (`FindTPCAtPosition`; -1 = outside)
  - per plane `p` of that TPC:
    - `wire_coordinate[:,p]`: `PlaneGeo::WireCoordinate`
    - `nearest_wire[:,p]`: `NearestWireID`; -1 if off the plane
    - `tick[:,p]`: `ConvertXToTicks(x)`
  - from the experiment's SCE provider, built with **all** `Enable{Sim,Cal}{Spatial,Efield}SCE`
    flags on so that both maps are validated (NaN if the provider could not be built):
    - `sce_pos_offsets`: `GetPosOffsets`
    - `sce_efield_offsets`: `GetEfieldOffsets`
    - `sce_cal_pos_offsets`: `GetCalPosOffsets(point, sce_cal_tpcid)`
  - `photonlib_voxel_id`: `sim::PhotonVoxelDef::GetVoxelID` (-1 outside the grid, -2 not computed)
- Points: 100 per TPC inside the active volume, 20 anywhere in the detector enclosure, and
  80 edge points (40 in the active-volume union grown by 6 cm, 40 in a slab |x| < 4 cm around
  the cathode) that exercise the SCE clamping.
- Use: unit tests of the python reimplementation (`tests/test_validation.py`).

## Changelog
- 1.4: MicroBooNE support. Every `*Filename` SCE map is stored (non-primary maps under
  `/spacecharge/histograms/<key>/`, e.g. `CalibrationInputFilename/`). The SCE reference provider is built
  with all maps enabled. The photon-library grid follows `UseCryoBoundary`.
- 1.3: `/optical_model` (light-model FHiCL tables, semi-analytic descriptors) and `/validation_optical`.
- 1.2: `/photon_visibility` + photon library file; SCE and voxel-id reference values and edge
  points in `/validation`; shuffle + gzip-9 compression.
- 1.1: `/validation`; SCE config read from the dumped services FHiCL.
- 1.0: initial.
