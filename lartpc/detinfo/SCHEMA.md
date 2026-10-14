# detinfo HDF5 schema (v1.1)

One file per (experiment, code version):
`$DETINFO_ROOT/<det>/<product>_<version>/detinfo_<det>_<version>.h5`
(default `DETINFO_ROOT=/cluster/tufts/wongjiradlab/larbys/detinfo`). It is written by
[`extract/extract_larsoft_detinfo.py`](extract/extract_larsoft_detinfo.py), which queries
the experiment's own LArSoft service providers. Inspect it with
[`dump_tables.py`](dump_tables.py).

Next to the file:
- `spacecharge/<map>.root`: verbatim copy of the SCE map. Its contents are also stored as arrays in `/spacecharge/histograms`.
- `aux/`: verbatim copies of experiment files that LArSoft geometry does not expose (e.g. `sbnd_pds_mapping.json`, the photon-detector type map).
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
- Use: unit tests of the python reimplementation.

## Changelog
- 1.1: `/validation`; SCE config read from the dumped services FHiCL.
- 1.0: initial.
