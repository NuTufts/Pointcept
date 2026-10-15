# detinfo: detector-agnostic detector constants for SBN experiments

Goal: one source of geometry, electronics-clock, detector-property, LAr-property
and space-charge constants for **SBND, MicroBooNE and ICARUS**. LArFormer and the
data converters use it in place of hard-coded MicroBooNE numbers (inventory in
[`../data_prep/sbnd_official/README.md`](../data_prep/sbnd_official/README.md) §6).

The constants are extracted **once per experiment code version** from the official
LArSoft service providers (cvmfs), and cached in a portable HDF5 file. Everything
downstream is python-only (numpy + h5py), with no ROOT / LArSoft / cvmfs at use time.

| piece | status |
|---|---|
| [`extract/run_extract.sh`](extract/run_extract.sh), [`extract/extract_larsoft_detinfo.py`](extract/extract_larsoft_detinfo.py) | done for all three: sbndcode v10_14_02, uboonecode v10_04_07_13, icaruscode v10_25_00 |
| [`SCHEMA.md`](SCHEMA.md), [`dump_tables.py`](dump_tables.py) | done (schema 1.5) |
| python utility: `Geometry`, `DetectorClocks`, `DetectorProperties`, `LArProperties`, `SpaceCharge` (SBND, MicroBooNE, ICARUS ports), `PhotonLibrary` (incl. ICARUS cryostat mapping), `Conventions` | done; **all checks vs LArSoft reference pass** for all three detectors ([`tests/test_validation.py`](tests/test_validation.py)) |
| plotly 3D geometry review page ([`viz/make_detinfo_html.py`](viz/make_detinfo_html.py)) | done |
| SBND in-TPC light: semi-analytic model (the SBND library covers only outside the active volume) | done in `lartpc/flashmatch/model/visibility/semianalytic.py` |
| MicroBooNE extraction | **uboonecode v10_04_07_13 done** (8/8 checks incl. a python SpaceChargeMicroBooNE port); v08_00_00_13 compared at the configuration level (section below) |
| ICARUS extraction | **icaruscode v10_25_00 done** (9/9 checks; section below) |
| fold MicroBooNE SCE / photonlib (lartpc/flashmatch) into this package | later |
| export back to larlite ROOT trees | later |

## Extracting

```bash
bash lartpc/detinfo/extract/run_extract.sh sbnd v10_14_02            # quals default e26:prof
# -> /cluster/tufts/wongjiradlab/larbys/detinfo/sbnd/sbndcode_v10_14_02/
#      detinfo_sbnd_v10_14_02.h5, spacecharge/*.root, aux/*, extract.log
```

How it works on the Tufts cluster:
- Runs the SL7 container `cvmfs_containers/uboone-devel-sl7-11` with cvmfs mounted by
  the EL7 `singcvmfs` dist, because sbndcode v10_14_02 is SL7-only.
- Mounts the sbn, sbnd, larsoft and fermilab repos. **`sbn.opensciencegrid.org` is required.**
- Sets up the experiment product and imports `SBNDservices` (the same service loader the
  TITUS event display uses) to build the real providers.
- The first run on a node takes about 7–10 min, mostly filling the node-local cvmfs cache.
- h5py for the LArSoft python 3.9 is an unpacked manylinux2014 wheel in
  `/cluster/tufts/wongjiradlab/larbys/detinfo/py39_site`.
- Each provider query is wrapped: failures are listed in `/provenance@failed_queries_json`
  and printed at the end, instead of aborting the run.

Installing h5py for the LArSoft python (once): `bash lartpc/detinfo/extract/install_py39_h5py.sh`
(the wheel goes to the shared area, never into the repo).

## Using

```python
from lartpc.detinfo import load
di = load("sbnd")                               # newest cached version; or load("sbnd", "v10_14_02")
print(di.summary())
g = di.geometry                                 # larlite-style names, (plane, tpc=0, cryo=0) order
g.NTPCs(); g.Nwires(2, tpc=1); g.GetPlane(0, tpc=1).WireCoordinate(points)   # points: (3,) or (N,3)
g.FindTPC(points); g.ChannelToWireID(ch); g.PlaneWireToChannel(plane, wire, tpc)
di.project(points)                              # TPC id + wire coordinate / nearest wire / tick per plane
di.detprop.ConvertXToTicks(x, plane, tpc); di.clocks.TPCClock().TickPeriod
di.sce.GetPosOffsets(points); di.sce.GetCalPosOffsets(points, tpcid)
di.photonlib.GetVisibility(points)              # lazy-loads photonlib_<det>_<ver>.h5
di.conventions.opdet_types(); di.conventions.crt_walls()
```

Checks and review page:
```bash
python lartpc/detinfo/tests/test_validation.py --detector sbnd
python lartpc/detinfo/viz/make_detinfo_html.py --detector sbnd -o sbnd.html [--crt-strips]
python lartpc/detinfo/viz/make_detinfo_html.py --detector sbnd --wires 0,1,0 [--wire-stride 4] -o wires.html
```

## Inspecting

```bash
python lartpc/detinfo/dump_tables.py FILE.h5                     # overview
python lartpc/detinfo/dump_tables.py FILE.h5 -g geometry/planes
python lartpc/detinfo/dump_tables.py FILE.h5 -g geometry/wires --where "(tpc==1)&(plane==2)" --head 5
python lartpc/detinfo/dump_tables.py FILE.h5 -a clocks
python lartpc/detinfo/dump_tables.py FILE.h5 -t provenance/services_fhicl
```

## MicroBooNE notes (2026-10-15)

```bash
bash lartpc/detinfo/extract/run_extract.sh uboone v10_04_07_13      # e26:prof, ~7 min
```

**How the providers are built.** uboonecode has no python service loader (no `UBooNEservices`,
no `LArSoftUtils` on its `PYTHONPATH`). `extract/larsoft_ref/providers.py` builds the providers with
cling, from the dumped `microboone_simulation_services` FHiCL:
- larcorealg's standalone geometry setup
- `geo::WireReadoutUBooNEGeom`
- lardataalg's provider test helpers for clocks, detector properties and LAr properties

Gotchas handled in the code:
- range-v3's `<span>` include must be disabled for cling's C++17 modulemap
  (`RANGES_WORKAROUND_MSVC_UNUSABLE_SPAN`).
- Every UPS `*_INC` directory goes on the include path (GSL for the SCE header).
- `SpaceChargeMicroBooNE` needs `TH1::AddDirectory(false)`: it closes its TH3 file, which deletes
  the histograms otherwise.
- The photon library sits under `pmtresponse/`.
- Its voxel grid follows `UseCryoBoundary`, i.e. the cryostat box, not X/Y/ZMin/Max.

**Validation** (`tests/test_validation.py --detector uboone`, 8/8):
- TPC lookup, wire coordinate, nearest wire and x→tick are exact
- the python `SpaceChargeMicroBooNE` port matches the LArSoft provider to ~1e-14: forward map
  (`SCEoffsets_dataDriven_combined_fwd_Jan18`), backward map (`..._bkwd_Jan18`) and E field
  (× `EfieldOffsetScale` 3.65)
- photon-library voxel ids agree

**Frame: the MicroBooNE pipeline (larlite/larcv, `merged_sp`, `lartpc/flashmatch`) is NOT the
LArSoft world frame.**
- detinfo (LArSoft world) active volume: x [−1.55, 254.8], y [−115.53, 117.47], z [0.1, 1036.9].
- Pipeline active volume: x [0, 256.35], y ±116.5, z [0, 1036.8].
- So pipeline = world − (−1.55, 0.97, 0.10). This is the "TPC origin" that
  `lartpc/flashmatch/build_photonlib_cache.py` subtracts.
- Any MicroBooNE use of detinfo positions must apply this offset.

**Photon library.**
- The official config loads `uboone_photon_library_v6_70kV.root` (`uboone_photon_propagation`).
- Our flash model uses `ublarcvapp/UBPhotonLib/dat/uboone_photon_library_v6_70kV_EnhancedExtraTPCVis.root`.
- Same grid and channel order, but the values differ:
  - outside the active volume, 711,919 of 1,485,000 voxels differ and total visibility is +49% (6,779 vs 4,565)
  - inside, 50,580 of 765,000 voxels differ and total visibility is +6%
- The calibrated MicroBooNE gamma belongs to the *enhanced* library.

**v08_00_00_13 (the overlay release) vs v10_04_07_13.** v08 runs larsoft v08 under python 2.7 and the
older geometry API, so the extractor does not run there. Comparing the resolved services FHiCL and file md5s:
- **Identical:** GDML (`microboonev12.gdml`), both SCE map files, the photon library file, and the
  `DetectorClocksService`. `DetectorPropertiesService` has the same E field, temperature, lifetime and ADC scale.
- **New in v10:**
  - `IncludeInterPlanePitchInXTickOffsets: true` and `UseIcarusMicrobooneDriftModel: false` (DetectorProperties)
  - `EfieldOffsetScale: 3.65` (SpaceCharge; v08 code default 1.0, i.e. E-field distortions 3.65× smaller in v08)
  - the CRT aux-det initializer/sorter (76 modules × 16 strips)
  - semi-analytic timing tables (PhotonVisibilityService)
- **Changed:** the LAr optical spectra (e.g. absorption length 2000 → 8000 cm), which affect only optical simulation.
- So for geometry, wires, SCE positions and the photon library, the v10 extraction applies to the v08 overlay.
  The v08 x↔tick (inter-plane pitch) and E-field SCE scale would need a check before relying on them.

## ICARUS notes (2026-10-15)

```bash
bash lartpc/detinfo/extract/run_extract.sh icarus v10_25_00         # e26:prof, ~11 min, ~18 GB RSS (photon library)
```
Any icaruscode release with an `slf7` build can be extracted the same way (on cvmfs: v09_60 … v10_25_00;
v10_25_00 is the newest, with larsoft v10_24_00, icarus_data v10_06_06).

**How the providers are built.** icarusalg ships `ICARUSservices.py`, so the SBND path is used
(`ServiceManager` with `WireReadoutICARUS`). The services table is `icarus_g4_services`
(`services_icarus_simulation.fcl`): the common services plus `PhotonVisibilityService`.
fhiclpy is not set up by icaruscode; the extractor parses `fhicl-dump` output itself instead.

**What it holds** (geometry `icarus_v4`, `icarus_refactored_nounderscore_20230918.gdml`):
- 2 cryostats × 4 TPCs. Each drift volume is split at z = 0 into two logical TPCs (the refactored
  geometry), so TPC (c, 0/1) and (c, 2/3) share an x range. Cathodes at x = ±210.2 cm.
  Active x: C0 [−358.49, −210.29] ∪ [−210.14, −61.94], C1 mirrored; y [−181.86, 134.96]; z ±894.95.
- Planes per TPC: 0 = first induction (horizontal wires, view kY, 1056 wires per z half),
  1 = second induction (kV, 3064), 2 = collection (kU, 3064); 0.3 cm pitch; 55,296 channels.
- 360 PMTs (180 per cryostat, at x = ±371.1 and ±49.4), 305 CRT aux dets.
- 2.5 MHz TPC clock (0.4 µs), 4096 samples, `TriggerOffsetTPC` −340 µs.
- E field 0.4938 kV/cm with `DriftVelFudgeFactor` 0.99733 → drift velocity 0.157144 cm/µs; T 87.5 K;
  lifetime 3 ms.

**Space charge** (`SCEoffsets_ICARUS_E500_voxelTH3.root`, a python `SpaceChargeICARUS` port). One map
in |x| serves all four drift volumes; the provider applies the signs per volume.
- The services table has every `Enable*` flag **false**. SCE jobs (`larg4_icarus*_sce.fcl`) add
  `enable_spacecharge_icarus.fcl`: same map, `is2DdriftSimHack: false`.
- **Upstream quirk:** the provider reads `pset.get<bool>("is2DdriftSimHack", "false")`. The string default
  converts to `true`, so the services table (no key) runs **with** the 2D-drift hack: the forward x
  offset gets the sign −(cryostat side) everywhere. The port follows the extracted configuration;
  `SpaceChargeICARUS(h5, is2DdriftSimHack=False)` gives the `enable_spacecharge_icarus.fcl` behaviour.
- Forward, backward (TPC ids 0–3, all cathode clamps) and E-field offsets match LArSoft **exactly**.
  This needed ROOT's own fixed-bin arithmetic in `Hist3D`: the clamp y = 134.96 is exactly the last bin
  centre, and edge-averaged centres put it on the other side of ROOT's test. The SBND and MicroBooNE
  ports are now exact (0 difference) too.

**Photon library** (`PhotonLibrary-20201209.root`, 253 MB; dense h5 616 MB).
- It covers **cryostat 0 only**: grid x [−395, −25], y [−215, 170], z ±985; 74 × 77 × 394 voxels;
  180 channels, fully populated (233.6 M entries); no reflected-light table.
- LArSoft serves cryostat 1 through `ICARUSPhotonMappingTransformations`: a point with x > 0 is
  **translated** (not mirrored) by −420.43 cm in x, and library channel i becomes optical channel i + 180.
  The tool needs art, so the extractor recomputes its parameters from the geometry (`mapping_*`
  attributes); `PhotonLibrary.GetVisibility()` applies them and returns 360 channels.
- Check without a LArSoft reference: for in-TPC points the brightest channel is always in the point's
  own cryostat and, for 99.9% of points, among the 4 nearest PMTs.
- ICARUS simulates light with `PDFastSimPVS` + this library (no semi-analytic VUV model); the
  `icarus_pdfastsim_pvs` table is stored under `/optical_model`.
