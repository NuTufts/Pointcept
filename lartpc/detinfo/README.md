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
| [`extract/run_extract.sh`](extract/run_extract.sh), [`extract/extract_larsoft_detinfo.py`](extract/extract_larsoft_detinfo.py) | done; SBND sbndcode v10_14_02 extracted with 0 failed queries |
| [`SCHEMA.md`](SCHEMA.md), [`dump_tables.py`](dump_tables.py) | done (schema 1.3: light-model tables + semi-analytic LArSoft reference) |
| python utility: `Geometry`, `DetectorClocks`, `DetectorProperties`, `LArProperties`, `SpaceCharge` (SBND), `PhotonLibrary`, `Conventions` | done; **8/8 checks vs LArSoft reference pass** ([`tests/test_validation.py`](tests/test_validation.py)) |
| plotly 3D geometry review page ([`viz/make_detinfo_html.py`](viz/make_detinfo_html.py)) | done |
| SBND in-TPC light: semi-analytic model (the SBND library covers only outside the active volume) | open |
| ICARUS / MicroBooNE extraction | `icarus` entry is a guess, untested; `uboone` not yet added |
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
