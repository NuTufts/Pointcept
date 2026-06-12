# Gen2 Flat Ntuple — Parsing Spec

Practical spec for reading the **DL Gen2 flat ntuples** (the "ntuple" datasets in
[`MicroBooNE_Datasets_on_Tufts.md`](MicroBooNE_Datasets_on_Tufts.md)). These are
plain flat ROOT files — no custom C++ classes, so **no `ubdl`/`larcv`/`larlite`
libraries are needed to read them**, just ROOT/PyROOT (or `uproot`). This
contrasts with the official `merged_dlreco`/`merged_dlana` files, which *do*
require the serialized-class libraries.

Authoritative branch list (and the maker that fills them) lives in the
[gen2ntuple repo](https://github.com/NuTufts/gen2ntuple): `README.md` (variable
docs) and `make_dlgen2_flat_ntuples.py` (the filling logic). This file
summarizes the parts that matter for **truth-level event selection** and records
the **non-obvious gotchas** verified on the Tufts copy
(`/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/gen2ntuple`).

---

## File layout

Each ntuple file is a single flat ROOT file with two TTrees:

| Tree | Entry granularity | Use |
|------|-------------------|-----|
| `EventTree` | one entry **per input data event** | all per-event reco + truth variables |
| `potTree` | one entry **per source `merged_dlreco` file** (MC only) | POT bookkeeping |

A "dataset ntuple" is usually a **single hadded file** covering the whole sample.
Example (BNB ν overlay, run 3b), verified Dec 2025:

```
/cluster/tufts/wongjiradlabnu/nutufts/data/ntuples/dlgen2_reco_v2me05_gen2ntuple_v7_run3b_bnb_nu_overlay_nocrtremerge.root
  EventTree : 290,538 entries
  potTree   :  15,513 entries   ->  sum(totGoodPOT) = 8.9832e20 POT
```

### POT normalization

`potTree` has `totPOT` and `totGoodPOT` (use **`totGoodPOT`**; the two are
generally equal). Total sample POT = sum over **all** `potTree` entries:

```python
pot = 0.0
for i in range(potTree.GetEntries()):
    potTree.GetEntry(i); pot += potTree.totGoodPOT
```

To scale a weighted event count to a target POT (e.g. runs 1-3 = 6.67e20):
`expected = sum(xsecWeight over selected) * targetPOT / pot`.
**Always weight MC events by `xsecWeight`** when producing physics counts/spectra.

---

## Event-loop idiom (PyROOT)

```python
import ROOT as rt
f  = rt.TFile(path)
et = f.Get("EventTree")
for i in range(et.GetEntries()):
    et.GetEntry(i)
    # scalars: et.trueVtxX, et.trueNuCCNC, et.xsecWeight, ...
    # arrays:  loop j in range(et.nTrueSimParts): et.trueSimPartPDG[j]
```

Per-particle variables are C arrays whose length is the matching `nXxx` scalar
(`trueSimPartPDG[nTrueSimParts]`, `trackPID[nTracks]`, `showerPID[nShowers]`,
`truePrimPartPDG[nTruePrimParts]`). Always bound array loops by the `nXxx`
counter, never by a fixed size.

---

## Event filtering already applied (important)

For MC, the maker **drops events before they ever reach the ntuple** when:
- the **SCE-corrected true ν vertex is outside the Wire-Cell fiducial volume**
  (> 3 cm from the SCE-corrected detector edge), or
- the event's cross-section weight is unknown or infinite.

Consequence for truth selections: **every ntuple event already has its true
vertex inside the WC fiducial volume**, which is a *strict subset* of the full
TPC active volume `x(0,255) y(-116.5,116.5) z(0,1036)` cm. A vertex-in-TPC cut
applied on the ntuple is therefore (nearly) a no-op — verified 290,538/290,538
pass. **Events in the TPC but outside the WC-FV are not in the ntuple at all**
and can only be recovered from the official files.

---

## Truth variables — two distinct particle families

The single most common confusion. There are **two** truth-particle array
families, filled from different sources, with different meanings:

### `truePrimPart*` — GENIE primaries (final-state, pre-detsim)
Stable final-state particles from the neutrino interaction (`StatusCode()==1`),
e.g. a μ⁻, a proton, **or a π⁰ (PDG 111)**. Filled from `mctruth` "generator".
A π⁰ appears here; its decay photons do **not** (they are made later by Geant4).

### `trueSimPart*` — Geant4/detsim-tracked particles
Filled from larlite **`mcreco` `MCTrack` + `MCShower`**
(`make_dlgen2_flat_ntuples.py` ~L702-743). This is where **photons (PDG 22)**,
electrons, and the actual tracked daughters live. Key fields:

| Field | Meaning / unit |
|-------|----------------|
| `trueSimPartPDG` | PDG code |
| `trueSimPartTID` / `trueSimPartMID` | trackID / mother trackID (ancestry) |
| `trueSimPartProcess` | 0 = primary from ν, 1 = decay, 2 = other |
| `trueSimPart{X,Y,Z}` | SCE-corrected **start** position (cm) |
| `trueSimPartEDep{X,Y,Z}` | **first energy-deposit** point (cm). For photons = first-conversion point (from `MCShower::DetProfile()`); equals start for other particles |
| `trueSimPart{Px,Py,Pz,E}` | initial 4-momentum, **MeV** / MeV/c |
| `trueSimPart{EndX,EndY,EndZ,EndE,...}` | end position / end 4-momentum |
| `trueSimPartContained` | 1 if SCE-corrected end is in WC-FV |

> **Unit gotcha:** `trueSimPart*` energies/momenta are in **MeV**, but the
> neutrino-level `trueNuE`, `trueLepE`, and `truePrimPartE` are in **GeV**.
> Reco energies (`recoNuE`, `trackRecoE`, `showerRecoE`) are in **MeV**.

### Overlay → `trueSimPart` is neutrino-only (no cosmics)

In an **overlay** sample the cosmic background is **real off-beam data** with no
simulated `MCParticle`s, so `mcreco` MCTrack/MCShower contain **only the
neutrino-induced particles**. Therefore in overlay every `trueSimPart` entry —
including every PDG-22 photon — is **neutrino-origin by construction**. Verified
Step-0 check: photon start-to-true-vertex distance has median ≈ 0 cm (π⁰ photons
at the vertex), not the detector-wide scatter cosmics would produce.

> Caveat: this nu-only property is specific to **overlay** MC. For a
> cosmics-included MC sample you would need the `trueSimPartProcess` / ancestry
> (`MID`) to separate origins. For **data** ntuples the truth arrays are absent.

### Worked example — "≥1 neutrino-origin photon ≥20 MeV depositing in the TPC"

```python
nphot = 0
for j in range(et.nTrueSimParts):
    if et.trueSimPartPDG[j] != 22:          # photon
        continue
    if et.trueSimPartE[j] <= 20.0:          # MeV; below 20 MeV can't make a 20 MeV cluster
        continue
    if not in_tpc(et.trueSimPartEDepX[j], et.trueSimPartEDepY[j], et.trueSimPartEDepZ[j]):
        continue                            # first conversion outside TPC -> invisible
    nphot += 1
```

Caveat on this proxy: many `trueSimPart` photons are **neutron-capture γ's**
(neutrons dominate the `trueSimPart` PDG census). The 20 MeV initial-energy cut
removes the ~9 MeV Ar-capture cascade but is only **necessary, not sufficient**
for detectability — the real "≥20 MeV in a *single ionization cluster*" cut needs
per-cluster ionization, which the ntuple does **not** carry and must be computed
on the official sim files (see [`MicroBooNE_Datasets_on_Tufts.md`](MicroBooNE_Datasets_on_Tufts.md)
and the single-photon study under `lartpc_data_prep/larformer_physics/single_photon/`).

---

## File identification: mapping an event back to its official file

Two identifiers per event let you find the source official file:

- `run`, `subrun`, `event` — the canonical event ID.
- `fileid` — **parsed from the source larflowreco filename's `fileid<NNNN>`
  token** (`make_dlgen2_flat_ntuples.py` ~L759-762). It is *intended* to be the
  **line number (index) into the sample's master filelist**, e.g.
  `/cluster/tufts/wongjiradlabnu/mrosen25/filelists/mcc9_v29e_dl_run3b_bnb_nu_overlay_nocrtremerge_filelist.txt`.

> **`fileid` is fragile.** It only stays correct if that master filelist is never
> reordered. **Do not trust it blindly** — confirm it, or skip it entirely and
> match on `(run, subrun, event)`.

### Robust match: `(run, subrun, event)` via `larlite_id_tree`

> **Verified June 2025:** the `fileid`→filelist-line mapping is **broken** for the
> run-3b BNB ν overlay ntuple — `fileid=0` events are run 16934, but filelist line
> 0 is run 14121. The list was reordered after the ntuples were made. **Use the
> `(run,subrun,event)` match below, not `fileid`.**

The official `merged_dlreco`/`merged_dlana` files expose the same event IDs
through the **`larlite_id_tree`** TTree — a simple tree whose branches are plain
`Int_t` `_run_id` / `_subrun_id` / `_event_id`. They are POD, so you can read them
with **bare ROOT (no larlite libraries)**:

```python
t = rt.TFile(path).Get("larlite_id_tree")
for i in range(t.GetEntries()):
    t.GetEntry(i)
    r, s, e = int(t._run_id), int(t._subrun_id), int(t._event_id)
```

Two verified facts make the lookup cheap (no full 15k-file scan needed):

1. **The on-disk path encodes the run.** Zero-pad the run to 8 digits and split
   into four 2-digit directories: run 16934 → `00016934` →
   `…/mcc9_v29e_dl_run3b_bnb_nu_overlay_nocrtremerge/data/00/01/69/34/`. All files
   for that run live in that one directory.
2. **A merged file holds *multiple* `(run,subrun)` pairs** — it is NOT one subrun
   per file. So index each candidate file under **every** `(run,subrun)` present in
   its `larlite_id_tree`, then confirm the specific `event` id is in the file.

Recipe: for each selected signal event, derive its run directory, `glob` the
`merged_dlreco_*.root` files there, read each one's `larlite_id_tree`, map
`(run,subrun) → path`, and confirm the event. The worked implementation is
`lartpc_data_prep/larformer_physics/single_photon/map_signal_to_files.py`
(validated: 20/20 selected events resolved, 100% confirmed, against 9 unique files).

---

## Reco variables (one-line orientation)

Not needed for truth selection but present for downstream analysis:

- **Vertex/event:** `foundVertex`, `vtxIsFiducial`, `vtxContainment`, `vtxScore`,
  `vtxFracHitsOnCosmic`, `recoNuE` (MeV), `eventPC*`, `kp*` (keypoints).
- **Tracks** (`nTracks`): `trackPID`, `track{El,Ph,Mu,Pi,Pr}Score`, `trackRecoE`,
  `trackProcess`, `trackStartPos/Dir*`, plus `trackTrue*` truth-match columns.
- **Showers** (`nShowers`): `showerPID`, `shower{El,Ph,...}Score`, `showerRecoE`,
  `showerProcess`, `showerStartPos/Dir*`, plus `showerTrue*` truth-match columns.

PID scores come from the **LArPID** CNN. Reco prongs are matched to the true
particle depositing the most charge (`*TruePID`, `*TrueComp`, `*TruePurity`).
Full per-branch docs: gen2ntuple `README.md`.

---

## Quick reference: running the readers in this repo

The single-photon study scripts (`lartpc_data_prep/larformer_physics/single_photon/`)
are the canonical worked example:

```bash
apptainer exec --bind /cluster:/cluster \
  /cluster/tufts/wongjiradlabnu/larbys/larbys-container/pointcept_cuml.sif \
  bash -c "source /cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/ubdl/setenv_pointcept_container.sh >/dev/null 2>&1; \
           python3 verify_ntuple.py -n 3000"          # Step-0 sanity checks
# select_single_photon_signal.py  -> raw + POT-scaled counts, signal_events.csv, signal_fileids.txt
```

> `source setenv_pointcept_container.sh` is only needed to put **ROOT** on the
> python path inside the pointcept container; the flat ntuple itself needs no
> serialized-class libraries.
