# Plan: per-detector flash model + port of the SBND semi-analytic light model

Status (2026-10-15):
- **P0 done**: detinfo schema 1.3 has `/optical_model` and `/validation_optical`.
- **P1 done**: `model/visibility/semianalytic.py` matches LArSoft to ~1e-10 relative on 865 points × 312 opdets.
  - It needs float64: float32 loses up to 17% on the rectangle solid angles.
  - Speed on an L40 GPU, 1e5 points, direct + reflected: 0.12 s for the 120 PMTs, 0.65 s for all 312.
  - On CPU (4 threads): 49 µs/point for the PMTs.
- **P2 done**: `model/base.py` (`FlashModel`: the unified formula, per-TPC drift correction,
  OOB fraction, Neyman score + normalized option) and `model/detectors/uboone.py` (`UBooNEFlashModel`).
  - `tests/test_uboone_regression.py` is **bit-identical** to the legacy code: 38/38 checks.
  - Covered: synthetic slices with 4 flash times × 2 producers, points outside the TPC, and 17 real events
    (run-1 overlay with truth points; run-3 EXT with dead PMT 15 and saturation holes 1 and 28).
- Next: P3 (SBND model).

Related:
- detector constants: [`../detinfo/`](../detinfo/README.md)
- the larcv2 → HDF5 analysis: [`../data_prep/sbnd_official/README.md`](../data_prep/sbnd_official/README.md)

## Why

- SBND's photon library (`SBND_OpLibOUT_v2.00`) only covers light produced **outside**
  the TPC active volume. Inside, SBND simulates light with larsim's semi-analytic model
  (`phot::SemiAnalyticalModel`, used by `PDFastSimPAR`).
- SBND's own flash matcher, `SBNDOpT0Finder` → sbncode `PhotonLibHypothesis` with
  `UseSemiAnalytical`, builds its hypotheses from the same model. So flash prediction
  for SBND needs a port of that model.
- Today's flash code (`lartpc/flashmatch/`, `pointcept/models/event_slicer/photonlib.py`)
  is MicroBooNE-only:
  - 32 PMTs
  - one TPC with its anode at x=0 (drift shift `x - 0.1098*t`)
  - Y-plane ADC charge, gamma tables by run period
  - hard-coded PMT positions, dead channels and TPC box
- There is no detector abstraction.

## Reference implementation (what we port)

Sources were copied from cvmfs for sbndcode v10_14_02, which uses larsim v10_09_01:
- `larsim/PhotonPropagation/SemiAnalyticalModel.{h,cxx}`
- `PhotonPropagationUtils.cxx` (`fast_acos`, `interpolate`, `interpolate2`)
- `sbndcode/OpPathTool/SBNDOpticalPath.h`
- `opticalsimparameterisations_sbnd.fcl`, `PDFastSim_sbnd.fcl`, `opt0finder_sbnd.fcl`
- `sbndcode/OpT0Finder/SBNDOpT0Finder_module.cc`
- sbncode `OpT0Finder/flashmatch/Algorithms/PhotonLibHypothesis.cxx`

### Model, per scintillation point P and optical detector d

**Direct (VUV) visibility**

- `SBNDOpticalPath` visibility rule: 0 unless `sign(P.x) == sign(d.x)` (same side of the cathode).
- 0 if `|P − d| > MaxPDDistance` (default 1000 cm).
- Detector type and orientation come from `OpDetGeo`:
  - sphere → dome PMT (type 1)
  - bar → (X-)ARAPUCA (type 0); the smallest dimension sets the orientation (SBND: all face x, orientation 0)
- Solid angle Ω:
  - rectangle: the 4-term `Rectangle_SolidAngle` decomposition
  - dome: `Omega_Dome_Model` with an apparent-radius table, radius 10.16 cm, break distance 5r
- Geometric visibility: `exp(−dist/λ_abs) · Ω/4π`, where `λ_abs = interpolate(AbsLengthSpectrum, 9.7 eV)`.
- Gaisser–Hillas correction:
  - parameters `GH_PARS_{flat,dome}[·][θ/10°]`
  - plus border slopes `GH_border_*` interpolated in θ, times r (distance from the cathode-centre axis)
  - the result is set to 0 unless finite and in [0, 10]
- `vis_direct = GH · vis_geo / cosθ`.

**Reflected (visible) visibility** (`DoReflectedLight: true` for SBND):

1. Cathode-foil "hotspot" at `(±|x_cathode|, P.y, P.z)`.
2. Cathode visibility: rectangle Ω of the whole cathode plane × absorption × on-axis GH (θ bin 0, flat parameters).
3. For each detector: `Ω_d/2π · cathode_vis · interpolate2(VIS_correction_*[k], distance to cathode, r) / cosθ_vis`.

**Predicted PE** (`PhotonLibHypothesis::FillEstimateSemiAnalytical`):

`PE_d = Σ_P Nγ(P) · (vis_direct_d · QE · VUVEff_d + vis_refl_d · QEr · VISEff_d)`

- `VUVEff` / `VISEff` are per-channel arrays in `opt0finder_sbnd.fcl` (312 entries):
  - coated PMT 0.035/0.03 VUV, 0.039 visible
  - uncoated PMT 0 VUV, 0.0375 visible
  - X-ARAPUCA VUV 0.021, X-ARAPUCA visible 0.014
- The matcher's default uses **PMTs only**.

**Photons per point** (`SBNDOpT0Finder`):

- Electrons: `Q_hit / CalAreaConstants[plane] · exp(t_drift/τ)`, with CalAreaConstants ≈ 0.0201 ADC/e⁻.
- Photons, one of:
  - `Nγ = dE/W_ph − dQ` (calorimetry), or
  - a constant factor (`ChargeToNPhotonsTrack` 1.0, `…Shower` 1.25)

**Masks**

- 30 `OpChannelsToSkip`: dead or uncalibrated PMTs, plus the APSIA channels.
- Optional exiting-track mask: direct-light detectors within 75 cm of a track's exit point.
- Opchannel parity gives the TPC (even = TPC 0, odd = TPC 1).

**Scoring**

QLLMatch chi2 on **normalized** flashes, `ChiErrorWidth` 0.1, saturation threshold 2500 PE.

### Quirks the port must reproduce exactly

- `fast_acos`: a cubic-polynomial approximation, used for θ and therefore for the GH bin index.
- `interpolate(..., extrapolate=false)` returns the **left** value `yL` for x beyond the right
  end of the table (not `yR`).
- `interpolate2` brackets x once, then interpolates each radial row.
- GH bin `j = size_t(θ/δθ)` truncates. For θ = 90° that is j = 9 with a 10° step; check the table length.
- `Disk_SolidAngle` (elliptic integrals) is only for disk PMTs, which SBND does not have.
  Leave it out at first and raise if a detector needs it.
- The cathode centre uses `fGeom.TPC()` (TPC 0) and `fActiveVolumes[0]` for y and z.

## Design

### Package layout (new, `lartpc/flashmatch/model/`)

```
flashmatch/model/
  __init__.py          make_flash_model(detector, detinfo=None, **cfg) -> FlashModel
  base.py              FlashModel: detector-agnostic pipeline
  config.py            FlashModelConfig dataclasses (per detector defaults, overridable)
  visibility/
    base.py            VisibilityModel.visibility(pos[N,3]) -> (direct[N,D], reflected[N,D] | None)
    library.py         LibraryVisibility  (wraps PhotonLibLookup; uboone; SBND "OUT" lib)
    semianalytic.py    SemiAnalyticVisibility (torch, vectorized over points x opdets)
    hybrid.py          inside-AV -> semi-analytic, outside-AV -> library (mirrors PDFastSim split)
  charge/
    uboone.py          current Y-plane ADC + dedup + gamma*readout (unchanged numbers)
    sbnd.py            hit charge -> electrons (CalArea, lifetime) -> photons
  detectors/
    uboone.py          UBooNEFlashModel  (adapter over today's functions; bit-identical)
    sbnd.py            SBNDFlashModel
```

### Pipeline (`FlashModel`)

```
predict(points, charge, cluster_id, n_clusters, t_flash_us) -> pe[n_clusters, n_opch]
  1. photons  = charge_model(points, charge)                 # per detector
  2. x_true   = drift_correct(points, t_flash_us)            # per TPC: x -= drift_dir_x * v * t
  3. vd, vr   = visibility(x_true)                           # (N, D), (N, D)|None
  4. pe       = index_add(photons * (vd*eff_vuv + vr*eff_vis))   # D = n optical channels
mask(observed, run) -> bool[D]                               # skip list, dead, saturation
score(pred, obs, mask) -> chi2                               # neyman (uboone) | normalized QLL (sbnd)
```

The detector-specific pieces are config plus small strategy objects. The geometry comes from
`lartpc.detinfo`, never from constants in this code:
- photon-detector positions, types and channel map
- TPC active volumes, drift direction, anode/cathode x
- drift velocity, LAr absorption spectrum

| piece | MicroBooNE | SBND |
|---|---|---|
| visibility | `LibraryVisibility` (v6_70kV LUT, trilinear) | `HybridVisibility` (semi-analytic in AV + OUT library), or semi-analytic only |
| channels D | 32 PMTs (OpDet order) | 312 opchannels; subset selected by type (default PMTs only, as OpT0Finder) |
| efficiencies | folded into gamma | `VUVEff` / `VISEff` per channel (opt0finder), × global gamma |
| charge model | Y-plane ADC (U/V fallback), dedup, × gamma(period) × readout | hit charge / CalArea × lifetime × photons-per-electron; dedup by `hit_key` |
| drift | one TPC, anode x=0 | per TPC (x<0 drifts −x, x>0 drifts +x), v from detinfo |
| masks | dead opdets by run period + saturation holes | `OpChannelsToSkip` + saturation (2500 PE) + optional exiting mask |
| score | Neyman chi2, absolute PE, f_sys 0.1 | option: normalized chi2 (QLL mode 0, width 0.1), or Neyman |
| calibration | `flash_calib` gamma table | new SBND gamma (MC first) |

### Semi-analytic implementation notes

- **Torch, vectorized.** Build opdet tensors once (centre, type, orientation, h, w, efficiencies) and evaluate the
  (N, D) grid in chunks of points. A slice of 1e4–1e5 points × 120 PMTs is 1e6–1e7 pair evaluations,
  which is fine on GPU and acceptable on CPU in chunks.
- **Tables.** Load GH and VIS tables as tensors. Do the `interpolate` / `interpolate2` bracketing with
  `searchsorted`, applying the right-end clamp quirk explicitly.
- **Optional dense in-AV table.** Tabulate the semi-analytic visibilities on a grid once per detector version
  (e.g. 2.5 cm in the AV, a few hundred MB float16) to make it a LUT like MicroBooNE's.
  Keep the analytic path as the reference; choose by benchmark.

## Data needed in detinfo (extractor schema 1.3, `/optical_model`)

1. **VUV/VIS hit parameterizations.** `fhicl-dump` of `sbnd_vuv_RS100cm_hits_parameterization` and
   `sbnd_vis_RS100cm_hits_parameterization` (from `opticalsimparameterisations_sbnd.fcl`),
   stored as arrays: `GH_PARS_flat/dome`, `GH_border_*`, `VIS_distances_*`, `VIS_correction_*`,
   `delta_angulo_*`, `PMT_radius`, `MaxPDDistance`, and the flags.
   - The flash-model FHiCL is not part of the services dump, so the extractor needs a second dump
     target: a per-detector list of (fcl, table) pairs.
2. **Per-opdet semi-analytic descriptor**, computed from larsoft `OpDetGeo` exactly as
   `opticalDetectors()` does: type (`isSphere` / `isBar`), orientation, h, w.
   Our current `dx/dy/dz` are local-frame TGeo half-sizes and would be a guess.
3. **Model geometry**: `ISTPC::extractActiveLArVolume` volumes, cathode centre, anode centre,
   drift distance (as the model builds them).
4. **Matcher constants**: `VUVEfficiency`, `VISEfficiency`, `OpChannelsToSkip`, `CalAreaConstants`,
   `ChargeToNPhotons*`, `PhotoDetectors` default, QLL settings (from `opt0finder_sbnd.fcl`),
   and `ScintTime` `SlowDecayTime` (1300 ns, `PDFastSim_sbnd.fcl`).
5. **Reference values for validation.** Compile a **patched copy** of `SemiAnalyticalModel.cxx` with cling
   in the extractor. The constructor gets the GeometryCore / WireReadoutGeom / LArProperties
   providers we already build instead of `art::ServiceHandle`; the physics code stays untouched.
   Store `sam_direct[N,312]` and `sam_reflected[N,312]` for the validation points
   (plus a dense line scan across the cathode and along x).
   If the cling build fails, fall back to running sbndcode's `PDFastSimPAR` in a tiny art job,
   which is heavier.

## Validation

1. **Unit:** the python semi-analytic model vs the cling reference, every point × channel, direct and
   reflected, to relative 1e-6 (`fast_acos` reproduced exactly). Add it to `detinfo/tests` style checks.
2. **Regression:** `UBooNEFlashModel` reproduces today's `pred_pe` bit-for-bit
   (`flashmodel_calib/scripts/validate_records.py` + `slicer_eval/smoke_test_lib.py`).
3. **Physics, SBND MC.** The larcv2 test file has true per-voxel energy (`sparse3d_pcluster`, MeV),
   reco charge and the `opflash_tpc0/1` flashes.
   - Predict the in-time flash from **true** deposits (photons = E × light yield) and compare the PE shape
     (cosine) and normalization per TPC.
   - Then from reco charge with the charge model, and fit the SBND gamma (procedure as in
     `flashmodel_calib/PROTOCOL.md`).
4. **Cross-check against OpT0Finder** on a few events, if SBND CAF/analysis-tree outputs are
   available: our hypothesis vs `opt0_hypo_pe`.

## Phases

| phase | work | output |
|---|---|---|
| P0 | extractor schema 1.3: `/optical_model` + cling reference values | updated SBND detinfo file |
| P1 | `visibility/semianalytic.py` + unit test vs reference | parity test passes |
| P2 | `FlashModel` base + `UBooNEFlashModel` adapter; no behaviour change | regression bit-identical |
| P3 | `SBNDFlashModel`: charge model, efficiencies, per-TPC drift, masks, score | predicts (312,) PE |
| P4 | call sites take `--detector` (default `uboone`): keypoint2 cascade `_flash_slice_table`, stage-3 `flash_recovery_keep`, `analyze_event`, `gammacal` | one code path |
| P5 | SBND MC closure + gamma fit (needs the larcv2 converter for reco charge) | calibrated SBND model |

P0–P3 do not depend on the larcv2 converter. P5 does, at least for reco charge.

## Decisions (2026-10-14)

1. **Photon detectors:** all sensors can be used (`detectors=` takes PD types or `"all"`).
   The **default is PMTs only**; X-ARAPUCAs are still being understood.
2. **Light from outside the active volume:** use the OUT library, as SBND's simulation does.
   Verified in `standard_g4_sbnd.fcl`:
   - `ionandscint` → `pdfastsim` (semi-analytic) inside the active volume
   - `ionandscintout` → `pdfastsimout` (library) outside it

   Reco space points are always inside the AV, so the library is loaded **lazily**, only when
   some point lies outside the AV (e.g. MC true deposits).
3. **Unified formula; MicroBooNE as a special case:**
   `PE_d = G · Σ_P n(P) · [ v_dir,d(P)·ε_dir,d + v_refl,d(P)·ε_refl,d ]`, with
   `n = q / CalArea[plane] · exp(t_drift/τ) · k_γ`.
   - **MicroBooNE:**
     - CalArea = 1, no lifetime correction, `k_γ` = 1
     - `q` = deduplicated Y-plane ADC with U/V fallback
     - `v_dir` = v6 library (trilinear), `v_refl` = 0, `ε_dir` = 1
     - `G` = 5.25 × period scale × readout factor
   - **SBND:**
     - semi-analytic direct and reflected visibilities in the AV, library outside
     - ε from the `opt0finder_sbnd.fcl` arrays
     - `k_γ` = 1 and one fitted `G`
   - Per detector: the charge rule, the per-TPC drift correction, and the lookup mode
     (trilinear vs nearest voxel as in LArSoft).
   - The MicroBooNE configuration must reproduce today's `pred_pe` bit-for-bit.
4. **Score:** the default for every detector is the existing absolute Neyman chi2.
   `score="normalized"` reproduces sbncode `QLLMatch` (QLLMode 0, `NormalizeFlash`):
   - zero the masked channels, then divide the hypothesis and the measurement by their own sums
   - keep only channels whose parity matches the flash TPC
   - floor O and H at 1e-6
   - each channel adds `(O−H)²/(H + (0.1·H)²)` (Pearson: the error is the hypothesis)
   - divide the sum by the number of channels used
   - beforehand, a PMT with hypothesis ≥ 2500 PE and observed 0 is treated as saturated:
     its hypothesis is set to 0
5. **No dense semi-analytic table.** The model is the memory-cheap option: tables of a few tens of KB,
   against ~500 MB for a dense float32 table on SBND's own 104×54×71×312 grid.
   - Per point, a lookup is cheaper than the analytic evaluation (a few hundred flops per pair).
   - At ~1e7 point×detector pairs per slice on GPU the cost is still small.
   - The benchmark (P1) only confirms the per-slice cost; a table is added only if it turns out to be a bottleneck.
