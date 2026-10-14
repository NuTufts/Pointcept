# Semi-analytic light model port: quirks, validation, benchmarks

The python/torch port of larsim's `phot::SemiAnalyticalModel` lives in
[`model/visibility/semianalytic.py`](model/visibility/semianalytic.py).
Plan and phases are in [`SBND_SEMIANALYTIC_PLAN.md`](SBND_SEMIANALYTIC_PLAN.md).

Reference version: larsim **v10_09_01**, as used by sbndcode v10_14_02:
- `larsim/PhotonPropagation/SemiAnalyticalModel.cxx`
- `larsim/PhotonPropagation/PhotonPropagationUtils.cxx`
- `sbndcode/OpPathTool/SBNDOpticalPath.h`

## Rerun

```bash
# CPU (here): validation against the LArSoft reference + a 20k-point CPU benchmark
bash lartpc/flashmatch/model/tests/run_semianalytic_checks.sh
# GPU: submits validation + a 1e5-point benchmark to gpu,preempt
bash lartpc/flashmatch/model/tests/run_semianalytic_checks.sh --gpu
```

**Where the reference comes from.** The reference values are `/validation_optical` in the detinfo
file (schema ≥ 1.3). The extractor (`lartpc/detinfo/extract/run_extract.sh`) computes them by
compiling larsim's own `SemiAnalyticalModel.cxx` with cling
(`lartpc/detinfo/extract/larsoft_ref/semianalytic_ref.py`). Only the constructor is patched: it
takes the gallery providers instead of art services. The physics code is unmodified.

**After a new sbndcode/larsim version:**
1. Re-extract.
2. Rerun the checks.
3. Diff the upstream `SemiAnalyticalModel.cxx` against the quirk list below.

## Calculation quirks reproduced exactly

These are properties of the upstream C++ that a "clean" reimplementation would get wrong.

1. **`fast_acos`.** Every angle (θ for direct light, θ_vis for reflected light, and inside the
   rectangle solid angle) uses larsim's cubic approximation
   `((a3·x + a2)·x + a1)·x + a0) · sqrt(1−x)`, with error up to ~1e-4 rad. It is not `acos`. This moves
   the Gaisser–Hillas (GH) and VIS angle-bin edges.
2. **Angle bins truncate.** `j = size_t(θ / delta_angulo_vuv)` and `k = size_t(θ_vis / delta_angulo_vis)`.
   Because of `fast_acos`, θ < 90° always, so j ≤ 8 for 10° bins.
3. **`interpolate(x, y, x0, extrapolate=false)` returns the *left* value `yL` past *both* ends.**
   Past the right end of the table it returns `y[n-2]`, not `y[n-1]`. The bracket for
   `x ≥ x[n-2]` is fixed at `n-2`.
4. **`interpolate2` brackets x once** (on the x-distance table) and reuses that bracket for every
   radial row, then interpolates the row values in r (no extrapolation).
5. **The border corrections for direct light *do* extrapolate:**
   `interpolate(GH_border_angulo_*, GH_border_*[i], θ, true)`. For reflected light they are evaluated at θ = 0.
6. **`Rectangle_SolidAngle(Dims, v, orientation)` always compares `|v.Z()|` against the
   width,** for every orientation. Only `d1` (|y| for orientation 0, |x| otherwise) and `d2`
   change with orientation.
7. **Machine-epsilon branching in the rectangle solid angle.** `isDefinitelyGreaterThan(a, b)` means
   `a−b > ε` or `a−b > max(|a|,|b|)·ε` (ε = 2.2e-16), and `isApproximatelyZero(a)` means `|a| ≤ ε`.
   Values in the gaps between the branch conditions fall through to 0. The branch order is:
   on-axis, both outside, both inside, d1 outside, z outside.
8. **The GH correction is set to 0 unless it is finite and in [0, 10].** This catches the NaN from
   `pow(negative, non-integer)` close to detectors.
9. **Optical path (SBND):** a detector sees a point only if `(P.x < 0) == (O.x < 0)`, i.e. it
   sees its own side of the cathode. Points at exactly x = 0 count as the x ≥ 0 side.
10. **`MaxPDDistance`** (default 1000 cm) zeroes direct light beyond that distance; reflected light has no such cut.
11. **Reflected light** is computed in three steps:
    - The hotspot is `(±|cathode.x|, P.y, P.z)`, with the sign from P.x.
    - The cathode visibility uses the solid angle of the **whole** active-volume y–z face, absorption over the
      distance to the cathode, and the GH correction from the *flat* parameters at θ-bin 0.
    - The detector term is `Ω_d/2π` (2π because of the reflective foils).
12. **Cathode singularity.** For a point exactly on the cathode (distance to cathode = 0), larsim
    returns **NaN** reflected visibilities (0/0) for the detectors on the point's side. This is
    reproduced; callers must not feed charge at x = 0 exactly (or must mask it).
13. **Dome PMT solid angle (`Omega_Dome_Model`).** It uses a fixed 9-bin apparent-radius table, with
    "close"/"far" parameters switching at 5 × PMT radius.
14. **Model geometry is taken from TPC 0 / active volume 0.** The cathode centre is
    `(TPC(0).cathode.x, AV[0].centre.y, AV[0].centre.z)`. The cathode plane is `AV[0]` y × z, where `ISTPC`
    merges SBND's two TPCs into one active volume, x ∈ [−201.3, 201.3].
15. **Detector shape and orientation** come from `OpDetGeo` as `opticalDetectors()` derives them:
    - sphere → dome (type 1)
    - bar → rectangle (type 0), with the orientation set by its smallest dimension
    - otherwise disk (type 2, **not ported**; construction raises)

    We store the model's own result in detinfo rather than re-deriving it from TGeo half-sizes.

## Validation results (2026-10-15, detinfo `sbnd/sbndcode_v10_14_02`, schema 1.3)

The reference is 865 points × 312 opdets:
- 150 random points per TPC
- an x scan, plus z and y scans at x = ±half drift
- 40 points in front of detectors

| check | float64 result |
|---|---|
| direct, all opdets | max \|Δ\| 1.4e-14, max rel 1.3e-10 (CPU); 7e-16 / 5.9e-11 (GPU) |
| reflected, all opdets | max \|Δ\| 3.2e-14, max rel 2.3e-10 (CPU); 7e-16 / 4.8e-12 (GPU) |
| PMT subset (120) | equals the reference columns, max rel ~1e-12 |
| NaN pattern (cathode point) | identical |
| SBND optical path | no direct light on the far side of the cathode (0 violations) |

The tolerance is `|Δ| ≤ 1e-9 + 1e-6·|ref|`.

**float32 is not safe** (measured against the same reference):

| quantity | float32 error |
|---|---|
| PMT direct | max rel 1.7e-4 (fine) |
| PMT reflected | max rel 15% per channel; per-point sum over PMTs 0.35% |
| X-ARAPUCA direct | max rel 17% per channel; sum 0.7% |
| X-ARAPUCA reflected | max rel 23% per channel |

The cause is cancellation in the 4-term rectangle solid angle, which is used for X-ARAPUCAs and for
the cathode plane in reflected light. **float64 is the default.**

## Benchmarks (direct + reflected, float64 unless noted)

| device | points | detectors | time | per point |
|---|---|---|---|---|
| NVIDIA L40 (job 5245611) | 1e5 | 120 PMTs | 0.12 s | 1.2 µs |
| NVIDIA L40 | 1e5 | all 312 | 0.65 s | 6.5 µs |
| NVIDIA L40, float32 | 1e5 | 120 PMTs / 312 | 0.11 s / 0.34 s | 1.1 / 3.4 µs |
| CPU, 4 threads (pax077) | 2e4 | 120 PMTs | 0.99 s | 49 µs |
| CPU, 4 threads | 2e4 | all 312 | 7.3 s | 364 µs |
| CPU, 4 threads, float32 | 2e4 | 120 PMTs / 312 | 0.64 s / 3.6 s | 32 / 180 µs |

For comparison, larsim C++ takes ~0.09 ms per point for 312 detectors, direct only, single thread.

Notes:
- On GPU, float64 costs about the same as float32 for PMTs.
- All-312 is dominated by the X-ARAPUCA rectangle solid angle: all five branches × four terms are
  evaluated for every pair.
- Each shape's formula is only evaluated on its own detector columns. Before that change, PMTs on CPU took 153 µs/point.

Possible further speed-ups, not done:
- evaluate the rectangle branches only on their masked elements
- merge charge points onto a ~1 cm grid before evaluating
