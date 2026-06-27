# Implementation Brief: Piecewise-Linear Trajectory Fitting from LArTPC Spacepoint Instances

## 0. How to use this document
This is background context for planning and building an implementation. The goal is to fit a
**piecewise-linear polyline to the 1D physical trajectory of a particle**, given a 3D spacepoint
cloud that is distributed in a roughly cylindrical "tube" around that trajectory. The cloud comes
from an upstream PTv3-based instance-segmentation model that has already isolated one particle per
instance.

Three methods are to be implemented and compared. Two of them compose into one pipeline; the third
is a standalone alternative:

- **Method A — Sliding-window PCA**: collapses the tube to an ordered, denoised centerline.
- **Method B — RDP with an MCS-tied threshold**: turns an *ordered* centerline into line segments,
  placing break points at physically real kinks. **Consumes the output of A.**
- **Method C — ElPiGraph**: an end-to-end elastic-principal-graph fit that does centerline
  extraction, ordering, and segmentation in one shot. **Standalone alternative to A+B.**

All three must emit the **same unified output schema** (Section 3) so they can be benchmarked head
to head against the same truth and metrics (Section 7).

---

## 1. Physics / domain context (brief)
- Detector: liquid-argon time projection chamber (LArTPC). Radiation length of liquid argon
  **X₀ ≈ 14.0 cm**. `[CONFIRM detector: MicroBooNE / SBND / DUNE — affects spatial resolution and
  fiducial scale, not X₀]`
- Trajectories are *not* straight: they curve and develop kinks from multiple Coulomb scattering
  (MCS, continuous small-angle) and from hard scatters / hadronic interactions (discrete large-angle).
  The **discrete kinks are the segment break points we want**; the MCS wiggle should be absorbed
  into straight segments.
- The point cloud has finite transverse width because of detector spatial resolution and the
  physical ionization spread, hence the "cylinder around a line" geometry.
- Tracks that fold back on themselves (low-momentum curlers, e±/γ conversions) are a known hard
  case for any method that relies on a single global ordering axis (see Method A failure modes).

---

## 2. Input data specification  *(PROPOSED — confirm before building)*
The upstream model produces, per event, a set of reconstructed 3D spacepoints with predicted
instance labels. Trajectory fitting operates **per instance**.

### 2.1 Per-event arrays `[CONFIRM exact layout]`
| field | dtype | shape | meaning |
|---|---|---|---|
| `coords` | float32 | `(N, 3)` | spacepoint positions `(x, y, z)` in **cm** `[CONFIRM units]`, detector frame `[CONFIRM: x=drift, y=vertical, z=beam?]` |
| `charge` | float32 | `(N,)` or `(N, F)` | per-point charge / ADC (and any extra features). Used as PCA weights and later for dE/dx. `[CONFIRM availability]` |
| `instance_id` | int32 | `(N,)` | predicted instance label per point; background / unclustered = `-1` `[CONFIRM sentinel]` |
| `semantic_id` | int32 | `(N,)` | optional: track / shower / etc., to filter to track-like instances `[CONFIRM availability]` |

### 2.2 Truth arrays (simulation only, for evaluation) `[CONFIRM availability + layout]`
| field | dtype | shape | meaning |
|---|---|---|---|
| `true_instance_id` | int32 | `(N,)` | per-point true particle/instance label, for truth-matching predicted instances |
| `true_traj` | list of float32 `(Mᵢ, 3)` | per true particle | Geant4 trajectory points (the ground-truth polyline) in cm |
| `true_pdg` | int32 | per particle | PDG code |
| `true_p` | float32 | per particle | true momentum (MeV/c) — needed to evaluate the MCS model and angular resolution |

### 2.3 Practical scale parameters `[CONFIRM]`
- Typical `N` per event: `[CONFIRM — thousands? millions?]`
- Typical points per instance: `[CONFIRM — tens? few thousand?]`
- Nominal point spacing / voxel pitch: `[CONFIRM — e.g. ~3 mm]`. This sets the floor on segment
  resolution and feeds the resolution term in the MCS threshold.
- Are instances delivered **pre-split into separate arrays**, or as **one labeled cloud** the
  implementation must group by `instance_id`? `[CONFIRM]`

### 2.4 Input contract handed to each method
Each method receives a single instance: `points: (n, 3) float32` and optional `weights: (n,)
float32`. It must not assume points are ordered. It must handle small `n` (e.g. < 10) gracefully
(return a degenerate 1–2 vertex polyline rather than crashing).

---

## 3. Unified output schema (all three methods return this)
```python
@dataclass
class TrajectoryFit:
    instance_id: int
    vertices: np.ndarray        # (M, 3) float32, ordered polyline vertices in cm
    seg_lengths: np.ndarray     # (M-1,) float32, cm
    seg_dirs: np.ndarray        # (M-1, 3) float32, unit direction of each segment
    residual_rms: float         # RMS perpendicular distance, cloud points -> polyline (cm)
    n_points: int               # points in the instance
    runtime_s: float            # wall-clock for this instance
    method: str                 # "sliding_pca_rdp" | "elpigraph"
    extra: dict                 # method-specific diagnostics
```
A reusable helper `point_to_polyline_distance(points, vertices)` (3D point-to-segment, min over
segments) should live in shared utils — it is needed for `residual_rms`, for RDP, and for evaluation.

---

## 4. Method A — Sliding-window PCA  (centerline extraction + ordering)
### 4.1 What it does
Produces an **ordered, denoised, dense centerline** by sliding a window along the track and taking a
charge-weighted local PCA in each window. The window centroid is a smoothed trajectory point; the
leading eigenvector is the local direction. This is the LArTPC-standard "3D sliding linear fit"
(cf. Pandora's `ThreeDSlidingFitResult`).

### 4.2 Algorithm
1. **Global PCA** on the instance to get the leading principal axis `e1`.
2. Project every point onto `e1` to get a scalar arc-length proxy `s = (p - centroid) · e1`.
3. **Sort points by `s`.**
4. **Slide a window** along `s` (either fixed width `w` in cm, or fixed `k` nearest-in-`s` points).
   For each window: compute charge-weighted mean (→ smoothed centerline point) and the covariance's
   leading eigenvector (→ local direction).
5. Output the ordered sequence of smoothed centerline points + per-point directions.
6. (Optional) re-parametrize by cumulative arc length along the smoothed points and **resample
   uniformly** so RDP sees evenly spaced input.

### 4.3 Key parameters
- `window` (cm) or `k` (points): smoothing scale. Too small → noisy, fails to denoise the tube;
  too large → rounds off real kinks. Tie loosely to tube radius and point spacing.
- `step`: window stride / output sampling density.
- `use_charge_weights`: bool.

### 4.4 Dependencies & GPU
- Pure PyTorch: `torch.linalg.eigh` on batched `(*, 3, 3)` covariance tensors. Windowing via sort +
  segment/gather reductions (or cumulative-sum trick for sliding covariances).
- Single track runs in ms on CPU; **GPU value comes from batching many instances** into one
  vectorized eigendecomp. Plan for a batched code path that processes a padded ragged batch of
  instances at once. `torch_cluster` optional for the MST fallback below.

### 4.5 Failure modes to handle
- **Non-monotonic ordering (curlers / hairpins)**: `s` from global PCA is not monotone along the
  true path → ordering scrambles. Detect via large local direction reversals or high fit residual;
  fall back to ordering via **MST on a kNN graph + longest-path / endpoint shortest-path**.
- Branched instances (primary + δ-ray still in one instance): sliding PCA assumes a single 1D
  thread. Either pre-split, or defer these to Method C.
- Very short tracks: return raw endpoints.

### 4.6 Output
Ordered centerline points + directions → **feeds Method B**. (Method A alone does not produce
segments; it produces the ordered curve B operates on.)

---

## 5. Method B — RDP with an MCS-tied threshold  (segmentation)
### 5.1 What it does
Ramer–Douglas–Peucker simplification turns the **ordered** centerline from Method A into a minimal
polyline. The novelty: the tolerance is **not a constant** — it is a function of chord length and
momentum derived from a multiple-Coulomb-scattering model, so MCS wiggle is absorbed into straight
segments while genuine hard scatters force a vertex.

### 5.2 Algorithm (modified, variable-tolerance RDP, recursive)
```
def rdp(P, i, j):                         # P: ordered (M,3); span endpoints i..j
    chord = segment(P[i], P[j])
    d_k, k = max_perp_distance(P[i+1:j], chord)   # 3D point-to-segment distance
    L = ||P[j] - P[i]||
    eps = tolerance(L)                    # <-- MCS-tied, NOT constant
    if d_k > eps:
        return rdp(P, i, k) + rdp(P, k, j)   # keep vertex k, recurse
    else:
        return [P[i], P[j]]               # collapse span to one segment
```
Because `eps = tolerance(L)` depends on the span, **off-the-shelf RDP packages (which assume a
constant ε, and some of which are 2D-only) are not sufficient** — implement a custom n-D RDP with a
callable tolerance.

### 5.3 The MCS-tied tolerance
Highland multiple-scattering angle for path length `x` in material of radiation length `X₀`:
```
θ₀ = (13.6 MeV / (β c p)) · z · sqrt(x / X₀) · [1 + 0.038 · ln(x z² / (X₀ β²))]
```
Transverse displacement RMS over chord length `L` scales like `σ_MCS(L) ≈ L · θ₀(L) / sqrt(3)`
(projected). Total tolerance adds detector resolution in quadrature:
```
eps(L) = κ · sqrt( σ_MCS(L)²  +  σ_reso² )
```
- `X₀ = 14.0 cm` (liquid argon).
- `σ_reso`: per-point reconstruction resolution `[CONFIRM ~ few mm]`; it is the residual floor.
- `p`, `β`: particle momentum. Either supply truth `p` for studies, or estimate from range/dE/dx and
  **iterate** (fit → estimate p → refit). Expose `momentum_source` ("truth" | "range" | "fixed").
- `κ`: significance multiplier (e.g. 3) controlling how aggressively kinks are kept.

**Important composition note:** RDP must run on the **smoothed centerline from Method A**, where the
residuals it sees are centerline-fit residuals (MCS + leftover smoothing). If run on raw cloud
points, `eps` would have to swallow the full tube radius and all kink sensitivity is lost. Keep the
A→B order.

### 5.4 Dependencies & GPU
- Inherently sequential, operates on a small ordered set (hundreds of points post-smoothing) → cheap
  on CPU. Do **not** GPU per-track; batch across instances with multiprocessing if needed.

### 5.5 Failure modes
- Wrong/garbled ordering from A propagates directly → garbage segments. Validate ordering first.
- Bad `p` estimate mis-scales `eps` (too many / too few vertices). Surface the chosen `p` per track
  in `extra`.

---

## 6. Method C — ElPiGraph  (end-to-end elastic principal graph)
### 6.1 What it does
Fits a graph of nodes+edges through the middle of the cloud by minimizing mean-squared distance
regularized by an **elastic energy** with separate **stretching** and **bending** penalties. Output
is already a piecewise-linear graph; for a single track, constrain to a curve/path topology. Handles
centerline + ordering + segmentation in one call, is robust to noise, and (in tree mode) handles
branch points.

### 6.2 API & parameters (`pip install elpigraph-python`)
- `elpigraph.computeElasticPrincipalCurve(X, NumNodes=...)` for single threads (path topology).
- `computeElasticPrincipalTree(...)` only if branched instances are expected.
- Key knobs:
  - `NumNodes`: resolution / approximate segment count. Too few → misses kinks; too many → fits
    noise. Consider scaling with track length / point count.
  - `Lambda` (stretching elasticity), `Mu` (bending rigidity): **higher `Mu` → straighter, smoother**.
    These are the MCS-vs-kink trade-off analog of `eps` in Method B.
  - `TrimmingRadius` (robust mode): ignore points beyond a radius from the graph — useful to reject
    δ-ray contamination.
- After fitting, **trace the path** through `NodePositions` + `Edges` to produce the ordered
  `vertices` for the unified schema.

### 6.3 Dependencies & GPU
- The Python implementation (`sysbio-curie/ElPiGraph.P` / `elpigraph-python`) supports **multi-core
  and GPU**; enable the GPU path via its flag (CuPy backend). GPU helps for large clouds / batching;
  for tiny per-track clouds, launch overhead may dominate → **benchmark CPU vs GPU per typical `n`**.

### 6.4 Failure modes / care
- Elastic energy is **scale-dependent**: normalize each instance's coordinates (e.g. by track extent)
  or scale `Lambda`/`Mu` accordingly, otherwise penalties mean different things on a 5 cm vs 200 cm
  track.
- Tree mode can hallucinate branches on a clean single track → prefer curve mode unless branching is
  expected.
- `NumNodes` is a hyperparameter; consider a model-selection criterion or a length-based heuristic.

---

## 7. Evaluation & comparison harness
Run all three (A+B, and C) on the same truth-matched instances and report per-instance + aggregate:
- **Transverse residual RMS**: cloud points → fitted polyline, and fitted polyline → `true_traj`.
- **Endpoint accuracy**: distance between fitted endpoints and true start/end.
- **Angular resolution at kinks**: angle between adjacent fitted segments vs true scattering angle.
- **Vertex placement**: distance from each fitted break to the nearest true hard-scatter vertex;
  matched/spurious/missed vertex counts (a precision/recall on kinks).
- **Segment-count fidelity**: fitted segments vs true kinks (over/under-segmentation).
- **Throughput**: wall-clock per instance and batched, CPU vs GPU.
Bin metrics by true momentum and by track length (reuse existing kinematic-binning utilities).
Provide a single comparison table + residual/angle histograms per method.

---

## 8. Suggested repo layout
```
trajfit/
  io.py              # load event arrays, group by instance_id, truth-match
  schema.py          # TrajectoryFit dataclass + validation
  geometry.py        # point_to_polyline_distance, arc-length resample, batched PCA
  ordering.py        # global-PCA ordering + kNN/MST fallback
  method_sliding_pca.py
  method_rdp.py      # variable-tolerance n-D RDP
  mcs.py             # Highland model, eps(L) tolerance
  method_elpigraph.py
  evaluate.py        # metrics + kinematic binning
  run_compare.py     # driver: runs all methods on a dataset, emits comparison table
```
Interfaces: each `method_*` exposes `fit(points, weights=None, **cfg) -> TrajectoryFit`. The driver
loops instances, dispatches, collects, evaluates. Build A+B first (it's the lowest-friction GPU-native
path), validate against truth, then add C as the robust comparator.

---

## 9. Open questions for the user to confirm
1. Detector (MicroBooNE / SBND / DUNE) and coordinate convention / units.
2. Exact input array names, dtypes, and whether instances arrive pre-split or as one labeled cloud.
3. Availability and format of charge/feature and of truth (`true_traj`, `true_pdg`, `true_p`).
4. Per-point spatial resolution `σ_reso` and nominal point spacing.
5. Typical `n` per instance and number of instances per event (drives the batching strategy).
6. Whether single instances can contain branches (primary + δ-rays), which decides curve-vs-tree
   mode for ElPiGraph and whether an MST ordering fallback is mandatory for A.
7. Momentum source for the MCS threshold: truth `p` for now, or range-based estimate with iteration?
