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

## 2. Input data specification

### 2.0 Detector, units, coordinate frame (confirmed)
- **Detector: MicroBooNE** (single 2.56 × 2.3 × 10.4 m LArTPC, 3 induction/collection wireplanes,
  32 PMTs). Dev sample is `bnb_nu_overlay` (BNB neutrino MC overlaid on cosmic data).
- **Units: centimeters** throughout (`*_cm`, `pos`, `start_pos`, keypoint `pos`). The detector
  coordinate convention is MicroBooNE `(x, y, z)`: x = drift (0–256 cm), y = vertical
  (−116…+116 cm), z = beam (0–1036 cm). All spacepoints carry **space-charge-corrected (SCE)**
  reco positions; the matching truth field is `start_pos_sce`.
- **Per-point spacing ≈ 3.3 mm** (median NN distance in a slice ≈ 0.33 cm; this is the spacepoint
  granularity set by the 3 mm MicroBooNE wire pitch). Use this for `σ_reso ≈ 3 mm` (the residual
  floor in §5.3) and as the natural lower bound for the sliding-window scale and ElPiGraph node
  spacing.
- **X₀ = 14.0 cm** (liquid argon) is correct for MicroBooNE; resolves the `[CONFIRM detector]` in §1.

### 2.1 The two file types and how they relate
Trajectory fitting consumes **`keypoint2_out/`** (one H5 per event, the cascade-inference product of
`tools/run_larformer_keypoint2_cascade_inference.py`). **`merged_sp/`** (one H5 per event, the raw
upstream event the cascade ran on) is the **truth source**; it is *optional* for fitting and *required
only* for richer evaluation (charge weights, full MC tree, true kink vertices). A `keypoint2_out`
file names its parent via the root attr `src_file` (= the basename of the `merged_sp` file).

**Critical fact (verified):** every point in `keypoint2_out:/slice/coord_cm` is an **exact** member of
`merged_sp:/entry_0/triplet_data/pos` (nearest-neighbour distance = 0.0). So per-point charge and
per-point truth can be recovered by a KD-tree position match `slice → triplet_data` (build the tree
once per event). No index map is stored; match on coordinates.

### 2.2 `keypoint2_out/keypoint2_event{NNNNN}_{ei}.h5` — fitting input (self-contained)
This is what each method ingests. **It alone is sufficient to run a fit and do a first-order
evaluation** (it carries the per-instance cloud *and* GT endpoints + GT cloud).

Root attrs: `n_particles:int`, `has_gt:bool`, `has_score_maps:bool`, `src_file:str` (parent
`merged_sp` basename); plus `run/subrun/event` when present.

| Path | Shape / dtype | Meaning |
|---|---|---|
| `slice/coord_cm` | `(N_slice, 3) f32` | The nu-slice spacepoint cloud in detector cm. **All `*point_idx` index into this array.** N_slice ~ O(2k). |
| `nu_vertex_cm` | `(3,) f32` | Predicted neutrino vertex (dense-head, score-weighted centroid). |
| `gt_nu_vertex_cm` | `(3,) f32` | True neutrino vertex (NaN if no MC). |
| `particle/{i}/` (attrs) | — | `cls:int` (predicted particle class, **see §2.4**), `gt_trackid:int`, `has_match:bool`, `iou:f32` (pred-vs-GT point IoU). |
| `particle/{i}/point_idx` | `(n_i,) i32` | **Indices into `slice/coord_cm` of this predicted instance.** `cloud = slice_coord[point_idx]` → the input to `fit()`. n_i ranges ~11–2100 here (median ~220). |
| `particle/{i}/start_cm` | `(3,) f32` | **Predicted** start keypoint (cm). |
| `particle/{i}/end_cm` | `(3,) f32` | **Predicted** end keypoint (cm); NaN if the model emitted no end (e.g. showers). |
| `particle/{i}/gt_point_idx` | `(m_i,) i32` | Indices into `slice/coord_cm` of the **matched GT** particle (per-point `trackid == gt_trackid`). Use `slice_coord[gt_point_idx]` as the *truth cloud* for residual eval. |
| `particle/{i}/gt_start_cm`, `gt_end_cm` | `(3,) f32` | **True** start/end of the matched particle (NaN if unavailable). NB: `gt_start` is the per-instance *visible/loss* start (origin for the trained head), not necessarily the photon conversion point. |
| `score_maps/{object,nu_vertex}/` | coords+score | (only if `has_score_maps`) dense keypoint head scores — keypoint diagnostics, **not needed** for trajectory fitting. |
| `gt_keypoints/{pos_cm,type,trackid}` | `(K,3)f32,(K,)i32,(K,)i32` | All raw MC keypoints for the event (types in §2.5). |

**Instances arrive pre-split, as point-index lists into one shared slice cloud** (answers §9.2). To
fit a track you take one `particle/{i}`, gather `slice_coord[point_idx]`, and (optionally) attach
charge/weights from `merged_sp`. There is **no per-point feature/charge stored in `keypoint2_out`** —
only geometry.

### 2.3 `merged_sp/..._entry{NNNNNN}.h5` — truth + charge source (optional enrichment)
One top group `entry_0` (attrs: `run, subrun, event, trigger_tick, usec_per_tick, n_pmts`). Relevant
subgroups for trajectory fitting:

- **`triplet_data/`** — the full reconstructed spacepoint cloud the cascade saw (`pos (M,3) f32`,
  superset of the slice; M ~ O(1e5), includes ghosts). Per-point fields usable as features/weights
  after the slice→triplet match: `pixval (M,3) f32` (per-plane wire charge U/V/Y — **the charge for
  weighting**; use Y or the mean), `trackid (M,) i64` (true GEANT trackid, −1/0 = ghost/none),
  `pid (M,) i32`, `edep (M,3) f32`, `lm_score`, `ssnet_label`, `kpscores (M,6)`.
- **`triplet_truth/`** — the **true (non-ghost) energy-deposit cloud**: `pos_reco (T,3) f32` (reco
  position), `trackid (T,) i64`, `pid`, `edep`. Use as an alternative, cleaner truth cloud per
  trackid for residual metrics.
- **`mc_particle_tree/`** — the per-particle MC truth (length = #particles): `trackid (P,) i32`,
  `pid (P,) i32` (PDG), `energy_mev (P,) f32`, `start_pos (P,3)` / `start_pos_sce (P,3)` f32,
  `parent_trackid (P,) i32`, `daughter_trackids`, `daughter_start_indices`, `num_daughters`,
  `process_code`, `origin`, `nu_vertices (1,3)`. **This is the source for `true_pdg`, `true_p`, and
  true kink/vertex locations** (see §2.6).
- **`mckeypoints/`** — labelled truth keypoints (length K): `pos (K,3) f32`, `kptype (K,) i32`
  (types in §2.5), `pid (K,) i32`, `trackid (K,) i32`, `startpos (K,3)`, `imgcoord (K,4)`. Per-track
  **true start (type 1) and end (type 2)** for endpoint metrics, keyed by `trackid`.
- Also present but not needed for tracks: `image_data/` (raw wireplane images), `flashes/`,
  `pmt_positions`, `shower_fragments/`.

### 2.4 Particle class codes (the `cls` attr) — which instances are tracks
The predicted `cls` comes from the Stage-3 particle segmenter, whose class list is
`["e", "gamma", "mu", "pi", "p", "other", "(unused)", "no_object"]`:

| cls | 0 | 1 | 2 | 3 | 4 | 5 | 7 |
|---|---|---|---|---|---|---|---|
| name | e | gamma | mu | pi | p | other | no_object |
| topology | shower | shower | **track** | **track** | **track** | (mixed) | — |

**Track reconstruction selects instances with `cls ∈ {2, 3, 4}`** (mu, pi, p). Showers
(`cls ∈ {0, 1}`) are deferred to a later project; the optional "trunk direction / mis-ID" study in
the brief preamble would run the same track fitter on `cls ∈ {0,1}` clouds.

### 2.5 Keypoint type codes (`mckeypoints/kptype`, `gt_keypoints/type`)
`0 = nu_vertex`, `1 = track_start` (also used as shower-start/conversion), `2 = track_end`,
`3 = shower-start`, … For tracks use type 1 (start) and type 2 (end) keyed by `trackid`.

### 2.6 Truth available for evaluation — and the key limitation
There is **NO stored GEANT4 step polyline** (`true_traj` does not exist as an array). For §7 metrics,
synthesize truth from what *is* stored:
- **Endpoint accuracy** → `mckeypoints` type-1/type-2 `pos` keyed by `trackid` (or the
  `gt_start_cm`/`gt_end_cm` already decoded into `keypoint2_out`).
- **Transverse residual** → fit the cloud, then measure fitted-polyline-to-**true-cloud** distance
  using the per-instance truth cloud (`slice_coord[gt_point_idx]`, or `triplet_truth` filtered by
  `trackid`). This replaces the missing `true_traj` for residual/RMS metrics.
- **True kink / hard-scatter vertices** → derive from `mc_particle_tree`: a daughter's
  `start_pos_sce` that lies on the parent's path is a scatter/decay vertex. Concretely, for a parent
  `trackid t`, collect the `start_pos_sce` of every particle whose `parent_trackid == t` (and of `t`
  itself for the production vertex); cluster coincident positions. In the dev event, trackid 2's
  daughters all start at `(231.1, 19.4, 140.8)` — a single hadronic-interaction vertex. These are the
  truth break points for vertex precision/recall and segment-count fidelity.
- **Momentum `true_p`** for the MCS tolerance (§5.3) → from `mc_particle_tree`: `p = sqrt(E_kin² +
  2 E_kin m)` using `energy_mev` (kinetic) and the PDG mass from `pid`. Expose `momentum_source ∈
  {truth, range, fixed}`; start with truth.

A small loader (`trajfit/io.py`, dev prototype in §10) should: open a `keypoint2_out` file, optionally
open its `src_file` sibling under a given `merged_sp` dir, KD-match the slice into `triplet_data` once,
and yield per-instance records `{points, charge, weights, pred_cls, pred_start/end, gt_start/end,
truth_cloud, trackid, pid, true_p, true_kinks}`.

### 2.7 Per-event scale (drives the batching strategy, §9.5)
Dev sample (8 events with a nu slice): **0–6 particle instances per event**, instance sizes
**~11 to ~2100 points (median ~220)**, slice cloud ~2k points. These are *small* clouds → per-track
ElPiGraph/PCA is cheap; GPU value (if any) comes from batching many instances, not from one track.

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

## 9. Open questions — status after the §2 data investigation
1. **Detector / units — RESOLVED.** MicroBooNE; centimeters; SCE-corrected detector `(x,y,z)`. X₀ = 14.0 cm. (§2.0)
2. **Array names / pre-split — RESOLVED.** Schema in §2.2–2.3. Instances arrive **pre-split** as
   `point_idx` lists into one shared `slice/coord_cm` cloud. (§2.2)
3. **Charge / truth — PARTIALLY RESOLVED.** Charge = `triplet_data/pixval` (3-plane), attached via
   the slice→triplet KD-match (not stored in `keypoint2_out`). `true_pdg`/`true_p` available from
   `mc_particle_tree`. **`true_traj` (GEANT step polyline) does NOT exist** — substitute the true
   point cloud + true kink vertices for residual/vertex metrics (§2.6). *Confirm this substitution is
   acceptable, or point us at a step-level truth product if one exists upstream.*
4. **σ_reso / spacing — RESOLVED.** Point spacing ≈ 3.3 mm; use `σ_reso ≈ 3 mm`. (§2.0)
5. **Per-instance / per-event scale — RESOLVED (dev sample).** 0–6 instances/event, ~11–2100
   points/instance, median ~220; slice ~2k points → small clouds, batch across instances for GPU. (§2.7)
   *Confirm whether production events (full BNB, more cosmics) are larger.*
6. **Branches within an instance — OPEN.** δ-rays can in principle merge into a primary instance;
   the segmenter aims for one particle per instance but is imperfect (see the low-IoU matches in the
   dev event). Recommend **curve mode** for ElPiGraph by default with `TrimmingRadius` to reject
   δ-ray contamination, and keep the MST ordering fallback for Method A. *User to confirm expected
   branch rate.*
7. **Momentum source — RECOMMEND truth first.** `true_p` from `mc_particle_tree` (§2.6) for initial
   studies; add range-based iteration later. (§5.3)

---

## 10. Quick-start dev scripts (ElPiGraph spike) — `trajfit/`
A standalone spike lives in `lartpc/larformer_reco/trajfit/` for trying
ElPiGraph on this data **before** committing to the full `trajfit/` package (§8):
- `trajfit_io.py` — the §2.6 loader: yields per-instance track records from a `keypoint2_out` file
  (+ optional `merged_sp` enrichment for charge/truth).
- `run_elpigraph.py` — fits `computeElasticPrincipalCurve` on track-class instances, traces the node
  graph into an ordered polyline, reports runtime (CPU vs `GPU=True`), residual RMS, endpoint error,
  and dumps a Plotly HTML per instance (cloud + fitted polyline + GT endpoints).
- `sweep_elpigraph.py` — grids `NumNodes`/`Lambda`/`Mu`/`TrimmingRadius` over the dev events and
  tabulates residual / endpoint error / #segments / runtime to tune the meta-parameters.
Run inside the container, e.g.
`./run_in_local_pointcept_container.sh python lartpc/larformer_reco/trajfit/run_elpigraph.py`.
