# Project Spec: Shower Direction Reconstruction & Nu-Vertex Connection (Prototype)

## 0. Scope & how to use this document
This is a planning/implementation spec for a **prototype** shower reconstruction
that plugs into the existing track+vertex reco
(`trajfit/nu_interaction.py`). The end goal is to attach EM showers
(`cls ∈ {e, gamma}` instances from the particle segmenter) to the reconstructed
neutrino interaction. The hard sub-problem is finding each shower's **trunk
direction**; everything downstream (vertex/keypoint matching) depends on it.

**Deliberately phased — do the cheap, decisive part first:**
- **Phase 1 (this spec's focus):** implement and compare **three shower-direction
  methods**, and evaluate each *only* on how well the shower connects to the
  **nu vertex** (the single most important connection point). Measure direction
  accuracy, connection performance, and compute cost. Decide which method(s) to
  carry forward.
- **Phase 2 (outlined in §8, deferred):** the full iterative attachment of all
  slice showers to *all* interaction connection points (nu vertex + every track
  connection point), greedy vs exhaustive. Do **not** build this until Phase 1
  picks a direction method.

### Resolved design decisions (from review)
- **Trunk anchor = the predicted shower-start keypoint** (each shower instance's
  `start_cm`). It is available at **reco time with no GT** (critical: real data has
  no truth), it marks the trunk start, and it **tags the trunk fragment** (the
  DBSCAN fragment containing / nearest that keypoint). The keypoint was *made* by
  finding the trunk and placing a point at its start, so it is the natural anchor.
- **One trunk per predicted shower instance** — that is what the reco must emit on
  real data. No GT dependence in the reco path.
- **PAF dropped**: the per-point direction field came from the deprecated old
  larmatch network; not used. Method (3) is geometric (PCA) only.
- **e and γ treated the same** for direction; leave a **class-specific hook** on
  the *connection* acceptance (`d_gap`/`cos_min` may later differ by class).
- **dE/dx and energy** deferred (its own project) — no energy binning for now.
- **Truth trunk direction**: for now derive from the trunk fragment tagged by the
  shower-start keypoint (§6); augmenting `merged_sp` with explicit trunk info is a
  later option, mainly for rigorous eval.

The three direction methods to implement and compare (§4):
- **(1) Whole-cluster PCA** — leading principal component of the entire shower
  cluster. Baseline; expected to be noisy.
- **(2) ElPiGraph trunk finding** — skeletonize the shower fragments, extract
  line-like graph segments above a minimum length as trunk candidates, pick the
  one closest to / pointing back at the vertex.
- **(3) Vertex-biased trunk** — port of LANTERN's
  `larflow/larflow/Reco/NuVertexShowerReco.cxx` `_make_trunk_cand`: anchor on the
  cluster point nearest the vertex, PCA the *local* trunk region, orient outward,
  score by vertex alignment.

---

## 1. Physics / domain context
- An EM shower (e±, γ) is a **branching cascade**, not a 1D track. In a LArTPC it
  appears as a roughly conical spray of ionization that **fans out** from a
  narrow **trunk** near the start.
- **The trunk is the only part that carries the original direction.** Past the
  first ~1 radiation length (X₀ ≈ 14 cm in LAr) the shower fans out and PCA of the
  full cloud is pulled toward the (wide) shower body, biasing the direction.
- Showers are **disjoint / fragmented**: gaps between the trunk and later
  fragments (sub-showers, bremsstrahlung photons that re-convert). The segmenter
  may emit a shower as **one instance with several spatial fragments**, or split
  it across **several instances**. Fragment clustering is therefore a pre-step.
- **Photons convert after a gap**: a γ trunk starts at the conversion point, which
  is displaced from the neutrino vertex (the pi0 decay point) by the conversion
  gap — so a shower's trunk **points back to** the vertex but does **not touch**
  it. This is why connection is by *direction* (impact parameter + back-pointing),
  not proximity, unlike track endpoints.
- **The trunk can be short** (early brem / hard e-scatter), which is exactly when
  whole-cluster PCA fails worst and a local/vertex-biased estimate matters most.

---

## 2. Input data
Per event, from `keypoint2_out` (see `trajectory_fitting_brief.md` §2):
- **Shower instances**: particle groups with `cls ∈ {0:e, 1:gamma}`. Each has
  `point_idx` into `slice/coord_cm` → the shower cloud. Loadable via
  `trajfit_io.load_instances(..., tracks_only=False)` then filter
  `cls in SHOWER_CLASSES` (already defined in `trajfit_io.py`).
- **Predicted shower-start keypoint** (the trunk anchor): each shower instance's
  `start_cm` — the per-particle keypoint model's predicted start. Available on
  real data (no GT). All methods use this as the trunk start / fragment tag.
- **Per-shower-instance truth** (for eval only): `gt_point_idx` (matched true
  cloud), `gt_start_cm` (true visible start = the trunk-start keypoint; for a γ
  the conversion point), `gt_trackid`, `iou`.
- **Vertex**: from the score-field fitter (`reco.KeypointRecoTorch`) as already
  wired in `nu_interaction.py` (`vertex_candidates`), or `gt_nu_vertex_cm`.
- **Shower keypoints**: `gt_keypoints` types include `3 = shower-start` (and
  `1 = track_start`); usable as a truth anchor for the trunk start.

**Not available in the current `keypoint2_out`** (note for method design):
- No **per-point direction field** — the LANTERN "PAF" came from the deprecated
  old larmatch network and is intentionally not used. Method (3) is PCA-only.
- No **per-point charge** in the slice (charge needs `merged_sp/triplet_data`
  via the KD-match in `trajfit_io`; `merged_sp` is currently empty for the pi0
  set). Treat charge-weighting as optional/when-available.
- True **trunk direction** is not stored directly — derive it for eval (§6).

**Fragment clustering pre-step (shared by all methods):** DBSCAN the shower
instance's points (reuse `cluster_fit_stitch.fit_fragments`, `eps≈1.5–2.5 cm`) to
get continuous fragments. Note: a slightly larger `eps` than track reco, because
shower fragments are sparser. Keep the fragment list per shower instance.

---

## 3. Unified shower-direction output (all three methods return this)
```python
@dataclass
class ShowerTrunk:
    start: np.ndarray        # (3,) trunk start point, cm (most upstream / vertex-near)
    direction: np.ndarray    # (3,) unit trunk direction, pointing AWAY from start
    trunk_points: np.ndarray # (K,3) points used to fit the direction
    length_cm: float         # extent of the trunk region used
    n_cluster: int           # points in the whole shower cluster
    quality: float           # method-specific direction-confidence (e.g. PCA elongation)
    method: str              # "pca" | "elpigraph" | "vertex_biased"
    runtime_s: float
    extra: dict
```
`direction` is oriented so that `start` is the upstream end (trunk start) and
`direction` points down the shower. For methods that need a vertex (3, and the
orientation step of 1 & 2), the orientation is "away from the vertex".

---

## 4. The three direction methods

All methods anchor `start` at the **predicted shower-start keypoint** (`start_cm`)
and orient `direction` to point **away from that start** (down the shower).

### 4.1 Method (1): whole-cluster PCA  *(baseline)*
- Charge-weighted (if available) 3D PCA of the **entire** shower cluster.
- `direction` = leading eigenvector, oriented away from the shower-start keypoint
  (sign chosen so `direction·(cluster_centroid − start) > 0`); `start` = the
  shower-start keypoint.
- `quality` = `λ0/(λ0+λ1+λ2)` (elongation; low for fanned showers).
- **Params:** none beyond charge weighting.
- **Compute:** one eigendecomposition; ~µs–ms. Cheapest.
- **Failure modes:** the shower body dominates the covariance → direction biased
  off the true trunk, especially for short-trunk / wide showers. This is the
  baseline we expect everything to beat; quantify *by how much*.

### 4.2 Method (2): ElPiGraph trunk finding
- **Skeletonize** each shower fragment with ElPiGraph
  (`elpigraph.computeElasticPrincipalTree` for branchy fragments, or
  `computeElasticPrincipalCurve` for simple ones; reuse `run_elpigraph.py`
  helpers and `trace_path`).
- From the node+edge graph, **extract line-like segments**: walk the graph and
  collect paths whose length ≥ `min_trunk_len` (e.g. 3–5 cm) and whose local
  straightness (residual / bending) is below a threshold → **trunk candidates**.
- **Select** the trunk candidate whose end is nearest the **shower-start
  keypoint** (the keypoint tags the trunk; no vertex needed). Tie-break by length.
- `direction` = candidate segment direction (oriented away from the start
  keypoint); `start` = the shower-start keypoint (or the candidate end snapped to
  it).
- **Params:** ElPiGraph `NumNodes`/`Lambda`/`Mu` (scale with fragment size),
  `min_trunk_len`, straightness threshold, `TrimmingRadius` for δ-ray rejection.
- **Compute:** ElPiGraph per fragment — the expensive option (10s–100s ms per
  cluster; see `run_elpigraph.py` timings). Measure carefully; this is the
  method whose cost we most need to justify.
- **Failure modes:** tree mode hallucinating branches on noise; `NumNodes`
  sensitivity; graph fragments not spanning a real gap. Curve-vs-tree choice
  matters.

### 4.3 Method (3): vertex-biased local trunk  *(LANTERN port)*
Port of `larflow/larflow/Reco/NuVertexShowerReco.cxx::_make_trunk_cand`
(read for reference). Algorithm, given the shower cluster and a vertex `V`:
1. Find the cluster hit **closest to `V`** → `minpos` (the trunk start anchor).
2. Collect **local hits** within `trunk_maxdist_from_closest_cm` (~5 cm) of
   `minpos` — the trunk region only, *not* the whole cluster.
3. Sub-cluster the local hits (DBSCAN) into trunk-candidate fragments.
4. For each candidate with ≥5 points: **PCA**; direction = leading axis, flipped
   to point from `minpos` toward the candidate centroid (outward).
5. **Score** each candidate by alignment of its PCA axis with the
   **vertex→minpos** line (`score = (V→minpos)·pca1`); the best-aligned candidate
   (points back to the vertex) wins.
6. Output `start = minpos`, `direction = pca1` of the winner.
- **Params:** `trunk_maxdist_from_closest_cm` (local-region radius), sub-cluster
  `eps`, min points.
- **Compute:** one nearest-search + local PCA(s); ~ms. Cheap, like (1), but uses
  only the local trunk region + vertex anchor → expected to be much more accurate
  than (1) for short trunks.
- **Note vs LANTERN:** the C++ also blends a **PAF** (network per-point direction)
  estimate and an XGBoost prong score; we **drop both** (no PAF feature in our
  data, no BDT) and keep the geometric PCA branch. Flag this as a known
  simplification.
- **Failure modes:** sensitive to vertex quality (a bad vertex anchors the trunk
  in the wrong place — couples to the vertex error we already measure); local
  radius too small → too few points, too large → includes the fan.

---

## 5. Connection to the nu vertex (Phase 1 evaluation target)
For each shower trunk `(start, direction)` and the nu vertex `V`, compute the same
geometric quantities LANTERN uses (`RecoShowerInfo_t`):
- **gap / dist2vtx** = `|start − V|` (the conversion gap; can be large for γ).
- **impact parameter** = perpendicular distance from `V` to the trunk **line**
  `{start + s·direction}` (small if the trunk points back at the vertex).
- **back-pointing cosine** = `cos∠(direction, start − V)` (≈ +1 if the trunk
  heads away from the vertex, i.e. originates there).
- **Connection decision (greedy, Phase 1):** attach to the vertex if
  `impact_par ≤ d_impact` AND `cosine ≥ cos_min` AND `gap ≤ d_gap`. Tune these on
  truth. (LANTERN's coarse legacy cut was `impactdist < 20 cm && nhits > 10`.)

This mirrors the track-attachment test in `nu_interaction.py` (`attach_cost`) but
with a **line/impact-parameter** test instead of an endpoint-proximity test —
because showers connect by pointing-back-across-a-gap, not by touching.

---

## 6. Truth & evaluation
**Truth sources** (per shower instance / true shower particle):
- True **trunk start**: `gt_start_cm` (the trained visible start = γ conversion
  point) and/or the `gt_keypoints` type-3 shower-start.
- True **trunk direction** (keypoint2-only, the agreed default for now): the GT
  trunk is the cluster region tagged by the **GT shower-start keypoint** —
  `normalize(centroid_of_GT_points_within_R_of(gt_start_cm) − gt_start_cm)`,
  `R ≈ 5 cm`. (Augmenting `merged_sp` with `shower_fragments/istrunk` to get the
  exact `_true_trunkdir` is a later option for rigorous eval.)
- True **vertex**: `gt_nu_vertex_cm`.

**Metrics (per shower, then aggregate):**
- **Angular error**: `∠(reco direction, true trunk direction)` — the headline
  number. Report median + tail (e.g. 68%/95%).
- **Trunk-start error**: `|reco start − true start|`.
- **Impact parameter & cosine vs vertex** (reco and true), to characterize the
  connection geometry.
- **Connection performance vs the nu vertex**: with truth of which showers
  actually originate at the vertex, report **precision/recall** of the greedy
  connection decision, swept over `d_impact`/`cos_min`.
- **Compute cost**: wall-clock per shower and per event, each method, CPU (and
  GPU for ElPiGraph if it helps) — the cost half of the cost/accuracy trade.
- **Bins**: by trunk length and conversion gap — the regimes where methods
  diverge (short trunk, large gap). Energy binning deferred (needs dE/dx, a
  separate project).

**Deliverable:** a single comparison table (method × {median angular err, 95%
angular err, connection precision/recall at a fixed working point, ms/shower})
plus angular-error and impact-parameter histograms per method. This is what
decides which method(s) advance to Phase 2.

---

## 7. Suggested code layout (prototype, under `trajfit/`)
```
trajfit/
  shower_trunk.py     # ShowerTrunk dataclass; fragment clustering; the 3 methods:
                      #   trunk_pca(), trunk_elpigraph(), trunk_vertex_biased()
  shower_connect.py   # impact-parameter / cosine / gap vs a point; greedy nu-vertex
                      #   connection decision
  run_shower_dir.py   # driver: load shower instances, run all 3 methods, evaluate
                      #   vs truth, emit the comparison table + histograms + a
                      #   plotly viz (cluster + 3 trunk arrows + true trunk + vertex)
```
Reuse: `trajfit_io` (loading, truth clouds), `cluster_fit_stitch.fit_fragments`
(DBSCAN), `run_elpigraph` (ElPiGraph + `trace_path`), `mcs_rdp`/geometry helpers,
and the `nu_interaction.py` viz patterns (synced 2-panel, GT overlay).
Integration target: a `shower_trunk` field usable by `nu_interaction.py` so
showers become attachable objects in the interaction tree (Phase 2).

---

## 8. Phase 2 (deferred — outline only)
Once a direction method is chosen, generalize connection from "nu vertex only" to
the full interaction:
- **Connection points** = nu vertex + every track connection point (track
  endpoints / kinks already produced by the track reco).
- **Ordering**: process connection points **closest-to-the-nu-vertex first**,
  outward (the user's proposed order), mirroring the BFS in `reco_interaction`.
- **Greedy** scheme: attach a shower to the first connection point it points back
  to within tolerance.
- **Exhaustive** scheme: score a shower against *all* connection points and take
  the best; compare cost & correctness vs greedy (does greedy lose much?).
- Showers then become nodes in the same interaction tree the tracks build, so the
  "what reco misses" GT panel (already in `nu_interaction.py`) directly measures
  shower-attachment completeness.

---

## 9. Resolved decisions (was: open questions)
1. **Truth trunk**: tag the trunk cluster with the shower-start keypoint; derive
   true direction from that fragment (§6). `merged_sp/shower_fragments` augmentation
   deferred (a later eval refinement).
2. **PAF**: deprecated (old larmatch) — not used. Method (3) is PCA-only.
3. **Energy / dE-dx**: deferred (separate project). No energy binning.
4. **One trunk per predicted shower instance** — matches what the reco emits on
   real data; no GT dependence in the reco path.
5. **e vs γ**: same direction treatment; keep a **class-specific hook** on the
   connection acceptance (`d_gap`/`cos_min`) for later tuning.
