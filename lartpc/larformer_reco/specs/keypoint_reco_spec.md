# Keypoint Reconstruction from LArFormer Cascade Score Maps — PyTorch Spec

Status: **v1 IMPLEMENTED (2026-06-27)** — score-field path in
[`reco/`](reco/) (`reco/README.md`). Synthetic test suite passes (incl. the
truncated / one-sided / off-support cases that drive the fitter choice). The
vote/offset head (§4.1 Method 3) remains future work.

This document specifies a PyTorch reconstruction tool that turns the **dense
keypoint score maps** produced by the attempt-2 LArFormer keypoint cascade into
discrete reconstructed keypoints (nu vertex + generic keypoints). It is a port
and adaptation of the C++ greedy peak-finder in
`larflow/larflow/Reco/KeypointReco.{h,cxx}` (`larflow::reco::KeypointReco`),
re-targeted from the *larmatch* `larflow3dhit` score columns to the larformer
cascade's dense per-level score outputs.

---

## 1. Motivation

The cascade inference tool
(`tools/run_larformer_keypoint2_cascade_inference.py`) already emits a
*single* decoded nu vertex (score-weighted centroid of spacepoints above a
threshold) and *per-particle* start/end points (from the query decoder). What it
does **not** do is reconstruct **all** keypoints directly from the dense score
fields the network predicts — i.e. find every local peak in the nu-vertex
spacepoint score map and in the voxel-level "object" keypoint score map.

The larflow C++ `KeypointReco` solves exactly this peak-finding problem for the
older larmatch network: it iteratively (1) finds the max-score point, (2)
isolates nearby points, (3) fits a 3D Gaussian to localize the peak sub-voxel,
(4) subtracts that Gaussian from the score field, and (5) repeats until the
residual max falls below threshold. We want the same algorithm, in PyTorch,
operating on the larformer cascade outputs.

This matters because the network's GT scores are *generated* as a sum/max of
Gaussians (`lartpc/data_prep/labels/keypoint_labels.py`):

```
kpscore_t(sp) = exp( -0.5 * (d_t / sigma)^2 )   for d_t = ||sp - nearest kp of type t||
sigma = 3.0 cm  (production default)
```

So the predicted score field is (ideally) a superposition of 3 cm Gaussians, one
per true keypoint. The reconstruction **inverts that generative model** by
peeling off one Gaussian at a time — which is precisely what `KeypointReco`
does. Using the *same* σ = 3 cm the labels were built with is the natural choice.

---

## 2. What the C++ `KeypointReco` does (reference algorithm)

Source: `larflow/larflow/Reco/KeypointReco.cxx`. Key entry points and steps:

| C++ piece | What it does |
|---|---|
| `_make_initial_pt_data` | Skim `larflow3dhit`s, keep those with keypoint-score column `[_lfhit_score_index] > kp_thresh` AND larmatch-score `[9] > lm_thresh`; store `(x,y,z, kp_score, lm_score)`. |
| `_skim_remaining_points` | Of the surviving points, keep those whose **current** score (decremented by prior subtractions) is still above threshold. |
| `cluster_sdbscan_spacepoints` | DBSCAN the skimmed points (`_max_dbscan_dist`, `min_cluster_size`) into candidate keypoint clusters. |
| `_characterize_cluster` | Per cluster: score×charge-weighted centroid, PCA, and the **max-score point**. |
| `_fit_cluster_CARUANA` | Localize the peak by fitting a Gaussian to the score pattern via **Caruana's method** (parabola fit to `ln(score)` per axis), solving a 3×3 moment system per dimension → `mean[dim]`. |
| score subtraction | For each point in the cluster: `new = score − max_score · exp(−0.5·d²/σ²)` where `d` = distance to the fitted center, `σ = _sigma` (bandwidth, default 5 cm). Clamp at 0 and mark "used". |
| `process` loop | Repeat `_num_passes` times, each pass re-skims by the current (decremented) score and re-clusters. |

The peeling is what enforces non-maximum suppression: once a keypoint is found,
its Gaussian is removed so the next iteration finds the *next* peak rather than a
neighbor of the same one.

### Caruana fit (the core math we keep)

For a 1-D Gaussian `y = A·exp(−(x−μ)²/2σ²)`, taking `ln y` gives a parabola in
`x`. Caruana solves the linear system for the parabola coefficients from data
moments. The C++ builds, per axis:

```
A = [[ N,        Σx,    Σx² ],
     [ Σx,       Σx²,   Σx³ ],
     [ Σx²,      Σx³,   Σx⁴ ]]
b = [ Σ ln y,   Σ x·ln y,   Σ x²·ln y ]
sol = A⁻¹ b   →   μ = −sol[1] / (2·sol[2]),   σ_fit = sqrt(−1/(2·sol[2]))
```

(Solved independently for x, y, z because the GT Gaussian is isotropic /
uncorrelated by construction.) See `_fit_cluster_CARUANA` for the goodness-of-fit
(`rmse`, `rsqr`) bookkeeping we will mirror.

---

## 3. Inputs — the cascade dense score maps

The new tool consumes the **score-map** output of
`tools/run_larformer_keypoint2_cascade_inference.py --save-score-maps`
(see `_decode_event` / `_write_event_h5` in that file). Per event H5:

```
slice/coord_cm                (N_slice, 3)   nu-slice spacepoints, detector cm
nu_vertex_cm                  (3,)           the existing single-centroid decode
gt_nu_vertex_cm               (3,)           GT (sim only)
score_maps/<head>/            group per dense head:
    coords_cm                 (M_head, 3)    token positions, detector cm
    score                     (M_head,)      sigmoid score in [0,1]
    attrs.level               e.g. "spacepoint" or "ptv3_dec2"
    attrs.kp_types            int list, the GT keypoint types this head targets
gt_keypoints/pos_cm,type,trackid             all GT mckeypoints (sim only)
```

Two heads are the inputs to this reco (from
`larformer-keypoint2-fullcascade.py` / the slice config's
`level_keypoint_heads`):

1. **`nu_vertex`** — `level = spacepoint`, `kp_types = [0]`. **Full spacepoint
   resolution** score field for the nu interaction vertex. `coords_cm` ==
   `slice/coord_cm` (one score per spacepoint).
2. **`object`** — `level = ptv3_dec2`, `kp_types = [1,2,3,4,5]`
   (track_start/track_end/shower/michel/delta lumped into one *object /
   no-object* score). **Lower, voxelized resolution** (one score per dec2 token,
   ~few-cm voxels). This is type-agnostic: it localizes *where* keypoints are,
   not *which* type (typing is the per-particle query decoder's job).

Keypoint type enumeration (`keypoint_labels.py`): `0 nu_vertex, 1 track_start,
2 track_end, 3 shower, 4 michel, 5 delta`.

**Coordinate frame:** both score maps' `coords_cm` are already in **detector cm**
(the inference applies the per-event affine `to_cm` from
`keypoint_eval._recover_affine`). The reco therefore works **entirely in cm** —
radius 10 cm, σ 3 cm, threshold 0.67 apply directly with no normalization. No
frame handling is needed inside the reco.

> If running the reco *inline* in the inference tool (option B, §6), the same
> two `score`/`coords_cm` arrays are available pre-H5 in `ev["level_kp"][name]`
> (`score`, `coords`) — `coords` is in recentered `coord_norm` there, so apply
> `to_cm` first, exactly as `_decode_event` already does for `score_maps`.

---

## 4. Algorithm (this tool)

Greedy iterative peeling, per score map, in cm. This is `KeypointReco` with
DBSCAN clustering replaced by **radius isolation around the running max** and σ
**fixed** at the label value (the user-specified simplification).

```
INPUT:  coords (M,3) cm,  scores (M,) in [0,1]
PARAMS: radius_cm = 10.0,  sigma_cm = 3.0,  score_thresh = 0.67,
        max_keypoints = K (safety cap),  min_neighbors = 4
STATE:  residual = scores.clone()
OUTPUT: list of keypoints {pos_cm, peak_score, fit_rmse, fit_rsqr, n_support}

loop:
  i*   = argmax(residual)
  smax = residual[i*]
  if smax < score_thresh:  break
  nbr  = { j : ||coords[j] - coords[i*]|| <= radius_cm }      # radius isolate
  if |nbr| < min_neighbors:
      residual[i*] = 0                # lone hot point — suppress, don't fit
      continue
  mu, A, rmse, rsqr = fit_gaussian_fixed_sigma(coords[nbr], residual[nbr],
                                               sigma_cm, seed=coords[i*])
  emit { pos_cm=mu, peak_score=smax, fit_rmse, fit_rsqr, n_support=|nbr| }
  # subtract the fitted Gaussian from ALL points (cheap; support ~ a few sigma)
  residual -= A * exp(-0.5 * ||coords - mu||^2 / sigma_cm^2)
  residual.clamp_(min=0)
  if len(output) >= max_keypoints: break
```

### 4.1 Fixed-σ Gaussian fit (reduced Caruana)

**Caveat that drives the fitter choice — truncated / one-sided support.** The
slicer and particle-segmentation masks frequently keep only *part* of a
keypoint's Gaussian, and track/shower **start** points are intrinsically
**half-Gaussians** (hits exist only on the downstream side; nothing before the
start). Worse, the true peak can lie **outside** the surviving point set when the
start hit itself is masked. This breaks naive moment estimators, but not all the
same way — so it's worth being precise about *which* "moment" fails:

- The **score-weighted centroid** `Σ x·y / Σ y` (the true *distribution* moment)
  is **strongly biased** by one-sided truncation — it is pulled toward the
  visible mass. Demote it to a last-resort fallback only.
- The **Caruana** system (C++ `_fit_cluster_CARUANA`) is **not** a centroid — it
  is a least-squares **parabola fit to `ln y`**, and its vertex `μ = −b/2c` can
  legitimately fall **outside** the data range. So it *can* represent an
  off-support peak. Its truncation problems are instead: (1) it is **unweighted**,
  so noisy low-score tail points get equal say and `ln` amplifies their noise;
  and (2) it must **estimate the curvature `c`** from a one-sided sample, and `μ`
  divides by `c` → a poorly-determined curvature throws the peak far off.

Fixing σ removes the *dominant* pathology (2): the curvature `c = −1/(2σ²)` is
now **known**, so the peak is a **stable linear extrapolation**, fine even from
one side. The methods below build on that, ordered most-robust first.

**Method 1 (recommended for the score-field path) — direct nonlinear weighted
Gaussian fit, σ fixed.** Fit the *shape*, not the centroid, so one-sidedness does
not bias it; weight by score so the noisy tail is down-weighted; let `μ` leave the
convex hull of the points:

```
minimize over (A, mu):   sum_j  w_j * ( y_j - A*exp(-||x_j - mu||^2 / (2 sigma^2)) )^2
    w_j = y_j  (or y_j^2),   y_j = residual[nbr],   sigma fixed = 3 cm
solve by 2-3 Gauss-Newton / Levenberg-Marquardt steps, seeded at mu = coords[i*].
```

Converges in a few steps (4 params: `A`, `mu_xyz`), no `ln` noise amplification,
naturally robust to truncation. This is the default fitter.

**Method 2 (fast closed-form seed / fallback) — fixed-σ, y²-weighted, joint-3D
log-linear fit.** The closed-form approximation to Method 1, and the right way to
do the "Caruana with σ fixed" idea. Two upgrades over the C++: it is **weighted**
(kills problem 1) and **jointly 3D** rather than per-axis (the per-axis C++ fit
assumes a separability that scattered points under an isotropic 3D Gaussian do
not satisfy):

```
zj = ln(yj) + ||xj||^2 / (2 sigma^2)          # = lnA - ||mu||^2/2sigma^2 + (mu·xj)/sigma^2
weighted linear LS (weights wj = yj^2), 4 unknowns jointly in 3D:
    [ c0 ; g ] = argmin  sum_j wj ( zj - c0 - g·xj )^2 ,   g in R^3
    mu = g * sigma^2                              # peak position (may be off-support)
```

One linear solve; use it to seed Method 1, or as the fit itself when speed
dominates.

**Amplitude.** For the subtraction term use the observed peak `smax`
(= C++ `kpc.max_score`), not the fitted `A` — robust against log-fit blow-ups.
With a masked peak `smax` under-estimates the true amplitude, but it only governs
*how much* is peeled, not *where*; the position comes from `mu`.

**Method 3 (proper structural fix — model-side) — a vote / offset (Hough)
head.** The cleanest answer to "keypoint outside the points" and one-sided starts
is to not fit the score field at all: have each point regress a **vector to its
keypoint**, so even a single tail point on one side points back to the true
(off-cloud) location, then cluster the votes. The codebase already has this —
`KeypointOffsetHead`, the per-SP `kp_dense_offset_head` (supervised by the cache
`kpoffsets`), and `keypoint_eval.decode_dense_votes` / `cluster_votes`. The v2
*level* heads are currently **score-only** and `--save-score-maps` writes only
`score`+`coords`, so this needs the keypoint model to emit per-token offsets and
the inference to save them — but it is the most robust observable for exactly the
masked/one-sided failure mode and should be the medium-term direction. (The
per-particle query decoder already does the analogous thing: a zero-init pos head
seeded at an anchor, refined by attention, so it can place a start off the
visible points.)

**Edge-case guards (apply with any method):**
- **One-sidedness flag.** Test whether the neighborhood is one-sided — neighbor
  directions subtend a cone rather than a full ball, or the neighborhood centroid
  is ≳ 1σ from `coords[i*]`. Flag such keypoints `extrapolated=True` with inflated
  uncertainty; prefer the vote (Method 3) there when available.
- **Clamp the extrapolation.** Cap `mu` to `coords[i*] ± a few σ` so a
  near-singular fit cannot fling the peak across the detector.
- Keep the C++ goodness-of-fit outputs (`rmse`, `rsqr` of residual vs fitted
  Gaussian) — low R² is the signal to distrust the fit / fall back.

**Fallback chain (degenerate neighborhoods):** `< min_neighbors` points or a
singular system → score²-weighted centroid (the C++ `center_avg_pt_v` path),
flagged low-confidence. Any `yj <= 0` is excluded from the log fit (already
filtered — it is the residual).

### 4.2 Why fixed σ and radius isolation (vs C++ DBSCAN + variable σ)

- **Fixed σ = 3 cm:** matches the GT label width, so the subtracted Gaussian has
  the same footprint as the object it removes — clean peeling. The C++ variable
  σ (Caruana) was needed because larmatch had no single known width; here we do.
- **Radius isolation vs DBSCAN:** the dense maps are already a smooth field on a
  (mostly) regular voxel/spacepoint support; a fixed radius around the running
  max is sufficient to grab the local peak's support and is trivially vectorized.
  DBSCAN's role (grouping connected above-threshold points) is replaced by the
  subtraction step, which removes a found peak's support before the next
  iteration. Keep DBSCAN out unless validation shows merged peaks.
- **Tight FIT radius vs wide isolation radius (`fit_radius_cm`).** The 10 cm
  isolation radius can engulf a *neighbor* peak, biasing the local Gaussian fit
  toward it. The fit is therefore done on a **tighter** neighborhood
  (`fit_radius_cm`, default 2σ = 6 cm) than the isolation/subtraction radius.
  Validated on synthetic 2σ-separated peaks: a 10 cm fit window pulls the mean
  3.2 cm off; the 6 cm window holds it to <1 cm and both peaks are resolved
  (`test_keypoint_reco.test_overlapping_peaks`). The off-support extrapolation
  still works because the masked-peak tail is sampled within `fit_radius_cm`.

### 4.3 Per-map specialization

- **nu vertex (`nu_vertex` head, spacepoint res):** physically **one** vertex per
  neutrino event. Run the loop but typically `max_keypoints` small (e.g. 3) and
  report the highest-score peak as *the* nu vertex (mirrors
  `decode_nu_vertex`). Keep the full list for multi-vertex / pile-up studies.
  Full-res support can be 4k–11k points → use a KDTree/torch radius query
  (§5.2), not an O(M²) `cdist`.
- **keypoints (`object` head, dec2 res):** expect **many** peaks (every visible
  particle start/end). `max_keypoints` larger (e.g. 64). Output is **type-less**
  (the object head doesn't separate types); optional post-hoc typing in §7.

---

## 5. Module / code layout

New package under the v2 spec dir, no larflow/ROOT dependency (pure
torch + numpy + scipy):

```
lartpc/larformer_reco/keypoint/
├── __init__.py
├── keypoint_reco.py        # KeypointRecoTorch (the algorithm + params)
├── gaussian_fit.py         # fit_gaussian_nls (Method 1, default), fit_gaussian_loglinear
│                           #   (Method 2, fixed-sigma y^2-weighted joint-3D), caruana_full (parity)
├── io.py                   # load_score_maps(h5) -> {name: (coords_cm, score, level, kp_types)}
└── run_keypoint_reco.py    # CLI: dir of *.h5 -> reco keypoints (+ optional eval/H5 out)
```

### 5.1 `KeypointRecoTorch` (mirrors the C++ class surface)

```python
@dataclass
class KeypointRecoParams:
    radius_cm: float = 10.0          # neighborhood isolation radius
    fit_radius_cm: float | None = None  # tight FIT neighborhood (None -> 2*sigma)
    sigma_cm: float = 3.0            # fixed Gaussian width (= GT label sigma)
    score_thresh: float = 0.67       # stop when running max < this
    max_keypoints: int = 64          # safety cap per map
    min_neighbors: int = 4           # below this: suppress point, no fit
    amplitude: str = "peak"          # subtraction amplitude: "peak" (C++) | "fit"
    fit_method: str = "nls"          # "nls" (Method 1) | "loglinear" (Method 2) | "centroid"
    clamp_extrap_sigma: float = 3.0  # cap |mu - argmax| at this * sigma_cm
    device: str = "cpu"              # "cuda" for big nu-slice maps

class Keypoint:                      # ~ KPCluster, trimmed to what we have
    pos_cm: np.ndarray   # (3,) fitted mean
    peak_score: float    # max residual score at detection (~ KPCluster.max_score)
    fit_rmse: np.ndarray # (3,)  ~ center_pt_rmse_v
    fit_rsqr: np.ndarray # (3,)  ~ center_pt_rsqr_v
    n_support: int       # neighborhood size used for the fit
    extrapolated: bool   # peak placed outside / from one-sided support (§4.1 guard)

class KeypointRecoTorch:
    def __init__(self, params: KeypointRecoParams): ...
    def reconstruct(self, coords_cm, scores) -> list[Keypoint]:
        """Greedy peeling on ONE score map (§4)."""
    def reconstruct_event(self, score_maps) -> dict:
        """Run on nu_vertex (-> single best + list) and object (-> list)."""
```

### 5.2 Performance notes

- The greedy outer loop is inherently **sequential** (each subtraction changes
  the next argmax). Per iteration the work — radius query + fit + subtract — is
  vectorized.
- Radius queries: build a `scipy.spatial.cKDTree` once on `coords_cm` (static)
  and `query_ball_point` per iteration (reuse the existing dependency —
  `keypoint_eval.cluster_votes` already uses `cKDTree`). For GPU, a one-shot
  `torch.cdist` is fine for the dec2 map (few-hundred tokens) but **not** for the
  full-res nu map; there, KDTree on CPU or a voxel-bucket radius query.
- The subtraction touches all M points but only matters within ~4σ; can restrict
  to `query_ball_point(mu, 4*sigma)` for speed. Functionally identical.

---

## 6. Integration — two options

**Option A (recommended first): standalone post-processor.** A CLI
`run_keypoint_reco.py <dir-of-score-map-h5>` that reads the `--save-score-maps`
H5s and writes reco keypoints. Decouples the reco from the (slow) cascade; lets
us iterate on params without re-running inference. Pairs with the existing
`eval_keypoint2_inference.py` style of dir-based evaluation.

**Option B: inline in the inference tool.** Add a `--reco-keypoints` flag to
`run_larformer_keypoint2_cascade_inference.py` that, inside `_decode_event`,
calls `KeypointRecoTorch.reconstruct_event` on `ev["level_kp"]` (apply `to_cm`
first) and writes `reco/nu_vertex_cm`, `reco/keypoints_cm`, `reco/keypoint_score`
alongside the existing decode. Do this after Option A is validated, to avoid
coupling.

### 6.1 Output H5 schema (Option A)

```
reco/nu_vertex_cm            (3,)         best nu-vertex peak (highest score)
reco/nu_vertex_score         scalar
reco/nu_candidates_cm        (Knu, 3)     all nu peaks (pile-up studies)
reco/nu_candidates_score     (Knu,)
reco/keypoints_cm            (Kobj, 3)    object-head peaks
reco/keypoints_score         (Kobj,)
reco/keypoints_rmse          (Kobj, 3)    fit QA
reco/keypoints_rsqr          (Kobj, 3)
attrs: radius_cm, sigma_cm, score_thresh  (provenance)
```

---

## 7. Evaluation & validation

1. **Synthetic unit test (parity / correctness).** Build a score map = sum of K
   known 3 cm Gaussians at random cm positions on a synthetic point support;
   assert the reco recovers all K centers to < 0.5 cm and stops at the right
   count. Add overlapping-peak cases (centers 4–8 cm apart) to probe the
   isolation radius / subtraction. **Critically, add truncated cases:** mask one
   hemisphere of the support (half-Gaussian, simulating a track start) and mask
   the peak entirely (only the descending tail survives, peak off-support), and
   assert Method 1/2 still recover the center to < 1 cm while the score-weighted
   centroid visibly biases toward the visible side — this is the test that
   justifies the fitter choice in §4.1.
2. **Caruana parity.** Unit-test `gaussian_fit.caruana_full` against a hand
   computation / the C++ formula on a single Gaussian (no fixed-σ), then confirm
   `fit_gaussian_fixed_sigma` matches it when the data's true σ = 3 cm.
3. **GT comparison on sim.** With `--save-score-maps` + `--with-gt`, compare reco
   keypoints to `gt_keypoints/pos_cm` (filtered by `type`): nu vertex vs type 0;
   object peaks vs types {1..5} pooled. Reuse the matching/metrics machinery in
   `keypoint_eval.py` (`match_points`, `accumulate_metrics`) — median/mean
   distance + recall@{1,3,10} cm, exactly like `eval_keypoint2_inference.py`.
4. **Cross-check vs existing decode.** The reco's best nu peak vs the inference
   tool's `nu_vertex_cm` (score-weighted centroid) — they should agree to a few
   cm; large disagreement flags a multi-peak nu map.
5. **Threshold sweep.** `score_thresh` ∈ {0.5, 0.67, 0.8}, `radius_cm` ∈ {6, 10,
   15} → precision/recall curves to confirm the defaults.

Note on threshold: score 0.67 ≈ a point ~2.7 cm from a true keypoint
(`exp(−0.5·d²/3²)=0.67 → d≈2.7 cm`), i.e. "within roughly one σ" — a sensible
floor for trusting a peak.

---

## 8. Parameters (defaults)

| Param | Default | Meaning / origin |
|---|---|---|
| `radius_cm` | 10.0 | neighborhood isolation radius (user spec) |
| `fit_radius_cm` | 2σ (6.0) | tight FIT neighborhood — avoids neighbor-peak bias (§4.2) |
| `sigma_cm` | 3.0 | fixed Gaussian width = GT label σ (`keypoint_labels.py`) |
| `score_thresh` | 0.67 | stop when running max < this (user spec) |
| `min_neighbors` | 4 | C++ skips clusters with `< 4` points |
| `max_keypoints` | 3 (nu) / 64 (object) | safety cap per map |
| `amplitude` | "peak" | subtraction amplitude = peak score (C++ `max_score`) |
| `fit_method` | "nls" | §4.1 — "nls" (default) / "loglinear" / "centroid" |
| `clamp_extrap_sigma` | 3.0 | cap `‖μ−argmax‖` at this × σ (off-support guard) |

---

## 9. Reuse map

- `pointcept/models/LArFormer/keypoint_eval.py`: `_recover_affine` / `denorm`
  (frame, only needed for Option B), `cluster_votes` (KDTree-greedy pattern to
  mirror), `decode_dense_votes` (the vote/Hough decode to reuse if the offset
  head — §4.1 Method 3 — is exposed), `match_points` / `accumulate_metrics` /
  `format_metrics_table` (evaluation).
- `pointcept/models/LArFormer/keypoint_heads.py`: `KeypointOffsetHead` (+ the
  per-SP `kp_dense_offset_head` in `model.py`, supervised by cache `kpoffsets`) —
  the existing vote-head machinery for §4.1 Method 3.
- `larflow/larflow/Reco/KeypointReco.cxx`: `_fit_cluster_CARUANA` (the fit math),
  the subtraction term, the pass loop structure, `KPCluster` field set.
- `lartpc/data_prep/labels/keypoint_labels.py`: σ and the keypoint-type enum — the
  generative model we invert.
- `tools/run_larformer_keypoint2_cascade_inference.py`: `--save-score-maps`
  writer (the input format), `_decode_event` (the `to_cm` affine, for Option B).

New code: the `reco/` package in §5 (no ROOT / larflow link).

---

## 10. Open questions / risks

- **Amplitude choice.** "peak" (C++) is robust but slightly over-subtracts when
  peaks overlap; "fit" can blow up on noisy logs. Default "peak"; revisit if the
  synthetic overlapping-peak test shows residual ghosts.
- **Object head is type-less.** The dec2 `object` head pools types 1–5 into one
  score, so the reco produces *generic* keypoints. Typing options (future): (a)
  add per-type dense heads to the keypoint model; (b) assign each reco keypoint
  the type of the nearest per-particle query start/end from the existing decoder;
  (c) nearest-neighbor to the `object` token's argmax type if per-type scores get
  exposed. Out of scope for v1 — v1 outputs positions + scores only.
- **Spacepoint-resolution nu map size.** 4k–11k points; confirm KDTree radius
  queries keep the per-event reco well under a second. If not, voxel-bucket the
  support or down-select to `score > 0.3` before the loop (the GT threshold floor
  already zeros most points).
- **Multi-pass?** C++ supports `_num_passes` (re-skim at lower thresholds). The
  subtraction-based single pass should suffice here; add passes only if
  validation shows missed low-but-real peaks.
- **Does the network actually emit clean 3 cm Gaussians?** The fit assumes it; if
  the predicted field is broader/asymmetric, the fixed-σ subtraction will leave
  residual. The `fit_rsqr` QA output is the diagnostic — low R² flags maps where
  a variable-σ (full Caruana) fallback is warranted.
- **Truncated / off-support keypoints (§4.1) are the main accuracy risk.** Masked
  partial Gaussians and one-sided track/shower starts bias any centroid estimator
  and stress even the shape fit when the peak is off-support. v1 mitigates with
  the σ-fixed shape fit (Method 1) + extrapolation clamp + one-sidedness flag; the
  durable fix is the vote/offset (Hough) head (§4.1 Method 3), which needs the
  keypoint model to emit per-token offsets and `--save-score-maps` to write them.
  Decision needed: ship v1 score-field-only and add offsets later, or wire the
  offset head first. Recommended: v1 score-field first (validate the peeling +
  metrics), then add the vote head as v2.
```
