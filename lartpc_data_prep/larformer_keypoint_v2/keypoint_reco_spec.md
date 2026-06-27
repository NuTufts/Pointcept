# Keypoint Reconstruction from LArFormer Cascade Score Maps — PyTorch Spec

Status: **PLAN (2026-06-27)** — not yet implemented.

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
Gaussians (`lartpc_data_prep/keypoint_labels.py`):

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

The user fixes σ = 3 cm and fits only the **mean** (and amplitude). With σ fixed,
the `x²` coefficient of the log-parabola is **known** (`−1/(2σ²)`), so per axis we
fit only two free parameters — a much better-conditioned linear fit than the full
3×3 Caruana system, and it cannot return an imaginary σ. Per axis `d ∈ {x,y,z}`:

```
let  yj = residual[nbr]  (>0),   xj = coords[nbr, d]
move the known curvature to the LHS:
    zj = ln(yj) + xj^2 / (2 sigma^2)            # = ln A + (mu/sigma^2) xj
weighted linear fit  zj ≈ a + b·xj   (weights wj = yj^2, score-weighted as in C++):
    b = mu_d / sigma^2   →   mu_d = b * sigma^2
A (amplitude) from the intercept a, averaged/blended across the 3 axes:
    ln A_d = a + ... ;  use A = residual[i*] (peak) as the robust default,
    or the fitted A clamped to <= 1.5 * peak.
```

Recommended: **use the fitted `mu` for position, and the observed peak score
`smax` for the subtraction amplitude `A`** (this is exactly what C++ does —
`kpc.max_score` is the amplitude in the subtraction term). This avoids amplitude
blow-ups from the log fit while still getting sub-voxel `mu`.

Keep the C++ goodness-of-fit outputs (`rmse`, `rsqr` per the residual vs the
fitted Gaussian) for QA / downstream filtering; compute them as in
`_fit_cluster_CARUANA`.

Fallbacks (guard against degenerate neighborhoods):
- `< min_neighbors` points, or the linear system is singular → use the
  **score²-weighted centroid** of the neighborhood as `mu` (the C++
  `center_avg_pt_v` path), no sub-voxel fit.
- Any `yj <= 0` excluded from the log fit (already filtered: it's the residual).

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
lartpc_data_prep/larformer_keypoint_v2/reco/
├── __init__.py
├── keypoint_reco.py        # KeypointRecoTorch (the algorithm + params)
├── gaussian_fit.py         # fit_gaussian_fixed_sigma (reduced Caruana) + caruana_full (parity)
├── io.py                   # load_score_maps(h5) -> {name: (coords_cm, score, level, kp_types)}
└── run_keypoint_reco.py    # CLI: dir of *.h5 -> reco keypoints (+ optional eval/H5 out)
```

### 5.1 `KeypointRecoTorch` (mirrors the C++ class surface)

```python
@dataclass
class KeypointRecoParams:
    radius_cm: float = 10.0          # neighborhood isolation radius
    sigma_cm: float = 3.0            # fixed Gaussian width (= GT label sigma)
    score_thresh: float = 0.67       # stop when running max < this
    max_keypoints: int = 64          # safety cap per map
    min_neighbors: int = 4           # below this: suppress point, no fit
    amplitude: str = "peak"          # "peak" (C++ behavior) | "fit"
    device: str = "cpu"              # "cuda" for big nu-slice maps

class Keypoint:                      # ~ KPCluster, trimmed to what we have
    pos_cm: np.ndarray   # (3,) fitted mean
    peak_score: float    # max residual score at detection (~ KPCluster.max_score)
    fit_rmse: np.ndarray # (3,)  ~ center_pt_rmse_v
    fit_rsqr: np.ndarray # (3,)  ~ center_pt_rsqr_v
    n_support: int       # neighborhood size used for the fit

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
   isolation radius / subtraction.
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
| `sigma_cm` | 3.0 | fixed Gaussian width = GT label σ (`keypoint_labels.py`) |
| `score_thresh` | 0.67 | stop when running max < this (user spec) |
| `min_neighbors` | 4 | C++ skips clusters with `< 4` points |
| `max_keypoints` | 3 (nu) / 64 (object) | safety cap per map |
| `amplitude` | "peak" | subtraction amplitude = peak score (C++ `max_score`) |

---

## 9. Reuse map

- `pointcept/models/LArFormer/keypoint_eval.py`: `_recover_affine` / `denorm`
  (frame, only needed for Option B), `cluster_votes` (KDTree-greedy pattern to
  mirror), `match_points` / `accumulate_metrics` / `format_metrics_table`
  (evaluation).
- `larflow/larflow/Reco/KeypointReco.cxx`: `_fit_cluster_CARUANA` (the fit math),
  the subtraction term, the pass loop structure, `KPCluster` field set.
- `lartpc_data_prep/keypoint_labels.py`: σ and the keypoint-type enum — the
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
```
