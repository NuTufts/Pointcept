# trajfit_dev — ElPiGraph spike for track trajectory fitting

Throwaway dev harness for trying **ElPiGraph** (Method C of
[`../trajectory_fitting_brief.md`](../trajectory_fitting_brief.md)) on real
cascade-inference output, before committing to the full `trajfit/` package
(brief §8). All scripts run **inside the pointcept container** (ElPiGraph 0.3.4 +
CuPy are installed there, not on the host):

```bash
# from Pointcept/
./run_in_local_pointcept_container.sh python \
    lartpc_data_prep/larformer_keypoint_v2/trajfit_dev/run_elpigraph.py
```

## Files
- **`trajfit_io.py`** — per-instance loader (brief §2.6). Reads a `keypoint2_out`
  event H5, yields one `InstanceRecord` per predicted particle
  (`points`, `pred_cls`, pred/GT start+end, `truth_cloud`). With
  `--merged-sp-dir` it KD-matches the slice into the parent `merged_sp`
  `triplet_data` to attach per-point **charge** (`pixval`) and pulls MC truth
  (`pid`, `true_p`, true kink vertices). Smoke test:
  `python trajfit_io.py <keypoint2_event*.h5> --merged-sp-dir <merged_sp dir>`.
- **`run_elpigraph.py`** — fits `computeElasticPrincipalCurve` on every
  track-class instance (`cls ∈ {mu,pi,p}`), traces the node graph into an
  ordered polyline, reports runtime / residual-RMS / endpoint error, and writes
  a Plotly HTML overlay per instance to `elpigraph_out/`. Key flags:
  `--num-nodes` (0 = length-adaptive), `--Lambda`, `--Mu`, `--trimming-radius`,
  `--gpu`, `--no-charge`, `--no-html`.
- **`sweep_elpigraph.py`** — grids node-density / `Lambda` / `Mu` /
  `TrimmingRadius` over all dev instances and tabulates median residual,
  endpoint error, #segments, runtime. For tuning the meta-parameters.
- **`nu_interaction.py`** + **`nu_interaction_spec.md`** — prototype **neutrino-
  interaction reco**: roots a particle tree at the nu vertex and attaches tracks
  whose endpoint is near the vertex AND whose initial ≤3 cm points back to it,
  then repeats at attached-track far ends (secondary vertices). Includes a
  cross-track-convergence **vertex snap** that recovers a badly-placed seed —
  but ONLY when the seed is *unsupported* (no track endpoint within
  `--snap-support-radius`, default 5 cm); a good fitter seed already sitting on a
  track start is left untouched (snapping it to the ~5-10 cm-scattered track-start
  convergence would only degrade it). On the dev set only 2/42 events snap; vtx-err
  median 0.7 cm. `--no-snap` disables. **Viz is a synced 2-panel plotly**: left =
  reco interaction (tracks by tree depth, gold primary vertex, dashed attachment
  bridges); right = the **true ionization** — ALL true particles (tracks AND
  showers), colored by type, with the ones the reco **missed** (showers,
  sub-threshold, unmatched) drawn as larger ✗ markers so misses stand out against
  what was reconstructed. Dragging either panel rotates both (shared camera).
  Type is coloured by true PID when `merged_sp` is present, else by predicted
  class. `--vertex-source {reco,pred,gt}`, `--d-vertex`, `--d-perp`,
  `--snap-radius`, `--max-gap`. Per-event summary reports `N GT particles missed`.

## Shower attachment into the interaction (Phase 2) — in `nu_interaction.py`
`reco_showers()` attaches predicted shower instances to the interaction's
**connection points** (nu vertex + track endpoints from the tree, ordered
closest-to-vertex first). Per (shower, connection point) the trunk is re-fit
biased toward that point (`trunk_vertex_biased`, the Phase-1 winner) and tested
with the impact-parameter + back-pointing cosine `connects()`. Two modes:
`--shower-mode greedy` (attach to the first point it points back to) vs
`exhaustive` (best over all). Attached showers draw as red trunk arrows + dashed
bridges on the left panel and flip to `reco` (no longer `MISSED`) in the GT panel.
Knobs: `--shower-d-impact`, `--shower-cos-min`, `--shower-d-gap`.

Connection points = nu vertex + track endpoints/junctions + **interior track
kinks** (`--kink-tol`, item b). The trunk anchors on fragments ≥ `min_frag_pts`
(default 10) so a tiny stray cluster near a connection point can't produce a
trunk "floating in space" (fixed). `shower_truth.py` provides the **provenance
truth**: a shower is *primary* iff its true `originpt` (from
`merged_sp/shower_fragments`, matched by nearest `startpt` to the instance's GT
conversion point) is within ~5 cm of the true nu vertex — NOT the matched
trackid's mc origin (shower electrons are created 100s of cm downstream).

**Findings (89 showers, bnb_pi0_valdata, merged_sp present):**
- **Greedy ≈ exhaustive**: both attach **57/89 (64%)**, differing on only **6/89**
  (which connection point) — cheap closest-first greedy loses ~nothing.
- **nu-vertex attachment vs provenance truth** (75/81 showers truly primary).
  Per-shower `impact`/`cosine` are in the viz legend so a confidence threshold can
  be chosen.
- **Parameter scan** (`scan_shower_attach.py`, sweeps `d_impact` × `cos_min`
  against the provenance truth — caches per-event geometry once, then applies cuts
  cheaply). Efficiency curve:

  | working point | d_impact | cos_min | precision | recall |
  |---|---|---|---|---|
  | tight (old default) | 10 | 0.90 | 1.00 | 0.67 |
  | precision-first | 10 | 0.80 | **1.00** | 0.72 |
  | **balanced (new default)** | 15 | 0.80 | 0.98 | **0.80** |
  | recall-first (best F1) | 30 | 0.50 | 0.96 | 0.85 |

  **Recall ceilings at ~0.85** even at the loosest cuts; precision degrades
  gracefully (1.00→0.96). Default is now the balanced point.

### Failure analysis (the recall ceiling)
Trunk-direction error vs the true photon direction (`startpt−originpt`) over 84
primary showers: **median ~10°, 85% under 30°**. The >60° tail (the recall
ceiling) decomposes by **vertex quality**, not by the trunk method:
- **On good-vertex events (77/84): only 2 failures (2.6%)** — the trunk method is
  essentially solved there. Both are genuine hard cases (very short / heavily
  fragmented showers where the local PCA picks the fan).
- **The other 3 failures are all on the few bad-vertex events** (reco vtx 20–48 cm
  off — the same events whose whole interaction fails). The vertex-biased trunk
  anchors at the wrong place when the vertex is wrong.
- **Tested fixes that do NOT help**: anchoring the trunk at the predicted
  shower-start keypoint instead of the vertex is *worse* (median 9.7→11.1°, BAD
  6%→15%, p90 41→130°) — the vertex gives a strong "points away from vertex"
  orientation prior the noisier keypoint lacks. Shrinking the trunk radius helps
  the angular proxy but slightly *hurts* attachment recall (kept R=5).

**Conclusion:** shower trunk direction is not the bottleneck — the recall ceiling
is set by **vertex reco on a handful of events**, which is upstream and also
breaks those whole interactions. The lever for the remaining tail is better /
flagged vertex reco, not the shower direction method.
- **Kinks (item b)**: 15 added across events; **0 used here** because these pi0
  showers are overwhelmingly primary (attach at the vertex). Kinks are
  infrastructure for secondary / mid-track-scatter showers, which this sample
  lacks.
- Several pi0 events fully reconstruct (event1, event11: 2/2 showers, 0 GT
  missed). GT panel colors by **true PID**; attached showers show as `reco`.

## Shower direction reco (Phase 1) — `shower_trunk.py`, `shower_connect.py`, `run_shower_dir.py`
Prototype of the three trunk-direction methods from
[`../shower_reco_spec.md`](../shower_reco_spec.md), evaluated on nu-vertex
connection only. All anchor the trunk at the predicted shower-start keypoint
(method 3 anchors at the vertex-nearest cluster point, its defining bias).
- `shower_trunk.py` — `ShowerTrunk` + `trunk_pca`, `trunk_elpigraph`,
  `trunk_vertex_biased` (LANTERN `NuVertexShowerReco::_make_trunk_cand` port).
- `shower_connect.py` — impact-parameter / back-pointing-cosine / gap geometry +
  greedy `connects()` decision (showers attach by pointing back across a gap).
- `run_shower_dir.py` — runs all 3 on every predicted shower instance, scores
  direction vs a truth trunk (GT start keypoint + local GT cloud), reports a
  comparison table + per-event arrow viz.

**Findings (100 showers, bnb_pi0_valdata):**
| method | ang median | ang p95 | start err | conn P/R | ms/shower |
|---|---|---|---|---|---|
| whole-cluster PCA | 17.6° | 166° | 5.65 | 0.96/0.33 | 0.1 |
| ElPiGraph | 14.5° | 164° | 5.65 | 0.75/0.32 | 5.9 |
| **vertex-biased** | **10.6°** | **121°** | **0.72** | 0.83/**0.67** | 0.7 |
- **Vertex-biased wins** on median angle, connection recall, and start error, at
  ~1 ms/shower. **ElPiGraph is less accurate AND ~8× slower** — skeletonization
  doesn't pay off for trunk direction here.
- All methods have a **large p95 tail (>120°)** — ~a few % of showers (noisy /
  very short trunks) get a flipped/garbage direction; that tail caps recall.
- The predicted shower-start keypoint sits ~5.6 cm from the true start (the PCA/
  elpigraph `start_med`); the vertex-nearest anchor is closer (0.7 cm) for showers
  that truly connect.

## Neutrino-interaction reco findings (dev events)
- **Vertex from the score-field fitter, not the centroid.** `--vertex-source reco`
  takes ranked peaks from `reco.KeypointRecoTorch` (the sibling `reco/` package's
  Gaussian peak-finder). This fixed the worst case: **event4 went from 0/6
  attached / 70 cm off (the `nu_vertex_cm` centroid) to 4/6 attached / 2.3 cm**,
  because the fitter found the true peak (0.98 score, 0.4 cm from truth) that the
  centroid had averaged away.
- **Every dev event now reconstructs a sensible primary**, vertex error **0.7–5 cm
  vs GT** (centroid was up to 70 cm): event3 4-prong (4/4), event0 & event4
  3-prong, event6 2-prong, etc.
- **Iterative seeding** tries candidates in score order and keeps the best-attaching
  one (a candidate that attaches nothing is skipped). The convergence **snap** still
  rescues cases where the top peak is on a hadronic secondary rather than the
  primary (event0: top peak is the neutron vertex; snap pulls 18.7 cm onto the
  true primary).
- **Two real limitations remain:** (1) neutral-induced secondary vertices
  (neutron/photon, no parent track) are unreachable by track-end chaining — their
  daughters stay unattached; (2) reconstructed track-start scatter is ~5-10 cm, so
  the attach gates are loose (`d_vertex=12`, `d_perp=4`).
- **`mcs_rdp.py`** — **Method B**: variable-tolerance n-D RDP with the Highland
  MCS-tied tolerance `eps(L)=kappa*sqrt(sigma_MCS(L)^2+sigma_reso^2)` (brief §5).
  Consumed by `cluster_fit_stitch.py` as the smoothing stage; `--kappa`,
  `--momentum-source {truth,fixed}`, `--fixed-p` control it.
- **`cluster_fit_stitch.py`** — the **cluster → per-fragment sliding-PCA →
  direction-scored stitch → MCS-RDP smoothing** pipeline (brief Method A + B +
  explicit gap handling),
  run head-to-head against single-call ElPiGraph. DBSCAN(eps≈1.2) splits the
  instance into gap-free fragments, each gets a windowed charge-weighted
  centerline, then fragments are stitched by linking endpoints scored on
  gap-length + end-tangent collinearity + a **`dead_region_fraction` hook**
  (stub returning 0 until a real dead-channel map is wired in). Writes colored
  per-fragment overlays (stitched=green, elpigraph=red) to `cfs_out/`. Endpoint
  extension (`--no-extend` to disable) pushes the global ends out to the extreme
  cloud points to recover range.

## Cluster-fit-stitch vs ElPiGraph (8 dev events, 22 track instances)
- **Fit quality (RMS over each track's own points): ~0.25 cm median for stitch
  vs 0.42 cm for ElPiGraph**, and far more robust on the cases ElPiGraph
  struggled with (e.g. a 262-pt muon: stitch 0.25 cm vs ElPiGraph 3.78 cm; an
  822-pt muon: 0.27 vs 1.37). Sliding-PCA hugs the centerline; ElPiGraph's
  elastic penalty rounds corners.
- **~15× faster: ~2 ms vs ~36 ms median** (ElPiGraph hits 0.5–1.2 s on the
  largest tracks; stitch stays single-digit ms).
- **Completeness is governed by `--max-gap` (the `max_gap_live` gate).** At a
  tight 3 cm only 17/22 stitch into one chain (gaps > 3 cm stay split); at the
  current default **`--max-gap 20` all 22/22 stitch** and the two large tracks
  (890-pt p, 2103-pt mu) recover from 55–62 % to 100 % coverage (range
  130→222 cm, 53→110 cm) with RMS unchanged (0.27). Aggressive bridging is the
  right default *within one predicted instance* (the segmenter already claims all
  points are one particle). The remaining guard is the `--min-dir` collinearity
  gate. **The dead-channel map (step b) is what will let long gaps be bridged
  only where physically justified while keeping live-region gaps tight** — the
  `dead_region_fraction` stub is the single insertion point.
- A few tiny leftover fragments (the stitch declined a low-collinearity or
  >3 cm link) — tunable via `max_gap_live` / `min_dir` / `short_n` in
  `stitch_fragments`.

## Method-B (MCS-RDP) — decoupled from the trajectory
**The trajectory is the DENSE sliding-PCA centerline, NOT the RDP output.** RDP is
a corner-cutter: its chords sit up to `eps` *off* the curve by construction, so
on continuous (cumulative-MCS) curvature it chords across the points. Using it as
the trajectory degraded a curving 78 cm pion from RMS 0.27 → 0.75 cm (and at
`kappa=3`, → 3.75 cm). MicroBooNE is unmagnetized, so that S-bend is real MCS
wander, not a discrete kink — exactly what RDP should *not* try to represent as a
trajectory.

So the two products are split:
- **Trajectory** (`cfs["polyline"]`, used for range / residual / local direction):
  the dense centerline. Hugs the points — median RMS **0.27 cm**, on par with
  ElPiGraph and far cheaper.
- **Kink candidates** (`cfs["kink_vertices"]`): interior vertices of MCS-RDP on
  the centerline = candidate hard-scatter break points, reported separately
  (`nkink` column, red diamonds in the overlay). NOT the trajectory.

Notes:
- **Range = dense centerline length** (with main-chain-only endpoint extension; a
  bug where extension shot the end to a far un-stitched fragment, 51.9→110 cm, is
  fixed). If `--max-gap` is set tight, a low-coverage main chain reports only the
  *covered* range and shows a large `sE`; widening `--max-gap` stitches the full
  track and range/`sE` recover.
- **`kappa` only affects the kink finder now**, not the trajectory. It's still
  unsettled: `kappa=1` over-finds (11 "kinks" on the smoothly curving 2103-pt
  muon — it mistakes MCS curvature for kinks). Setting `kappa` (or replacing the
  `sigma_MCS=L*theta0/sqrt3` sagitta estimate) needs the kink precision/recall
  metric — **step (c)**. Until then treat `nkink` as diagnostic only.

## First-look findings (8 dev events, 23 track instances)
- **Runs out of the box.** Steady-state **~34 ms median / instance** on CPU
  (max ~1.1 s for a 2103-pt muon); the first call eats ~1–7 s of numba JIT,
  which the scripts warm up and exclude. Clouds are small (11–2100 pts), so
  **CPU is the right path** — matches the brief's prediction; GPU launch
  overhead won't pay off per-track (batch instead). GPU flag is wired but needs
  a visible CUDA device.
- **Centerline quality is good:** cloud→polyline residual RMS **~0.42 cm
  median** (≈ the 3.3 mm point spacing). The curve sits cleanly down the tube.
- **Endpoint accuracy is mixed:** end error often < 2 cm but start error has a
  long tail (tens of cm) — the elastic curve overshoots/undershoots track ends
  and δ-ray/secondary contamination pulls the principal thread. This is the
  thing to tune (node density, `Mu`) and the reason endpoint metrics matter.
- **`TrimmingRadius` set tight (≈2 cm) HURTS** here — it discards real on-track
  points and raises RMS; leave it `inf` unless a specific δ-ray case needs it.
- **No GEANT step polyline exists** for truth (brief §2.6): evaluation uses the
  true point cloud + endpoints + daughter-derived kink vertices, all loaded by
  `trajfit_io.py`.

Outputs (`elpigraph_out/`) are git-ignored scratch; open the HTML files in a
browser to eyeball cloud + fitted polyline + GT endpoints + true kinks.
