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
  cross-track-convergence **vertex snap** (`--no-snap` to disable) that recovers
  moderately-off predicted vertices. **Viz is a synced 2-panel plotly**: left =
  reco interaction (tracks by tree depth, gold primary vertex, dashed attachment
  bridges); right = the **true ionization** (GT particle spacepoints, colored to
  match the left panel) for direct reco-vs-truth comparison. Dragging either
  panel rotates both (shared camera). `--vertex-source {reco,pred,gt}`,
  `--d-vertex`, `--d-perp`, `--snap-radius`, `--max-gap`.

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
