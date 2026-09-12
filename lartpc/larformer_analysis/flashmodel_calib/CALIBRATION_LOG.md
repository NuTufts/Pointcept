# Cross-sample flash light-yield (gamma) calibration — working log & handoff

Owner: hand-off doc for a dedicated flashmatch-calibration session.
Last updated: 2026-09-11.

## 1. Why this matters right now

Two live problems depend on the answer:

1. **EXT data/prediction mismatch.** The pi0 and single-photon comparisons show a
   data excess over prediction in the EXT (beam-off) component, concentrated at
   low flash PE
   (`physics/single_photon/plots_cew6bdt_novtx/diagnostics_{vertex,novertex}/preChi2/flashPE.png`).
   If the EXT sample was reconstructed with the wrong light-yield scale, its
   `pred_pe` — and therefore its flash-chi2 — is systematically biased, which
   changes how many EXT events survive the chi2 cut. That makes this a
   *reconstruction* problem, not only a normalization one.
2. **The Run-1 alignment campaign is gated on it.** `gamma_scale` multiplies
   `gamma_beam` at
   `tools/larformer/run_larformer_keypoint2_cascade_inference.py:569`, i.e.
   BEFORE the flash-chi2 and the `chi2_rank` that assigns the `nu` vs
   `flashmatch` stream. It is baked into the GPU pass, so it must be right
   before production inference; getting it wrong means redoing all of it.
   (Staging + merged_sp conversion are unaffected and are already running —
   the converter has no gamma/pred_pe/chi2 dependence at all.)

## 2. What is deployed today

`lartpc/flashmatch/dead_channels.py` — the single source of truth, keyed on
**run number only**:

```python
GAMMA_SCALE_BY_PERIOD = {1: 0.80,  # bnb5e19 run1: measured (muon 0.79 / shower 0.85)
                         2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0}   # 3 = gamma_beam reference
DEAD_OPDETS_BY_PERIOD = {1: (), 2: (), 3: (15,), 4: (15,), 5: (15,)}
_PERIOD_BOUNDS = ((1,0), (2,7771), (3,13697), (4,18961), (5,22270))
```

`gamma_beam = 5.25` (inference default), tuned to **run-3 MC**. Resolution is
per-event from the run number via `resolve_gamma_scale` / `resolve_dead_opdets`
(specs `auto` by default). Two TODOs are recorded in that module's docstring and
are exactly what this work is about:
(a) replace the dead list with the official per-period bad-optical-channel list;
(b) **measure the run3-DATA gamma from the full EXT sample; if run-3 data also
offsets from run-3 MC, the effect is data/MC and not purely per-run, and the
table must be split accordingly.**

**Key structural point: the table cannot distinguish samples within a run
period.** bnb5e19, a Run-1 EXT sample and a Run-1 overlay are all
`run_period()==1`. So do NOT edit `GAMMA_SCALE_BY_PERIOD` in place to encode a
per-sample finding — it would retroactively change every validated Run-1
product. Pass `INF_EXTRA_ARGS="--gamma-run-scale <value>"` per sample instead
(the hook exists at `larformer_reco/slurm/submit_extbnb_chain.sh:106`). If a
data/MC split is genuinely required, add it **additively** (e.g. a separate
`GAMMA_SCALE_BY_PERIOD_MC` plus an `"automc"` spec) so the data path stays
bit-identical.

## 3. Two estimators — know which one you are reading

| | `fit_gamma_run.py` ("clean muon") | `measure_bulk_gamma.py` ("bulk") |
|---|---|---|
| population | in-time stopping/entering MIP muons, proton-vetoed, with a pred/obs centroid spatial match | every event with a nu-candidate slice and an in-time flash |
| statistic | `gamma = gamma_beam * median(sum_live obs/sum_live pred)` | `r = median(sum_live obs / sum_live pred)` |
| N per sample | 10^2 | 10^4–10^5 |
| intended use | absolute-ish | **differential only** (see its docstring) |
| selection knobs | `--min-muon-len 50 --x-entry-min 125 --proton-veto-e 50 --min-pred-pe 50 --match-dz 100 --match-dy 60` | `--min-pred-pe` |

Both read `slices/pred_pe` and `flash/observed_pe` out of the **already-written**
cascade tables, so neither needs re-inference. Both resolve dead PMTs per run
(opdet 15 is dead in run 3 but LIVE in run 1 — hardcoding `(15,)` silently drops
a live tube from both sums).

### The arithmetic trap (applies to every number below)

`pred_pe` stored in a cascade file was computed at that production's
`gamma_eff = gamma_beam * gamma_scale`, which is recorded in the file:

```python
h5py.File(f)['flash'].attrs  # -> gamma_beam, gamma_scale, gamma_eff, dead_opdets
```

`fit_gamma_run.py:204` does `gamma_fit = args.gamma_beam * median(ratio)` —
it multiplies blindly by whatever you pass. So:

* pass `--gamma-beam <the production's gamma_eff>`, not the default 5.25;
* the comparable quantity across samples is
  **`absolute_scale = production_gamma_scale x r`**.

Productions to date: bnb5e19 cew6 ran at **scale 0.80** (`gamma_eff` 4.20); the
run-3 MC and run-3 EXT cew6 productions ran at **scale 1.0** (`gamma_eff` 5.25);
the Run-1 overlay gamma pilot was deliberately forced to **scale 1.0**.
`run_bulk_gamma_refs.sh` prints raw `r` ratios that are **not** corrected for
this, so its printed "run1/run3" number is misleading as-is.

## 4. Measurements (all on the cew6 chain)

`production` = gamma_scale the sample was reconstructed at.
`absolute` = production x median ratio = the scale that would make pred match obs.

| sample | kind | production | bulk r (N) | bulk absolute | clean-muon absolute (N) |
|---|---|---|---|---|---|
| run-3 MC overlay 67k | overlay | 1.00 | 0.8653 ± 0.0023 (34,568) | **0.865** | 0.924 ± 0.014 (768) |
| run-3 EXT 200k | pure data | 1.00 | 0.7047 ± 0.0051 (66,293) | **0.705** | 0.402 ± 0.058 (334) |
| run-1 bnb5e19 beam | pure data | 0.80 | 0.7959 ± 0.0037 (67,387) | **0.637** | 0.605 ± 0.021 (660) |
| run-1 overlay (pilot, TRAINPOOL) | overlay | 1.00 | *not yet measured* | — | 0.910 ± 0.038 (135) |

Clean-muon errors are `1.253*sigma/sqrt(N)` from the quoted 16–84% band; bulk
errors are the bootstrap value stored in the npz.

### Where the estimators agree, and where they blow up

* run-1 bnb5e19: 0.637 vs 0.605 — agree to ~5%.
* run-3 MC: 0.865 vs 0.924 — agree to ~6%.
* **run-3 EXT: 0.705 vs 0.402 — disagree by 75%.**

The run-3 EXT clean-muon point is the outlier and is the one to distrust: its
spatial match **dropped 77%** of candidates (1113 of 1447) versus 26–29% for the
overlays, and its per-event spread is enormous (16–84%: 0.12–1.81). In beam-off
data the in-time flash frequently belongs to a *different* cosmic than the
reconstructed slice, so the muon↔flash association is ambiguous and the
surviving sample is plausibly biased. **Prefer the bulk number for pure-cosmic
data.** Drop rates per sample are worth recording every time you run the
clean-muon fit; they are printed as `spatial in-time match: kept X, dropped Y`.

### What the numbers do and do not say

Taking the bulk column at face value:

* data is dimmer than MC at the same run: run-3 data/MC = 0.705/0.865 = **0.815**
  (consistent in direction with the earlier CC flash-chi2 study, which found
  obs/pred 0.809 MC vs 0.736 data — `SLICER_RETRAIN_PLAN.md` 2026-08-31);
* run-1 data vs run-3 data = 0.637/0.705 = **0.90** — i.e. run-1 data comes out
  *dimmer*, which is **opposite** to the impurity/light-yield-vs-time
  expectation and opposite to the clean-muon result (1.51). **This contradiction
  is unresolved and is the single most important open question.**
* the deployed run-1 value 0.80 is not reproduced by either estimator on the
  cew6 chain (bulk 0.637, clean-muon 0.605). It was measured on an older chain;
  the slicer/segmenter changed what charge lands in the nu slice, so `pred`
  changed.

Do **not** promote any absolute number from this table into `dead_channels.py`
yet. `measure_bulk_gamma.py`'s docstring is explicit that the bulk population is
not where `gamma_beam=5.25` was tuned (run-3 MC gives r≈0.81–0.87, not 1.0), so
its absolute values carry an unknown common offset; only ratios between arms
produced at the *same* scale are trustworthy, and two of our three arms were not.

## 5. Open questions for the dedicated session

1. **Resolve the run1-vs-run3 data direction.** Bulk says run-1 is dimmer, the
   clean-muon estimator says brighter, and detector physics says run-3 should be
   dimmer. Candidate causes: the 0.80-vs-1.0 production-scale correction (verify
   it is being applied), a beam-on vs beam-off population difference (bnb5e19
   has real neutrino light; EXT does not), different processing versions
   (`mcc9_v28` vs `mcc9_v29e`), or dead-PMT handling (opdet 15 live in run 1 —
   confirm both sums use the per-run list).
2. **Re-measure both arms at a common scale.** The cleanest experiment: re-run
   inference on a modest subset of bnb5e19 with `--gamma-run-scale 1.0` so it is
   directly comparable to the run-3 arms with no correction arithmetic at all.
   That removes the trap in §3 entirely.
3. **Is the offset per-run, per-data/MC, or per-sample?** The Run-1 overlay
   pilot (0.910 clean-muon) sits close to run-3 MC (0.924), hinting the split is
   simulated-vs-real light rather than run period — but that arm has no bulk
   measurement yet and an overlay mixes data cosmic light with simulated
   neutrino light, so it is not a clean probe of either.
4. **Does the bulk median have a selection bias?** The 16–84% bands are wide
   (run-3 EXT 0.15–3.31), so the median may be sensitive to `--min-pred-pe` and
   to slice quality. Scan the cut and check stability; consider a charge-weighted
   or fitted estimator instead of a median.
5. **Second, independent handle.** `test_gamma_scale_chi2.py` (pi0 shower slices)
   was the cross-check that produced the original "muon 0.79 / shower 0.85"
   pair. Re-run it on the cew6 productions.
6. **Only then**: decide the table structure and whether a full re-inference is
   justified. Re-inference changes flash-chi2 for every sample and would
   invalidate the current working points and tables.

## 6. Reproducing / extending

Tools (all in this directory):
* `fit_gamma_run.py` — clean-muon estimator. **Pass `--gamma-beam <gamma_eff>`.**
* `measure_bulk_gamma.py` — bulk estimator (differential use).
* `run_gamma_2x2.sh` — runs the clean-muon fit on the three existing cew6
  productions with the correct per-production `gamma_eff` already wired in.
* `run_bulk_gamma_refs.sh` — runs the bulk estimator on the same three.
  **Caveat: its printed ratios are not production-scale corrected.**
* `test_gamma_scale_chi2.py`, `rank1_gamma_check.py` — independent handles.
* saturation tooling (`saturated_pmt_study.py`, `make_saturation_pmt_table.py`,
  `ophit_saturation_probe.py`, `saturation_vs_badchannel.py`) — relevant because
  saturated tubes report 0 PE and are masked; defaults were tuned on run-3b.

Stored results:
* clean-muon: `gamma_run1_data_bnb5e19.npz`, `gamma_run3_data_extbnb.npz`,
  `gamma_run3_mc_overlay.npz`, `gamma_mc_run1ovl.npz` (+ older
  `gamma_bnb5e19_run1.npz`, `gamma_extbnb_run3.npz`, `gamma_mc_run3.npz`)
* bulk: `out/bulk_run3_mc_cew6.npz`, `out/bulk_run3_ext_cew6.npz`,
  `out/bulk_run1_bnb5e19_cew6.npz` (fields: `ratio`, `median`, `boot_err`,
  `runs`, `reco_cc`, plus `vtx/nofile/noslice/lowpred` counters)
* plots: `plots/gamma_fit_<tag>.png`
* RSE caches `rse_*_keypoint2_streams.npz` (large; regenerable)

Input productions (cew6 chain), under
`/cluster/tufts/wongjiradlab/larbys/data/ub_on_tufts/`:
`larformer_mcoverlay67k_s1ep2p8cew6`, `larformer_extbnb200k_s1ep2p8cew6`,
`larformer_bnb5e19_s1ep2p8cew6` — each with `dlgen2_larformer_ntuple_*.root`
and `keypoint2_streams/`. Run-1 overlay pilot:
`/cluster/tufts/wongjiradlab/larbys/data/larformer/gammapilot_run1ovl_g100/`
(12,812 events; TRAINPOOL, so **calibration use only, never physics**).

Always check what a production was made at before using it:

```python
import h5py, glob
f = glob.glob('<cascade_dir>/**/*_0.h5', recursive=True)[0]
print(dict(h5py.File(f)['flash'].attrs))   # gamma_beam, gamma_scale, gamma_eff, dead_opdets
```

## 7. Related context

* `README.md` in this directory — flash model, `f_sys`, saturation masking.
* `lartpc/larformer_analysis/physics/pilot_matrix/SLICER_RETRAIN_PLAN.md` —
  rolling campaign log; see 2026-08-30/31 (CC flash-chi2 shape: data peak
  shifted high and broadened; global pred-scale + resolution) and 2026-09-11
  (the measurements above).
* `lartpc/larformer_analysis/model_and_output_file_versions.md` — which
  checkpoints/samples define the current analysis version (v2_s1ep2p8cew6).
* Run-1 campaign plan (staging/conversion already running, inference gated on
  this work): `~/.claude/plans/zany-whistling-pine.md`.

---

## 2026-09-11 (later) — New procedure (gammacal), first cross-sample measurement

Everything above this line was measured with the NU-UNION prediction stored in
the cascade files. That approach is now understood to be invalid for the
question being asked, and a new procedure replaces it (`PROTOCOL.md`, code in
`gammacal/` + `scripts/`). Section numbering restarts here.

### 1. Why the old numbers contradicted each other

* The cew6 productions' `slices/` tables contain ONLY the nu-union row: no
  cosmic rows at all (300 files/sample checked; the deployed slicer assigns no
  cosmic-class queries, which is also why the flashmatch stream is empty,
  ~5/100k). Every stored-table estimator therefore scored the slicer's nu
  union, which in EXT is a bag of several cosmics (median ~2,200 points), mostly
  out of time. Their charge inflates the prediction while producing no in-time
  light, so obs/pred is dominated by mis-association: EXT's log10(obs/pred) is
  flat over two decades with no core, and the "bulk median" tracked sample
  composition (MC clean, beam data mixed, EXT junk). The 77% spatial-match drop
  and the 0.40 EXT clean-muon value were this effect, not light yield.
* An overlay's in-time flash is the SIMULATED nu light (unbiased-trigger
  beam-off cosmics rarely coincide), so in-time muons in an overlay measure the
  MC light scale. The earlier "0.91 = mixture of 0.80 data and 1.0 MC" reading
  of the run-1 pilot was wrong.
* In-time flash windows differ per sample (bnb5e19 run-1 ~[2.8,5.0] us, EXT
  run-3 ~[3.2,5.4], MC run-3 ~[3.6,5.2]); the cascade picks the brightest
  producer-0 flash with NO time cut (~10% of its choices are out of window).

### 2. New procedure in one paragraph

Calibration event = isolated one-boundary MIP muon that is the in-time flash
source, defined identically in EXT, beam data and overlays (candidates from
nu_reco tracks AND vertex-less kp2 instances, no vertex requirement). The light
is predicted from the muon's OWN spacepoints at the frozen reference
gamma_beam = 5.25 (CPU PhotonLib, slice-wide dedup comb charge; the full-union
rebuild reproduces the stored `slices/pred_pe` exactly on every event, so the
pipeline is validated), so the scale is `s = median(obs/pred_ref)` with no
production-gamma arithmetic. Flash-source test = scale-free cosine similarity of
the observed and predicted PMT patterns (>= 0.9) + light-centroid match. Live
PMTs = run-period dead list + saturation holes. Gates: trimmed core median
within 5% of the median and holding >= 50% of the events.

### 3. Results on the cew6 chain (multiplier `s` on gamma_beam = 5.25)

Primary = protocol defaults (`<tag>__muon.json`). Variants: `cos98` = cosine
>= 0.98 (purest source match), `loose_cos95` = no track/shower isolation, no
charge-fraction cut, cosine >= 0.95 (max statistics; the only selection that
gives EXT a usable sample). Full table: `scripts/summarize_results.py`.

| cell | sample | primary s (N) | gates | cos98 (N) | loose_cos95 (N) | truth-nu union (N) |
|---|---|---|---|---|---|---|
| (data, 1) | bnb5e19 | **0.670 +- 0.013** (292) | OK | 0.612 +- 0.053 (25) | 0.734 +- 0.010 (788) | — |
| (data, 3) | EXT 200k | 0.79 +- 0.85 (7) — unusable | FAIL | — (0) | **0.843 +- 0.058** (93) | — |
| (mc, 3) | numu overlay 67k | **0.989 +- 0.013** (486) | OK | 0.914 +- 0.016 (220) | 1.056 +- 0.011 (1023) | 0.858 +- 0.003 (23,688) |
| (mc, 1) | run-1 overlay pilot (TRAINPOOL) | **0.947 +- 0.028** (76) | FAIL (core med 0.898, 5.2%) | 0.853 +- 0.031 (31) | 1.021 +- 0.027 (178) | 0.863 +- 0.007 (4,320) |
| (mc, 3) shower | nue overlay 79k | no muons (as expected) | — | — | — | 0.772 +- 0.001 (44,540) |

Systematics visible in the variants (same direction in every sample):
dropping isolation/charge-fraction raises s by ~7-10% (extra in-time light
from other particles in the slice); tightening the shape match to >= 0.98
lowers it by ~8% (the best-matched muons sit lower). The truth-nu union arm is
biased low by cosmic charge mixed into the nu union (`nu_qfrac` is
completeness, not purity), which is why the muon-only prediction is primary.
The shower-dominated nue sample reads ~10% lower than numu on the same arm: a
particle-type dependence to keep in mind for shower-heavy selections.

Robustness (primary, bnb5e19 and numu MC): flat vs run number within the
period (+-0.03), vs brightness, vs flash position in the window, vs muon
length; the boundary-end-x scan shows muons ending near the anode (x < 125)
read ~10% lower, so the cathode-side cut stays.

### 4. Decomposition: it is the light, not the charge

Same muons (loose_cos95 selection, N = 788 / 93 / 1023 / 178):

| sample | MIP comb charge per cm | observed PE per cm | s |
|---|---|---|---|
| bnb5e19 run-1 data | 187 | 3.59 | 0.734 |
| EXT run-3 data | 187 | 3.87 | 0.843 |
| numu overlay run-3 MC | 195 | 4.89 | 1.056 |
| run-1 overlay MC | 199 | 4.43 | 1.021 |

The charge side is the same within 5% across all four; the observed light per
cm is ~25% lower in data than in MC, and ~7% lower in run-1 data than run-3
data. So: (a) the dominant split is DATA vs MC, not run period; (b) run-1 data
is NOT brighter than run-3 data in reconstructed PE per cm — the expected
scintillation-yield direction does not show up in the PE the flash model is
compared to (whatever compensates it sits in the optical reconstruction /
PE calibration); (c) `gamma_beam = 5.25` is right for run-3 MC muons
(s = 0.99): the earlier 0.865 "reference offset" was union impurity.

### 5. What the deployed table gets wrong today

`GAMMA_SCALE_BY_PERIOD = {1: 0.80, else 1.0}` applies 0.80 to run-1 overlays
(measured ~0.95) and 1.0 to run-3 data (measured ~0.8), and 0.80 to run-1
data (measured 0.67). Nothing is promoted yet (PROTOCOL section 7: closure
first, and re-inference invalidates the current working points), but the
additive resolver `lartpc/flashmatch/flash_calib.py` (specs `auto:data`,
`auto:mc`, `table`) and the cascade/stage-3/chain/exporter plumbing are in
place so a per-(kind, period) table can be deployed without touching the
legacy `auto` path (verified bit-identical on 18/20 regression events; the
other 2 differ upstream in the slicer partition).

### 6. Open items

1. **EXT needs a calibration inference mode.** With the production slicing the
   in-time cosmic muon is almost never the slice that carries the flash; only
   93 loose muons survive in 200k events. Either tabulate every slice
   (`--all-slices` is too expensive; a flash-only per-slice table would do) or
   run a dedicated pass that predicts per segmenter instance over the full
   event. Same limitation will hit the Run-1 EXT sample.
2. Closure re-inference per cell (>= 5k events) and the decision whether to
   re-infer bnb5e19 (0.80 -> ~0.67) and the run-3 data (1.0 -> ~0.8).
3. Run-1 overlay full sample -> (mc, 1) with real statistics (the pilot is
   TRAINPOOL and marginal on the gate); Run-1 EXT once converted.
4. The shape-quality dependence (cos98 lower by ~8%) points at a residual
   PMT-pattern mismatch (per-PMT Neyman-weighted gamma is 10-15% below the
   median ratio everywhere): a flash-model SHAPE issue, separate from the scale.

---

## 2026-09-12 — Dedicated flash-calibration inference mode (first EXT test)

`run_larformer_keypoint2_cascade_inference.py --flash-calib-mode`: after the
slicer forward, the deghosted cloud is clustered into connected components
(4 cm linkage; model-independent, since the deployed slicer emits no cosmic
slices), light is predicted for every cluster, the in-time flash SOURCE is the
cluster with the best pattern cosine (>= 0.9), Stage-3 runs on that cluster
only (new explicit-mask forced slice) and the event is written as
stream='calib' with the full per-cluster table (`slices/shape_cos`). Driver:
`slurm/run_calib_inference.sh`; list: `scripts/make_calib_list.py`; records
work without nu_reco (`nu_reco_dir=None` samples).

300 EXT events (gidx 30000-30299): 167 flash-matched clusters (50 events had
no in-window flash, 83 no cluster with cosine >= 0.9), against 7 usable muons
from the production nu stream on the same events.

Object definition matters (same events, track-like clusters, >= 1 boundary):

| instance coverage of the cluster | N | r (whole cluster) | r (largest segmenter instance) |
|---|---|---|---|
| < 0.5 | 18 | 0.43 | 5.4 |
| 0.5-0.8 | 16 | 0.55 | 0.86 |
| 0.8-0.9 | 16 | 0.52 | 0.64 |
| > 0.9 | 9 | 0.47 | 0.49 |

The segmenter instance holds 78% of the cluster points on the median event
(the user saw half-muon masks in the viz); its ratio is biased HIGH by the
missing charge and converges to the cluster value as coverage -> 1, while the
cluster ratio is coverage-independent. So the whole flash-matched cluster is
the calibration object (`fit_gamma --calib-object cluster`), and the
production-stream muon arm above (instances) carries an incompleteness bias
that must be quantified on the same events (jobs below).

Cluster object, EXT run-3, protocol cuts (track-like rms_perp < 4 cm, lin >
0.95, every boundary end at x > 125 cm):

| selection | N | s |
|---|---|---|
| >= 1 boundary | 37 | 0.529 +- 0.043 (gates OK) |
| 1 boundary | 20 | 0.507 +- 0.062 |
| 2 boundaries (through-going) | 17 | 0.529 +- 0.040 |

Open: the cluster ratio falls with track length (80-120 cm 0.70, 120-180
0.55, 180-300 0.39) and with containment (contained 0.75, 1 boundary 0.52,
through-going 0.44, N = 10/51/71 without the x cut), with no brightness or
flash-time-in-window trend. Running the calibration mode on the first 3000
events of EXT, bnb5e19 and the numu overlay (2 GPU shards each) to see whether
the length trend and the cluster-vs-instance offset are EXT-specific.
Example pages: `results/s1ep2p8cew6/viz/extbnb200k_calib_test300_cluster/`.

---

## 2026-09-12 — Calibration mode on 3,000 events of EXT, bnb5e19 and numu overlay: run-1 vs run-3 direction resolved

`slurm/run_calib_inference.sh`, first 3,000 events of each merged_sp list, 2
GPU shards each (~2 h/shard). Flash-matched clusters written: EXT 1,771,
bnb5e19 1,644, numu overlay 1,571. Samples `*_calib3k` in `gammacal/samples.py`;
records `results/s1ep2p8cew6/records/<tag>.shard0000000.npz`; results
`<tag>__muon_<variant>.json`.

Object = the WHOLE flash-matched cluster (`--calib-object cluster`), track-like
(rms_perp < 4 cm, linearity > 0.95), every boundary end at x > 125 cm, cosine
>= 0.9, centroid match; `--no-require-mu-class`, isolation cuts off (the
cluster is one object by construction). The short-cluster fragment bias (a
connected component broken at a gap looks like a stopping muon with too little
charge -> ratio high; visible as the falling length trend in every sample) is
removed by requiring **through-going (2 boundaries) and length > 120 cm**:
complete by construction. On that selection the ratio is flat vs length,
boundary x, run number and flash position in the window, and the gates pass.

| cell | sample | through-going, L > 120 cm | any boundary, L > 120 cm | instance object (any boundary, L > 50) |
|---|---|---|---|---|
| (data, 1) | bnb5e19 calib3k | **0.533 +- 0.010 (107)** core 0.87 OK | 0.545 +- 0.006 (255) OK | 0.727 +- 0.027 (84) FAIL |
| (data, 3) | EXT calib3k | **0.414 +- 0.008 (110)** core 0.92 OK | 0.428 +- 0.006 (275) OK | 0.558 +- 0.033 (71) OK |
| (data, 3) | EXT 300-event test | 0.475 +- 0.045 (15) | 0.457 +- 0.031 (30) | 0.44 (3) |
| (mc, 3) | numu overlay calib3k | 0.94 +- 0.13 (13) FAIL | 0.827 +- 0.026 (87) OK | 1.025 +- 0.040 (65) OK |

**Run-1 data / run-3 data = 1.29 +- 0.03** on the identical object: run 1 IS
brighter, as the scintillation-yield history says. The earlier reversal
("run 1 dimmer", sections above) was an artefact of the instance objects and
the nu-union slicing; the instance object is biased high by 30-35% in data
(incomplete masks) and ~20% in MC.

Caveats that remain:
* MC: an overlay's in-time light is the neutrino's, so the through-going
  (cosmic) selection has no statistics (13) and the any-boundary selection is
  a mix of exiting CC muons (right object) and cosmics spatially aligned with
  a bright nu flash (ratio outliers of 2-12 at > 1000 PE). The (mc, 3) value
  0.83 therefore carries an association systematic; the production-stream
  arms (instance 0.99, cosine >= 0.98: 0.91, truth-nu union 0.86) bracket it.
  A purity condition (no other cluster predicting a comparable share of the
  observed light) is the next refinement for overlays.
* The 4 cm linkage merges delta rays and Michels (real in-time charge, fine)
  and, rarely, a crossing track (rejected by the track-likeness cut).
* MC run 1 has no calibration-mode run yet (pilot, instance object: 0.95).

Implied data/MC light ratios: run 3 = 0.41 / 0.83 ~ 0.50; run 1 ~ 0.53 / 0.85
~ 0.6. Deployed table (run 1: 0.80, else 1.0) is far from all of these; any
promotion needs the closure step (PROTOCOL section 7) and a decision on
re-inference. Pages: `results/s1ep2p8cew6/viz/{bnb5e19,extbnb200k}_calib3k_cluster_2b/`.

---

## 2026-09-12 — 2x2 complete: run-1 overlay calibration run; data vs MC and run 1 vs run 3

Run-1 overlay pilot (mcc9_v28, TRAINPOOL, calibration only), 6,000 events in
calibration mode -> 3,245 flash-matched clusters (`run1ovl_calib6k`). Its
flash window was measured on the pilot production: [3.6, 5.2] us, same as
run-3 MC.

MC purity. In an overlay the in-time light is the neutrino's, so a cluster can
win the shape match while a bright nu flash (or a dim unrelated one) is what
the PMTs saw: the ratio then sits at ~3-4 (> 3000 PE) or ~0.1 (< 300 PE) and
the core fraction drops to ~0.5. The records carry the true-nu charge fraction
of the calibration slice, and `fit_gamma --nu-qfrac-min 0.3` (MC only) removes
those associations: core fraction 0.85-0.87, flat in brightness. Data cells
cannot use it and do not need it (single cosmic per in-time flash).

Object for all four cells: whole flash-matched cluster, track-like, length
> 120 cm, >= 1 boundary end, every boundary end at x > 125 cm (cathode half);
MC additionally nu_qfrac >= 0.3. Multipliers on gamma_beam = 5.25:

| | run 1 | run 3 | run 1 / run 3 |
|---|---|---|---|
| **data** | bnb5e19 **0.545 +- 0.006** (255) [through-going: 0.533 +- 0.010 (107)] | EXT **0.428 +- 0.006** (275) [through-going: 0.414 +- 0.008 (110)] | **1.27 +- 0.02** |
| **mc** | run-1 overlay **0.858 +- 0.022** (75) [1 boundary: 0.846 +- 0.022 (65)] | numu overlay **0.833 +- 0.025** (62) [1 boundary: 0.827 +- 0.026 (54)] | 1.03 +- 0.04 |
| **data / mc** | 0.64 +- 0.02 | 0.51 +- 0.02 | |

Reading: the simulated light scale is the same in both periods (the
simulation does not follow the light-yield history), real data is brighter
in run 1 than in run 3 by 27%, and data is dimmer than simulation by 36%
(run 1) and 49% (run 3). All four cells pass the gates and are flat in run
number, flash time in window, boundary x and length. Object caveat: the MC
cells are exiting CC muons (truth-selected), the data cells are cosmic muons;
the instance-based production-stream arms (0.99 MC, 0.67 / 0.84 data) are
biased high by mask incompleteness and are superseded.

Implication for the deployed table `{run 1: 0.80, else 1.0}`: every cell is
off (MC ~0.85 both periods; run-1 data 0.55; run-3 data 0.43). Promotion
still needs closure (PROTOCOL section 7) and the re-inference decision; the
`flash_calib.py` table stays empty until then. Result JSONs:
`results/s1ep2p8cew6/{bnb5e19,extbnb200k,mcoverlay67k,run1ovl}_calib*__muon_clu_*.json`.

---

## 2026-09-12 — Deployed: calibrated table in production; Run-1 EXT tranche A launched

`lartpc/flashmatch/flash_calib.GAMMA_SCALE_TABLE` now holds the four cells
(written by `scripts/make_table.py --no-closure-check --write` from the
`*_clu_anyb_L120[_nuq].json` results):

| cell | scale | err | N | result |
|---|---|---|---|---|
| (data, 1) | 0.5449 | 0.0061 | 255 | bnb5e19_calib3k |
| (data, 3) | 0.4281 | 0.0057 | 275 | extbnb200k_calib3k |
| (mc, 1)   | 0.8576 | 0.0217 | 75  | run1ovl_calib6k |
| (mc, 3)   | 0.8325 | 0.0253 | 62  | mcoverlay67k_calib3k |

`submit_extbnb_chain.sh` defaults: `GAMMA_SPEC=table`, `SAMPLE_KIND=auto`,
`FLASH_WINDOW=auto`; EXT samples pass the window explicitly. The legacy
`GAMMA_SCALE_BY_PERIOD` (spec `auto`) is untouched and marked legacy; the
resolver identity test still passes. Periods 2/4/5 and any unmeasured cell
raise instead of defaulting.

Run-1 EXT (mcc9_v29e_dl_run1_C1_extbnb, stride-2 list, tranche A = filenos
1-1000, 15,381 merged_sp events, runs 4953-6998): in-time window measured on
the merged_sp flashes = [3.2, 5.4] us (same as run-3 EXT; NOT the beam-on
run-1 window 2.8-5.0). Chain launched with `TAG=extbnb_run1_A NINF=8 NNR=8
NEXP=4 SAMPLE_KIND=data FLASH_WINDOW=3.2,5.4` -> gamma_eff = 5.25 x 0.5449 =
2.861, default LArPID weights, jobs 3589639-3589651; ntuple
`run1_C1_extbnb/dlgen2_larformer_ntuple_extbnb_run1_A.root`.

Closure (PROTOCOL section 7) is folded into this production: run the
calibration mode on a subset of the same files and check that the cluster
arm returns 0.545 +- 0.05 (the calibration is independent of the production
gamma, so this tests the period/kind assignment and the window, not the
arithmetic); and check obs / stored pred_pe ~ 1 on the production's own
flash-matched nu slices.

This is the FIRST production made with a non-legacy gamma. It is not
comparable in flash-chi2 to the cew6 run-3 EXT (gamma_eff 5.25) or bnb5e19
(4.20) productions until those are re-inferred with the table.

**Correction (same day).** The first launch (jobs 3589639-3589651) ran with
the WRONG chain: `submit_inference_shard.sh` takes the cascade config and the
slicer/segmenter checkpoints from environment variables that the cew6
productions had exported by hand, and a launch from a clean shell fell back to
the base config + default checkpoints. Symptom: cosmic slice rows in the
tables and 311 flashmatch-stream vs 66 nu-stream files (the cew6 chain has no
cosmic rows). Cancelled; `submit_extbnb_chain.sh` now carries the frozen
v2_s1ep2p8cew6 chain version as checked, exported defaults (config, slicer,
segmenter, both shower BDTs, LLR attachment tables) and prints it. Relaunched
as jobs 3589903-3589911 (prep clears the bad outputs first). The gamma
provenance attrs of the first (wrong-chain) file were nevertheless correct:
gamma_spec=table, gamma_scale 0.5449, gamma_eff 2.861, sample_kind data,
flash_window 3.2,5.4, dead_opdets '' (run 4983).
