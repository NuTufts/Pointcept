# Flash-model calibration & the flash-χ² saturation fix

This folder holds the studies that diagnosed and corrected the LArFormer
flash-match χ² for the run-3 MicroBooNE samples, plus the per-run light-yield
(γ) calibration. The headline result: the flash-χ² used to rank neutrino vs
cosmic slices had **two separable defects**, and fixing them raised the fraction
of events where the true-neutrino slice is the rank-1 flash match from **40.9 %
to 72.1 %** on the full run-3b overlay MC.

---

## TL;DR for a report

The flash-match compares the **observed** in-time PMT flash to the light
**predicted** from each candidate charge slice (PhotonLib × charge × γ), via a
Neyman χ². The lowest-χ² slice is taken as the neutrino. Two problems corrupted
that χ²:

1. **Unmasked dead PMT (opdet 15).** Dead in run 3 (MC + EXT); reads 0 PE while
   the prediction still puts light there, so its Neyman term `pred²/ε` (~10⁶)
   dominated the sum and pushed genuine-neutrino slices into the cosmic range.
   Masking it is the larger, **earlier** fix.

2. **Unmodeled PMT saturation (this work).** A PMT under a bright flash
   saturates: the pulse rails past the 12-bit ADC ceiling (4096), baseline
   restoration undershoots, the ophit integral goes negative, the reconstructed
   PE collapses to ≤ 0, and the opflash writes **exactly 0 PE** for that tube.
   The correct slice then eats a ~10⁶ χ² penalty on that one PMT and the ranking
   hands you a near-empty slice somewhere else. This is a **run-3 optical
   simulation artifact**, not a detector effect (see "Evidence" below).

### nu-slice rank-1 decomposition (full run-3b overlay MC, 36 849 events, identical events, only the mask differs)

| flash-χ² mask                | nu slice is rank-1 | increment |
|------------------------------|--------------------|-----------|
| no mask (old production)     | 40.9 %             | —         |
| dead-PMT {15} only           | 65.4 %             | **+24.5** |
| dead + saturation (current)  | 72.1 %             | **+6.7**  |

Cross-checked four independent ways: no-mask 40.9 % reproduces the old full
production; dead-only 65.4 % reproduces the 30 k dead-only pilot (66.0 %);
dead+sat 72.1 % reproduces the new full run and the new 30 k pilot (72.6 %). The
recompute-from-stored-arrays and the actual GPU reruns agree.

**Do not quote 40.9 → 72.1 as "the saturation fix."** That folds in the separate
dead-PMT fix. The saturation contribution is the **+6.7**.

---

## The hole-finding algorithm

Implemented in [`lartpc/flashmatch/saturation.py`](../../flashmatch/saturation.py)
(`find_saturated`). Input is the 32-element observed PE vector of the in-time
beam flash (opdet-indexed) plus the run's known-dead opdets. For each PMT *i*:

1. **Neighbourhood** — its 3 geometrically nearest other PMTs (v12 opdet
   positions); known-dead tubes are excluded from every neighbourhood.
2. **Require light present** — skip *i* unless its **brightest** neighbour
   exceeds 100 PE (dim detector regions are never flagged).
3. **Hole test** — flag *i* if `pe_obs[i] < 0.02 × max(neighbour PE)`.

Candidates are ranked by neighbour brightness and truncated to at most 4 per
event. The χ² mask is `dead_opdets ∪ saturated_opdets`.

Three deliberate design choices (each learned from a real failure case):

- **Observed PE only, never the prediction** — the mask is a property of the
  event, so every slice is scored on the same PMT subset. A prediction-dependent
  rule would let any slice excuse its own mismatches (circular). It also lets the
  identical mask apply to data and MC.
- **Relative threshold, not an absolute PE floor** — RSE (15014,234,11701) opdet
  20 reads 62 PE where 5326 was predicted; an absolute "≤ 5 PE" test misses it,
  yet it alone carries 2.7×10⁵ of χ².
- **Brightest neighbour, not the median** — broken tubes cluster; in that same
  event opdet 20's neighbours 21 and 24 are themselves holes, so a median
  reference is 0 and the test can never fire.

Wired into both inference callers:
[`run_larformer_keypoint2_cascade_inference.py`](../../../tools/larformer/run_larformer_keypoint2_cascade_inference.py)
(`--mask-saturated`, `--max-saturated-pmts`; records `saturated_opdets` /
`chi2_masked_opdets` per event) and
[`run_larformer_stage3_inference.py`](../../../tools/larformer/run_larformer_stage3_inference.py)
`flash_recovery_keep` (where χ² is a **cut**, so a poisoned value was silently
dropping the right slice's spacepoints).

---

## Evidence it is a simulation artifact (not the detector)

- **Hole rate vs flash brightness**, over all nu-slice events, MC vs the two
  real-data samples (`hole_rate_vs_brightness.py`):

  | total observed PE | MC run3b | EXT run3 | bnb5e19 run1 |
  |-------------------|----------|----------|--------------|
  | 2000–3000         |  8.3 %   |  0.9 %   |  0.6 %       |
  | 4000–6000         | 20.6 %   |  0.6 %   |  0.1 %       |
  | 8000+             | 34.7 %   |  0.0 %   |  0.9 %       |

  MC climbs steeply with brightness; both real-data samples stay flat at ~0–1 %
  at **every** brightness, and the samples overlap in brightness — so this is not
  a brightness-selection effect. n = 36 742 / 5 647 / 52 359.

- **Independent reco cross-check.** For RSE (15014,234,11701) the PeLEE
  `tpellee` branch `flash_pe_flash_matching_v` (a separate reconstruction chain)
  is opdet-indexed (corr with our obs 0.9995), agrees with our flash to 0.3 % in
  total and 1.6 % on bright tubes, and shows the **same** holes: opdets
  18/20/21/24 carry 16 027 PE of prediction and deliver 62 (ours) / 33 (PeLEE).
  So our converter and the DL optical stream are both exonerated — the light is
  genuinely absent from the simulated waveforms, at or below the ophit level.
  (The run-3 optical sim cannot be re-run; the mask is the mitigation.)

- **The ophit signature** (`ophit_saturation_probe.py`): the tag is the
  PE/amplitude ratio, not integrated area alone. Healthy tubes run 0.05–0.30
  (median 0.115); saturated tubes < 0.01, with an empty gap at 0.01–0.02.
  Amplitude > 4096 alone is NOT the failure tag — those are the *recovered*
  low-gain hits. Diagnostic only; does not fully reproduce the observed-PE hole
  finder because the worst tubes have no reconstructed pulse at all.

---

## Per-run light-yield (γ) calibration

Separate from the saturation work but part of making the flash prediction
comparable across run periods. `fit_gamma_run.py` fits γ (PhotonLib PE scale)
from in-time MIP muons per sample; the results are baked into
[`lartpc/flashmatch/dead_channels.py`](../../flashmatch/dead_channels.py)
(`GAMMA_SCALE_BY_PERIOD`). Key finding: run-1 beam data runs ~0.80× the MC light
scale (a charge/PMT-PE calibration offset, opposite the light-yield direction),
while run-3 MC/EXT use 1.0. Fitted γ arrays are the `gamma_*_run*.npz` files.

---

## Downstream physics impact

Plots: `../physics/pi0mass_peak/plots_ext_cut1e4_satfix/` (full MC =
`mcc9_bnbnu_overlay_1500_full_satfix`, 67 211 events; data = bnb5e19 full; EXT =
run-3 val). Working point: reco-CC muon tag KE > 50 MeV; flash-χ² cut reco-CC
log₁₀ < 4 (10 000), reco-NC log₁₀ < 3.25 (1778); EXT scale 4.1258. (All three are
CLI flags: `--mu-ke-min`, `--flashchi2-cut`, `--flashchi2-cut-nc`.)

- **pi0 mass peak**: unchanged by the saturation mask (it only touches the
  flash-match / fm stream, not the nu-stream pi0 selection). Signal mass median
  CC 140.0 → 140.0 MeV, NC 134.7 → 134.4 MeV; total true-π0 signal −0.1 %.
- **flash-χ² as a cut**: MC reco-NC eq2 high-χ² fraction 48.6 → 9.5 %. With the
  cut and the EXT cosmic included, the reco-NC near-peak (100–170 MeV)
  data/prediction ratio is **1.00 (eq2) / 1.02 (≥2)**. That near-peak high-χ²
  population is cosmic-dominated — once EXT is included there is no unexplained
  excess. Tightening the NC cut 3.5 → 3.25 removes a further ~20 % of the cosmic
  (EXT 41 → 33 near-peak) at essentially zero NC-signal cost; the removed
  log₁₀ ∈ [3.25, 3.5] slice is 78 % cosmic with ~1 true NC-π0 event and data
  (28) ≈ pred (27), i.e. it sits on the EXT contribution, not a data excess.

### reco-NC pi0 purity levers (near-peak 100–170 MeV, eq2)

True NC-π0 purity of the reco-NC sample, building up the selection (full MC +
bnb5e19-full + EXT-val):

  | selection                                  | NC-π0 purity |
  |--------------------------------------------|--------------|
  | muon tag 100 MeV, NC cut 3.5               | 0.420        |
  | muon tag **50** MeV, NC cut 3.5            | 0.492        |
  | muon tag 50 MeV, NC cut **3.25**           | 0.511        |
  | + **0 charged-π veto** (n_cpi_reco == 0)   | **0.543**    |

Three handles, in order of leverage:
1. **Muon tag 100 → 50 MeV** (+7 pt): most reco-NC CC feed-down was soft muons
   below the old 100 MeV tag, not muon→π misID. Justified by `stage_mu.png` — the
   mu instance finder is usable down to ~50 MeV KE. Biggest single lever.
2. **NC flash-χ² cut 3.5 → 3.25** (+2 pt): removes cosmic, no signal cost (above).
3. **Charged-π veto** (+3 pt): rejects a CC-rich sub-sample (of its signal, CC
   outweighs NC ~3:1), but only catches the ~18 % of CC feed-down that is
   muon→π misID.

Stacked, these bring reco-NC 0-charged-π NC-π0 purity to **0.543**, at the ~54 %
benchmark. The complementary stream-matched result: reco-CC isolates true CC π0
at ~0.88 (97 % right-type); reco-NC's residual impurity after these cuts is the
remaining CC feed-down (missed muons) plus the ~13 % cosmic that survives.

Statistics caveat: MC is full (67 k); the **data and EXT sides are still
bnb5e19-full / EXT-val**. The full bnb5e19 (176 k) + EXT (668 k) reruns (Phase 2)
shrink the reco-NC data/pred ratio error from ~±0.09 toward ±0.03. Tables carry
`n_cpi_reco`/`n_cpi_true` (charged-π counts) and a `reco_cc` built at the chosen
`--mu-ke-min`; `flashchi2_from_tables.py` rebuilds the flash-χ² plots from the
tables when the cascades are unavailable (e.g. mid-reprocessing).

---

## Files

### Saturation (the fix)
- `hole_rate_vs_brightness.py` — hole rate vs total observed PE, MC vs data vs
  EXT (the sim-artifact evidence).
- `ophit_saturation_probe.py` — does the ophit PE/amplitude signature tag the
  same PMTs as the observed-PE hole finder? Traces cascade → source dlmerged.
- `make_saturation_event_list.py` — emits (run,subrun,event) + opdet/**opchannel**
  + ophit evidence for events with a saturated PMT, for the official-production
  cross-check. Output: `saturation_events_mc_run3b.csv`.
- `make_saturation_pmt_table.py` — full 32-PMT table + true generator ν vertex
  for a saturated event. Output: `saturation_pmt_15014_234_11701.csv`.
- `dump_wc_pmt_info.C` / `.py` — dump a Wire-Cell `T_BDTvars` per-PMT
  `vector<double>` for a given RSE (ROOT macro, no uproot needed) and compare to
  our obs/pred. Handles the opdet-vs-opchannel indexing question explicitly.
- `saturated_pmt_study.py`, `saturation_vs_badchannel.py` — the original
  diagnosis: is the residual high-χ² CC population saturation or extra bad
  channels? (Answer: saturation.)

### γ calibration & flash-fix impact
- `fit_gamma_run.py` — per-run γ from in-time MIP muons. → `gamma_*_run*.npz`.
- `test_gamma_scale_chi2.py` — cross-check the muon γ on π0 shower slices.
- `compare_chi2_mc_data.py`, `compare_predobs_mc_data.py` — reco-NC 2γ nu-slice
  χ² and pred/obs PE, MC vs bnb5e19 data.
- `flashmatch_impact.py` — OLD (buggy-flash) vs NEW (fixed-flash) cascade impact.
- `plots/` — outputs of the above.

## Related code
- `lartpc/flashmatch/saturation.py` — the hole finder.
- `lartpc/flashmatch/flash_chi2.py` — `neyman_chi2(..., dead_opdets=)`.
- `lartpc/flashmatch/dead_channels.py` — per-run dead opdets + γ scale.
- `lartpc/larformer_analysis/physics/pi0mass_peak/` — the π0 / flash-χ²
  analyses (`flashchi2_ncpi0.py`, `datamc_ext_overlay.py`, `flash_correction.py`
  with `--saturation-mask`, `saturation_mask_test.py`).
