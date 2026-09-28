# Single-photon selection on the RUN-1 table-gamma samples

Updated 2026-09-27 (truth energy scale fixed; the 2026-09-23 legacy-scale
numbers are kept below for comparison).

## TRUTH ENERGY SCALE (2026-09-27) -- `plots_run1_novtx_dep*/` are current

The truth "visible energy" constant 0.0253 MeV/ADC was a July-2026 RECO fit
(pre-label-expansion, pre-recal); on the truth charge it overstates deposited
energy by x2.30, so the 20 MeV visibility threshold meant ~9 MeV deposited
while the reco cut demanded 20 MeV. Measured on well-contained true particles
(`lartpc/larformer_analysis/truth_calib.py`): photons 0.0110, electrons
0.0101 MeV/ADC, identical on run-1 and run-3; reco showerRecoE over the
truth-calibrated deposited energy has a median of 1.04. Visibility is now
DEPOSITED energy >= 20 MeV (`--truth-scale deposited`, the default;
`legacy` reproduces the old numbers). Analysis-level only: no reco or export
rerun.

Signal denominators (w @ 4.4e19, both MC halves): combined 805.7 (was 882.4)
| in-FV 157.6 (137.9) | 1g0X 56.2 (48.8) | entering 648.1 (744.5) = no-mu
415.7 + CC-mu 232.4. Raw signal 8,507 events.

| final step | legacy scale, chi2<316 | **deposited, chi2<316** | deposited, chi2<178 |
|---|---|---|---|
| eff combined | 0.254 | **0.325** | 0.280 |
| eff in-FV | 0.402 | **0.466** | 0.400 |
| eff 1g0X | 0.376 | **0.416** | 0.350 |
| eff entering no-mu / CC-mu | 0.245 / 0.194 | **0.310 / 0.257** | 0.269 / 0.217 |
| purity | 0.320 | **0.375** | 0.454 |
| MC bkg / EXT | 294 / 180 | **256 / 180** | 200 / 71 |
| data / pred | 1.03 | **1.03** | 1.00 |

Cutflow (deposited, chi2 < 316): S1 0.817 | S2 0.522 | S3 0.470 | S4 0.325;
purity 0.010 / 0.052 / 0.080 / 0.375; data/pred 1.03 / 1.06 / 0.99 / 1.03.
Prediction and data are unchanged (reco is untouched); only the truth labels
of MC events moved: ~5% of former in-FV >=2g events, whose second photon
deposits under 20 MeV, are now single-photon signal.

Vertex vs no-vertex (deposited, chi2 < 316): VERTEX 135.6 sig / 149.9 MC bkg /
102.8 EXT, purity 0.349 (in-FV 0.170), d/p 1.10; NO-VERTEX 126.5 / 106.0 /
77.5, purity 0.408 (entering 0.384), d/p 0.93.

Missed second photon (deposited): 847 selected in-FV >=2g events (was 1,085),
864 missed photons: photon prong with reco E <= 20 MeV 0.41 (was 0.44 --
now a genuine threshold-edge/resolution population: deposited >= 20 but
reconstructed < 20), no prong 0.32, merged 0.10, unattached BDT-fail 0.09,
track 0.05, attached BDT-fail 0.02, electron 0.01. Second-photon veto at
reco E > 10 MeV: removes 43% of in-FV >=2g and 51% of entering >=2g for a
9% / 8% / 8% cost on 1g0X / 1g+X / entering signal (the cost rose from 2-8%
because signal events now legitimately contain soft second photons).

## 2x2 split: (vertex | no-vertex) x (passes | fails the second-photon veto)
(`plots_run1_novtx_dep/paths_veto2g/`, deposited scale, chi2 < 316; the tag =
another nu-stream photon-labelled prong with reco E > 10 MeV, any BDT score,
any vertex; `--veto2g-emin`)

| final step (w) | signal | in-FV sig | MC bkg | EXT | pred | data | d/p | purity | in-FV purity |
|---|---|---|---|---|---|---|---|---|---|
| VERTEX, passes veto | 125.7 | 60.3 | 113.8 | 69.1 | 308.7 | 333 | 1.08 | 0.407 | 0.195 |
| VERTEX, fails veto | 9.9 | 5.7 | 36.0 | 33.7 | 79.5 | 95 | 1.19 | 0.124 | 0.071 |
| NO-VERTEX, passes veto | 114.7 | 6.7 | 68.9 | 57.7 | 241.3 | 223 | 0.92 | 0.475 | 0.028 |
| NO-VERTEX, fails veto | 11.9 | 0.7 | 37.1 | 19.8 | 68.8 | 66 | 0.96 | 0.172 | 0.010 |

The "fails" halves are background-rich sidebands (purity 0.12 / 0.17) that
hold 24% (vertex) and 35% (no-vertex) of the MC background and 33% / 26% of
the EXT for 7% / 9% of the signal; the in-FV >=2g background splits 31.1
(pass) vs 18.8 (fail) in the vertex sample. Using the pass halves alone:
vertex purity 0.349 -> 0.407 (in-FV 0.170 -> 0.195), no-vertex 0.408 ->
0.475. The vertex-fail sideband shows the largest data excess (d/p 1.19 on
95 events).

## Flash PE ratio cut (2026-09-27) -- `plots_run1_novtx_dep_pe3*/` are current

New last step S5: nu-slice flashObsPE / nuSlicePredPE < 3.0
(`--flashpe-ratio-max`, <=0 disables; skipped on ntuples without the
table-gamma flash branches). Above 3 the sample is almost pure EXT.

| final step, chi2 < 316 | before (S4) | **after S5** |
|---|---|---|
| eff combined | 0.325 | **0.309** |
| eff in-FV / 1g0X | 0.466 / 0.416 | **0.451 / 0.395** |
| eff entering no-mu / CC-mu | 0.310 / 0.257 | **0.293 / 0.239** |
| purity | 0.375 | **0.406** |
| MC bkg / EXT | 256 / 180 | **242 / 122** |
| data / pred | 1.03 | **1.02** |

EXT -32% for -5% signal; MC background -5%. At chi2 < 178: eff 0.271, purity
0.466, EXT 56, d/p 0.99. Vertex-path-only baseline: purity 0.343 -> 0.372,
EXT 109 -> 73, d/p 1.13.

2x2 with the PE-ratio cut (`plots_run1_novtx_dep_pe3/paths_veto2g/`):

| final step (w) | signal | in-FV sig | MC bkg | EXT | pred | data | d/p | purity | in-FV purity |
|---|---|---|---|---|---|---|---|---|---|
| VERTEX, passes veto | 119.8 | 58.7 | 107.4 | 40.9 | 268.1 | 289 | 1.08 | 0.447 | 0.219 |
| VERTEX, fails veto | 9.5 | 5.6 | 35.3 | 27.0 | 71.8 | 86 | 1.20 | 0.132 | 0.078 |
| NO-VERTEX, passes veto | 108.0 | 6.2 | 63.6 | 35.5 | 207.0 | 187 | 0.90 | 0.522 | 0.030 |
| NO-VERTEX, fails veto | 11.3 | 0.6 | 36.0 | 18.6 | 65.9 | 61 | 0.93 | 0.171 | 0.009 |

The vertex-pass cell (the in-FV physics sample) went EXT 69.1 -> 40.9 (-41%)
for signal 125.7 -> 119.8 (-5%), purity 0.407 -> 0.447, in-FV purity
0.195 -> 0.219; no-vertex-pass EXT 57.7 -> 35.5 (-38%), purity 0.475 -> 0.522.
Per-cell diagnostics: `diagnostics_{vertex,novertex}_{pass,fail}/` with
`preChi2/`, `preRatio/` (the ratio N-1) and `final/`.

---
# LEGACY-SCALE RESULTS (2026-09-23, A_GAMMA = 0.0253) -- superseded
 Supersedes `CEW6_SINGLE_PHOTON_TABLES.md` (run-3 MC/EXT
against run-1 beam data). Same cew6 chain and both cew6 BDTs; what changed is
the samples: run-1 simulation and run-1 beam-off, so data and prediction now
come from the same run period, with the calibrated flash-gamma table
(mc,1 4.5024 / data,1 2.8607) making the flash chi2 mutually comparable.

## Samples (versions doc section 3c)

| role | ntuple(s) | events | normalization |
|---|---|---|---|
| MC | `$L/run1_bnboverlay_half{,2}/dlgen2_larformer_ntuple_bnbovl_run1_half{,2}.root` | 370,580 (after dropping the 49 shared rows) | POT 4.6533e20 -> x0.09456 to 4.4e19, times xsecWeight |
| EXT | `$L/run1_C1_extbnb_half{,2}/dlgen2_larformer_ntuple_extbnb_run1_half{,2}.root` | 219,154 | x**0.601** (measured, see below) |
| data | `$L/bnb5e19_run1_table/dlgen2_larformer_ntuple_bnb5e19_run1_table.root` | 176,302 | 4.4e19 POT, unit weight |

`$L=/cluster/tufts/wongjiradlab/larbys/data/larformer`. Every `--*-ntuple`
takes a comma-separated list: the halves are concatenated, POT summed and
de-duplicated by (run,subrun,event).

**No BDT-hygiene halves.** Both cew6 BDTs were trained on RUN-3 overlay and
RUN-3 EXT, so the entire run-1 MC and EXT are held out: `--ext-row-min 0`, no
`--mc-odd-only`. Verified in the ntuples: `showerCosmicScore` and
`showerNoVtxScore` both reproduce the cew6 models (the vertex-free branch
differs from the ep8 model for 99.2% of showers).

**No `--recal-gamma-*`**: recal3 is baked into `showerRecoE`. Scripts whose
recal default is non-None (`photon_bdt_study.py`) need the identity constants
`--recal-gamma-a 0.020101 --recal-gamma-b -15.49`.

## EXT normalization — measured, not the spill count

The spill-count ratio (bnb5e19 10,375,708 / both EXT halves 11,545,474 =
0.8987) over-predicts cosmics. Measuring the scale in the pure-cosmic
flash-chi2 sideband of the pre-chi2 selection (`ext_norm_sideband.py`,
scale = (data - MC nu) / EXT raw):

| sideband | data | MC nu | EXT raw | -> scale |
|---|---|---|---|---|
| log10 chi2 > 3.0 | 2492 | 436.2 | 3376.0 | 0.609 |
| log10 chi2 > 3.5 | 1892 | 348.3 | 2522.0 | 0.612 |
| log10 chi2 > 4.0 | 1320 | 263.4 | 1756.0 | 0.602 |
| log10 chi2 > 4.5 | 803 | 178.2 | 1091.0 | 0.573 |

0.601 is adopted. It is self-consistent: data/prediction is 1.00-1.03 at
EVERY flash-chi2 cut from log10 2.25 to open, whereas 0.8987 gives 0.75-0.93
and a slope in chi2. This is the same ~30% deficit the pi0 suite flagged for
the run-1 EXT spill count (versions doc section 3c NOTE). One EXT half:
double the scale (1.202).

## Cutflow (both halves, EXT x0.601, cosmic BDT >= 0.164, vertex-free >= 0.5)

Efficiency denominators (w @ 4.4e19): combined 882.4 | in-FV 137.9 | entering
744.5 (no-mu 466.3 + CC-mu 278.2) | strict 1g0X 48.8. Raw signal events 9,296
(the run-3 cew6 sample had 1,658).

flash chi2 < 316 (log10 2.5) — `plots_run1_novtx/`:

| step | sig | eff | effFV | effOut | eff1g0X | pur | MC bkg | EXT | pred | data | d/p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S1 nu slice | 635.9 | 0.721 | 0.721 | 0.721 | 0.786 | 0.009 | 12175 | 54272 | 67083 | 68985 | 1.03 |
| S2 1 photon | 366.9 | 0.416 | 0.577 | 0.386 | 0.562 | 0.046 | 2117 | 5567 | 8051 | 8522 | 1.06 |
| S3 no vis X | 329.5 | 0.373 | 0.521 | 0.346 | 0.522 | 0.070 | 974 | 3409 | 4712 | 4659 | 0.99 |
| S4 flash chi2 | 223.8 | 0.254 | 0.402 | 0.226 | 0.376 | 0.320 | 294 | 180 | 698 | 717 | 1.03 |

flash chi2 < 178 (log10 2.25, the eff x purity optimum) —
`plots_run1_novtx_chi2_225/`: final eff 0.220, in-FV 0.352, 1g0X 0.320,
purity **0.390**, MC bkg 232, EXT 71, pred 496, data 495, **d/p 1.00**.

Cut scan (EXT x0.601):

| log10 chi2 | eff | effFV | eff1g0X | purity | d/p |
|---|---|---|---|---|---|
| 2.00 | 0.177 | 0.280 | 0.250 | 0.446 | 0.91 |
| 2.25 | 0.220 | 0.352 | 0.320 | 0.390 | 1.00 |
| 2.50 | 0.254 | 0.402 | 0.376 | 0.320 | 1.03 |
| 2.75 | 0.280 | 0.444 | 0.436 | 0.247 | 1.03 |
| 3.00 | 0.298 | 0.457 | 0.448 | 0.183 | 1.01 |
| open | 0.352 | 0.501 | 0.496 | 0.080 | 1.01 |

Final composition at chi2 < 316 (weighted): sig 1g0X 18.3 | sig 1g+X 37.1 |
sig entering no-mu 114.3 | sig entering CC-mu 54.1 | bkg entering >=2g 61.9 |
entering CC 60.2 | entering other 11.4 | in-FV CC numu 57.0 | in-FV CC nue 9.6
| in-FV >=2 visible g 79.2 | no visible g 14.9 | EXT 180.3 | data 717.

## Vertex vs no-vertex samples (chi2 < 316, `plots_run1_novtx/paths/`)

| final step | VERTEX sample | NO-VERTEX sample |
|---|---|---|
| sig 1g0X (eff) | 15.9 (0.326) | 2.5 (0.050) |
| sig 1g+X (eff) | 34.2 (0.384) | 2.8 (0.032) |
| sig entering no-mu / CC-mu | 40.6 / 21.3 | 73.7 / 32.8 |
| total signal | 112.0 | 111.8 |
| MC bkg / EXT | 173.4 / 102.8 | 120.8 / 77.5 |
| purity (in-FV / entering) | 0.289 (0.129 / 0.160) | 0.360 (0.017 / 0.343) |
| data / pred | 1.10 | 0.93 |

The no-vertex sample stays ~95% entering by signal content, as on run 3.

## Run-1 vs the run-3 comparison it replaces

| final step, chi2 < 316 | run-3 cew6 fresh pair | **run-1 table-gamma** |
|---|---|---|
| eff combined | 0.218 | 0.254 |
| eff in-FV | 0.405 | 0.402 |
| eff 1g0X | 0.525 | 0.376 |
| purity | 0.342 | 0.320 |
| data / pred | 1.08 | **1.03** |
| raw signal events | 1,658 (odd half x2) | **9,296** |

The data/MC comparison is the real gain: 0.99-1.06 at every cut step, and
1.00-1.03 across the whole chi2 range, versus a 8-35% excess before. The
1g0X efficiency moves 0.525 -> 0.376, but the run-3 number rested on ~84 raw
events in the hygiene half; run 1 measures it on 516 raw 1g0X events, so the
lower value is the better-determined one (run-1 detector conditions also
differ: lower light yield, noisier electronics).
