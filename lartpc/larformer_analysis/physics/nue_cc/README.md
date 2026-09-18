# Electron Neutrino CC Analysis

Selecting electron-neutrino charged-current (CC) events from MicroBooNE LArTPC data.

**Chain version: v2_s1ep2p8cew6** (see
`../../model_and_output_file_versions.md`). recal3 is baked into `showerRecoE`,
so **no analysis-side `--recal-gamma-*` is applied anywhere in this folder**.

## Truth definition

 - true vertex in the WireCell FV (`trueVtxInWCFV==1`; ~10 cm from the wall)
 - nue CC (`|trueNuPDG|==12 & trueNuCCNC==0`)
 - a primary electron with E > 20 MeV

## Selection

 - a reco nu-stream vertex in the FV:
   `foundVertex==1 & primaryVtxStream==0 & vtxIsFiducial==1`
 - `>= 1` PRIMARY electron shower:
   `showerLArFormerPID==11 & showerIsSecondary==0 & showerRecoE > 20 MeV`
 - **observable** = leading (most energetic) primary-electron `showerRecoE`
 - `log10(flash_chi2) < 3.0` (provisional; choose it from `flashchi2.png`)
 - LArPID / LArFormer discriminants, chosen with `nue_cc_scan.py`

**LANTERN benchmark: ~55% efficiency at ~90% purity**, state of the art in
MicroBooNE.

## RESULTS on RUN 1, table-gamma (2026-09-18)  -- use these for data vs MC

All four samples are run 1 with the calibrated flash light-yield table
(versions doc section 3c), so run-1-vs-run-3 differences can no longer explain
a data/MC disagreement. Tables: `tables_run1/` (`slurm_build_tables.sh` with
`SET=run1_tg`); every table records its sample set and the plotters refuse to
mix sets.

| sample | ntuple (`$L=/cluster/tufts/wongjiradlab/larbys/data/larformer`) | norm |
|---|---|---|
| nu_e CC signal | `$L/run1_nueintrinsics_half/..._nue_run1_half.root` | POT 4.9065e22 |
| BNB nu bkg | `$L/run1_bnboverlay_half/..._bnbovl_run1_half.root` | POT 2.1292e20 after dropping the 350 filenos that fed segmenter training (`--exclude-rows pi0mass_peak/run1ovl_trainpool_exclusion.npz`) |
| EXT cosmic | `$L/run1_C1_extbnb_half/..._extbnb_run1_half.root` | 23,090,946 beam spills / (0.25 x 94,414,115 EXT spills) = **0.9783** per event |
| data | `$L/bnb5e19_run1_table/..._bnb5e19_run1_table.root` | 4.4e19 POT |

No classifier hygiene on run 1 (no BDT ever saw a run-1 event).

| selection | purity | eff | data / pred | plots |
|---|---|---|---|---|
| flash only | 0.017 | 0.748 | 4032 / 3377 = 1.19 | `plots_run1_flash/` |
| **elconf>9 & prim>4** | **0.942** | **0.452** | **40 / 37.5 = 1.07** | `plots_run1_wp_e9p4/` |
| **elconf>8 & prim>4** | **0.835** | **0.506** | 40 / 47.5 = 0.84 | `plots_run1_wp_e8p4/` |
| same, EXT at sideband fit 1.44 | 0.819 | 0.506 | 40 / 48.4 = 0.83 | `plots_run1_wp_e8p4_extfit/` |

**The signal-region data deficit is gone on run 1.** At `elconf>9 & prim>4`
data/pred = 1.07 (+-0.17 stat) where every run-3 working point sat at
0.72-0.78. The long-standing deficit was a run-1-data vs run-3-MC mismatch.

**Flash chi2 SHAPE now agrees** (`plots_run1_extnorm/flashchi2_extnorm.png`):
data/pred is flat at ~1.3 from log10 chi2 ~1.9 to 5.5. The localized
data/pred bump at log10 chi2 2.7-3.0 in the July analysis (run-1 data vs run-3
MC/EXT light yield) is gone. Residual: below log10 chi2 ~1.8 MC still has
somewhat better-matched flashes than data (ratio 0.2-0.8).

**Open -- the overall normalization at the LOOSE selection is ~1.3 high.**
Three estimates of the EXT per-event weight disagree:
spill-based 0.978 (nominal) | beam-trigger count ~1.31 (176,302 triggers minus
~39,300 weighted overlay events, over 104,516 EXT) | cosmic-sideband fit 1.44.
But the excess is FLAT in flash chi2, including the neutrino-dominated region
(log10 chi2 1.9-2.5, mostly numuCC), where raising EXT alone cannot close it:
it looks like a global factor rather than an EXT-only problem. Check the
bnb5e19 POT (4.4e19) and spill count before tuning EXT. The tight working
points are insensitive: EXT is 0.00 at `elconf>9 & prim>4` and 1.96 -> 2.88
at `elconf>8 & prim>4`.

**Cut tuning:** `run_nuecc_cutflow.sh` (defaults to run 1). Every step folder
gets the headline plots plus, for each of 19 candidate variables at that
step, `var_*.png` (stack + data | eff & purity vs threshold) and `cat_*.png`
(stacked by what the e-candidate really is), with `scan_summary.md` ranking
the best next cut. Override values from the env (`ELCONF=9 PRIM=2 ...`);
`VARPLOTS=0` skips the variable plots (~20 s/step).

To test a data-driven EXT correction, every plotter takes `--ext-rescale F`
(multiplies the spill-based weight; `--ext-scale W` replaces it outright), and
the cutflow driver takes `EXTRESCALE=F`. Reference points: x1.34 = trigger
count (weight ~1.31), x1.47 = cosmic-sideband fit (weight 1.44).

Also: `data_prep/README.md` quotes "25% of the full run-1 EXT -> 5,772,737
spills", which is 25% of the BEAM count (23,090,946), not of the EXT count
(94,414,115). The pi0 run-1 remake and versions doc section 3c used 5.77M.

Efficiency is lower than on run 3 (0.452 vs 0.501 at `elconf>9 & prim>4`):
different MC sample, and run 1 uses the LArPID DEFAULT weights while run 3 used
the alternate ones, so the run-3-tuned `elconf`/`primariness` thresholds are not
guaranteed optimal here; re-tune from `plots_run1_flash/var_*.png`.

## RESULTS on cew6 (2026-09-10)

All four samples on v2_s1ep2p8cew6, `log10(flashchi2)<3`, POT 4.4e19.

| selection | purity | eff | data/pred | plots |
|---|---|---|---|---|
| flash only | 0.021 | 0.742 | 1.10 | `plots_cew6_scan_flash/` |
| elconf>8 | 0.570 | 0.564 | 0.95 | |
| elconf>9 | 0.815 | 0.501 | 0.84 | `plots_cew6_scan_elconf9/` |
| **elconf>9 & primariness>4** | **0.903** | **0.501** | 0.78 | `plots_cew6_wp/` |
| **elconf>8 & primariness>4** | **0.877** | **0.548** | 0.69 | `plots_cew6_wp_e8p4/` |
| elconf>10 | 0.955 | 0.421 | 0.78 | |
| (July chain, elconf>9 & vtxmu<-3.7) | 0.90 | 0.39 | - | |
| (LANTERN) | 0.90 | 0.55 | - | |

**+28% relative efficiency at matched purity vs the July chain** (0.39 -> 0.50 at
purity 0.90). Base acceptance improved too: on identical events the cew6 chain
finds FV vertices 0.881 -> 0.917 and >=1 primary-e candidate 0.736 -> 0.776.

The two boxed rows are the recommended operating points: `elconf>9 & prim>4`
if you want to beat LANTERN on purity, `elconf>8 & prim>4` if you want to match
it on efficiency. Both put EXT cosmic at exactly 0.00.

**The remaining efficiency loss is a low-energy turn-on, not the PID cuts.** At
`elconf>8 & prim>4` the efficiency vs true electron KE is 0.03 (50 MeV) / 0.18
(150) / 0.34 (250) / 0.50 (400) / 0.58 (650) / 0.64 (1000) / **0.68 plateau**
(>1.6 GeV) -- see `plots_cew6_wp_e8p4/eff_vs_true_ele_ke.png`. The integrated
0.548 is dominated by soft electrons below ~400 MeV. Loosening PID further
cannot recover them; the handle is reconstruction of low-energy showers.

### Which background actually dominates -- it is NOT photons

Truth composition of the reco'd electron candidate at `elconf>9` (background
total 8.99):

| category | yield | share |
|---|---|---|
| **nu e (secondary)** -- Michel / delta ray | 3.94 | **44%** |
| EXT cosmic | 2.36 | 26% |
| nu photon (pi0 mis-ID) | 1.57 | 17% |
| nu muon | 0.60 | 7% |
| nu e (primary) in the bnb sample | 0.52 | 6% |

The July chain had photons at 41% of background; on cew6 they are **17%**,
consistent with the segmenter's gamma gains and EXT cosmic photon candidates
-35%. So the photon-BDT handles are aimed at the third-largest background, and
the photon vetoes measure as **net-negative** at this working point: adding
`n_good_photons<=0` costs efficiency 0.501 -> 0.449 for purity 0.903 -> 0.893.

The dominant background is secondary electrons, whose natural handle is
**primariness** (LArPID primary vs from-charged/from-neutral) -- exactly what
Michels and delta rays are. It is the top-ranked discriminant in the scan and it
also removes EXT cosmic entirely (2.36 -> 0.00), since cosmic-induced showers are
not primary either.

> This REVERSES the July conclusion that "primariness is net-negative here".
> That held when photons were 41% of the background; it does not hold on cew6.

Open: `data/pred ~ 0.78` at the best working points (the long-standing deficit),
and the EXT cosmic-sideband fit is 1.17x nominal (July was 0.95x).

## Samples (v2_s1ep2p8cew6)

Paths live in one place: `nue_cc_common.SAMPLES`.

| role | ntuple |
|---|---|
| nu_e CC signal | `<nue dir>/dlgen2_larformer_ntuple_nue_overlay_s1ep2p8cew6_run3.root` (POT 4.709e22; 2231 good filenos) |
| BNB nu background | `$D/larformer_mcoverlay67k_s1ep2p8cew6/dlgen2_larformer_ntuple_mc_overlay_s1ep2p8cew6_run3.root` (67,211 evts; POT 8.394e19) |
| EXT cosmic | `$D/larformer_extbnb200k_s1ep2p8cew6/dlgen2_larformer_ntuple_extbnb200k_s1ep2p8cew6.root` (200,000 evts) |
| bnb5e19 beam data | `$D/larformer_bnb5e19_s1ep2p8cew6/dlgen2_larformer_ntuple_bnb5e19_s1ep2p8cew6.root` (176,302 evts) |

`$D = /cluster/tufts/wongjiradlab/larbys/data/ub_on_tufts`. The intrinsic-nue
sample is 100% CC, so it is a pure nu_e-CC signal source; the bnb sample supplies
numu-CC / NC background with its true nue CC **vetoed** (`is_nuecc`) to avoid
double counting.

### EXT normalization -- CHANGED, do not carry the old number forward

Per-EXT-event weight is `0.17682554549 / f`, `f` = fraction of the FULL
668,388-event EXT sample processed. The July analysis used the full sample, so
its weight was the bare spill ratio. **The cew6 EXT file is a 200,000-event
subset**, so:

| EXT selection | fraction | per-event weight |
|---|---|---|
| all rows | 0.29923 | **0.5909** |
| rows >= 100k (shower-BDT analysis half) | 0.14962 | **1.1818** |
| rows >= 100k AND odd `event` | 0.07481 | **2.3636** |

Using the old 0.17683 undercounts EXT by 3.3x. `nue_cc_common.ext_scale()`
picks the right one, and switches to the hygiene half automatically whenever a
cut uses a shower-BDT score (EXT rows < 100k are that BDT's training half).

## The shower BDT scores: what they are NOT

`showerCosmicScore` and `showerNoVtxScore` are **cosmic-vs-neutrino-PHOTON**
classifiers. They are **not** e/gamma discriminators, and the exporter
hard-assigns every `LArFormerPID==11` shower a score of exactly **1.0**
(`export_gen2ntuple.py:532-534`, `:479-481`). Verified on the cew6 MC overlay:
10,838 / 10,838 electron-PID showers score exactly 1.0. So a `score >= t` cut
passes **every electron candidate unconditionally** and buys nothing.

They are still useful here, applied to *photons*, which is their training domain:

- `n_good_photons` -- photons at the electron's vertex with
  `showerCosmicScore >= 0.164` (cew6 WP). A sharper pi0 veto than the raw
  `n_photons`, which counts cosmic-contaminated showers too and so fires on
  signal events with a stray nearby cosmic (~9% signal cost in July).
- `n_novtx_photons` -- VERTEX-LESS photons (`showerVtxIdx == -1`,
  `showerStream == 0`) with `showerNoVtxScore >= 0.5`, catching pi0 partner
  photons that failed vertex attachment. Not possible before the novtx export.

To use a BDT on the electron candidate itself you must either re-score it
analysis-side (all 22 cosmic-BDT features are ntuple branches and are stored in
the table) or train a dedicated e-vs-gamma model. Which is worth doing is
decided from `bg_composition.png`, not assumed.

> `showerNoVtxScore` in the canonical cew6 ntuples is the **cew6** vertex-free
> model as of commit `0c3acef` (verified: 69% of photon showers differ from the
> archived `*_novtxep8.root`). No analysis-side re-scoring needed.

## Truth categories (`bg_cat`)

`nue_cc_analysis.py` tags what the reco'd electron candidate **actually is**:

`nu e (primary)` = the neutrino's primary electron (SIGNAL) · `nu e (fragment)`
= an EM daughter of it (shower split) · `nu e (secondary)` = Michel / delta ray
· `nu photon` = pi0 or other gamma mis-ID · `nu muon` / `nu pion` / `nu proton`
· `nu other` · `cosmic` · `unresolved`.

The cosmic test runs FIRST and is `showerTrueTID <= 0 | unlabeledPurity >= 0.5`.
It deliberately does **not** copy `single_photon/photon_bdt_study.py`'s
`showerTruePID <= 0`, which has two bugs:

- `<= 0` swallows every **negative PDG**. On cew6 electron candidates that
  mislabels 294 genuinely nu-matched showers as cosmic (257 positrons, 31 pi-,
  6 mu+) -- and not one of them has `unlabeledPurity >= 0.5`, i.e. none is
  actually cosmic-contaminated.
- `== 0` still mislabels the ~2.5% of showers with `truePID==0` but `TID>0`:
  neutrino-origin particles mcreco simply did not save.

In MCC9 overlay the cosmics are real off-beam DATA carrying no G4 labels, so
`origin==1 <=> trackid>0` exactly -- `TID<=0` *is* the cosmic tag, and
`showerTrueUnlabeledPurity` is the cosmic-contamination fraction of the shower.
This is also why label completion matters: without it the genuine-neutrino
shower periphery is unlabeled too, inflating `unl` and blurring the boundary.

## Analysis scripts

`nue_cc_common.py` is the single source of truth for sample paths, the EXT
scale and hygiene, the component/colour vocabulary, the derived-variable
formulas, the cut definitions, and one canonical `base()` mask. Every script
imports it; nothing re-declares those any more. (They used to, and had drifted:
`nue_cc_larpid_scores` applied a two-sided flash band where `nue_cc_overlay`
applied a one-sided cut, on the same tables.)

1. **`nue_cc_analysis.py`** `--ntuple X.root --out tables/tag.npz [--data]`
   One gen2ntuple -> a per-event table. MC weight
   `w = xsecWeight * (--pot / sum potTree.totGoodPOT)`, `--pot` default 4.4e19.
   Stores the selection, the observable, LArPID + LArFormer scores, the cew6
   handles (objectness, cosmic/novtx scores, slice flash-chi2, vertex quality,
   the raw cosmic-BDT features for re-scoring), the photon-veto counts, `row`
   (needed for EXT hygiene) and, for MC, `bg_cat`.
   Degrades gracefully on old-chain ntuples -- it names the missing branches.

2. **`nue_cc_scan.py`** -- *the cut-definition tool*. For every candidate
   variable at the current selection: `var_<key>.png` (stacked prediction +
   data, and efficiency/purity vs threshold with the best working point marked)
   and `cat_<key>.png` (the same variable stacked by **truth category**, which
   is what says whether a variable separates the background you actually have),
   plus `bg_composition.png` and `scan_summary.md`. Replaces the hand-edited
   `run_nuecc_cutflow.sh` loop.

3. **`nue_cc_overlay.py`** -- the headline plots: `flashchi2.png` (drawn with
   NO cuts applied, since it is what the flash cut is chosen from),
   `reco_ele_energy.png`, the `eff_vs_*.png` turn-ons, and
   `bg_truth_category.png`. Prints purity + efficiency.

Supporting studies (all now on the shared module):
`nue_cc_larpid_scores.py` (per-LArPID-variable stack + cut scan),
`nue_cc_ext_norm.py` (cosmic-sideband EXT-scale fit -- run this FIRST on any new
stack to confirm the normalization before trusting any physics),
`add_observed_pe.py` + `nue_cc_observed_pe.py` (in-time flash observed PE, not
in the ntuple; scanned from the cascade `keypoint2_streams`),
`nue_cc_bg_vars.py` (vertex-muon and reco-to-true-vertex diagnostics).

## Known background topologies (July, to be re-measured on cew6)

At `elconf>9` the background split by truth was electron 43% / photon 41% /
muon 16%:

1. **muon at the e-vertex** -- a true e-shower from a decay mu/pi whose decay
   muon merged into a track. `vtx_mu_score` targets it; `vtxmu<-3.7` on top of
   `elconf>9` lifted purity 0.87 -> 0.90 at eff 0.42 -> 0.39.
2. **pi0 / mis-identified gamma** -- a pi0 photon reco'd as the electron. The
   photon-count veto and the vertex-muon veto are COMPLEMENTARY (NC-pi0 vs
   numuCC): at `elconf>9`, `nphoton<=0` cut NC 1.57 -> 1.05 but left numuCC
   2.24 unchanged.
3. **secondary-interaction chain** (n -> pi -> mu -> e far from the nu vertex).
   No reco proxy yet; diagnosed via `vtx_dist_true` (MC only): >5 cm is 29% of
   background vs 6% of signal.

Also open: a persistent **data/pred ~ 0.72-0.74** in the signal-dominated
selection -- intrinsic-nue over-prediction, or a data/MC electron-ID efficiency
difference. Not a selection artifact.

  elconf      = showerElScore - 0.5*(showerPiScore + showerPhScore)
  primariness = showerPrimaryScore - max(showerFromNeutralScore, showerFromChargedScore)
  mu          = showerMuScore
(all for the leading primary e-shower; LArPID scores are log-softmax, and the
LArFormer probabilities are stored as log(prob) so the same formulas apply.)
