# Electron Neutrino CC Analysis

This folder is dedicated to selecting electron neutrino charged current (CC) events from the MicroBooNE LArTPC data.

## Samples

Intrinsic nue events are only 0.5% of BNB flux. So we need a dedicated MC sample for them. Right now the sample is on Tier 2 and needs to be copied over. It's location is 

  tier2:wongjiradlab/larbys/data/mcc9/mcc9_v29e_dl_run3b_bnb_intrinsic_nue_overlay_nocrtremerge

and takes up 766.21 GB of space.

## Truth definition

 - events with true vertex 10 cm away from the wall
 - nue CC events
 - primary electron energy > 20 MeV
 - primary electron visible energy (using the deduped wire plane pixel sum energy) > 20 MeV

## Selection

 - one primary electron attached to one of the nu candidate slice's vertices
 - vertex in the WireCell Fiducial Volume
 - log10( flashmatch_chi2 ) < 3.0. The cut value shoud be determined based on a plot comparing the log(flashmatch_chi2)
 - in LANTERN, additional cuts were used:
    - an "electron confidence score: logit(p_e) - 0.5*(logit(p_pi) + logit(p_gamma)) < 0.0
    - an muon confidence cut: logit(p_mu) < -3.7 
    - a vertex score cut to remove cosmics
    - hopefully cuts like these are not required
    - an 'electron primariness' based on LArPID outputs: primary_score > fromcharged_score && primary_score > fromneutral_score"
    - for these LArPID based cuts, the variables should be plotted first to see how they separate signal from background.

# LANTERN benchmarck

Inclusive nue CC events were selected with about 55% efficiency and 90% purity, which was state-of-art in MicroBooNE.

# Analysis scripts

Two-stage, mirroring `../pi0mass_peak/` (build per-sample tables, then stack):

1. `nue_cc_analysis.py --ntuple <ntuple.root> --out tables/<tag>.npz [--data]`
   Reads one gen2ntuple, applies the reco selection, writes a per-event table.
   MC weight `w = xsecWeight * (--pot / sum potTree.totGoodPOT)` (`--pot` default
   4.4e19 = the Tufts bnb5e19 beam livetime). `--data` = unit weights, no truth.
   Stores: `sel`, `reco_ele_E`, `flash_chi2`, `w`, `nu_pdg`, `ccnc`, `is_nuecc`
   (veto flag), `is_nuecc_fv` (signal), the leading e-shower LArPID scores
   (log-softmax), the LArFormer/segmentation-model scores (`lf_*_score` =
   log(prob); the raw LArFormer outputs are softmax PROBABILITIES, logged here so
   the same confidence formulas apply), `vtx_mu_score` / `vtx_lf_mu_score`
   (max muon score of OTHER particles at the e-vertex, LArPID / LArFormer), and
   `vtx_dist_true`.
2. `nue_cc_overlay.py --nue-npz --bnb-npz --ext-npz --data-npz --plots <dir>`
   Stacks nu_e-CC signal (nue overlay) + bnb-nu background (numu CC / NC, with
   true nue CC VETOED via `is_nuecc` to avoid double counting) + EXT cosmic
   (`--ext-scale` default 0.17682554549 spill ratio), overlays bnb5e19 data.
   Also makes MC-truth validation plots (need the MC tables): signal
   `eff_vs_true_ele_ke.png` + `eff_vs_true_ele_vise.png` (selection efficiency
   vs true electron KE / visible energy = A_GAMMA x primary-e pixel charge) and
   `bg_truth_pid.png` (truth-matched particle of the reco'd electron for MC bkg
   passing the cuts). Makes `reco_ele_energy.png` + `flashchi2.png` + `var_*.png` (a stacked
   prediction+data, log-y, for EVERY candidate cut variable at the current
   selection, with the cut line drawn if set) -- so a cutflow is just repeated
   runs adding one flag + a new `--plots` folder each step. Prints purity +
   true-signal efficiency. Cut flags: `--flashchi2-cut` (log10), `--elconf-cut`,
   `--primariness-cut`, `--mu-cut` (e-shower muon), `--vtxmu-cut` (vertex muon);
   `--no-var-plots` to skip the var_*.png.

Supporting studies:
- `nue_cc_larpid_scores.py` -- per-LArPID-variable stacked prediction + data and
  efficiency/purity-vs-cut scan (`--flash-lo/--flash-hi` band). Ranks the
  discriminants; e-confidence is strongest.
- `nue_cc_ext_norm.py` -- EXT-normalization diagnostic: flash-chi2 stacked + data
  with a data/pred ratio panel + a cosmic-sideband EXT-scale fit. Result: EXT is
  correctly normalized (fit 0.95x spill ratio); the data excess is a LOCALIZED
  bump at log10(flashchi2)~2.7-3.0, not a global/EXT offset.
- `add_observed_pe.py` + `nue_cc_observed_pe.py` -- the in-time flash observed PE
  is NOT in the ntuple; add_observed_pe scans the cascade `keypoint2_streams`
  (`flash/observed_pe`) by (run,subrun,event) and adds it to a table; the plotter
  overlays data vs prediction to test the Run-1(data)/Run-3(MC+EXT) light-yield
  hypothesis for the flash-chi2 excess.

## Samples (ntuples)

| role | ntuple |
|---|---|
| nu_e CC signal | `.../mcc9_v29e_dl_run3b_bnb_intrinsic_nue_overlay_nocrtremerge/dlgen2_larformer_ntuple_mcc9_v29e_nue_overlay.root` (POT 4.709e22; 2231 good files) |
| BNB nu background | `../../../larformer_reco/output/mcc9_bnbnu_overlay_1500_full_satfix/dlgen2_larformer_ntuple_*.root` (POT 8.394e19) |
| EXT cosmic | `.../mcc9_v29e_dl_run3_G1_extbnb_full/dlgen2_larformer_ntuple_extbnb_full.root` (668388 evts; spill weight 0.17682554549) |
| bnb5e19 beam data | `.../mcc9_v28_wctagger_bnb5e19/dlgen2_larformer_ntuple_bnb5e19_full.root` (176336 evts) |

Note: the intrinsic-nue sample is 100% CC (all `trueNuCCNC==0`), so it is a pure
nu_e-CC signal source; the bnb sample supplies numu-CC / NC background.

# Implemented Reco selection (first pass)

- `foundVertex==1 & primaryVtxStream==0 & vtxIsFiducial==1` (a reco nu-stream
  vertex in the fiducial volume)
- `>= 1` PRIMARY electron shower: `showerLArFormerPID==11 & showerIsSecondary==0
  & showerRecoE > 20 MeV`
- **observable** = leading (most energetic) primary-electron `showerRecoE`
- **flash-chi2 cut** `log10(flash_chi2) < 3.0` (provisional; `flash_chi2` = the
  primary nu-vtx `recoVtxFlashChi2`). Tune from `flashchi2.png`.

## LArPID electron cuts (from the score-separation study)

LArPID scores are LOG-softmax [e,gamma,mu,pi,p] + process [primary,fromN,fromC].
Single-variable max purity @ eff>=0.8 (in the flash-cut selection): e-confidence
0.37 (best), primariness 0.25, muon 0.21, e-score 0.13, e/gamma 0.10. NOTE the
README's "< 0.0" for e-confidence was wrong-signed -- signal is at HIGH
e-confidence; useful cut is `> ~7`.

**Selections** (flash + LArPID). e-confidence ALONE traces a better purity/eff
curve than combining cuts -- adding primariness+muon removes more signal than
background at matched efficiency, so they are net-negative here.

| selection (all with log10(flashchi2)<3) | purity | eff | plots |
|---|---|---|---|
| elconf>7 & primariness>0 & mu<-3.7 | 0.75 | 0.54 | `plots_selected/` |
| **elconf>9 alone** | **0.87** | 0.42 | `plots_selected_elconf9/` |
| (LANTERN benchmark) | 0.90 | 0.55 | |

elconf>9 alone reaches ~LANTERN purity with a single cut. In BOTH selections
data/pred ~= 0.72-0.74 (data below pred in the signal-dominated region) -> a
persistent ~28% deficit pointing to intrinsic-nue signal over-prediction OR a
data/MC electron-ID efficiency difference (not a selection artifact).

## Hard-background variables (`nue_cc_bg_vars.py`, `plots_bgvars/`)

Two background topologies (analyzer domain knowledge):
1. **muon at the e-vertex** (true e-shower from a decay mu/pi whose decay muon
   was merged into a track). `vtx_mu_score` = max LArPID muon score among the
   OTHER reco particles sharing the leading e-shower's vertex (tracks + other
   showers; distinct from the e-shower's OWN muon score). numu CC piles at
   log p_mu ~ 0 (a muon-like track at the vertex); signal is low. Usable RECO
   cut: `--vtxmu-cut`. On top of elconf>9, `vtxmu<-3.7` cuts numuCC 2.24->1.65
   and lifts purity **0.87 -> 0.90** (LANTERN benchmark) at eff 0.42->0.39
   (`plots_selected_elconf9_vtxmu/`).
1b. **pi0 / mis-identified gamma** (a pi0 photon reco'd as the electron, its
   partner photon still present). `n_photons` = # reco photons (LArFormerPID==22,
   >20 MeV) attached to the nu vertex. Cut `--nphoton-max` (e.g. 0). At elconf>9,
   `nphoton<=0` removes NC 1.57->1.05 (NC IS pi0, sits at n_photons=1) but leaves
   numuCC 2.24 UNCHANGED (the numuCC residual is NOT pi0 -> use the vertex-muon
   veto instead) at ~9% signal cost (signal brems photons). Photon-count and
   vertex-muon vetoes are COMPLEMENTARY (NC-pi0 vs numuCC).
2. **secondary-interaction chain** (n travels, makes a secondary interaction
   relabeled "true", pi->mu->e far from the nu vertex). No reco proxy yet;
   DIAGNOSED via `vtx_dist_true` = reco-vtx to SCE-corrected true-vtx distance
   (already in the ntuple as `vtxDistToTrue`, MC only). Non-nueCC bkg has a
   clear long tail: **>5cm = 29% of bkg vs 6% of signal** (extends to ~400cm).
   Worth finding a reco handle (displaced-vertex / vertex-activity tag).

  elconf = showerElScore - 0.5*(showerPiScore + showerPhScore)
  primariness = showerPrimaryScore - max(showerFromNeutralScore, showerFromChargedScore)
  mu = showerMuScore
(all for the leading primary e-shower.)


