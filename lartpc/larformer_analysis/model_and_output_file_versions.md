# Model & output-file versions for analysis work

Maintained reference for which checkpoints / calibrations / files define an
analysis version, so a new session can run additional final-state analyses
(CC nue, CC numu inclusive, ...) against a consistent chain. Update this file
when a version is promoted.

**CURRENT VERSION: v2_s1ep2p8cew6** (second section below).

---

## VERSION v2_s1ep2p8  (PREVIOUS — superseded by v2_s1ep2p8cew6 below;
## files retained as comparators, do not delete)

Frozen 2026-08-27; classifier/calibration additions through 2026-09-05.
All paths relative to the kpv2 checkout
`/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept` (= $K)
unless absolute. Everything runs inside the container
`/cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif`
(`run_in_tufts_pointcept_container.sh`).

### 1. Chain checkpoints (reco inference)

| stage | checkpoint | how it is selected |
|---|---|---|
| deghoster | v6-lantern LoRA ep25 | baked into the cascade config |
| slicer | `exp/larformer_slicer_s1_mixenriched_v1/model/epoch_2.pth` | env `LARFORMER_BATTERY_SLICER_CKPT` |
| segmenter (stage-3 particle) | `exp/larformer_particle_s1cache_m2frecipe/model/epoch_8.pth` | env `LARFORMER_KP_PARTICLE_CKPT` |
| keypoint | attempt-2 (old) | baked into the cascade config |
| cascade config | `configs/lartpc/larformer/stage4_keypoint/larformer-keypoint2-fullcascade-v6lantern-envslicer.py` | `--config` |
| LArPID | alternate network weights when the sample TAG contains `run3` (MC); default weights for data/EXT | automatic in `larpid/apply_larpid.py` |

Inference: `tools/larformer/run_larformer_keypoint2_cascade_inference.py`
with `--output-tree` and WITHOUT `--no-flash` for anything needing flash-chi2
(satfix knobs default "auto"). Since 2026-09-03 the kp2 particle groups also
persist dedup provenance attrs (`runnerup_class/prob`, `n_absorbed`).

CANDIDATE segmenter upgrades (NOT promoted; mu-PID rebalance campaign):
- `exp/larformer_particle_s1cache_m2frecipe_rebal_v2_warm/model/epoch_4.pth`
  (validated: mu->mu .723->.815, mu->e .164->.107 on the 2000-event subset,
  gamma held; pi->mu up .095->.19)
- `exp/larformer_particle_s1cache_m2frecipe_rebal_v2_warm2_cew/` — 6-epoch
  CE-weighted continuation TRAINING NOW (job 3297938). Gate = confusion A/B
  on `lartpc/larformer_reco/inputlists/merged_sp_confusion_test2000.txt`.

### 2. Reco-internal calibrations / tables

- nu_reco attachment: `--attach-llr-tables
  lartpc/larformer_reco/trajfit/data/attachment_llr_tables_s1ep2p8.npz
  --attach-llr-thr 4.0`
- `lartpc/larformer_reco/trajfit/data/calo_calib.npz` (deployed):
  e_a 0.019743, gamma_a 0.01553, gamma_b -12.80 ("recal3").
  CAUTION: the v2_s1ep2p8 NTUPLES below were exported while the OLD gamma
  constants were deployed (NTUP_G_A 0.020101, NTUP_G_B -15.49), so photon
  energies in ntuple `showerRecoE` need the analysis-side recal:
  `--recal-gamma-a 0.01553 --recal-gamma-b -12.80` (invert-and-reapply;
  supported by pi0_mass_analysis / sbnd_cc1pi0 / datamc_diagnostics).
  A future re-export will bake recal3 in; then drop the flags.
- SCE fwd/bkwd: `lartpc/flashmatch/sce_microboone.py` (self-contained npz).
- Flash model: gamma_beam 5.25; dead PMT {15}; saturation-hole mask.

### 3. Classifiers (analysis level)

- Per-shower cosmic BDT (OFFICIAL): `lartpc/larformer_reco/export/data/
  shower_cosmic_bdt.joblib` — exporter fills ntuple branch
  `showerCosmicScore` when env `LARFORMER_SHOWER_BDT` points at it
  (electrons autopass 1.0; photons get score; -9 for RecoE<=0).
  Trained on the segmenter-training pi0 corpus + EXT rows<100k.
- Event-level EXT BDT (flash-blind): `lartpc/larformer_analysis/physics/
  pi0mass_peak/ext_bdt_model_flashblind.joblib` (thr 0.280 standalone).
  Trained on even-event signal-MC + even-event EXT.
- STAGED 97% working point (adopted for pi0 plots): shower score >= 0.192
  then event BDT >= 0.21; flash-chi2 CC<1e4 / NC<1778 per stream, or
  combined-stream chi2 < 3162 (log10 3.5).
- Muon finder for CC tagging: `--muon-finder union` (segmenter-mu OR
  larpid-mu-argmax on primary tracks) measured best (CC eff .648 vs .607).

### 4. Analysis samples — FINAL ntuples (current exporter: expanded labels,
###    charge/TID purities, showerCosmicScore)

Base `D=/cluster/tufts/wongjiradlab/larbys/data/ub_on_tufts`:

| sample | ntuple | events | notes |
|---|---|---|---|
| MC BNB-nu overlay (run3b) | `$D/larformer_mcoverlay67k_s1ep2p8/dlgen2_larformer_ntuple_mc_overlay_s1ep2p8_run3.root` | 67,211 | POT 8.394e19 -> scale 0.5242 to 4.4e19; labels EXPANDED; archives `*_prescore/_pointpurity/_qpurity_origlabels.root` |
| bnb5e19 beam data (run1) | `$D/larformer_bnb5e19_s1ep2p8/dlgen2_larformer_ntuple_bnb5e19_s1ep2p8.root` | 176,302 | POT 4.4e19 |
| EXT-BNB beam-off (run3) | `$D/larformer_extbnb200k_s1ep2p8_flash/dlgen2_larformer_ntuple_extbnb200k_s1ep2p8f.root` | 200,000 | full-sample spill scale 0.17682554549; this 200k subset 0.5909 |
| pi0-BDT training corpus | `/cluster/tufts/wongjiradlab/larbys/data/ub_on_tufts/larformer_overlaytrain_pi0_bdt/dlgen2_larformer_ntuple_overlaytrain_pi0_bdt_run3.root` | 40,589 | DATA-MODE export (no truth sidecar); training only, never plot |

EXT HYGIENE (must respect in ALL analyses):
- rows < 100,000: per-shower-BDT training half — exclude from plots.
- even `event`%2: event-BDT training half — exclude when that BDT is used.
- analysis convention: rows>=100k (+odd if event BDT), scale 1.1818
  (x2 more if odd-only).
MC hygiene: even-event true-signal (pi0 table cat<2) trained the event BDT.
The per-shower BDT never saw the analysis MC.

Per-sample intermediates (same directories): `keypoint2_streams/` (cascade
h5, flash-matched; per-PMT obs/pred arrays), `nu_reco_larpid_{nu,fm}/`,
per-event RSE caches `pi0mass_peak/rse_*_keypoint2_streams.npz`.
MC truth sidecar: `lartpc/larformer_reco/output/
mcc9_bnbnu_overlay_1500_full_satfix/truth_sidecar` (chain-independent).
Input lists: `lartpc/larformer_reco/inputlists/merged_sp_<TAG>.txt`;
kp2 stream lists: `outputlists/keypoint2_out_<TAG>_{nu,fm}.txt`.
Selection tables (row-aligned npz; w + flash_chi2 + sel masks):
`lartpc/larformer_analysis/physics/pi0mass_peak/
{mc,data,ext}_s1ep2p8_recal3[_staged|_stagedun|_stagedlp]{,_bdt,_nochi2bdt,
_analysishalf...}_table.npz`.

### 4b. "novtx" re-export (2026-09-06; exporter change, same reco chain)

Exporter (`lartpc/larformer_reco/export/export_gen2ntuple.py`) now also writes
VERTEX-LESS prongs: every segmenter particle of the nu-/fm-slice kp2 file that
is not part of a nu_reco interaction (incl. all particles of events with no
interaction) appears with `{track,shower}VtxIdx = -1` (start = kp2 start_cm,
PCA direction, comb Charge, calo showerRecoE, LArPID defaults, truth match by
kp2 gt_trackid with a dominant-labelled-TID fallback; `--orphan-min-points 10`).
New branches: `{track,shower}Objectness` (1-P(no_object)), `{track,shower}Stream`,
`nuSliceFlashChi2`/`fmSliceFlashChi2` (+`*NParticles`), `showerNoVtxScore`
(vertex-free per-shower BDT via env `LARFORMER_SHOWER_BDT_NOVTX`; -9 without).
showerRecoE is now RECOMPUTED at export from the comb Charge with the deployed
`calo_calib.npz` (recal3 baked in) so attached and vertex-less showers share one
calibration -> analyses must NOT apply `--recal-gamma-*` to these files.
`showerCosmicScore` is unchanged w.r.t. the canonical ntuples except showers
whose recalibrated energy clips to 0 (-9 sentinel). Canonical files untouched.

| sample | novtx ntuple |
|---|---|
| MC BNB-nu overlay | `$D/larformer_mcoverlay67k_s1ep2p8/dlgen2_larformer_ntuple_mc_overlay_s1ep2p8_run3_novtx.root` |
| EXT-BNB 200k | `$D/larformer_extbnb200k_s1ep2p8_flash/dlgen2_larformer_ntuple_extbnb200k_s1ep2p8f_novtx.root` |
| bnb5e19 data | `$D/larformer_bnb5e19_s1ep2p8/dlgen2_larformer_ntuple_bnb5e19_s1ep2p8_novtx.root` |

Start finder v2 (2026-09-07, `shower_start_dir()`): DBSCAN clusters with
>= 5 MeV of charge, PCA extrema, keypoint origin picks the start end; the
`*_novtx.root` files above carry it (v1 kept as `*_novtx_v1kpstart.root`).
Vertex-free shower BDT: `lartpc/larformer_analysis/physics/single_photon/
shower_novtx_bdt.py` (signal = MC EVEN-event nu photons, bkg = EXT rows<100k;
analyses using it take ODD MC x2 + EXT rows>=100k) -> model
`lartpc/larformer_reco/export/data/shower_novtx_bdt.joblib` (held-out AUC
0.957; top features objectness, piS, sdwall, muS). The three novtx ntuples
above were re-exported with `LARFORMER_SHOWER_BDT_NOVTX` set, so
`showerNoVtxScore` is FILLED (validated equal to analysis-side scoring to
3e-8; every other branch identical). Single-photon result with the
vertex-less candidate path (NoVtx BDT >= 0.5, flash chi2 < 316): combined
efficiency 0.099 -> 0.196 at purity 0.323 -> 0.357 (see single_photon/README).

### 5. Samples NOT yet at v2_s1ep2p8 (need reprocessing before use)

- intrinsic-nue overlay (run3b, `mcc9_v29e_nue_overlay`): merged_sp +
  truth_sidecar exist (`lartpc/larformer_reco/output/mcc9_v29e_nue_overlay/`,
  79,356 events) but the ntuple + kp2/nu_reco there are JULY OLD-CHAIN
  (no showerCosmicScore/TrueUnlabeledPurity; labels NOT expanded).
  To use for CC-nue at v2_s1ep2p8: complete labels
  (`lartpc/data_prep/uboone_official/complete_labels.py`, idempotent),
  then rerun kp2 (env ckpts above, no --no-flash) -> nu_reco (LLR s1ep2p8)
  -> larpid -> export with TRUTH_DIR + LARFORMER_SHOWER_BDT. Follow the
  submit pattern in `lartpc/larformer_reco/slurm/` (see README "Reproducing
  with the v2 chain" in pi0mass_peak).
- Any other July-era ntuple under `larformer_reco/output/*` (check for the
  `showerCosmicScore` branch as the version fingerprint).

### 6. Training / eval assets (for retrains & performance evals)

- Training-data ledger (hygiene splits, TRAINPOOL vs RESERVED, EVAL_LOCKED):
  `lartpc/data_prep/uboone_official/training_data_ledger/` (see LEDGER.md).
  RESERVED halves are the analysis-side pools for new-sample processing.
- Overlay training corpus: `/cluster/tufts/wongjiradlab/larbys/data/
  larformer/overlay_train/<sample>/merged_sp/` — label-completed; repacked
  2026-09-05 (68% smaller, contents bitwise-verified). Includes the NEW
  generic-numu tranche (run3b 20,773 + run1 12,812 events, nu-deposit>=20pt
  dirt filter; dud filenos in `generic_mu_tranche1_dud_filenos.txt`).
- Stage-1/2 training cache (segmenter): `/cluster/tufts/wongjiradlab/
  larbys/data/ub_on_tufts/hdf5/larformer_cache_stage12__s1ep2_v6lantern_tau020/`
  (train+val; particle_class_id-augmented). Rebalanced train list:
  `training_data_ledger/cachelist_rebalanced_train_v2.txt` (281,452 events;
  census: e 214.6k / g 248.1k / mu 114.7k / pi 153.6k / p 360.7k).
  Older caches (m2fv2ep4, highermaxsp) DELETED 2026-09-05 (regenerable).
- Reco performance eval (this chain): records
  `lartpc/larformer_reco/results_eval_reco_mc_overlay_s1ep2p8.npz`, plots
  `lartpc/larformer_reco/plots/{eval_reco_mc_overlay_s1ep2p8_streams_nu_intpc,
  ntuple_compare_mc_overlay_s1ep2p8}/`.
- Confusion test subset (checkpoint A/B):
  `lartpc/larformer_reco/inputlists/merged_sp_confusion_test2000.txt`.

### 7. Known conventions / caveats carried by this version

- A_GAMMA visible-energy convention is ~2.2x actual photon energy
  (signal-definition threshold 20 MeV E_vis ~ 9 MeV actual). Unresolved.
- Flash model: data chi2 peak shifted+broadened vs MC (global scale+
  resolution); set aside — do not retune per-analysis.
- Segmenter mu->e confusion ~10% (KE-flat) in THIS version's checkpoint;
  mitigations available (union muon finder; larpid relabel needs track
  refit for momentum). Rebalance retrain in progress (see §1 candidates).
- LArTPCDataset larmatch key mismatch: MC h5 `lm_score`, EXT `larmatch_score`.
- Rolling log of all decisions/jobs:
  `lartpc/larformer_analysis/physics/pilot_matrix/SLICER_RETRAIN_PLAN.md`.


---

## VERSION v2_s1ep2p8cew6  (CURRENT / OFFICIAL — promoted 2026-09-11)

Reprocessing COMPLETE 2026-09-09; benchmark suite + talk tables done
2026-09-10. Same conventions as v2_s1ep2p8 except as noted. Base
`$D=/cluster/tufts/wongjiradlab/larbys/data/ub_on_tufts`.

### 1. Chain deltas vs v2_s1ep2p8
- Segmenter (stage-3): `exp/larformer_particle_s1cache_m2frecipe_rebal_v2_warm2_cew/model/epoch_6.pth`
  (rebalanced list + inverse-sqrt CE weights, warm from epoch_8-era
  rebal_v2_warm/epoch_4). Chain-level confusion (full 67k, primaries
  KE>50, purity>0.5): mu->mu .783->.868, mu->e .105->.054, gamma->gamma
  .912->.919, pi->mu .082->.128. EXT cosmic photon candidates -35%.
- Everything else identical (deghoster/slicer/keypoint, LLR s1ep2p8
  thr 4.0, LArPID weights rule, shower-BDT env model).
- kp2 particle groups now also carry dedup provenance attrs
  (runnerup_class/prob, n_absorbed).

### 2. Calibration — IMPORTANT DIFFERENCE
The cew6 chain re-ran nu_reco fresh with the deployed calo_calib, so
recal3 (gamma a=0.01553, b=-12.80) is BAKED INTO showerRecoE.
- Do NOT pass --recal-gamma-* to analysis scripts.
- For scripts whose recal args have non-None defaults, pass the
  IDENTITY constants: `--recal-gamma-a 0.020101 --recal-gamma-b -15.49`.
- Verified: matched-photon RecoE/TrueE median 0.945 (old ntuples 1.218).

### 3. Final ntuples (current exporter: expanded-label truth, charge/TID
###    purities, showerCosmicScore; entries verified)
| sample | ntuple | events |
|---|---|---|
| MC BNB-nu overlay | `$D/larformer_mcoverlay67k_s1ep2p8cew6/dlgen2_larformer_ntuple_mc_overlay_s1ep2p8cew6_run3.root` | 67,211 |
| bnb5e19 beam data | `$D/larformer_bnb5e19_s1ep2p8cew6/dlgen2_larformer_ntuple_bnb5e19_s1ep2p8cew6.root` | 176,302 |
| EXT-BNB beam-off | `$D/larformer_extbnb200k_s1ep2p8cew6/dlgen2_larformer_ntuple_extbnb200k_s1ep2p8cew6.root` | 200,000 |
Normalizations, EXT/MC hygiene halves: identical to v2_s1ep2p8 (same
underlying events, same row conventions).

### 4. Intermediates (per sample dir)
`keypoint2_streams/` (flash-matched cascade h5, --output-tree layout),
`nu_reco_streams_{nu,fm}/`, `nu_reco_larpid_{nu,fm}/`, plus
`$K/lartpc/larformer_reco/inputlists/merged_sp_<TAG>.txt` and
`outputlists/keypoint2_out_<TAG>_{nu,fm}.txt` with TAGs
`mc_overlay_s1ep2p8cew6_run3`, `bnb5e19_s1ep2p8cew6`,
`extbnb200k_s1ep2p8cew6`. MC truth sidecar: reused
(`larformer_reco/output/mcc9_bnbnu_overlay_1500_full_satfix/truth_sidecar`).

### 5. Talk working point + classifier status
- WP: shower cosmicScore >= 0.075 (eff-0.97 on cew6; old 0.192 -> eff
  0.938), event BDT >= 0.21, UNION muon finder, chi2 1e4/1778
  (combined-stream 3162).
- Shower BDT RETRAINED on cew6 (2026-09-10): model
  `export/data/shower_cosmic_bdt_cew6.joblib` (AUC 0.968; eff 0.97 ->
  rej 0.773 @ thr 0.164; corpus reprocessed at
  $D/larformer_overlaytrain_pi0bdt_cew6/). NOT yet promoted — the
  ntuple showerCosmicScore branches still carry the ep8 model; promote
  via LARFORMER_SHOWER_BDT at the next re-export, or re-score
  analysis-side (features are ntuple branches).
- Event BDT RETRAINED on cew6 (2026-09-10): model
  `pi0mass_peak/ext_bdt_model_flashblind_cew6.joblib` (AUC 0.957;
  staged te 0.21 -> sig 0.985 / EXT rej 0.487 vs stale 0.333).
- FRESH PAIR PROMOTED (2026-09-11): ntuples re-exported with
  shower_cosmic_bdt_cew6 -> showerCosmicScore branch is now the cew6
  model (ep8-score versions archived as *_ep8score.root). FINAL talk
  WP: shower >= 0.164, event >= 0.21, union. Suite + tables rebuilt
  (CEW6_TALK_TABLES.md = fresh-pair numbers). Net: intermediate-stage
  purity way up (pre-chi2 EXT -38%), post-chi2 selections unchanged;
  data/pred 0.85-0.90 dip persists (NOT a classifier-staleness effect
  — chain-level, still open).

### 6. Selection tables + results docs (pi0mass_peak/)
- Base tables (WP, no BDT cuts): `{mc,data,ext}_s1ep2p8cew6_table.npz`
- Talk-WP tables: `{mc,data,ext}_cew6_ts0075_table.npz`
  (+`_bdt` = event-BDT+hygiene applied; `_nochi2bdt` = chi2-open BDT
  scoring for chi2 spectra/step tables; `ext_*_analysishalf_chi2inf` =
  SBND ext gate). A/B alternate: `*_cew6_ts0192_*`.
- TABLES + numbers: `CEW6_TALK_TABLES.md` (pi0 step tables, SBND
  ordered cutflow, ep8 comparators, conventions).
- Plot dirs: `plots_cew6_overlay_ts0075/` (and _ts0192),
  `plots_cew6_flashchi2_ts0075/`, `plots_cew6_sbnd/`,
  `plots_cew6_sbnd_sbndflow/`, `plots_{mc,data,ext}_cew6_base/`.

### 7. Benchmark summary (vs v2_s1ep2p8)
- pi0 combined ge2 final: 0.690 eff @ 0.675 purity (ep8 0.690 @ 0.700)
- CC eq2: 0.553 @ 0.754 (ep8-seg 0.514 @ 0.798); NC eq2: 0.547 @ 0.572
  (ep8 0.547 @ 0.557)
- SBND ordered cutflow final: 0.500 @ 0.874 (ep8-union 0.515 @ 0.875);
  standing format 0.534 @ 0.778 (0.561 @ 0.788)
- Passing SIGNAL core verified identical across chains (1,087 shared
  events) — differences are in admitted background.
- Confusion A/B subset: `larformer_reco/inputlists/merged_sp_confusion_test2000.txt`;
  kp2 outputs `larformer_reco/output/kp2_conf_test2000_{old,new,cew6}/`.

### 7c. Single-photon study, FRESH-PAIR ntuples (2026-09-10)
Verified on the promoted ntuples: `showerCosmicScore` = cew6 model (WP 0.164),
but `showerNoVtxScore` was still the EP8 vertex-free model — the exporter
default `export/data/shower_novtx_bdt.joblib` held ep8 while the cew6 retrain
sat as `shower_novtx_bdt_cew6.joblib`. FIXED: the default is now the cew6
model (ep8 archived `shower_novtx_bdt_ep8.joblib`); the next re-export bakes
it, until then pass `--novtx-model`. Single-photon result with both cew6
BDTs: eff 0.218 / purity 0.342, EXT 85, data/pred 1.08 (ep8 chain 0.198 /
0.350; cew6 + stale BDTs 0.219 / 0.307). Tables:
`physics/single_photon/CEW6_SINGLE_PHOTON_TABLES.md`.

### 7b. Single-photon study on cew6 (2026-09-09)
cew6 ntuples verified to carry the vertex-less prongs, objectness, slice chi2
and baked showerNoVtxScore (v2 start finder). Same cuts as ep8: combined eff
0.198 -> 0.219, purity 0.350 -> 0.307 (EXT 97 -> 142 with the ep8-trained
cosmic BDT); with the vertex-free BDT retrained on cew6
(`export/data/shower_novtx_bdt_cew6.joblib`, NOT yet deployed as default)
purity 0.334 at EXT 93. Details: `physics/single_photon/README.md`.

### 8. Status
v2_s1ep2p8cew6 is the TALK version. v2_s1ep2p8 files (section above)
remain in place as comparators — do not delete before the talk.
Post-talk queue: retrain shower+event BDTs on cew6, data/pred dip
investigation, segmenter options in docs/reference/Segmenter_Improvement_Options.md.
