# Domain-shift measurement plan: data / val-sim / overlay-sim triangle
### (object-conditioned spacepoint embeddings for CC1pi0 photons + muon)

*Written 2026-08-16. Companion to `DOMAIN_STUDY_RESULTS.md` (same dir),
which records why: the v2 chain's overlay collapse is a feature-level
domain shift; this plan measures it against DATA and adjudicates which sim
is closer to data (diffusion suspicion: the overlay may carry the bad
parameter).*

## Locked design decisions

| choice | value | why |
|---|---|---|
| selection chain | **OLD chain** (LoRA@0.5 + old slicer + old stage-3) | measured domain-flat (CC1pi0 eff 0.398 val / 0.402 overlay) — selection composition not a function of domain |
| data working point | **WITH flash cut** (pi0mass_peak WP, chi2<1e4) | user choice: purity over stats |
| embedding space | frozen **P5B.3** encoder, full-event forward, object points selected post-hoc | deployment-realistic context; flagship space of the isambard domain study |
| P5B.3 checkpoint | `/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/isambard_pointcept/sonata/p5b/P5B.3-mix_larmatch-s0/model/epoch_18.pth` (config `isambard_pointcept/configs/lartpc/p05/pretrain-sonata-p5b3-mix-larmatch-detsym.py`) | |
| footing parity | lm >= 0.15 on ALL samples | val already cut at production; overlay via `pilot174_lmscored`; DATA needs a stepA0 pass |
| stats unit | EVENT-level bootstrap + same-sample split-half nulls | points within an event correlated (isambard convention) |
| tool location | ported INTO kpv2: `lartpc/larformer_analysis/domain_shift/` | user choice |

## Samples and object isolation

Per event, three objects: photon-1, photon-2 (the two selected showers),
muon (the selected primary track). Points = kp2 `particle/{i}/point_idx`
of the reco objects chosen by the selection (NOT truth masks — procedure
identical across samples; data has no truth).

1. **DATA**: bnb5e19 pi0 candidates at the flash-cut WP.
   - Selection source: `lartpc/larformer_analysis/physics/pi0mass_peak/`
     (working point per its README; selection tables `*_satfix_table.npz`,
     data ntuple + kp2 files under
     `lartpc/larformer_reco/output/<bnb5e19 tags>/` — lists
     `outputlists/keypoint2_out_bnb5e19_full_{nu,fm}.txt`).
   - Object indices: rebuild per-event from the ntuple's selected shower/
     track rows -> `part_inst_idx` -> kp2 `particle/{inst}` groups (the
     exporter stores `inst_idx`; same path the pilot studies used).
   - NEW: stepA0 larmatch pass over the candidate events' dlmerged
     (`lartpc/data_prep/uboone_official/run_stepA0_larmatch.sh`, data mode:
     ADC/chstatus `wire`, `-tb`, `--is-data` semantics at attach) + score
     attach -> lm>=0.15 parity copies.
2. **VAL twin** (200 evts): `NTUP/val_cc1pi0/` + old-chain reco
   (`nu_reco_val_oldchain`, kp2 `NTUP/kp2_val_oldchain/`). Apply the SAME
   reco selection (light cutflow objects). Truth cross-check: fraction of
   selected object points truly photon/muon (contamination systematic).
3. **OVERLAY pilot** (174 evts): old-chain reco
   (`nu_reco_pilot_old_bnbnu_pred`, kp2 subset list
   `outputlists/keypoint2_out_bnbnu_satfix_pilot10k_nu.txt`); lm-scored
   copies exist (`NTUP/pilot174_lmscored/`). Same selection + truth check.

`NTUP` = `lartpc/larformer_reco/output/pilot_ntuples/`.

## Measurements

**M1 — MMD triangle.** Per object class (gamma pts, mu pts separately):
pairwise multi-kernel MMD^2 for {data-val, data-overlay, val-overlay},
reported as MMD^2/null95 (same-sample event-level split-half null) +
permutation p + linear & kNN C2ST AUC. Tooling: ported
`domain_metrics.py`, `bootstrap.py`, `compute_metrics.py`.
Headline: which sim is closer to data, per object class.

**M2 — Witness-function forensics.** For each pair: kernel witness
f(z) = mean_k(z,A) − mean_k(z,B) on held-out points; rank |witness|;
(a) correlation table witness vs {drift x, charge, per-plane pixval,
lm_score, local density, dist-to-dead-channel, hit width from M3};
(b) exemplar event-display crops of top-|witness| points.
(Small extension to domain_metrics.py.)

**M3 — Waveform hit widths (diffusion adjudication).** New tool: larcv
image2d reader; for MUON-track manifest points on small-angle track
segments (3D direction cut to limit path-length broadening), extract
per-plane waveform in a ±N-tick window at (plane, wire); measure Gaussian
sigma / FWHM, amplitude, integral. Headline: **width vs drift distance
slope per sample** = effective longitudinal diffusion, three-way.
Conventions: data/overlay = tick-backward, ADC `wire`; val-sim =
tick-forward, ADC `wiremc`; dlmerged sources: data = bnb5e19 official
list, overlay = `inputlists/dlmerged_scale1500_resolved.txt` (fileno=line),
val-sim = `lantern` dlmerged under
`/cluster/tufts/wongjiradlab/larbys/data/ub_on_tufts/dlmerged_scratch/`.

## Execution steps (tick as done)

- [x] S0 port isambard tools -> `lartpc/larformer_analysis/domain_shift/`
      (`extract_features.py`, `domain_metrics.py`, `bootstrap.py`,
      `compute_metrics.py`; source:
      `isambard_pointcept/lartpc/pretraining_studies/domain_shift/`)
- [x] S1 object-manifest builder
      (`lartpc/larformer_analysis/domain_shift/build_object_manifest.py`).
      Sim legs DONE (job 2462123, `NTUP/build_domain_manifests.sh`):
      `NTUP/domain_manifests/manifest_{val,ovl}_oldchain.npz` — val 78
      evts / 152k pts, ovl 70 evts / 150k pts (both == light-cutflow mgg
      counts; 0 cell-join misses). Charge purity comparable across sims
      (gamma 0.82-0.90, mu ~0.89) so contamination won't confound M1.
      Real lm on both legs (ovl via pilot174 copies; 99.5% >= 0.15).
- [x] S2 data-leg parity. FOUND: data nu_reco shards
      (`.../mcc9_v28_wctagger_bnb5e19/nu_reco_streams_nu`, event_N = kp2
      nu-list position) carry `flash_chi2` as an event attr, so the flash
      cut is applied in the builder (`--flash-chi2-max 1e4`) — no
      ntuple-table join needed. Candidate run = job 2462208
      (`NTUP/build_domain_manifest_data.sh`) ->
      `manifest_data_prelm.npz` (pt_lm = stepA dummy): **261 flash-cut
      CC1pi0 candidates / 534k object points** from 54,626 nu-stream
      events. stepA0 data smoke clean (job 2462331, fileno 132). Full
      chain IN FLIGHT: stepA0 array 2462498 (258 files, lists
      `data_pi0_filenos.txt` + `merged_sp_data_pi0_261.txt`) -> attach
      2462529 (aftercorr) -> `NTUP/data_pi0_lmscored/` -> manifest
      rebuild -> data feats -> full triangle. COMPLETE (rerun chain
      2463783-86 after aftercorr preemption-cancel hiccup; all 258
      larmatch files + 261 lm copies present).
      Data-file quirk: empty mc_particle_tree => pt_true_class=0
      (no-truth) rather than -1; treat data truth as absent.
- [x] S3 embedding extraction — implemented as companion script
      `domain_shift/extract_manifest_features.py` (imports the ported
      extractor's helpers; whole-event forward, lm>=0.15 parity cut,
      manifest point -> upcast feature via raw-cm NN, tol 0.5 cm).
      ALL THREE LEGS DONE (sims job 2462348, data job 2463785):
      `NTUP/domain_manifests/feats_{val,ovl}_oldchain.npz` +
      `feats_data_bnb5e19.npz` — 822/822 ckpt keys, match rates
      0.92/0.90/0.90, median match dist 0.346 cm everywhere.
- [ ] S4 run M1 battery. Tools written: `domain_shift/triangle_metrics.py`
      (event-block MMD^2 + block permutation + split-half nulls +
      GroupKFold C2ST) and `domain_shift/run_triangle.py`.
      **FULL TRIANGLE DONE** (job 2463786, `results_triangle.json`).
      VERDICT: data ~ OVERLAY (gamma pooled MMD^2 0.0043, p=0.088,
      x0.5 null95, kNN AUC 0.51; mu 0.0029, p=0.25, x0.2, kNN 0.47) —
      statistically indistinguishable. VAL far from BOTH (data-val
      gamma x19.2 null95 linAUC 0.988, mu x5.9 linAUC 1.000; val-ovl
      gamma x9.9, mu x4.8). Decision rule fires: route 2 REJECTED —
      overlay stays as expectation; the NEW sim carries the
      mismodeling; hardening/fix targets come from S5/S6.
- [x] S5 witness extension + M2 outputs DONE (job 2464878,
      `NTUP/domain_manifests/witness_*.npz`; tables in the log +
      DOMAIN_STUDY_RESULTS.md section 18). Headline axes of the val-sim
      direction: NN-spacing (val sparser; mu ovl-val rho -0.38 strongest)
      + per-plane pixval (+0.19..+0.26) + q_cell; mu ovl-val drift
      correlation +0.22 (diffusion-suggestive). data-ovl residual: no
      density component. BONUS probe (job 2465705, section 18): cap-100k
      eval — slicers density-INSENSITIVE at eval (v2 domain gap identical
      at both densities => exploited features are charge-spectral, not
      density texture); deghosters ARE density-sensitive (keep full
      density at inference). Old-cap-robustness hypothesis: eval-side
      form refuted; training-side form only testable by retrain ablation;
      underfit-old-run alternative raised (see section 18).
- [x] S6 DONE (results: DOMAIN_STUDY_RESULTS.md section 19; fits
      `NTUP/domain_manifests/hitwidth_fits.json`). VERDICT: val == ovl
      in effective D_L (Y 6.3 vs 6.3-6.4 cm^2/s — same sim setting);
      BOTH sims ~1.7x data (Y 3.67 +- 0.19, matches published uB
      measurement). Diffusion is NOT the val-vs-ovl axis; second-order
      in embedding space. val-manifest leg ran on 77 tier2-restored
      dlmerged (now in dlmerged_scratch); truth-vs-manifest modes agree.
      DESIGN CHANGE: dlmerged image2d are 6-tick-row
      compressed (cannot resolve diffusion-scale widths) but ALL THREE
      samples carry `hit_gaushit_tree` (larlite, raw-tick RMS from the
      same hit finder) => widths from GAUSHIT RMS, not image fits.
      Tools: `domain_shift/waveform_hit_width.py` (muon points from
      manifest OR truth-h5 mode; local-PCA |dir_x|<0.25 segment cut;
      per-sample SELF-CALIBRATED tick<->PeakTime transform — all three
      legs land on tick=peak+~2397; smokes validated on hand-checked
      pulses) + `domain_shift/hitwidth_fit.py` (binned-median RMS^2 vs
      drift WLS -> sigma_0 + effective D_L per sample/plane, event
      bootstrap). VAL LEG CAVEAT: manifest events' prod2 dlmerged were
      scratch-cleaned (1/78 survive) => val uses TRUTH-selected primary
      muons from surviving filenos 1-199 (same production; list
      `inputlists/val_prod2_surviving_dlmerged_h5.txt`); overlay runs
      BOTH manifest and truth modes to bound the selection systematic.
      Full runs = job 2490926 -> `NTUP/domain_manifests/hitwidth_*.npz`.
      dlmerged lists: data `mcc9_v28_wctagger_bnb5e19.txt`, ovl
      `dlmerged_scale1500_resolved.txt`, val `bnb_nu_corsika_prod2.txt`
      + path-remap into dlmerged_scratch (fileno=line everywhere).
- [x] S7 synthesis DONE — DOMAIN_STUDY_RESULTS.md section 20. Route:
      overlay stays as expectation (route 2 rejected); fix axes =
      charge response/spectra + triplet-formation sparsity (NOT
      diffusion, NOT eval-density); chain remedy = overlay-domain
      training files / charge augmentation (+ density dropout cheap);
      sim remedy = match charge response, retune D_L to ~3.7 for both
      sims; the S1-S6 tooling is a standing checkpoint-gate battery.

## Decision rules

- val closer to data than overlay (both object classes) => route 2: new
  sim becomes the analysis expectation; overlay demoted to cross-check.
- overlay closer => overlay stays; hardening targets = witness-identified
  axes.
- Either way: M2/M3 outputs define the augmentation set (charge scale,
  drift-dependent smearing, dead-channel dropout) for the chain-wide
  retrain, and the overlay+data batteries stay as checkpoint gates.

## Queued studies (user-approved, not yet run)

- **Triangle battery in v6-noghosts feature space**: rerun
  extract_manifest_features + run_triangle with the v6-noghosts
  config/ckpt (`pretrain-sonata-v1m1-lartpc-v6-logspace-resume` /
  `lartpc_v6_h200_noghosts_pretrain_logspace_resume/model/epoch_42.pth`)
  on the SAME three manifests. Prediction (section 24 encoder story):
  val-vs-ovl separability collapses vs P5B.3's linAUC 0.99 — the robust
  encoder "can barely see the domain". Doubles as a quantitative
  encoder-selection gate for the retrain campaign (min domain
  separability at fixed probe performance).
- NOT queued (deemed redundant, 2026-08-18): v6-lantern deghoster +
  v2-slicer cascade on overlay — the section 6 hybrid already measured
  this to within the deghoster difference (v6-lantern == deployed LoRA
  overlay keep at matched ga); expected outcome = collapse persists.
  Slicer questions are answered by TRAINING-side A/Bs only (+ optional
  early-checkpoint underfit probe).

## Open/pending elsewhere (context for whoever resumes)

- Data-isolation training cell `sonata/lora_deghost_v6noghosts_lantern`
  (job 2450950): judge at ~ep25 with `tools/larformer/run_deghost_eval.py`
  + `photon_keep_from_preal.py` two-domain battery.
- P5B.3-LoRA training job 2360962 may still hold 2xA100 (user to decide).
- Hygiene fixes pending: stepA converter chstatus product name ("wire"
  hardcode vs "wiremc" files); octal fileno bug in
  `NTUP/converter_study/convert_one.sh`.
