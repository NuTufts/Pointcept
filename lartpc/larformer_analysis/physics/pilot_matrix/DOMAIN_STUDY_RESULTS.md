# Reco-chain domain study — results log (2026-08-11 … 2026-08-16)

Campaign question: the retrained (v2) LArFormer chain improved every
training/val metric yet **selected fewer CC1pi0 events on the official
overlay pilot** than the old production chain. This document records every
study run to localize, mechanism-test, and quantify that regression, with
script and output locations for each result.

**Conventions used throughout**
- "old chain" = LoRA deghoster (τ=0.5) + old slicer + old stage-3.
- "v2 chain" = ft PTv3-decoder deghoster (τ=0.2) + m2f-v2 slicer + m2f stage-3.
- Overlay pilot = first 10k events of the official run3b bnb-nu overlay
  (satfix); its 174 CC1pi0 signal events / 348 photons are the standard
  probe set (`inputlists/merged_sp_pi0sig_174.txt`).
- Photon-charge metrics are de-double-counted (`dedup_charge` q_comb),
  charge-weighted, with a 0.25 cm cell-inheritance join to the kp2/sliceid
  point sets (exact-position joins are WRONG — kp2 stores one representative
  per dedup cell).
- Paths below are relative to the kpv2_pointcept repo root. `NTUP` =
  `lartpc/larformer_reco/output/pilot_ntuples/`.

---

## 1. Pilot matrix: {old, new chain} × {pred, true vertex} efficiency/purity

True-vertex seeding (`--true-vertex` in
`lartpc/larformer_reco/scripts/run_nu_reco.py`) decouples reco from the
stale stage-4 keypoint model. 8 ntuples over bnb-nu 10k + nue 5k pilots.

| nue CC (step-5 WP, MC-only) | eff | purity |
|---|---|---|
| old pred / true | 0.543 / 0.557 | 0.707 / 0.757 |
| v2 pred / true | 0.465 / 0.481 | 0.837 / 0.915 |

CC1pi0 (174 sig): old 0.391 → v2 0.316 eff at ~0.70 purity.

- Scripts: `compare_pilot_matrix.py` (this dir),
  `lartpc/larformer_reco/scripts/run_nu_reco.py` (`--true-vertex`).
- Outputs: `NTUP/dlgen2_pilot_{old,new}_{bnbnu,nue}_{pred,true}.root`,
  `NTUP/summary_pilot_matrix.txt`, nue tables in `NTUP/nue_tables/`.

## 2. CC1pi0 signal cutflow — where events die

Same 174 events, ntuple cutflow (vtx → ≥2 conf-γ → +μ → +0cπ → +mgg):
old 166→100→74→70; v2 169→89→58→56. Photon loss persists without
AttConfident ⇒ photon PID/reco; extra muon-step loss ⇒ track KE / μ-PID.
- Script: inline in session; superseded by `light_cc1pi0_cutflow.py` (§15).

## 3. Root cause pass 1 — photon charge flow through the chain

Per-photon dedup-charge fate through kp2 (slice membership + instance class)
and the full-cloud sliceid decomposition (ghost / nu-slice / cosmic /
unclustered):

| photon charge | old | v2 |
|---|---|---|
| nu slice | 0.427 | 0.292 |
| cosmic slice | 0.008 | 0.009 |
| unclustered | 0.311 | 0.230 |
| **ghosted** | 0.247 | **0.417** |

⇒ segmenter labeling clean both chains; cosmic leakage negligible;
**deghoster is the largest single term**. Keep-vs-τ curves: LoRA 0.854@0.2 /
0.746@0.5; ft 0.531@0.2 and no τ recovers it (0.696 even at τ=0.05).
Also: the 300k spacepoint cap costs 5.2% of photon charge (500k: 0.7%) —
inference configs now use 500k.

- Scripts: `pi0_photon_charge_flow.py`, `pi0_photon_sliceid_decomp.py`,
  `photon_keep_from_preal.py` (all this dir).
- Outputs: `NTUP/photon_charge_flow/` (records, summary, PNGs),
  `NTUP/sliceids_old/`, `NTUP/sliceids_new/` (per-SP slice_id + p_real via
  `--slice-ids-only` in
  `tools/larformer/run_larformer_keypoint2_cascade_inference.py`).

## 4. Plumbing validation — deghoster inference setup is correct

`tools/larformer/run_deghost_eval.py`: trainval mode reproduces training
val (mIoU 0.788/mAcc 0.887 = reported 0.77/0.89); cascade-feed mode on the
same 2k files matches (real-recall@0.2 0.926 vs 0.933). Config review:
LArTPCDataset vs LArFormerDataset feeds identical (same LogTransform_v6
strength, coord match to 0.06%).
- Outputs: `logs/pilot_matrix/deghost_eval.*.log`.

## 5. Input-domain measurements

- Pixval spectra {LANTERN val vs overlay}, split true-nu vs rest
  (`pixval_domain_compare.py` → `NTUP/pixval_domain/`): true-nu (MC vs MC!)
  medians −10%, p95 −20…−36% in overlay; rest: 3× more Y-plane pixval≤0
  (data cosmics dead/saturated channels).
- Context-proximity discriminator (`context_proximity_recall.py`): MC-nu
  deghost recall vs distance to nearest data-cosmic point. Context
  corruption REJECTED — ft recall *declines* with isolation (0.574 → 0.022),
  i.e. it kills isolated MC deposits (where photons live); LoRA healthy.

## 6. Hybrid interim test — exposes the slicer

LoRA@0.2 + new slicer/stage-3 (configs
`stage3_particle/larformer-fullcascade-hybrid-loradeghost-tau020.py`,
`stage4_keypoint/larformer-keypoint2-fullcascade-hybrid.py`): photon
in-slice 0.383 (not recovered); cutflow 167→93→58→54 ≈ v2. Ratio
in-slice/delivered: corsika ~0.98 vs overlay 0.45–0.57 for every slicer ⇒
slicer also degrades on overlay.
- Outputs: `NTUP/photon_charge_flow_hybrid/`, `NTUP/kp2_hybrid_pi0sig/`,
  `NTUP/dlgen2_hybrid_pi0sig.root`.

## 7. Encoder A/B (P5B.3 sim+data mix) + operating-point methodology

New cells trained (configs in `stage1_deghost/`):
`deghost-ptv3decoder-p5b3mix-v1.py` (crop, ep17 used),
`lorafinetune-sonata-p5b3mix-deghost.py` (ep25; LANTERN-data caveat).
Judged with keep curves + **matched in-domain ghost acceptance** (fixed-τ
comparisons are calibration-confounded; LoRAs are much looser at equal τ —
`keep_significance.py` paired bootstrap + recall-at-matched-ghost-acceptance;
`keep_matched_overlay.py` for the cross-domain verdict):

| overlay keep @ τ*(ga=0.20) | v7-LoRA | v7-ft | v7-crop | P5B3-dec | P5B3-LoRA |
|---|---|---|---|---|---|
| overlay / transfer | **0.792** / 0.97 | 0.619 / 0.68 | 0.716 / 0.79 | 0.640 / 0.72 | 0.627 / 0.74 |

⇒ mixed encoder buys no robustness (both rows); **the full-event ft step
itself damaged transfer** (crop 0.79 vs ft 0.68 at identical in-domain);
deployed LoRA essentially domain-free.
- Outputs: `NTUP/preal_{v7cropdec,p5b3dec,p5b3lora}/`,
  `logs/pilot_matrix/{ab_*,matched_ab,keep_significance}*.log`.

## 8. Lineage correction (wandb git hash 60e8452)

The deployed robust LoRA sits on the **v6-noghosts** pretrain (sim-only SSL,
ghosts dropped, cosmics suppressed) — not v7-extbnb; same encoder as slicer
+ stage-3 (whole chain shares one foundation). prod4 (v2_expandedclasses)
and LANTERN (v3_larmatch) are v2/v3 conversions of the **same user sim**.
Loader charge-handling code unchanged since the May-14 training commit.

## 9. Two-converter experiment — conversion exonerated

15 user-sim dlmerged converted with the stepA converter and compared to the
v3 files of the same events: **pixval exactly equal** on >99.99% of shared
points; **v3 = exact strict subset** of the stepA cloud (the lm≥0.15 cut is
the only difference; an earlier 26.4% "v3-only" figure was a script bug).
Same PrepMatchTriplets config (threshold 10.0) verified in both pipelines.
⇒ the overlay-vs-LANTERN nu spectral shift is genuine **official-sim vs
user-sim**, not converter-induced.
- Outputs/scripts: `NTUP/converter_study/` (stepA_h5, membership_check.py,
  v3only_characterize.py, dlmerged_15.txt).
- Known wart (harmless here, fix pending): stepA converter hardcodes
  chstatus product "wire" (files may carry "wiremc"); bash octal fileno bug
  in the study wrapper.

## 10. LArMatch fold-in for official files + parity tests — composition exonerated

New production-shaped stage A0:
`lartpc/data_prep/uboone_official/run_stepA0_larmatch.sh` (LArMatch deploy
only — official files already carry SSNet/ubspurn; container warts
documented in-script), `attach_larmatch_scores.py` ((tick,u,v,y) join →
real lm_score into h5 copies), `submit_stepA0_pi0sig_array.sh`.
Satfix production input list = `inputlists/dlmerged_scale1500_resolved.txt`
(fileno = line). Deploy ~16 s/event CPU. The 0.15 cut costs only **0.4% of
photon charge** while removing ~⅓ of triplets.

**Deghoster parity test** (cut applied via dataset `lm_score_val_threshold`
0.15, `run_deghost_eval --lm-threshold`): keep@0.2 improves only +0.01…+0.06
(ft 0.531→0.586) ⇒ refuted as dominant mechanism.
**Slicer parity test** (hybrid ± cut, config
`larformer-keypoint2-fullcascade-hybrid-lmcut.py`): in-slice 0.383→0.387 ⇒
refuted at the slicer too.
- Outputs: `NTUP/larmatch_pilot/` (deploy), `NTUP/pilot174_lmscored/`
  (scored copies + preal dumps + parity keep table in
  `logs/pilot_matrix/parity_curves*.log`), `NTUP/slicer_lmcut_decomp.txt`,
  `NTUP/sliceids_hybrid_{uncut,lmcut}/`.

## 11. Old-slicer control at matched footing

Configs `stage4_keypoint/larformer-keypoint2-fullcascade-oldslicer-tau020{,-lmcut}.py`:

| photon in-slice (overlay) | uncut | lm≥0.15 |
|---|---|---|
| old slicer, LoRA@0.2 | **0.467** | 0.460 |
| new slicer, LoRA@0.2 | 0.383 | 0.387 |

⇒ slicer retrain regression = −0.084 (18% rel) at identical deghosting;
**old slicer + LoRA@τ0.2 = best zero-retraining config** (+0.040 over
production τ0.5; purity side unchecked but cosmic leakage 0.007).
- Outputs: `NTUP/oldslicer_tau02_decomp.txt`,
  `NTUP/sliceids_oldslicer_tau02_{uncut,lmcut}/`.

## 12. Val CC1pi0 twin sample (in-domain probe)

`select_cc1pi0_from_val.py`: 200 events / 400 photons from the LANTERN val
list, SBND truth definition mirrored (tree stores the pi0 directly —
pid 111 primaries, photons by parent linkage; energy_mev = KE; orphan
fallback for old productions; **LArFormerDataset sorts data lists** —
selector emits sorted artifacts).
- Outputs: `NTUP/val_cc1pi0/` (files.txt, photon_records.npz, preal_* for
  5 deghosters, keep tables in `logs/pilot_matrix/valcc_curves*.log`).
- In-domain keep@0.2: v7-LoRA 0.884 | v7-ft 0.860 | v7-crop 0.860 |
  P5B3-dec 0.870 | P5B3-LoRA 0.927 — all fine in-domain; ft's overlay
  collapse (0.860→0.531) = pure domain, isolation pathology overlay-only.

## 13. Slicer-stage ceiling (val vs overlay)

`NTUP/val_ceiling_decomp.txt` (sliceids in `NTUP/sliceids_val_{oldchain,v2chain}/`):

| photon in-slice | val | overlay | Δ |
|---|---|---|---|
| old chain | 0.470 | 0.427 | −0.043 |
| v2 chain | **0.558** | 0.292 | −0.266 |

Retrain works in-domain (+0.088); v2 is ~6× more domain-sensitive.
Honest ceiling: ~0.29 of CC1pi0 photon charge unclustered even in-domain
(soft detached showers). NOTE: not comparable to the earlier 0.79
slicer-completeness number (different sample + metric footing).

## 14. Statistical/methodology notes

- Paired event-level bootstrap for keep differences (`keep_significance.py`)
  — all pairwise cell differences significant, not sample luck.
- Cross-model comparisons must be made at matched ghost acceptance
  (in-domain-anchored τ*), never fixed τ.

## 15. Full-chain physics payoff (the decision table)

`light_cc1pi0_cutflow.py` — CC1pi0 reco cutflow directly from nu_reco
shards (LArFormer-side cuts only; validated against the ntuple cutflow to
±1 event/step), pre-flash, one implementation across all 4 cells (val runs
are flash-less, `--no-flash`):

| CC1pi0 eff (pre-flash) | val (in-domain) | overlay | Δ |
|---|---|---|---|
| old chain | 0.398 | 0.402 | **flat** |
| v2 chain | **0.490** | 0.322 | −0.168 |

**Domain-hardening payoff: +0.088 absolute (+22% relative) CC1pi0
efficiency over deployed physics; the old chain proves domain-flatness is
achievable. Remaining gap to SPINE (~0.65) is in-domain reconstruction
(soft-photon clustering), not domain.**

### Cutflow table — where signal events are lost, old vs new chain

Event counts surviving each cut (light cutflow, pre-flash; denominators =
true signal counts: 200 val / 174 overlay). Percentages in parentheses =
survival relative to the PREVIOUS row (step survival), so a low value marks
the cut where events die.

| cut step | val OLD | val V2 | overlay OLD | overlay V2 |
|---|---|---|---|---|
| true CC1pi0 signal | 200 | 200 | 174 | 174 |
| nu slice found + reco'd | 196 (98%) | 199 (100%) | 169 (97%) | 173 (99%) |
| primary vtx in FV | 193 (98%) | 197 (99%) | 167 (99%) | 171 (99%) |
| ≥2 confident γ (E>20) | 118 (61%) | 129 (65%) | 101 (60%) | **89 (52%)** |
| + primary μ (KE>143) | 81 (69%) | **104 (81%)** | 73 (72%) | **58 (65%)** |
| + 0 charged π (KE>25) | 78 (96%) | 98 (94%) | 70 (96%) | 56 (97%) |
| + m(γγ) < 400 MeV | 78 (100%) | 98 (100%) | 70 (100%) | 56 (100%) |
| **eff (pre-flash)** | **0.390** | **0.490** | **0.402** | **0.322** |

Readings:
- Slice/vertex finding is near-lossless everywhere and slightly BETTER in
  the v2 chain (its one domain-robust improvement).
- The two big cuts everywhere are **≥2γ** (~35–48% step loss) and **μ**
  (~19–35%); cπ-veto and mγγ are nearly free.
- The v2 chain's in-domain gains concentrate at the **μ step** (81% vs 69%
  survival) with a smaller γ-step gain (65% vs 61%).
- On overlay the v2 chain loses exactly those gains and more: γ step drops
  to 52% (old 60%) and μ to 65% (old 72%) — the same track-KE/PID and
  photon quantities the feature-level domain shift degrades.
- The old chain's per-step survivals are domain-flat (61/69 val vs 60/72
  overlay), mirroring its flat total efficiency.

Footnote: the val-OLD log prints eff 0.398 using its 196 kp2-found events
as denominator; the table uses the honest 200-signal denominator (0.390).

### Truth-matched decomposed cutflow — clustering vs classification vs attachment

`cutflow_decomposed.py` splits the photon and muon steps into sub-steps
(interleaved as ADDITIONAL truth-matched requirements, so efficiencies here
are lower than the selection cutflow above — this is the
"correctly-reconstructed" cutflow; the plain selection can pass with wrong
objects):
- *found*: some kp2 instance holds >=20% of the true particle's dedup
  charge (both photons / highest-KE primary muon) — clustering success
- *ID*: that >=20% instance is correctly classed — classification success
- *cut*: the original reco requirement passes — attachment / KE threshold

| step (survival vs prev) | val OLD | val V2 | ovl OLD | ovl V2 |
|---|---|---|---|---|
| vtx in FV | 193 | 197 | 167 | 171 |
| γ **found** (both ≥20%) | 122 (63%) | 142 (72%) | 100 (60%) | **53 (31%)** |
| γ ID | 118 (97%) | 140 (99%) | 98 (98%) | 52 (98%) |
| γ cut (≥2 conf attached) | 96 (81%) | 117 (84%) | 78 (80%) | 46 (88%) |
| μ found (≥20%) | 83 (86%) | 114 (97%) | 68 (87%) | 42 (91%) |
| μ ID | 78 (94%) | 112 (98%) | 67 (99%) | 37 (88%) |
| μ cut (primary, KE>143) | 62 (79%) | 94 (84%) | 52 (78%) | 27 (73%) |
| + cπ veto + mγγ | 60 | 88 | 49 | 27 |
| **eff (truth-matched)** | **0.300** | **0.440** | **0.282** | **0.155** |

Readings:
- **The v2 chain's overlay collapse is a γ-FOUND collapse**: 72% in-domain
  → 31% on overlay, while γ-ID is 98–99% in every cell (classifier fully
  exonerated) and γ-attachment is flat (80–88%). The domain shift destroys
  upstream instance FORMATION (deghost + slice + cluster), matching the
  charge-level story exactly.
- Muon sub-steps are secondary: found/ID healthy; the v2 overlay μ-cut dip
  (73%) is the residual track-KE/attachment term.
- γ-found is the biggest in-domain loss too (63–72%) — the soft-photon
  clustering ceiling in cutflow form.
- On the truth-matched metric the payoff GROWS: hardening ceiling 0.440 vs
  deployed old 0.282 = **+0.158 (+56% relative)** correctly-reconstructed
  CC1pi0 efficiency — the old chain passes the plain selection more often
  with wrong objects, which the truth-matched cutflow does not credit.
- Outputs: `logs/pilot_matrix/cutflow_decomposed.*.log`; script
  `cutflow_decomposed.py` (this dir).

- Outputs: `NTUP/kp2_val_{oldchain,v2chain}/`,
  `lartpc/larformer_reco/output/nu_reco_val_{oldchain,v2chain}/`,
  `logs/pilot_matrix/light_cutflow.*.log`.

## 16. Mechanism verdict + open items

Eliminated with measurements: operating threshold (§3), inference plumbing
(§4), conversion pipeline (§9), larmatch-cut composition (§10 both stages),
context corruption (§5), encoder choice (§7). Remaining:
**feature-level official-sim vs user-sim charge/response shift** (real,
measured §5/§9) interacting with head capacity (LoRA tolerant, big heads
fragile), + bounded data-cosmic context term.

Pending / next:
- Data-isolation cell `sonata/lora_deghost_v6noghosts_lantern` (job
  2450950; config `lorafinetune-sonata-v6noghosts-lantern-deghost.py`) —
  robust ⇒ capacity×shift story (augmentation-first remedy); collapse ⇒
  training-data story (mixed-sim training remedy). Judge at ~ep25 with the
  standard two-domain battery.
- {val, overlay, DATA} photon domain-shift measurement (beam-on pi0-selected
  photons = data leg; EXT beam-off = cosmic data leg) — adjudicates which
  sim is closer to data (diffusion suspicion; drift-axis structure already
  seen in the P5B.3 DCTR study).
- Hardening options ranked: mixed-sim supervised training; physics
  augmentations (charge scale ±30%, drift-dependent smearing, dead-channel
  dropout); EXT weak-label training; DANN/CDAN + DCTR as second wave.
  Overlay + data batteries become mandatory checkpoint gates.
