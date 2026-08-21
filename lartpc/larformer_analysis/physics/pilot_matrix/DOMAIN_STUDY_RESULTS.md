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

**γ-SLICE sub-step added (2026-08-17): slicer delivery vs segmenter
recognition.** New requirement interleaved between vtx-FV and γ-found:
BOTH true photons have >`--gslice-min` of dedup charge in the predicted
nu slice. Two thresholds run:

*Matched bar, 0.2 (jobs 2512592 + 2512743, script default; γ-deghost
from the --slice-ids-only sidecars, slice_id==-2 = ghosted).* Nested
chain — deghost-surviving ⊇ in-slice ⊇ instance charge — so γ-found
onward are UNCHANGED from the main table (consistency check passed):

| step (survival vs prev) | val OLD | val V2 | ovl OLD | ovl V2 |
|---|---|---|---|---|
| vtx in FV | 193 | 197 | 167 | 171 |
| γ deghost (both >20% survive) | 189 (98%) | 197 (100%) | 162 (97%) | **146 (85%)** |
| γ slice (both >20% in-slice) | 148 (78%) | 179 (91%) | 133 (82%) | **100 (68%)** |
| γ found \| slice | 122 (82%) | 142 (79%) | 100 (75%) | **53 (53%)** |
| γ ID             | 118 (97%) | 140 (99%) | 98 (98%) | 52 (98%) |
| γ cut (≥2 conf attached) | 96 (81%) | 117 (84%) | 78 (80%) | 46 (88%) |
| μ found (≥20%) | 83 (86%) | 114 (97%) | 68 (87%) | 42 (91%) |
| μ ID | 78 (94%) | 112 (98%) | 67 (99%) | 37 (88%) |
| μ cut (primary, KE>143) | 62 (79%) | 94 (84%) | 52 (78%) | 27 (73%) |
| + cπ veto + mγγ | 60 | 88 | 49 | 27 |
| **eff (truth-matched)** | **0.300** | **0.440** | **0.282** | **0.155** |

Reading: the v2 overlay collapse is DISTRIBUTED across the three
upstream stages — deghost −15 pts vs in-domain, slice −23, found −26
(multiplicative closure 0.85 x 0.68 x 0.53 = 0.31 = γ-found/vtx; old
chain is domain-flat at deghost 97-98% and slice 78-82%, closure
0.97 x 0.82 x 0.75 = 0.60). CAVEAT on stage attribution: all bars share
the photon's TOTAL charge as denominator, so deghosted charge never
reaches the slicer — the deghoster's charge-level damage (0.42 of
ovl-v2 photon charge ghosted, §3) largely surfaces DOWNSTREAM as
photons entering later steps hollowed out and hovering near the bars;
85% deghost survival only means ghosting alone rarely pushes a photon
below 20%. The v2 conditional γ-found dip (53%) is largely
marginal-delivery composition — photons delivered at 20-50% in-slice
can barely support a ≥20%-of-total instance (ovl-v2 mean in-slice 0.29
piles up just above the bar) — see the 0.5-bar control below.
Sidecar provenance: sliceid runs used cap 500k vs the v2 kp2
production's 300k; nesting is exact in these numbers.

*Delivery-controlled bar, 0.5 (job 2512447)* — when both photons are
majority-delivered, segmenter recognition is FLAT across all cells:

| step (survival vs prev) | val OLD | val V2 | ovl OLD | ovl V2 |
|---|---|---|---|---|
| vtx in FV | 193 | 197 | 167 | 171 |
| γ **slice** (both >50% in nu slice) | 57 (30%) | 92 (47%) | 26 (16%) | **2 (1%)** |
| γ found \| slice (both ≥20% inst) | 50 (88%) | 80 (87%) | 23 (88%) | 2 (100%) |
| γ ID | 47 (94%) | 78 (98%) | 22 (96%) | 2 |
| γ cut | 39 (83%) | 67 (86%) | 16 (73%) | 1 |
| μ found/ID/cut → final | 23 | 52 | 7 | 1 |
| eff (both-γ-majority-in-slice) | 0.115 | 0.260 | 0.040 | 0.006 |

Combined reading — **the γ-found collapse is dominated by SLICER
delivery; segmenter recognition is domain-flat once delivery is
controlled**:
- At the 0.5 bar (delivery comfortably above the instance requirement)
  the segmenter carves a ≥20% instance at 87–88% in EVERY cell,
  including overlay-v2 (2/2, thin stats but no counter-evidence).
- The strict both->50% bar also exposes the absolute slicer ceiling
  in-domain: val-old 30%, val-v2 47% of vtx-passing events — most
  selection passes in the plain cutflow ride on photons that are only
  partially in-slice (instances holding 20–50% of photon charge).
- Interpretation joins section 23: the periphery/label pathology and
  domain shift both act at the deghost+slice stage; the segmenter
  classifies and clusters what it is given, robustly. Retrain-campaign
  effort should weight the slicer (and its labels/cache) accordingly.

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

## 17. Domain-shift triangle: data / val-sim / overlay-sim (M1 verdict)

Design + execution log: `DOMAIN_SHIFT_MEASUREMENT_PLAN.md` (this dir).
CC1pi0 object-conditioned measurement: identical OLD-chain reco selection
on all three samples (val twin 78 evts, overlay pilot 70, bnb5e19 DATA
261 flash-cut chi2<1e4 candidates); objects = the two selected photons +
selected primary muon; per-object spacepoints from nu_reco part_inst_idx
-> kp2 particle groups; frozen P5B.3 (mix sim+data SSL) embeddings,
whole-event forward, lm>=0.15 parity on all legs.

| pair | class | pooled MMD2 (p) | /null95 | points MMD2 (p) | /null95 | lin AUC | kNN AUC |
|---|---|---|---|---|---|---|---|
| data-val | gamma | 0.08916 (0.002) | 19.2 | 0.03833 (0.002) | 1.6 | 0.988 | 0.895 |
| data-val | mu | 0.08985 (0.002) | 5.9 | 0.03875 (0.002) | 1.8 | 1.000 | 0.895 |
| data-ovl | gamma | 0.004256 (0.088) | 0.5 | 0.008941 (0.479) | 0.3 | 0.672 | 0.513 |
| data-ovl | mu | 0.002926 (0.248) | 0.2 | 0.006543 (0.752) | 0.3 | 0.627 | 0.474 |
| val-ovl | gamma | 0.07968 (0.002) | 9.9 | 0.03785 (0.002) | 1.3 | 0.977 | 0.908 |
| val-ovl | mu | 0.07621 (0.002) | 4.8 | 0.04138 (0.002) | 1.6 | 0.987 | 0.863 |

(pooled = per-object mean-pooled embedding, 2 gamma + 1 mu rows/event;
points = per-point embeddings, event-proportional subsample to 3000/side;
/null95 = MMD2 over the larger of the two samples' event split-half null
95% quantiles; p = event-block permutation; AUCs = GroupKFold C2ST on
pooled rows.)

**Verdict: DATA ~ OVERLAY — statistically indistinguishable** (below the
split-half null, kNN at chance) **while the new val-sim is far from
both** (x5-19 the null, linear AUC 0.99-1.00). The diffusion-suspicion
direction is inverted: the official overlay production models data well
in exactly the representation the chain heads consume; the new
corsika+nu sim carries the mismodeling. Background contamination in the
data leg could only inflate data-ovl, making the null result stronger.
Consequences:
- Route "replace overlay with new sim as expectation" is REJECTED.
- The v2-chain overlay/data regression = trained on val-sim features far
  from both data and overlay.
- Fix targets: retrain on overlay-domain (or corrected-sim) features;
  witness (S5) + waveform-width (S6) analyses identify the physical axes
  (charge scale / diffusion / response) to correct or augment over.

Scripts: `lartpc/larformer_analysis/domain_shift/` —
`build_object_manifest.py` (S1/S2), `extract_manifest_features.py` (S3),
`triangle_metrics.py` + `run_triangle.py` (S4). Outputs:
`NTUP/domain_manifests/` (manifest_*.npz, feats_*.npz,
results_triangle.json); stepA0 data leg: `NTUP/larmatch_data_pi0/`,
`NTUP/data_pi0_lmscored/`; logs `logs/pilot_matrix/{dom_*,triangle_*}`,
`logs/stepA0/data_arr.*`.

## 18. Witness forensics + cap-100k density probe (M2; old-cap hypothesis test)

**Witness correlations** (`domain_shift/witness_forensics.py`, job 2464878,
`NTUP/domain_manifests/witness_*.npz`; positive witness = first-named
sample). Top |Spearman| axes vs the val-sim direction:
- gamma: NN-spacing ("local_density" = mean 10-NN dist; sparser = val-like)
  rho -0.15 (data-val) / -0.21 (ovl-val); pixval_v +0.26/+0.19;
  q_cell -0.17/-0.10.
- mu: NN-spacing -0.21 (data-val) / **-0.38 (ovl-val, strongest single
  variable)**; pixval_v/pixval_y ~ +0.20/+0.21 (data-val); ovl-val
  x_drift/tick +0.22 (drift-dependent mismatch — diffusion-suggestive,
  S6 measures directly).
- data-ovl residual (below-null pair): density ~0 for gamma (+0.001);
  mild charge axes (pixval_y ~ +0.20, q_cell -0.21).
Direct density measurement (same 0.25 cm dedup): median 10-NN spacing
val 0.488 cm vs ovl 0.427 / data 0.441 — val-sim clouds ~10-14% sparser.

**Training-provenance confirmation at the old-slicer git hash b90a73b**
(old checkout): old slicer cap=100k with FRESH np.random.permutation per
__getitem__ (per-epoch density randomization; bit 66.7% of events);
lm settings 0.0 but training files = LANTERN v3 lists (already lm>=0.15
at file production, so composition identical to the new training — the
"no lm cut => ghost-dense" half of the hypothesis is moot). New slicer:
same lists, cap 300k. Old stage-3 cache (ptv3crosslevelslicer_iter_75750)
also built at cap 100k.

**Cap-100k eval probe** (job 2465705, `cap100k_density_probe.sh`;
sliceids_{val,ovl}_{oldchain,v2chain}_cap100k;
{val,ovl}_cap100k_decomp.txt). Raw in-slice fractions drop everywhere
(cap deletes charge: no-cell 0.20 val / 0.49 ovl). STAGE-SEPARATED:
- Slicer conditional in-slice (nu / (delivered - ghosted)):
  val: old 0.621 vs full-density 0.616; v2 0.622 vs 0.650.
  ovl: old 0.564 vs 0.567; v2 0.490 vs 0.501.
  => SLICERS ARE DENSITY-INSENSITIVE AT EVAL; the v2 slicer's domain gap
  (~0.63 -> ~0.50) is IDENTICAL at both densities.
- Deghoster ghosted-of-delivered RISES on sparsified input for both:
  val old 0.373 vs 0.237, v2 0.198 vs 0.140; ovl old 0.432 vs 0.247,
  v2 0.612 vs 0.417 (deghosters are the density-sensitive stage; full
  density at inference is right for everyone).

**Verdict on the old-cap hypothesis**: the EVAL-side version is refuted —
sparsifying to the old training density does not close (or move) the v2
slicer's domain gap, so the sim-discriminative features it exploits
survive 2x uniform dropout: per-point charge/response spectra (+ shape),
not absolute density texture. The TRAINING-side version (cap as
density-randomization regularizer) is only testable by a retrain
ablation, but its weight drops. Rising alternative for old-slicer
robustness: the old run was effectively UNDERFIT (flat 1e-5 LR, matching
agreement stuck ~0.1 per the m2f-recipe review) — robustness via limited
fit to sim microstructure; the m2f-recipe v2 run fits the (mismodeled)
new-sim charge texture much better and pays for it off-domain.
**Hardening priority reordered**: (1) charge/pixval-space augmentation
(per-plane scale + spectral jitter) and/or overlay-domain training files;
(2) density dropout still cheap/harmless to include; (3) S6 pulse-shape
measurement to identify the physical sim parameter (val muon witness
drift-dependence hints at diffusion).

## 19. M3 hit-width vs drift: effective longitudinal diffusion, three-way

Tool: `domain_shift/waveform_hit_width.py` (gaushit RMS at manifest/truth
muon points — dlmerged image2d are 6-tick compressed, cannot resolve
diffusion widths; gaushit is raw-tick, same hit finder in all three
productions; per-sample self-calibrated tick<->PeakTime, all legs land on
+~2397; |local dir_x|<0.25 PCA segment cut; mult==1, 5-95% amp window).
Fits: `domain_shift/hitwidth_fit.py` (binned-median RMS^2 vs x WLS,
sigma_t^2 = sigma_0^2 + 2 D_L x / v_d^3, event bootstrap). Jobs
2490926 + 2491619 (val-manifest on the 77 tier2-restored dlmerged);
outputs `NTUP/domain_manifests/hitwidth_*.npz` + `hitwidth_fits.json`.

| sample | Y-plane D_L eff (cm^2/s) | U | V | n_Y hits |
|---|---|---|---|---|
| val (truth, 1000 evts)     | 6.34 +- 0.03 | 7.09 | 6.73 | 158k |
| val (manifest, 78 evts)    | 6.27 +- 0.11 | 7.09 | 5.79 | 8.6k |
| ovl (manifest, 70 evts)    | 6.42 +- 0.18 | 7.60 | 7.98 | 8.8k |
| ovl (truth, 174 evts)      | 6.30 +- 0.08 | 6.64 | 6.71 | 31k |
| data (manifest, 261 evts)  | **3.67 +- 0.19** | 4.91 | 4.60 | 31k |

(sigma_0 ~ 1.4 us everywhere; induction planes read slightly higher D_L
in data — deconvolution/response effects; Y is the clean headline.)

**Verdicts:**
- **val == overlay in diffusion** (Y ratio 0.99-1.01; all planes within
  errors): both productions carry the SAME effective D_L ~ 6.3-6.4
  cm^2/s (the old LArSoft default). Diffusion is NOT the val-vs-overlay
  axis — the colleagues' "one sim has the bad diffusion parameter" story
  does not apply to this pair.
- **both sims over-diffuse vs data by ~1.7x** (data Y 3.67 +- 0.19,
  consistent with the published MicroBooNE measurement ~3.7). A real
  data-sim mismatch — but note the M1 triangle found data ~ overlay in
  P5B.3 embedding space, so the 0.25 cm dedup + pixval features largely
  integrate hit width out; diffusion is second-order for the chain.
- Selection-mode systematic bounded: truth vs manifest agree at 1.00-1.16
  (within errors) on both sims.

## 20. S7 synthesis: what actually needs fixing

Combining M1 (triangle), M2 (witness + cap-100k probe), M3 (widths):
1. **Overlay = data** in the deployment-relevant representation
   (MMD below split-half null, kNN at chance). Overlay stays as the
   analysis expectation. The new sim is the outlier (x5-19 null).
2. The val-sim outlier axes are **charge response/spectra** (per-plane
   pixval witness rho +0.19..+0.26; official-vs-user charge medians
   -10%, p95 -20..-36% from section 5/9) **and cloud sparsity**
   (val 10-NN spacing ~10-14% larger; mu ovl-val rho -0.38) — NOT
   diffusion (M3: val==ovl) and NOT density texture exploitable at eval
   (cap-100k probe: slicer domain gap identical at both densities).
   Sparsity origin: non-diffusion — prime suspect is hit/triplet
   formation (thresholds, multiplicity) and charge-response amplitude
   interacting with thresholds; the val sim also lacks the overlay's
   real-noise wires.
3. **Chain remedy (priority order)**: (a) train on overlay-domain files
   (truth exists) or add strong per-plane charge-scale/spectral
   augmentation; (b) keep density dropout as cheap insurance; (c) old
   chain's robustness likely included an underfit component (flat-LR
   run) — the v2 recipe's better optimization needs the domain handled
   by data/augmentation, not by hoping.
4. **Sim remedy**: to make the new sim usable as expectation, match its
   charge response to overlay/data (the pixval axes) and retune
   D_L -> measured ~3.7 for BOTH sims if pulse-shape realism matters
   downstream; validate against `hitwidth_fit.py` + the triangle
   battery (rerun costs one GPU-hour).
5. Standing infrastructure: manifests + frozen-encoder triangle +
   witness + width tools are now a reusable domain battery — run them
   as checkpoint gates for every retrain and for any new production.

## 21. Truth-parity density check (user-requested): the sparsity is real

Q: is the val cloud sparsity a selection artifact? Select points by
trackid (truth) + REAL larmatch NETWORK score on both legs (val files
are pre-cut at the 0.15 deploy floor — min lm exactly 0.150; overlay via
stepA0-attached scores; hasmatch NOT used, but trackid>0 => hasmatch==1
by construction, fractions match exactly). Network-missed (<0.15) true
points absent from val files: bounded by overlay where they are 0.2% of
true-particle points — cannot explain the deficit.
Tool: `domain_shift/truth_density_parity.py` (job 2492644,
`NTUP/domain_manifests/truth_density_parity.npz`).

median per-particle 10-NN spacing [median pts | pts per 1e3 dedup-q]:
| class | variant | val | ovl |
|---|---|---|---|
| gamma | dedup lm>=0.15 | 0.510 [636 | 124] | 0.436 [833 | 165] |
| mu    | dedup lm>=0.15 | 0.437 [2077 | 145] | 0.390 [2629 | 192] |
| gamma | raw   lm>=0.15 | 0.415 [872 | 166] | 0.387 [1016 | 198] |

**VERDICT: genuine sim property, stronger at truth parity than in the
manifest measurement — val has ~16% fewer raw triplets and ~25% fewer
0.25 cm dedup cells PER UNIT DEPOSITED CHARGE.** With diffusion measured
identical (section 19), the origin is 2D signal/hit formation: fewer
wire-tick pixels above threshold per deposit. Candidate sim parameters:
charge amplitude/gain (compounds with lower pixval medians), zero-
suppression/sparsification thresholds, electron lifetime/attenuation.
Unifies the witness axes: one charge-response deficit both shifts pixval
spectra and thins the triplet cloud via thresholds.

## 22. Label-definition audit: section 21's interpretation CORRECTED

User hypothesis (2026-08-17): the density deficit is the TRUTH-LABEL
definition, not detector response — overlay truth = 2D projection match
(inclusive along deposits); new-sim truth = 3D proximity to edep segment
CENTROIDS with a max radius (misses segment ends). Test:
`domain_shift/label_definition_audit.py` (job 2493137,
`NTUP/domain_manifests/label_definition_audit.npz`): per photon, local
cloud = ALL lm>=0.15 triplets within 0.75 cm of labeled points.

| median over photons | val | ovl |
|---|---|---|
| local-cloud NN10 (label-free) | 0.358 | 0.360 |
| labeled-only NN10 | 0.415 | 0.389 |
| labeled fraction of local cloud | 0.547 | 0.702 |
| dedup-q labeled/unlabeled | 5.03/3.51 | 3.99/3.08 |
| pixval_V labeled/unlabeled | 43.0/36.3 | 41.5/34.9 |

**CONFIRMED.** (1) Label-free clouds are IDENTICAL in density (0.6%) —
detector response / triplet formation exonerated; section 21's "2D hit
formation deficit" interpretation is WITHDRAWN. (2) Val under-labels by
~15 points of labeled-fraction with a charge-core bias (centroid
signature). (3) The truth-conditioned pixval shift largely dissolves
label-free (U 36.4/36.7, V 36.3/34.9, Y 37.5/40.7).

Consequences:
- SIM FIX = the label maker: use full segment extent (sample along
  segments or apply the overlay-style 2D projection), not centroid+
  radius. No charge-response change indicated.
- TRAINING implication (CORRECTED 2026-08-17, per user): the simch-label
  mechanism is COMMON to every sim-trained stage of BOTH chains — the
  old LoRA trained on prod4 simch/hasmatch labels, the old slicer on the
  same LANTERN labels as v2. So the periphery bias ("deposit periphery
  != particle/real") is a SHARED label pathology, not an old-vs-v2
  differentiator; what differs between chains is how sharply each model
  ENFORCES it (capacity/optimization — big heads fit the thin labels
  faithfully, LoRAs underfit them). On OVERLAY evaluation the truth
  mechanism switches to the more inclusive 2D-image labels, so
  simch-trained models are charged for killing periphery their training
  truth never credited — part of the measured overlay keep gap is this
  train/eval label-mechanism mismatch, largest for high-capacity heads.
  Relabel (label completion) + retrain remains a first-class remedy for
  ALL stages, and it also removes the train/eval mismatch.
- Embedding triangle reinterpretation: with nu-deposit clouds now shown
  near-identical, the val outlier (linAUC ~0.99) most plausibly
  reflects EVENT CONTEXT — simulated corsika cosmics + sim noise vs
  real cosmics/noise in overlay+data (consistent with data ~ ovl).
  Testable: rerun the MMD battery on cosmic-region points, or on
  context-free local crops around nu objects.

## 23. Label pipelines traced to code + offset signature; recommended fix

Code trace (with user): overlay conversion runs `--mcc9` ->
`process_mcc9_sim()` -> truth from official 2D instance/ancestor/segment
images (`process_truth_labels` + `TripletTruthFixer`); official overlay
dlmerged has NO simch product. New sim runs the simch path
(`MCPixelLabelMaker::make_truthlabels_fromsimch`): each wirecell IDE
(drifted depo centroid; ~1.4 IDEs per (ch,tdc), median 0.01 MeV) stamps
ONE (u,v,y,row) cell, bleed dwire=1 / drow=0 — set identically in both
converters. New-sim dlmerged lacks instance/ancestor images (has
segment + simch), so the unified mcc9 path cannot run on it as-is.
Detsim configs (`uboonecode/wcls.fcl`): new sim = plain `driftsim`;
official overlay = `driftsim_overlay` (adds YZ per-plane charge corr +
database lifetime); identical diffusion 6.4 (matches measured 6.3).

Offset signature (`domain_shift/label_offset_signature.py`, job 2512114):
unlabeled-local points sit in a THIN SHELL around labels in both sims
(95% within drow<=2 & dwire<=2; shape similar val vs ovl) — val misses
more of the shell (45% of local cloud unlabeled vs ovl 30%), spread over
same-row wire-combo misses and +-1-row misses, not one clean axis.

**RECOMMENDED FIX: spacepoint-level label completion at conversion** —
after any base labeling, attach unlabeled lm>=0.15 triplets within
~0.75 cm (3D) of a particle's labeled points to that particle. Applies
identically to BOTH productions (production-independent truth), lifts
overlay's 70% coverage too, and removes the "periphery != particle"
training pathology. Bleed expansion (drow 1-2 / dwire 2) is the blunter
single-knob alternative. Either way: relabel -> rebuild caches ->
retrain with the overlay+data battery as gates.

## 24. Data-isolation cell verdict (job 2450950, ep25): preparation exonerated

Pre-registered cell (`stage1_deghost/lorafinetune-sonata-v6noghosts-
lantern-deghost.py`): deployed-LoRA recipe (v6-noghosts encoder frozen +
LoRA) with ONLY the data swapped prod4-uncut -> LANTERN lm>=0.15
pre-cut. Same simch label mechanism both (user correction: labels are
common to old and new sim trainings; only the OVERLAY derives truth by
the 2D-image mechanism). Battery: preal legs job 2512787, matched-tau*
job 2512788 (`matched_v6lantern.*.log`).

| ga=0.20 | tau* | keep_in | keep_OVL | transfer |
|---|---|---|---|---|
| v7-lora (deployed) | 0.373 | 0.816 | 0.792 | 0.970 |
| v6-lantern ep25 | 0.348 | **0.863** | **0.786** | 0.911 |
| p5b3-lora | 0.420 | 0.846 | 0.627 | 0.741 |
| ft/crop/p5b3 decoders | — | 0.85-0.91 | 0.62-0.72 | 0.53-0.79 |

Verdicts:
- **Pre-cut preparation largely exonerated for the deghoster**: overlay
  keep matches the deployed LoRA (0.786 vs 0.792); recipe/capacity is
  the robustness carrier, per the pre-registration's second branch.
- Second-order margin erosion is real: transfer 0.91/0.88/0.84 at
  ga 0.20/0.15/0.10 vs deployed 0.97/0.99/1.02 — grows at tight
  operating points, the hard-negative-margin signature. Remedy if
  wanted: randomized lm threshold in training (dataset supports
  larmatch_threshold_range).
- **Sharpest encoder result of the campaign**: same LoRA recipe + same
  LANTERN data, v6-noghosts encoder 0.911 transfer vs P5B.3 mix 0.741 —
  the sim-only v6-noghosts SSL features carry the robustness; the
  sim+data-mix encoder is ACTIVELY worse, not merely equal (sharpens
  section 7).
- v6-lantern beats the deployed LoRA in-domain at every matched point
  with equal overlay keep — best deghoster cell measured; candidate
  deployable upgrade pending fuller eval (val mIoU battery + later
  epochs; training still running at ep32+).
- Campaign consequence: deghoster lane settled (v6-noghosts + LoRA,
  prep as-is +- threshold randomization); domain-hardening effort goes
  to the SLICER (cosmic-context + label completion + recipe-preserving
  regularization; see the mechanism discussion preceding this section).

## 25. Early-checkpoint slicer probe + v6-lantern cascade rows (job 2513133)

Standard sliceids footing; v6-lantern deghoster (ep25, tau0.2) upstream;
m2frecipe-v2 slicer at ep4 (deployed) vs ep1 (underfit; OneCycle still
ramping). Outputs: sliceids_{val,ovl}_v6lantern{,_ep1slicer},
v6lantern_slicerprobe_{val,ovl}.txt.

| in-slice photon charge | val | ovl | ratio |
|---|---|---|---|
| ep4 | 0.558 | 0.383 | 0.69 |
| ep1 | 0.509 | 0.403 | 0.79 |
| (old slicer + LoRA@0.2 ref) | 0.470 | 0.467 | ~0.99 |

- Fit-transfer trade CONFIRMED in direction, MODEST in size: epochs 2-4
  bought +0.049 val, cost -0.020 ovl. Under-optimization is NOT the main
  source of the old slicer's transfer (ep1 0.403 << old 0.467) —
  remaining candidates: training-time density randomization (100k cap)
  and/or recipe elements (128 vs 48 queries, per-level matching, DN).
  SLICER_RETRAIN_PLAN: C2 demoted to minor; S3 (density dropout) stock
  up; recipe-element ablation added as optional cell.
- Redundancy prediction verified exactly: v6-lantern + ep4 overlay 0.383
  == hybrid (LoRA + ep4) 0.383 — no deghoster windfall, per matched keep.
- v6-lantern deghoster cascade rows: ghosted 0.116 val (best measured;
  ft 0.140, LoRA 0.237) and 0.139 ovl (== LoRA) — best on BOTH domains
  simultaneously; deployable-upgrade case firmed.

## 26. S1 retrain vs the reference chains — full cutflow tables

Cascade for the S1 columns: v6-lantern deghoster @tau0.2 + S1 slicer
(mix-enriched + completed labels + masked-no-object) + **unchanged v2
m2f stage-3** + old attempt-2 keypoint. All columns share the same thin
(unexpanded) truth ruler and the same gslice/gfound bar (0.2).

### Truth-matched (decomposed) cutflow — OVERLAY, 174 CC1pi0 signal

| step (survival vs prev) | old | v2 | S1 ep1 | S1 ep2 |
|---|---|---|---|---|
| reco'd | 169 | 173 | 173 | 173 |
| vtx in FV | 167 | 171 | 168 | 171 |
| γ deghost (>20% survives) | 162 (97%) | 146 (85%) | 168 (100%) | 171 (100%) |
| γ slice (>20% in nu slice) | 133 (82%) | 100 (68%) | 160 (95%) | **166 (97%)** |
| γ found (≥20% instance) | 100 (75%) | 53 (53%) | 128 (80%) | **133 (80%)** |
| γ ID | 98 (98%) | 52 (98%) | 126 (98%) | 127 (95%) |
| γ cut (≥2 conf attached) | 78 (80%) | 46 (88%) | 99 (79%) | 90 (71%) |
| μ found | 68 (87%) | 42 (91%) | 95 (96%) | 86 (96%) |
| **μ ID** | 67 (**99%**) | 37 (88%) | 71 (**75%**) | 71 (**83%**) |
| μ cut | 52 (78%) | 27 (73%) | 56 (79%) | 55 (77%) |
| + cπ veto + mγγ | 49 | 27 | 54 | 54 |
| **eff (truth-matched)** | **0.282** | **0.155** | **0.310** | **0.310** |

### Truth-matched cutflow — VAL twin, 200 CC1pi0 signal

| step | old | v2 | S1 ep1 | S1 ep2 |
|---|---|---|---|---|
| vtx in FV | 193 | 197 | 197 | 198 |
| γ deghost | 189 (98%) | 197 (100%) | 197 (100%) | 198 (100%) |
| γ slice | 148 (78%) | 179 (91%) | 179 (91%) | 189 (95%) |
| γ found | 122 (82%) | 142 (79%) | 146 (82%) | 155 (82%) |
| γ ID | 118 (97%) | 140 (99%) | 141 (97%) | 152 (98%) |
| γ cut | 96 (81%) | 117 (84%) | 117 (83%) | 129 (85%) |
| μ found | 83 (86%) | 114 (97%) | 113 (97%) | 124 (96%) |
| μ ID | 78 (94%) | 112 (98%) | 111 (98%) | 121 (98%) |
| μ cut | 62 (79%) | 94 (84%) | 95 (86%) | 103 (85%) |
| + cπ + mγγ | 60 | 88 | 90 | 100 |
| **eff** | **0.300** | **0.440** | **0.450** | **0.500** |

### Selection (light) cutflow — pre-flash efficiency

| cell | reco'd | vtxFV | ≥2γ | +μ | +0cπ | +mγγ | eff |
|---|---|---|---|---|---|---|---|
| VAL old | 196 | 193 | 118 | 81 | 78 | 78 | 0.398 |
| VAL v2 | 199 | 197 | 129 | 104 | 98 | 98 | 0.490 |
| VAL S1 ep1 | 199 | 197 | 126 | 103 | 97 | 97 | 0.487 |
| VAL S1 ep2 | 200 | 198 | 134 | 107 | 104 | 104 | **0.520** |
| OVL old | 169 | 167 | 101 | 73 | 70 | 70 | **0.402** |
| OVL v2 | 173 | 171 | 89 | 58 | 56 | 56 | 0.322 |
| OVL S1 ep1 | 173 | 168 | 107 | 60 | 58 | 58 | 0.333 |
| OVL S1 ep2 | 173 | 171 | 98 | 59 | 57 | 57 | 0.328 |

Readings:
- **Photon lane fixed on overlay**: γ-slice 68% (v2) -> 97% (S1 ep2),
  γ-found 53% -> 80% — both best measured, exceeding even the old chain
  (82% / 75%). Truth-matched eff doubles v2 and passes deployed-old.
- **Domain gap narrowed**: overlay/val truth-matched ratio 0.35 (v2) ->
  0.62 (S1 ep2); old chain was 0.94 but at a much lower ceiling.
- **The bottleneck moved to STAGE-3 (unchanged in these runs)**:
  μ-ID 98% on VAL vs 83% on OVERLAY with the SAME slicer and segmenter
  => domain-specific, not an artifact of larger slices. The v2 stage-3
  was already 88% overlay (old stage-3: 99%).
- ep1->ep2 on overlay: delivery improved (γ-slice 95->97%, γ-found
  128->133) but γ-cut fell 79->71% and γ-ID 98->95%, so eff stayed
  0.310 — losses migrated downstream into stage-3 / attachment.
- SELECTION eff on overlay (0.328) still trails old (0.402) because the
  plain selection can pass with wrong objects; the truth-matched metric
  (which does not credit those) is where S1 wins.
