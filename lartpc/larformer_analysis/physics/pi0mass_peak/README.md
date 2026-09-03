The goal for this study is to select two photon events and create a pi0 mass peak.

The signal events we want to isolate are neutrino interactions that satisfy the following:
- the interaction vertex occurs within the WireCell FV inside the TPC,
- the interaction produces exactly one primary final state pi0 that decays into two photons that are detectable (see below for definition of detectable)

We are interested in tagging the signal as either charged-current or neutral current.
We will want to split the sample into these two categories.

The photon detectability is defined as using the true visible energy of the photon to be above 20 MeV. We use the true visible energy definition used in the evaluation script in lartpc/larformer_reco/export/compare_to_legacy_ntuple.py.

As for the selection criteria using the reconstructed variables, we select events that:
- has two reconstructed photons above 20 MeV,
- has a vertex inside the WireCell FV.
Like with the signal definition, we split the sample into charged-current and neutral current. This is defined by finding at least one muon created at the vertex.

I would like a plot of the invariant mass of the two photons for selected events, divided into charged-current and neutral current events.  Within the CC or NC plots, for events selected from the simulation sample, we tag events as either CC and NC and whether they fall into the signal or background categories. 

We can use the invariant mass formula: $m_{\gamma\gamma} = \sqrt{2 E_1 E_2 (1 - \cos \theta_{12})}$, where $E_1$ and $E_2$ are the energies of the two photons and $\theta_{12}$ is the angle between them. 

We start with the simulated sample processed to evaluate the reconstruction performance.
We use the output of the export ntuple script in lartpc/larformer_reco/export/export_gen2ntuple.py.

The current ntuple is at lartpc/larformer_reco/output/mcc9_bnbnu_overlay_1500_full/dlgen2_larformer_ntuple_mcc9_bnbnu_overlay_1500_full_67k_pre_llr_attach.root.

To start, use only the neutrino slice stream. Use only confidently attached photons as for determining if the event passes the selection criteria. 

Besides the invariant mass plot, I would also be interested efficiency of finding events as well as a function of the total true visible energy of the two photons.

## Secondary SBND-SPINE Comparison

SBND has a CC 1pi0 selection using the SPINE reconstruction to compare to.
While not on MicroBooNE, this is closer to what I would think is the state-of-the-art 
reconstruction for this channel in current LArTPC experiments.

The signal definition is:

- Flash Matched
    - Interaction is matched to ‘valid_flashmatch’ variable
- Fiducial Volume
    - Require interaction vertex to be at least 20 cm from
    - X,Y detector boundaries and 10 [50] cm from
    - upstream [downstream] Z detector boundaries
- Topology
    - 1 primary muon: muon kinetic energy > 143.425 MeV (50 cm long)
    - 1 primary neutral pion (2 primary photons)
        - photon kinetic energy > 20 MeV
        - diphoton mass < 400 MeV/c2
    - 0 charged pions
        - pion kinetic energy > 25 MeV
    - Inclusive to all other particles

From doc-db: https://sbn-docdb.fnal.gov/cgi-bin/sso/RetrieveFile?docid=47857&filename=June%2026%20SBND%20Collaboration.pdf

After their selection, they report:
  - 86% purity
  - 65% efficiency

---

# Processing chain & scripts (documented 2026-08-28)

Two selections live in this folder:
1. **MicroBooNE-style CC/NC pi0** (the primary analysis above; working point
   frozen behind `plots_ext_cut1e4_satfix/` — do NOT modify those scripts),
2. **SBND-SPINE-style CC 1pi0** (`sbnd_cc1pi0.py`, self-contained benchmark
   vs sbn-docdb 47857; plots in `plots_sbnd_cc1pi0/`).

## Upstream production chain (per sample)

All analysis scripts consume **gen2ntuple ROOT files** produced by:

```
merged_sp (stepA)                                        [per-event H5 + truth for MC]
  -> kp2 cascade inference, WITH flash-match streams     [GPU; slurm/submit_inference_shard.sh]
       tools/larformer/run_larformer_keypoint2_cascade_inference.py
       flags: --output-tree (NO --no-flash!); satfix knobs default "auto":
       --dead-opdets/--gamma-run-scale/--mask-saturated  (run1 data: dead='',
       gamma 0.80; run3 MC-overlay/EXT: dead=15, gamma 1.0 — auto-detected)
       => keypoint2_streams/ with nu (event*_0.h5) + fm (event*_fm_0.h5) files
  -> regen: split streams into nu / fm kp2 lists
  -> run_nu_reco.py per stream                           [CPU; submit_nu_reco_shard.sh]
       LLR shower attachment (see trajfit/shower_attach_llr.py; OP history below)
  -> larpid per stream                                   [CPU; submit_larpid_shard.sh]
       SAMPLE_TAG picks checkpoint: 'run3' in tag -> alternate weights (MC),
       else default (bnb5e19 data, EXT)
  -> export_gen2ntuple.py                                [submit_export_shard.sh]
       MC: TRUTH_DIR=<truth_sidecar> (truth branches + potTree + xsecWeight);
       data/EXT: TRUTH_DIR absent -> data mode
  -> hadd                                                [submit_export_merge.sh]
```

One-command orchestrator for the whole chain (any sample; set TRUTH_DIR for
MC): `lartpc/larformer_reco/slurm/submit_extbnb_chain.sh`
(TAG=, DATADIR=, NINF=, NNR=, NEXP=, NU_RECO_EXTRA_ARGS=, TRUTH_DIR=,
LARPID_SAMPLE_TAG=, MSP_LIST_SRC= for subset lists).

## The three samples (July-2026 old-chain "satfix" campaign)

| sample | events | ntuple / dir | notes |
|---|---|---|---|
| MC nu-overlay | 67,211 | `larformer_reco/output/mcc9_bnbnu_overlay_1500_full_satfix/` | run3; truth_sidecar/ REUSABLE (truth is chain-independent) |
| bnb5e19 beam data | 176,336 | `/cluster/tufts/wongjiradlab/larbys/data/larformer/mcc9_v28_wctagger_bnb5e19/` | run1; POT 4.4e19 |
| EXT-BNB (beam-off) | 668,388 | `/cluster/tufts/wongjiradlabnu/nutufts/data/larformer/mcc9_v29e_dl_run3_G1_extbnb_full/` | run3; spill scale **0.17682554549** (full sample; divide by processed fraction for subsets) |

## Analysis scripts

Table builders / primary plots:
- `pi0_mass_analysis.py` — signal/background truth categories + reco selection from an
  ntuple; writes the per-event selection table (`--out *.npz`) and plots.
  **Working point (July 2026):** `--mu-ke-min 50 --flashchi2-cut 1e4
  --flashchi2-cut-nc 1778 --cpi-ke-min 60` (+ `--data` for beam data).
  Tables: `mc_full_satfix_table.npz`, `data_satfix_table.npz`,
  `ext_satfix_table.npz` (row-aligned to ntuples; carry w + flash_chi2).
- `datamc_pi0_overlay.py` — data points over truth-category-stacked MC
  (m_gg, p_pi0) from two tables.
- `datamc_ext_overlay.py` — 3-component overlay: MC (xsecWeight to 4.4e19)
  + EXT cosmic (spill-scaled) vs beam data, from the three tables.
- `flashchi2_ncpi0.py` / `flashchi2_from_tables.py` — flash-chi2 shapes and
  cut optimization (from ntuples+cascades / from the pre-built tables).

Flash/saturation studies (analysis-level previews of the cascade satfix):
- `flash_correction.py` (dead-PMT mask), `saturation_mask_test.py`
  (dead+saturation cap), `pi0_compare_masks.py` (physics A/B of the two
  masks). All recompute chi2 from per-PMT arrays in keypoint2_streams
  WITHOUT a GPU re-run; the real fix (slice re-ranking) needs the cascade.
- `vtx_datamc_pi0.py` — vtx position/dwall data-vs-MC (cosmic spatial check).
- `datamc_distributions.py` — area-normalized reco-shape checkup.

SBND-style benchmark:
- `sbnd_cc1pi0.py` — tight-FV CC 1pi0 (mu KE>143.425, 0 cpi KE>25, 2
  confident photons E>20, m_gg<400) + flash cut. July result: purity 0.69 /
  eff 0.39 (SBND-SPINE ref: 0.86 / 0.65).
- `sbnd_photon_loss.py` — photon-loss stage breakdown from
  `eval/eval_reco_performance.py --out` records
  (`sbnd_photon_records.npz`). July: 25.4% photons lost, 72% of loss from
  missSlice+noInst — THE diagnosis that launched the 2026-08 slicer/
  segmenter retraining campaign.

`rse_*_keypoint2_streams.npz` — cached (run,subrun,event)->cascade-file maps
(entry-index != cascade-index on the beam sample after veto5M surgery).

## Attachment operating-point history
- pre-2026-07-11: hard cuts (impact<=10, cos>=0.9, gap<=60).
- 2026-07-11 (the July ntuples): LLR union rule, old-chain tables
  (`trajfit/data/attachment_llr_tables.npz`), thr +5.0.
- 2026-08-28 (v2 chain): refit tables
  (`trajfit/data/attachment_llr_tables_s1ep2p8.npz`), recommended thr +4.0
  (see SLICER_RETRAIN_PLAN.md threshold-curve entry).

## Reproducing with the v2 chain (s1ep2p8, 2026-08)

Frozen chain: v6-lantern LoRA deghoster ep25 + S1 mix-enriched slicer ep2 +
s1cache-m2frecipe segmenter ep8 + old attempt-2 keypoint. Cascade config
`configs/lartpc/larformer/stage4_keypoint/larformer-keypoint2-fullcascade-v6lantern-envslicer.py`
with env overrides:
```
LARFORMER_BATTERY_SLICER_CKPT=exp/larformer_slicer_s1_mixenriched_v1/model/epoch_2.pth
LARFORMER_KP_PARTICLE_CKPT=exp/larformer_particle_s1cache_m2frecipe/model/epoch_8.pth
```
nu_reco attachment: `NU_RECO_EXTRA_ARGS="--attach-llr-tables
lartpc/larformer_reco/trajfit/data/attachment_llr_tables_s1ep2p8.npz
--attach-llr-thr 4.0"`.
NOTE: the 2026-08 no-flash kp2 productions (overlay 67k, EXT 200k) canNOT
feed this analysis — the flash-chi2 cuts need the flash-matched streams, so
inference must be re-run WITHOUT --no-flash. MC truth_sidecar and all
analysis scripts are reused as-is; only the ntuples are regenerated.

## Official additions (2026-08-30)

- **Photon energy calibration (v2 chain)**: `trajfit/data/calo_calib.npz` =
  gamma E = 0.01556*Q − 11.47 (e unchanged). Derived by the per-E-bin
  median-charge profile vs ACTUAL photon energy (|p_true|), 8,286
  truth-matched photons, new-chain MC; validation peak 132.3 MeV. Old
  constants: `calo_calib_oldchain_backup.npz`. For already-produced
  ntuples use `pi0_mass_analysis.py --recal-gamma-a 0.01556
  --recal-gamma-b -11.47`.
- **EXT-rejection BDT (official cut)**: flash-blind topology BDT
  (`ext_bdt.py`; features dist1/dist2/vtxScore/E1/...; logchi2+flashPE
  excluded for data/MC robustness — flash-shape variants showed d/p
  instability). Model `ext_bdt_model_flashblind.joblib`; official WP
  score >= 0.280 (97% signal eff, 86% EXT rejection). Trained on
  even-event signal-MC/EXT halves; plots must use odd halves (w x2) —
  `ext_bdt.py --holdout-plots`. Near-peak: EXT 103->33, signal fraction
  0.74->0.82, data/pred 0.95.
- CAVEAT (documented 2026-08-29): `A_GAMMA x trueSimPartPixelSumQ`
  "visible energy" is ~2.2x the ACTUAL photon energy on both chains
  (fully-true m_gg reconstructs at ~300 MeV) — the 20-MeV detectability
  threshold is ~9 MeV actual. Longstanding convention; pending explicit
  redefinition decision.
