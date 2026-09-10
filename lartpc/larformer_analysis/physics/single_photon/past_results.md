# Past results: first-pass single-photon selection (v1 chain, June-July 2026)

Condensed from the original README (full text preserved in
[`archive/v1_stage3pred_workflow/README_v1.md`](archive/v1_stage3pred_workflow/README_v1.md)).
The scripts that produced these numbers consumed the *old* reco outputs
(`merged_sp` + `stage3pred` H5 from the stage-3 cascade, checkpoints
`model_iter_182304` / `model_iter_98652`) and the legacy `dlgen2_reco_v2me05`
gen2 ntuple. None of them run on the current v2_s1ep2p8 chain, whose analysis
inputs are the LArFormer gen2ntuple ROOT files (see README "Scripts"). They are
kept in `archive/v1_stage3pred_workflow/` for reference only.

## What was learned (still relevant)

- **Signal is diffuse in the sample.** Any-photon truth cut keeps ~90% of files
  (neutron-capture gammas); only the visible-energy cut isolates signal. On the
  1500-file (~10%, 67,211-event) run-3b BNB nu overlay sample:
  6,743 events with >=1 visible photon, of which **829 (12.3%) were 1g+0X**
  (visibility = nSP_trunk>=80 proxy; X thresholds p>50, pi/K/mu>30, e>20 MeV KE).
- **Photon finding (per-photon IoU match) was good:** efficiency 0.72, purity
  0.83 on 5,441 detectable photons; energy turn-on 0.16 (0-20 MeV) -> 0.65
  (150-400 MeV); falls with vertex distance 0.82 (0-5 cm) -> 0.44 (120+ cm).
- **The 1g+0X EVENT selection was poor: efficiency 0.15-0.17, purity 0.41-0.44**
  (prior LANTERN reference ~0.10 / ~0.40). Deghost-threshold sweeps did not help;
  flash rescue (K=1) gave +12% relative efficiency at similar purity.
- **Loss budget (where 1g0X photons went):** the slicer put only 49% of them in a
  nu-slice; 34% got their OWN slice mislabeled cosmic; 16% merged into cosmic
  slices; 2% lost. FN breakdown: 71% no photon query (slicer drop), 15% photon
  split, 14% false extra particle. This diagnosis (with the pi0 photon-loss study)
  launched the 2026-08 slicer/segmenter retraining that produced the v2 chain.
- **Flash recovery prototype:** for own-cosmic-slice photons, PhotonLib
  prediction + Neyman chi2 to the in-time flash ranked the photon slice first in
  32% of cases (top-3: 56%).
- **Reproducibility bugs found and fixed:** `shuffle_orders=True` on the deghoster
  PTv3 and the un-gated cross-level token subsample made per-event output depend
  on the run membership; now bit-exact with `--deterministic`
  (`docs/reference/LArFormer_Reproducibility.md`).

## Numbers table (v1 chain, 3000-event capped sample, deterministic rerun)

| Arm | deghost tau | flash rescue | efficiency | purity |
|---|---|---|---|---|
| base | 0.5 | no | 0.156 | 0.437 |
| dg0p4 | 0.4 | no | 0.151 | 0.442 |
| dg0p3 | 0.3 | no | 0.148 | 0.412 |
| rescue | 0.5 | K=1 | 0.167 | 0.420 |
| full 829 truth-1g0X: base / rescue | | | 0.151 / 0.169 | 0.424 / 0.406 |

Benchmarks quoted then: MicroBooNE WireCell inclusive 1g: 7% eff / 40% purity;
SBND SPINE: 37% / 56%.
