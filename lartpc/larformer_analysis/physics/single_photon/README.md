# Single Photon Selection

The goal of this analysis is to look at the performance of the LArFormer Full
Cascade model on single photon events. Specifically, we want to estimate the selection performance
of the MicroBooNE inclusive single photon event which is looking for an excess of single photon
MicroBooNE events that would be consistent with the MiniBooNE single-shower excess.

The signal definition considers the following:

 - We want to isolate events with exactly 1 visible photon in the final state. Visibility is determined by the true DEPOSITED energy in MeV, calculated by scaling the true "deduplicated charge" (`trueSimPartPixelSumQ`): E_dep = a × PixelSumQ with a_gamma = 0.0110 and a_e = 0.0101 MeV/ADC (`lartpc/larformer_analysis/truth_calib.py`, measured 2026-09-27 on well-contained true particles; reco showerRecoE / E_dep median 1.04, so truth and reco are on one scale). A visible shower has E_dep >= 20 MeV, for photons and electrons alike. (Until 2026-09-27 the legacy constant 0.0253017, a July reco fit, was used; it overstated deposited energy x2.3 so "20 MeV" meant ~9 MeV deposited while the reco cut was 20 MeV. `--truth-scale legacy` reproduces those results.)
 - MiniBooNE was a Ring Imagine Ring Cherenkov detector and so it was blind for particles below Cherenkov threshold. 
   To mimic selections in MicroBooNE, visibility for non-photon particles is based on the Cherenkov threshold KE. We follow the thresholds used in the MicroBooNE inclusive selection (https://arxiv.org/pdf/2502.06064). Muons are not visible is below its KE<100 MeV. Charged pions and protons are always considered as not visible by MiniBooNE. This is not a great assumption for pions -- but their momentum is hard to estimate accurately due to their propensity to reinteract or decay in flight.
 - These visibility thresholds apply to both primary and secondary final state particles. (Primaries are those particles produced by the initial nu interaction+FSI. Because neutral pions produced by nu interactions decay effectively immediately, their daugher photons are also considered primary.)

The signal nu interaction criteria becomes

- 1 visible photon and no visible non-photon final state particles

Our target is to isolate this somewhat inclusive definition. But we do want to split the result into truly single photons and the rest.
This is because the MicroBooNE inclusive selection found a local 2-sigma excess in this sample. As a result, we are interested in
how well the new reco chain, based on the LArFormer models can select such events.

We also want to split out the single photon events where no other primary particles are visible from the point of view of MicroBooNE's detector. This means applying lower thresholds.
 - muons+pions+other mesons: < 30 MeV
 - protons and other baryons: < 60 MeV
 - electrons+photons: the same 20 MeV visible threshold
We then want plots for both the single photon inclusive sample, the single photon only (1g0X) sample, and the rest.

The figure of merits for the selection is efficiency and purity for each of these individual samples.
Benchmarks include:
 - MicroBooNE's inclusive selection using WireCell: 7% efficiency and 40% purity
 - SBND inclusive selection using SPINE: 37% efficienct and 56% purity

The way we intend to find these events:

1. neutrino interactions with reco vertex position post-space charge correction that is further than 10 cm from the TPC boundary. The boundary is [0,256] for x, [-117,117] for y, and [0,1036] for z, all in cm.
2. the neutrino interaction must have one photon passing the single photon BDT score, threshold score value to be determined. All other particles must pass the other threshold KE thresholds.
3. The interaction must have a flashchi2 value below some threshold to be determined, which cuts on interactions that match the scintillation signal in time with the beam.

We want to define a selection based on the LArFormer outputs and estimate the true positive, false positive, and false negative rates.



## References

- LArFormer model information: [`docs/LArFormer.md`](../../../docs/LArFormer.md)
- MicroBooNE data set information: [`docs/reference/MicroBooNE_Datasets_on_Tufts.md`](../../../docs/reference/MicroBooNE_Datasets_on_Tufts.md)
- Flat-ntuple parsing spec: [`docs/reference/Gen2_Flat_Ntuple_Spec.md`](../../../docs/reference/Gen2_Flat_Ntuple_Spec.md)
- Cluster job submission: [`docs/reference/Tufts_SLURM_Job_Guide.md`](../../../docs/reference/Tufts_SLURM_Job_Guide.md)
- LArFormer cascade data pipeline: [`larformer_scripts/LARFORMER_DATAPREP.md`](../../larformer_scripts/LARFORMER_DATAPREP.md)
- Current version of model checkpoints and datasets: [`lartpc/larformer_analysis/model_and_output_file_versions.md`](../../model_and_output_file_versions.md)
- Pass Results from a first pass attempt selection which helped develop the reco: [`past_results.md`](past_results.md)

## Steps

1. Define the signal definition and determine the number of events in the dataset that satisfy this definition.
2. Define additional side band regions to estimate contributions from background events.
3. Isolate the files and events in the dataset that contain signal events.
4. Process the signal and side band events through the LArFormer Full Cascade model.
5. Estimate the efficiency (true positive) selection rate and the background contamination in the signal region.

---

## Why single photons are hard to define

Photons mostly come from π⁰s (from the ν interaction or secondary hadronic
interactions) and become *single-photon* topologies when only one photon is
"visible." We define visible as **≥20 MeV of ionization deposited in a single
cluster**. That quantity is not in the flat ntuples, so the study runs in two
passes over two data tiers (see [`MicroBooNE_Datasets_on_Tufts.md`](../../../docs/reference/MicroBooNE_Datasets_on_Tufts.md)):


---

## Scripts (v2_s1ep2p8 chain, ntuple-only; 2026-09-05)

The analysis now runs entirely on the LArFormer gen2ntuple ROOT files listed in
[`model_and_output_file_versions.md`](../../model_and_output_file_versions.md)
(MC overlay 67k + EXT-BNB 200k; the flash chi2 is the `recoVtxFlashChi2` of the
nu-stream vertex, which equals the masked chi2 used by the pi0 study). The earlier
stage3pred/merged_sp workflow and its scripts are archived under
[`archive/v1_stage3pred_workflow/`](archive/v1_stage3pred_workflow/) — they do not
run on the current chain; their findings are summarised in
[`past_results.md`](past_results.md).

| Script | Purpose |
|---|---|
| `single_photon_selection.py` | truth signal definition (inclusive 1g+X and strict 1g0X), cumulative reco cut steps S1-S5, stacked candidate-energy plots, N-1 flash chi2, efficiency vs true photon energy and purity vs reco photon energy per step, BDT / chi2 threshold scans, `cutflow.txt`, row-aligned tables in `workdir/` |
| `photon_bdt_study.py` | per-shower cosmic-BDT (`showerCosmicScore`) performance vs threshold, split by the event's number of true detectable photons (signal 1g / all 1g / 2g / >=3g), EXT and overlay-cosmic rejection, per-event effect, ROC, pass fraction vs reco E and vs distance to vertex; `bdt_table.txt` |
| `shower_novtx_bdt.py` | vertex-FREE per-shower cosmic BDT (features that exist for `showerVtxIdx=-1` prongs: E, angles, start dwall, objectness, class scores, nHits, charge, hasVtx, slice multiplicity); trained on EVEN MC nu photons vs EXT rows<100k; saves the joblib the exporter applies via `LARFORMER_SHOWER_BDT_NOVTX` (`showerNoVtxScore`) |
| `single_photon_diagnostics.py` | stacked MC/EXT + data distributions for selected events (final step and pre-chi2): flash PE, candidate start x/y/z, cos theta_beam, n reco protons, PE-centroid vs shower-line distance at x=0, distance to wall along +/- direction, flash PE vs E; needs the cascade dirs (flash PE) and reuses the pi0 study's cached RSE maps; `slurm/submit_single_photon_diagnostics.sh` |
| `single_photon_truth.py` | truth characterization of SIGNAL events stacked by signal category with the selected subset overlaid: true vertex x/y/z, detectable-photon true E / E_vis / cos theta_beam / conversion point + dwall, non-detectable photon true E / E_vis, summed KE of non-photon primaries, primary p / pi+- / mu counts; `slurm/submit_single_photon_truth.sh` |
| `single_photon_path_tables.py` | efficiency / purity tables split into FIVE paths: VERTEX sample (candidate on the primary in-FV vertex) x passes the SECOND-PHOTON veto (another nu-stream photon prong with reco E > `--veto2g-emin`, default 10 MeV) x (FULLY / PARTIALLY contained: any nu-stream track prong start or end < `--contain-dwall` 10 cm from a wall), plus vertex veto-fail, plus NO-VERTEX (vertex-less / non-FV-vertex candidate) x (pass / fail veto); per step from S2 on, per-category efficiencies, purities, a five-path summary and one final energy stack per cell; `slurm/submit_single_photon_path_tables.sh` |
| `single_photon_missed_photon.py` | missed-second-photon anatomy for selected in-FV >=2-visible-photon events: per missed photon a reco STAGE (no prong / merged into candidate / track prong / electron prong / photon prong E<=20 / attached BDT-fail / unattached vertex-free-BDT-fail), truth E, E_vis, vertex->conversion distance, conversion x/y/z + dwall, prong reco E + score, provisional m_gg with the accepted candidate (vs true m_gg), and a second-photon VETO scan (bkg removed vs signal kept per category); `slurm/submit_single_photon_missed_photon.sh` |
| `single_photon_ccmu_anatomy.py` | CC-numu background anatomy for selected events: muon failure mode (not reconstructed / reco'd as shower / track with wrong PID / mu-PID but soft because exiting, broken, or genuinely soft / not vetoed) x the accepted candidate's truth (visible photon, sub-threshold photon, the muon, electron, hadron, cosmic) + the true detectable-photon count; per-mode muon KE / reco KE / completeness plots; an exiting-muon veto scan (mu-PID track ending < d cm from a wall, any E); `slurm/submit_single_photon_ccmu_anatomy.sh` |
| `slurm/submit_single_photon_selection.sh`, `slurm/submit_photon_bdt_study.sh` | batch-partition wrappers (`sbatch ... [extra args]`; env `MC`, `EXT`, `DATA`, `PLOTS`, `OUTDIR`) |

Decisions folded into the code (2026-09-05): the X-visibility thresholds apply to
all nu-origin sim particles (primary + secondary; both reco secondaries and
non-primary muon tracks count too); photons entering from nu interactions outside
the FV/TPC are signal (own stack category "sig entering g"); the per-shower BDT
DEFINES the photon-candidate set before the multiplicity cut
(`--bdt-after-multiplicity` restores the first-pass ordering); beam data
(bnb5e19, 4.4e19 POT) is overlaid as points. Defaults: shower BDT >= 0.192 (the
pi0 staged working point; scan plot provided), flash chi2 < 316 (log10 2.5; chosen
from the N-1 plot, 2026-09-05), muon veto KE > 100 MeV with the union muon finder,
photon energy recalibrated with `--recal-gamma-a 0.01553 --recal-gamma-b -12.80`.
Plots land in `plots_v2_s1ep2p8/` (BDT study in `plots_v2_s1ep2p8/bdt_study/`).

### Vertex-less candidates (2026-09-06)

The exporter now writes segmenter particles that belong to no nu_reco
interaction (`showerVtxIdx = -1`, see `model_and_output_file_versions.md` §4b),
which is where 26% of the entering-photon signal was lost (another 30% sat on a
vertex outside the FV). `single_photon_selection.py --novtx` adds path B: any
nu-stream photon shower not attached to the primary in-FV vertex is a candidate
if `showerNoVtxScore >= --novtx-bdt-min` (`--novtx-model` scores from the ntuple
branches when the branch is unfilled). Use it on the `*_novtx.root` ntuples with
`--mc-odd-only` (the vertex-free BDT trained on even MC events) and WITHOUT the
`--recal-gamma-*` flags (calibration baked in at export).

First results (2026-09-06, novtx ntuples, odd-event MC x2, EXT analysis half,
cosmic BDT >= 0.192 for vertex-attached candidates, flash chi2 < 316):

| selection | eff combined | eff in-FV | eff entering | eff 1g0X | purity | EXT | data/pred |
|---|---|---|---|---|---|---|---|
| vertex path only (baseline) | 0.099 | 0.324 | 0.065 | 0.400 | 0.323 | 50.8 | 1.35 |
| + vertex-less path, NoVtx BDT >= 0.3 | 0.207 | 0.351 | 0.185 | 0.475 | 0.338 | 128.8 | 1.23 |
| + vertex-less path, NoVtx BDT >= 0.5 | 0.196 | 0.351 | 0.173 | 0.475 | 0.357 | 87.5 | 1.20 |
| + vertex-less path, NoVtx BDT >= 0.7 | 0.181 | 0.342 | 0.157 | 0.450 | 0.362 | 66.2 | 1.16 |

About half of the final signal comes through the vertex-less path. Vertex-free
BDT (held-out AUC 0.957; top features objectness, piS, sdwall, muS):
`plots_v2_s1ep2p8/novtx_bdt/`. Run dirs: `plots_v2_s1ep2p8_novtx_base/`,
`plots_v2_s1ep2p8_novtx_t{0.3,0.5,0.7}/`.

Out-of-FV backgrounds are split (2026-09-06) into "entering >=2 g" (a second
visible photon entered; same pi0 origin as most entering signal), "entering CC
(vis mu/e)" (a visible lepton entered; takes priority over the photon count) and
"entering other" (no visible photon, no lepton). At the vertex-less working
point these are 55.6 / 37.2 / 16.1 weighted events against 134.4 entering signal.

### TODO

- (DONE 2026-09-07, see "Vertex-less start finder v2" below; kept for the record)
  **Vertex-less shower start / direction (exporter).** For `showerVtxIdx=-1`
  prongs the start is the stage-4 keypoint model's predicted start (trained
  toward the photon ORIGIN, i.e. the nu vertex, not the conversion point:
  median 27 cm from the true conversion point, 16% outside the TPC) and the
  direction is the PCA axis signed away from that start (median 17 deg to the
  true photon direction, 19% flipped, vs 8 deg / 7% for vertex-attached
  showers). Fix: take the PCA axis, choose the trunk end as the extreme shower
  point closer to the keypoint start (fallback: narrower transverse profile),
  snap the start to that point, orient trunk -> tail. Needs a re-export and a
  retrain of the vertex-free BDT (which uses start dwall + direction cosines).

Diagnostics (2026-09-06, `plots_v2_s1ep2p8_novtx/diagnostics/{preChi2,final}/`):
signal peaks at low flash PE (median ~400-500 vs ~900-1000 for in-FV
backgrounds) and at 0 reco protons; entering signal starts within ~25 cm of a
wall along -dir (median 26 cm) and is forward (cos theta median 0.3); the
PE-centroid vs shower-line distance is not discriminating (many near-parallel
lines land in overflow). Data exceeds prediction at low flash PE, at 0
protons, at forward cos theta and for starts 100-350 cm from the wall.

Truth characterization (2026-09-06, `plots_v2_s1ep2p8_novtx/truth/`): entering
signal vertices lie in the LAr just outside the field cage on ALL sides (each
vertex is outside in at least one coordinate while the other two span the TPC;
peaks just beyond x=0, x=256 and |y|=117 cm), i.e. photons enter through every
face, with the cathode / anode sides and top / bottom most common; their
E_vis/E_true is ~0.7 (vs ~2.2 in-FV) because they deposit only part of the
shower; conversion points sit within ~10 cm of the wall; the non-detectable
second photon is usually >200 MeV (it misses the TPC); the hadronic system
carries ~440 MeV median KE but stays outside.

Max-KE plots (`truth/maxKE_{p,mu,pi}[_dep].png`, any sim particle, primary or
secondary; `_dep` = deposits charge in the TPC; lines at the MiniBooNE oil
Cherenkov thresholds p 342 / mu 39 / pi 51 MeV and the 100 MeV muon
convention). Fractions of ALL events in the category above threshold:
p>342: 1g0X 0 / 1g+X 0.21 / entering 0.14 (depositing: 0 / 0.21 / 0.04);
mu>39: 0.03 / 0.13 / 0.42 (depositing 0 / 0.13 / 0.01); mu>100 (any): 0.03 /
0 / 0.38 -- i.e. 38% of entering-signal events are CC with a muon above the
analysis threshold that never enters the TPC (the deposit requirement in the
signal definition keeps them as signal); pi>51: 0 / 0.25 / 0.18 (depositing
0 / 0.25 / 0.04). Overlay line = SIGNAL events passing the full selection.

Entering-signal split (2026-09-06): "sig entering g (no mu>100)" vs "sig
entering g, CC (mu>100 outside TPC)" -- the latter has a muon with KE >= 100
MeV that never deposits in the TPC (MiniBooNE would have seen it; MicroBooNE
cannot). Weighted: 485 no-mu + 293 CC-mu of the 778 entering signal; at the
vertex-less working point 90.2 + 44.2 selected (eff 0.186 / 0.151). All three
plot sets (selection, diagnostics, truth) carry the split; the efficiency
figures have six panels (in-FV, entering no-mu, entering CC-mu, combined, all
entering, strict 1g0X) and the cutflow has effOutNoMu / effOutMu columns.
Whether the CC-mu class stays in the signal definition is an open decision.

Vertex vs no-vertex samples (2026-09-06, `plots_v2_s1ep2p8_novtx/paths/`; final
step, weighted): VERTEX sample signal 84.9 (in-FV 37.8 + entering 47.2; eff
1g0X 0.40 / 1g+X 0.28 / entering 0.06), bkg MC 117 + EXT 46, purity 0.34,
in-FV purity 0.15, data/pred 1.34. NO-VERTEX sample signal 90.3 (in-FV 3.1 +
entering 87.2; eff entering 0.12 / 0.09), bkg MC 111 + EXT 41, purity 0.37,
entering purity 0.36, in-FV purity 0.01, data/pred 1.05. The no-vertex sample
is >96% entering by signal content -> usable as the entering-photon
sideband; the vertex sample is still 55% entering signal + entering bkg 20.

LATER: the EXT-BNB sample is Run 3 while the beam data is Run 1 (lower light
yield and quieter electronics in Run 3) -- a likely driver of the data/EXT
mismatch in the flash-chi2 and flash-PE shapes. Resolution = process a Run 1
EXT-BNB sample; deferred until the in-FV / entering separation is settled.
Diagnostics can be run per cell: `--path 0|1` (vertex / no-vertex),
`--veto2g pass|fail` (second-photon veto), `--contain fc|pc` (containment,
`--contain-dwall`); folders `preChi2/`, `preRatio/` (N-1 for the PE-ratio
step) and `final/`. Run-1 five-path cells:
`plots_run1_novtx_dep/diagnostics_vertex_pass_{FC,PC}/` (2026-09-28) plus the
older `diagnostics_{vertex,novertex}/`.

Per-path diagnostics (2026-09-06): in the VERTEX sample the backward wall
distance separates entering from in-FV signal (entering median 57-71 cm but
peaked < 50; in-FV 1g0X 168, 1g+X 76). MC-only scan at the final step
(inFV 37.7 / entering 47.2 / MC bkg 116.7): distFromWall > 50 cm keeps
30.4 / 18.9 / 85.3, > 100 cm keeps 19.9 / 9.4 / 61.8 -- the in-FV purity
among MC rises only 0.15 -> 0.22 because in-FV nu backgrounds (>=2 g, CC numu)
also sit deep inside. EXT in the vertex sample also peaks at < 50 cm. In the
NO-VERTEX sample everything (signal and entering bkg) is at < 30 cm, so the
cut is not useful there. Data/pred in the vertex sample at 100-350 cm stays
above 1 after the split.

### Vertex-less start finder v2 (2026-09-07)

Exporter `shower_start_dir()`: DBSCAN (eps 2 cm, min 5 points) on the shower's
points; clusters with >= 5 MeV of calibrated comb charge (slope only) are kept
(all points if none); the two extreme kept points along the 1st principal axis
are the candidate ends; the end closer to the segmenter/keypoint origin point
is the start; direction = principal axis oriented start -> other end. Flags
`--orphan-dbscan-eps`, `--orphan-min-cluster-mev`. Re-exported as
`*_novtx_v2.root`, validated against truth (`validate_novtx_start.py`), then
promoted; vertex-free BDT retrained (v1 model kept as
`export/data/shower_novtx_bdt_v1_kpstart.joblib`).

LOCKED IN (2026-09-07): start finder v2 validated on the full MC (494
vertex-less true photons): |start - true conversion| median 27 -> 5.8 cm,
90% 80 -> 58 cm, start inside the TPC 85% -> 99%; direction unchanged (median
16.4 deg, 18% sign-flipped -- the sign is still chosen by keypoint proximity;
a transverse-width rule is the next candidate if needed). Vertex-free BDT v2
(held-out AUC 0.958; vertex-less pass 0.64 at 0.5, EXT vertex-less rejection
0.97; sdwall now the top feature). All three `*_novtx.root` ntuples carry the
v2 start + baked v2 score; the v1 files are kept as `*_novtx_v1kpstart.root`.
Final selection (vertex-less path, BDT 0.5, chi2 < 316): combined eff 0.198 /
purity 0.350 (vertex sample 84.9 sig / 163 bkg, no-vertex sample 92.4 sig /
167 bkg). This is the reco state for the upcoming larger BNB-nu overlay and
EXT (incl. Run 1) campaigns.

### v2_s1ep2p8cew6 (new segmenter, 2026-09-09) -- `plots_cew6_novtx*/`

Verified the cew6 ntuples carry everything (vertex-less prongs, objectness,
slice chi2, baked showerNoVtxScore, v2 start finder: vertex-less start median
5.4 cm from the true conversion point, 98% in TPC). Same selection, same cuts
(cosmic BDT 0.192, NoVtx BDT 0.5, chi2 < 316), odd-event MC x2; both BDTs
are still the ep8-trained models unless noted.

| final step | ep8 | cew6 | cew6 + NoVtx BDT retrained on cew6 |
|---|---|---|---|
| combined eff / purity | 0.198 / 0.350 | 0.219 / 0.307 | 0.218 / 0.334 |
| in-FV eff (1g0X eff) | 0.351 (0.475) | 0.414 (0.525) | 0.405 (0.525) |
| entering eff no-mu / CC-mu | 0.190 / 0.151 | 0.210 / 0.158 | 0.212 / 0.155 |
| MC bkg / EXT | 233 / 97 | 301 / 142 | 296 / 93 |
| data / pred | 1.16 | 1.14 | 1.10 |

cew6 finds more signal at every step (S1 in-FV 0.72 -> 0.78, S2 0.56 -> 0.63)
but the ep8-trained per-shower cosmic BDT is looser on cew6 showers (EXT
photon rejection at 0.2: 0.83 -> 0.72; per-event EXT >=1 passing 0.10 ->
0.15), so EXT and in-FV backgrounds rise (>=2 g 66 -> 86, CC numu 39 -> 54).
Retraining the vertex-free BDT on cew6 (`export/data/shower_novtx_bdt_cew6.joblib`,
AUC 0.961) restores the EXT level (93) at unchanged efficiency; the cosmic
(vertex) BDT still needs its cew6 retrain (pi0 folder trainer). Per path:
vertex sample 96.5 sig / 165 MC bkg / 59 EXT (purity 0.30), no-vertex 99.8 /
136 / 83 (0.31). Runs: `plots_cew6_novtx{,_base,_ts0075,_bdtcew6}`,
`plots_cew6_novtx/{paths,diagnostics_vertex,diagnostics_novertex,truth,
bdt_study,novtx_bdt}`. The ep8 plots (`plots_v2_s1ep2p8_novtx*`) are kept.

### cew6 fresh-pair BDTs (2026-09-10) -- `plots_cew6bdt_novtx*/`

Re-ran everything on the re-exported cew6 ntuples. Verified score provenance:
`showerCosmicScore` IS the cew6-retrained model (81% of scores changed vs the
`*_ep8score.root` archives; its eff-0.97 WP is 0.164, used here), but
`showerNoVtxScore` was still the ep8 vertex-free model (reproduces it to
3e-8) because the export defaulted to `export/data/shower_novtx_bdt.joblib`,
which held the ep8 model while the cew6 retrain sat under a `_cew6` name.
The cew6 vertex-free model is now the deployed default (ep8 archived as
`shower_novtx_bdt_ep8.joblib`), and these runs applied it analysis-side with
`--novtx-model`. Numbers, cutflow and the vertex/no-vertex tables:
[`CEW6_SINGLE_PHOTON_TABLES.md`](CEW6_SINGLE_PHOTON_TABLES.md). Headline:
eff 0.218 / purity 0.342 (ep8 chain 0.198 / 0.350; cew6 with stale BDTs
0.219 / 0.307), EXT 85 (was 142), data/pred 1.08.

## RUN-1 TABLE-GAMMA SAMPLES (2026-09-23) -- the current default

The scripts now default to the run-1 simulation + run-1 beam-off + run-1
bnb5e19 data (versions doc section 3c), so prediction and data come from the
same run period with the calibrated flash-gamma table. Numbers and cut tables:
[`RUN1_SINGLE_PHOTON_TABLES.md`](RUN1_SINGLE_PHOTON_TABLES.md).

Script changes for it:
- `--mc/ext/data-ntuple` take a COMMA-SEPARATED list; `load()` concatenates,
  sums potTree and drops duplicate (run,subrun,event) rows (the two run-1
  BNB-overlay halves share one dlmerged file, 49 events). Default = both MC
  halves (370,580 events, POT 4.6533e20) and both EXT halves (219,154).
- No hygiene halves: the cew6 BDTs were trained on RUN-3 samples, so all of
  run-1 MC/EXT is held out. Wrappers pass `--ext-row-min 0` and drop
  `--mc-odd-only`.
- `--ext-scale` default 0.601, MEASURED in the pure-cosmic flash-chi2 sideband
  (`ext_norm_sideband.py`), not the spill ratio 0.8987 -- the spill value
  leaves data 10-25% under prediction with a slope in chi2, the measured one
  gives data/pred 1.00-1.03 at every chi2 cut. Same ~30% EXT spill deficit the
  pi0 suite reports.
- `single_photon_diagnostics.py` now reads flash quantities from the ntuple
  (`flashObsPE`, `nuSlicePredPE`, `flashTimeUs`) instead of the cascade files,
  and adds `flashPredPE`, `flashPERatio` (observed/predicted) and
  `flashTimeUs` plots; `--*-cascade` is only needed for the PE-centroid
  variables now and its RSE maps are cached on first use. Wall distances are
  vectorized and computed only for plotted events.
- No `--recal-gamma-*` (recal3 baked); `photon_bdt_study.py` needs the
  identity constants `--recal-gamma-a 0.020101 --recal-gamma-b -15.49`.

Headline: eff 0.254 / purity 0.320 at flash chi2 < 316 with data/pred **1.03**
(0.99-1.06 at every step), or eff 0.220 / purity 0.390 at chi2 < 178 with
data/pred 1.00. 9,296 raw signal events, 5.6x the run-3 sample.

### Missed second photon (2026-09-27, `plots_run1_novtx/missed_photon/`)

All the information is in the ntuple: the vertex-less export makes every
nu-slice segmenter particle a prong with its truth TID, so for each missed
photon we can tell whether a mask existed, whether it was attached, and which
cut it failed. The only thing the ntuple cannot split is the "no prong" class
(not found vs deghosted vs mis-sliced vs the fragment fell below the 10-point
orphan cut) -- that needs the slice-id sidecar inference pass, not the
standard cascade files (which only hold nu-slice particles too).

Run-1, 1,085 selected in-FV events with >=2 detectable photons (838 in the
>=2g category, 229 CC numu, 18 CC nue), 1,109 missed photons (weighted):

| stage | fraction |
|---|---|
| photon prong found, reco E <= 20 MeV | 0.44 |
| no prong | 0.34 |
| merged into the accepted candidate (photon purity - TID purity > 0.2) | 0.08 |
| photon prong not attached, vertex-free BDT fail | 0.07 |
| found as a track prong (mis-PID) | 0.05 |
| photon prong attached, cosmic BDT fail | 0.02 |
| found as an electron prong | 0.01 |

The dominant class is a photon the segmenter DID find but whose calibrated
energy is below the 20 MeV reco cut, while it is "visible" by the truth
convention because A_GAMMA x Q is ~2.2 x the true energy (missed-photon
true E peaks at 20-40 MeV, E_vis at 40-60). These prongs are usable: a
SECOND-PHOTON VETO on any other nu-stream photon prong with reco E > e_min
(no BDT) removes, at e_min = 10 MeV, 39% of the in-FV >=2g background and 39%
of the entering >=2g background at a cost of 2% (in-FV 1g+X), 6% (1g0X) and
8% (entering) of the signal; at 5 MeV, 49% / 46% removed for 4-10% signal
cost. Requiring m_gg(candidate, prong) in [50,250] MeV weakens the veto
(the soft prong's energy is under-measured, so the provisional m_gg peaks at
30-100 MeV, not 135; the TRUE m_gg is a clean pi0 peak for all stages). The
provisional m_gg is still a useful constraint plot for the leakage. Full scan
in `missed_photon_stages.txt`.

### Truth energy scale fixed (2026-09-27) -- `plots_run1_novtx_dep*/`

See `RUN1_SINGLE_PHOTON_TABLES.md` (top section) and
`lartpc/larformer_analysis/truth_calib.py`. Analysis-level only. All
single-photon scripts take `--truth-scale deposited|legacy` (default
deposited). Run-1 headline moves from eff 0.254 / purity 0.320 to
**0.325 / 0.375** at chi2 < 316 (1g0X 0.376 -> 0.416), data/pred unchanged
at 1.03. The pi0, nue and sigma0 scripts still hardcode 0.0253 and should
import the shared module (not changed here).

### Second-photon veto as a selection split (2026-09-27)

`single_photon_selection.build_steps` now tags every event with
`R["other_gE_max"]` (highest reco E among the OTHER nu-stream photon-labelled
prongs, any BDT score, any vertex); `single_photon_path_tables.py` splits the
final selection 2x2 by path x veto outcome (`--veto2g-emin`, default 10 MeV).
Results in `RUN1_SINGLE_PHOTON_TABLES.md` and `plots_run1_novtx_dep/paths_veto2g/`.

Diagnostics for the four cells (2026-09-27): `single_photon_diagnostics.py`
takes `--veto2g pass|fail` (with `--veto2g-emin`) next to `--path`, giving
`plots_run1_novtx_dep/diagnostics_{vertex,novertex}_{pass,fail}/` with the
usual `preChi2/` and `final/` sets (flash PE / predicted PE / ratio / time,
log chi2, start x/y/z, cos theta, n protons, PE-centroid distance, wall
distances, candidate E) and `medians.txt` per cell.

### Flash PE ratio cut (2026-09-27) -- `plots_run1_novtx_dep_pe3*/`

S5 = observed / predicted nu-slice flash PE < 3.0 (`--flashpe-ratio-max`),
appended after the chi2 cut in `build_steps` and therefore in every script
(N-1 plot `flashperatio_nminus1.png`; diagnostics gain a `preRatio/` set).
Run 1: EXT -32% for -5% signal, purity 0.375 -> 0.406 (vertex-pass cell
0.407 -> 0.447). Numbers in `RUN1_SINGLE_PHOTON_TABLES.md`.

### CC-numu background anatomy (2026-09-27, `plots_run1_novtx_dep/ccmu_anatomy/`)

Run 1, 592 selected in-FV CC-numu events (57.0 w). Muon failure modes:
mu-PID track but reco KE <= 100 MeV because the muon EXITS (range energy =
contained part only) 0.53 | track with wrong PID (p/pi) 0.13 | genuinely soft
after mu PID 0.13 | not reconstructed at all 0.12 | broken track 0.05 | reco'd
as a shower 0.05 (1.7 w where that shower IS the accepted photon). Only 24% of
the true muons are contained. The photon candidate is a REAL visible photon in
72% of events (event has 1 detectable photon 43%, >=2 30%, 0 27%); cosmic /
unlabeled 15%, electron 8%, the muon 3%, hadron 2%.
The dominant mode has a cheap fix: 72% / 83% of the exiting-mode reco tracks
end within 5 / 10 cm of a wall. An EXITING-MUON VETO (mu-PID track, any reco
E, end < 10 cm from a wall) keeps 0.55 of CC numu and 0.78 of entering CC
while keeping 0.996 / 0.951 / 0.982 of 1g0X / 1g+X / entering signal and
0.985 of the >=2g background (i.e. it is orthogonal to the second-photon
veto). Full scan in `ccmu_anatomy.txt`.

### Five paths: containment split (2026-09-27, `plots_run1_novtx_dep/paths5/`)

`single_photon_selection.reco_tables` now stores `track_min_dwall` (smallest
wall distance of any nu-stream track prong's start or end); the path tables
split the vertex veto-pass cell into FULLY / PARTIALLY contained
(`--contain-dwall`, 10 cm). Numbers in `RUN1_SINGLE_PHOTON_TABLES.md`. The
PC cell collects 65% of the vertex veto-pass CC-numu background (27.1 of
41.8) with only 5.4 EXT; the FC cell keeps 64% of the vertex-pass signal at
purity 0.43. NOTE: the selection gained a final flash PE-ratio step
(`--flashpe-ratio-max`, default 3.0, added in a parallel session and included
in commit 40f5f61); tables report both the chi2 step and that last step.

The flash PE-ratio cut (`--flashpe-ratio-max 3.0`) is adopted as the 6th
selection step (2026-09-28): it removes EXT from the vertex-pass cells
(FC EXT 63.7 -> 38.5, PC 5.4 -> 2.4) at ~5% signal cost.

### Plot-suite convention (2026-09-28)

`slurm/run_suite.sh <ROOT> [--flashpe-ratio-max X]` submits the whole
single-photon plot suite (selection + variants, five-path tables, per-cell
diagnostics incl. FC/PC, truth, missed-photon, CC-numu anatomy).
`plots_run1_novtx_dep*` = PE-ratio step OFF (final = S4 flash chi2);
`plots_run1_novtx_dep_pe3*` = PE-ratio step ON at 3.0 (final = S5; the
adopted selection). Every file in each tree was regenerated under its own
convention; stale PE-ratio outputs were removed from the OFF tree.

### Truth containment + FC/PC efficiencies (2026-09-28)

`truth_tables` now provides `pc_true` / `true_track_min_dwall` (any depositing
charged track-like particle with true start or end < 10 cm from a TPC wall;
mirrors the reco criterion up to endpoints-vs-trajectory). The path tables
print the truth-vs-reco agreement (0.94 on selected vertex-pass signal; the
disagreement is truth-PC events with a soft near-wall proton and no reco
track) and per-(category x truth containment) efficiencies into each reco
cell: 1g0X FC->FC 0.30, PC->PC 0.20; 1g+X FC->FC 0.43, PC->PC 0.32
(adopted selection). See `RUN1_SINGLE_PHOTON_TABLES.md`.

### Entering-vs-in-FV separators, due diligence (2026-09-28)

Diagnostics gained `vtxDwall` (reco primary-vertex wall distance; 0 for the
no-vertex path) and `startDwall` (candidate shower start wall distance).
Medians in the vertex veto-pass FC cell (adopted selection): in-FV signal vs
entering signal = 24-30 vs 18-21 cm (vtxDwall), 24-29 vs 18-19 cm
(startDwall), 130-144 vs 45-51 cm (distFromWall). The two new variables
barely separate the classes (entering photons convert a few cm inside, right
where in-FV vertices also sit), while the BACKWARD wall distance along the
shower direction separates them by a factor ~3 -- it stays the variable of
choice. Plots: `plots_run1_novtx_dep*/diagnostics_*/{preChi2,preRatio,final}/
{vtxDwall,startDwall}.png`.

### Six paths: FC split by backward wall distance (2026-09-28)

`single_photon_selection.build_steps` stores `cand_dist_from_wall` (and
`cand_dist_to_wall`, shared `rays_to_wall`); `path_tables` splits the vertex
veto-pass FC cell at `--dfw-cut` 100 cm into in-TPC (>= 100) and
entering-like (< 100 or start outside) cells -> six leaf paths (`paths6/`).
Diagnostics take `--dfw in|ent`; `run_suite.sh` now produces
`diagnostics_vertex_pass_FC_{in,ent}` and `paths6`. Numbers in
`RUN1_SINGLE_PHOTON_TABLES.md`: the in-TPC cell keeps 61% of the FC in-FV
signal at in-FV purity 0.25 with entering signal down 3.6x.
