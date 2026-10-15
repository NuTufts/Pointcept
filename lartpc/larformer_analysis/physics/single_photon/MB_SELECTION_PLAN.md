# Plan: a MiniBooNE-targeted single-photon selection (2026-10-01)

Goal: a high-PURITY sample of events that MiniBooNE would have seen as one
EM ring (truth definition `--signal-def mb`), kept in a separate script
(`single_photon_mb_selection.py`) so the LAr-rate selection
(`single_photon_selection.py`, `--signal-def lar`) is untouched. Purity is
prioritised over efficiency. RULE: no cuts on photon kinematics (energy,
direction, E_QE) -- BSM photons may sit elsewhere in phase space.

## Starting point (mb truth, adopted reco selection, primary cell)

Vertex path / 2nd-photon veto pass / FC / backward wall distance >= 100 cm,
final step, 4.4e19 POT (`plots_run1_novtx_mbdef_pe3/paths6`):

| component | events | note |
|---|---|---|
| signal (in-FV 13.6 = 1g0X 7.0 + 1g+X 6.6; entering 3.7) | 17.3 | purity 0.167, in-FV purity 0.131 |
| ring-making mu/pi/K/p above Cherenkov threshold | 25.6 | nProtons median 2, flash PE median 1374 |
| >=2 visible photons in FV | 20.8 | 2nd photon not reconstructed as a prong > 10 MeV |
| entering CC / >=2g / other (vertex outside, hard mu or hadron anywhere) | 16.5 | cosBeam median ~0, startX ~191 cm; dfw does NOT separate here |
| no visible photon (candidate is not a photon) | 2.8 | cosmic BDT median 0.51 |
| EXT cosmic | 20.4 | cosmic BDT median 0.47 (signal 0.96), flash PE 295, PE ratio 1.8 |

Cosmic BDT >= 0.5 alone (`paths6_tightbdt`): EXT 20.4 -> 9.0, signal 17.3 ->
15.6, purity 0.19. In-FV mb efficiency into this cell is 0.27-0.29 (truth FC,
in-TPC rows), so the cell already holds most of what the reco can deliver;
the work is background removal.

## Directions, in the order to try them

1. **MB-compatible hadronic system (targets the 25.6).** MiniBooNE allows
   only sub-threshold hadrons, so require every non-candidate prong to be a
   proton-PID track with reco KE < 342 MeV and no mu/pi-PID track with KE
   above ~40 / 50 MeV (the lar selection vetoes muons only above 100 MeV).
   Needs `trackRecoE` per PID; a Michel-electron tag at the track end would
   recover soft muons the PID misses. Then SPLIT into 1g0p / 1gNp (reco
   protons > 35 MeV, the WC convention): MiniBooNE's excess and the WC 2.5
   sigma are 0p; the 0p half loses the ring-making background automatically
   (nProtons median 2) and halves the >=2g background (median 1).
2. **Tight FV, with the 0p / Np distinction the user raised.** For Np events
   the reco vertex is the proton vertex: require vertex dwall >= X (scan 10 /
   20 / 30 cm; medians are 28-32 cm so 20 cm costs ~40%). For 0p events the
   "vertex" is the shower start, so use the shower start dwall AND the
   backward wall distance (already >= 100; scan 150 / 200) instead. Report
   the mb efficiency against a TIGHT-FV truth denominator (true vertex dwall
   >= X) so the concession to the small detector is explicit, and treat the
   out-of-tight-FV vertices as a separate (constraint) sample.
3. **Geometry against entering photons (NOT photon kinematics).** Rule for
   this selection: no cuts on photon energy, direction (cosBeam), E_QE or any
   other shower kinematic, because BSM single photons can populate a different
   region of phase space than the SM MB-like events; those variables are
   REPORTED (spectra for the MiniBooNE comparison) but never cut on.
   Allowed handles are geometric / topological: the backward wall distance
   (already >= 100; scan tighter), the tight FV of item 2, and -- within
   1gNp -- the vertex-to-shower-start conversion distance, which separates a
   real pi0 photon (converts 0-50 cm from the proton vertex) from an entering
   photon whose "vertex" IS the conversion point (distance 0). The entering
   photons that survive geometry are treated as an irreducible component
   constrained from the no-vertex and FC-entering cells, not cut further.
4. **Beam-likeness against EXT (targets the 20.4), kinematics-free.** Cosmic
   BDT >= 0.5 (EXT 20.4 -> 9.0 for 10% signal; check which BDT inputs are
   shower kinematics and consider a retrain without them), flash PE RATIO
   (observed / predicted, normalised so it does not track shower energy) < 2,
   and log10 chi2 < 2.25. Do NOT use a raw flash-PE floor: light yield scales
   with shower energy, so it is an energy cut in disguise.
5. **Hadronic-side and topology variables only.** Anything on the proton /
   track side (multiplicity, KE, containment, Michel tags) and on event
   topology (stray charge, second showers, vertex geometry) is fair game;
   anything that is a function of the candidate photon's energy or direction
   is not.
6. **Second-photon completeness (targets the 20.8).** Lower the 2nd-photon
   prong threshold 10 -> 5 MeV and include electron-labelled and vertex-less
   showers from ANY stream; add a "stray charge" veto = slice charge not in
   the candidate or in accepted proton tracks (the missed-photon study found
   34% of lost photons leave no prong). The per-prong `Charge` branch exists;
   a per-slice total charge branch does not, so the stray-charge veto needs
   one exporter addition (sum of nu-slice charge) -- flag before relying on it.
7. **Last resort: a dedicated MB BDT** on the variables above, trained on mb
   truth signal vs everything else in the cell. Previously tabled over the
   prior-enforcement concern; keep the cut-based chain as the headline and use
   the BDT only as a cross-check.

## Script shape

- Reuse `SPS.load / reco_tables / truth_tables(mb) / build_steps` up to the
  current final step, then apply M1..M6 as additional cumulative steps with
  the existing cutflow / stacked-energy / efficiency-vs-true-E machinery.
- Every new cut takes a threshold argument and produces an N-1 plot, so the
  scans above come from one job per setting (`run_suite.sh`-style wrapper).
- Report: efficiency vs mb truth (all in-FV, and tight-FV denominator), purity
  (all signal and in-FV), EXT, data/pred, split 1g0p / 1gNp, plus the shower
  energy, cosBeam and E_QE spectra (reported, never cut) for the final sample
  and for the entering and out-of-FV constraint samples (no-vertex pass, FC
  entering, PC).

## Results of the first pass (2026-10-01, `single_photon_mb_selection.py`)

Implemented M1 (MB hadrons), M2 (extra prong energy), M3 (tight FV; 0p via
start dwall + dfw), M4 (Np conversion distance, off by default), M5
(beam-likeness) on the primary cell; mb truth; 4.4e19 POT; one-at-a-time
threshold scan in `plots_run1_mbsel_scan/` (`mb_scan_summary.py`).

Default chain (tight 20 cm, 0p dfw >= 150, BDT >= 0.5), cumulative:

| step | signal | in-FV | effFV | purity | MC bkg | EXT |
|---|---|---|---|---|---|---|
| M0 primary cell | 17.3 | 13.6 | 0.148 | 0.167 | 66.0 | 20.4 |
| M1 MB hadrons (mu >= 40, pi >= 50, p >= 342 MeV vetoed) | 17.3 | 13.6 | 0.148 | 0.207 | 47.7 | 18.6 |
| M2 extra prong energy <= 10 MeV | 16.7 | 13.3 | 0.145 | 0.207 | 46.7 | 17.4 |
| M3 tight FV 20 cm (0p also dfw >= 150) | 9.9 | 8.7 | 0.095 | 0.193 | 31.9 | 9.6 |
| M5 beam-like (BDT >= 0.5, PE ratio < 2, logchi2 < 2.25) | 8.3 | 7.6 | 0.082 | 0.273 | 21.5 | 0.6 |

Scan findings: M1 is free (ring-making 25.6 -> 8.7 at no signal cost; a
100 MeV muon threshold is worse, 0.246); M2 does almost nothing (second
photons leave NO prong); tight FV costs signal without buying purity (20 cm:
-40% signal, purity 0.207 -> 0.193; 10 cm is the sweet spot, 0 cm is as pure);
the extra 0p dfw cut only costs signal; BDT >= 0.7 is the single best purity
lever (EXT -> 0, MC bkg -17%); conversion distance and a 60 MeV proton
threshold do not help. Chosen defaults: tight FV 10 cm, no extra dfw, BDT >= 0.7
(`comb_t10_bdt07`): signal 11.1 (in-FV 10.4), effFV 0.107, purity 0.305
(1gNp 0.333, 1g0p 0.258), MC bkg 24.0, EXT 1.2, data 29 / pred 36.

PURITY CEILING ~0.30 with the information in the ntuple. Remaining background
in the chosen selection (24 MC + 1.2 EXT): >=2-photon 14.2, of which the
second photon deposited < 20 MeV (invisible in LAr) 6.0, deposited >= 20 MeV
but left NO reconstructed prong 5.4, soft prong 2.8; entering photons 6.1
(mostly in 1g0p: 5.1 of its 9.5 background); ring-making mis-PID 3.6 (no
mu/pi track reconstructed at all). What would move it: (a) per-slice stray
charge from the exporter (the 5.4 no-prong events plus part of the 2.8);
(b) better track PID / finding for the 3.6; (c) the 6.0 invisible second
photons and most of the 6.1 entering photons are irreducible without photon
kinematics -> constrain them (no-vertex and FC-entering cells) rather than cut.
1g0p at 6.34e20 POT: ~50 signal at purity 0.26 (WC paper: 247 at 0.40 over all
five categories).

### Second pass: pi0-cone veto and the variable-insight study (2026-10-01)

- N-1 bookkeeping fixed (per-step conditions, `mb_conditions` / `all_but`);
  corrected N-1 at the chosen selection (sig | MC bkg | EXT | purity):
  without M1 hadron vetoes 11.1 | 32.3 | 1.8 | 0.245; without M2 extra energy
  11.3 | 24.5 | 1.2 | 0.307; without M3 tight FV 11.7 | 25.6 | 1.8 | 0.300;
  without M5 beam-like 15.6 | 44.1 | 15.6 | 0.207. M1 and M5 carry the purity;
  M2 and M3 are nearly neutral.
- `--pi0-cone` (M2b): a soft other-shower prong (E <= 10 MeV) whose angle to
  the candidate implies a pi0 partner energy E2 = m^2 / (2 E1 (1 - cos theta))
  in [5, 600] MeV and >= its own energy. It removes 30% of signal and 30% of
  background alike (M2 -> M2b: signal 16.7 -> 11.7, bkg 46.7 -> 32.6, purity
  0.207 -> 0.204; final 7.8 sig at 0.308 vs 11.1 at 0.305). NOT adopted: the
  soft clusters near real single photons are as common as near pi0 photons,
  and a single photon only constrains the partner to a cone family (one free
  parameter + azimuth), so the test is close to a flat soft-cluster veto.
- `mb_variable_study.py` (`plots_run1_mbsel_vars/`): HistGradientBoosting,
  odd/even folds, 30 kinematics-free variables, trained on the M1 cell
  (signal 17.3 / nu bkg 47.7 / EXT 18.6 weighted; 183 / 502 / 31 raw).
  * signal vs nu background: test AUC 0.60 -- essentially NO separation.
    Top single-variable AUCs: flash PE ratio 0.61, cosmic BDT 0.61, candidate
    objectness 0.58, photon score 0.57, start dwall 0.56. Mean BDT score is
    0.46-0.48 for EVERY background category, i.e. the nu backgrounds that
    survive M1 look like signal in everything but the photon kinematics.
  * signal vs EXT: test AUC 0.70; cosmic BDT 0.84, log10 chi2 0.77, PE ratio
    0.76, objectness 0.74 (all already in M5). Nothing else matters.
  Conclusion: with the present ntuple content and the no-kinematics rule the
  M1 + M5 chain is at the information limit; the next gain needs new
  information (per-slice stray charge; better soft-track finding / PID) or a
  constraint strategy for the >=2g and entering components.

### Third pass: pre-path BDT study, WC-style (2026-10-01)

`mb_variable_study.py --stage S3|pre` (all paths, before any path split;
path-defining quantities included as inputs; `plots_run1_mbsel_vars_{S3,pre}/`).
Pre stage (base S5, all paths): signal 85.0 (in-FV 36.8, entering 48.3), nu bkg
405.7 (entering CC 153, ring-making 113, >=2g 68 + entering >=2g 40, entering
other 20, no-photon 11), EXT 122. Targeted classifiers, test AUC and
background rejection at 0.9 / 0.8 / 0.7 signal efficiency:

| target (stage pre) | AUC | rej @0.9 / 0.8 / 0.7 | drivers |
|---|---|---|---|
| CC-numu-like (ring-making + entering CC) | 0.75 | 0.50 / 0.59 / 0.67 | n ring-making tracks, max mu/pi KE, max proton KE |
| >=2 photon (in-FV + entering) | 0.65 | 0.27 / 0.42 / 0.52 | start dwall, 2nd-photon prong E, other-shower E, dwalls |
| other nu | 0.74 | 0.23 / 0.52 / 0.67 | start dwall, vertex-free BDT, cosmic BDT, unified score |
| all nu | 0.69 | 0.42 / 0.50 / 0.58 | n ring-making, mu/pi KE, other-shower E, charge fraction |
| EXT | 0.85 | 0.63 / 0.79 / 0.85 | vertex-free BDT, log chi2, cosmic BDT, unified score |
| in-FV vs ENTERING mb signal | 0.89 | 0.71 / 0.82 / 0.90 | dfw, unified score, max proton KE, containment, n tracks, start dwall |

At S3 (before the flash cuts) the same picture with the flash chi2 taking
over the EXT / other-nu / numu separation (AUC 0.89 / 0.85 / 0.80).

Reading: (1) the entering-vs-in-FV separator is the strongest classifier in
the study (0.89) and it scores the entering BACKGROUNDS as entering (mean
0.36-0.40 vs ring-making 0.62) -> it is the natural first split: an in-FV-like
sample and an entering sideband, replacing the hand-made path / dfw cells.
(2) The CC-numu-like BDT is a hadronic-side classifier (ring-making tracks,
mu/pi/p KE) and can replace the hard M1 vetoes with a tunable score; (3) the
>=2-photon BDT is weak (0.65) and geometric (dwalls, 2nd-prong E): there is no
hidden second-photon information in the ntuple -> this background can only be
constrained, not cut, without stray-charge information; (4) EXT is handled by
the existing scores. The proposed structure (next step): pre-selection =
S3 (nu slice, one candidate, no visible X) -> BDT_ext (flash chi2, vertex-free
and cosmic scores, objectness) -> BDT_numu (hadronic side) -> BDT_enter split
into in-FV-like / entering-like samples -> within in-FV-like: 1g0p / 1gNp by
reco protons; the >=2g component is constrained by its own sideband
(2nd-photon-veto-fail cell) rather than cut.

### Fourth pass: the WC-style BDT chain (2026-10-01, `single_photon_mb_bdt_chain.py`)

Hygiene: BDTs trained on the run-1 overlay TRAINPOOL half + EXT first half;
evaluated and plotted on the RESERVED overlay half + EXT second half (scale
1.202) + bnb5e19 data, all at 4.4e19 POT. Pre-selection = S3 (nu slice, one
candidate on either path, no visible X); BDT_ext (signal vs EXT), BDT_numu
(signal vs CC-numu-like), BDT_enter (in-FV vs entering mb signal) on 39
kinematics-free inputs incl. the path-split quantities; `--no-flash` drops
log chi2 and PE ratio. Thresholds are quoted as TRAINING-signal efficiencies
(`plots_run1_mbchain/mb_bdt_chain_{flash,noflash}.txt`, models saved as
joblib).

In-FV-like sample (BDT_enter above threshold), flash variant:

| e_ext / e_numu / e_enter | signal | effFV | purity | MC bkg | EXT | 1g0p sig / pur | 1gNp sig / pur |
|---|---|---|---|---|---|---|---|
| 0.9 / 0.9 / 0.9 | 45.1 | 0.390 | 0.087 | 207.8 | 264.4 | 14.5 / 0.07 | 30.6 / 0.10 |
| 0.8 / 0.8 / 0.8 | 31.5 | 0.321 | 0.174 | 103.0 | 46.9 | 6.7 / 0.14 | 24.8 / 0.19 |
| 0.7 / 0.7 / 0.7 | 23.1 | 0.245 | 0.234 | 62.5 | 13.2 | 4.1 / 0.20 | 19.0 / 0.24 |
| 0.7 / 0.7 / 0.5 | 14.3 | 0.156 | 0.251 | 38.0 | 4.8 | 0.2 / - | 14.2 / 0.25 |
| fixed scores 0.7 / 0.7 / 0.7 (first run) | 9.9 | 0.110 | 0.385 | 15.8 | 0.0 | 0.2 / - | 9.7 / 0.38 |

Versus the cut-based chain (11.1 signal, effFV 0.107, purity 0.305): the BDT
chain reaches purity 0.385 at the same efficiency, or 2.3x the efficiency
(0.245) at purity 0.23. The entering sideband (BDT_enter below threshold)
holds 50 entering signal / 193 nu bkg / 89 EXT at the loose point with
data/pred 0.92. The purity ceiling (~0.25-0.4 depending on efficiency) is the
same information limit as before; the chain mainly converts it into efficiency.

Flash vs no-flash: score data/MC agreement at P0 is the same for both
(data/pred 0.955; chi2/ndf ext 28/16 vs 27/14, numu 20/12 vs 21/11, enter
45/15 vs similar) and after BDT_ext the numu / enter scores agree at 10/12
and 23/15. But the no-flash BDT_ext cannot replace the flash chi2: at the
0.7/0.7/0.7 point EXT is 89 vs 13 and purity 0.134 vs 0.234. Keep the flash
variables in BDT_ext (its agreement is acceptable) and revisit if the
post-selection flash distributions disagree.

Note: in-FV-like 1g0p at the loose default point has data 59 vs predicted 49
(d/p 1.20) -- low statistics (one overlay half), not to be read into yet.

### Fifth pass: score range, log-y plots, data deficit in the signal-like tail (2026-10-02)

- WHY the scores sat in 0.2-0.8: the class-balanced sample weights were
  normalised to sum 1, so every event carried a weight ~1e-4; HistGradientBoosting's
  L2-regularised leaf values (G / (H + lambda), lambda = 1) collapse when the
  weighted hessian H is that small, and the boosted log-odds never left ~[-1.5, 1.5].
  Fixed by normalising the weights to a mean of 1 (raw scores now span 0-1; signal
  quantiles for BDT_ext 0.43-0.97). The plotted / cut score is now FLATTENED
  (raw score -> training-signal ECDF): signal is uniform in [0,1] and a cut at
  s >= 1-e keeps a fraction e of the signal. Every plot also has a `_logy` copy.
- After the fix the chain is tighter per nominal efficiency (0.7/0.7/0.7: 14.3 sig,
  effFV 0.158, purity 0.263, EXT 4.8; 0.8/0.8/0.8: 20.5, 0.222, 0.260, 8.4).
- DATA DEFICIT in the most signal-like region: the flattened BDT_ext at P0 has
  data/pred ~0.3 in its top three bins (the 12% most signal-like, nu-MC dominated)
  with the flash inputs, ~0.5 without them; after B2/B3 the in-FV-like sample has
  data/pred 0.73 (flash) / 0.65 (no-flash). Per-input agreement (`mb_input_agreement.py`,
  `plots_run1_mbchain_inputs/input_agreement.txt`): at S5 the flash PE RATIO has
  the clearest shape mismatch (data shifted to higher obs/pred: data/pred 0.5-0.8
  below 1.2, 1.3-1.5 above; chi2/ndf 83/25), log chi2 81/25, and the vertex-found
  fraction differs (data 1.12x for vertex events, 0.86x for no-vertex); most other
  inputs sit at chi2/ndf ~2 with statistical errors only. The pe-ratio mismatch is
  a flash-model (light yield) calibration effect, i.e. the user's warning; it pushes
  data away from the "ratio ~ 1" signal-like region.
- Next: BDT_ext without pe_ratio (`--drop-vars pe_ratio`, tag `noperatio`) and the
  vertex-found fraction to be understood (the path variable feeds BDT_enter).

### Sixth pass: chain retrained as cross-fitted, early-stopped ensembles (2026-10-02)

Finding: the first chain's BDT_enter was overtrained -- 17.5% of the held-out
in-FV signal sat below flattened 0.05 (ideal 5%), 35% below 0.2, 60% below
0.5 -- because the flattening map was built from the memorised training
scores (264 in-FV training events, 300 iterations, min leaf 20). Retrain
(`FoldEnsemble` in `single_photon_mb_bdt_chain.py`): two fold models per BDT
split by event parity, each early-stopped on a held-out 20% (iterations now
86-183 instead of 300), min_samples_leaf 40, l2 3; evaluation scores average
the folds; the flattening map uses the OUT-OF-FOLD scores. Calibration on the
evaluation sample (fraction of signal below flattened 0.05 / 0.10 / 0.20 /
0.50; ideal = x), flash model: ext 0.042 / 0.104 / 0.230 / 0.546; numu 0.051 /
0.125 / 0.224 / 0.546; enter 0.070 / 0.119 / 0.234 / 0.619. The flattened
axis now means what it says to within a few percent.

Working points keep their flattened values (BDT_enter > 0.05, BDT_ext > 0.2,
BDT_numu > 0.4 for the tightbdt path; 0.7/0.7/0.7 efficiency point for the
chain cutflow) -- they now correspond to ~93% / ~77% / ~55-60% of the unseen
in-FV signal. Chain at 0.7/0.7/0.7 (flash): in-FV-like 24.2 sig / 68.0 MC /
7.2 EXT, purity 0.243, effFV 0.253, data/pred 0.90 (was 0.73: the honest
map moved the cut to a less extreme raw score where the data deficit is
milder). no-PE-ratio model: 25.7 / 72.7 / 13.2, purity 0.230, d/p 0.95.
The default model used by the diagnostics / path tables stays the flash one.

### Fifth pass: chain scores on the vertex + veto-pass cell, no containment split (2026-10-02, lar truth)

`plots_run1_novtx_dep_pe3/diagnostics_vertex_pass_reserved/` (RESERVED halves
only, EXT scale 1.202, default chain model `mb_bdt_chain_noflash_novtx`,
flattened scores) and `plots_run1_mbsel_vars_vpass_lar/` (study at the same
stage, `--stage vpass`). Final-step medians of the flattened BDT_enter
(1 = in-FV-like): 1g+X 0.62, 1g0X 0.39 | entering signal 0.14, entering >=2g
0.19, entering CC 0.16, entering other 0.11, EXT 0.28 | CC numu 0.55, >=2
visible g 0.55, CC nue 0.54. So BDT_enter separates entering from in-FV
WITHOUT the containment cut, but (a) in-FV backgrounds score in-FV-like, as
they should, and (b) 1g0X sits between in-FV and entering because the
separator leans on hadronic activity (study at vpass: in-FV vs entering AUC
0.876, drivers max proton KE 0.83, n tracks 0.83, conversion distance 0.73,
dfw 0.72, start dwall 0.64). BDT_numu medians: CC numu 0.02 vs signal 0.5-0.8
(AUC 0.85, max mu/pi KE). Data vs prediction: the two lowest BDT_enter bins
(most entering-like) show data 20 + 19 vs predicted ~22 (one overlay half,
EXT one half; watch, do not interpret yet).

Containment revision: the PC criterion (any nu-stream track start/end < 10 cm
from a wall, any PID, any KE) diverts ~27% of the vertex-pass in-FV signal
(15.6 of 58.7) into the PC leaf, mostly 1g+X with a proton or soft track near
a wall. Two options from the study: (1) restrict containment to mu/pi-PID
tracks or tracks above a KE threshold (the exiting-muon anatomy that motivated
PC); (2) drop the containment leaf and use BDT_enter (its top inputs are
exactly the hadronic-side quantities) with a 0p-specific treatment, since 0p
events carry no hadronic handle and must rely on conversion distance / dfw /
start dwall.

### Sixth pass: vertex + veto-pass split by chain BDT_numu at 0.1 (2026-10-05, lar truth)

`diagnostics_vertex_pass_{ccnumu,nonccnumu}` (RESERVED halves, EXT 1.202;
`--chain-numu-max 0.1` / `--chain-numu-min 0.1`, flattened score). Final step:

| cell | signal (in-FV: 1g0X / 1g+X; entering) | CC numu | >=2g | entering bkg | other |
|---|---|---|---|---|---|
| whole vertex + veto-pass | 101.3 (14.9 / 36.7; 49.7) | 36.5 | 28.9 | 14.4 | 7.9 |
| BDT_numu <= 0.1 (CC-numu-like) | 13.1 (0.0 / 10.8; 2.3) | 31.6 | 7.8 | 3.0 | 3.4 |
| BDT_numu > 0.1 (non-CC-numu) | 88.2 (14.9 / 25.9; 47.4) | 4.9 | 21.0 | 11.4 | 4.5 |

The 0.1 cut removes 87% of the CC numu (36.5 -> 4.9) for 13% of the signal
(all of it 1g+X, 10.8 of 36.7 = 29% of 1g+X; 1g0X and entering untouched).
In the non-CC-numu cell the in-FV vs entering split is visible in both
candidate variables: BDT_enter medians 1g+X 0.64 / 1g0X 0.39 vs entering
signal 0.13-0.15 and entering backgrounds 0.11-0.18 (CC numu 0.53, >=2g 0.53
stay in-FV-like); distFromWall medians 1g0X 111 / 1g+X 88 cm vs entering
42-46 cm (>=2g 159, entering other 93). Gallery: sections "vpass-ccnumu" and
"vpass-nonccnumu".

### Seventh pass: which truth trains BDT_numu (2026-10-05)

The chain defaulted to the mb truth, so BDT_numu learned to reject lar 1g+X
events carrying a pi > 51 MeV, a proton > 342 MeV or a 39-100 MeV muon
(mb "ring-making" background). Retrained with `--signal-def lar` (tag
`noflash_novtx_lar`) and again with muon-aware inputs added to the shared
feature builder (`mu_kemax`, `pi_kemax`, `mu_min_dwall`, `pi_min_dwall`,
`mu_exiting`, `mu_score_max`, `n_mu_tracks`; tag `noflash_novtx_lar_mu`).
Vertex + veto-pass cell split at flattened BDT_numu 0.1, lar truth, final step:

| BDT_numu model | 1g+X removed (of 36.7) | CC numu removed (of 36.5) | CC numu left | 1g0X / entering removed |
|---|---|---|---|---|
| mb-trained (noflash_novtx) | 10.8 (29%) | 31.6 (87%) | 4.9 | 0 / 2.3 |
| lar-trained | 5.0 (14%) | 28.4 (78%) | 8.1 | 0 / 1.3 |
| lar-trained + muon-aware inputs | 5.8 (16%) | 28.9 (79%) | 7.6 | 0 / 1.3 |

The truth definition was the issue; the muon-aware inputs change nothing at
this working point (the BDT already had the mu/pi/p counts and energies).
Recommended default: `mb_bdt_chain_noflash_novtx_lar.joblib` for the lar
analysis. Gallery sections "vpass-ccnumu-lar" / "vpass-nonccnumu-lar".
Variable study at the vertex + veto-pass stage with the muon-aware inputs
(`plots_run1_mbsel_vars_vpass_lar_mu/`): for signal vs CC-numu-like the
combined `mupi_kemax` still carries almost all the importance (0.30; `mu_kemax`
0.0045 with a single-variable AUC of 0.83, i.e. redundant); for in-FV vs
entering none of the new inputs enters the top list (max proton KE, n tracks,
conversion distance, dfw, start dwall unchanged). A muon-near-the-wall rule
has no separating power beyond what the existing counts already encode.

Operating-point comparison on the vertex + veto-pass cell (lar truth, final step;
`diagnostics_vertex_pass_{ccnumu,nonccnumu}_lar025`):

| cut | 1g+X removed (of 36.7) | CC numu removed (of 36.5) | CC numu left | entering sig removed | >=2g left |
|---|---|---|---|---|---|
| mb-trained BDT_numu <= 0.1 | 10.8 (29%) | 31.6 (87%) | 4.9 | 2.3 | 21.0 |
| lar-trained <= 0.1 | 5.0 (14%) | 28.4 (78%) | 8.1 | 1.3 | 23.8 |
| lar-trained <= 0.25 | 11.5 (31%) | 34.3 (94%) | 2.1 | 4.9 | 17.9 |

At matched signal cost (~30% of 1g+X) the lar model removes 94% of CC numu vs
87% for the mb model (2.1 vs 4.9 left) and also takes 3 more >=2g events; at
matched CC-numu rejection (~87%) the lar model would sit between its 0.1 and
0.25 points, costing roughly 20% of 1g+X instead of 29%. A modest but real
gain, not the factor-two the 0.1-vs-0.1 comparison suggested.

### Eighth pass: adopted BDT_numu cut and the in-FV / entering split scan (2026-10-05)

Adopted: lar-trained chain BDT_numu (flattened) > 0.25 on the vertex +
veto-pass cell. `enter_split_scan.py` on that cell (RESERVED halves, final
step; in-FV signal 39.3 = 1g0X 15.4 + 1g+X 23.9, entering 55.7, EXT 12.0):

| keep in-FV signal | distFromWall: entering removed | BDT_enter (lar): entering removed | 1g0X kept (dfw / BDT) |
|---|---|---|---|
| ~95% | 0.21 (x = 20 cm) | 0.52 (s = 0.10) | 0.96 / 0.93 |
| ~88% | 0.34 (30 cm) | 0.81 (s = 0.20) | 0.90 / 0.74 |
| ~80% | 0.48 (40 cm) | 0.93 (s = 0.30) | 0.84 / 0.55 |
| ~70% | 0.60 (60 cm) | 0.96 (s = 0.35) | 0.72 / 0.50 |
| ~50% | 0.78 (100 cm) | 0.99 (s = 0.55) | 0.53 / 0.24 |

BDT_enter dominates distFromWall everywhere: at 80% in-FV signal kept it
removes 93% of entering events vs 48% for the wall distance, and the kept
sample's in-FV purity is 0.55 vs 0.36. The two chain models (lar- and
mb-trained BDT_enter) give the same curve. The caveat is 1g0X: BDT_enter buys
its separation from the hadronic side, so at s = 0.30 it keeps 93% of 1g+X
but only 55% of 1g0X, while distFromWall treats both alike (84% / 82% at
40 cm). For a 1g0X-targeted sample either use distFromWall, or a lower
BDT_enter threshold for 0p events (s = 0.15: 83% of 1g0X kept, 72% of
entering removed, vs distFromWall 50 cm: 74% / 55%), which still beats the
wall distance on the 1g0X-only curve. Working points: s = 0.30 (all) or
s = 0.15 for 0p / 0.30 for Np. Plot: gallery section "split-scan".

### Final cells (2026-10-05, lar truth, RESERVED halves, EXT half2 x 1.202, bnb5e19 data, 4.4e19 POT)

Chain: base S1-S6 (nu slice, one candidate, no visible X, flash chi2 < 316,
PE ratio < 3, vertex cosmic BDT >= 0.6) -> vertex path -> 2nd-photon veto
pass -> lar-trained chain BDT_numu (flattened) > 0.25 -> BDT_enter >= 0.30
with an accepted proton (>= 35 MeV) / >= 0.15 without.

| cell | signal (1g0X / 1g+X / entering) | CC numu | >=2g | entering bkg | other | EXT | data | pred | in-FV purity |
|---|---|---|---|---|---|---|---|---|---|
| before the enter split | 84.1 (14.2 / 25.1 / 44.8) | 2.1 | 17.9 | 10.9 | 3.4 | 12.0 | 129 | 130.4 | 0.30 |
| **in-FV (MB target)** | 46.2 (11.0 / 23.6 / 11.6) | 2.1 | 16.8 | 2.4 | 1.7 | 4.8 | 68 | 74.0 | **0.47** |
| in-FV, 1g0p | 16.8 (6.3 / 0.9 / 9.5) | 0.2 | 4.3 | 1.8 | 0.4 | - | - | - | 0.31 (no EXT) |
| in-FV, 1gNp | 29.5 (4.7 / 22.7 / 2.1) | 2.0 | 12.5 | 0.6 | 1.3 | - | - | - | 0.60 (no EXT) |
| entering sideband | 37.9 (3.2 / 1.5 / 33.2) | 0.0 | 1.1 | 8.5 | 1.7 | 7.2 | 61 | 56.3 | 0.08 |

The in-FV cell keeps 34.6 of the 39.3 in-FV signal (88%; 1g0X 77%, 1g+X
94%) and holds 11.6 entering signal plus 2.4 entering background; its
remaining background is >=2 visible photons (16.8), i.e. the information
limit found earlier. The entering sideband is 88% entering (signal + bkg),
with data/pred 1.08; the in-FV cell has data/pred 0.92. In the 1g0p cell the
entering signal (9.5) still rivals the in-FV signal (7.2): a 0p-only tighter
BDT_enter or a distFromWall cut is the remaining lever there. Gallery
sections "final-infv" / "final-entering"; diagnostics options
`--enter in|ent --enter-cut 0.30 --enter-cut-0p 0.15`.

### Ninth pass: the 0p sample (2026-10-07)

`--protons 0p|np` added to the diagnostics. Cells
`diagnostics_vertex_pass_nonccnumu_lar025_{infv_0p,infv_np,0p}` (the last =
0p BEFORE the entering split, with `enter_split_scan.py` incl. a 2-D
distFromWall x BDT_enter grid). The 0p sample before the split is 9.5 in-FV
signal (1g0X 8.6) vs 49.8 entering (signal 39.7 + bkg 10.1) and 8.4 EXT, a
5:1 entering-to-in-FV ratio, so no cut on it reaches high purity:

| 0p, BDT_enter >= s | 1g0X kept | entering removed | EXT | in-FV purity |
|---|---|---|---|---|
| 0.10 | 0.89 | 0.57 | 3.6 | 0.21 |
| 0.15 (adopted) | 0.74 | 0.76 | 1.2 | 0.28 |
| 0.20 | 0.59 | 0.87 | 0.0 | 0.33 |
| 0.30 | 0.37 | 0.97 | 0.0 | 0.43 |
| distFromWall >= 100 cm | 0.57 | 0.78 | 2.4 | 0.23 |

Adding distFromWall on top of BDT_enter buys almost nothing (2-D grid: at
BDT >= 0.20, dfw >= 60 moves purity 0.33 -> 0.35 at the same 1g0X
efficiency); the two variables carry the same information for 0p events. In
the adopted 0p in-FV cell the medians still separate (distFromWall 1g0X
167 cm vs entering 92-139 cm) but start / vertex dwall do not (23 vs 18-24).
Why BDT_enter is weak for 0p: its strongest inputs (max proton KE, n tracks,
conversion distance) are all degenerate for proton-free events (no track, the
vertex IS the shower start), so only the geometric inputs act. A dedicated
0p separator trained on 0p events only (lar truth, inputs: dfw, dtw, start
dwall / position, candidate shape scores) is the obvious next experiment; the
ceiling is set by the 5:1 prior, so even a perfect 0p separator at 80% 1g0X
efficiency would need ~95% entering rejection to reach purity ~0.6.

### Tenth pass: second-photon veto geometry (2026-10-08, hand scan + `single_photon_veto2g_study.py`)

Hand scan of the 43 data events in the vertex / veto-FAIL cell below 200 MeV
(user, `larformer_reco/viz/scan_vertex_fail_lowE`): 25 plausible 2-photon
events, 15 vetoed by an unrelated small cluster far from the candidate (often
the other side of the detector), 4 unparseable; the second shower often falls
below threshold because the segmenter drops simply-connected chunks of the
photon. Study at the vertex stage (base final step, vertex path, before the
veto; MC 226 w, data 260 events): per candidate/other-prong pair the axis
crossing point, its distance to both starts (dcmax) and to the vertex, impact
parameters, closest approach, 'behind' flag; MC truth class of the other prong.

Current veto (photon prong E > 10 MeV) decides on, per fail event (w): in-FV
signal 5.3 = partner 2.9 / fragment 0.6 / nu non-photon 0.8 / junk 1.0;
>=2g bkg 19.7 = partner 16.6 / fragment 0.9 / junk 1.0. So in MC the veto fires
mostly on real partners; fragments and junk cost ~1.6 signal events. The
FAIL cell has data 57 vs pred 42.8 (d/p 1.33): the excess the scan saw.

| veto variant | sigFV pass | >=2g pass | EXT pass | data pass (d/p) | >=2g fail | data fail (d/p) |
|---|---|---|---|---|---|---|
| current | 54.9 | 33.7 | 16.2 | 203 (0.99) | 19.7 | 57 (1.33) |
| + crossing point within 100 cm of both starts | 55.9 | 35.2 | 19.8 | 225 (1.05) | 18.2 | 35 (1.03) |
| + crossing within 50 cm of the reco vertex | 56.0 | 36.8 | 19.2 | 228 (1.05) | 16.6 | 32 (1.01) |
| + start-to-start < 100 cm | 55.7 | 35.4 | 19.2 | 220 (1.03) | 18.0 | 40 (1.14) |
| + behind and skew < 30 | 57.6 | 40.7 | 19.2 | 239 (1.07) | 12.7 | 21 (0.86) |
| + other E > 20 MeV | 59.9 | 50.8 | 18.0 | 246 (1.03) | 2.6 | 14 (1.52) |

Requiring the two axes to cross within 100 cm of both starts (or within 50 cm
of the vertex) removes the data excess in the fail cell (1.33 -> 1.03) for
1.5 more >=2g in the pass cell and +1 in-FV signal, i.e. the extra data
fail events are exactly the unrelated-cluster vetoes; those clusters are also
partly EXT (EXT pass 16 -> 20). The 'behind + skew' requirement is too loose
(>=2g pass 40.7). Recommended veto: photon prong E > 10 MeV AND dcmax < 100 cm
(or dcv < 50 cm). The fragment problem is real but small in MC (0.6 signal);
its fix belongs in the segmenter/attachment, not the veto.

### Eleventh pass: geometric veto + BDT-path tree, held-out samples (2026-10-08)

Suite `plots_run1_bdtpaths_pe3/` (`slurm/run_bdtpaths_suite.sh`; tree table
`single_photon_bdt_paths.py`). lar truth; RESERVED overlay half + EXT half2
(x1.202) + bnb5e19 data (nothing trained on). Veto: photon prong E > 10 MeV
AND the two start axes cross within 100 cm of both starts (`--veto2g-dcmax
100`, new `build_steps` option; the BDT-chain feature keeps the old any-prong
quantity). Chain model lar-trained; BDT_numu > 0.25; BDT_enter >= 0.30 (Np) /
0.15 (0p). In-FV signal denominator 170.0.

| cell | signal (in-FV / entering) | eff in-FV | MC bkg | EXT | pred | data | d/p | purity | in-FV purity |
|---|---|---|---|---|---|---|---|---|---|
| vertex, veto FAIL (sideband) | 6.7 (4.8 / 1.9) | 0.028 | 24.9 | 3.6 | 35.3 | 35 | 0.99 | 0.19 | 0.14 |
| vertex, veto PASS | 103.1 (52.5 / 50.6) | 0.309 | 91.4 | 21.6 | 216.2 | 225 | 1.04 | 0.48 | 0.24 |
| vertex_pass, CC-numu-like | 17.6 (12.7 / 4.9) | 0.074 | 55.5 | 7.2 | 80.2 | 84 | 1.05 | 0.22 | 0.16 |
| vertex_pass, non-CC-numu | 85.6 (39.8 / 45.7) | 0.234 | 35.9 | 14.4 | 135.9 | 141 | 1.04 | 0.63 | 0.29 |
| non-CC-numu, ENTERING | 38.6 (4.7 / 34.0) | 0.027 | 11.6 | 8.4 | 58.7 | 65 | 1.11 | 0.66 | 0.08 |
| **non-CC-numu, IN-FV (MB target)** | 47.0 (35.2 / 11.8) | **0.207** | 24.3 | 6.0 | 77.3 | 76 | 0.98 | 0.61 | **0.455** |
| in-FV, 1g0p | 16.9 (7.3 / 9.7) | 0.043 | 8.5 | 2.4 | 27.9 | 27 | 0.97 | 0.61 | 0.26 |
| in-FV, 1gNp | 30.0 (27.9 / 2.1) | 0.164 | 15.8 | 3.6 | 49.4 | 49 | 0.99 | 0.61 | 0.57 |
| no-vertex, veto FAIL | 9.1 (0.6 / 8.6) | 0.003 | 23.4 | 7.2 | 39.8 | 35 | 0.88 | 0.23 | 0.01 |
| no-vertex, veto PASS (entering sideband) | 117.8 (7.8 / 110.0) | 0.046 | 68.9 | 52.9 | 239.6 | 213 | 0.89 | 0.49 | 0.03 |

Old any-prong veto on the same samples: vertex FAIL data 57 vs pred 45.7
(d/p 1.25), no-vertex FAIL 61 vs 58.3; with the geometric veto both fail
cells agree (0.99, 0.88) and the vertex PASS cell gains +1.8 signal / +3.8 bkg
/ +4.8 EXT. The MB-target cell: 47.0 signal (35.2 in-FV), purity 0.61 (in-FV
0.455), data/pred 0.98, 1g0p 27 data vs 27.9 predicted. Compared with the
2026-10-05 in-FV cell (46.2 signal, purity 0.47, d/p 0.92) the numbers are
the same within the veto change; the difference is that every cell now has
data/pred within 0.9-1.1. Remaining backgrounds in the MB cell: >=2 visible
photons 17.5, CC numu 2.3, entering 2.8, EXT 6.0. Gallery (separate page):
see artifact 'BDT-Path Suite Gallery'.
