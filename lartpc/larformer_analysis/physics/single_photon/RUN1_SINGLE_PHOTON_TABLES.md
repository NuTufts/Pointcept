# Single-photon selection on the RUN-1 table-gamma samples

Updated 2026-09-27 (truth energy scale fixed; the 2026-09-23 legacy-scale
numbers are kept below for comparison).

## TRUTH ENERGY SCALE (2026-09-27) -- `plots_run1_novtx_dep*/` are current

The truth "visible energy" constant 0.0253 MeV/ADC was a July-2026 RECO fit
(pre-label-expansion, pre-recal); on the truth charge it overstates deposited
energy by x2.30, so the 20 MeV visibility threshold meant ~9 MeV deposited
while the reco cut demanded 20 MeV. Measured on well-contained true particles
(`lartpc/larformer_analysis/truth_calib.py`): photons 0.0110, electrons
0.0101 MeV/ADC, identical on run-1 and run-3; reco showerRecoE over the
truth-calibrated deposited energy has a median of 1.04. Visibility is now
DEPOSITED energy >= 20 MeV (`--truth-scale deposited`, the default;
`legacy` reproduces the old numbers). Analysis-level only: no reco or export
rerun.

Signal denominators (w @ 4.4e19, both MC halves): combined 805.7 (was 882.4)
| in-FV 157.6 (137.9) | 1g0X 56.2 (48.8) | entering 648.1 (744.5) = no-mu
415.7 + CC-mu 232.4. Raw signal 8,507 events.

| final step | legacy scale, chi2<316 | **deposited, chi2<316** | deposited, chi2<178 |
|---|---|---|---|
| eff combined | 0.254 | **0.325** | 0.280 |
| eff in-FV | 0.402 | **0.466** | 0.400 |
| eff 1g0X | 0.376 | **0.416** | 0.350 |
| eff entering no-mu / CC-mu | 0.245 / 0.194 | **0.310 / 0.257** | 0.269 / 0.217 |
| purity | 0.320 | **0.375** | 0.454 |
| MC bkg / EXT | 294 / 180 | **256 / 180** | 200 / 71 |
| data / pred | 1.03 | **1.03** | 1.00 |

Cutflow (deposited, chi2 < 316): S1 0.817 | S2 0.522 | S3 0.470 | S4 0.325;
purity 0.010 / 0.052 / 0.080 / 0.375; data/pred 1.03 / 1.06 / 0.99 / 1.03.
Prediction and data are unchanged (reco is untouched); only the truth labels
of MC events moved: ~5% of former in-FV >=2g events, whose second photon
deposits under 20 MeV, are now single-photon signal.

Vertex vs no-vertex (deposited, chi2 < 316): VERTEX 135.6 sig / 149.9 MC bkg /
102.8 EXT, purity 0.349 (in-FV 0.170), d/p 1.10; NO-VERTEX 126.5 / 106.0 /
77.5, purity 0.408 (entering 0.384), d/p 0.93.

Missed second photon (deposited): 847 selected in-FV >=2g events (was 1,085),
864 missed photons: photon prong with reco E <= 20 MeV 0.41 (was 0.44 --
now a genuine threshold-edge/resolution population: deposited >= 20 but
reconstructed < 20), no prong 0.32, merged 0.10, unattached BDT-fail 0.09,
track 0.05, attached BDT-fail 0.02, electron 0.01. Second-photon veto at
reco E > 10 MeV: removes 43% of in-FV >=2g and 51% of entering >=2g for a
9% / 8% / 8% cost on 1g0X / 1g+X / entering signal (the cost rose from 2-8%
because signal events now legitimately contain soft second photons).

## 2x2 split: (vertex | no-vertex) x (passes | fails the second-photon veto)
(`plots_run1_novtx_dep/paths_veto2g/`, deposited scale, chi2 < 316; the tag =
another nu-stream photon-labelled prong with reco E > 10 MeV, any BDT score,
any vertex; `--veto2g-emin`)

| final step (w) | signal | in-FV sig | MC bkg | EXT | pred | data | d/p | purity | in-FV purity |
|---|---|---|---|---|---|---|---|---|---|
| VERTEX, passes veto | 125.7 | 60.3 | 113.8 | 69.1 | 308.7 | 333 | 1.08 | 0.407 | 0.195 |
| VERTEX, fails veto | 9.9 | 5.7 | 36.0 | 33.7 | 79.5 | 95 | 1.19 | 0.124 | 0.071 |
| NO-VERTEX, passes veto | 114.7 | 6.7 | 68.9 | 57.7 | 241.3 | 223 | 0.92 | 0.475 | 0.028 |
| NO-VERTEX, fails veto | 11.9 | 0.7 | 37.1 | 19.8 | 68.8 | 66 | 0.96 | 0.172 | 0.010 |

The "fails" halves are background-rich sidebands (purity 0.12 / 0.17) that
hold 24% (vertex) and 35% (no-vertex) of the MC background and 33% / 26% of
the EXT for 7% / 9% of the signal; the in-FV >=2g background splits 31.1
(pass) vs 18.8 (fail) in the vertex sample. Using the pass halves alone:
vertex purity 0.349 -> 0.407 (in-FV 0.170 -> 0.195), no-vertex 0.408 ->
0.475. The vertex-fail sideband shows the largest data excess (d/p 1.19 on
95 events).

## Flash PE ratio cut (2026-09-27) -- `plots_run1_novtx_dep_pe3*/` are current

New last step S5: nu-slice flashObsPE / nuSlicePredPE < 3.0
(`--flashpe-ratio-max`, <=0 disables; skipped on ntuples without the
table-gamma flash branches). Above 3 the sample is almost pure EXT.

| final step, chi2 < 316 | before (S4) | **after S5** |
|---|---|---|
| eff combined | 0.325 | **0.309** |
| eff in-FV / 1g0X | 0.466 / 0.416 | **0.451 / 0.395** |
| eff entering no-mu / CC-mu | 0.310 / 0.257 | **0.293 / 0.239** |
| purity | 0.375 | **0.406** |
| MC bkg / EXT | 256 / 180 | **242 / 122** |
| data / pred | 1.03 | **1.02** |

EXT -32% for -5% signal; MC background -5%. At chi2 < 178: eff 0.271, purity
0.466, EXT 56, d/p 0.99. Vertex-path-only baseline: purity 0.343 -> 0.372,
EXT 109 -> 73, d/p 1.13.

2x2 with the PE-ratio cut (`plots_run1_novtx_dep_pe3/paths_veto2g/`):

| final step (w) | signal | in-FV sig | MC bkg | EXT | pred | data | d/p | purity | in-FV purity |
|---|---|---|---|---|---|---|---|---|---|
| VERTEX, passes veto | 119.8 | 58.7 | 107.4 | 40.9 | 268.1 | 289 | 1.08 | 0.447 | 0.219 |
| VERTEX, fails veto | 9.5 | 5.6 | 35.3 | 27.0 | 71.8 | 86 | 1.20 | 0.132 | 0.078 |
| NO-VERTEX, passes veto | 108.0 | 6.2 | 63.6 | 35.5 | 207.0 | 187 | 0.90 | 0.522 | 0.030 |
| NO-VERTEX, fails veto | 11.3 | 0.6 | 36.0 | 18.6 | 65.9 | 61 | 0.93 | 0.171 | 0.009 |

The vertex-pass cell (the in-FV physics sample) went EXT 69.1 -> 40.9 (-41%)
for signal 125.7 -> 119.8 (-5%), purity 0.407 -> 0.447, in-FV purity
0.195 -> 0.219; no-vertex-pass EXT 57.7 -> 35.5 (-38%), purity 0.475 -> 0.522.
Per-cell diagnostics: `diagnostics_{vertex,novertex}_{pass,fail}/` with
`preChi2/`, `preRatio/` (the ratio N-1) and `final/`.

## Folder convention (2026-09-28)

Two complete, internally consistent plot suites (selection, chi2-225 and
vertex-only variants, five-path tables, per-cell diagnostics incl. the FC/PC
cells, truth, missed-photon and CC-numu anatomies; `slurm/run_suite.sh`):

| folder | cuts | final step |
|---|---|---|
| `plots_run1_novtx_dep*` | S1-S4 (flash chi2 < 316), NO flash PE-ratio cut (`--flashpe-ratio-max 0`) | S4 flash chi2 |
| `plots_run1_novtx_dep_pe3*` | + S5 flash PE ratio obs/pred < 3.0 (ADOPTED 2026-09-28) | S5 flash PE ratio |

Full selection, both conventions (combined eff / purity / EXT / data/pred):
no PE-ratio 0.325 / 0.375 / 180 / 1.03; with PE-ratio 0.309 / 0.406 / 122 /
1.01 (`plots_run1_novtx_dep_pe3/cutflow.txt`). Missed-photon and CC-numu
anatomies in the pe3 suite: 828 and 574 selected events (847 / 592 without).

## FIVE paths: vertex veto-pass split into fully / partially contained
(`plots_run1_novtx_dep/paths5/` without and `plots_run1_novtx_dep_pe3/paths5/`
with the PE-ratio step; PARTIALLY contained = any nu-stream track prong, any
PID / energy, whose reco start or end is < 10 cm from a TPC wall,
`--contain-dwall`; targets the exiting-muon loophole found by the CC-numu
anatomy.)

At the flash-chi2 step (chi2 < 316):

| cell (w) | signal | in-FV sig | MC bkg | EXT | data | d/p | purity | in-FV purity | CC numu |
|---|---|---|---|---|---|---|---|---|---|
| VERTEX, veto-pass, FULLY contained | 103.6 | 44.3 | 73.5 | 63.7 | 268 | 1.11 | 0.430 | 0.184 | 14.7 |
| VERTEX, veto-pass, PARTIALLY contained | 22.1 | 16.0 | 40.3 | 5.4 | 65 | 0.96 | 0.326 | 0.236 | 27.1 |
| VERTEX, fails veto | 9.9 | 5.7 | 36.0 | 33.7 | 95 | 1.19 | 0.124 | 0.071 | 8.0 |
| NO-VERTEX, veto-pass | 114.7 | 6.7 | 68.9 | 57.7 | 223 | 0.92 | 0.475 | 0.028 | 4.6 |
| NO-VERTEX, fails veto | 11.9 | 0.7 | 37.1 | 19.8 | 66 | 0.96 | 0.172 | 0.010 | 0.9 |

After the additional PE-ratio step (< 3.0): FC 98.4 sig / 67.7 MC / 38.5 EXT /
228 data, purity 0.481 (in-FV 0.211), d/p 1.11; PC 21.4 / 39.6 / 2.4 / 61,
purity 0.337, d/p 0.96; vertex-fail 9.5 / 35.3 / 27.0 / 86, d/p 1.20;
no-vertex pass 108.0 / 63.6 / 35.5 / 187, purity 0.522, d/p 0.90; no-vertex
fail 11.3 / 36.0 / 18.6 / 61, d/p 0.93.

The containment split does what it was meant to: of the vertex veto-pass
CC-numu background (41.8), 27.1 lands in the PC cell and 14.7 in FC, while
the in-FV signal splits 44.3 FC / 16.0 PC (1g+X carries most of the PC share:
its protons/pions reach the walls). The FC cell has 64% of the vertex-pass
signal at purity 0.43 with CC numu down to 15% of its background; the PC
cell is a CC-numu-dominated (67% of its MC background) sideband with very
little EXT (5.4) and in-FV purity 0.24.

## SIX paths: the FC cell split by backward wall distance (2026-09-28)

The candidate's BACKWARD wall distance (start point to the TPC box along
-direction; `cand_dist_from_wall` in the shared selection, NaN = start
outside the box, counted as entering-like) splits the vertex / veto-pass /
fully-contained cell at 100 cm (`--dfw-cut`): `paths6/` and the
`diagnostics_vertex_pass_FC_{in,ent}/` cells in both suites.

Adopted selection (`plots_run1_novtx_dep_pe3/paths6/path_tables.txt`):

| cell (w) | signal | in-FV sig | entering sig | MC bkg | EXT | data | d/p | purity | in-FV purity |
|---|---|---|---|---|---|---|---|---|---|
| FC, dfw >= 100 (in-TPC) | 41.8 | 26.4 | 15.4 | 41.5 | 20.4 | 119 | 1.15 | 0.403 | **0.254** |
| FC, dfw < 100 (entering-like) | 56.6 | 16.7 | 39.9 | 26.3 | 18.0 | 109 | 1.08 | 0.561 | 0.166 |
| vertex veto-pass PC | 21.4 | 15.6 | 5.8 | 39.6 | 2.4 | 61 | 0.96 | 0.337 | 0.246 |
| vertex veto-fail | 9.5 | 5.6 | 3.9 | 35.3 | 27.0 | 86 | 1.20 | 0.132 | 0.078 |
| no-vertex veto-pass | 108.0 | 6.2 | 101.8 | 63.6 | 35.5 | 187 | 0.90 | 0.522 | 0.030 |
| no-vertex veto-fail | 11.3 | 0.6 | 10.7 | 36.0 | 18.6 | 61 | 0.93 | 0.171 | 0.009 |

The in-TPC cell carries 61% of the FC in-FV signal (26.4 of 43.1) with the
entering signal cut from 55.4 to 15.4 and the entering backgrounds from 10.4
to 5.1; its residual background is in-FV physics (>=2g 19.8, CC numu 10.1)
plus 20 EXT, so its in-FV purity 0.25 is now limited by the in-FV
backgrounds, not by entering photons. The entering-like cell keeps 40 of the
55 entering signal at purity 0.56 and becomes the second entering sideband.
Efficiencies vs truth containment: 1g0X truth-FC -> in-TPC cell 0.199,
1g+X 0.256 (entering-like cell 0.104 / 0.169). Without the PE-ratio step
(`plots_run1_novtx_dep/paths6/`): in-TPC cell 44.0 signal / 46.3 MC bkg /
31.6 EXT, entering-like 59.6 / 27.2 / 32.1.

## Truth analogue of the backward wall distance + in-TPC / entering efficiencies (2026-09-29)

`truth_tables` now carries `g_dfw_true`: the signal photon's first energy
deposit (`trueSimPartEDep*`, always inside the TPC box) traced back along the
true momentum to the box, the same construction as the reco
`cand_dist_from_wall` (for an in-TPC vertex the ray passes the vertex and
continues to the wall in both). `dfw_in_true` = truth dfw >= 100 cm. Plot:
`paths6/dfw_truth_vs_reco.png`; truth spectra `truth/gDistFromWall.png`.

Truth in-TPC fraction of ALL true events (median truth dfw): 1g0X 0.62 (137 cm),
1g+X 0.54 (116 cm), entering no-mu 0.06 (19 cm), entering CC-mu 0.04 (16 cm).
Truth vs reco class agreement on the selected vertex / veto-pass / FC signal:
0.795 (reco entering-like among truth in-TPC 0.11; reco in-TPC among truth
entering-like 0.24); |reco - truth| dfw median 13 cm, 68% 38 cm.

Efficiency into the two FC cells, denominator = all true events of the
(category x truth FC x truth dfw) row, adopted selection
(`plots_run1_novtx_dep_pe3/paths6/path_tables.txt`):

| row | N(w) | -> FC in-TPC | -> FC entering | any cell |
|---|---|---|---|---|
| 1g0X, truth FC, in-TPC | 31.7 | **0.274** | 0.021 | 0.394 |
| 1g0X, truth FC, entering | 17.3 | 0.060 | 0.257 | 0.404 |
| 1g+X, truth FC, in-TPC | 38.0 | **0.378** | 0.042 | 0.525 |
| 1g+X, truth FC, entering | 22.9 | 0.054 | 0.380 | 0.512 |
| entering no-mu, truth FC, entering | 313.0 | 0.026 | 0.077 | 0.276 |
| entering CC-mu, truth FC, entering | 193.7 | 0.031 | 0.066 | 0.231 |

Selected signal in the FC in-TPC cell by truth class: 24.7 truth in-TPC / 17.2
truth entering-like; FC entering cell: 3.0 / 53.5. Without the PE-ratio step
(`plots_run1_novtx_dep/paths6/`): in-TPC efficiencies 0.286 (1g0X) / 0.383
(1g+X), agreement 0.791.

## Expected signal under the WireCell LEE definition, arXiv:2502.06064 (2026-09-30)

`single_photon_wc_lee_definition.py` (wrapper `slurm/submit_single_photon_wc_lee.sh`,
output `plots_run1_novtx_dep_pe3/wc_lee_definition/`): exactly one reconstructable
photon shower (true E > 20 MeV, first deposit >= 3 cm inside the TPC box;
same-vertex pairs with opening angle < 20 deg merged), no muon KE >= 100 MeV,
no primary e+-, NO vertex requirement. Run-1 BNB overlay (both halves), 4.4e19 POT:

| | WC definition | present analysis (cats 0-3) |
|---|---|---|
| total | 760.8 | 805.7 |
| true vertex inside the TPC box | 174.0 (23%) | 157.6 in WC FV |
| true vertex outside the TPC box | 586.8 (77%) | 648.1 |
| inside / outside the WC FV flag | 142.3 / 618.6 | 157.6 / 648.1 |
| CC / NC | 194.7 / 566.1 | |

Sensitivity: no opening-angle merging 729.4; photon also required to deposit
labelled charge 682.9 (inside 207.7 / outside 475.1); muon veto on primaries
only 764.6. Merging changes 158 w events, 31.5 of them become signal.

Overlap is only 297.7: WC-only 463.1 (350 from our "entering other", 47 from
"entering >=2 g"; 365 of them are single true photons with DEPOSITED E < 20
MeV, i.e. true-E-above-20 photons that leave < 20 MeV in the TPC). Present-only
508.0: 262.6 have a second true photon above 20 MeV true E that deposits < 20
MeV (WC counts it, we do not), 125.2 have the shower start < 3 cm from the
boundary, 123.6 fail the WC muon veto (our "entering CC-mu outside TPC" category,
232.4 w, is entirely excluded by WC: the muon is vetoed whether or not it
enters the TPC). Per category, 51-55% of our 1g0X / 1g+X / entering-no-mu
events are WC signal.

## FC in-TPC cell with a tighter candidate BDT cut (2026-09-30)

Extra cut on the FC in-TPC cell ONLY: leading-candidate cosmic BDT score >=
`--fc-in-bdt-min` (the candidate set is still defined by 0.164 before the
multiplicity cut, so the other five cells are unchanged; failing events go to
no path). `plots_run1_novtx_dep_pe3/paths6_tightbdt/` (cut 0.5) and
`diagnostics_vertex_pass_FC_in_tightbdt/` (`--cand-bdt-min 0.5`). In-FV
efficiency denominator = all true in-FV 1g events (157.6 w).

Scan (adopted selection, final step; `paths6_tightbdt/path_tables.txt`):

| BDT >= | in-FV sig | entering sig | MC bkg | EXT | data | d/p | eff in-FV | eff 1g0X | eff 1g+X | purity | in-FV purity | eff x purFV |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.164 (none) | 26.4 | 15.5 | 41.5 | 20.4 | 119 | 1.15 | 0.167 | 0.178 | 0.161 | 0.403 | 0.254 | 0.0426 |
| 0.5 | 24.7 | 11.8 | 35.6 | 9.0 | 86 | 1.06 | 0.157 | 0.160 | 0.155 | 0.450 | 0.304 | 0.0476 |
| 0.7 | 23.4 | 9.9 | 28.5 | 6.6 | 70 | 1.02 | 0.148 | 0.148 | 0.148 | 0.487 | 0.341 | 0.0506 |
| 0.8 | 21.1 | 7.8 | 24.2 | 4.2 | 56 | 0.98 | 0.134 | 0.131 | 0.135 | 0.504 | 0.368 | 0.0493 |
| 0.9 | 17.8 | 5.1 | 18.4 | 3.0 | 41 | 0.93 | 0.113 | 0.103 | 0.118 | 0.517 | 0.402 | 0.0453 |

At 0.5: EXT 20.4 -> 9.0, entering signal 15.5 -> 11.8, in-FV signal -6%,
in-FV purity 0.254 -> 0.304; eff x purity peaks near 0.7. Truth-FC in-TPC ->
FC in-TPC efficiencies at 0.5: 0.250 (1g0X) / 0.366 (1g+X); confusion diagonal
0.320. Diagnostics medians in the cell: cosmic BDT 0.96 for in-FV signal vs
0.86-0.90 entering.

## tightbdt BDT_numu cut loosened to 0.1 (2026-10-02)

User: the 0.4 numu working point was too aggressive -> tightbdt path =
BDT_enter > 0.3 & BDT_ext > 0.2 & BDT_numu > 0.1 (`run_suite.sh`). Final step:

| suite | signal | in-FV sig | MC bkg | EXT | data | d/p | in-FV purity |
|---|---|---|---|---|---|---|---|
| mb (numu > 0.1) | 17.9 | 17.1 | 34.8 | 3.6 | 44 | 0.78 | 0.304 |
| mb (numu > 0.4, previous) | 17.4 | 16.6 | 28.8 | 3.6 | 40 | 0.80 | 0.334 |
| wc | 19.9 | 18.7 | 32.8 | 3.6 | 44 | 0.78 | 0.333 |
| lar | 32.4 | 28.3 | 20.2 | 3.6 | 44 | 0.78 | 0.502 |

As expected a small change: +0.5 signal, +6 background (ring-making 4.8 ->
7.4, >=2g 19.0 -> 21.6); 1g+X efficiency 0.293 -> 0.303.

## Adopted chain configuration: FLASH-FREE, vertex-free model, BDT_enter 0.3 (2026-10-02, final)

User decision: the chain BDTs use NO flash variables (log chi2, PE ratio
dropped; the flash model is known to be poorly modelled) and no vertex-found
inputs (vtx_ok and the candidate path dropped, after their data/MC mismatch
in the input scan). Model `plots_run1_mbchain/mb_bdt_chain_noflash_novtx.joblib`
(35 inputs; same cross-fitted, early-stopped training; held-out calibration
ext 0.039 / 0.119 / 0.233 / 0.552, numu 0.046 / 0.137 / 0.233 / 0.558, enter
0.061 / 0.127 / 0.283 / 0.619 below 0.05 / 0.1 / 0.2 / 0.5). The chain's OWN
cutflow starts at S3 (no flash cuts), so its EXT is large without flash
inputs (B1 397 EXT); in the suites the base S4 / S5 / S6 steps are applied
before the paths, so the BDTs only act on the residual. Final step, 4.4e19:

| suite / cell | signal | in-FV sig | MC bkg | EXT | data | d/p | purity | in-FV purity |
|---|---|---|---|---|---|---|---|---|
| mb tightbdt (enter > 0.3, ext > 0.2, numu > 0.4) | 17.4 | 16.6 | 28.8 | 3.6 | 40 | 0.80 | 0.349 | **0.334** |
| mb enter split, in-FV-like / entering-like | 18.0 / 15.7 | 17.2 / 3.2 | 56.5 / 47.3 | 4.8 / 10.2 | 75 / 81 | 0.95 / 1.11 | 0.227 / 0.214 | 0.217 / 0.044 |
| wc tightbdt | 18.9 | 17.8 | 27.3 | 3.6 | 40 | 0.80 | 0.380 | 0.357 |
| lar tightbdt | 29.9 | 26.0 | 16.3 | 3.6 | 40 | 0.80 | 0.600 | 0.522 |
| lar enter split, in / ent | 38.0 / 48.0 | 33.9 / 6.1 | 36.4 / 15.0 | 4.8 / 10.2 | 75 / 81 | 0.95 / 1.11 | | 0.427 / 0.084 |

Versus the no-PE-ratio model (mb tightbdt 18.7 / 17.5 / 33.7 / 4.2, in-FV
purity 0.309, d/p 0.79): the flash-free model gives the same or slightly
better cell (0.334) and the data/pred stays 0.80. mb tightbdt background:
>=2 visible g 19.0, ring-making 4.8, entering CC 3.0, entering >=2g 1.0;
in-FV efficiencies 1g0X 0.204 / 1g+X 0.293 (truth-FC in-TPC rows).

## Adopted chain configuration: no-PE-ratio model, BDT_enter 0.3 (2026-10-02)

User decisions: the default chain model is the retrained no-PE-ratio ensemble
(`plots_run1_mbchain/mb_bdt_chain_noperatio.joblib`; same cross-fitted
training, held-out calibration enter 0.061 / 0.156 / 0.279 / 0.607 below
0.05 / 0.1 / 0.2 / 0.5) and BOTH the BDT_enter split path (`paths6_enter/`)
and the tightbdt path use BDT_enter > 0.3. Final step, 4.4e19 POT:

| suite / cell | signal | in-FV sig | MC bkg | EXT | data | d/p | purity | in-FV purity |
|---|---|---|---|---|---|---|---|---|
| mb tightbdt (enter > 0.3, ext > 0.2, numu > 0.4) | 18.7 | 17.5 | 33.7 | 4.2 | 45 | 0.79 | 0.330 | 0.309 |
| mb enter split, in-FV-like | 19.1 | 17.8 | 61.1 | 7.2 | 81 | 0.93 | 0.218 | 0.203 |
| mb enter split, entering-like | 14.6 | 2.6 | 42.6 | 7.8 | 75 | 1.15 | 0.224 | 0.041 |
| wc tightbdt | 20.4 | 18.7 | 32.0 | 4.2 | 45 | 0.79 | 0.360 | 0.330 |
| lar tightbdt | 33.9 | 27.9 | 18.5 | 4.2 | 45 | 0.79 | 0.599 | 0.492 |
| lar enter split, in / ent | 41.4 / 44.6 | 35.1 / 4.9 | 38.8 / 12.6 | 7.2 / 7.8 | 81 / 75 | 0.93 / 1.15 | | 0.401 / 0.076 |

Versus the flash model at the same cuts (mb tightbdt 17.9 / 16.8 / 30.3 / 4.2,
in-FV purity 0.321, d/p 0.76): essentially the same cell, data/pred 0.76 ->
0.79 (45 vs 57). mb tightbdt background: >=2 visible g 21.2, ring-making 5.3,
entering CC 4.2, entering >=2g 1.7; in-FV efficiencies 1g0X 0.216 / 1g+X 0.298
(truth-FC in-TPC rows).

## Chain retrained; working points re-derived (2026-10-02, supersedes the two sections below)

The first chain's BDT_enter was overtrained (17.5% of held-out in-FV signal
below flattened 0.05, ideal 5%). Retrained as 2-fold cross-fitted,
early-stopped ensembles with the flattening map from out-of-fold scores
(MB_SELECTION_PLAN.md, sixth pass); evaluation-sample calibration is now
within a few percent (enter: 0.07 / 0.12 / 0.23 / 0.62 below 0.05 / 0.1 / 0.2 /
0.5). With the honest map the old "0.05" cut (really raw 0.62) corresponds to
about flattened 0.3, so the mb vertex / veto-pass / FC cell was re-scanned
(`plots_run1_novtx_mbdef_pe3/scan_enter/`), with BDT_ext > 0.2 & BDT_numu > 0.4:

| BDT_enter > | signal | in-FV sig | MC bkg | EXT | data | d/p | in-FV purity | eff 1g0X / 1g+X (truth-FC in-TPC) |
|---|---|---|---|---|---|---|---|---|
| 0.05 | 28.8 | 20.0 | 62.4 | 7.8 | 91 | 0.92 | 0.201 | 0.247 / 0.303 |
| 0.1 | 27.3 | 20.0 | 57.3 | 7.8 | 86 | 0.93 | 0.216 | 0.247 / 0.303 |
| 0.2 | 23.8 | 19.6 | 48.8 | 6.6 | 70 | 0.88 | 0.247 | 0.247 / 0.303 |
| **0.3 (adopted)** | 17.9 | 16.8 | 30.3 | 4.2 | 40 | 0.76 | **0.321** | 0.196 / 0.298 |
| 0.5 | 13.1 | 13.0 | 20.8 | 3.0 | 26 | 0.70 | 0.351 | 0.118 / 0.279 |

tightbdt path now = BDT_enter > 0.3 & BDT_ext > 0.2 & BDT_numu > 0.4
(`run_suite.sh`). Final cells: mb 17.9 / 16.8 / 30.3 / 4.2, in-FV purity 0.321
(bkg: >=2 visible g 19.9, ring-making 4.7, entering CC 3.6, entering >=2g 1.0);
wc 19.5 / 18.1 / 28.7 / 4.2, 0.345; lar 31.5 / 26.8 / 16.7 / 4.2, 0.511.
Data 40 vs predicted 52 (d/p 0.76) in that cell -- the signal-like-tail
deficit persists. The BDT_enter split path (`paths6_enter/`) is still at
0.05: with the honest map that keeps 93% of in-FV signal but leaves only
~4 entering signal in the sideband; 0.2-0.3 would be the equivalent of the
earlier split (user decision pending).

## "tightbdt" path redefined as the MB-chain cuts (2026-10-02)

With the cosmic BDT >= 0.6 now a standard step (S6), `paths6_tightbdt/` and
`diagnostics_vertex_pass_FC_in_tightbdt/` are redefined: in-FV-like cell =
vertex / veto-pass / FC with BDT_enter > 0.05 & BDT_ext > 0.2 & BDT_numu > 0.4
(flattened chain scores; events failing the ext/numu cuts land in no path, the
entering-like cell is the BDT_enter <= 0.05 sideband). Final step, 4.4e19:

| suite | signal | in-FV sig | MC bkg | EXT | data | d/p | purity | in-FV purity |
|---|---|---|---|---|---|---|---|---|
| mb | 18.3 | 17.4 | 29.6 | 3.0 | 37 | 0.73 | 0.360 | **0.342** |
| mb, BDT_enter split only | 20.1 | 18.9 | 61.3 | 7.8 | 84 | 0.94 | 0.226 | 0.212 |
| wc | 20.0 | 18.5 | 27.9 | 3.0 | 37 | 0.73 | 0.394 | 0.364 |
| lar | 32.1 | 26.3 | 15.8 | 3.0 | 37 | 0.73 | 0.632 | 0.516 |

mb in-FV-like background after the chain cuts: >=2 visible g 18.8, ring-making
4.2, entering CC 3.8, entering >=2g 1.8, other 1.3; EXT 3.0. The ext/numu cuts
halve the background (61 -> 30) for 8% of the in-FV signal; the ring-making
component drops 27.6 -> 4.2 and EXT 7.8 -> 3.0. Remaining background is 63%
in-FV >=2 photon. mb in-FV efficiencies into the cell: 1g0X truth-FC in-TPC
0.227, 1g+X 0.279 (plus the entering-class rows 0.138 / 0.355).
CAUTION: data 37 vs predicted 51 (d/p 0.73) -- the same signal-like-tail
deficit seen on the chain scores (fifth pass of MB_SELECTION_PLAN.md); the
cell is small (37 data events).

## Alternative FC split by the MB-chain BDT_enter score (2026-10-02)

`--fc-split enter --enter-cut 0.05` (path tables; diagnostics `--enter in|ent`)
splits the vertex / veto-pass / FC cell by the chain's in-FV-vs-entering score
instead of the backward wall distance. Separate outputs (`paths6_enter/`,
`diagnostics_vertex_pass_FC_enter_{in,ent}/`); the dwall split (`paths6/`) is
untouched. Final step (after S6), 4.4e19 POT:

| suite / cell | signal | in-FV sig | MC bkg | EXT | data | d/p | purity | in-FV purity |
|---|---|---|---|---|---|---|---|---|
| lar, BDT_enter >= 0.05 (in-FV-like) | 43.4 | 36.1 | 38.0 | 7.8 | 84 | 0.94 | 0.487 | **0.405** |
| lar, BDT_enter < 0.05 (entering-like) | 42.6 | 3.9 | 13.4 | 7.2 | 72 | 1.14 | 0.673 | 0.061 |
| lar, dwall >= 100 (in-TPC) | 35.3 | 24.0 | 31.8 | 7.2 | 79 | 1.06 | 0.475 | 0.323 |
| lar, dwall < 100 (entering-like) | 50.7 | 16.0 | 19.6 | 7.8 | 77 | 0.99 | 0.649 | 0.205 |
| wc, BDT_enter in / ent | 26.0 / 16.6 | 23.9 / 1.7 | 55.4 / 39.4 | 7.8 / 7.2 | 84 / 72 | | 0.291 / 0.263 | 0.268 / 0.027 |
| wc, dwall in / ent | 19.5 / 23.2 | 16.1 / 9.6 | 47.6 / 47.1 | | 79 / 77 | | 0.262 / 0.297 | 0.216 / 0.122 |
| mb, BDT_enter in / ent | 20.1 / 13.5 | 18.9 / 1.5 | 61.3 / 42.5 | 7.8 / 7.2 | 84 / 72 | | 0.226 / 0.214 | 0.212 / 0.024 |
| mb, dwall in / ent | 14.8 / 18.9 | 12.4 / 8.0 | 52.4 / 51.4 | | 79 / 77 | | 0.198 / 0.242 | 0.167 / 0.103 |

The BDT split moves in-FV signal into the in-FV-like cell (lar 24.0 -> 36.1,
+50%; mb 12.4 -> 18.9) and empties the entering-like cell of in-FV signal
(lar 16.0 -> 3.9; mb 8.0 -> 1.5), i.e. it is a much cleaner in-FV / entering
partition of the SIGNAL. In-FV purity of the in-FV-like cell rises 0.32 -> 0.41
(lar), 0.22 -> 0.27 (wc), 0.17 -> 0.21 (mb). The price: more entering
background in the in-FV-like cell (lar entering CC+>=2g+other ~8 -> 12) and
the entering-like sideband loses purity as a constraint sample for the
in-FV-like region. In-FV efficiencies into the in-FV-like cell: lar 1g+X
truth-FC 0.323 (in-TPC) / 0.368 (entering-class), i.e. BDT_enter keeps the
in-FV events that the dwall split had mis-assigned. Truth-dfw agreement
drops (0.83 -> 0.75 lar, 0.89 -> 0.73 mb) because the BDT classifies by vertex
location, not by the wall-distance proxy. Note the entering-like cell has
d/p 1.14 (72 vs 63), low statistics.

(Fixed in passing: the "reco entering-like among truth in-TPC" line in
`path_tables.txt` was corrupted by a bitwise-not on the integer code; now uses
explicit == 0 / == 1.)

## S6 pre-selection: vertex-path cosmic BDT >= 0.6 (2026-10-02)

`build_steps` now ends with S6: vertex-path (path 0) candidates must have
cosmic BDT >= `--vtx-cosmic-bdt-min` (default 0.6; 0 disables); vertex-less
candidates pass unchanged. Applied in every suite (`lar`, `wc`, `mb`) and in
the MB cut-based selection (its M0 cell starts from S6). Final step, 4.4e19:

| | S5 (old final) | S6 |
|---|---|---|
| lar: signal / eff / purity / MC bkg / EXT / data / d/p | 252.1 / 0.302 / 0.411 / 238.7 / 122.0 / 623 / 1.02 | 234.6 / 0.281 / 0.450 / 210.6 / 76.3 / 508 / 0.97 |
| wc: signal / eff / purity | 116.3 / 0.153 / 0.190 | 108.3 / 0.142 / 0.208 |
| mb: signal / eff / purity | 85.0 / 0.165 / 0.139 | 78.7 / 0.153 / 0.151 |
| FC in-TPC cell purity, lar / mb | 0.405 / 0.167 | 0.475 / 0.198 |

EXT falls 122 -> 76 for 7% of the signal; the vertex veto-fail cell is now
d/p 1.33 (57 data vs 43 predicted, low statistics). Diagnostics folders gain
a `preBDT/` stage (after S5, before S6) and three new variables: `mupiKEmax`
(max mu/pi-PID track KE, -1 = none) and the MB-chain scores `bdtChainExt`,
`bdtChainNumu`, `bdtChainEnter` (flattened, from
`plots_run1_mbchain/mb_bdt_chain_flash.joblib`; `--mb-chain-model`).
Final-step medians in the mb vertex sample: BDT_enter 0.33 (1g0X) / 0.69
(1g+X) vs 0.00-0.01 for the entering backgrounds and 0.62 for ring-making;
BDT_numu 0.78-0.81 (signal) vs 0.00 (ring-making); mupiKEmax is -1 for every
category except ring-making (69 MeV).

## BDT-path tree with the geometric second-photon veto (2026-10-08)

See MB_SELECTION_PLAN.md 'Eleventh pass' (suite `plots_run1_bdtpaths_pe3/`, RESERVED halves): MB-target cell 47.0 signal (35.2 in-FV), in-FV purity 0.455, data 76 / pred 77; every cell data/pred 0.9-1.1 after the veto change (old veto fail cell was 1.25).

## MiniBooNE-targeted selection: final cells (2026-10-05)

See MB_SELECTION_PLAN.md 'Final cells' for the chain and the table (in-FV cell: 46.2 signal, 34.6 in-FV, purity 0.47, EXT 4.8, data 68 / pred 74; entering sideband 37.9 / data 61 / pred 56).

## MiniBooNE-targeted selection, first pass (2026-10-01)

`single_photon_mb_selection.py` (`plots_run1_mbsel/`, scan in
`plots_run1_mbsel_scan/`, plan + numbers in `MB_SELECTION_PLAN.md`): primary
cell + MB hadron vetoes (mu >= 40, pi >= 50, p >= 342 MeV) + extra prong
energy <= 10 MeV + tight FV 10 cm + cosmic BDT >= 0.7 / PE ratio < 2 /
logchi2 < 2.25; no photon-kinematic cuts. mb truth, 4.4e19 POT: signal 11.1
(in-FV 10.4), in-FV eff 0.107, purity 0.305 (1gNp 7.6 sig / 14.5 bkg = 0.333;
1g0p 3.5 / 9.5 = 0.258), EXT 1.2, data 29 vs pred 36. Purity ceiling ~0.30:
remaining bkg = >=2 photon 14.2 (second photon invisible 6.0 / no prong 5.4 /
soft prong 2.8), entering 6.1, ring-making mis-PID 3.6.

## Three signal definitions: lar / wc / mb (2026-10-01)

`--signal-def lar|wc|mb` (README has the definition table; `present` = `lar`).
`lar` now excludes decay (Michel) electrons from the electron veto, which moved
~30 selected events from "visible e" into signal relative to the earlier
"present" numbers. Same reco selection, final step (S5), 4.4e19 POT:

| | lar (detectable in LAr) | wc (arXiv:2502.06064) | mb (MiniBooNE ring) |
|---|---|---|---|
| true signal: all / in-FV / entering | 834.9 / 174.5 / 660.4 | 760.8 / 153.0 / 607.9 | 513.9 / 92.0 / 421.9 |
| selected signal | 252.1 | 116.3 | 85.0 |
| eff: combined / in-FV / entering / 1g0X | 0.302 / 0.418 / 0.271 / 0.387 | 0.153 / 0.323 / 0.110 / 0.328 | 0.165 / 0.400 / 0.114 / 0.354 |
| purity: all / in-FV | 0.411 / 0.119 | 0.190 / 0.081 | 0.139 / 0.060 |
| MC bkg | 238.7 | 374.5 | 405.7 |
| FC in-TPC cell: sig / MC bkg / purity | 42.0 / 41.3 / 0.405 | 22.5 / 60.8 / 0.217 | 17.3 / 66.0 / 0.167 |

EXT 122.0, data 623, d/p 1.02 in all three (reco identical). Under `mb` the
true signal is a third smaller than `wc` because any pi above 51 MeV, K above
180 MeV or proton above 342 MeV (ring-making in oil) vetoes; the selected
"entering CC" background grows to 153 because entering events with a hard
muon or hadron anywhere are vetoed, and category 7 ("ring-making mu/pi/K/p")
holds 11 selected events. In-FV efficiencies are similar across definitions
(0.32-0.42); the headline differences are in what counts as signal, not in
what the reco keeps.

## Signal-definition switch: present vs WC-like, and the true opening angle (2026-10-01)

`--signal-def present|wc` on every script (README). Suites:
`plots_run1_novtx_dep_pe3` (present) and `plots_run1_novtx_wcdef_pe3` (WC-like:
photon TRUE E > 20 MeV with first deposit >= 3 cm inside the TPC, same-vertex
pairs < 20 deg merged, veto any mu KE >= 100 MeV incl. outside the TPC or any
primary e, FV = vertex >= 3 cm inside the box). Same reco selection, final
step (S5 flash PE ratio), 4.4e19 POT:

| | present | WC-like |
|---|---|---|
| true signal (in-FV / entering) | 805.7 (157.6 / 648.1) | 760.8 (153.0 / 607.9) |
| selected signal | 248.6 | 116.3 |
| efficiency: combined / in-FV / entering / 1g0X | 0.309 / 0.451 / 0.274 / 0.395 | 0.153 / 0.323 / 0.110 / 0.328 |
| purity (all signal / in-FV) | 0.406 / 0.116 | 0.190 / 0.081 |
| MC bkg / EXT / data / d/p | 242.2 / 122.0 / 623 / 1.02 | 374.5 / 122.0 / 623 / 1.02 |

Where the 132 selected events move (present signal -> WC background): entering
CC-mu (mu > 100 MeV outside the TPC, vetoed by WC) 55.5; second photon with
TRUE E > 20 MeV that deposits < 20 MeV -> ">=2 visible g" +32.5 and "entering
>=2 g" +10.3; FV/vertex-box reclassification -> "entering other" +20.3; muons
with KE >= 100 that do not deposit +10.4; "visible e" falls 8.0 -> 1.6 (Michel
electrons no longer veto). Six-path cells under the WC-like definition: FC
in-TPC 22.5 sig / 60.8 MC bkg (purity 0.22), FC entering 26.2 / 56.6,
no-vertex pass 41.1 / 130.5.

TRUE OPENING ANGLE (smallest angle between detectable photons, true momenta;
`trueOpenAngle` in the diagnostics, MC only). Present definition, final step,
all paths: of the selected ">=2 visible g" background 2% (1.2 of 60.9) have a
pair below 20 deg; of "entering >=2 g" 9% (3.8 of 41.8); CC numu with a second
photon 2%. The distribution peaks at 25-70 deg for entering >=2g and is broad
(20-150 deg) for in-FV >=2g: COLLINEAR PHOTONS ARE NOT WHAT MAKES THE
TWO-PHOTON BACKGROUND. Under the WC-like definition the same-vertex pairs
below 20 deg are merged into the signal by construction (2-3% of selected
signal events carry such a pair; ">=2 visible g" then has 0% below 20 deg).

## WireCell photon-like LEE definition with a PRIMARY tag (2026-09-30)

`single_photon_wc_lee_definition.py` (`plots_run1_novtx_dep_pe3/wc_lee_definition/`,
both Run-1 halves at 4.4e19 POT). WC definition: one reconstructable photon
(true E > 20 MeV, start >= 3 cm inside the TPC, same-vertex pairs < 20 deg
merged), no mu KE >= 100 MeV, no primary e, no vertex requirement. PRIMARY
tag = photon start within 1 cm of the true nu vertex (direct GENIE photon or
immediate primary pi0/eta decay; `--prim-dist`).

| variant | total | vtx inside TPC | vtx outside | CC | NC |
|---|---|---|---|---|---|
| nominal | 760.8 | 174.0 | 586.8 | 194.7 | 566.1 |
| nominal & the single shower is primary | 702.9 | 160.0 | 542.9 | 174.7 | 528.2 |
| primary-only counting (secondary photons ignored as X) | 717.4 | 164.6 | 552.8 | 179.1 | 538.3 |

Primary fraction of the nominal WC signal 0.92 (inside 0.92, outside 0.93);
the 58 non-primary events are photons from secondary pi0s (33) or from
mothers below the mcreco table threshold (24), starting a median 32 cm from
the vertex. Present-analysis signal that is primary by the same tag: 1g0X
0.97, 1g+X 0.88, entering no-mu 0.75, entering CC-mu 0.70. The primary tag
therefore moves the WC expectation by only 8%; the large levers between the
two definitions remain true-E vs deposited-E visibility (365 WC-only events
with deposited E < 20 MeV) and the 77% out-of-TPC vertex share.

### Paper rate check (Table I/III of arXiv:2502.06064: 247 selected 1g, eff 7.0%, 6.34e20 POT)

Ntuple prediction in the paper's five categories (FV = 3 cm inside the TPC box):

| category | @4.4e19 | @6.34e20 |
|---|---|---|
| out of FV 1g | 607.9 | 8758.6 |
| NC pi0 1g | 121.1 | 1745.3 |
| NC Delta / direct-photon 1g | 7.6 | 109.0 |
| NC other 1g | 0.9 | 13.6 |
| numu CC 1g (mu KE < 100) | 23.4 | 336.5 |
| in-FV total | 153.0 | 2204.5 |
| all five | 760.8 | 10963.1 |

Paper-implied total signal 247 / 0.070 = 3529 at 6.34e20 (244.9 at 4.4e19).
(a) If the 7.0% denominator were all five categories the ntuple over-predicts
by 3.1x. (b) If the denominator is the in-FV signal (the four in-FV
categories, the usual WC convention) and the 29.5% out-of-FV share of the
selected signal (73 events) is additive, the implied in-FV signal is 2488 at
6.34e20 = 172.6 at 4.4e19 vs the ntuple's 153.0 (ratio 0.89, within the
paper's 21% pre-constraint uncertainty), and the implied out-of-FV efficiency
is 0.83%. The paper's selected in-FV composition (NC pi0 82%, Delta 11%,
CC 5%, other 2%) vs the ntuple's pre-selection shares (79% / 5% / 15% / 1%)
is consistent with a ~2x higher efficiency for Delta photons and ~3x lower
for soft-muon CC 1g. Conclusion: the numbers are consistent once the
efficiency is read as in-FV; out-of-FV 1g dominates the true rate (80%) but
is selected at <1% by WC.

GOTCHA (mcreco bookkeeping): a primary pi0 is absent from the trueSimPart
table when the vertex is inside the TPC but present (Process = primary, at the
vertex) when the vertex is outside, so "mother not in table" is NOT a primary
criterion; the geometric tag is the one to use. Pre-tag summary kept as
`wc_lee_definition_noprimtag.txt`.

## Truth-path -> reco-path confusion table (2026-09-29)

Truth path = (true vertex in the WC FV) x (second true photon with deposited
E > 10 MeV = "fail") x (truth PC) x (truth dfw < 100 = entering-like); the
truth analogues of the six reco leaf paths. Rows are normalised; "no path" =
fails the selection. Adopted selection, all signal (cats 0-3),
`plots_run1_novtx_dep_pe3/paths6/path_tables.txt` (per-category blocks and a
by-event-type block with the MC backgrounds are in the file):

| truth path | N(w) | FC in-TPC | FC entering | vtx-pass PC | vtx-fail | novtx-pass | novtx-fail | no path |
|---|---|---|---|---|---|---|---|---|
| vtx, pass, FC, in-TPC | 50.8 | **0.339** | 0.037 | 0.006 | 0.017 | 0.047 | 0.002 | 0.553 |
| vtx, pass, FC, entering | 32.3 | 0.064 | **0.342** | 0.015 | 0.009 | 0.035 | 0.003 | 0.532 |
| vtx, pass, PC | 40.7 | 0.026 | 0.030 | **0.326** | 0.009 | 0.035 | 0.000 | 0.574 |
| vtx, fail | 33.8 | 0.179 | 0.076 | 0.048 | **0.120** | 0.039 | 0.011 | 0.527 |
| no-vtx, pass | 596.5 | 0.025 | 0.061 | 0.009 | 0.006 | **0.159** | 0.016 | 0.726 |
| no-vtx, fail | 51.7 | 0.015 | 0.070 | 0.011 | 0.011 | 0.135 | 0.026 | 0.733 |

Diagonal fractions 0.33-0.34 for the three vertex veto-pass truth paths; the
truth "vtx, fail" row mostly ends up in the FC in-TPC cell (0.18) rather than
the reco veto-fail cell (0.12), i.e. the 10-20 MeV second photon is usually
not reconstructed. Entering signal: 0.16 into the no-vertex pass cell, 0.09
into vertex cells, 0.73 no path. Per category the in-TPC diagonal is 0.275
(1g0X) / 0.396 (1g+X). Fraction of each true event type reaching NO path:
1g0X 0.60, 1g+X 0.52, entering 0.71 / 0.76, entering >=2g 0.82, >=2 visible g
0.91, CC nue 0.98, CC numu 0.995, entering CC 0.991, no visible g 0.996.

## FC / PC efficiencies against TRUTH containment (2026-09-28)

Truth definition (single_photon_selection.truth_tables, `pc_true`): any
charged track-like nu-origin particle (mu, pi+-, K+-, p, charged baryons;
primary or secondary) that deposits charge whose true START or END point is
< 10 cm from a TPC wall (or outside). This mirrors the reco criterion (any
nu-stream track prong start/end < 10 cm) except that the ntuple has only
particle endpoints, not trajectories. NOTE `trueSimPartContained` is a
DIFFERENT flag (end in the WireCell FV) and is not used.

Agreement on selected vertex veto-pass signal (adopted selection): 0.940
overall; 1.8% of truth-FC events are reco PC, 21.8% of truth-PC events are
reco FC (16% / 11% if the truth track floor is KE >= 20 / 50 MeV: the
disagreement is dominated by soft protons near a wall that leave no reco
track). Reco-FC selected signal is 92.9 truth-FC / 5.5 truth-PC; reco-PC is
1.7 / 19.7.

Efficiency into each reco cell, denominator = all true events of the row
(adopted selection, PE-ratio ON; `plots_run1_novtx_dep_pe3/paths5/path_tables.txt`):

| true signal, truth containment | N(w) | -> vtx-pass FC | -> vtx-pass PC | vtx-fail | novtx-pass | novtx-fail | any |
|---|---|---|---|---|---|---|---|
| 1g0X, FC | 49.0 | **0.303** | 0.008 | 0.031 | 0.048 | 0.008 | 0.397 |
| 1g0X, PC | 7.2 | 0.145 | **0.197** | 0.013 | 0.026 | 0.000 | 0.382 |
| 1g+X, FC | 60.9 | **0.425** | 0.012 | 0.043 | 0.036 | 0.003 | 0.520 |
| 1g+X, PC | 40.5 | 0.033 | **0.322** | 0.033 | 0.037 | 0.000 | 0.425 |
| entering no-mu, FC | 330.2 | 0.100 | 0.001 | 0.007 | 0.145 | 0.015 | 0.268 |
| entering no-mu, PC | 85.6 | 0.030 | 0.042 | 0.004 | 0.276 | 0.039 | 0.390 |
| entering CC-mu, FC | 199.4 | 0.095 | 0.001 | 0.005 | 0.119 | 0.008 | 0.228 |
| entering CC-mu, PC | 33.0 | 0.017 | 0.049 | 0.009 | 0.200 | 0.029 | 0.304 |

Without the PE-ratio step (`plots_run1_novtx_dep/paths5/`) the FC->FC and
PC->PC numbers are 0.316 / 0.211 (1g0X) and 0.435 / 0.327 (1g+X).

---
# LEGACY-SCALE RESULTS (2026-09-23, A_GAMMA = 0.0253) -- superseded
 Supersedes `CEW6_SINGLE_PHOTON_TABLES.md` (run-3 MC/EXT
against run-1 beam data). Same cew6 chain and both cew6 BDTs; what changed is
the samples: run-1 simulation and run-1 beam-off, so data and prediction now
come from the same run period, with the calibrated flash-gamma table
(mc,1 4.5024 / data,1 2.8607) making the flash chi2 mutually comparable.

## Samples (versions doc section 3c)

| role | ntuple(s) | events | normalization |
|---|---|---|---|
| MC | `$L/run1_bnboverlay_half{,2}/dlgen2_larformer_ntuple_bnbovl_run1_half{,2}.root` | 370,580 (after dropping the 49 shared rows) | POT 4.6533e20 -> x0.09456 to 4.4e19, times xsecWeight |
| EXT | `$L/run1_C1_extbnb_half{,2}/dlgen2_larformer_ntuple_extbnb_run1_half{,2}.root` | 219,154 | x**0.601** (measured, see below) |
| data | `$L/bnb5e19_run1_table/dlgen2_larformer_ntuple_bnb5e19_run1_table.root` | 176,302 | 4.4e19 POT, unit weight |

`$L=/cluster/tufts/wongjiradlab/larbys/data/larformer`. Every `--*-ntuple`
takes a comma-separated list: the halves are concatenated, POT summed and
de-duplicated by (run,subrun,event).

**No BDT-hygiene halves.** Both cew6 BDTs were trained on RUN-3 overlay and
RUN-3 EXT, so the entire run-1 MC and EXT are held out: `--ext-row-min 0`, no
`--mc-odd-only`. Verified in the ntuples: `showerCosmicScore` and
`showerNoVtxScore` both reproduce the cew6 models (the vertex-free branch
differs from the ep8 model for 99.2% of showers).

**No `--recal-gamma-*`**: recal3 is baked into `showerRecoE`. Scripts whose
recal default is non-None (`photon_bdt_study.py`) need the identity constants
`--recal-gamma-a 0.020101 --recal-gamma-b -15.49`.

## EXT normalization — measured, not the spill count

The spill-count ratio (bnb5e19 10,375,708 / both EXT halves 11,545,474 =
0.8987) over-predicts cosmics. Measuring the scale in the pure-cosmic
flash-chi2 sideband of the pre-chi2 selection (`ext_norm_sideband.py`,
scale = (data - MC nu) / EXT raw):

| sideband | data | MC nu | EXT raw | -> scale |
|---|---|---|---|---|
| log10 chi2 > 3.0 | 2492 | 436.2 | 3376.0 | 0.609 |
| log10 chi2 > 3.5 | 1892 | 348.3 | 2522.0 | 0.612 |
| log10 chi2 > 4.0 | 1320 | 263.4 | 1756.0 | 0.602 |
| log10 chi2 > 4.5 | 803 | 178.2 | 1091.0 | 0.573 |

0.601 is adopted. It is self-consistent: data/prediction is 1.00-1.03 at
EVERY flash-chi2 cut from log10 2.25 to open, whereas 0.8987 gives 0.75-0.93
and a slope in chi2. This is the same ~30% deficit the pi0 suite flagged for
the run-1 EXT spill count (versions doc section 3c NOTE). One EXT half:
double the scale (1.202).

## Cutflow (both halves, EXT x0.601, cosmic BDT >= 0.164, vertex-free >= 0.5)

Efficiency denominators (w @ 4.4e19): combined 882.4 | in-FV 137.9 | entering
744.5 (no-mu 466.3 + CC-mu 278.2) | strict 1g0X 48.8. Raw signal events 9,296
(the run-3 cew6 sample had 1,658).

flash chi2 < 316 (log10 2.5) — `plots_run1_novtx/`:

| step | sig | eff | effFV | effOut | eff1g0X | pur | MC bkg | EXT | pred | data | d/p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S1 nu slice | 635.9 | 0.721 | 0.721 | 0.721 | 0.786 | 0.009 | 12175 | 54272 | 67083 | 68985 | 1.03 |
| S2 1 photon | 366.9 | 0.416 | 0.577 | 0.386 | 0.562 | 0.046 | 2117 | 5567 | 8051 | 8522 | 1.06 |
| S3 no vis X | 329.5 | 0.373 | 0.521 | 0.346 | 0.522 | 0.070 | 974 | 3409 | 4712 | 4659 | 0.99 |
| S4 flash chi2 | 223.8 | 0.254 | 0.402 | 0.226 | 0.376 | 0.320 | 294 | 180 | 698 | 717 | 1.03 |

flash chi2 < 178 (log10 2.25, the eff x purity optimum) —
`plots_run1_novtx_chi2_225/`: final eff 0.220, in-FV 0.352, 1g0X 0.320,
purity **0.390**, MC bkg 232, EXT 71, pred 496, data 495, **d/p 1.00**.

Cut scan (EXT x0.601):

| log10 chi2 | eff | effFV | eff1g0X | purity | d/p |
|---|---|---|---|---|---|
| 2.00 | 0.177 | 0.280 | 0.250 | 0.446 | 0.91 |
| 2.25 | 0.220 | 0.352 | 0.320 | 0.390 | 1.00 |
| 2.50 | 0.254 | 0.402 | 0.376 | 0.320 | 1.03 |
| 2.75 | 0.280 | 0.444 | 0.436 | 0.247 | 1.03 |
| 3.00 | 0.298 | 0.457 | 0.448 | 0.183 | 1.01 |
| open | 0.352 | 0.501 | 0.496 | 0.080 | 1.01 |

Final composition at chi2 < 316 (weighted): sig 1g0X 18.3 | sig 1g+X 37.1 |
sig entering no-mu 114.3 | sig entering CC-mu 54.1 | bkg entering >=2g 61.9 |
entering CC 60.2 | entering other 11.4 | in-FV CC numu 57.0 | in-FV CC nue 9.6
| in-FV >=2 visible g 79.2 | no visible g 14.9 | EXT 180.3 | data 717.

## Vertex vs no-vertex samples (chi2 < 316, `plots_run1_novtx/paths/`)

| final step | VERTEX sample | NO-VERTEX sample |
|---|---|---|
| sig 1g0X (eff) | 15.9 (0.326) | 2.5 (0.050) |
| sig 1g+X (eff) | 34.2 (0.384) | 2.8 (0.032) |
| sig entering no-mu / CC-mu | 40.6 / 21.3 | 73.7 / 32.8 |
| total signal | 112.0 | 111.8 |
| MC bkg / EXT | 173.4 / 102.8 | 120.8 / 77.5 |
| purity (in-FV / entering) | 0.289 (0.129 / 0.160) | 0.360 (0.017 / 0.343) |
| data / pred | 1.10 | 0.93 |

The no-vertex sample stays ~95% entering by signal content, as on run 3.

## Run-1 vs the run-3 comparison it replaces

| final step, chi2 < 316 | run-3 cew6 fresh pair | **run-1 table-gamma** |
|---|---|---|
| eff combined | 0.218 | 0.254 |
| eff in-FV | 0.405 | 0.402 |
| eff 1g0X | 0.525 | 0.376 |
| purity | 0.342 | 0.320 |
| data / pred | 1.08 | **1.03** |
| raw signal events | 1,658 (odd half x2) | **9,296** |

The data/MC comparison is the real gain: 0.99-1.06 at every cut step, and
1.00-1.03 across the whole chi2 range, versus a 8-35% excess before. The
1g0X efficiency moves 0.525 -> 0.376, but the run-3 number rested on ~84 raw
events in the hygiene half; run 1 measures it on 516 raw 1g0X events, so the
lower value is the better-determined one (run-1 detector conditions also
differ: lower light yield, noisier electronics).
