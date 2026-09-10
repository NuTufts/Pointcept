# Single-photon selection on v2_s1ep2p8cew6 (fresh-pair BDTs)

Updated 2026-09-10. Conventions: POT-weighted to 4.4e19, MC odd-event half
(weights x2), EXT rows >= 100,000 x 1.1818, beam data unit weight. No
`--recal-gamma-*` (recal3 is baked into `showerRecoE` on cew6).

## Score provenance in the current ntuples (verified 2026-09-10)

| branch | model in the ntuple | check |
|---|---|---|
| `showerCosmicScore` | **cew6** (`shower_cosmic_bdt_cew6.joblib`) | 81% of photon-shower scores changed vs `*_ep8score.root`; median 0.1125 -> 0.0850; the model's eff-0.97 WP is 0.164 |
| `showerNoVtxScore` | **ep8** (`shower_novtx_bdt.joblib` as of the export) | branch reproduces the ep8 model to 3e-8 and differs from the cew6 model for 99.6% of showers |

So only the vertex-attached BDT was promoted into the branch. The runs below
therefore score the vertex-free path analysis-side with
`--novtx-model .../shower_novtx_bdt_cew6.joblib` (identical to what the
exporter would bake). ACTION TAKEN: `export/data/shower_novtx_bdt.joblib` is
now the cew6 model (ep8 kept as `shower_novtx_bdt_ep8.joblib`), so the next
re-export bakes it and the flag becomes unnecessary.

Working point: cosmic BDT >= **0.164** (the cew6 model's eff-0.97 point;
0.192 was the ep8 model's), vertex-free BDT >= 0.5, flash chi2 < 316,
muon veto KE > 100 MeV (union finder).

## Final-step comparison

| | ep8 chain | cew6, ep8 BDTs | cew6, cew6 novtx only | **cew6 fresh pair** |
|---|---|---|---|---|
| cosmic-BDT WP | 0.192 | 0.192 | 0.192 | **0.164** |
| eff combined | 0.198 | 0.219 | 0.218 | **0.218** |
| eff in-FV | 0.351 | 0.414 | 0.405 | **0.405** |
| eff 1g0X | 0.475 | 0.525 | 0.525 | **0.525** |
| eff entering no-mu / CC-mu | 0.190 / 0.151 | 0.210 / 0.158 | 0.212 / 0.155 | **0.210 / 0.158** |
| purity | 0.350 | 0.307 | 0.334 | **0.342** |
| MC bkg / EXT | 233 / 97 | 301 / 142 | 296 / 93 | **291 / 85** |
| data / pred | 1.16 | 1.14 | 1.10 | **1.08** |
| plot dir | `plots_v2_s1ep2p8_novtx` | `plots_cew6_novtx` | `plots_cew6_novtx_bdtcew6` | `plots_cew6bdt_novtx` |

## Cutflow, cew6 fresh pair (`plots_cew6bdt_novtx/cutflow.txt`)

denominators (w): combined 894.0 | in-FV 116.4 | entering 777.6 (no-mu 484.8
+ CC-mu 292.8) | strict 1g0X 41.9

| step | sig | eff | effFV | effOut | eff1g0X | pur | MC bkg | EXT | pred | data | d/p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S1 nu slice | 627.8 | 0.702 | 0.775 | 0.691 | 0.850 | 0.010 | 11897 | 47513 | 60038 | 69039 | 1.15 |
| S2 1 photon | 357.4 | 0.400 | 0.622 | 0.367 | 0.650 | 0.049 | 2026 | 4854 | 7237 | 8522 | 1.18 |
| S3 no vis X | 313.2 | 0.350 | 0.568 | 0.318 | 0.625 | 0.074 | 950 | 2977 | 4240 | 4648 | 1.10 |
| S4 flash chi2 | 195.2 | 0.218 | 0.405 | 0.190 | 0.525 | 0.342 | 291 | 85 | 571 | 616 | 1.08 |

Final composition (weighted): sig 1g0X 22.0 | sig 1g+X 25.2 | sig entering
no-mu 101.7 | sig entering CC-mu 46.3 | bkg entering >=2g 69.2 | entering CC
41.2 | entering other 13.5 | in-FV CC numu 54.6 | in-FV CC nue 12.9 | in-FV
>=2 visible g 86.0 | no visible g 13.6 | EXT 85.1 | data 616.

Vertex-path-only baseline (`plots_cew6bdt_novtx_base`): eff 0.110, in-FV
0.369, 1g0X 0.425, purity 0.288, EXT 57.9, d/p 1.20.
A/B at the old WP 0.192 (`plots_cew6bdt_novtx_ts0192`): identical final
numbers (eff 0.218 / purity 0.342, EXT 82.7, d/p 1.06) — the chi2 cut absorbs
the difference; the WP only matters pre-chi2 (S3 EXT 2977 vs 2820).

## Vertex vs no-vertex samples (`plots_cew6bdt_novtx/paths/path_tables.txt`)

| final step | VERTEX sample | NO-VERTEX sample |
|---|---|---|
| sig 1g0X (eff) | 17.8 (0.425) | 4.2 (0.100) |
| sig 1g+X (eff) | 25.2 (0.338) | 0.0 (0.000) |
| sig entering no-mu / CC-mu | 34.6 / 18.9 | 67.1 / 27.4 |
| total signal | 96.5 | 98.7 |
| MC bkg / EXT | 163.6 / 52.0 | 127.5 / 33.1 |
| purity (in-FV / entering) | 0.309 (0.138 / 0.171) | 0.381 (0.016 / 0.365) |
| data / pred | 1.21 | 0.91 |

The no-vertex sample stays ~96% entering by signal content (in-FV purity
0.016) and its data/pred is now 0.91; the vertex sample carries all the in-FV
1g+X signal and sits at 1.21.

## Per-shower cosmic BDT on the cew6-scored branch
(`plots_cew6bdt_novtx/bdt_study/bdt_table.txt`)

At threshold 0.2: signal-photon pass 0.818 (was 0.866 with the ep8 score),
EXT photon rejection 0.772 (was 0.721), per-event EXT >=1-passing 0.112 (was
0.151). The retrained model is strictly better traded: ~5 points of
per-shower signal for ~5 points of EXT rejection and a 26% cut in EXT events.
