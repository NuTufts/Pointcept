# Polaris (A100-SXM4, ALCF) vs Tufts (A100-PCIe, driver 575.57.08): cross-platform conformance

Measured 2026-09-16 on the same 2,000 bnb5e19 events (production indices
0-1999) run through the identical container, code (commit 43a186d), checkpoints
and flags on both clusters (`lartpc/polaris/compare_platforms.py`; inputs in
`/cluster/tufts/wongjiradlab/larbys/data/larformer/polaris_it_tufts/{gpu/tests/throughput,from_polaris}`).
Tufts vs its own earlier cew6 production reproduces 200/200 events bit for bit,
so every difference below is the platform (driver + GPU SKU), cf.
`LArFormer_Reproducibility.md` §4.3.

## Cascade level (keypoint2 files)
| quantity | value |
|---|---|
| nu-file presence flips | 0 / 2000 |
| same n_particles | 77.0 +- 0.9 % |
| bit-identical partition | 47.6 +- 1.1 % |
| n_particles Polaris - Tufts | mean +0.005 (no bias); >= 2 in 7 % |
| slice point-set Jaccard | median 0.9998; < 0.9 in 3.5 %; < 0.5 in 1.0 % |
| nu vertex distance | median 0.5 cm; > 5 cm 24 %; > 20 cm 11 % |

## Analysis level (gen2ntuple, matched by run/subrun/event)
| event-level flip | rate |
|---|---|
| foundVertex | 3.80 +- 0.43 % (1270 vs 1280 found: no bias) |
| primaryVtxStream | 3.80 % |
| nTracks / nShowers / nProngs | 10.3 / 17.9 / 21.6 % |
| LArFormer PID multiset / LArPID multiset | 24.9 / 26.1 % |
| sum showerRecoE > 10 % | 14.4 % (mean shift +1 MeV, std 46 MeV) |
| max showerCosmicScore > 0.1 | 6.9 % |
| nuSliceFlashChi2 > 10 % | 3.3 % |
| recoNuE > 10 % | 12.9 % |
| vertex > 5 cm (both found) | 5.7 % (median 0.01 cm, 90th pct 0.33 cm) |

## Verdict
Polaris is a separate conformance family: event-level churn ~4 % (twice the
Hopper family's 1.9 % that was ruled non-conforming), prong-level ~20-25 %,
unbiased. Data reconstructed on Polaris against MC/EXT reconstructed at Tufts
would carry this as an uncontrolled per-event systematic. Decision 2026-09-15:
the run-1 table-gamma campaign (EXT half, overlay half, bnb5e19) is processed
entirely at Tufts. Polaris is usable only for a campaign processed wholly there
(data + MC + EXT), or after the divergence source (driver/library stack, see
the Hopper GEMM analysis) is understood.
