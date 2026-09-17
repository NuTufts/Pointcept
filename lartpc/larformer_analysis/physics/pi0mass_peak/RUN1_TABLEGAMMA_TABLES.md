# Run-1 table-gamma pi0 tables (2026-09-17)

Inputs, procedure and script changes: `README.md`, section "Run-1 table-gamma
remake". Working point = cew6 talk WP (shower cosmicScore >= 0.164, event BDT
>= 0.21 with `ext_bdt_model_flashblind_cew6.joblib`, union muon finder KE > 50,
chi2 CC < 1e4 / NC < 1778, combined 3162). MC = run-1 overlay half minus the
segmenter-training filenos 1-350 (kept POT 2.1292e20 -> scale 0.2067 to 4.4e19,
times xsecWeight); data = bnb5e19 run-1 table-gamma; EXT = run-1 C1 half
x 1.7974 (10,375,708 / 5,772,737 spills). No classifier hygiene. Jobs
3775254 (tables), 3775586 (suite; 3775255 was the 94.4M-spill first pass).

**NORMALISATION IS OPEN.** `data_prep/README.md` gives 94,414,115 bnb5e19
spills; the July normalisation used 10,375,708 (consistent with 4.4e19 POT).
At 94.4M the cosmic-only prediction is 1.71M events for 176,302 beam triggers
and the EXT band sits 9x above the data in the pure-cosmic flash-chi2 sideband
(`plots_run1tg_*_spills94M/`), so everything below uses 10.4M. Even so, the
sideband (chi2 above the cut) prefers an EXT scale of 1.2-1.5 and the
trigger-level count prefers 1.31, versus 1.80 used: the EXT-half spill count
(25% file fraction x 23,090,946) may be ~25-30% low. Until the official gate
counts are confirmed, treat every EXT number here as an upper estimate
(purities as lower estimates, data/pred as lower estimates).

## pi0 step tables (`plots_run1tg_overlay_ts0164/step_tables.txt`)

    # run1tg_ts0164: shower score >= 0.164, event BDT >= 0.21, union muon finder (KE>50), chi2 CC<10000 / NC<1778; EXT scale 1.7974; hygiene: none (all rows)
    
    == COMBINED CC+NC, >=2 photons (signal = true CC+NC pi0; chi2 per-stream) | signal denominator (POT-wtd): 1583.8 ==
    cut                             signal    eff  purity  MC other      EXT    data   d/p
    all events (stream-tagged)      1583.8  1.000   0.007   37725.8 187853.3  176302  0.78
    reco vtx (nu-stream, FV)        1533.6  0.968   0.017   20805.2  66736.1   71469  0.80
    >=2 photons >20 MeV             1176.6  0.743   0.089    1781.8  10201.8   10088  0.77
    + shower score >=0.164          1160.1  0.732   0.295     757.0   2020.2    3264  0.83
    + mass ok (pair found)          1159.1  0.732   0.295     755.6   2014.8    3259  0.83
    + event BDT >=0.21              1137.9  0.718   0.342     664.3   1527.8    2775  0.83
    + flash chi2                    1076.2  0.679   0.616     416.1    253.4    1588  0.91
    
    == reco-CC stream, exactly 2 (signal = true CC pi0) | signal denominator (POT-wtd): 990.8 ==
    cut                             signal    eff  purity  MC other      EXT    data   d/p
    all events (stream-tagged)       784.2  0.792   0.014   16479.3  39525.8   50236  0.88
    reco vtx (nu-stream, FV)         781.6  0.789   0.022   12155.2  22032.1   30928  0.88
    exactly 2 photons >20 MeV        544.4  0.549   0.167     514.8   2201.8    2872  0.88
    + shower score >=0.164           545.5  0.551   0.439     254.2    442.2    1161  0.93
    + mass ok (pair found)           545.5  0.551   0.439     254.2    442.2    1161  0.93
    + event BDT >=0.21               541.4  0.546   0.518     221.9    282.2     990  0.95
    + flash chi2                     518.3  0.523   0.693     159.1     70.1     739  0.99
    
    == reco-NC stream, exactly 2 (signal = true NC pi0) | signal denominator (POT-wtd): 593.1 ==
    cut                             signal    eff  purity  MC other      EXT    data   d/p
    all events (stream-tagged)       554.4  0.935   0.003   21491.7 148327.4  126066  0.74
    reco vtx (nu-stream, FV)         517.7  0.873   0.010    8884.4  44704.0   40541  0.75
    exactly 2 photons >20 MeV        350.7  0.591   0.053     881.8   5329.2    4731  0.72
    + shower score >=0.164           349.7  0.590   0.193     393.1   1067.6    1370  0.76
    + mass ok (pair found)           348.6  0.588   0.193     391.8   1062.2    1365  0.76
    + event BDT >=0.21               337.7  0.569   0.228     343.1    799.8    1128  0.76
    + flash chi2                     319.5  0.539   0.483     220.9    120.4     543  0.82

## Near-peak (100-170 MeV) data vs prediction

| selection | signal | other MC | EXT | purity | data | pred | d/p | cew6 talk d/p (EXT-corrected) |
|---|---|---|---|---|---|---|---|---|
| CC+NC combined, >=2, chi2<3162 | 570.8 | 192.6 (incl. EXT) | - | 0.748 | 696 | 763.4 | 0.91 | 0.90 |
| CC+NC combined, eq2, chi2<3162 | 529.8 | 136.2 (incl. EXT) | - | 0.795 | 623 | 666.0 | 0.94 | 0.88 |
| reco-NC, >=2, chi2<1778 | - | MC 278.7 | 59.3 | - | 266 | 338.0 | 0.79 | 0.83 |
| reco-NC, eq2, chi2<1778 | - | MC 244.3 | 46.7 | - | 233 | 291.1 | 0.80 | 0.80 |
| reco-NC eq2, 0 charged-pi, true-NC-pi0 purity | 151.9 | 67.0 | 41.3 | 0.584 | | | | 0.65 (cew6 uncorrected) |

(cew6 "EXT-corrected": the talk overlays under-weighted EXT by 2x, see README
item 5; the NC rows above are recomputed with EXT x2 from the 3485634 log,
combined rows unchanged within rounding.)

## Flash-chi2 spectra, before (cew6) vs after (run 1)  (`plots_run1tg_flashchi2_ts0164/flashchi2_before_after.png`)

    stream                  ver  data<cut  pred<cut   d/p  data>cut  pred>cut   d/p
    reco-CC, exactly 2      old       711     795.0  0.89       249     256.9  0.97
    reco-CC, exactly 2      new       739     747.5  0.99       170     190.4  0.89
    reco-NC, exactly 2      old       517     600.8  0.86       544     576.2  0.94
    reco-NC, exactly 2      new       543     660.8  0.82       439     584.1  0.75
    CC+NC combined, >=2     old      1509    1668.7  0.90      1132    1155.2  0.98
    CC+NC combined, >=2     new      1576    1745.9  0.90       903    1126.9  0.80

Shape: with the calibrated gamma the data peak moves from log10 chi2 ~2.3
(cew6, right of the MC peak) to ~1.9, on top of the run-1 MC peak; the CC
peak in data is now sharper than MC. The high-chi2 sideband shape (EXT) now
tracks the data in both streams; its normalisation is the open item above.
Events without an in-window flash (2% of true 1pi0 signal, 12-19% of all
vertex-found events) now fail the chi2 step outright.

## Benchmark summary vs cew6 (eff @ purity after all cuts)

| selection | run-1 remake | cew6 talk |
|---|---|---|
| pi0 combined >=2 | 0.679 @ 0.616 | 0.687 @ 0.675 |
| CC eq2 | 0.523 @ 0.693 | 0.557 @ 0.758 |
| NC eq2 | 0.539 @ 0.483 | 0.540 @ 0.566 |
| SBND ordered cutflow final | 0.449 @ 0.838 (d/p 1.06) | 0.500 @ 0.869 (0.90) |
| SBND standing format (after flash cut) | 0.486 @ 0.751 | 0.534 @ 0.776 |

Efficiencies are on the run-1 overlay (independent sample, 7% of files removed)
and are within a few % of cew6 except the SBND flow (-5%), where the run-1
"exactly 1 primary mu > 143" step is 0.752 vs 0.779. Purities are lower mainly
because the run-1 EXT half contributes 2-2.5x more cosmic events than the
run-3 EXT did at the talk WP (combined final EXT 253 vs 109); part of that is
the open EXT normalisation.

## SBND-style suite

    cut                           signal    eff  purity  MC other     EXT   data   d/p
    all events                     524.6  1.000   0.002   38785.1187853.3 176302  0.78
    reco vtx in tight FV           512.9  0.978   0.009   14871.0 40735.5  45332  0.81
    exactly 1 primary mu >143      394.7  0.752   0.030    6640.8  6274.6  12729  0.96
    no primary cpi >25             375.8  0.716   0.032    5577.6  5769.5  11200  0.96
    exactly 2 photons              253.6  0.483   0.691      64.9    48.5    373  1.02
    m_gg in [30,300]               245.3  0.468   0.742      45.6    39.5    335  1.01
    flash chi2 < 1e4               235.7  0.449   0.838      36.7     9.0    297  1.06
    standing format: BEFORE flash cut pred 419 | signal 266 | data 424 | purity 0.635 | eff 0.508
                     AFTER  flash cut pred 340 | signal 255 | data 359 | purity 0.751 | eff 0.486

## Plot directories
- per-sample tables/plots: `plots_{mc,data,ext}_run1tg_ts0164/`
- data vs MC+EXT overlays (m_gg, p_pi0, purity curves, NC charged-pi veto): `plots_run1tg_overlay_ts0164/`
- flash-chi2 spectra + before/after: `plots_run1tg_flashchi2_ts0164/`
- SBND: `plots_run1tg_sbnd_ts0164/`, `plots_run1tg_sbnd_sbndflow_ts0164/`
- 94.4M-spill first pass (do not use): `*_spills94M/`
