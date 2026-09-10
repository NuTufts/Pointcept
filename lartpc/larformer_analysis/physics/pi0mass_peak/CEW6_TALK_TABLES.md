# cew6 talk tables (v2_s1ep2p8cew6, FRESH classifier pair)

FINAL talk working point (2026-09-11): shower cosmicScore >= 0.164
(shower_cosmic_bdt_cew6, eff-0.97), event BDT >= 0.21
(ext_bdt_model_flashblind_cew6), union muon finder, chi2 CC<1e4/NC<1778
(combined 3162). Ntuples PROMOTED with cew6-model showerCosmicScore
(ep8-score versions archived as *_ep8score.root). recal3 BAKED (identity
constants a=0.020101 b=-15.49 where recal defaults exist). Hygiene:
odd-event x2 signal estimator; EXT rows>=100k AND odd x2.3636.
Sources: job 3485634 (suite), 3485597-3485602 (re-export).
Stale-pair comparators: earlier version of this file / SLICER_RETRAIN_PLAN.

## pi0 step tables

    == COMBINED CC+NC, >=2 photons (signal = true CC+NC pi0; chi2 per-stream 1e4/1778) | signal denominator (POT-wtd, odd x2): 1655.1 ==
    cut                             signal    eff  purity  MC other     EXT
    all events (stream-tagged)      1655.1  1.000   0.011   37938.1117709.6
    reco vtx (nu-stream, FV)        1589.0  0.960   0.025   20723.5 41750.6
    >=2 photons >20 MeV             1255.7  0.759   0.137    1772.0  6161.9
    + shower score >=0.164          1243.1  0.751   0.365     805.2  1356.7
    + mass ok (pair found)          1239.9  0.749   0.365     803.6  1354.3
    + event BDT >=0.21              1224.2  0.740   0.414     706.7  1025.8
    + flash chi2                    1137.2  0.687   0.675     439.9   108.7
    
    == reco-CC stream, exactly 2 (signal = true CC pi0; chi2<1e4) | signal denominator (POT-wtd, odd x2): 1076.4 ==
    cut                             signal    eff  purity  MC other     EXT
    all events (stream-tagged)       860.6  0.800   0.020   16807.4 25404.0
    reco vtx (nu-stream, FV)         857.4  0.797   0.032   12218.0 13952.3
    exactly 2 photons >20 MeV        629.9  0.585   0.251     508.6  1368.5
    + shower score >=0.164           637.2  0.592   0.527     261.3   309.6
    + mass ok (pair found)           637.2  0.592   0.527     261.3   309.6
    + event BDT >=0.21               634.0  0.589   0.597     215.8   212.7
    + flash chi2                     599.5  0.557   0.758     162.5    28.4
    
    == reco-NC stream, exactly 2 (signal = true NC pi0; chi2<1778) | signal denominator (POT-wtd, odd x2): 578.7 ==
    cut                             signal    eff  purity  MC other     EXT
    all events (stream-tagged)       547.3  0.946   0.005   21367.8 92305.7
    reco vtx (nu-stream, FV)         498.0  0.861   0.013    8730.5 27798.3
    exactly 2 photons >20 MeV        349.1  0.603   0.079     894.9  3188.5
    + shower score >=0.164           349.1  0.603   0.230     420.4   749.3
    + mass ok (pair found)           346.0  0.598   0.229     418.8   746.9
    + event BDT >=0.21               338.6  0.585   0.278     319.3   560.2
    + flash chi2                     312.4  0.540   0.566     185.0    54.4

Fresh-vs-stale pair: big gains at INTERMEDIATE stages (combined purity
after shower score 0.283 -> 0.365; pre-chi2 EXT -38%), final post-chi2
selections statistically unchanged (chi2 was already removing the same
events). => the fresh pair matters most where chi2 is loose/absent
(single-photon, no-vertex streams). data/pred unchanged (~0.85-0.90):
the dip is NOT a stale-classifier effect.

## SBND-ordered cutflow

    >>> true SBND signal (POT-wtd): 561.3 (raw 1068)
    
    cut                           signal    eff  purity  MC other     EXT   data   d/p
    all events                     561.3  1.000   0.004   39006.8118180.0 176302  1.12
    reco vtx in tight FV           552.4  0.984   0.014   14743.3 25409.9  45318  1.11
    exactly 1 primary mu >143      437.2  0.779   0.039    6800.6  4032.3  12754  1.13
    no primary cpi >25             420.8  0.750   0.043    5681.3  3740.4  11217  1.14
    exactly 2 photons              302.9  0.540   0.735      70.3    39.0    374  0.91
    m_gg in [30,300]               299.2  0.533   0.801      46.0    28.4    336  0.90
    flash chi2 < 1e4               280.6  0.500   0.869      41.0     1.2    292  0.90
    mgg_sbndflow_2gamma.png: in-window signal 299.2 | other 74.4 | purity 0.801
    mgg_sbndflow_final.png: in-window signal 280.6 | other 42.1 | purity 0.869
    >>> plots -> /cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept/lartpc/larformer_analysis/physics/pi0mass_peak/plots_cew6_sbnd_sbndflow_cew6_ts0164
    >>> muon finder: union
    >>> per-shower cosmic-BDT cut: score >= 0.164
    >>> showerRecoE recal: gamma_a=0.020101 e_a=None
    >>> loading MC ...
    >>> loading data ...
    >>> loading EXT ...
    
    == SBND CC 1pi0 selection ==
      [BEFORE flash cut] MC pred 494 | signal 319 | data 425 | purity 0.646 | efficiency 0.569 (true signal 561)
      [AFTER flash cut ] MC pred 386 | signal 300 | data 352 | purity 0.776 | efficiency 0.534 (true signal 561)
    >>> plots -> /cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept/lartpc/larformer_analysis/physics/pi0mass_peak/plots_cew6_sbnd_cew6_ts0164
    DONE_CEW6_FRESH

Standing-format SBND: 0.534 eff @ 0.776 purity (after flash cut).

## Plot dirs (fresh pair)
- pi0 overlays: plots_cew6_overlay_cew6_ts0164/
- flashchi2 spectra: plots_cew6_flashchi2_cew6_ts0164/
- SBND: plots_cew6_sbnd_cew6_ts0164/, plots_cew6_sbnd_sbndflow_cew6_ts0164/
- classifier training: plots_cew6_sbdt_train/, plots_cew6_extbdt_train/
Stale-pair versions (comparators): *_ts0075* tables and plots dirs.
