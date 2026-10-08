# mcc9.10 run-4/5 ISOTROPIC SINGLE-PHOTON gun (CORSIKA cosmics, period 4 by run number), 248 files /
# 54,670 events, MC mode + truth sidecars, one batch. Sources are on tier1 (globus/) -> the stager
# symlinks instead of copying. ~6.3 MB/event -> ~350 GB merged_sp.
# TICK=forward: mcc9.10 larcv images are tick-forward. The first production (2026-10-07,
# default -tb) mirrored the drift coordinate (x_reco ~ -x_true - 80 cm) and was discarded.
isogamma_run45 /cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept/lartpc/data_prep/uboone_official/inputlists/mcc9.10_isotropic_one_gamma_run45.txt /cluster/tufts/wongjiradlab/larbys/data/larformer/mcc9.10_isotropic_one_gamma_run45/merged_sp mc /cluster/tufts/wongjiradlab/larbys/data/larformer/mcc9.10_isotropic_one_gamma_run45/truth_sidecar /cluster/tufts/wongjiradlab/larbys/data/mcc9_scratch/tier2_staging/isogamma_run45 /cluster/tufts/wongjiradlab/larbys/data/larformer/mcc9.10_isotropic_one_gamma_run45/markers 1 248 forward
