# mcc9.10 Run4a BNB nu OVERLAY (MCC9.10_Run4acd5_v10_04_07_20_BNB_nu_overlay_retuple_retuple_dlreco_4a, "mergeonly"
# list = the 119 merged_dlreco_*_merge_*.root files): 103,384 events (per file 151/900/1463), runs 18961-19750 ->
# period 4, MC + truth sidecars, one batch (sources on tier1 wongjiradlabnu/globus -> symlink staging).
# TICK=forward (mcc9.10 larcv images; verified in the smoke, see logs/data_prep/run4ovl4a_smoke.*.log).
# DATADIR on wongjiradlabnu (wongjiradlab had 432 GB free; merged_sp ~5.5 MB/event -> ~570 GB).
# This sample is the Tufts(stepA) -> Polaris(stepB GPU) -> Tufts(stepC) split-workflow demonstration:
# docs/reference/Polaris_Split_Workflow.md. The finisher packs merged_sp into squashfs instead of launching the chain.
# Survey: logs/data_prep/run4ovl4a_survey.5263874.log (NOTE: its totPOT sum is garbage (2.9e273) for this
# production -- check the sidecar POT before normalizing anything).
run4ovl_4a /cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept/lartpc/data_prep/uboone_official/inputlists/MCC9.10_Run4acd5_v10_04_07_20_BNB_nu_overlay_retuple_retuple_dlreco_4a_mergeonly.txt /cluster/tufts/wongjiradlabnu/nutufts/data/larformer/mcc9.10_run4_bnbnu_overlay_4a/merged_sp mc /cluster/tufts/wongjiradlabnu/nutufts/data/larformer/mcc9.10_run4_bnbnu_overlay_4a/truth_sidecar /cluster/tufts/wongjiradlabnu/nutufts/data/larformer/mcc9.10_run4_bnbnu_overlay_4a/staging /cluster/tufts/wongjiradlabnu/nutufts/data/larformer/mcc9.10_run4_bnbnu_overlay_4a/markers 1 119 forward
