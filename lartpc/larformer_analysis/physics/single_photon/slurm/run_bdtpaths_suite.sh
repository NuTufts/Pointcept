#!/bin/bash
# BDT-path suite (2026-10-08): geometric 2nd-photon veto + chain-BDT splits after vertex_pass,
# on the fully held-out samples (RESERVED overlay half, EXT second half x 1.202, bnb5e19 data).
#   bash slurm/run_bdtpaths_suite.sh <PLOT_ROOT> [extra args, e.g. --veto2g-dcmax 100]
ROOT=$1; shift
X="$@"
L=/cluster/tufts/wongjiradlab/larbys/data/larformer
export MC=$L/run1_bnboverlay_half2/dlgen2_larformer_ntuple_bnbovl_run1_half2.root
export EXT=$L/run1_C1_extbnb_half2/dlgen2_larformer_ntuple_extbnb_run1_half2.root
export EXT_SCALE=1.202
LAR=plots_run1_mbchain/mb_bdt_chain_noflash_novtx_lar.joblib
MB=plots_run1_mbchain/mb_bdt_chain_noflash_novtx.joblib
A="--novtx --novtx-bdt-min 0.5 --shower-bdt-min 0.164 --flashpe-ratio-max 3.0 --signal-def lar --veto2g-dcmax 100 --mb-chain-model $LAR --mb-chain-model-extra $MB $X"
E="--enter-cut 0.30 --enter-cut-0p 0.15"
cd "$(dirname "$0")/.."
J=""
J="$J $(PLOTS=$ROOT/tree sbatch --parsable slurm/submit_single_photon_bdt_paths.sh $A)"
J="$J $(PLOTS=$ROOT/diagnostics_vertex_fail sbatch --parsable slurm/submit_single_photon_diagnostics.sh $A --path 0 --veto2g fail)"
J="$J $(PLOTS=$ROOT/diagnostics_vertex_pass sbatch --parsable slurm/submit_single_photon_diagnostics.sh $A --path 0 --veto2g pass)"
J="$J $(PLOTS=$ROOT/diagnostics_vertex_pass_ccnumu sbatch --parsable slurm/submit_single_photon_diagnostics.sh $A --path 0 --veto2g pass --chain-numu-max 0.25)"
J="$J $(PLOTS=$ROOT/diagnostics_vertex_pass_nonccnumu sbatch --parsable slurm/submit_single_photon_diagnostics.sh $A --path 0 --veto2g pass --chain-numu-min 0.25)"
J="$J $(PLOTS=$ROOT/diagnostics_vertex_pass_nonccnumu_entering sbatch --parsable slurm/submit_single_photon_diagnostics.sh $A --path 0 --veto2g pass --chain-numu-min 0.25 --enter ent $E)"
J="$J $(PLOTS=$ROOT/diagnostics_vertex_pass_nonccnumu_infv sbatch --parsable slurm/submit_single_photon_diagnostics.sh $A --path 0 --veto2g pass --chain-numu-min 0.25 --enter in $E)"
J="$J $(PLOTS=$ROOT/diagnostics_vertex_pass_nonccnumu_infv_0p sbatch --parsable slurm/submit_single_photon_diagnostics.sh $A --path 0 --veto2g pass --chain-numu-min 0.25 --enter in $E --protons 0p)"
J="$J $(PLOTS=$ROOT/diagnostics_vertex_pass_nonccnumu_infv_np sbatch --parsable slurm/submit_single_photon_diagnostics.sh $A --path 0 --veto2g pass --chain-numu-min 0.25 --enter in $E --protons np)"
J="$J $(PLOTS=$ROOT/diagnostics_novertex_fail sbatch --parsable slurm/submit_single_photon_diagnostics.sh $A --path 1 --veto2g fail)"
J="$J $(PLOTS=$ROOT/diagnostics_novertex_pass sbatch --parsable slurm/submit_single_photon_diagnostics.sh $A --path 1 --veto2g pass)"
echo $J
