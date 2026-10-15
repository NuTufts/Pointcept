#!/bin/bash
# Gun-sample twin of run_bdtpaths_suite.sh: the same 10 diagnostics cells on the isotropic 1-gamma gun.
#   bash slurm/run_isogamma_diag_suite.sh <PLOT_ROOT> [extra args]
ROOT=$1; shift
LAR=plots_run1_mbchain/mb_bdt_chain_noflash_novtx_lar.joblib
MB=plots_run1_mbchain/mb_bdt_chain_noflash_novtx.joblib
A="--novtx --novtx-bdt-min 0.5 --shower-bdt-min 0.164 --flashpe-ratio-max 3.0 --signal-def lar --veto2g-dcmax 100 --mb-chain-model $LAR --mb-chain-model-extra $MB $@"
E="--enter-cut 0.30 --enter-cut-0p 0.15"
cd "$(dirname "$0")/.."
J=""
J="$J $(PLOTS=$ROOT/diagnostics_all_selected sbatch --parsable slurm/submit_isogamma_diagnostics.sh $A)"
J="$J $(PLOTS=$ROOT/diagnostics_vertex_fail sbatch --parsable slurm/submit_isogamma_diagnostics.sh $A --path 0 --veto2g fail)"
J="$J $(PLOTS=$ROOT/diagnostics_vertex_pass sbatch --parsable slurm/submit_isogamma_diagnostics.sh $A --path 0 --veto2g pass)"
J="$J $(PLOTS=$ROOT/diagnostics_vertex_pass_ccnumu sbatch --parsable slurm/submit_isogamma_diagnostics.sh $A --path 0 --veto2g pass --chain-numu-max 0.25)"
J="$J $(PLOTS=$ROOT/diagnostics_vertex_pass_nonccnumu sbatch --parsable slurm/submit_isogamma_diagnostics.sh $A --path 0 --veto2g pass --chain-numu-min 0.25)"
J="$J $(PLOTS=$ROOT/diagnostics_vertex_pass_nonccnumu_entering sbatch --parsable slurm/submit_isogamma_diagnostics.sh $A --path 0 --veto2g pass --chain-numu-min 0.25 --enter ent $E)"
J="$J $(PLOTS=$ROOT/diagnostics_vertex_pass_nonccnumu_infv sbatch --parsable slurm/submit_isogamma_diagnostics.sh $A --path 0 --veto2g pass --chain-numu-min 0.25 --enter in $E)"
J="$J $(PLOTS=$ROOT/diagnostics_vertex_pass_nonccnumu_infv_0p sbatch --parsable slurm/submit_isogamma_diagnostics.sh $A --path 0 --veto2g pass --chain-numu-min 0.25 --enter in $E --protons 0p)"
J="$J $(PLOTS=$ROOT/diagnostics_vertex_pass_nonccnumu_infv_np sbatch --parsable slurm/submit_isogamma_diagnostics.sh $A --path 0 --veto2g pass --chain-numu-min 0.25 --enter in $E --protons np)"
J="$J $(PLOTS=$ROOT/diagnostics_novertex_fail sbatch --parsable slurm/submit_isogamma_diagnostics.sh $A --path 1 --veto2g fail)"
J="$J $(PLOTS=$ROOT/diagnostics_novertex_pass sbatch --parsable slurm/submit_isogamma_diagnostics.sh $A --path 1 --veto2g pass)"
echo $J
