#!/bin/bash
#SBATCH --job-name=stepA0_arr
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=6:00:00
#SBATCH --partition=batch
#SBATCH --output=logs/stepA0/arr.%A_%a.log
#SBATCH --error=logs/stepA0/arr.%A_%a.err
set -u
W=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
export FILENO=$(sed -n "${SLURM_ARRAY_TASK_ID}p" ${W}/lartpc/data_prep/uboone_official/inputlists/pi0sig_filenos.txt)
export INPUTLIST=${W}/lartpc/larformer_reco/inputlists/dlmerged_scale1500_resolved.txt
export OUTDIR=${W}/lartpc/larformer_reco/output/pilot_ntuples/larmatch_pilot
unset SLURM_ARRAY_TASK_ID
bash ${W}/lartpc/data_prep/uboone_official/run_stepA0_larmatch.sh
