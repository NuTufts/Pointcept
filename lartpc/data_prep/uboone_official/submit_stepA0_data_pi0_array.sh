#!/bin/bash
#SBATCH --job-name=stepA0_data
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
#SBATCH --partition=batch,preempt
#SBATCH --output=logs/stepA0/data_arr.%A_%a.log
#SBATCH --error=logs/stepA0/data_arr.%A_%a.err
# stepA0 larmatch over the bnb5e19 CC1pi0 flash-cut candidates' dlmerged
# files (S2 of DOMAIN_SHIFT_MEASUREMENT_PLAN.md). Array index -> line of
# data_pi0_filenos.txt -> fileno (1-based line of the bnb5e19 dlmerged
# list). Data conventions = script defaults (ADC "wire", -tb).
set -u
W=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
export FILENO=$(sed -n "${SLURM_ARRAY_TASK_ID}p" ${W}/lartpc/data_prep/uboone_official/inputlists/data_pi0_filenos.txt)
export INPUTLIST=${W}/lartpc/data_prep/uboone_official/inputlists/mcc9_v28_wctagger_bnb5e19.txt
export OUTDIR=${W}/lartpc/larformer_reco/output/pilot_ntuples/larmatch_data_pi0
unset SLURM_ARRAY_TASK_ID
bash ${W}/lartpc/data_prep/uboone_official/run_stepA0_larmatch.sh
