#!/bin/bash
#
# submit_stageA_convert.sh — Pass-2 Stage A (convert) SLURM array for the
# single-photon subsample. CPU-only, fanned out wide on the batch partition
# (one array task per input file). Cheap; scales to the full sample.
#
#   sbatch submit_stageA_convert.sh
#
# Array sizing: the config has stride=1, so array task K processes input-list
# line K+1. The subsample list (signal_files_subsample.txt) currently has ~20
# files -> --array=0-19. For the FULL sample (~14k signal files), bump stride in
# the config and/or widen --array (e.g. stride=140 + --array=0-99 = 14000 lines,
# 100 parallel tasks). SLURM caps array size ~1000-10000, so use stride to cover.

#SBATCH --job-name=sp_convert
#SBATCH --partition=batch
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8000
#SBATCH --array=0-19
#SBATCH --output=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/lartpc_data_prep/larformer_physics/single_photon/slurm/logs/stageA/conv.%A_%a.log
#SBATCH --error=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/lartpc_data_prep/larformer_physics/single_photon/slurm/logs/stageA/conv.%A_%a.err

LARFORMER_SCRIPTS=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/lartpc_data_prep/larformer_scripts
CONFIG=${LARFORMER_SCRIPTS}/larformer_configs/single_photon_subsample.conf

# apptainer is needed on the bare node; the driver re-execs into the pointcept
# container itself.
module load apptainer/1.4.0 2>/dev/null || true

# Driver: Stage A only (RUN_CASCADE=0 in the config). SLURM_ARRAY_TASK_ID selects
# the input line. Outputs are copied to OUTPUT_DIR/<lineno/1000>/<lineno/100>/.
source ${LARFORMER_SCRIPTS}/run_larformer_wconfig.sh ${CONFIG}
