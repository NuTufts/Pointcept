#!/bin/bash
#
# submit_stageA_convert_scale1500.sh — Stage A (convert) over 1500 BNB nu overlay
# files (~10% of the sample). CPU batch array, 15 files/task x 100 tasks = 1500.
#   sbatch submit_stageA_convert_scale1500.sh
# Output (~430 GB) -> larbys data area (see config). Both the converter out_of_range
# fix and per-event guard are in place, so files should convert fully.

#SBATCH --job-name=sp_conv_1500
#SBATCH --partition=batch
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16000
#SBATCH --array=0-99
#SBATCH --output=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/lartpc/larformer_analysis/physics/single_photon/slurm/logs/stageA_scale1500/conv.%A_%a.log
#SBATCH --error=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/lartpc/larformer_analysis/physics/single_photon/slurm/logs/stageA_scale1500/conv.%A_%a.err

LARFORMER_SCRIPTS=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/lartpc/data_prep/uboone_official
CONFIG=${LARFORMER_SCRIPTS}/larformer_configs/single_photon_scale1500.conf

module load apptainer/1.4.0 2>/dev/null || true
source ${LARFORMER_SCRIPTS}/run_larformer_wconfig.sh ${CONFIG}
