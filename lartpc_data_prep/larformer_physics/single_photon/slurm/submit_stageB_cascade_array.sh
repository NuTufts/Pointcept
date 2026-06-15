#!/bin/bash
#
# submit_stageB_cascade_array.sh — Stage B (LArFormer cascade) over the capped
# ~3000 visible-photon events, PARALLELIZED as a GPU array (100 events/task).
# New-cluster A100s (gpu partition). For old-cluster P100s use the _p100 variant.
#   sbatch submit_stageB_cascade_array.sh
#
# Each task processes its 100-line slice of workdir_scale/cascade_inputs.txt and
# writes stage3pred_<name>.h5 into <DATADIR>/stage3pred/ (unique names, no clash).
# The Stage-3 checkpoint is pre-prefixed once (model_iter_182304.particle_segmenter.pth
# in stage3pred/), so tasks skip re-prefixing -> no race.

#SBATCH --job-name=sp_casc_arr
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8000
#SBATCH --time=8:00:00
#SBATCH --array=0-29
#SBATCH --output=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/lartpc_data_prep/larformer_physics/single_photon/slurm/logs/stageB_scale/casc.%A_%a.log
#SBATCH --error=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/lartpc_data_prep/larformer_physics/single_photon/slurm/logs/stageB_scale/casc.%A_%a.err

SP=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/lartpc_data_prep/larformer_physics/single_photon
CONFIG=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/lartpc_data_prep/larformer_scripts/larformer_configs/single_photon_scale1500.conf
SIF=/cluster/tufts/wongjiradlabnu/larbys/larbys-container/pointcept_cuml.sif
DATADIR=/cluster/tufts/wongjiradlab/larbys/data/larformer/mcc9_v29e_dl_run3b_bnb_nu_overlay

INLIST=${SP}/workdir_scale/cascade_inputs.txt
OUTDIR=${DATADIR}/stage3pred
SLICEDIR=${DATADIR}/stageB_slices
mkdir -p ${SLICEDIR} ${OUTDIR}

PER=100
start=$(( SLURM_ARRAY_TASK_ID * PER + 1 ))
end=$(( start + PER - 1 ))
SLICE=${SLICEDIR}/slice_${SLURM_ARRAY_TASK_ID}.txt
sed -n "${start},${end}p" ${INLIST} > ${SLICE}
echo "task ${SLURM_ARRAY_TASK_ID}: lines ${start}-${end}  ($(wc -l < ${SLICE}) events)"

module load apptainer/1.4.0 2>/dev/null || true
apptainer exec --nv --bind /cluster:/cluster ${SIF} \
    bash -c "source ${SP}/run_stageB_capped.sh ${CONFIG} ${SLICE} ${OUTDIR}"
