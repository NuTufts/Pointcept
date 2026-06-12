#!/bin/bash
#
# submit_stageB_cascade.sh — Pass-2 Stage B (LArFormer full cascade) on the
# CAPPED list of signal events (workdir/cascade_inputs.txt from
# select_cascade_events.py). GPU. One job over the whole capped list.
#
#   sbatch submit_stageB_cascade.sh
#
# GPU target: the `gpu` partition (A100), visible/submittable from login.pax.tufts.edu.
# (The `wongjiradlab` P100 partition currently has 0 nodes, so it is NOT usable; if
# it returns, --partition=wongjiradlab --gres=gpu:p100:1 is the priority fallback.)
# The cap is the point — keep cascade_inputs.txt small (e.g. <=200 events) so one
# GPU job finishes quickly. For larger caps, split the list and submit an array.

#SBATCH --job-name=sp_cascade
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8000
#SBATCH --time=8:00:00
#SBATCH --output=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/lartpc_data_prep/larformer_physics/single_photon/slurm/logs/stageB/casc.%A.log
#SBATCH --error=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/lartpc_data_prep/larformer_physics/single_photon/slurm/logs/stageB/casc.%A.err

SP=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/lartpc_data_prep/larformer_physics/single_photon
CONFIG=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/lartpc_data_prep/larformer_scripts/larformer_configs/single_photon_subsample.conf
SIF=/cluster/tufts/wongjiradlabnu/larbys/larbys-container/pointcept_cuml.sif

INPUT_LIST=${SP}/workdir/cascade_inputs.txt
OUTPUT_DIR=${SP}/workdir/stage3pred

module load apptainer/1.4.0 2>/dev/null || true

apptainer exec --nv --bind /cluster:/cluster ${SIF} \
    bash -c "source ${SP}/run_stageB_capped.sh ${CONFIG} ${INPUT_LIST} ${OUTPUT_DIR}"
