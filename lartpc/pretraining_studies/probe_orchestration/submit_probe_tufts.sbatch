#!/bin/bash
# One linear-probe job on the Tufts GPU partition. Submitted by
# launch_probe_sweep.py as:
#   sbatch submit_probe_tufts.sbatch <config.py> <weight.pth> <save_path> [extra --options tokens...]
#
# mkdir -p logs BEFORE the first sbatch (SLURM opens --output before the
# script runs; missing logs/ kills the job instantly with exit code 53).
#
#SBATCH --job-name=p05probe
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100          # PIN one arch for the whole sweep (PI decision
                                   # 2026-07-16): batch 64 fits 40G and 80G variants;
                                   # do NOT mix archs (L40S/H100) within a comparison.
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16         # probe configs use num_worker=14
#SBATCH --mem=48G
#SBATCH --output=logs/p05probe.%j.log
#SBATCH --error=logs/p05probe.%j.err

set -u

REPO=${LOCAL_REPO:-/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/isambard_pointcept}
CONTAINER=${CONTAINER:-/cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif}

CONFIG=${1:?config}; WEIGHT=${2:?weight}; SAVE_PATH=${3:?save_path}; shift 3

module load modtree/deprecated
module load apptainer/1.2.4-suid

apptainer exec --nv --bind /cluster:/cluster "$CONTAINER" bash -c "cd ${REPO} && export PYTHONPATH=${REPO}:\$PYTHONPATH && exec python3 tools/train.py --config-file ${CONFIG} --num-gpus 1 --options weight=${WEIGHT} save_path=${SAVE_PATH} $*"
RC=$?
echo "probe exited rc=$RC"
[ $RC -eq 0 ] && touch "${REPO}/${SAVE_PATH}/.done"
exit $RC
