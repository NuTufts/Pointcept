#!/bin/bash
# One feature-extraction job on the Tufts GPU partition. Usage:
#   cd lartpc/pretraining_studies/domain_shift && mkdir -p logs
#   sbatch submit_extract_tufts.sbatch <extract_features.py args...>
# All arguments are passed through to extract_features.py.
#
#SBATCH --job-name=dsextract
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100          # flash_attn needs sm80+; a100 matches the
                                   # probe-sweep pin (do not use p100)
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/dsextract.%j.log
#SBATCH --error=logs/dsextract.%j.err

set -u

REPO=${LOCAL_REPO:-/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/isambard_pointcept}
CONTAINER=${CONTAINER:-/cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif}

module load modtree/deprecated
module load apptainer/1.2.4-suid

apptainer exec --nv --bind /cluster:/cluster "$CONTAINER" bash -c \
  "cd ${REPO} && export PYTHONPATH=${REPO}:\${PYTHONPATH:-} && \
   exec python3 lartpc/pretraining_studies/domain_shift/extract_features.py $*"
