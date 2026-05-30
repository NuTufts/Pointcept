#!/bin/bash

#SBATCH --job-name=larformer
#SBATCH --mem=64G
#SBATCH --cpus-per-task=24
#SBATCH --time=2-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:2
#SBATCH --signal=USR1@600
#SBATCH --output=logs/larformer.v1.cascaded_crosslevelonly.%j.%N.log
#SBATCH --error=logs/larformer.v1.cascaded_crosslevelonly.%j.%N.err

set -u

# set the location of your copy of the repo here
WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/
CONFIG=${WORKDIR}/configs/lartpc/larformer-slicer-v1-cascaded-crosslevel.py
#SQASHFILE=/projects/u6jo/datasets/combined_pretrain-sonata-v7-extbnb-larmatch.sqsh

# location of the sl7 container here
#container=/projects/u6jo/containers/pointcept-sandbox/
container=/cluster/tufts/wongjiradlabnu/larbys/larbys-container/pointcept_cuml.sif

# setup singularity on the node
module load apptainer

# start job inside container
# apptainer exec --nv --bind /cluster:/cluster $container bash -c "cd ${WORKDIR} && \
#   cd /cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/ && \
#   source setenv_pointcept_only.sh && \
#   cd /cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept && \
#   python3 tools/train.py --config ${CONFIG} --num-gpus 2"

CHECKPOINT="${WORKDIR}/exp/larformer_slicer_v1_cascaded_crosslevelrefiner_mixedq_maskdn/model/model_last.pth"

# resume job script inside container
apptainer exec --nv --bind /cluster:/cluster $container bash -c "cd ${WORKDIR} && \
  cd /cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/ && \
  source setenv_pointcept_only.sh && \
  cd /cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept && \
  python3 tools/train.py --config ${CONFIG} --num-gpus 2 --options resume=True weight=${CHECKPOINT}"