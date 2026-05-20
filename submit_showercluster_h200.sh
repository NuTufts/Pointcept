#!/bin/bash

#SBATCH --job-name=showercluster
#SBATCH --output=logs/showercluster.h200.%j.%N.log
#SBATCH --mem-per-cpu=2000
#SBATCH --cpus-per-task=32
#SBATCH --time=2-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:2
#SBATCH --error=logs/showercluster.h200.%j.%N.err

# set the location of your copy of the repo here
WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/

# location of the sl7 container here
container=/cluster/tufts/wongjiradlabnu/larbys/larbys-container/pointcept_cuml.sif

# setup singularity on the node
module load apptainer

# run job script inside container
#apptainer exec --nv --bind /cluster:/cluster $container bash -c "cd ${WORKDIR}/../ && \
#  source setenv_pointcept_only.sh && \
#  cd ${WORKDIR} && \
#  python tools/train.py --config configs/lartpc/shower-cluster-sonata-v1-h200.py --num-gpus 2"

# run job script inside container: resume


CHECKPOINT_FOLDER="/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/exp/shower_clustering/run2_h200/"
#resume_checkpoint="${CHECKPOINT_FOLDER}/model_epoch30.pth"
#resume_checkpoint="${CHECKPOINT_FOLDER}/model_epoch48.pth"
resume_checkpoint="${CHECKPOINT_FOLDER}/model_epoch60.pth"
apptainer exec --nv --bind /cluster:/cluster $container bash -c "cd ${WORKDIR}/../ && \
  source setenv_pointcept_only.sh && \
  cd ${WORKDIR} && \
  python tools/train.py --config configs/lartpc/shower-cluster-sonata-v1-h200.py --num-gpus 2 --options resume=True weight=${resume_checkpoint}"

