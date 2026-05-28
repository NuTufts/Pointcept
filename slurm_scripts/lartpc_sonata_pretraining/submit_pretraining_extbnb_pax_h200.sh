#!/bin/bash

#SBATCH --job-name=larsonata
#SBATCH --mem-per-cpu=2000
#SBATCH --cpus-per-task=32
#SBATCH --time=2-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:2
#SBATCH --output=logs/larsonata.v7.extbnb-larmatch.lowerlr.%j.%N.log
#SBATCH --error=logs/larsonata.v7.extbnb-larmatch.lowerlr.%j.%N.err

# set the location of your copy of the repo here
WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept

# location of the sl7 container here
container=/cluster/tufts/wongjiradlabnu/larbys/larbys-container/pointcept_cuml.sif

# setup singularity on the node
module load apptainer

CONFIG="/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/configs/lartpc/pretrain-sonata-v7-extbnb-larmatch.py"

# resume checkpoints
#CHECKPOINT="/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/sonata/lartpc_v7_h200_extbnb_larmatch_lowerlr/model/epoch_6.pth"
#CHECKPOINT="/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/sonata/lartpc_v7_h200_extbnb_larmatch_lowerlr/model/epoch_18.pth"
#CHECKPOINT="/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/sonata/lartpc_v7_h200_extbnb_larmatch_lowerlr/model/epoch_24.pth"
CHECKPOINT="/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/sonata/lartpc_v7_h200_extbnb_larmatch_lowerlr/model/epoch_24.pth"


# run job script inside container
apptainer exec --nv --bind /cluster:/cluster $container bash -c "cd ${WORKDIR} && \
  cd /cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/ && \
  source setenv_pointcept_only.sh && \
  cd /cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept && \
  python3 tools/train.py --config ${CONFIG} --num-gpus 2 --options resume=True weight=${CHECKPOINT}"
