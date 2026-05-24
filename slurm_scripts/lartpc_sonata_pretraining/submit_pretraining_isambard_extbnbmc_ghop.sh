#!/bin/bash

#SBATCH --job-name=larsonata
#SBATCH --mem=256G
#SBATCH --cpus-per-task=32
#SBATCH --time=1-00:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --output=logs/larsonata.v8.isambard_ghopper_extbnb_mc_combined_larmatch.%j.%N.log
#SBATCH --error=logs/larsonata.v8.isambard_ghopper_extbnb_mc_combined_larmatch.%j.%N.err

# set the location of your copy of the repo here
WORKDIR=/home/u6jo/twongj01.u6jo/ubpointcept/pointcept
CONFIG=${WORKDIR}/configs/lartpc/pretrain-sonata-v8-extbnb-mc-combined-larmatch.py
SQASHFILE=/projects/u6jo/datasets/combined_pretrain-sonata-v7-extbnb-larmatch.sqsh

# location of the sl7 container here
container=/projects/u6jo/containers/pointcept-sandbox/

# run job script inside container: new run
apptainer exec --nv --bind $SQASHFILE:/data:image-src=,ro $container bash -c "cd ${WORKDIR} && \
  source setenv.sh && \
  python3 tools/train.py --config ${CONFIG} --num-gpus 4"

