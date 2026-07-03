#!/bin/bash

#SBATCH --job-name=shower_origin_inference
#SBATCH --output=shower_origin_inference.%j.%N.log
#SBATCH --mem-per-cpu=4000
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --partition=wongjiradlab,gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|p100" # Choose either A100 or P100
#SBATCH --error=shower_origin_inference.%j.%N.err

# set the location of your copy of the repo here
WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/

# location of the sl7 container here
container=/cluster/tufts/wongjiradlabnu/larbys/larbys-container/pointcept_cuml.sif

# setup singularity on the node
module load apptainer/1.2.4-suid

# run job script inside container
#apptainer exec --nv --bind /cluster:/cluster $container bash -c "cd ${WORKDIR} && source run_shower_origin_inference.sh"
apptainer exec --nv --bind /cluster:/cluster $container bash -c "cd ${WORKDIR} && source run_shower_origin_reco_inference.sh"

