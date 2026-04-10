#!/bin/bash

#SBATCH --job-name=larsonata_v6
#SBATCH --output=logs/larsonata_v6_new%j.%N.log
#SBATCH --mem-per-cpu=4000
#SBATCH --cpus-per-task=40
#SBATCH --time=15:00:00
#SBATCH --partition=wongjiradlab
#SBATCH --gres=gpu:p100:4
#SBATCH --error=logs/larsonata_v6_new%j.%N.err

# set the location of your copy of the repo here
WORKDIR=/cluster/tufts/wongjiradlabnu/vdasil01/Pointcept

# location of the sl7 container here
container=/cluster/tufts/wongjiradlabnu/larbys/larbys-container/pointcept_cuml.sif

# setup singularity on the node
module load apptainer/1.2.4-suid
#export WANDB_API_KEY="wandb_v1_JzdGdSIxN1z8lEAT8URJxXolzDo_lvZ"
# run job script inside container
apptainer exec --nv --bind /cluster:/cluster \
    --env WANDB_API_KEY="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" \
    --env WANDB_MODE=offline \
    $container bash -c "cd ${WORKDIR} && source run_lora_seg.sh"
#apptainer exec --nv --bind /cluster:/cluster $container bash -c "cd ${WORKDIR} && source run_lora_finetune_lartpc.sh"
