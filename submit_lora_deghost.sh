#!/bin/bash
#SBATCH --job-name=LoRa_v_deghoster
#SBATCH --output=logs/LoRa_v_deghoster_%j.%N.log
#SBATCH --mem-per-cpu=4000
#SBATCH --cpus-per-task=40
#SBATCH --time=16:00:00
#SBATCH --partition=wongjiradlab
#SBATCH --gres=gpu:p100:4
#SBATCH --error=logs/LoRa_v_deghoster_%j.%N.err

# set the location of your copy of the repo here
WORKDIR=/cluster/tufts/wongjiradlabnu/vdasil01/Pointcept

# location of the sl7 container here
container=/cluster/tufts/wongjiradlabnu/larbys/larbys-container/pointcept_cuml.sif

# setup singularity on the node
module load apptainer/1.2.4-suid

# run job script inside container
apptainer exec --nv --bind /cluster:/cluster \
    --env WANDB_API_KEY="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" \
    --env WANDB_MODE=offline \
    $container bash -c "cd ${WORKDIR} && source run_lora_finetune_lartpc_deghost.sh"
