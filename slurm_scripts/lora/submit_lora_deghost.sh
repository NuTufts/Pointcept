#!/bin/bash
#SBATCH --job-name=LoRa_v_deghoster
#SBATCH --output=logs/LoRa_v_deghoster_%j.%N.log
#SBATCH --mem-per-cpu=2000
#SBATCH --cpus-per-task=32
#SBATCH --time=2-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:2
#SBATCH --error=logs/LoRa_v_deghoster_%j.%N.err

# set the location of your copy of the repo here
WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept

# location of the sl7 container here
container=/cluster/tufts/wongjiradlabnu/larbys/larbys-container/pointcept_cuml.sif

# setup singularity on the node
#module load apptainer/1.2.4-suid
module load apptainer

# run job script inside container
apptainer exec --nv --bind /cluster:/cluster \
    $container bash -c "cd ${WORKDIR} && source run_lora_finetune_lartpc_deghost.sh"
