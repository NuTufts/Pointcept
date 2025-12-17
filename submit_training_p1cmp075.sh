#!/bin/bash

#SBATCH --job-name=larsonata
#SBATCH --output=larsonata.%j.%N.log
#SBATCH --mem-per-cpu=4000
#SBATCH --cpus-per-task=10
#SBATCH --time=6-00:00:00
#SBATCH --partition=wongjiradlab
#SBATCH --gres=gpu:p100:4
#SBATCH --error=larsonata.%j.%N.err

# set the location of your copy of the repo here
WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/

# location of the sl7 container here
container=/cluster/tufts/wongjiradlabnu/larbys/larbys-container/pointcept_cuml.sif

# setup singularity on the node
module load apptainer/1.2.4-suid

# run job script inside container
apptainer exec --nv --bind /cluster:/cluster $container bash -c "cd ${WORKDIR} && source run_training_p1cmp075.sh"

