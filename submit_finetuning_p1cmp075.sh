#!/bin/bash

#SBATCH --job-name=larsonataft
#SBATCH --output=larsonata.finetune.v4.%j.%N.log
#SBATCH --mem-per-cpu=1000
#SBATCH --cpus-per-task=43
#SBATCH --time=6-00:00:00
#SBATCH --partition=wongjiradlab
#SBATCH --gres=gpu:p100:6
#SBATCH --error=larsonata.finetune.v4.%j.%N.err

# set the location of your copy of the repo here
WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/

# location of the sl7 container here
container=/cluster/tufts/wongjiradlabnu/larbys/larbys-container/pointcept_cuml.sif

# setup singularity on the node
module load apptainer/1.2.4-suid

# run job script inside container
apptainer exec --nv --bind /cluster:/cluster $container bash -c "cd ${WORKDIR} && source run_decoder_finetune_lartpcv2_noghost_p1cmp075.sh"

