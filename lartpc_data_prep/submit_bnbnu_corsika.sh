#!/bin/bash

#SBATCH --job-name=dl2h5
#SBATCH --output=logs/bnbnu_corsika/gridlog_dl2h5.%j.%A_%a.%N.log
#SBATCH --mem-per-cpu=8000
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00:00
#SBATCH --partition=wongjiradlab,batch
#SBATCH --error=logs/bnbnu_corsika/griderr_dl2h5.%j.%A_%a.%N.err
#SBATCH --array=100-399

# set the location of your copy of the repo here
WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/lartpc_data_prep/

# location of the sl7 container here
container=/cluster/tufts/wongjiradlabnu/larbys/larbys-container/pointcept.sif

# setup singularity on the node
module load apptainer/1.2.4-suid

# run job script inside container
apptainer exec --bind /cluster:/cluster $container bash -c "cd ${WORKDIR} && source run_corsika_bnb_nu.sh"

