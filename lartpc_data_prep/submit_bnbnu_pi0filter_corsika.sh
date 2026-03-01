#!/bin/bash

#SBATCH --job-name=dl2h5
#SBATCH --output=logs/bnbnu_pi0filter_corsika/gridlog_dl2h5.%j.%A_%a.%N.log
#SBATCH --mem-per-cpu=8000
#SBATCH --cpus-per-task=1
#SBATCH --time=3-00:00:00
#SBATCH --partition=batch
#SBATCH --error=logs/bnbnu_pi0filter_corsika/griderr_dl2h5.%j.%A_%a.%N.err
#SBATCH --array=1-83

# set the location of your copy of the repo here
WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/lartpc_data_prep/

# location of the sl7 container here
container=/cluster/tufts/wongjiradlabnu/larbys/larbys-container/pointcept_cuml.sif

# setup singularity on the node
module load apptainer/1.2.4-suid

# run job script inside container
apptainer exec --bind /cluster:/cluster $container bash -c "cd ${WORKDIR} && source run_corsika_bnb_nu_pi0filter.sh"

