#!/bin/bash

#SBATCH --job-name=pointcept_lantern
#SBATCH --output=logs/pointcept_lantern/grid_bnb_nu_corsika_set2_prod2.%j.%A_%a.%N.log
#SBATCH --error=logs/pointcept_lantern/grid_bnb_nu_corsika_set2_prod2.%j.%A_%a.%N.err
#SBATCH --mem-per-cpu=8000
#SBATCH --cpus-per-task=2
#SBATCH --time=1-00:00:00
#SBATCH --partition=batch
#SBATCH --array=0-217

WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/lartpc/data_prep/training_data

# Create log directories
mkdir -p ${WORKDIR}/logs/pointcept_lantern

# Load apptainer module (needed on bare node to call apptainer exec)
# command for old cluster
module load apptainer/1.2.4-suid

# Run the main job script on the bare node (NOT inside a container)
# The job script itself calls apptainer exec for the lantern and pointcept containers
source ${WORKDIR}/run_lantern_wconfig.sh ${WORKDIR}/lantern_configs/bnb_nu_corsika_set2.conf
