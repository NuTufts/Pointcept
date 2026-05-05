#!/bin/bash

#SBATCH --job-name=pointcept_lantern
#SBATCH --output=logs/pointcept_lantern/gridlog_bnbnu_coriska_pointcept_lantern.%j.%A_%a.%N.log
#SBATCH --error=logs/pointcept_lantern/griderr_bnbnu_coriska_pointcept_lantern.%j.%A_%a.%N.err
#SBATCH --mem-per-cpu=16000
#SBATCH --cpus-per-task=1
#SBATCH --time=6-00:00:00
#SBATCH --partition=batch
#SBATCH --array=1-49

WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/lartpc_data_prep/lantern_scripts

# Create log directories
mkdir -p ${WORKDIR}/logs/pointcept_lantern

# Load apptainer module (needed on bare node to call apptainer exec)
module load apptainer/1.2.4-suid

# Run the main job script on the bare node (NOT inside a container)
# The job script itself calls apptainer exec for the lantern and pointcept containers
source ${WORKDIR}/run_lantern_wconfig.sh ${WORKDIR}/lantern_configs/bnbnu_coriska.conf
