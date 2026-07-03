#!/bin/bash

#SBATCH --job-name=bnbnu_chargedpiplus_pointcept_lantern
#SBATCH --output=logs/pointcept_lantern/gridlog_bnbnu_chargedpiplus_pointcept_lantern.%j.%A_%a.%N.log
#SBATCH --error=logs/pointcept_lantern/griderr_bnbnu_chargedpiplus_pointcept_lantern.%j.%A_%a.%N.err
#SBATCH --mem-per-cpu=8000
#SBATCH --cpus-per-task=3
#SBATCH --time=2-00:00:00
#SBATCH --partition=batch
#SBATCH --array=0-52

WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/lartpc/data_prep/training_data

# Create log directories
mkdir -p ${WORKDIR}/logs/pointcept_lantern

# Load apptainer module (needed on bare node to call apptainer exec)
module load apptainer/1.2.4-suid

# Run the main job script on the bare node (NOT inside a container)
# The job script itself calls apptainer exec for the lantern and pointcept containers
source ${WORKDIR}/run_lantern_wconfig.sh ${WORKDIR}/lantern_configs/bnbnu_chargedpiplus_corsika.conf
