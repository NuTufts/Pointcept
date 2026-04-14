#!/bin/bash

#SBATCH --job-name=lm_extbnb
#SBATCH --output=logs/gridlog_lm_extbnb.%j.%A_%a.%N.log
#SBATCH --error=logs/griderr_lm_extbnb.%j.%A_%a.%N.err
#SBATCH --mem-per-cpu=8000
#SBATCH --cpus-per-task=1
#SBATCH --time=3-00:00:00
#SBATCH --partition=batch
#SBATCH --array=200-399

WORKDIR=/cluster/tufts/wongjiradlab/twongj01/mphys/Pointcept/lartpc_data_prep/extbnb_larmatch

# Create log directories
mkdir -p ${WORKDIR}/logs

# Load apptainer module (needed on bare node to call apptainer exec)
module load apptainer/1.2.4-suid

# Run the main job script on the bare node (NOT inside a container)
# The job script itself calls apptainer exec for the lantern and pointcept containers
source ${WORKDIR}/run_extbnb_larmatch.sh
