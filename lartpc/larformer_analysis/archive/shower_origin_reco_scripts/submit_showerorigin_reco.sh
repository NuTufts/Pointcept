#!/bin/bash

#SBATCH --job-name=shwreco
#SBATCH --output=logs/shower_origin_reco/gridlog_shwreco.%j.%A_%a.%N.log
#SBATCH --error=logs/shower_origin_reco/griderr_shwreco.%j.%A_%a.%N.err
#SBATCH --mem-per-cpu=8000
#SBATCH --cpus-per-task=1
#SBATCH --time=3-00:00:00
#SBATCH --partition=batch
#SBATCH --array=800-831

WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/lartpc/larformer_analysis/archive/shower_origin_reco_scripts

# Create log directories
mkdir -p ${WORKDIR}/logs/shower_origin_reco

# Load apptainer module (needed on bare node to call apptainer exec)
module load apptainer/1.2.4-suid

# Run the main job script on the bare node (NOT inside a container)
# The job script itself calls apptainer exec for the lantern and pointcept containers
source ${WORKDIR}/run_showerorigin_reco.sh
