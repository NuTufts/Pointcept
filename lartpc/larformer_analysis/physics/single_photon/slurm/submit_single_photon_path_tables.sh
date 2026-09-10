#!/bin/bash
# Vertex vs no-vertex sample tables (single_photon_path_tables.py).
#   sbatch slurm/submit_single_photon_path_tables.sh [extra args]
#   sbatch slurm/submit_single_photon_diagnostics.sh [extra args]
#SBATCH --job-name=sp_paths
#SBATCH --partition=batch
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=24000
#SBATCH --output=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept/lartpc/larformer_analysis/physics/single_photon/slurm/logs/sp_paths.%j.log
#SBATCH --error=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept/lartpc/larformer_analysis/physics/single_photon/slurm/logs/sp_paths.%j.err

K=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
SP=${K}/lartpc/larformer_analysis/physics/single_photon
D=/cluster/tufts/wongjiradlab/larbys/data/ub_on_tufts
MCD=$D/larformer_mcoverlay67k_s1ep2p8; EXTD=$D/larformer_extbnb200k_s1ep2p8_flash; DAD=$D/larformer_bnb5e19_s1ep2p8
MC=${MC:-$MCD/dlgen2_larformer_ntuple_mc_overlay_s1ep2p8_run3_novtx.root}
EXT=${EXT:-$EXTD/dlgen2_larformer_ntuple_extbnb200k_s1ep2p8f_novtx.root}
DATA=${DATA:-$DAD/dlgen2_larformer_ntuple_bnb5e19_s1ep2p8_novtx.root}
PLOTS=${PLOTS:-plots_v2_s1ep2p8_novtx/paths}

cd ${SP}
${K}/run_in_tufts_pointcept_container.sh python3 single_photon_path_tables.py \
    --mc-ntuple ${MC} --ext-ntuple ${EXT} --data-ntuple ${DATA} --ext-scale 1.1818 --plots ${PLOTS} "$@"
