#!/bin/bash
# Diagnostic distributions (single_photon_diagnostics.py) for the selected
# single-photon events; needs the cascade dirs for flash PE.
#   sbatch slurm/submit_single_photon_diagnostics.sh [extra args]
#SBATCH --job-name=sp_diag
#SBATCH --partition=batch
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=24000
#SBATCH --output=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept/lartpc/larformer_analysis/physics/single_photon/slurm/logs/sp_diag.%j.log
#SBATCH --error=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept/lartpc/larformer_analysis/physics/single_photon/slurm/logs/sp_diag.%j.err

K=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
SP=${K}/lartpc/larformer_analysis/physics/single_photon
D=/cluster/tufts/wongjiradlab/larbys/data/ub_on_tufts
MCD=${MCD:-$D/larformer_mcoverlay67k_s1ep2p8}; EXTD=${EXTD:-$D/larformer_extbnb200k_s1ep2p8_flash}; DAD=${DAD:-$D/larformer_bnb5e19_s1ep2p8}
MC=${MC:-$MCD/dlgen2_larformer_ntuple_mc_overlay_s1ep2p8_run3_novtx.root}
EXT=${EXT:-$EXTD/dlgen2_larformer_ntuple_extbnb200k_s1ep2p8f_novtx.root}
DATA=${DATA:-$DAD/dlgen2_larformer_ntuple_bnb5e19_s1ep2p8_novtx.root}
PLOTS=${PLOTS:-plots_v2_s1ep2p8_novtx/diagnostics}

cd ${SP}
${K}/run_in_tufts_pointcept_container.sh python3 single_photon_diagnostics.py \
    --mc-ntuple ${MC} --ext-ntuple ${EXT} --data-ntuple ${DATA} \
    --mc-cascade $MCD/keypoint2_streams --ext-cascade $EXTD/keypoint2_streams \
    --data-cascade $DAD/keypoint2_streams --ext-scale 1.1818 --plots ${PLOTS} "$@"
