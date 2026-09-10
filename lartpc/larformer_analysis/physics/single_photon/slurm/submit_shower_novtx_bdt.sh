#!/bin/bash
# Train the vertex-free per-shower cosmic BDT (shower_novtx_bdt.py) on the
# novtx MC + EXT ntuples and save the exporter model.
#   sbatch slurm/submit_shower_novtx_bdt.sh [extra args]
#SBATCH --job-name=sp_novtxbdt
#SBATCH --partition=batch
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8000
#SBATCH --output=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept/lartpc/larformer_analysis/physics/single_photon/slurm/logs/sp_novtxbdt.%j.log
#SBATCH --error=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept/lartpc/larformer_analysis/physics/single_photon/slurm/logs/sp_novtxbdt.%j.err

K=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
SP=${K}/lartpc/larformer_analysis/physics/single_photon
D=/cluster/tufts/wongjiradlab/larbys/data/ub_on_tufts
MC=${MC:-$D/larformer_mcoverlay67k_s1ep2p8/dlgen2_larformer_ntuple_mc_overlay_s1ep2p8_run3_novtx.root}
EXT=${EXT:-$D/larformer_extbnb200k_s1ep2p8_flash/dlgen2_larformer_ntuple_extbnb200k_s1ep2p8f_novtx.root}
PLOTS=${PLOTS:-plots_v2_s1ep2p8/novtx_bdt}
MODEL=${MODEL:-${K}/lartpc/larformer_reco/export/data/shower_novtx_bdt.joblib}

cd ${SP}
${K}/run_in_tufts_pointcept_container.sh python3 shower_novtx_bdt.py \
    --mc-ntuple ${MC} --ext-ntuple ${EXT} --plots ${PLOTS} --save-model ${MODEL} "$@"
