#!/bin/bash
# Per-shower cosmic-BDT performance study (photon_bdt_study.py) on batch.
#   sbatch slurm/submit_photon_bdt_study.sh
#   sbatch slurm/submit_single_photon_selection.sh [extra args]
#SBATCH --job-name=sp_bdt
#SBATCH --partition=batch
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=24000
#SBATCH --output=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept/lartpc/larformer_analysis/physics/single_photon/slurm/logs/sp_bdt.%j.log
#SBATCH --error=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept/lartpc/larformer_analysis/physics/single_photon/slurm/logs/sp_bdt.%j.err

K=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
SP=${K}/lartpc/larformer_analysis/physics/single_photon
D=/cluster/tufts/wongjiradlab/larbys/data/ub_on_tufts
MC=${MC:-$D/larformer_mcoverlay67k_s1ep2p8/dlgen2_larformer_ntuple_mc_overlay_s1ep2p8_run3.root}
EXT=${EXT:-$D/larformer_extbnb200k_s1ep2p8_flash/dlgen2_larformer_ntuple_extbnb200k_s1ep2p8f.root}
DATA=${DATA:-$D/larformer_bnb5e19_s1ep2p8/dlgen2_larformer_ntuple_bnb5e19_s1ep2p8.root}
PLOTS=${PLOTS:-plots_v2_s1ep2p8/bdt_study}
OUTDIR=${OUTDIR:-workdir}

cd ${SP}
${K}/run_in_tufts_pointcept_container.sh python3 photon_bdt_study.py \
    --mc-ntuple ${MC} --ext-ntuple ${EXT} --ext-scale 1.1818 \
    --plots ${PLOTS} "$@"
