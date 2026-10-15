#!/bin/bash
# Expected signal under the WireCell photon-like LEE definition (arXiv:2502.06064)
# from the run-1 BNB nu overlay ntuples (single_photon_wc_lee_definition.py).
#   sbatch slurm/submit_single_photon_wc_lee.sh [extra args]
#SBATCH --job-name=sp_wclee
#SBATCH --partition=batch
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=24000
#SBATCH --output=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept/lartpc/larformer_analysis/physics/single_photon/slurm/logs/sp_wclee.%j.log
#SBATCH --error=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept/lartpc/larformer_analysis/physics/single_photon/slurm/logs/sp_wclee.%j.err

K=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
SP=${K}/lartpc/larformer_analysis/physics/single_photon
L=/cluster/tufts/wongjiradlab/larbys/data/larformer
MC=${MC:-$L/run1_bnboverlay_half/dlgen2_larformer_ntuple_bnbovl_run1_half.root,$L/run1_bnboverlay_half2/dlgen2_larformer_ntuple_bnbovl_run1_half2.root}
PLOTS=${PLOTS:-plots_run1_novtx_dep_pe3/wc_lee_definition}

cd ${SP}
${K}/run_in_tufts_pointcept_container.sh python3 single_photon_wc_lee_definition.py \
    --mc-ntuple ${MC} --pot 4.4e19 --plots ${PLOTS} "$@"
