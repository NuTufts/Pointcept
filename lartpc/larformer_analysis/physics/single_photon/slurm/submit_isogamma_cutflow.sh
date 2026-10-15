#!/bin/bash
# Isotropic 1-gamma gun through the single-photon cutflow / BDT-path tree (single_photon_isogamma_cutflow.py).
#   sbatch slurm/submit_isogamma_cutflow.sh [extra args]
#SBATCH --job-name=sp_isogamma
#SBATCH --partition=batch
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=24000
#SBATCH --output=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept/lartpc/larformer_analysis/physics/single_photon/slurm/logs/sp_isogamma.%j.log
#SBATCH --error=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept/lartpc/larformer_analysis/physics/single_photon/slurm/logs/sp_isogamma.%j.err
K=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
SP=${K}/lartpc/larformer_analysis/physics/single_photon
L=/cluster/tufts/wongjiradlab/larbys/data/larformer
MC=${MC:-$L/mcc9.10_isotropic_one_gamma_run45/dlgen2_larformer_ntuple_isogamma_run45.root}
PLOTS=${PLOTS:-plots_isogamma/cutflow}
SCRIPT=${SCRIPT:-single_photon_isogamma_cutflow.py}
cd ${SP}
${K}/run_in_tufts_pointcept_container.sh python3 ${SCRIPT} --mc-ntuple ${MC} --plots ${PLOTS} "$@"
