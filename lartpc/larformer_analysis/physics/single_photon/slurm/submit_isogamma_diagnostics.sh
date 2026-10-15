#!/bin/bash
# single_photon_diagnostics.py on the isotropic 1-gamma gun ntuple only (unit weights, no EXT / data).
#   PLOTS=<dir> sbatch slurm/submit_isogamma_diagnostics.sh [cell args]
#SBATCH --job-name=sp_isog_diag
#SBATCH --partition=batch
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=24000
#SBATCH --output=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept/lartpc/larformer_analysis/physics/single_photon/slurm/logs/sp_isog_diag.%j.log
#SBATCH --error=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept/lartpc/larformer_analysis/physics/single_photon/slurm/logs/sp_isog_diag.%j.err
K=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
SP=${K}/lartpc/larformer_analysis/physics/single_photon
L=/cluster/tufts/wongjiradlab/larbys/data/larformer
MC=${MC:-$L/mcc9.10_isotropic_one_gamma_run45/dlgen2_larformer_ntuple_isogamma_run45.root}
PLOTS=${PLOTS:-plots_isogamma/diagnostics}
cd ${SP}
${K}/run_in_tufts_pointcept_container.sh python3 single_photon_diagnostics.py --mc-ntuple ${MC} --unit-weight --plots ${PLOTS} "$@"
