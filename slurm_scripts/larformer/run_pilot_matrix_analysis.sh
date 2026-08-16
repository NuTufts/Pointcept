#!/bin/bash
#
# Stage-4 analysis of the pilot ntuple matrix: build the 8 nue tables
# (nue_cc_analysis.py) then run compare_pilot_matrix.py -> summary table of
# CC-1pi0 + nue-CC efficiency/purity per {chain} x {vertex mode} cell.
#
# Submit after the export hadds:  sbatch run_pilot_matrix_analysis.sh
# ---------------------------------------------------------------------------

#SBATCH --job-name=pilotcmp
#SBATCH --mem=24G
#SBATCH --cpus-per-task=2
#SBATCH --time=2:00:00
#SBATCH --partition=batch
#SBATCH --output=logs/pilot_matrix/analysis.%j.log
#SBATCH --error=logs/pilot_matrix/analysis.%j.err

set -eu

WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
container=/cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif
NTUP=${WORKDIR}/lartpc/larformer_reco/output/pilot_ntuples
TABDIR=${NTUP}/nue_tables

mkdir -p "${WORKDIR}/logs/pilot_matrix" "${TABDIR}"
module load apptainer 2>/dev/null || true

apptainer exec --bind /cluster:/cluster "${container}" bash -c "
  cd ${WORKDIR} && source setenv_pointcept_only.sh && \
  set -e
  for CELL in old_bnbnu_pred old_bnbnu_true old_nue_pred old_nue_true \
              new_bnbnu_pred new_bnbnu_true new_nue_pred new_nue_true; do
    if [ ! -f ${TABDIR}/nuetab_\${CELL}.npz ]; then
      echo \">>> nue table \${CELL}\"
      PYTHONPATH=./ python3 lartpc/larformer_analysis/physics/nue_cc/nue_cc_analysis.py \
        --ntuple ${NTUP}/dlgen2_pilot_\${CELL}.root \
        --out ${TABDIR}/nuetab_\${CELL}.npz
    fi
  done
  PYTHONPATH=./ python3 lartpc/larformer_analysis/physics/pilot_matrix/compare_pilot_matrix.py \
    --ntuple-dir ${NTUP} --nue-table-dir ${TABDIR} \
    --out ${NTUP}/summary_pilot_matrix.txt
"
echo "DONE pilot matrix analysis"
