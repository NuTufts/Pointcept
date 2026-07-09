#!/bin/bash
#
# Sharded truth-sidecar extraction: each array task runs
# export/extract_truth_sidecar.py over a contiguous chunk of the resolved
# dlmerged list, writing truth_fileno<NNNNN>.h5 per source file (fileno =
# 1-based line number in the list = the merged_sp fileno tag).
#
# Submit:
#   NSHARDS=50 sbatch --array=0-49 submit_truth_sidecar_shard.sh
# ---------------------------------------------------------------------------
#SBATCH --job-name=truthsc
#SBATCH --output=logs/export/truthsc.%A_%a.%N.log
#SBATCH --error=logs/export/truthsc.%A_%a.%N.err
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=8:00:00
#SBATCH --partition=batch
#SBATCH --array=0-49

set -eu
NSHARDS=${NSHARDS:-50}
WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
container=/cluster/tufts/wongjiradlabnu/larbys/larbys-container/pointcept_cuml.sif
UBDL_DIR=${UBDL_DIR:-/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/ubdl}
RECODIR=${WORKDIR}/lartpc/larformer_reco

INPUT_LIST=${INPUT_LIST:-${RECODIR}/inputlists/dlmerged_scale1500_resolved.txt}
OUTPUT_DIR=${OUTPUT_DIR:-${RECODIR}/output/mcc9_bnbnu_overlay_1500/truth_sidecar/}

NLINES=$(grep -c . "${INPUT_LIST}")
PER=$(( (NLINES + NSHARDS - 1) / NSHARDS ))
START=$(( SLURM_ARRAY_TASK_ID * PER + 1 ))
END=$(( START + PER - 1 )); [ ${END} -gt ${NLINES} ] && END=${NLINES}
mkdir -p "${OUTPUT_DIR}" "${WORKDIR}/logs/export"
[ ${START} -gt ${NLINES} ] && { echo "nothing to do"; exit 0; }
echo ">>> shard ${SLURM_ARRAY_TASK_ID}: filenos ${START}-${END} -> ${OUTPUT_DIR}"

module load apptainer 2>/dev/null || true
apptainer exec --bind /cluster:/cluster "${container}" bash -c "
  cd ${UBDL_DIR} && source setenv_pointcept_container.sh >/dev/null 2>&1
  cd ${WORKDIR}
  for FILENO in \$(seq ${START} ${END}); do
    Z=\$(printf '%05d' \${FILENO})
    OUT=${OUTPUT_DIR}/truth_fileno\${Z}.h5
    [ -s \${OUT} ] && { echo \"  fileno \${FILENO}: exists, skip\"; continue; }
    IN=\$(sed -n \"\${FILENO}p\" ${INPUT_LIST})
    [ -f \"\${IN}\" ] || { echo \"  fileno \${FILENO}: MISSING \${IN}\"; continue; }
    python3 ${RECODIR}/export/extract_truth_sidecar.py \
      --input-dlmerged \${IN} --out \${OUT} || echo \"  fileno \${FILENO}: FAILED\"
  done
"
echo "DONE shard ${SLURM_ARRAY_TASK_ID}"
