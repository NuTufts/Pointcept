#!/bin/bash
#
# Sharded Stage-A conversion (dlmerged ROOT -> merged_sp H5) on the batch
# partition, standalone-config driven. Each array task processes STRIDE
# consecutive lines of the config's INPUTLIST starting at
# OFFSET + STRIDE*TASK_ID + 1, running run_stepA_convert_wconfig.sh per line
# (it self-re-execs into the pointcept container), then copies the merged H5
# from the node-local workdir to the config's OUTPUT_DIR (the standalone mode
# leaves outputs in the workdir; only the full driver copies them).
#
# Submit:
#   CONF=larformer_configs/bnb5e19_tufts.conf STRIDE=5 \
#     sbatch --array=0-3 submit_stepA_shard.sh     # -> lines 1..20
# ---------------------------------------------------------------------------

#SBATCH --job-name=stepA
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=8:00:00
#SBATCH --partition=batch
#SBATCH --output=logs/data_prep/stepA.%A_%a.%N.log
#SBATCH --error=logs/data_prep/stepA.%A_%a.%N.err

set -u
# NOTE: sbatch runs a COPY from /var/spool/slurm, so BASH_SOURCE cannot be
# used to locate the repo -- hardcode like the other slurm scripts.
WORKROOT=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
SCRIPT_DIR=${WORKROOT}/lartpc/data_prep/uboone_official
CONF=${CONF:-${SCRIPT_DIR}/larformer_configs/bnb5e19_tufts.conf}
STRIDE=${STRIDE:-5}
OFFSET=${OFFSET:-0}
mkdir -p "${WORKROOT}/logs/data_prep"

# source the config directly for TAG / OUTPUT_DIR / INPUTLIST (it only sets
# variables; POINTCEPT_DIR/REPO_ROOT must be defined first)
POINTCEPT_DIR=${WORKROOT}
REPO_ROOT=${SCRIPT_DIR}
source "${CONF}"
# merged_sp_leaf() -- write Stage A output straight into the 2-level fileno tree
# (no post-hoc reorg_merged_sp.py needed). Sourcing only defines functions.
source "${SCRIPT_DIR}/_wconfig_common.sh"
WORKDIR_BASE=${WORKDIR_BASE:-/tmp}
NLINES=$(grep -c . "${INPUTLIST}")
mkdir -p "${OUTPUT_DIR}"
module load apptainer 2>/dev/null || true

nfail=0
for i in $(seq 0 $((STRIDE - 1))); do
    LINENO_=$((OFFSET + STRIDE * SLURM_ARRAY_TASK_ID + i + 1))
    [ "${LINENO_}" -gt "${NLINES}" ] && break
    # skip if this line's output already exists (resume-friendly)
    ZF=$(printf "%05d" ${LINENO_})
    LEAFDIR=$(merged_sp_leaf "${OUTPUT_DIR}" "${LINENO_}")
    if ls "${LEAFDIR}"/merged_${TAG}_fileno${ZF}_entry*.h5 >/dev/null 2>&1; then
        echo ">>> line ${LINENO_}: output exists, skip"
        continue
    fi
    echo ">>> line ${LINENO_}: converting"
    bash -c "cd '${SCRIPT_DIR}' && source '${SCRIPT_DIR}/run_stepA_convert_wconfig.sh' '${CONF}' ${LINENO_}"
    rc=$?
    WD=$(printf "%s/lantern_%s_jobid%04d_line%05d" "${WORKDIR_BASE}" \
         "${TAG}" "${SLURM_ARRAY_TASK_ID}" "${LINENO_}")
    if [ ${rc} -eq 0 ] && ls "${WD}"/merged_${TAG}_fileno*_entry*.h5 >/dev/null 2>&1; then
        mkdir -p "${LEAFDIR}"
        cp "${WD}"/merged_${TAG}_fileno*_entry*.h5 "${LEAFDIR}/"
        echo ">>> line ${LINENO_}: $(ls "${WD}"/merged_${TAG}_fileno*_entry*.h5 | wc -l) files -> ${LEAFDIR}"
    else
        echo ">>> line ${LINENO_}: FAILED (rc=${rc})"
        nfail=$((nfail + 1))
    fi
    rm -rf "${WD}"
done
echo "DONE task ${SLURM_ARRAY_TASK_ID} (${nfail} failures)"
[ ${nfail} -eq 0 ]
