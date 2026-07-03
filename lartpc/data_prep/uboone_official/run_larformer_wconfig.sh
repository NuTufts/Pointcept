#!/bin/bash
#
# run_larformer_wconfig.sh
# Top driver for the LArFormer cascade data path. Config-driven.
#
# For each input line (OFFSET + stride*SLURM_ARRAY_TASK_ID + i, i=1..stride):
#   Stage A  convert merged_dlreco.root -> merged_<TAG>_fileno<N>_entry*.h5
#   Stage B  (if RUN_CASCADE=1) full LArFormer cascade -> stage3pred_*.h5
#   copy outputs to OUTPUT_DIR/<lineno/1000>/<lineno/100>/  (3-level hash)
#   rm -rf workdir   (unless KEEP_INTERMEDIATES=1)
#
# Everything runs in the pointcept container (NO lantern container — LArMatch
# and SSNet are not used; the LoRA deghoster handles deghosting).
#
# Usage (bare node / inside container):
#   source run_larformer_wconfig.sh configs/<dataset>.conf
#   (SLURM_ARRAY_TASK_ID selects the stride block; defaults to 0)
#

WCONFIG_CONTAINER_TYPE=pointcept
source "$(dirname "${BASH_SOURCE[0]}")/_wconfig_common.sh"
wconfig_bootstrap "$@"; _bs_rc=$?
if [ "${WCONFIG_RE_EXECED:-0}" = "1" ] || [ ${_bs_rc} -ne 0 ]; then
    return ${_bs_rc} 2>/dev/null || exit ${_bs_rc}
fi
# Not a config file -> nothing to do (this script is config-only).
if [ "${WCONFIG_STANDALONE:-0}" != "1" ]; then
    echo "ERROR: usage: source run_larformer_wconfig.sh configs/<dataset>.conf"
    return 1 2>/dev/null || exit 1
fi

RUN_CASCADE=${RUN_CASCADE:-1}
ISDATAFLAG=${ISDATAFLAG:-""}
jobid=${SLURM_ARRAY_TASK_ID:-0}
startline=$(( OFFSET + stride * jobid ))

mkdir -p "${OUTPUT_DIR}"
echo "=========================================================="
echo "LArFormer data path  TAG=${TAG}"
echo "  INPUTLIST   : ${INPUTLIST}"
echo "  OUTPUT_DIR  : ${OUTPUT_DIR}"
echo "  jobid=${jobid} stride=${stride} startline=${startline}"
echo "  adc=${ADCNAME} tb='${TBFLAG}' mcc9='${MCC9FLAG}' isdata='${ISDATAFLAG}'"
echo "  RUN_CASCADE=${RUN_CASCADE}  KEEP_INTERMEDIATES=${KEEP_INTERMEDIATES}"
echo "=========================================================="

for (( i=1; i<=stride; i++ )); do
    lineno=$(( startline + i ))
    inputfile=$(sed -n "${lineno}p" "${INPUTLIST}")
    if [ -z "${inputfile}" ]; then
        echo "LINE ${lineno}: empty, skipping"; continue
    fi
    if [ ! -f "${inputfile}" ]; then
        echo "LINE ${lineno}: file not found: ${inputfile}"; continue
    fi
    echo "---- LINE ${lineno}: $(basename ${inputfile}) ----"

    local_jobdir=$(printf "%s/larformer_${TAG}_jobid%04d_line%05d" \
        "${WORKDIR_BASE}" ${jobid} ${lineno})
    if [ "${KEEP_INTERMEDIATES}" != "1" ]; then
        rm -rf "${local_jobdir}"
    fi
    mkdir -p "${local_jobdir}"

    # ---- Stage A: convert ----
    source "${SCRIPT_DIR}/run_stepA_convert_wconfig.sh" \
        "${local_jobdir}" "${lineno}" "${TAG}" "${ADCNAME}" \
        "${TBFLAG}" "${MCC9FLAG}" "${ISDATAFLAG}" "${MAX_EVENTS}" \
        "${KEEP_INTERMEDIATES}"
    if [ $? -ne 0 ]; then
        echo "Stage A FAILED for line ${lineno}"
        [ "${KEEP_INTERMEDIATES}" != "1" ] && rm -rf "${local_jobdir}"
        continue
    fi

    # ---- Stage B: cascade (optional) ----
    if [ "${RUN_CASCADE}" = "1" ]; then
        source "${SCRIPT_DIR}/run_stepB_cascade_wconfig.sh" \
            "${local_jobdir}" "${lineno}" "${TAG}" "${ISDATAFLAG}" \
            "${KEEP_INTERMEDIATES}"
        if [ $? -ne 0 ]; then
            echo "Stage B FAILED for line ${lineno} (merged H5 still available)"
        fi
    fi

    # ---- Copy outputs to 3-level hashed output dir ----
    nsubdir1=$(( lineno / 1000 )); zsubdir1=$(printf "%03d" ${nsubdir1})
    nsubdir2=$(( lineno / 100 ));  zsubdir2=$(printf "%03d" ${nsubdir2})
    outfolder="${OUTPUT_DIR}/${zsubdir1}/${zsubdir2}"
    mkdir -p "${outfolder}"

    nmerged=$(ls "${local_jobdir}"/merged_${TAG}_fileno*_entry*.h5 2>/dev/null | wc -l)
    if [ "${nmerged}" -gt 0 ]; then
        cp "${local_jobdir}"/merged_${TAG}_fileno*_entry*.h5 "${outfolder}/"
        echo "Copied ${nmerged} merged H5 -> ${outfolder}"
    fi
    if [ "${RUN_CASCADE}" = "1" ]; then
        npred=$(ls "${local_jobdir}"/stage3pred_*.h5 2>/dev/null | wc -l)
        if [ "${npred}" -gt 0 ]; then
            cp "${local_jobdir}"/stage3pred_*.h5 "${outfolder}/"
            echo "Copied ${npred} stage3pred H5 -> ${outfolder}"
        fi
    fi

    # ---- Cleanup ----
    if [ "${KEEP_INTERMEDIATES}" != "1" ]; then
        rm -rf "${local_jobdir}"
    fi
done

echo "run_larformer_wconfig.sh: done (jobid ${jobid})"
