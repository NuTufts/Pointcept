#!/bin/bash
#
# run_stepA_convert_wconfig.sh
# Stage A of the LArFormer cascade data path. Runs inside the pointcept
# container. Config-driven.
#
# merged_dlreco.root  ->  merged_<TAG>_fileno<NNNNN>_entry<N>.h5
#   (SimChTripletLabelMaker triplets + mc_particle_tree + dummy lm_score
#    + folded-in detector flashes/pmt_positions).  NO LArMatch, NO SSNet.
#
# Two invocation modes:
#   1) Driver mode (from run_larformer_wconfig.sh):
#        source run_stepA_convert_wconfig.sh <workdir> <fileno> <tag> <adc> <tb> <mcc9> <isdata> [max_events] [keep]
#   2) Standalone mode (bare node or inside pointcept container):
#        source run_stepA_convert_wconfig.sh <config_file> [lineno]
#

WCONFIG_CONTAINER_TYPE=pointcept
source "$(dirname "${BASH_SOURCE[0]}")/_wconfig_common.sh"
wconfig_bootstrap "$@"; _bs_rc=$?
if [ "${WCONFIG_RE_EXECED:-0}" = "1" ] || [ ${_bs_rc} -ne 0 ]; then
    return ${_bs_rc} 2>/dev/null || exit ${_bs_rc}
fi
if [ "${WCONFIG_STANDALONE:-0}" = "1" ]; then
    ISDATAFLAG=${ISDATAFLAG:-""}
    set -- "${WCONFIG_WORKDIR}" "${WCONFIG_FILENO}" "${TAG}" "${ADCNAME}" \
           "${TBFLAG}" "${MCC9FLAG}" "${ISDATAFLAG}" "${MAX_EVENTS}" \
           "${KEEP_INTERMEDIATES}"
fi

WORKDIR_PATH=$1
FILENO=$2
FILETAG=$3
ADCNAME=${4:-wiremc}
TBFLAG=${5:-""}
MCC9FLAG=${6:-""}
ISDATAFLAG=${7:-""}
MAX_EVENTS=${8:--1}
KEEP_INTERMEDIATES=${9:-0}

if [ -z "${WORKDIR_PATH}" ] || [ -z "${FILENO}" ] || [ -z "${FILETAG}" ]; then
    echo "ERROR: usage: source run_stepA_convert_wconfig.sh <workdir> <fileno> <tag> ..."
    return 1 2>/dev/null || exit 1
fi
ZFILENO=$(printf "%05d" ${FILENO})

# Resolve repo paths (the larformer_scripts dir is REPO_ROOT/SCRIPT_DIR).
if [ -z "${SCRIPT_DIR:-}" ]; then
    SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
fi
PY_DIR=${SCRIPT_DIR}
CONVERT_PY="${PY_DIR}/convert_dlmerged_to_larformer_h5.py"
if [ ! -f "${CONVERT_PY}" ]; then
    echo "ERROR: converter not found at ${CONVERT_PY}"
    return 1 2>/dev/null || exit 1
fi

# Resolve the input dlmerged for this line.
INPUT_ROOT=""
if [ -n "${INPUTLIST}" ] && [ -f "${INPUTLIST}" ]; then
    INPUT_ROOT=$(sed -n "${FILENO}p" "${INPUTLIST}")
fi
if [ -z "${INPUT_ROOT}" ] || [ ! -f "${INPUT_ROOT}" ]; then
    echo "ERROR: line ${FILENO} of ${INPUTLIST} empty or missing: '${INPUT_ROOT}'"
    return 1 2>/dev/null || exit 1
fi

cd "${WORKDIR_PATH}" || { echo "cannot cd to ${WORKDIR_PATH}"; return 1 2>/dev/null || exit 1; }
echo "Stage A workdir: $(pwd)"
echo "  input    : ${INPUT_ROOT}"
echo "  adc=${ADCNAME} tb='${TBFLAG}' mcc9='${MCC9FLAG}' isdata='${ISDATAFLAG}'"
echo "  MAX_EVENTS=${MAX_EVENTS}  KEEP_INTERMEDIATES=${KEEP_INTERMEDIATES}"

# ubdl env (larlite + larcv + larflow + h5py)
UBDL_DIR=${UBDL_DIR:-/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/ubdl}
cd "${UBDL_DIR}"
source setenv_pointcept_container.sh
cd "${WORKDIR_PATH}"

# Stage-level skip
has_merged=$(ls "${WORKDIR_PATH}"/merged_${FILETAG}_fileno${ZFILENO}_entry*.h5 2>/dev/null | head -1)
if [ "${KEEP_INTERMEDIATES}" = "1" ] && [ -n "${has_merged}" ]; then
    echo "Stage A: skipped (merged H5 present)"
    return 0 2>/dev/null || exit 0
fi

NENTFLAG=""
if [ "${MAX_EVENTS:-0}" -gt 0 ] 2>/dev/null; then
    NENTFLAG="-n ${MAX_EVENTS}"
fi

CMD="python3 ${CONVERT_PY} \
    -i ${INPUT_ROOT} \
    -o ${WORKDIR_PATH} \
    --tag ${FILETAG} \
    --fileno-tag fileno${ZFILENO} \
    --adc ${ADCNAME} ${TBFLAG} ${MCC9FLAG} ${ISDATAFLAG} ${NENTFLAG}"
echo "${CMD}"
${CMD}
stepA_status=$?
echo "Stage A exit status: ${stepA_status}"
if [ ${stepA_status} -ne 0 ]; then
    echo "ERROR: Stage A conversion failed"
    return 1 2>/dev/null || exit 1
fi

echo "Stage A output files:"
ls -lh "${WORKDIR_PATH}"/merged_${FILETAG}_fileno${ZFILENO}_entry*.h5 2>/dev/null
echo "Stage A complete"
