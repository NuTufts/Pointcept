#!/bin/bash
#
# run_step1_lantern_wconfig.sh
# Runs inside the lantern container. Config-driven variant.
#
# Two invocation modes:
#   1) Driver mode (called from run_showerorigin_reco_wconfig.sh):
#        source run_step1_lantern_wconfig.sh <workdir> <adcname> <tbflag> [max_events] [keep]
#   2) Standalone mode (from bare node or from inside lantern container):
#        source run_step1_lantern_wconfig.sh <config_file> [lineno]
#      In standalone mode the bootstrap auto-apptainer-execs into the lantern
#      container when run from a bare node. Defaults lineno to OFFSET + 1.
#

WCONFIG_CONTAINER_TYPE=lantern
source "$(dirname "${BASH_SOURCE[0]}")/_wconfig_common.sh"
wconfig_bootstrap "$@"; _bs_rc=$?
if [ "${WCONFIG_RE_EXECED:-0}" = "1" ] || [ ${_bs_rc} -ne 0 ]; then
    return ${_bs_rc} 2>/dev/null || exit ${_bs_rc}
fi
if [ "${WCONFIG_STANDALONE:-0}" = "1" ]; then
    set -- "${WCONFIG_WORKDIR}" "${ADCNAME}" "${TBFLAG}" "${MAX_EVENTS}" "${KEEP_INTERMEDIATES}"
fi

WORKDIR_PATH=$1
ADCNAME=${2:-wire}
TBFLAG=${3:-""}
MAX_EVENTS=${4:--1}
KEEP_INTERMEDIATES=${5:-0}

if [ -z "${WORKDIR_PATH}" ]; then
    echo "ERROR: workdir path not provided"
    return 1 2>/dev/null || exit 1
fi

cd "${WORKDIR_PATH}" || { echo "cannot cd to ${WORKDIR_PATH}"; return 1 2>/dev/null || exit 1; }
echo "Step 1 workdir: $(pwd)"
echo "MAX_EVENTS: ${MAX_EVENTS}"
echo "KEEP_INTERMEDIATES: ${KEEP_INTERMEDIATES}"
echo "Files at start:"
ls -lh

# Lantern env (container-internal paths)
shopt -s expand_aliases
alias python=python3

export LANTERN_DIR=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/lantern_ubdl/
export UBDL_DIR=${LANTERN_DIR}/ubdl/
export LARMATCH_DIR=${UBDL_DIR}/larflow/larmatchnet/larmatch/
export SSNET_DIR=${LANTERN_DIR}/uresnet_pytorch/
export LANTERN_SCRIPTS=${LANTERN_DIR}/lantern_scripts/

cd ${UBDL_DIR}
#source setenv_pointcept_container.sh
source setenv_py3_container.sh
source configure_container.sh

cd ${UBDL_DIR}/larflow/larmatchnet
source set_pythonpath.sh
export PYTHONPATH=${LARMATCH_DIR}:${PYTHONPATH}
export PYTHONPATH=${PYTHONPATH}:${SSNET_DIR}
cd "${WORKDIR_PATH}"
echo "Environment setup complete"

larsoftout=$(find . -maxdepth 1 -type f -name 'merged_dlreco*.root' -printf '%f\n' | head -1)
if [ -z "${larsoftout}" ]; then
    larsoftout=$(find . -maxdepth 1 -type f -name 'merged_dlana*.root' -printf '%f\n' | head -1)
fi
if [ -z "${larsoftout}" ]; then
    larsoftout=$(find . -maxdepth 1 -type f -name 'dlmerged*.root' -printf '%f\n' | head -1)
fi
if [ -z "${larsoftout}" ]; then
    echo "ERROR: no input ROOT file in workdir"
    ls -la
    return 1 2>/dev/null || exit 1
fi
echo "Input file: ${larsoftout}"

baseinput="merged_dlreco_with_ssnet.root"
CONFIG_FILE="${LANTERN_SCRIPTS}/config_larmatchme_deploycpu.yaml"
WEIGHT_FILE="larmatch_ckpt78k.pt"
INPUTSTEM="merged_dlreco_with_ssnet"
lm_outfile=$(echo ${baseinput} | sed "s|${INPUTSTEM}|larmatchme|g")
baselm=$(echo ${baseinput} | sed "s|${INPUTSTEM}|larmatchme|g" | sed 's|.root|_larlite.root|g')
echo "larmatch outfile: ${lm_outfile}"
echo "baselm: ${baselm}"

# Copy local bug-fixed lantern scripts
LOCAL_LANTERN_SCRIPTS=${LANTERN_DIR}/lantern_scripts
cp ${LOCAL_LANTERN_SCRIPTS}/inference_sparse_ssnet_uboone.py .
cp ${LOCAL_LANTERN_SCRIPTS}/recreate_ubspurn.py .
echo "Copied local lantern scripts to workdir"

NENTFLAG=""
LMFLAG_N=""
if [ "${MAX_EVENTS:-0}" -gt 0 ] 2>/dev/null; then
    NENTFLAG="-n ${MAX_EVENTS}"
    LMFLAG_N="-n ${MAX_EVENTS}"
    echo "Limiting Step 1 to ${MAX_EVENTS} entries"
fi

# ---- SSNet inference ----
if [ "${KEEP_INTERMEDIATES}" = "1" ] && [ -f "ssnet_output.root" ]; then
    echo "SSNet: ssnet_output.root present, skipping"
else
    echo "----------------------------------------------"
    echo "Running SSNet inference"
    CMD="python3 ./inference_sparse_ssnet_uboone.py -i ${larsoftout} -w ${SSNET_DIR}/weights -o ssnet_output.root --adcname ${ADCNAME} ${TBFLAG} ${NENTFLAG}"
    echo ${CMD}
    ${CMD}
    if [ $? -ne 0 ]; then
        echo "ERROR: SSNet inference failed"
        return 1 2>/dev/null || exit 1
    fi
fi

# ---- Merge SSNet output into dlmerged copy ----
if [ "${KEEP_INTERMEDIATES}" = "1" ] && [ -f "${baseinput}" ]; then
    echo "SSNet merge: ${baseinput} present, skipping"
else
    echo "Merging SSNet output with merged_dlreco..."
    cp ${larsoftout} ${baseinput}
    chmod a+rw ${baseinput}
    rootcp ssnet_output.root:sparseimg_sparseuresnetout_tree ${baseinput}
    python3 ./recreate_ubspurn.py -i ${baseinput} -o ssnet_ubspurn_output.root --adcname ${ADCNAME} ${TBFLAG} ${NENTFLAG}
    inputTrees=$(rootls ssnet_ubspurn_output.root)
    for tree in ${inputTrees}; do
        rootcp ssnet_ubspurn_output.root:${tree} ${baseinput}
    done
fi

# ---- LArMatch deploy ----
if [ "${KEEP_INTERMEDIATES}" = "1" ] && [ -f "${baselm}" ]; then
    echo "LArMatch: ${baselm} present, skipping"
else
    echo "----------------------------------------------"
    echo "Running LArMatch deploy"
    CMD="python3 ${LARMATCH_DIR}/deploy_larmatchme.py --config-file ${CONFIG_FILE} --supera ${baseinput} --weights ${LARMATCH_DIR}/${WEIGHT_FILE} --output ${lm_outfile} --min-score 0.5 --adc-name ${ADCNAME} --chstatus-name ${ADCNAME} --device-name cpu --use-skip-limit --save-larmatch-feats ${TBFLAG} ${LMFLAG_N}"
    echo ${CMD}
    ${CMD}
    if [ $? -ne 0 ]; then
        echo "ERROR: LArMatch deploy failed"
        return 1 2>/dev/null || exit 1
    fi
fi

if [ "${KEEP_INTERMEDIATES}" != "1" ]; then
    rm -f ssnet_output.root ssnet_ubspurn_output.root
    rm -f larmatchme_larcv.root
fi

echo "Step 1 complete. Files in workdir:"
ls -lh
