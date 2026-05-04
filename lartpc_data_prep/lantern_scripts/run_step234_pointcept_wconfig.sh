#!/bin/bash
#
# run_step234_pointcept_wconfig.sh
# Runs inside the pointcept container. Config-driven variant.
# Steps 2 (reco H5) / 3 (truth H5) / 4 (merge).
#
# Uses migrated Python entrypoints from ${REPO_ROOT}/ubshowerorginreco/,
# not the pointcept copies.
#
# Two invocation modes:
#   1) Driver mode (called from run_showerorigin_reco_wconfig.sh):
#        source run_step234_pointcept_wconfig.sh <workdir> <fileno> <tag> <adc> <tb> <mcc9> [max_events] [keep]
#   2) Standalone mode (from bare node or inside pointcept container):
#        source run_step234_pointcept_wconfig.sh <config_file> [lineno]
#      Expects Step 1 outputs (larmatchme_larlite.root + merged_dlreco_with_ssnet.root)
#      already present in the workdir the config resolves to.
#

WCONFIG_CONTAINER_TYPE=pointcept
source "$(dirname "${BASH_SOURCE[0]}")/_wconfig_common.sh"
wconfig_bootstrap "$@"; _bs_rc=$?
if [ "${WCONFIG_RE_EXECED:-0}" = "1" ] || [ ${_bs_rc} -ne 0 ]; then
    return ${_bs_rc} 2>/dev/null || exit ${_bs_rc}
fi
if [ "${WCONFIG_STANDALONE:-0}" = "1" ]; then
    set -- "${WCONFIG_WORKDIR}" "${WCONFIG_FILENO}" "${TAG}" "${ADCNAME}" "${TBFLAG}" "${MCC9FLAG}" "${MAX_EVENTS}" "${KEEP_INTERMEDIATES}"
fi

WORKDIR_PATH=$1
FILENO=$2
FILETAG=$3
ADCNAME=${4:-wire}
TBFLAG=${5:-""}
MCC9FLAG=${6:-""}
MAX_EVENTS=${7:--1}
KEEP_INTERMEDIATES=${8:-0}

if [ -z "${WORKDIR_PATH}" ] || [ -z "${FILENO}" ] || [ -z "${FILETAG}" ]; then
    echo "ERROR: usage: source run_step234_pointcept_wconfig.sh <workdir> <fileno> <tag> ..."
    return 1 2>/dev/null || exit 1
fi

ZFILENO=$(printf "%05d" ${FILENO})

cd "${WORKDIR_PATH}" || { echo "cannot cd to ${WORKDIR_PATH}"; return 1 2>/dev/null || exit 1; }
echo "Steps 2-4 workdir: $(pwd)"
echo "MAX_EVENTS: ${MAX_EVENTS}"
echo "KEEP_INTERMEDIATES: ${KEEP_INTERMEDIATES}"
echo "Files at start:"
ls -lh

# Resolve repo paths. Prefer bootstrap/driver-exported values (which are
# always absolute). Only re-derive if neither was set — and do it *before*
# any subsequent cd, while BASH_SOURCE[0] is still meaningful.
if [ -z "${REPO_ROOT:-}" ]; then
    SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
    REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
fi
PY_DIR=${REPO_ROOT}

UBDL_DIR=${UBDL_DIR:-/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/ubdl}
cd "${UBDL_DIR}"
source setenv_pointcept_container.sh
cd "${WORKDIR_PATH}"
echo "Environment setup complete. Python scripts from: ${PY_DIR}"

LARLITE_FILE="${WORKDIR_PATH}/larmatchme_larlite.root"
DLMERGED_FILE="${WORKDIR_PATH}/merged_dlreco_with_ssnet.root"

if [ ! -f "${LARLITE_FILE}" ]; then
    echo "ERROR: larmatchme_larlite.root not found"
    return 1 2>/dev/null || exit 1
fi
if [ ! -f "${DLMERGED_FILE}" ]; then
    echo "ERROR: merged_dlreco_with_ssnet.root not found"
    return 1 2>/dev/null || exit 1
fi

RECO_DIR=${WORKDIR_PATH}/reco_h5
TRUTH_DIR=${WORKDIR_PATH}/truth_h5
MERGED_DIR=${WORKDIR_PATH}
mkdir -p "${RECO_DIR}" "${TRUTH_DIR}"
mkdir -p "${MERGEFILE_OUTPUT_DIR}"

NENTFLAG=""
if [ "${MAX_EVENTS:-0}" -gt 0 ] 2>/dev/null; then
    NENTFLAG="-n ${MAX_EVENTS}"
    echo "Limiting Steps 2-3 to ${MAX_EVENTS} entries"
fi

# ---- STEP 2 ----
has_reco=$(ls "${RECO_DIR}"/showerorigin_*.h5 2>/dev/null | head -1)
if [ "${KEEP_INTERMEDIATES}" = "1" ] && [ -n "${has_reco}" ]; then
    echo "STEP 2: skipped (reco H5 present)"
else
    echo "----------------------------------------------"
    echo "STEP 2: convert_larlite_to_pointcept_h5.py"
    CMD="python3 ${PY_DIR}/convert_larlite_to_pointcept_h5.py \
        -i ${LARLITE_FILE} \
        --input-larcv ${DLMERGED_FILE} \
        -o ${RECO_DIR} --adc ${ADCNAME} ${TBFLAG} ${NENTFLAG} \
        --fileno-tag fileno${ZFILENO}"
    echo ${CMD}
    ${CMD}
    step2_status=$?
    echo "Step 2 exit status: ${step2_status}"
    if [ ${step2_status} -ne 0 ]; then
        echo "ERROR: Step 2 failed"
        return 1 2>/dev/null || exit 1
    fi
fi
echo "Step 2 output files:"
ls -lh "${RECO_DIR}"/showerorigin_*.h5 2>/dev/null

# ---- STEP 3 ----
has_truth=$(ls "${TRUTH_DIR}"/pointceptdata_*.h5 2>/dev/null | head -1)
if [ "${KEEP_INTERMEDIATES}" = "1" ] && [ -n "${has_truth}" ]; then
    echo "STEP 3: skipped (truth H5 present)"
else
    echo "----------------------------------------------"
    echo "STEP 3: process_dlmerged_to_hdf5_event_files.py"
    CMD="python3 ${PY_DIR}/process_dlmerged_to_hdf5_event_files.py \
        -i ${DLMERGED_FILE} --adc ${ADCNAME} ${TBFLAG} ${MCC9FLAG} ${NENTFLAG} \
        --fileno-tag fileno${ZFILENO}"
    echo ${CMD}
    ${CMD}
    step3_status=$?
    echo "Step 3 exit status: ${step3_status}"
    if [ ${step3_status} -ne 0 ]; then
        echo "ERROR: Step 3 failed"
        return 1 2>/dev/null || exit 1
    fi
    mv "${WORKDIR_PATH}"/pointceptdata_*.h5 "${TRUTH_DIR}"/ 2>/dev/null
fi
echo "Step 3 output files:"
ls -lh "${TRUTH_DIR}"/pointceptdata_*.h5 2>/dev/null

# ---- STEP 4 ----
echo "----------------------------------------------"
echo "STEP 4: merge_reco_truth_showerorigin.py"
nmerged=0
nfailed=0
for reco_file in "${RECO_DIR}"/showerorigin_*.h5; do
    if [ ! -f "${reco_file}" ]; then
        echo "No reco H5 files found"
        break
    fi
    reco_basename=$(basename "${reco_file}")
    entry_part=$(echo "${reco_basename}" | grep -oP 'entry\d+')
    if [ -z "${entry_part}" ]; then
        echo "WARNING: could not extract entry from ${reco_basename}, skipping"
        continue
    fi
    truth_file=$(ls "${TRUTH_DIR}"/pointceptdata_*_${entry_part}.h5 2>/dev/null | head -1)
    if [ -z "${truth_file}" ] || [ ! -f "${truth_file}" ]; then
        echo "WARNING: no matching truth file for ${entry_part}, skipping"
        let nfailed+=1
        continue
    fi
    merged_outfile=${MERGED_DIR}/merged_${FILETAG}_fileno${ZFILENO}_${entry_part}.h5
    if [ "${KEEP_INTERMEDIATES}" = "1" ] && [ -f "${merged_outfile}" ]; then
        echo "Step 4: ${entry_part} already merged, skipping"
        let nmerged+=1
        continue
    fi
    CMD="python3 ${PY_DIR}/merge_reco_truth_showerorigin.py \
        --reco-h5 ${reco_file} \
        --truth-h5 ${truth_file} \
        --output ${merged_outfile}"
    echo ${CMD}
    ${CMD}
    if [ $? -eq 0 ]; then
        let nmerged+=1
    else
        echo "WARNING: merge failed for ${entry_part}"
        let nfailed+=1
    fi
done

echo "----------------------------------------------"
echo "Step 4 complete: ${nmerged} merged, ${nfailed} failed"
echo "Final merged files:"
ls -lh "${MERGED_DIR}"/merged_*.h5 2>/dev/null

if [ "${KEEP_INTERMEDIATES}" != "1" ]; then
    rm -rf "${RECO_DIR}" "${TRUTH_DIR}"
    rm -f "${WORKDIR_PATH}"/larmatchme_larlite.root
    rm -f "${WORKDIR_PATH}"/merged_dlreco_with_ssnet.root
    rm -f "${WORKDIR_PATH}"/*.root
fi
echo "Steps 2-4 complete"
