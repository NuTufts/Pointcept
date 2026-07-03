#!/bin/bash
#
# run_step5_flashinfo_pointcept_wconfig.sh
# Runs inside the pointcept container. Config-driven.
# Step 5: prepare auxiliary flashinfo H5 files paired with the merged H5s.
#
# Two invocation modes:
#   1) Driver mode (called from run_lantern_wconfig.sh, integrated pipeline):
#        source run_step5_flashinfo_pointcept_wconfig.sh <workdir> <fileno> <tag> [dtick_threshold] [keep]
#      Expects merged H5 files in <workdir>, ROOT input also in <workdir>.
#      Writes flashinfo_*.h5 into <workdir>; the driver cp's them to
#      ${FLASHINFO_OUTPUT_DIR}/<F/1000>/<F/100>/ afterwards.
#
#   2) Standalone mode (from bare node or directly inside pointcept container):
#        source run_step5_flashinfo_pointcept_wconfig.sh <config_file> [lineno]
#      Reads merged H5 from ${MERGEFILE_OUTPUT_DIR}/<F/1000>/<F/100>/
#      Reads ROOT input directly from the path on line ${lineno} of ${INPUTLIST}.
#      Writes flashinfo_*.h5 DIRECTLY to ${FLASHINFO_OUTPUT_DIR}/<F/1000>/<F/100>/
#      (.tmp + rename atomicity is handled by the Python script).
#

WCONFIG_CONTAINER_TYPE=pointcept
source "$(dirname "${BASH_SOURCE[0]}")/_wconfig_common.sh"
wconfig_bootstrap "$@"; _bs_rc=$?
if [ "${WCONFIG_RE_EXECED:-0}" = "1" ] || [ ${_bs_rc} -ne 0 ]; then
    return ${_bs_rc} 2>/dev/null || exit ${_bs_rc}
fi
if [ "${WCONFIG_STANDALONE:-0}" = "1" ]; then
    DTICK_THRESHOLD=${DTICK_THRESHOLD:-3.0}
    set -- "${WCONFIG_WORKDIR}" "${WCONFIG_FILENO}" "${TAG}" "${DTICK_THRESHOLD}" "${KEEP_INTERMEDIATES}"
fi

WORKDIR_PATH=$1
FILENO=$2
FILETAG=$3
DTICK_THRESHOLD=${4:-3.0}
KEEP_INTERMEDIATES=${5:-0}

if [ -z "${WORKDIR_PATH}" ] || [ -z "${FILENO}" ] || [ -z "${FILETAG}" ]; then
    echo "ERROR: usage: source run_step5_flashinfo_pointcept_wconfig.sh <workdir> <fileno> <tag> [dtick] [keep]"
    return 1 2>/dev/null || exit 1
fi

ZFILENO=$(printf "%05d" ${FILENO})
nsubdir1=$(( FILENO / 1000 ))
zsubdir1=$(printf "%03d" ${nsubdir1})
nsubdir2=$(( FILENO / 100 ))
zsubdir2=$(printf "%03d" ${nsubdir2})

echo "----------------------------------------------"
echo "STEP 5: flashinfo prep"
echo "  workdir:            ${WORKDIR_PATH}"
echo "  tag:                ${FILETAG}"
echo "  fileno (lineno):    ${FILENO}"
echo "  dtick_threshold:    ${DTICK_THRESHOLD}"
echo "  REPO_ROOT:          ${REPO_ROOT}"

# Resolve REPO_ROOT (Pointcept/lartpc_data_prep dir, parent of lantern_scripts)
if [ -z "${REPO_ROOT:-}" ]; then
    SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
    REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
fi
PY_SCRIPT="${REPO_ROOT}/../prepare_flashinfo_h5.py"
if [ ! -f "${PY_SCRIPT}" ]; then
    echo "ERROR: prepare_flashinfo_h5.py not found at ${PY_SCRIPT}"
    return 1 2>/dev/null || exit 1
fi

# Source ubdl env (larlite + larcv + h5py)
UBDL_DIR=${UBDL_DIR:-/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/ubdl}
cd "${UBDL_DIR}"
source setenv_pointcept_container.sh
cd "${WORKDIR_PATH}" || { echo "cannot cd to ${WORKDIR_PATH}"; return 1 2>/dev/null || exit 1; }

# ---- Resolve INPUT_ROOT (ROOT dlmerged file) ----
# Prefer the local workdir copy (integrated pipeline). Fall back to the
# original cluster path from the INPUTLIST (standalone reprocessing).
INPUT_ROOT=""
inputbasename=""
if [ -n "${INPUTLIST}" ] && [ -f "${INPUTLIST}" ]; then
    inputline=$(sed -n "${FILENO}p" "${INPUTLIST}")
    if [ -n "${inputline}" ]; then
        inputbasename=$(basename "${inputline}")
        if [ -f "${WORKDIR_PATH}/${inputbasename}" ]; then
            INPUT_ROOT="${WORKDIR_PATH}/${inputbasename}"
        elif [ -f "${inputline}" ]; then
            INPUT_ROOT="${inputline}"
        fi
    fi
fi
if [ -z "${INPUT_ROOT}" ]; then
    # Last-ditch: any .root in the workdir that isn't an intermediate.
    INPUT_ROOT=$(ls "${WORKDIR_PATH}"/*.root 2>/dev/null \
                 | grep -v 'larmatchme_' \
                 | grep -v 'merged_dlreco_with_ssnet' \
                 | head -1)
fi
if [ -z "${INPUT_ROOT}" ] || [ ! -f "${INPUT_ROOT}" ]; then
    echo "ERROR: could not resolve INPUT_ROOT (lineno ${FILENO})"
    [ -n "${INPUTLIST}" ] && echo "  INPUTLIST line: ${inputline}"
    return 1 2>/dev/null || exit 1
fi
echo "  INPUT_ROOT:         ${INPUT_ROOT}"

# ---- Resolve MERGED_DIR (where merged_<TAG>_fileno<F>_entry*.h5 live) ----
# Prefer the workdir (integrated). Fall back to the production output tree.
MERGED_DIR=""
if ls "${WORKDIR_PATH}"/merged_${FILETAG}_fileno${ZFILENO}_*.h5 >/dev/null 2>&1; then
    MERGED_DIR="${WORKDIR_PATH}"
elif [ -n "${MERGEFILE_OUTPUT_DIR}" ]; then
    candidate="${MERGEFILE_OUTPUT_DIR}/${zsubdir1}/${zsubdir2}"
    if ls "${candidate}"/merged_${FILETAG}_fileno${ZFILENO}_*.h5 >/dev/null 2>&1; then
        MERGED_DIR="${candidate}"
    fi
fi
if [ -z "${MERGED_DIR}" ]; then
    echo "WARNING: no merged_${FILETAG}_fileno${ZFILENO}_entry*.h5 found "
    echo "         (neither in workdir nor in MERGEFILE_OUTPUT_DIR/${zsubdir1}/${zsubdir2})."
    echo "         Step 5 skipped for this line."
    return 0 2>/dev/null || exit 0
fi
echo "  MERGED_DIR:         ${MERGED_DIR}"

# ---- Resolve OUTPUT location ----
# Integrated mode writes into the workdir; the driver cp's afterwards.
# Standalone mode writes directly to the final tree (with .tmp + rename).
if [ "${WCONFIG_STANDALONE:-0}" = "1" ]; then
    if [ -z "${FLASHINFO_OUTPUT_DIR}" ]; then
        echo "ERROR: standalone Step 5 requires FLASHINFO_OUTPUT_DIR in the config"
        return 1 2>/dev/null || exit 1
    fi
    OUTPUT_DIR_LOCAL="${FLASHINFO_OUTPUT_DIR}/${zsubdir1}/${zsubdir2}"
    mkdir -p "${OUTPUT_DIR_LOCAL}"
else
    OUTPUT_DIR_LOCAL="${WORKDIR_PATH}"
fi
echo "  OUTPUT_DIR_LOCAL:   ${OUTPUT_DIR_LOCAL}"

# ---- Resume support: if START_ENTRY is in the env, forward it ----
# In integrated mode the driver sets START_ENTRY for Steps 2-4 from a scan
# of the MERGEFILE_OUTPUT_DIR. For Step 5 we want a separate scan of the
# FLASHINFO_OUTPUT_DIR so it can resume independently. Do that scan here
# whenever the final output tree is reachable.
FLASH_START_ENTRY=0
flash_outfolder=""
if [ -n "${FLASHINFO_OUTPUT_DIR}" ]; then
    flash_outfolder="${FLASHINFO_OUTPUT_DIR}/${zsubdir1}/${zsubdir2}"
    if [ -d "${flash_outfolder}" ]; then
        existing=$(ls "${flash_outfolder}"/flashinfo_${FILETAG}_fileno${ZFILENO}_entry*.h5 2>/dev/null \
                    | sed -E 's|.*entry([0-9]+)\.h5$|\1|' | sort -n -u)
        expected=0
        for n in ${existing}; do
            n10=$((10#$n))
            if [ ${n10} -ne ${expected} ]; then break; fi
            expected=$(( expected + 1 ))
        done
        FLASH_START_ENTRY=${expected}
    fi
fi
START_FLAG=""
if [ "${FLASH_START_ENTRY}" -gt 0 ] 2>/dev/null; then
    START_FLAG="--start-entry ${FLASH_START_ENTRY}"
    echo "  resume: skipping flashinfo entries 0..$(( FLASH_START_ENTRY - 1 )) "
    echo "          (already present in ${flash_outfolder})"
fi

# In integrated mode the per-line workdir is fresh so START_FLAG isn't
# strictly needed; but harmless. In standalone mode it's the resume hook.

# ---- Stage-level skip (KEEP_INTERMEDIATES=1) ----
if [ "${KEEP_INTERMEDIATES}" = "1" ] && [ "${WCONFIG_STANDALONE:-0}" != "1" ]; then
    has_flash=$(ls "${OUTPUT_DIR_LOCAL}"/flashinfo_${FILETAG}_fileno${ZFILENO}_*.h5 2>/dev/null | head -1)
    if [ -n "${has_flash}" ]; then
        echo "STEP 5: skipped (flashinfo H5 present in workdir, KEEP_INTERMEDIATES=1)"
        echo "Step 5 complete"
        return 0 2>/dev/null || exit 0
    fi
fi

# ---- Run the Python batch ----
CMD="python3 ${PY_SCRIPT} --batch \
    --input-dlmerged ${INPUT_ROOT} \
    --merged-dir ${MERGED_DIR} \
    --output-dir ${OUTPUT_DIR_LOCAL} \
    --tag ${FILETAG} \
    --fileno ${FILENO} \
    --dtick-threshold ${DTICK_THRESHOLD} \
    ${START_FLAG}"
echo "${CMD}"
${CMD}
step5_status=$?
echo "Step 5 exit status: ${step5_status}"

if [ ${step5_status} -ne 0 ]; then
    echo "WARNING: Step 5 (flashinfo) failed (exit ${step5_status}). Non-fatal — "
    echo "         merged outputs are unaffected; rerun with the standalone driver."
    return ${step5_status} 2>/dev/null || exit ${step5_status}
fi

echo "Step 5 output files:"
ls -lh "${OUTPUT_DIR_LOCAL}"/flashinfo_${FILETAG}_fileno${ZFILENO}_entry*.h5 2>/dev/null

echo "Step 5 complete"
