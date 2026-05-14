#!/bin/bash
#
# run_flashinfo_wconfig.sh
#
# Top-level driver for the STANDALONE flashinfo pipeline. Use this to
# (re)process flashinfo H5 files for datasets whose merged H5 files have
# already been produced (Steps 1-4 done in a previous run).
#
# Usage:
#   source run_flashinfo_wconfig.sh <config_file>
#
# Reads the same config as run_lantern_wconfig.sh — just uses different
# variables. Required:
#     INPUTLIST                 — same line-keyed list as the integrated driver
#     TAG                       — dataset tag
#     MERGEFILE_OUTPUT_DIR      — where the existing merged_*.h5 tree lives
#     FLASHINFO_OUTPUT_DIR      — where flashinfo_*.h5 are written
#     POINTCEPT_CONTAINER       — pointcept container image
#
# Honors the same per-line scheme as the integrated driver:
#     lineno = OFFSET + stride * SLURM_ARRAY_TASK_ID + i   (i in 1..stride)
# OR rerun mode via RERUN_LINES_FILE.
#
# Per-line skip rules:
#   - If <TAG>_fileno<F>.flashinfo.complete exists      -> already done, skip.
#   - If <TAG>_fileno<F>.complete is MISSING            -> Steps 1-4 not run
#                                                          yet for this line; warn + skip.
#   - Otherwise                                         -> call Step 5 standalone.
#

CONFIG_FILE="${1:-${CONFIG_FILE}}"
if [ -z "${CONFIG_FILE}" ]; then
    echo "ERROR: no config file provided. Usage: source run_flashinfo_wconfig.sh <config_file>" >&2
    return 1 2>/dev/null || exit 1
fi
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "ERROR: config file not found: ${CONFIG_FILE}" >&2
    return 1 2>/dev/null || exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
UBDL_DIR=${UBDL_DIR:-/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/ubdl}
POINTCEPT_DIR=${POINTCEPT_DIR:-/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept}
export SCRIPT_DIR REPO_ROOT UBDL_DIR POINTCEPT_DIR

echo "Running 'run_flashinfo_wconfig.sh'"
echo "Sourcing config file: ${CONFIG_FILE}"
# shellcheck disable=SC1090
source "${CONFIG_FILE}"

module load apptainer/1.2.4-suid 2>/dev/null || true

DTICK_THRESHOLD=${DTICK_THRESHOLD:-3.0}
stride=${stride:-1}
OFFSET=${OFFSET:-0}
SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

for v in INPUTLIST TAG MERGEFILE_OUTPUT_DIR FLASHINFO_OUTPUT_DIR POINTCEPT_CONTAINER; do
    if [ -z "${!v}" ]; then
        echo "ERROR: config must set ${v}" >&2
        return 1 2>/dev/null || exit 1
    fi
done
export TAG INPUTLIST MERGEFILE_OUTPUT_DIR FLASHINFO_OUTPUT_DIR
export POINTCEPT_CONTAINER DTICK_THRESHOLD

POINTCEPT_BIND="/cluster:/cluster"
case "${MERGEFILE_OUTPUT_DIR}" in /cluster*) ;; *) POINTCEPT_BIND="${POINTCEPT_BIND},${MERGEFILE_OUTPUT_DIR}:${MERGEFILE_OUTPUT_DIR}" ;; esac
case "${FLASHINFO_OUTPUT_DIR}" in /cluster*) ;; *) POINTCEPT_BIND="${POINTCEPT_BIND},${FLASHINFO_OUTPUT_DIR}:${FLASHINFO_OUTPUT_DIR}" ;; esac

jobid=${SLURM_ARRAY_TASK_ID}
startline=$(( OFFSET + stride * jobid ))

RERUN_LINES_FILE=${RERUN_LINES_FILE:-}
echo "RERUN_LINES_FILE: ${RERUN_LINES_FILE}"
if [ -n "${RERUN_LINES_FILE}" ] && [ ! -f "${RERUN_LINES_FILE}" ]; then
    echo "ERROR: RERUN_LINES_FILE set but not found: ${RERUN_LINES_FILE}" >&2
    return 1 2>/dev/null || exit 1
fi
export RERUN_LINES_FILE

jobworkdir=$(printf "%s/workdir/flashinfo_${TAG}_jobid_%04d" "${REPO_ROOT}" "${jobid}")
mkdir -p "${jobworkdir}"
mkdir -p "${FLASHINFO_OUTPUT_DIR}"

echo "Made job workdir: ${jobworkdir}"

local_logfile=${jobworkdir}/log_flashinfo_${TAG}_jobid${jobid}.txt
echo "local logfile: ${local_logfile}"

{
    echo "======================================"
    echo "Flashinfo standalone job started: $(date)"
    echo "CONFIG_FILE:           ${CONFIG_FILE}"
    echo "SLURM_ARRAY_TASK_ID:   ${jobid}"
    echo "startline:             ${startline}"
    echo "stride:                ${stride}"
    echo "MERGEFILE_OUTPUT_DIR:  ${MERGEFILE_OUTPUT_DIR}"
    echo "FLASHINFO_OUTPUT_DIR:  ${FLASHINFO_OUTPUT_DIR}"
    echo "DTICK_THRESHOLD:       ${DTICK_THRESHOLD}"
    echo "RERUN_LINES_FILE:      ${RERUN_LINES_FILE:-(unset, normal sequential mode)}"
    echo "REPO_ROOT:             ${REPO_ROOT}"
    echo "======================================"
} > "${local_logfile}"

# for debug
cat ${local_logfile}

for (( i=1; i<=stride; i++ )); do
    if [ -n "${RERUN_LINES_FILE}" ]; then
        rerun_idx=$(( stride * jobid + i ))
        lineno=$(sed -n "${rerun_idx}p" "${RERUN_LINES_FILE}")
        if [ -z "${lineno}" ]; then
            echo "RERUN idx ${rerun_idx}: past end of ${RERUN_LINES_FILE}, stopping" >> "${local_logfile}"
            break
        fi
        if ! [[ "${lineno}" =~ ^[0-9]+$ ]]; then
            echo "RERUN idx ${rerun_idx}: not a line number ('${lineno}'), skipping" >> "${local_logfile}"
            continue
        fi
    else
        lineno=$(( startline + i ))
    fi

    inputfile=$(sed -n "${lineno}p" "${INPUTLIST}")
    if [ -z "${inputfile}" ]; then
        echo "LINE ${lineno}: empty, skipping" >> "${local_logfile}"
        continue
    fi
    if [ ! -f "${inputfile}" ]; then
        echo "LINE ${lineno}: dlmerged ROOT not found: ${inputfile}" >> "${local_logfile}"
        continue
    fi

    nsubdir1=$(( lineno / 1000 ))
    zsubdir1=$(printf "%03d" ${nsubdir1})
    nsubdir2=$(( lineno / 100 ))
    zsubdir2=$(printf "%03d" ${nsubdir2})
    ZFILENO=$(printf "%05d" ${lineno})

    merged_folder="${MERGEFILE_OUTPUT_DIR}/${zsubdir1}/${zsubdir2}"
    flash_folder="${FLASHINFO_OUTPUT_DIR}/${zsubdir1}/${zsubdir2}"
    merged_sentinel="${merged_folder}/${TAG}_fileno${ZFILENO}.complete"
    flash_sentinel="${flash_folder}/${TAG}_fileno${ZFILENO}.flashinfo.complete"

    if [ -f "${flash_sentinel}" ]; then
        echo "LINE ${lineno}: flashinfo sentinel present, skipping (${flash_sentinel})" >> "${local_logfile}"
        continue
    fi
    if [ ! -f "${merged_sentinel}" ]; then
        echo "LINE ${lineno}: merged sentinel MISSING (${merged_sentinel}). Steps 1-4 not complete; skipping." >> "${local_logfile}"
        continue
    fi
    has_merged=$(ls "${merged_folder}"/merged_${TAG}_fileno${ZFILENO}_entry*.h5 2>/dev/null | head -1)
    if [ -z "${has_merged}" ]; then
        echo "LINE ${lineno}: merged sentinel present but no merged_${TAG}_fileno${ZFILENO}_entry*.h5 in ${merged_folder}; skipping" >> "${local_logfile}"
        continue
    fi

    {
        echo ""
        echo "============================================"
        echo "FLASHINFO line ${lineno}: $(basename "${inputfile}")"
        echo "Start: $(date)"
        echo "merged dir: ${merged_folder}"
        echo "output dir: ${flash_folder}"
        echo "============================================"
    } >> "${local_logfile}"
    # for debug
    cat ${local_logfile}

    mkdir -p "${flash_folder}"

    apptainer exec --nv --bind "${POINTCEPT_BIND}" \
        "${POINTCEPT_CONTAINER}" \
        bash -c "source ${SCRIPT_DIR}/run_step5_flashinfo_pointcept_wconfig.sh ${CONFIG_FILE} ${lineno}" \
        >> "${local_logfile}" 2>&1
    step5_status=$?
    echo "Step 5 exit status: ${step5_status}" >> "${local_logfile}"

    # Tally what got produced and decide whether to write the sentinel.
    if [ -d "${flash_folder}" ]; then
        n_flash=$(ls "${flash_folder}"/flashinfo_${TAG}_fileno${ZFILENO}_entry*.h5 2>/dev/null | wc -l)
        n_merged=$(ls "${merged_folder}"/merged_${TAG}_fileno${ZFILENO}_entry*.h5 2>/dev/null | wc -l)
        echo "Counts: flashinfo=${n_flash}  merged=${n_merged}" >> "${local_logfile}"
        if [ "${step5_status}" -eq 0 ] && [ "${n_flash}" -ge "${n_merged}" ] && [ "${n_merged}" -gt 0 ]; then
            touch "${flash_sentinel}"
            echo "Wrote flashinfo sentinel: ${flash_sentinel}" >> "${local_logfile}"
        else
            echo "Sentinel not written (step5_status=${step5_status} n_flash=${n_flash}/${n_merged})" >> "${local_logfile}"
        fi
    fi

    echo "Finish: $(date)" >> "${local_logfile}"
done

{
    echo ""
    echo "======================================"
    echo "Flashinfo job finished: $(date)"
    echo "======================================"
} >> "${local_logfile}"
