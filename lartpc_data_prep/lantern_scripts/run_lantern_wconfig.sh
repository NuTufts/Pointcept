#!/bin/bash
#
# run_lantern_wconfig.sh
#
# Usage:
#   source run_lantern_wconfig.sh <config_file>
#
# All pipeline options live in a sourced bash config (see example lantern_configs/bnbnu_coriska.conf)
# so the file can use ${REPO_ROOT}, ${UBDL_DIR}, ${POINTCEPT_DIR} for interpolation.
#
# Dev-knob options read from the config:
#   WORKDIR_BASE         parent dir for per-file workdirs (default /tmp)
#   KEEP_INTERMEDIATES   0/1; if 1, do not delete workdir and skip any stage
#                        whose canonical outputs are already present
#   MAX_EVENTS           int; passed as --nentries to Step 1/2/3 entrypoints (-1 = all)
#

CONFIG_FILE="${1:-${CONFIG_FILE}}"
if [ -z "${CONFIG_FILE}" ]; then
    echo "ERROR: no config file provided. Usage: source run_showerorigin_reco_wconfig.sh <config_file>" >&2
    return 1 2>/dev/null || exit 1
fi
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "ERROR: config file not found: ${CONFIG_FILE}" >&2
    return 1 2>/dev/null || exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}" && pwd)
UBDL_DIR=${UBDL_DIR:-/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/ubdl}
POINTCEPT_DIR=${POINTCEPT_DIR:-/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept}
export SCRIPT_DIR REPO_ROOT UBDL_DIR POINTCEPT_DIR

# shellcheck disable=SC1090
echo "Sourcing config file: ${CONFIG_FILE}"
source "${CONFIG_FILE}"

module load apptainer/1.2.4-suid
cvmfs_config probe uboone.opensciencegrid.org

WORKDIR_BASE=${WORKDIR_BASE:-/tmp}
KEEP_INTERMEDIATES=${KEEP_INTERMEDIATES:-0}
MAX_EVENTS=${MAX_EVENTS:--1}
# Cap on larmatch hits per event passed to Step 2; oversized events become
# empty placeholders. Configs may override; <=0 disables the cap.
MAX_HITS=${MAX_HITS:-1000000}
stride=${stride:-1}
OFFSET=${OFFSET:-0}
ADCNAME=${ADCNAME:-wiremc}
TBFLAG=${TBFLAG:-""}
MCC9FLAG=${MCC9FLAG:-""}
DEVICE=${DEVICE:-cuda}
SAVE_INFERENCE_H5=${SAVE_INFERENCE_H5:-0}
INFERENCE_H5_OUTPUT_DIR=${INFERENCE_H5_OUTPUT_DIR:-${OUTPUT_DIR}}
MERGEFILE_OUTPUT_DIR=${MERGEFILE_OUTPUT_DIR:-${OUTPUT_DIR}}
SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

for v in INPUTLIST TAG OUTPUT_DIR LANTERN_CONTAINER POINTCEPT_CONTAINER SHOWER_ORIGIN_CONFIG SHOWER_ORIGIN_CKPT; do
    if [ -z "${!v}" ]; then
        echo "ERROR: config must set ${v}" >&2
        return 1 2>/dev/null || exit 1
    fi
done

export TAG INPUTLIST OUTPUT_DIR MERGEFILE_OUTPUT_DIR ROOT_OUTPUT_DIR ADCNAME TBFLAG MCC9FLAG DEVICE
export LANTERN_CONTAINER POINTCEPT_CONTAINER
export SHOWER_ORIGIN_CONFIG SHOWER_ORIGIN_CKPT
export WORKDIR_BASE KEEP_INTERMEDIATES MAX_EVENTS MAX_HITS
export SAVE_INFERENCE_H5 INFERENCE_H5_OUTPUT_DIR

# Apptainer bind lists — add WORKDIR_BASE if outside the default tree
LANTERN_BIND="/cluster/tufts:/cluster/tufts"
case "${WORKDIR_BASE}" in
    /cluster/tufts*) ;;
    *) LANTERN_BIND="${LANTERN_BIND},${WORKDIR_BASE}:${WORKDIR_BASE}" ;;
esac
POINTCEPT_BIND="/cluster:/cluster"
case "${WORKDIR_BASE}" in
    /cluster*) ;;
    *) POINTCEPT_BIND="${POINTCEPT_BIND},${WORKDIR_BASE}:${WORKDIR_BASE}" ;;
esac

jobid=${SLURM_ARRAY_TASK_ID}
startline=$(( OFFSET + stride * jobid ))

# RERUN_LINES_FILE: optional rerun mode. When set, each array task pulls
# `stride` original line numbers from this file at positions
# stride*jobid+1 .. stride*jobid+stride (1-indexed), instead of computing
# lineno = OFFSET + stride*jobid + i. The line numbers are the original
# 1-indexed lines of INPUTLIST, so the hashed output dir, sentinel path,
# and START_ENTRY logic all resolve identically to the first run.
# Generate the file with:
#   bash write_completion_sentinels.sh <config> --report-only --write-rerun-list <path>
RERUN_LINES_FILE=${RERUN_LINES_FILE:-}
if [ -n "${RERUN_LINES_FILE}" ] && [ ! -f "${RERUN_LINES_FILE}" ]; then
    echo "ERROR: RERUN_LINES_FILE set but not found: ${RERUN_LINES_FILE}" >&2
    return 1 2>/dev/null || exit 1
fi
export RERUN_LINES_FILE

jobworkdir=$(printf "%s/workdir/${TAG}_jobid_%04d" "${REPO_ROOT}" "${jobid}")
mkdir -p "${jobworkdir}"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${MERGEFILE_OUTPUT_DIR}"


local_logfile=${jobworkdir}/log_${TAG}_jobid${jobid}.txt
{
    echo "======================================"
    echo "Job started: $(date)"
    echo "CONFIG_FILE:        ${CONFIG_FILE}"
    echo "SLURM_ARRAY_TASK_ID: ${jobid}"
    echo "startline:          ${startline}"
    echo "stride:             ${stride}"
    echo "WORKDIR_BASE:       ${WORKDIR_BASE}"
    echo "KEEP_INTERMEDIATES: ${KEEP_INTERMEDIATES}"
    echo "MAX_EVENTS:         ${MAX_EVENTS}"
    echo "RERUN_LINES_FILE:   ${RERUN_LINES_FILE:-(unset, normal sequential mode)}"
    echo "======================================"
} > "${local_logfile}"

for (( i=1; i<=stride; i++ )); do
    if [ -n "${RERUN_LINES_FILE}" ]; then
        # Rerun mode: pull the original lineno from RERUN_LINES_FILE at
        # position stride*jobid + i (1-indexed). Past end of file ⇒ skip.
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
        echo "LINE ${lineno}: file not found: ${inputfile}" >> "${local_logfile}"
        continue
    fi

    inputbasename=$(basename "${inputfile}")

    # ---- Resume support ----
    # Compute output folder and per-file completion sentinel up front so we
    # can (a) skip the whole line if the previous run wrote the sentinel,
    # or (b) start Steps 2-3 at the first entry index that isn't already in
    # MERGEFILE_OUTPUT_DIR. The hashed dir layout matches the cp at the
    # end of the loop.
    nsubdir1=$(( lineno / 1000 ))
    zsubdir1=$(printf "%03d" ${nsubdir1})
    nsubdir2=$(( lineno / 100 ))
    zsubdir2=$(printf "%03d" ${nsubdir2})
    ZFILENO=$(printf "%05d" ${lineno})
    outfolder="${MERGEFILE_OUTPUT_DIR}/${zsubdir1}/${zsubdir2}"
    sentinel="${outfolder}/${TAG}_fileno${ZFILENO}.complete"

    if [ -f "${sentinel}" ]; then
        echo "LINE ${lineno}: completion sentinel present, skipping (${sentinel})" >> "${local_logfile}"
        continue
    fi

    # Find the first missing entry index from existing merged_*.h5 outputs.
    # If we have a contiguous run 0..K-1, the next index is K. If there's a
    # gap (e.g. missing entry 3 with 0,1,2,4,5 present), restart at the gap.
    START_ENTRY=0
    if [ -d "${outfolder}" ]; then
        existing_entries=$(ls "${outfolder}/merged_${TAG}_fileno${ZFILENO}_entry"*.h5 2>/dev/null \
                            | sed -E 's|.*entry([0-9]+)\.h5$|\1|' \
                            | sort -n -u)
        expected=0
        for n in ${existing_entries}; do
            n10=$((10#$n))
            if [ ${n10} -ne ${expected} ]; then
                break
            fi
            expected=$(( expected + 1 ))
        done
        START_ENTRY=${expected}
    fi
    export START_ENTRY

    {
        echo ""
        echo "============================================"
        echo "Processing line ${lineno}: ${inputbasename}"
        echo "Start time: $(date)"
        echo "Output folder: ${outfolder}"
        echo "START_ENTRY:   ${START_ENTRY} (existing entries inform Steps 2-3 to skip)"
        echo "============================================"
    } >> "${local_logfile}"

    local_jobdir=$(printf "%s/lantern_${TAG}_jobid%04d_line%05d" "${WORKDIR_BASE}" ${jobid} ${lineno})
    if [ "${KEEP_INTERMEDIATES}" != "1" ]; then
        rm -rf "${local_jobdir}"
    fi
    mkdir -p "${local_jobdir}"

    if [ ! -f "${local_jobdir}/${inputbasename}" ]; then
        echo "Copying input file to ${local_jobdir}" >> "${local_logfile}"
        cp "${inputfile}" "${local_jobdir}/"
    else
        echo "Input file already present in workdir, skipping copy" >> "${local_logfile}"
    fi

    # ---- STEP 1 (lantern) ----
    if [ "${KEEP_INTERMEDIATES}" = "1" ] \
       && [ -f "${local_jobdir}/larmatchme_larlite.root" ] \
       && [ -f "${local_jobdir}/merged_dlreco_with_ssnet.root" ]; then
        echo "STEP 1: skipped (outputs present, KEEP_INTERMEDIATES=1)" >> "${local_logfile}"
    else
        echo "--- STEP 1: lantern workflow ---" >> "${local_logfile}"
        # apptainer exec --bind "${LANTERN_BIND}" \
        #     "${LANTERN_CONTAINER}" \
        #     bash -c "source ${SCRIPT_DIR}/run_step1_lantern_wconfig.sh ${local_jobdir} ${ADCNAME} ${TBFLAG} ${MAX_EVENTS} ${KEEP_INTERMEDIATES}" \
        #     >> "${local_logfile}" 2>&1
        apptainer exec --bind "${LANTERN_BIND}" \
            "${LANTERN_CONTAINER}" \
            bash -c "source ${SCRIPT_DIR}/run_step1_lantern_wconfig.sh ${CONFIG_FILE} ${lineno}" \
            >> "${local_logfile}" 2>&1            
        step1_status=$?
        echo "Step 1 exit status: ${step1_status}" >> "${local_logfile}"
        if [ ${step1_status} -ne 0 ]; then
            echo "Step 1 FAILED for line ${lineno}" >> "${local_logfile}"
            [ "${KEEP_INTERMEDIATES}" != "1" ] && rm -rf "${local_jobdir}"
            continue
        fi
    fi

    if [ ! -f "${local_jobdir}/larmatchme_larlite.root" ] || [ ! -f "${local_jobdir}/merged_dlreco_with_ssnet.root" ]; then
        echo "Step 1 output files missing, skipping remaining steps" >> "${local_logfile}"
        ls -la "${local_jobdir}/" >> "${local_logfile}"
        [ "${KEEP_INTERMEDIATES}" != "1" ] && rm -rf "${local_jobdir}"
        continue
    fi

    # ---- STEPS 2-4 (pointcept) ----
    has_merged=$(ls "${local_jobdir}"/merged_*.h5 2>/dev/null | head -1)
    if [ "${KEEP_INTERMEDIATES}" = "1" ] && [ -n "${has_merged}" ]; then
        echo "STEPS 2-4: skipped (merged_*.h5 present, KEEP_INTERMEDIATES=1)" >> "${local_logfile}"
    else
        echo "--- STEPS 2-4: pointcept processing ---" >> "${local_logfile}"
        # apptainer exec --nv --bind "${POINTCEPT_BIND}" \
        #     "${POINTCEPT_CONTAINER}" \
        #     bash -c "source ${SCRIPT_DIR}/run_step234_pointcept_wconfig.sh ${local_jobdir} ${lineno} ${TAG} ${ADCNAME} ${TBFLAG} ${MCC9FLAG} ${MAX_EVENTS} ${KEEP_INTERMEDIATES}" \
        #     >> "${local_logfile}" 2>&1
        apptainer exec --nv --bind "${POINTCEPT_BIND}" \
            "${POINTCEPT_CONTAINER}" \
            bash -c "source ${SCRIPT_DIR}/run_step234_pointcept_wconfig.sh ${CONFIG_FILE} ${lineno}" \
            >> "${local_logfile}" 2>&1
        step234_status=$?
        echo "Steps 2-4 exit status: ${step234_status}" >> "${local_logfile}"
        if [ ${step234_status} -ne 0 ]; then
            echo "Steps 2-4 FAILED for line ${lineno}" >> "${local_logfile}"
            [ "${KEEP_INTERMEDIATES}" != "1" ] && rm -rf "${local_jobdir}"
            continue
        fi
    fi

    # # ---- STEPS 5-7 (pointcept) ----
    # has_root=$(ls "${local_jobdir}"/showerreco_*.root 2>/dev/null | head -1)
    # if [ "${KEEP_INTERMEDIATES}" = "1" ] && [ -n "${has_root}" ]; then
    #     echo "STEPS 5-7: skipped (showerreco_*.root present, KEEP_INTERMEDIATES=1)" >> "${local_logfile}"
    #     step567_status=0
    # else
    #     echo "--- STEPS 5-7: shower origin pipeline ---" >> "${local_logfile}"
    #     apptainer exec --nv --bind "${POINTCEPT_BIND}" \
    #         "${POINTCEPT_CONTAINER}" \
    #         bash -c "source ${SCRIPT_DIR}/run_step567_pointcept_wconfig.sh ${local_jobdir} ${lineno} ${TAG} ${KEEP_INTERMEDIATES}" \
    #         >> "${local_logfile}" 2>&1
    #     step567_status=$?
    #     echo "Steps 5-7 exit status: ${step567_status}" >> "${local_logfile}"
    #     if [ ${step567_status} -ne 0 ]; then
    #         echo "Steps 5-7 FAILED for line ${lineno} (H5 output still available)" >> "${local_logfile}"
    #     fi
    # fi

    mkdir -p "${outfolder}"
    nmerged=$(ls "${local_jobdir}"/merged_*.h5 2>/dev/null | wc -l)
    echo "Number of merged H5 files: ${nmerged}" >> "${local_logfile}"
    if [ "${nmerged}" -gt 0 ]; then
        cp "${local_jobdir}"/merged_*.h5 "${outfolder}/"
        ls -lh "${outfolder}"/merged_*.h5 >> "${local_logfile}" 2>/dev/null
    else
        echo "WARNING: No merged H5 files produced for line ${lineno}" >> "${local_logfile}"
    fi

    # Mark the line complete so future jobs can skip it cheaply. We write
    # the sentinel both when the run produced new merged files and when a
    # resume run produced none (Step 2 hit input EOF immediately, meaning
    # OUTPUT_DIR already had every entry the input file contains).
    if [ "${nmerged}" -gt 0 ] || [ "${START_ENTRY}" -gt 0 ]; then
        touch "${sentinel}"
        echo "Wrote completion sentinel: ${sentinel}" >> "${local_logfile}"
    fi

    # nroot=$(ls "${local_jobdir}"/showerreco_*.root 2>/dev/null | wc -l)
    # if [ "${nroot}" -gt 0 ] && [ -n "${ROOT_OUTPUT_DIR}" ]; then
    #     mkdir -p "${ROOT_OUTPUT_DIR}/${zsubdir1}/${zsubdir2}/"
    #     cp "${local_jobdir}"/showerreco_*.root "${ROOT_OUTPUT_DIR}/${zsubdir1}/${zsubdir2}/"
    #     echo "Copied ${nroot} ROOT file(s) to ${ROOT_OUTPUT_DIR}/${zsubdir1}/${zsubdir2}/" >> "${local_logfile}"
    # fi

    # if [ "${SAVE_INFERENCE_H5}" = "1" ]; then
    #     ninference=$(ls "${local_jobdir}"/showerinference_*.h5 2>/dev/null | wc -l)
    #     if [ "${ninference}" -gt 0 ] && [ -n "${INFERENCE_H5_OUTPUT_DIR}" ]; then
    #         inf_outfolder=${INFERENCE_H5_OUTPUT_DIR}/${zsubdir1}/${zsubdir2}/
    #         mkdir -p "${inf_outfolder}"
    #         cp "${local_jobdir}"/showerinference_*.h5 "${inf_outfolder}/"
    #         echo "Copied ${ninference} inference H5 file(s) to ${inf_outfolder}" >> "${local_logfile}"
    #         ls -lh "${inf_outfolder}"/showerinference_*.h5 >> "${local_logfile}" 2>/dev/null
    #     fi
    # fi

    echo "Finish time: $(date)" >> "${local_logfile}"

    if [ "${KEEP_INTERMEDIATES}" != "1" ]; then
        rm -rf "${local_jobdir}"
    else
        echo "KEEP_INTERMEDIATES=1 — workdir preserved at ${local_jobdir}" >> "${local_logfile}"
    fi
done

{
    echo ""
    echo "======================================"
    echo "Job finished: $(date)"
    echo "======================================"
} >> "${local_logfile}"
