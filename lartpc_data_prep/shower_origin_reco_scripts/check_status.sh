#!/bin/bash
#
# check_status.sh
# Check completion status of shower origin reco jobs.
# Reports total files, completed, and missing/failed.
# Optionally writes a rerun file list.
#
# Usage: bash check_status.sh [--rerun rerun_list.txt]
#

POINTCEPT_DIR=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept
INPUTLIST=${POINTCEPT_DIR}/lartpc_data_prep/inputlists/bnb_nu_pi0filter_corsika.txt
OUTPUT_DIR=/cluster/tufts/wongjiradlab/larbys/data/ub_on_tufts/hdf5/shower_origin_reco/bnb_nu_pi0filter_corsika

RERUN_FILE=""
if [ "$1" == "--rerun" ] && [ -n "$2" ]; then
    RERUN_FILE=$2
    > ${RERUN_FILE}  # clear file
fi

total=0
completed=0
missing=0

nlines=$(wc -l < ${INPUTLIST})

echo "Checking status of shower origin reco production"
echo "Input list: ${INPUTLIST}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Total lines in input list: ${nlines}"
echo "============================================"

for (( lineno=1; lineno<=nlines; lineno++ )); do

    inputfile=$(sed -n ${lineno}p ${INPUTLIST})
    if [ -z "${inputfile}" ]; then
        continue
    fi

    let total+=1

    # Compute expected output subdirectory
    let nsubdir1=$(expr "${lineno}/1000")
    zsubdir1=$(printf %03d ${nsubdir1})
    let nsubdir2=$(expr "${lineno}/100")
    zsubdir2=$(printf %03d ${nsubdir2})
    outfolder=${OUTPUT_DIR}/${zsubdir1}/${zsubdir2}

    # Check if any merged H5 files exist for this input
    nmerged=$(ls ${outfolder}/merged_*.h5 2>/dev/null | wc -l)

    if [ ${nmerged} -gt 0 ]; then
        let completed+=1
    else
        let missing+=1
        if [ -n "${RERUN_FILE}" ]; then
            echo "${lineno}" >> ${RERUN_FILE}
        fi
    fi

done

echo ""
echo "============================================"
echo "RESULTS"
echo "============================================"
echo "Total input files:  ${total}"
echo "Completed:          ${completed}"
echo "Missing/Failed:     ${missing}"

if [ -n "${RERUN_FILE}" ]; then
    echo ""
    echo "Rerun list written to: ${RERUN_FILE}"
    echo "Lines in rerun list: $(wc -l < ${RERUN_FILE})"
fi
