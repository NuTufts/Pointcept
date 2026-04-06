#!/bin/bash
#
# check_status.sh
# Check completion status of extbnb larmatch processing.
# Reports how many input files have produced H5 output.
#
# Usage: bash check_status.sh [--rerun rerun_list.txt]
#

INPUTLIST=/cluster/tufts/wongjiradlab/hmcgui01/mphys/Pointcept/lartpc_data_prep/inputlists/inputlist_run3_G1_extbnb_dlreco.txt
OUTPUT_DIR=/cluster/tufts/wongjiradlab/larbys/data/ub_on_tufts/hdf5/v3_ext_larmatch/extbnb_run3_G1/

RERUN_FILE=""
if [ "$1" == "--rerun" ] && [ -n "$2" ]; then
    RERUN_FILE=$2
    > ${RERUN_FILE}
fi

total=0
found=0
missing=0

nlines=$(wc -l < ${INPUTLIST})

for lineno in $(seq 1 ${nlines}); do
    inputfile=$(sed -n ${lineno}p ${INPUTLIST})
    if [ -z "${inputfile}" ]; then
        continue
    fi

    total=$((total + 1))

    # Compute expected output path
    let nsubdir1=$(expr "${lineno}/1000")
    zsubdir1=$(printf %03d ${nsubdir1})
    let nsubdir2=$(expr "${lineno}/100")
    zsubdir2=$(printf %03d ${nsubdir2})

    outfolder=${OUTPUT_DIR}/${zsubdir1}/${zsubdir2}/

    # Check if any H5 files exist for this input
    nh5=$(ls ${outfolder}/pointceptdata_*_entry*.h5 2>/dev/null | wc -l)

    if [ ${nh5} -gt 0 ]; then
        found=$((found + 1))
    else
        missing=$((missing + 1))
        if [ -n "${RERUN_FILE}" ]; then
            echo "${lineno}" >> ${RERUN_FILE}
        fi
    fi

    # Progress update every 1000 files
    if [ $((total % 1000)) -eq 0 ]; then
        echo "Checked ${total}/${nlines}: found=${found}, missing=${missing}"
    fi
done

echo ""
echo "========================================="
echo "Total input files: ${total}"
echo "Completed (H5 found): ${found}"
echo "Missing/failed: ${missing}"
echo "========================================="

if [ -n "${RERUN_FILE}" ]; then
    echo "Rerun list written to: ${RERUN_FILE} (${missing} entries)"
fi
