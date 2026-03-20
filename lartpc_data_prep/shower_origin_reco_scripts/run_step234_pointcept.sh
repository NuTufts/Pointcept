#!/bin/bash
#
# run_step234_pointcept.sh
# Runs inside the pointcept container.
# Sources pointcept environment, then runs Steps 2-4:
#   Step 2: convert_larlite_to_showerorigin_h5.py (reco H5)
#   Step 3: process_dlmerged_to_hdf5_event_files.py (truth H5)
#   Step 4: merge_reco_truth_showerorigin.py (merged H5)
#
# Usage: source run_step234_pointcept.sh <workdir_path> <fileno> <tag>
#   fileno: the input file's line number in the input list, used to make output filenames unique
#   tag: dataset label inserted into merged output filenames
#

WORKDIR_PATH=$1
FILENO=$2
FILETAG=$3

if [ -z "${WORKDIR_PATH}" ]; then
    echo "ERROR: workdir path not provided"
    exit 1
fi

if [ -z "${FILENO}" ]; then
    echo "ERROR: fileno not provided"
    exit 1
fi

if [ -z "${FILETAG}" ]; then
    echo "ERROR: tag not provided"
    exit 1
fi

ZFILENO=$(printf "%05d" ${FILENO})

cd ${WORKDIR_PATH}
echo "Steps 2-4 workdir: $(pwd)"
echo "Files at start:"
ls -lh

# Setup pointcept container environment
UBDL_DIR=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/ubdl
POINTCEPT_DIR=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept
SCRIPT_DIR=${POINTCEPT_DIR}/lartpc_data_prep

cd ${UBDL_DIR}
source setenv_pointcept_container.sh
cd ${WORKDIR_PATH}

echo "Environment setup complete"

# Locate Step 1 output files
LARLITE_FILE="${WORKDIR_PATH}/larmatchme_larlite.root"
DLMERGED_FILE="${WORKDIR_PATH}/merged_dlreco_with_ssnet.root"

if [ ! -f "${LARLITE_FILE}" ]; then
    echo "ERROR: larmatchme_larlite.root not found"
    exit 1
fi
if [ ! -f "${DLMERGED_FILE}" ]; then
    echo "ERROR: merged_dlreco_with_ssnet.root not found"
    exit 1
fi

# Create subdirectories for intermediate outputs
RECO_DIR=${WORKDIR_PATH}/reco_h5
TRUTH_DIR=${WORKDIR_PATH}/truth_h5
MERGED_DIR=${WORKDIR_PATH}  # merged files go directly in workdir for easy copy
mkdir -p ${RECO_DIR} ${TRUTH_DIR}

# ---- STEP 2: Convert larlite to shower origin H5 (reco fragments) ----
echo "----------------------------------------------"
echo "STEP 2: convert_larlite_to_showerorigin_h5.py"
CMD="python3 ${SCRIPT_DIR}/convert_larlite_to_showerorigin_h5.py \
    -i ${LARLITE_FILE} \
    --input-larcv ${DLMERGED_FILE} \
    -o ${RECO_DIR}"
echo ${CMD}
${CMD}
step2_status=$?
echo "Step 2 exit status: ${step2_status}"

if [ ${step2_status} -ne 0 ]; then
    echo "ERROR: Step 2 failed"
    exit 1
fi

echo "Step 2 output files:"
ls -lh ${RECO_DIR}/showerorigin_*.h5 2>/dev/null

# ---- STEP 3: Process dlmerged to truth H5 ----
echo "----------------------------------------------"
echo "STEP 3: process_dlmerged_to_hdf5_event_files.py"
CMD="python3 ${SCRIPT_DIR}/process_dlmerged_to_hdf5_event_files.py \
    -i ${DLMERGED_FILE}"
echo ${CMD}
${CMD}
step3_status=$?
echo "Step 3 exit status: ${step3_status}"

if [ ${step3_status} -ne 0 ]; then
    echo "ERROR: Step 3 failed"
    exit 1
fi

# Step 3 outputs go to current directory by default; move to truth dir
mv ${WORKDIR_PATH}/pointceptdata_*.h5 ${TRUTH_DIR}/ 2>/dev/null

echo "Step 3 output files:"
ls -lh ${TRUTH_DIR}/pointceptdata_*.h5 2>/dev/null

# ---- STEP 4: Merge reco + truth for each event ----
echo "----------------------------------------------"
echo "STEP 4: merge_reco_truth_showerorigin.py"

# Match reco and truth files by entry number
nmerged=0
nfailed=0
for reco_file in ${RECO_DIR}/showerorigin_*.h5; do
    if [ ! -f "${reco_file}" ]; then
        echo "No reco H5 files found"
        break
    fi

    reco_basename=$(basename ${reco_file})
    # Extract entry number from filename: showerorigin_<stem>_entry<NNNNNN>.h5
    entry_part=$(echo ${reco_basename} | grep -oP 'entry\d+')

    if [ -z "${entry_part}" ]; then
        echo "WARNING: could not extract entry from ${reco_basename}, skipping"
        continue
    fi

    # Find matching truth file
    truth_file=$(ls ${TRUTH_DIR}/pointceptdata_*_${entry_part}.h5 2>/dev/null | head -1)

    if [ -z "${truth_file}" ] || [ ! -f "${truth_file}" ]; then
        echo "WARNING: no matching truth file for ${entry_part}, skipping"
        let nfailed+=1
        continue
    fi

    # Output merged filename — includes tag and fileno to avoid collisions across input files/datasets
    merged_outfile=${MERGED_DIR}/merged_${FILETAG}_fileno${ZFILENO}_${entry_part}.h5

    CMD="python3 ${SCRIPT_DIR}/merge_reco_truth_showerorigin.py \
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
ls -lh ${MERGED_DIR}/merged_*.h5 2>/dev/null

# Clean up intermediate files (keep only merged files)
rm -rf ${RECO_DIR} ${TRUTH_DIR}
rm -f ${WORKDIR_PATH}/larmatchme_larlite.root
rm -f ${WORKDIR_PATH}/merged_dlreco_with_ssnet.root
rm -f ${WORKDIR_PATH}/*.root

echo "Steps 2-4 complete"
