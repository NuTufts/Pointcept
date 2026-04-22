#!/bin/bash
#
# run_step2_pointcept.sh
# Runs inside the pointcept container.
# Converts larmatch larlite output to HDF5 with ghost scores.
#
# Usage: source run_step2_pointcept.sh <workdir_path> <fileno> <tag> <adcname> <tbflag> <original_rootfile>
#

WORKDIR_PATH=$1
FILENO=$2
FILETAG=$3
ADCNAME=$4
TBFLAG=$5
ORIGINAL_ROOTFILE=$6

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

cd ${WORKDIR_PATH}
echo "Step 2 workdir: $(pwd)"
echo "Files at start:"
ls -lh

# Setup pointcept container environment
UBDL_DIR=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/ubdl
SCRIPT_DIR=/cluster/tufts/wongjiradlab/twongj01/mphys/Pointcept/lartpc_data_prep/extbnb_larmatch

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

# Output goes to workdir; will be copied to final location by run script
H5_DIR=${WORKDIR_PATH}/h5_output
mkdir -p ${H5_DIR}

# Convert larmatch output to HDF5
echo "----------------------------------------------"
echo "STEP 2: convert_larmatch_to_h5.py"
# Build naming flags
NAMING_FLAGS="--file-id ${FILENO}"
if [ -n "${ORIGINAL_ROOTFILE}" ]; then
    NAMING_FLAGS="${NAMING_FLAGS} --original-filename ${ORIGINAL_ROOTFILE}"
fi

CMD="python3 ${SCRIPT_DIR}/convert_larmatch_to_h5.py \
    -i ${LARLITE_FILE} \
    --input-larcv ${DLMERGED_FILE} \
    -o ${H5_DIR} \
    --adc ${ADCNAME} ${NAMING_FLAGS} ${TBFLAG}"
echo ${CMD}
${CMD}
step2_status=$?
echo "Step 2 exit status: ${step2_status}"

if [ ${step2_status} -ne 0 ]; then
    echo "ERROR: Step 2 failed"
    exit 1
fi

echo "Step 2 output files:"
ls -lh ${H5_DIR}/pointceptdata_*.h5 2>/dev/null

# Clean up ROOT files
rm -f ${WORKDIR_PATH}/larmatchme_larlite.root
rm -f ${WORKDIR_PATH}/merged_dlreco_with_ssnet.root
rm -f ${WORKDIR_PATH}/*.root

echo "Step 2 complete"
