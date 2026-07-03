#!/bin/bash
#
# run_step567_pointcept.sh
# Runs inside the pointcept container.
# Sources pointcept environment, then runs Steps 5-7:
#   Step 5: Shower origin model inference on all fragments
#   Step 6: Fragment merging into reconstructed showers
#   Step 7: ROOT TTree output via uproot
#
# Usage: source run_step567_pointcept.sh <workdir_path> <fileno> <tag>
#
# Expects merged_*.h5 files in workdir_path (output of Steps 2-4).
# Produces one ROOT file per merged H5 file.
#
# Environment variables (set by caller or defaults used):
#   SHOWER_ORIGIN_CONFIG  - model config file path
#   SHOWER_ORIGIN_CKPT    - model checkpoint path
#   DEVICE                - cuda or cpu (default: cpu)
#   ORIGIN_PROXIMITY_RADIUS - cm (default: 5.0)
#   CONE_HALF_ANGLE         - degrees (default: 20.0)
#   CONE_MAX_LENGTH         - cm (default: 300.0)
#   MIN_POINTS_FRACTION_IN_CONE - 0-1 (default: 0.5)
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

cd ${WORKDIR_PATH}
echo "Steps 5-7 workdir: $(pwd)"

# Setup pointcept container environment
UBDL_DIR=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/ubdl
POINTCEPT_DIR=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept

cd ${UBDL_DIR}
source setenv_pointcept_container.sh
cd ${WORKDIR_PATH}

echo "Environment setup complete"

# Default paths (can be overridden by env vars)
SHOWER_ORIGIN_CONFIG=${SHOWER_ORIGIN_CONFIG:-${POINTCEPT_DIR}/configs/lartpc/shower_origin/archive/shower-origin-sonata-v1m1-v3-reco-fragments-p1cmp075.py}
SHOWER_ORIGIN_CKPT=${SHOWER_ORIGIN_CKPT:-${POINTCEPT_DIR}/shower_origin/sonata_v1m1_v3_v6backbone_pax_pi0filter_recofragments/model/model_epoch165.pth}
DEVICE=${DEVICE:-cuda}

# Merger parameters (defaults match MergerConfig)
ORIGIN_PROXIMITY_RADIUS=${ORIGIN_PROXIMITY_RADIUS:-5.0}
CONE_HALF_ANGLE=${CONE_HALF_ANGLE:-20.0}
CONE_MAX_LENGTH=${CONE_MAX_LENGTH:-300.0}
MIN_POINTS_FRACTION_IN_CONE=${MIN_POINTS_FRACTION_IN_CONE:-0.5}

echo "Config: ${SHOWER_ORIGIN_CONFIG}"
echo "Checkpoint: ${SHOWER_ORIGIN_CKPT}"
echo "Device: ${DEVICE}"
echo "Merger params: radius=${ORIGIN_PROXIMITY_RADIUS}, angle=${CONE_HALF_ANGLE}, length=${CONE_MAX_LENGTH}, frac=${MIN_POINTS_FRACTION_IN_CONE}"

# Check for merged H5 files
MERGED_FILES=$(ls ${WORKDIR_PATH}/merged_*.h5 2>/dev/null)
if [ -z "${MERGED_FILES}" ]; then
    echo "No merged H5 files found in ${WORKDIR_PATH}, skipping Steps 5-7"
    exit 0
fi

nfiles=$(echo "${MERGED_FILES}" | wc -l)
echo "Found ${nfiles} merged H5 file(s) to process"

# Create temporary file list for the pipeline script
FILELIST=${WORKDIR_PATH}/step567_filelist.txt
echo "${MERGED_FILES}" > ${FILELIST}

# Create output directory for ROOT files
ROOT_OUTDIR=${WORKDIR_PATH}/root_output
mkdir -p ${ROOT_OUTDIR}

# ---- STEPS 5-7: inference + merging + ROOT output ----
echo "----------------------------------------------"
echo "STEPS 5-7: shower origin pipeline"
CMD="python3 ${POINTCEPT_DIR}/tools/run_shower_origin_pipeline_step567.py \
    -c ${SHOWER_ORIGIN_CONFIG} \
    --checkpoint ${SHOWER_ORIGIN_CKPT} \
    --input-list ${FILELIST} \
    --output-dir ${ROOT_OUTDIR} \
    --device ${DEVICE} \
    --origin-proximity-radius ${ORIGIN_PROXIMITY_RADIUS} \
    --cone-half-angle ${CONE_HALF_ANGLE} \
    --cone-max-length ${CONE_MAX_LENGTH} \
    --min-points-fraction-in-cone ${MIN_POINTS_FRACTION_IN_CONE}"
echo ${CMD}
${CMD}
step567_status=$?
echo "Steps 5-7 exit status: ${step567_status}"

if [ ${step567_status} -ne 0 ]; then
    echo "ERROR: Steps 5-7 failed"
    exit 1
fi

# Verify ROOT output
nroot=$(ls ${ROOT_OUTDIR}/showerreco_*.root 2>/dev/null | wc -l)
echo "ROOT files produced: ${nroot}"
ls -lh ${ROOT_OUTDIR}/showerreco_*.root 2>/dev/null

# Move ROOT files to workdir root for easy pickup by the caller
mv ${ROOT_OUTDIR}/showerreco_*.root ${WORKDIR_PATH}/ 2>/dev/null

# Clean up
rm -rf ${ROOT_OUTDIR}
rm -f ${FILELIST}

echo "Steps 5-7 complete"
