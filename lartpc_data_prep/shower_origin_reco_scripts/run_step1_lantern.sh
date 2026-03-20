#!/bin/bash
#
# run_step1_lantern.sh
# Runs inside the lantern container.
# Sources lantern environment, then runs the lantern workflow on the
# input ROOT file already copied into the workdir.
#
# Usage: source run_step1_lantern.sh <workdir_path>
#

WORKDIR_PATH=$1

if [ -z "${WORKDIR_PATH}" ]; then
    echo "ERROR: workdir path not provided"
    exit 1
fi

cd ${WORKDIR_PATH}
echo "Step 1 workdir: $(pwd)"
echo "Files at start:"
ls -lh

# Setup lantern environment
echo "Setting up LANTERN environment"
shopt -s expand_aliases
alias python=python3

export UBDL_DIR=/cluster/home/ubdl/
export RECO_TEST_DIR=/cluster/home/ubdl/larflow/larflow/Reco/test/
export LARMATCH_DIR=/cluster/home/ubdl/larflow/larmatchnet/larmatch/
export SSNET_DIR=/cluster/home/uresnet_pytorch/
export NTMAKER_DIR=/cluster/home/gen2ntuple/
export LARPID_DIR=/cluster/home/prongCNN/models/checkpoints/
export LANTERN_SCRIPTS=/cluster/home/lantern_scripts/

source ${UBDL_DIR}/setenv_py3_container.sh
source ${UBDL_DIR}/configure_container.sh

LASTDIR="$(pwd)"
cd ${UBDL_DIR}/larflow/larmatchnet
source set_pythonpath.sh
export PYTHONPATH=${LARMATCH_DIR}:${PYTHONPATH}
export PYTHONPATH=${PYTHONPATH}:${NTMAKER_DIR}
export PYTHONPATH=${PYTHONPATH}:${SSNET_DIR}
cd ${LASTDIR}

echo "Environment setup complete"

# Find the input merged_dlreco ROOT file
larsoftout=$(find . -maxdepth 1 -type f -name 'merged_dlreco*.root' -printf '%f\n' | head -1)
if [ -z "${larsoftout}" ]; then
    larsoftout=$(find . -maxdepth 1 -type f -name 'dlmerged*.root' -printf '%f\n' | head -1)
fi

if [ -z "${larsoftout}" ]; then
    echo "ERROR: No input ROOT file found in workdir"
    ls -la
    exit 1
fi

echo "Input file: ${larsoftout}"

baseinput="merged_dlreco_with_ssnet.root"
CONFIG_FILE="${LANTERN_SCRIPTS}/config_larmatchme_deploycpu.yaml"
WEIGHT_FILE="larmatch_ckpt78k.pt"
INPUTSTEM="merged_dlreco_with_ssnet"

# Derived names
lm_outfile=$(echo ${baseinput} | sed "s|${INPUTSTEM}|larmatchme|g")
baselm=$(echo ${baseinput} | sed "s|${INPUTSTEM}|larmatchme|g" | sed 's|.root|_larlite.root|g')

echo "larmatch outfile: ${lm_outfile}"
echo "baselm: ${baselm}"

# Copy local (bug-fixed) lantern scripts to workdir.
# The container versions use "sparsessnet" as the sparseimg producer/tree name,
# but these local copies use "sparseuresnetout" which fixes a naming mismatch bug.
LOCAL_LANTERN_SCRIPTS=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/ubdl/lantern_scripts
cp ${LOCAL_LANTERN_SCRIPTS}/inference_sparse_ssnet_uboone.py .
cp ${LOCAL_LANTERN_SCRIPTS}/recreate_ubspurn.py .
echo "Copied local lantern scripts to workdir"

# ---- SSNet inference (using local bug-fixed script) ----
echo "----------------------------------------------"
echo "Running SSNet inference"
CMD="python3 ./inference_sparse_ssnet_uboone.py -i ${larsoftout} -w ${SSNET_DIR}/weights -o ssnet_output.root --adcname wiremc"
echo ${CMD}
${CMD}

if [ $? -ne 0 ]; then
    echo "ERROR: SSNet inference failed"
    exit 1
fi

# ---- Merge SSNet output ----
echo "Merging SSNet output with merged_dlreco..."
cp ${larsoftout} ${baseinput}
rootcp ssnet_output.root:sparseimg_sparseuresnetout_tree ${baseinput}
python3 ./recreate_ubspurn.py -i ${baseinput} -o ssnet_ubspurn_output.root --adcname wiremc
inputTrees=$(rootls ssnet_ubspurn_output.root)
for tree in ${inputTrees}; do
    rootcp ssnet_ubspurn_output.root:${tree} ${baseinput}
done

# ---- LArMatch deploy ----
echo "----------------------------------------------"
echo "Running LArMatch deploy"
CMD="python3 ${LARMATCH_DIR}/deploy_larmatchme.py --config-file ${CONFIG_FILE} --supera ${baseinput} --weights ${LARMATCH_DIR}/${WEIGHT_FILE} --output ${lm_outfile} --min-score 0.5 --adc-name wire --chstatus-name wire --device-name cpu --use-skip-limit"
echo ${CMD}
${CMD}

if [ $? -ne 0 ]; then
    echo "ERROR: LArMatch deploy failed"
    exit 1
fi

# Clean up intermediate files (keep larmatchme_larlite.root and merged_dlreco_with_ssnet.root)
rm -f ssnet_output.root ssnet_ubspurn_output.root
rm -f larmatchme_larcv.root

echo "Step 1 complete. Files in workdir:"
ls -lh
