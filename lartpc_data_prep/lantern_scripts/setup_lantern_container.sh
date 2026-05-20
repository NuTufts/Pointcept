#!/bin/bash

source /cluster/home/lantern_scripts/setup_lantern_container.sh

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
cd $LASTDIR
