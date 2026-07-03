#!/bin/bash
shopt -s extdebug

# WE NEED TO SETUP ENV VARIABLES FOR ROOT, CUDA, OPENCV
alias python=python3

#Force use of container options
#(needed if building ubdl in container recipe on one of the named machines in setenv_py3.sh)

echo "DEFAULT SETUP (COMPAT WITH SINGULARITY CONTAINER)"

# ROOT
# Source thisroot.sh from /opt/root to avoid shell-detection issues
# (getTrueShellExeName fails on some login nodes, so we cd there instead)
pushd /opt/root > /dev/null && source bin/thisroot.sh 2>/dev/null && popd > /dev/null

# CUDA
export CUDA_HOME=/usr/local/cuda/
[[ ":$LD_LIBRARY_PATH:" != *":${CUDA_HOME}/lib64:"* ]] && export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# OPENCV
export OPENCV_INCDIR=/usr/include/opencv4/
export OPENCV_LIBDIR=/usr/lib/x86_64-linux-gnu/

# UBDL
UBDL_BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Build against libtorch built with c++11 ABI standard
# This is different than the version in python3.8
# Careful about mixing these up.
#export LIBTORCH_DIR=/usr/local/libtorch1.9.0_cxx11abi/libtorch
#export LIBTORCH_LIBRARY_DIR=${LIBTORCH_DIR}/lib
#export LIBTORCH_CMAKE_DIR=${LIBTORCH_DIR}/share/cmake/Torch
#export LIBTORCH_BIN_DIR=${LIBTORCH_DIR}/bin

#[[ ":$LD_LIBRARY_PATH:" != *":${LIBTORCH_LIBRARY_DIR}:"* ]] && LD_LIBRARY_PATH="${LIBTORCH_LIBRARY_DIR}:${LD_LIBRARY_PATH}"
#[[ ":$PATH:" != *":${LIBTORCH_BIN_DIR}:"* ]] && PATH="${LIBTORCH_BIN_DIR}:${PATH}"


# Add prongCNN folder
#export PRONGCNN_DIR=${UBDL_BASEDIR}/prongCNN
#export LARPID_DIR=${PRONGCNN_DIR}/larpid/build/installed
#export LARPID_LIBDIR=${LARPID_DIR}/lib
#export LARPID_INCDIR=${LARPID_DIR}/include
#[[ ":$LD_LIBRARY_PATH:" != *":${LARPID_LIBDIR}:"* ]] && export LD_LIBRARY_PATH="${LARPID_LIBDIR}:${LD_LIBRARY_PATH}"
#[[ ":$LD_LIBRARY_PATH:" != *":${LARPID_LIBDIR}:"* ]] && export LD_LIBRARY_PATH="${LARPID_LIBDIR}:${LD_LIBRARY_PATH}"
#[[ ":$PATH:" != *":${LARPID_BINDIR}:"* ]] && export PATH="${LARFLOW_BINDIR}:${PATH}"
#[[ ":${PYTHONPATH}:" != *":${PRONGCNN_DIR}:"* ]] && export PYTHONPATH="${PRONGCNN_DIR}:${PYTHONPATH}"
#[[ ":${PYTHONPATH}:" != *":${PRONGCNN_DIR}/models:"* ]] && export PYTHONPATH="${PRONGCNN_DIR}/models:${PYTHONPATH}"

source configure_container.sh

# Put conda lib in front
export LD_LIBRARY_PATH=/opt/conda/lib/:${LD_LIBRARY_PATH}

# Suppress HDF5 version check - needed because system OpenCV links to GDAL
# which pulls in system HDF5 (1.10.7), conflicting with conda HDF5 (1.14.6)
# used by HighFive in our code. Setting to 2 suppresses warnings entirely.
export HDF5_DISABLE_VERSION_CHECK=2

