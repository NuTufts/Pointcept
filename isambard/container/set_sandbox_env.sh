#!/bin/bash

# Required env vars
export SPCONV_DISABLE_JIT="1"
export CUMM_CUDA_VERSION="12.4"
export CUMM_CUDA_ARCH_LIST="9.0"
export CUMM_DISABLE_JIT="1"
export TORCH_CUDA_ARCH_LIST="9.0"
export MAX_JOBS=16

export APPTAINER_TMPDIR=/local/user/$(id -u)/apptainer-tmp
export APPTAINER_CACHEDIR=/local/user/$(id -u)/apptainer-cache
export TMPDIR=/local/user/$(id -u)/tmp

mkdir -p $APPTAINER_TMPDIR $APPTAINER_CACHEDIR $TMPDIR