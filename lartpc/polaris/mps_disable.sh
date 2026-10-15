#!/bin/bash
# Stop the CUDA MPS control daemon on THIS node (ALCF disable_mps_polaris.sh).
export CUDA_MPS_PIPE_DIRECTORY=${CUDA_MPS_PIPE_DIRECTORY:-/tmp/nvidia-mps}
export CUDA_MPS_LOG_DIRECTORY=${CUDA_MPS_LOG_DIRECTORY:-/tmp/nvidia-log}
if command -v nvidia-cuda-mps-control >/dev/null 2>&1 && [ -S "$CUDA_MPS_PIPE_DIRECTORY/control" ]; then
  echo quit | nvidia-cuda-mps-control && echo ">>> MPS stopped on $(hostname)"
else
  echo ">>> MPS not running on $(hostname)"
fi
