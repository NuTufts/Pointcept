#!/bin/bash
# Start the CUDA MPS control daemon on THIS node (ALCF recipe,
# docs.alcf.anl.gov/polaris/running-jobs/using-gpus: enable_mps_polaris.sh).
# Run once per node (node_launcher.sh does it when POL_MPS=1, or
# `mpiexec -n $NN --ppn 1 bash lartpc/polaris/mps_enable.sh`). Clients must
# export the same CUDA_MPS_PIPE_DIRECTORY (polaris_env.sh does) and may still
# pick their GPU with CUDA_VISIBLE_DEVICES (the daemon sees all four).
export CUDA_MPS_PIPE_DIRECTORY=${CUDA_MPS_PIPE_DIRECTORY:-/tmp/nvidia-mps}
export CUDA_MPS_LOG_DIRECTORY=${CUDA_MPS_LOG_DIRECTORY:-/tmp/nvidia-log}
command -v nvidia-cuda-mps-control >/dev/null 2>&1 || { echo "ERROR: nvidia-cuda-mps-control not found on $(hostname)" >&2; exit 2; }
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
if [ -S "$CUDA_MPS_PIPE_DIRECTORY/control" ] && echo get_server_list | nvidia-cuda-mps-control >/dev/null 2>&1; then
  echo ">>> MPS already running on $(hostname)"; exit 0
fi
CUDA_VISIBLE_DEVICES=${POL_MPS_DEVICES:-0,1,2,3} nvidia-cuda-mps-control -d || { echo "ERROR: MPS daemon failed to start on $(hostname)" >&2; exit 1; }
sleep 1
echo "start_server -uid $(id -u)" | nvidia-cuda-mps-control
echo ">>> MPS enabled on $(hostname): pipe=$CUDA_MPS_PIPE_DIRECTORY servers=$(echo get_server_list | nvidia-cuda-mps-control | tr '\n' ' ')"
