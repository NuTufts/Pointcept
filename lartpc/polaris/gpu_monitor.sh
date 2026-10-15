#!/bin/bash
# GPU utilization sampler for one node: appends nvidia-smi samples to a CSV
# (one line per GPU per interval) until killed. Started per node by
# node_launcher.sh (POL_GPU_MONITOR=1); summarize with summarize_gpu_util.py.
#   gpu_monitor.sh <out.csv> [interval s]
# Columns: timestamp, gpu index, util.gpu %, util.mem %, mem.used MiB, mem.total MiB, power W
# The file is on Eagle, so `tail -f` from a login node shows the run live.
OUT=${1:?out csv}; INT=${2:-10}
mkdir -p "$(dirname "$OUT")"
echo "# host=$(hostname -s) start=$(date -Is) interval=${INT}s driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)" >> "$OUT"
exec stdbuf -oL nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
  --format=csv,noheader,nounits -l "$INT" >> "$OUT"
