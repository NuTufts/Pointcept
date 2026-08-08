#!/bin/bash
# MEMORY SMOKE for the 300k-cap slicer retrain on ONE A100-80GB.
#
# Trains a handful of iters on the ~40 BIGGEST events (>220k SP, built by
# cap_study_spacepoints.py --dump-list) at batch_size=4 -- i.e. the exact
# per-GPU load the 2-GPU production run will carry (batch_size=8 total). Every
# batch is near the 300k cap, so this is a worst-case memory probe. Samples GPU
# memory.used throughout and prints the peak. If it survives and peaks well
# under 80GB, the production 4/GPU config is safe.
#
#   sbatch smoke_larformer_slicer_cap300k_a100.sh

#SBATCH --job-name=lf-slicer-smoke
#SBATCH --mem=96G
#SBATCH --cpus-per-task=12
#SBATCH --time=00:40:00
#SBATCH --partition=gpu,preempt
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=pax141
#SBATCH --output=logs/lf-slicer-smoke.%j.%N.log
#SBATCH --error=logs/lf-slicer-smoke.%j.%N.err

set -u

WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
CONFIG=${WORKDIR}/configs/lartpc/larformer/stage2_slicer/larformer-slicer-v1-cascaded-ptv3hybrid-crosslevel.py
container=/cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif
BIGLIST=${WORKDIR}/slurm_scripts/larformer/smoke_biglist_cap300k.txt
SAVE=${WORKDIR}/exp/_smoke_slicer_cap300k
mkdir -p "${SAVE}"

module load apptainer

# Sample GPU memory (host side) every 3s while training runs.
MEMLOG=${SAVE}/gpumem.log
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits \
  -l 3 > "${MEMLOG}" 2>&1 &
SMPID=$!

# 1 GPU, batch 4 = production per-GPU load. Tiny data loop, eval OFF, wandb OFF.
apptainer exec --nv --bind /cluster:/cluster $container bash -c \
  "cd ${WORKDIR} && source setenv_pointcept_only.sh && \
   python3 tools/train.py --config ${CONFIG} --num-gpus 1 --options \
     save_path=${SAVE} \
     data.train.data_list_file=${BIGLIST} \
     data.train.loop=3 \
     data.val.data_list_file=${BIGLIST} \
     epoch=1 eval_epoch=1 evaluate=False enable_wandb=False \
     batch_size=4 gradient_accumulation_steps=1 \
     batch_size_val=2 num_worker=6 num_worker_val=2"
RC=$?

kill ${SMPID} 2>/dev/null
echo "=============================================="
echo "smoke train exit code: ${RC}"
awk -F',' 'BEGIN{m=0}{gsub(/ /,"",$2); if($2+0>m)m=$2+0} END{
  print "PEAK GPU MEM USED: "m" MB  (A100-80GB = 81920 MB)"}' "${MEMLOG}"
echo "  (RC=0 and peak << 81920 -> 4/GPU is safe for production)"
echo "=============================================="
