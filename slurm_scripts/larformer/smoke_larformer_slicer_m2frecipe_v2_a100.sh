#!/bin/bash
# CONFIG + CODE-PATH SMOKE for the M2F-recipe V2 slicer on ONE A100-80GB.
#
# V2 = vectorized pair loss + num_queries 48 + match_per_layer (per-layer
# Hungarian). The last is a NEW CODE PATH in LArFormerLoss.forward — this
# smoke exercises it end-to-end on the 40 biggest events (many GT instances,
# so per-layer matching runs on every supervision layer with K ~ 20-30).
# Also re-checks memory at the production per-GPU load (batch 4).
#
# What to check in the log:
#   - RC=0, finite losses, nonzero diag_mask_logit_p95 by the last iters
#   - n_matched <= 48 (query count applied)
#   - iter time ~<= the vecloss bench's 3.8 s/iter ballpark (48 queries +
#     7 extra tiny scipy solves should not add measurable cost)
#
#   sbatch smoke_larformer_slicer_m2frecipe_v2_a100.sh

#SBATCH --job-name=lf-slicer-smoke-m2fv2
#SBATCH --mem=96G
#SBATCH --cpus-per-task=12
#SBATCH --time=00:40:00
#SBATCH --partition=gpu,preempt
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=pax141
#SBATCH --output=logs/lf-slicer-smoke-m2fv2.%j.%N.log
#SBATCH --error=logs/lf-slicer-smoke-m2fv2.%j.%N.err

set -u

WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
CONFIG=${WORKDIR}/configs/lartpc/larformer/stage2_slicer/larformer-slicer-v1-cascaded-ptv3hybrid-crosslevel-m2frecipe-v2.py
container=/cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif
BIGLIST=${WORKDIR}/slurm_scripts/larformer/smoke_biglist_cap300k.txt
SAVE=${WORKDIR}/exp/_smoke_slicer_m2frecipe_v2
mkdir -p "${SAVE}"

module load apptainer

MEMLOG=${SAVE}/gpumem.log
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits \
  -l 3 > "${MEMLOG}" 2>&1 &
SMPID=$!

apptainer exec --nv --bind /cluster:/cluster $container bash -c \
  "cd ${WORKDIR} && source setenv_pointcept_only.sh && \
   python3 tools/train.py --config ${CONFIG} --num-gpus 1 --options \
     save_path=${SAVE} \
     data.train.data_list_file=${BIGLIST} \
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
echo "  (RC=0, finite losses, p95>0, n_matched<=48 -> V2 path is healthy)"
echo "=============================================="
