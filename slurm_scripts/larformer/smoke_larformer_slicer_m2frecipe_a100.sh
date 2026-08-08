#!/bin/bash
# CONFIG + MEMORY SMOKE for the M2F-recipe slicer retrain on ONE A100-80GB.
#
# Same worst-case probe as smoke_larformer_slicer_cap300k_a100.sh (a few iters
# on the ~40 BIGGEST events at batch_size=4 = the production per-GPU load) but
# pointed at the M2F-recipe config. The model/data are identical to the
# cap300k run (memory already validated at 41.5 GB peak), so this smoke is
# mainly a CONFIG-WIRING check for the recipe changes:
#   - AdamW no_decay_on_1d_and_embeddings param-group split (builder logs the
#     decay / no_decay group sizes -- check the log)
#   - OneCycleLR builds with trainer-injected total_steps
#   - loss_kwargs: no_object_weight=0.1, cost_origin=0.0, log_diagnostics=True
#     (expect train_batch/loss_diag_* keys in the iter log)
#   - clip_grad=0.1 (grad_norm logged pre-clip)
# plus a re-confirmation that memory behavior is unchanged.
#
#   sbatch smoke_larformer_slicer_m2frecipe_a100.sh

#SBATCH --job-name=lf-slicer-smoke-m2f
#SBATCH --mem=96G
#SBATCH --cpus-per-task=12
#SBATCH --time=00:40:00
#SBATCH --partition=gpu,preempt
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=pax141
#SBATCH --output=logs/lf-slicer-smoke-m2f.%j.%N.log
#SBATCH --error=logs/lf-slicer-smoke-m2f.%j.%N.err

set -u

WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
CONFIG=${WORKDIR}/configs/lartpc/larformer/stage2_slicer/larformer-slicer-v1-cascaded-ptv3hybrid-crosslevel-m2frecipe.py
container=/cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif
BIGLIST=${WORKDIR}/slurm_scripts/larformer/smoke_biglist_cap300k.txt
SAVE=${WORKDIR}/exp/_smoke_slicer_m2frecipe
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
echo "Also check the train log for:"
echo "  - optimizer decay/no_decay group split (no_decay_on_1d_and_embeddings)"
echo "  - loss_diag_* keys (log_diagnostics=True)"
echo "  - grad_norm values vs clip_grad=0.1"
echo "=============================================="
