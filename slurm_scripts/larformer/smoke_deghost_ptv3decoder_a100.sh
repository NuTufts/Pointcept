#!/bin/bash
# CONFIG SMOKE for the PTv3-decoder deghoster experiment on ONE A100.
#
# Validates on a 400-train / 200-val file subset:
#   - SonataFinetuneCheckpointLoader remap (log should report loaded_count
#     for teacher.backbone.* -> backbone.*; decoder + head stay random)
#   - param group split: Params Group 2 (backbone.enc) + 3 (backbone.embedding)
#     at lr 0.0, Group 1 (decoder + seg_head) at base_lr
#   - PT-v3m2 enc_mode=False forward through DefaultSegmentorV2 (64-ch dec0
#     -> Linear(64,2) seg head), finite Focal+Lovasz losses
#   - SemSegEvaluator runs and prints real/ghost per-class IoU
#   - GPU memory at batch 32 (production is 96 -> scale x3 for headroom check)
#
#   sbatch smoke_deghost_ptv3decoder_a100.sh

#SBATCH --job-name=lf-deghost-smoke
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --time=00:40:00
#SBATCH --partition=gpu,preempt
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=pax141,pax007
#SBATCH --output=logs/lf-deghost-smoke.%j.%N.log
#SBATCH --error=logs/lf-deghost-smoke.%j.%N.err

set -u

WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
CONFIG=${WORKDIR}/configs/lartpc/larformer/stage1_deghost/deghost-ptv3decoder-v1-frozenenc-extbnb.py
container=/cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif
LISTDIR=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/lartpc_data_prep/lantern_scripts/h5lists
SAVE=${WORKDIR}/exp/_smoke_deghost_ptv3dec
mkdir -p "${SAVE}"

# Small subsets of the real train/val lists (v3 LANTERN — the prod4 v2
# dataset was deleted from disk; see config docstring).
head -400 ${LISTDIR}/h5list_mcall_lantern_train.txt > ${SAVE}/smoke_train.txt
head -200 ${LISTDIR}/h5list_mcall_lantern_val.txt   > ${SAVE}/smoke_val.txt

module load apptainer

MEMLOG=${SAVE}/gpumem.log
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits \
  -l 3 > "${MEMLOG}" 2>&1 &
SMPID=$!

apptainer exec --nv --bind /cluster:/cluster $container bash -c \
  "cd ${WORKDIR} && source setenv_pointcept_only.sh && \
   python3 tools/train.py --config ${CONFIG} --num-gpus 1 --options \
     save_path=${SAVE} \
     data.train.data_list_file=${SAVE}/smoke_train.txt \
     data.val.data_list_file=${SAVE}/smoke_val.txt \
     epoch=1 eval_epoch=1 evaluate=True enable_wandb=False \
     batch_size=32 batch_size_val=32 num_worker=6 num_worker_val=4"
RC=$?

kill ${SMPID} 2>/dev/null
echo "=============================================="
echo "smoke train exit code: ${RC}"
awk -F',' 'BEGIN{m=0}{gsub(/ /,"",$2); if($2+0>m)m=$2+0} END{
  print "PEAK GPU MEM USED: "m" MB at batch 32 (A100-80GB = 81920 MB; production batch 96 ~ 3x)"}' "${MEMLOG}"
echo "Check err log for: SONATA remap loaded_count, Params Group lr values,"
echo "  finite losses, SemSegEvaluator real/ghost IoU."
echo "=============================================="
