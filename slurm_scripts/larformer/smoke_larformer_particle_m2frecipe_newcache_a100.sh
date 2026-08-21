#!/bin/bash
# CONFIG + CODE-PATH SMOKE for the Stage-3 M2F-recipe (V2) particle segmenter
# on ONE A100-80GB, using the OLD stage-1+2 cache (iter_75750 slicer).
#
# The production V2 run needs a NEW cache built from the m2frecipe-v2 slicer
# checkpoint; this smoke only validates the config + code wiring, which is
# cache-content-independent:
#   - overlay parses; optimizer no_decay group split logged
#   - OneCycleLR builds with trainer-injected total_steps
#   - match_per_layer + use_vectorized_pair_loss on the Stage-3 (origin-head
#     active, 8-class, soft-presence cls) path
#   - LArFormerParticleEvaluator probe_freq kwarg forwarding (new)
#   - finite losses, nonzero diag_mask_logit_p95, n_matched <= 32
#
#   sbatch smoke_larformer_particle_m2frecipe_a100.sh

#SBATCH --job-name=lf-particle-smoke-newcache
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --time=00:40:00
#SBATCH --partition=gpu,preempt
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=pax141,pax007
#SBATCH --output=logs/lf-particle-smoke-newcache.%j.%N.log
#SBATCH --error=logs/lf-particle-smoke-newcache.%j.%N.err

set -u

WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
CONFIG=${WORKDIR}/configs/lartpc/larformer/stage3_particle/larformer-particle-v2-cached-ptv3crosslevel-m2frecipe.py
container=/cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif
OLD_CACHE=/cluster/tufts/wongjiradlab/larbys/data/ub_on_tufts/hdf5/larformer_cache_stage12__m2fv2ep4_ftdeghost_tau020
SAVE=${WORKDIR}/exp/_smoke_particle_m2frecipe_newcache
mkdir -p "${SAVE}"

# Small file lists from the old cache: 400 train / 80 val events.
TRAIN_LIST=${SAVE}/smoke_train_list.txt
VAL_LIST=${SAVE}/smoke_val_list.txt
find ${OLD_CACHE}/train -name '*.h5' 2>/dev/null | sort | head -400 > "${TRAIN_LIST}"
find ${OLD_CACHE}/val   -name '*.h5' 2>/dev/null | sort | head -80  > "${VAL_LIST}"
echo "[smoke] train list: $(wc -l < ${TRAIN_LIST}) files; val list: $(wc -l < ${VAL_LIST}) files"

module load apptainer

MEMLOG=${SAVE}/gpumem.log
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits \
  -l 3 > "${MEMLOG}" 2>&1 &
SMPID=$!

# 400 samples / batch 8 = 50 iters. (The valprobe won't fire in 50 iters at
# probe_freq=1000, but the new kwarg forwarding is validated at hook
# construction — an unaccepted kwarg would TypeError at startup.)
apptainer exec --nv --bind /cluster:/cluster $container bash -c \
  "cd ${WORKDIR} && source setenv_pointcept_only.sh && \
   python3 tools/train.py --config ${CONFIG} --num-gpus 1 --options \
     save_path=${SAVE} \
     data.train.data_root=${OLD_CACHE}/train \
     data.train.data_list_file=${TRAIN_LIST} \
     data.val.data_root=${OLD_CACHE}/val \
     data.val.data_list_file=${VAL_LIST} \
     data.test.data_root=${OLD_CACHE}/val \
     epoch=1 eval_epoch=1 evaluate=False enable_wandb=False \
     batch_size=8 gradient_accumulation_steps=1 \
     batch_size_val=8 num_worker=6 num_worker_val=2"
RC=$?

kill ${SMPID} 2>/dev/null
echo "=============================================="
echo "smoke train exit code: ${RC}"
awk -F',' 'BEGIN{m=0}{gsub(/ /,"",$2); if($2+0>m)m=$2+0} END{
  print "PEAK GPU MEM USED: "m" MB  (A100-80GB = 81920 MB)"}' "${MEMLOG}"
echo "Check the err log for: no_decay group split, loss_diag_* keys,"
echo "  n_matched<=32, nonzero diag_mask_logit_p95, finite losses."
echo "=============================================="
