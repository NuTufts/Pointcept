#!/bin/bash
# VECTORIZED-PAIR-LOSS BENCHMARK for the M2F-recipe slicer on ONE A100-80GB.
#
# Three stages:
#   0. Function-level parity + speed (test_pair_loss_parity.py): the loop and
#      vectorized per-pair BCE/Dice on identical synthetic inputs — mean-level
#      agreement + ms/call (fwd+bwd), incl. the small-instance corner.
#   1. phase loop : ...-m2frecipe.py            (use_vectorized_pair_loss=False)
#   2. phase vec  : ...-m2frecipe-vecloss.py    (use_vectorized_pair_loss=True)
# Phases 1-2: identical 100-iter training (batch 4, seed 42, 40 biggest events
# x10), same harness as the AMP benchmark. Reports avg sec/iter, peak GPU
# memory, mean loss over the last 15 iters.
#
#   sbatch bench_larformer_slicer_vecloss_a100.sh

#SBATCH --job-name=lf-slicer-vecbench
#SBATCH --mem=96G
#SBATCH --cpus-per-task=12
#SBATCH --time=01:30:00
#SBATCH --partition=gpu,preempt
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=pax141
#SBATCH --output=logs/lf-slicer-vecbench.%j.%N.log
#SBATCH --error=logs/lf-slicer-vecbench.%j.%N.err

set -u

WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
CFGDIR=${WORKDIR}/configs/lartpc/larformer/stage2_slicer
CONFIG_LOOP=${CFGDIR}/larformer-slicer-v1-cascaded-ptv3hybrid-crosslevel-m2frecipe.py
CONFIG_VEC=${CFGDIR}/larformer-slicer-v1-cascaded-ptv3hybrid-crosslevel-m2frecipe-vecloss.py
container=/cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif
BIGLIST=${WORKDIR}/slurm_scripts/larformer/smoke_biglist_cap300k.txt
BENCHLIST=${WORKDIR}/slurm_scripts/larformer/bench_biglist_x10.txt
BASE=${WORKDIR}/exp/_bench_vecloss

module load apptainer
mkdir -p "${BASE}"

# 40 biggest events x10 = 400 samples = exactly 100 iters at batch 4.
: > "${BENCHLIST}"
for i in $(seq 10); do cat "${BIGLIST}" >> "${BENCHLIST}"; done

echo "========== stage 0: function-level parity + speed =========="
apptainer exec --nv --bind /cluster:/cluster $container bash -c \
  "cd ${WORKDIR} && source setenv_pointcept_only.sh && \
   python3 slurm_scripts/larformer/test_pair_loss_parity.py" \
  2>&1 | tee "${BASE}/parity_test.log"
echo "[bench] parity test exit code: $?"

run_phase () {
  local NAME=$1 CFG=$2
  local SAVE=${BASE}/${NAME}
  rm -rf "${SAVE}"; mkdir -p "${SAVE}"
  nvidia-smi --query-gpu=index,memory.used,memory.total \
    --format=csv,noheader,nounits -l 3 > "${SAVE}/gpumem.log" 2>&1 &
  local SMPID=$!
  apptainer exec --nv --bind /cluster:/cluster $container bash -c \
    "cd ${WORKDIR} && source setenv_pointcept_only.sh && \
     python3 tools/train.py --config ${CFG} --num-gpus 1 --options \
       save_path=${SAVE} seed=42 \
       data.train.data_list_file=${BENCHLIST} \
       data.val.data_list_file=${BIGLIST} \
       epoch=1 eval_epoch=1 evaluate=False enable_wandb=False \
       batch_size=4 gradient_accumulation_steps=1 \
       batch_size_val=2 num_worker=6 num_worker_val=2" \
    > "${SAVE}/train.log" 2>&1
  local RC=$?
  kill ${SMPID} 2>/dev/null
  echo "[bench] phase ${NAME} exit code: ${RC}"
}

summarize () {
  local NAME=$1
  local SAVE=${BASE}/${NAME}
  local AVG LOSS PEAK NITER P95
  NITER=$(grep -c "Train: \[1/1\]" "${SAVE}/train.log" 2>/dev/null || echo 0)
  AVG=$(grep "Train: \[1/1\]" "${SAVE}/train.log" | tail -1 \
        | grep -oE "Batch [0-9.]+ \([0-9.]+\)" | grep -oE "\([0-9.]+\)" | tr -d '()')
  LOSS=$(grep -oE " loss: [0-9.]+" "${SAVE}/train.log" | tail -15 \
         | awk '{s+=$2; n++} END{if(n>0) printf "%.3f", s/n; else print "n/a"}')
  P95=$(grep -oE "diag_mask_logit_p95: [0-9.]+" "${SAVE}/train.log" | tail -1 | awk '{print $2}')
  PEAK=$(awk -F',' 'BEGIN{m=0}{gsub(/ /,"",$2); if($2+0>m)m=$2+0} END{print m}' \
         "${SAVE}/gpumem.log" 2>/dev/null)
  echo "  ${NAME}: iters_logged=${NITER}  avg_sec_per_iter=${AVG:-n/a}  mean_loss_last15=${LOSS}  final_logit_p95=${P95:-n/a}  peak_gpu_mem_MB=${PEAK:-n/a}"
}

echo "========== stage 1+2: end-to-end 100-iter training =========="
run_phase loop "${CONFIG_LOOP}"
run_phase vec  "${CONFIG_VEC}"

echo "=============================================="
echo "VECLOSS BENCHMARK SUMMARY (batch 4, seed 42, 100 iters, 40-biggest-events x10)"
summarize loop
summarize vec
echo "  (parity bar: vec mean_loss_last15 within a few % of loop AND"
echo "   final_logit_p95 nonzero/similar — the AMP failure signature was p95=0)"
echo "=============================================="
