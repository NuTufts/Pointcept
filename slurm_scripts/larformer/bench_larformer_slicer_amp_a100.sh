#!/bin/bash
# AMP BENCHMARK for the M2F-recipe slicer on ONE A100-80GB.
#
# Runs the SAME 100-iteration training twice on the same GPU:
#   phase noamp : larformer-slicer-v1-cascaded-ptv3hybrid-crosslevel-m2frecipe.py
#   phase bf16  : ...-m2frecipe-bf16amp.py  (enable_amp=True, bfloat16)
# Both phases: batch_size=4 (production per-GPU load), seed=42 (identical data
# order + augmentations), 400 samples = 40 biggest events x10 (list built
# below; the dataset's `loop` option proved unreliable under --options in the
# earlier smoke, so the list file is repeated explicitly).
#
# Reports, per phase and as a comparison:
#   - avg sec/iter (IterationTimer's running avg at iter 100, warmup excluded)
#   - peak GPU memory (nvidia-smi sampled every 3 s)
#   - mean loss over the last 15 iters (same seed/data -> parity check;
#     bit-identical is impossible with nondeterministic scatter/sampling)
#
#   sbatch bench_larformer_slicer_amp_a100.sh

#SBATCH --job-name=lf-slicer-ampbench
#SBATCH --mem=96G
#SBATCH --cpus-per-task=12
#SBATCH --time=01:30:00
#SBATCH --partition=gpu,preempt
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=pax141
#SBATCH --output=logs/lf-slicer-ampbench.%j.%N.log
#SBATCH --error=logs/lf-slicer-ampbench.%j.%N.err

set -u

WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
CFGDIR=${WORKDIR}/configs/lartpc/larformer/stage2_slicer
CONFIG_NOAMP=${CFGDIR}/larformer-slicer-v1-cascaded-ptv3hybrid-crosslevel-m2frecipe.py
CONFIG_BF16=${CFGDIR}/larformer-slicer-v1-cascaded-ptv3hybrid-crosslevel-m2frecipe-bf16amp.py
container=/cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif
BIGLIST=${WORKDIR}/slurm_scripts/larformer/smoke_biglist_cap300k.txt
BENCHLIST=${WORKDIR}/slurm_scripts/larformer/bench_biglist_x10.txt
BASE=${WORKDIR}/exp/_bench_amp

module load apptainer
mkdir -p "${BASE}"

# 40 biggest events x10 = 400 samples = exactly 100 iters at batch 4.
: > "${BENCHLIST}"
for i in $(seq 10); do cat "${BIGLIST}" >> "${BENCHLIST}"; done

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
  local AVG LOSS PEAK NITER
  NITER=$(grep -c "Train: \[1/1\]" "${SAVE}/train.log" 2>/dev/null || echo 0)
  # IterationTimer running avg at the last logged iter (warmup_iter=2 excluded)
  AVG=$(grep "Train: \[1/1\]" "${SAVE}/train.log" | tail -1 \
        | grep -oE "Batch [0-9.]+ \([0-9.]+\)" | grep -oE "\([0-9.]+\)" | tr -d '()')
  LOSS=$(grep -oE " loss: [0-9.]+" "${SAVE}/train.log" | tail -15 \
         | awk '{s+=$2; n++} END{if(n>0) printf "%.3f", s/n; else print "n/a"}')
  PEAK=$(awk -F',' 'BEGIN{m=0}{gsub(/ /,"",$2); if($2+0>m)m=$2+0} END{print m}' \
         "${SAVE}/gpumem.log" 2>/dev/null)
  echo "  ${NAME}: iters_logged=${NITER}  avg_sec_per_iter=${AVG:-n/a}  mean_loss_last15=${LOSS}  peak_gpu_mem_MB=${PEAK:-n/a}"
}

run_phase noamp "${CONFIG_NOAMP}"
run_phase bf16  "${CONFIG_BF16}"

echo "=============================================="
echo "AMP BENCHMARK SUMMARY (batch 4, seed 42, 100 iters, 40-biggest-events x10)"
summarize noamp
summarize bf16
echo "  (parity bar: bf16 mean_loss_last15 within a few % of noamp;"
echo "   bit-identical is impossible — nondeterministic scatter + sampling)"
echo "=============================================="
