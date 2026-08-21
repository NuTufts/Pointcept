#!/bin/bash
# CELL S1 SMOKE: few iters on a 12-overlay + 8-LANTERN mixed dev list.
# Validates: mixed-schema loading, completed labels, partial-truth events
# through masked_no_object, DN min_gt guard, v6-lantern cascade weights.
#SBATCH --job-name=lf-s1-smoke
#SBATCH --mem=96G
#SBATCH --cpus-per-task=12
#SBATCH --time=00:50:00
#SBATCH --partition=gpu,preempt
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=pax141,pax007
#SBATCH --output=logs/lf-s1-smoke.%j.log
#SBATCH --error=logs/lf-s1-smoke.%j.err
set -u
WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
CONFIG=${WORKDIR}/configs/lartpc/larformer/stage2_slicer/larformer-slicer-s1-mixenriched-v1.py
container=/cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif
LIST=${WORKDIR}/slurm_scripts/larformer/smoke_s1_mixlist.txt
SAVE=${WORKDIR}/exp/_smoke_slicer_s1
mkdir -p "${SAVE}"
module load apptainer
apptainer exec --nv --bind /cluster:/cluster $container bash -c \
  "cd ${WORKDIR} && source setenv_pointcept_only.sh && \
   python3 tools/train.py --config ${CONFIG} --num-gpus 1 --options \
     save_path=${SAVE} \
     data.train.data_list_file=${LIST} \
     data.train.loop=2 \
     data.val.data_list_file=${LIST} \
     epoch=1 eval_epoch=1 evaluate=False enable_wandb=False \
     batch_size=4 gradient_accumulation_steps=1 \
     batch_size_val=2 num_worker=6 num_worker_val=2"
echo "smoke exit: $?"
