#!/bin/bash
#SBATCH --job-name=gcal_infer
#SBATCH --partition=gpu,preempt --gres=gpu:1 --mem=32G --cpus-per-task=4 --time=12:00:00
#SBATCH --output=logs/flashmodel_calib/calib_infer_%x.%A_%a.log --error=logs/flashmodel_calib/calib_infer_%x.%A_%a.err
# Flash-calibration inference (cascade --flash-calib-mode) over a merged_sp list.
#   MSP_LIST=<merged_sp list> OUT_DIR=<dir> KIND=data|mc [GT=0|1] [NSHARDS=2] [N_TOTAL=3000]
#   sbatch --array=0-1 --job-name=gcal_ext lartpc/larformer_analysis/flashmodel_calib/slurm/run_calib_inference.sh
# Uses the cew6 production checkpoints; gamma is irrelevant for the calibration
# (predictions are recomputed at the reference) but is pinned to 1.0 for provenance.
set -u
module load apptainer 2>/dev/null || true
K=${K:-/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept}
C=/cluster/tufts/wongjiradlabnu/larbys/larbys-container/pointcept_cuml.sif
[ -f $C ] || C=/cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif
MSP_LIST=${MSP_LIST:?}; OUT_DIR=${OUT_DIR:?}; KIND=${KIND:?data|mc}
GT=${GT:-0}; NSHARDS=${NSHARDS:-2}; N_TOTAL=${N_TOTAL:-3000}; TASK=${SLURM_ARRAY_TASK_ID:-0}
PER=$(( (N_TOTAL + NSHARDS - 1) / NSHARDS )); START=$(( TASK * PER ))
GTFLAG="--no-gt"; [ "$GT" = "1" ] && GTFLAG=""
mkdir -p "$K/logs/flashmodel_calib" "$OUT_DIR"
echo ">>> calib inference: $MSP_LIST [$START, +$PER) kind=$KIND gt=$GT -> $OUT_DIR"
apptainer exec --nv --bind /cluster:/cluster $C bash -c "
  cd $K && source setenv_pointcept_only.sh && \
  export LARFORMER_BATTERY_SLICER_CKPT=exp/larformer_slicer_s1_mixenriched_v1/model/epoch_2.pth && \
  export LARFORMER_KP_PARTICLE_CKPT=exp/larformer_particle_s1cache_m2frecipe_rebal_v2_warm2_cew/model/epoch_6.pth && \
  python3 -u tools/larformer/run_larformer_keypoint2_cascade_inference.py \
    --config configs/lartpc/larformer/stage4_keypoint/larformer-keypoint2-fullcascade-v6lantern-envslicer.py \
    --input-list $MSP_LIST --output-dir $OUT_DIR/ --start-event $START --n-events $PER \
    --deterministic --device cuda --output-tree $GTFLAG \
    --flash-calib-mode --sample-kind $KIND --gamma-run-scale 1.0"
echo "DONE shard $TASK"
