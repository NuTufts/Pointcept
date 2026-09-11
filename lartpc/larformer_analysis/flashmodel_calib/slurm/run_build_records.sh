#!/bin/bash
#SBATCH --job-name=gcal_records
#SBATCH --mem=24G --cpus-per-task=2 --time=24:00:00
#SBATCH --partition=batch,preempt,wongjiradlab
#SBATCH --output=logs/flashmodel_calib/records_%x.%A_%a.log
#SBATCH --error=logs/flashmodel_calib/records_%x.%A_%a.err
# Usage (from the repo root):
#   SAMPLE=extbnb200k_cew6 NSHARDS=8 sbatch --array=0-7 \
#       lartpc/larformer_analysis/flashmodel_calib/slurm/run_build_records.sh
# Optional: EXTRA="--union-every" (MC truth-nu arm), MIN_LEN=30
set -u
module load apptainer 2>/dev/null || true
K=${K:-/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept}
C=${C:-/cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif}
SAMPLE=${SAMPLE:?set SAMPLE=<tag from gammacal/samples.py>}
NSHARDS=${NSHARDS:-8}
EXTRA=${EXTRA:-}
MIN_LEN=${MIN_LEN:-30}
TASK=${SLURM_ARRAY_TASK_ID:-0}
mkdir -p "${K}/logs/flashmodel_calib"

LIST=$(apptainer exec --bind /cluster:/cluster "$C" bash -c "
  cd $K && export PYTHONPATH=$K && python3 -c \"
from lartpc.larformer_analysis.flashmodel_calib.gammacal import samples
print(samples.get('${SAMPLE}')['kp2_nu_list'])\"")
NLINES=$(grep -c . "${LIST}")
PER=$(( (NLINES + NSHARDS - 1) / NSHARDS ))
START=$(( TASK * PER ))
[ "${START}" -ge "${NLINES}" ] && { echo "shard ${TASK}: nothing to do"; exit 0; }
echo ">>> ${SAMPLE} shard ${TASK}/${NSHARDS}: start=${START} n=${PER} of ${NLINES}"

apptainer exec --bind /cluster:/cluster "$C" bash -c "
  cd $K && export PYTHONPATH=$K && \
  python3 lartpc/larformer_analysis/flashmodel_calib/scripts/build_records.py \
    --sample ${SAMPLE} --start ${START} --n ${PER} --min-len ${MIN_LEN} ${EXTRA}"
echo "DONE ${SAMPLE} shard ${TASK}"
