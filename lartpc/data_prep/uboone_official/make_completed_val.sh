#!/bin/bash
#SBATCH --job-name=val_completed
#SBATCH --mem=12G --cpus-per-task=2 --time=6:00:00
#SBATCH --partition=batch,preempt
#SBATCH --output=logs/overlay_train/valcomp.%A_%a.log
#SBATCH --error=logs/overlay_train/valcomp.%A_%a.err
set -u -o pipefail
I=${SLURM_ARRAY_TASK_ID:?}
KPV2=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
LED=$KPV2/lartpc/data_prep/uboone_official/training_data_ledger
VOUT=/cluster/tufts/wongjiradlab/larbys/data/larformer/overlay_train/_lantern_val_completed
N=1500; PER=150
FIRST=$(( (I-1)*PER + 1 )); LAST=$(( I*PER < N ? I*PER : N ))
sed -n "${FIRST},${LAST}p" $LED/lantern_val_subset_1500.txt > /tmp/vc_$$.txt
module load apptainer 2>/dev/null || true
apptainer exec --bind /cluster:/cluster,/tmp:/tmp \
  /cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif bash -c "
  cd $KPV2 && export PYTHONPATH=$KPV2 && \
  xargs -a /tmp/vc_$$.txt python3 lartpc/data_prep/uboone_official/complete_labels.py \
    --out-dir $VOUT --h5"
rm -f /tmp/vc_$$.txt
