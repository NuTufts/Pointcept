#!/bin/bash
#SBATCH --job-name=ovl_train_conv
#SBATCH --mem=12G
#SBATCH --cpus-per-task=2
#SBATCH --time=4:00:00
#SBATCH --partition=batch,preempt
#SBATCH --output=logs/overlay_train/conv.%A_%a.log
#SBATCH --error=logs/overlay_train/conv.%A_%a.err
# Overlay training-data conversion (SLICER_RETRAIN_PLAN P2).
# NOTE: tier2 is NOT mounted on compute nodes (verified 2026-08-18) —
# staging happens on the LOGIN side (stage_overlay_train_batch.sh);
# each array task expects its file already at
# $STAGE/fileno<NNNNN>.root, converts all entries with stepA (--mcc9,
# adc wire, tick-backward; NO larmatch per the no-lm policy), verifies,
# then deletes the staged copy (tier1 scratch is writable from compute).
#
#   sbatch --export=ALL,SAMPLE=<name> --array=1-N%60 \
#       submit_overlay_train_convert.sh
# SAMPLE = ledger basename, e.g. mcc9_v29e_dl_run1_bnb_intrinsic_nue_LowE
set -u -o pipefail
SAMPLE=${SAMPLE:?set SAMPLE}
KPV2=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
LIST=$KPV2/lartpc/data_prep/uboone_official/training_data_ledger/${SAMPLE}_TRAINPOOL.txt
FILENO=${SLURM_ARRAY_TASK_ID:?}
ZFN=$(printf "%05d" $FILENO)
STAGE=/cluster/tufts/wongjiradlab/larbys/data/mcc9_scratch/tier2_staging/${SAMPLE}
OUT=/cluster/tufts/wongjiradlab/larbys/data/larformer/overlay_train/${SAMPLE}/merged_sp/$(printf "%03d" $((FILENO/100)))
mkdir -p "$OUT"

# skip if this fileno already produced output
if ls "$OUT"/merged_${SAMPLE}_fileno${ZFN}_*.h5 >/dev/null 2>&1; then
    rm -f "$STAGE/fileno${ZFN}.root"
    echo "exists, skip fileno $ZFN"; exit 0
fi

LOCAL=$STAGE/fileno${ZFN}.root
[ -f "$LOCAL" ] || { echo "ERROR: staged file missing: $LOCAL (login-side staging required)"; exit 1; }

module load apptainer 2>/dev/null || true
apptainer exec --bind /cluster:/cluster \
  /cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif bash -c "
  cd /cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/ubdl && \
  source setenv_pointcept_container.sh >/dev/null 2>&1 && cd $KPV2 && \
  python3 lartpc/data_prep/uboone_official/convert_dlmerged_to_larformer_h5.py \
    -i $LOCAL -o $OUT --tag $SAMPLE --fileno-tag fileno${ZFN} \
    --adc wire -tb --mcc9 -v 1"
RC=$?
NOUT=$(ls "$OUT"/merged_${SAMPLE}_fileno${ZFN}_*.h5 2>/dev/null | wc -l)
rm -f "$LOCAL"
echo "fileno $ZFN: converter rc=$RC  h5s=$NOUT  (staged copy removed)"
[ "$RC" = "0" ] && [ "$NOUT" -gt 0 ] || exit 1

# A2 post-step: in-place label completion (r=0.5, shell +-2 locked by
# ablation; idempotency-guarded in the tool)
apptainer exec --bind /cluster:/cluster \
  /cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif bash -c "
  cd $KPV2 && export PYTHONPATH=$KPV2 && \
  python3 lartpc/data_prep/uboone_official/complete_labels.py \
    --h5 $OUT/merged_${SAMPLE}_fileno${ZFN}_*.h5" | tail -2
echo "fileno $ZFN: label completion done"
