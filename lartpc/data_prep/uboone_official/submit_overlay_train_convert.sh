#!/bin/bash
#SBATCH --job-name=ovl_train_conv
#SBATCH --mem=12G
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --partition=batch,preempt,wongjiradlab
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
# Generalized 2026-09-11 (see stage_overlay_train_batch.sh); defaults reproduce
# the original TRAINPOOL/MC behavior.
LIST=${LIST:-$KPV2/lartpc/data_prep/uboone_official/training_data_ledger/${SAMPLE}_TRAINPOOL.txt}
MODE=${MODE:-mc}                 # mc | data
TRUTH_DIR=${TRUTH_DIR:-}         # MC only
# fileno = FILENO_BASE + array index (array indices must stay < MaxArraySize=2000;
# FILENO_BASE=0 keeps the old "index == fileno" behaviour)
FILENO=$(( ${FILENO_BASE:-0} + ${SLURM_ARRAY_TASK_ID:?} ))
ZFN=$(printf "%05d" $FILENO)
STAGE=${STAGE:-/cluster/tufts/wongjiradlab/larbys/data/mcc9_scratch/tier2_staging/${SAMPLE}}
OUT_ROOT=${OUT_ROOT:-/cluster/tufts/wongjiradlab/larbys/data/larformer/overlay_train/${SAMPLE}/merged_sp}
OUT=$OUT_ROOT/$(printf "%03d" $((FILENO/100)))
MARK=${MARK:-$(dirname "$OUT_ROOT")/markers}
CLOG=$MARK/logs
mkdir -p "$OUT" "$MARK" "$CLOG"
case "$MODE" in
  mc)   CONVFLAGS="--mcc9"    ;;
  data) CONVFLAGS="--is-data" ;;
  *)    echo "ERROR: MODE must be mc|data (got '$MODE')"; exit 1 ;;
esac

# skip if this fileno already produced output
if [ -s "$MARK/fileno${ZFN}.ok" ] || ls "$OUT"/merged_${SAMPLE}_fileno${ZFN}_*.h5 >/dev/null 2>&1; then
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
    --adc wire -tb $CONVFLAGS -v 1" 2>&1 | tee "$CLOG/fileno${ZFN}.log"
RC=${PIPESTATUS[0]}
NOUT=$(ls "$OUT"/merged_${SAMPLE}_fileno${ZFN}_*.h5 2>/dev/null | wc -l)
# "Done." is printed only after the entry loop completes; its absence means the
# job died mid-file (partial output) even when h5s exist.
DONE=$(grep -c '^Done\.' "$CLOG/fileno${ZFN}.log" 2>/dev/null || true)

# MC: extract the truth sidecar BEFORE deleting the staged dlmerged -- tier2 is
# not mounted on compute nodes, so this is the only chance to read it.
TRUTH_N=-1
if [ "$MODE" = "mc" ] && [ -n "$TRUTH_DIR" ]; then
    mkdir -p "$TRUTH_DIR"
    if [ ! -s "$TRUTH_DIR/truth_fileno${ZFN}.h5" ]; then
        apptainer exec --bind /cluster:/cluster \
          /cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif bash -c "
          cd /cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/ubdl && \
          source setenv_pointcept_container.sh >/dev/null 2>&1 && cd $KPV2 && \
          python3 lartpc/larformer_reco/export/extract_truth_sidecar.py \
            --input-dlmerged $LOCAL --out $TRUTH_DIR/truth_fileno${ZFN}.h5" | tail -2
    fi
    [ -s "$TRUTH_DIR/truth_fileno${ZFN}.h5" ] && TRUTH_N=1 || TRUTH_N=0
fi

rm -f "$LOCAL"
echo "fileno $ZFN: converter rc=$RC  h5s=$NOUT  done=$DONE  truth=$TRUTH_N  (staged copy removed)"

# A2 post-step: in-place label completion (MC only; r=0.5, shell +-2 locked by
# ablation; idempotency-guarded in the tool). DECOUPLED from the converter exit
# code -- a teardown crash (rc=129/133) still leaves valid h5s that must be
# completed, otherwise they silently enter training/analysis unlabeled.
if [ "$MODE" = "mc" ] && [ "$NOUT" -gt 0 ]; then
    apptainer exec --bind /cluster:/cluster \
      /cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif bash -c "
      cd $KPV2 && export PYTHONPATH=$KPV2 && \
      python3 lartpc/data_prep/uboone_official/complete_labels.py \
        --h5 $OUT/merged_${SAMPLE}_fileno${ZFN}_*.h5" | tail -2
    echo "fileno $ZFN: label completion done"
fi

# marker records a VERIFIED conversion (consumed by the stager's skip test and
# by verify_tranche); only written when the entry loop actually finished.
if [ "$NOUT" -gt 0 ] && [ "${DONE:-0}" -ge 1 ]; then
    echo "rc=$RC nout=$NOUT done=$DONE truth=$TRUTH_N mode=$MODE" > "$MARK/fileno${ZFN}.ok"
    exit 0
fi
echo "fileno $ZFN: INCOMPLETE (rc=$RC nout=$NOUT done=$DONE) -- no marker written"
exit 1
