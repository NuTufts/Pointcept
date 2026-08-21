#!/bin/bash
# LOGIN-SIDE stager for overlay training conversion (tier2 not mounted on
# compute nodes). Stages a fileno range from the sample's TRAINPOOL list
# into tier1 scratch, then submits the conversion array for that range
# (array tasks delete their staged file when done).
#
#   bash stage_overlay_train_batch.sh <SAMPLE> <FIRST> <LAST> [%THROTTLE]
set -eu
SAMPLE=${1:?SAMPLE}
FIRST=${2:?first fileno}
LAST=${3:?last fileno}
THR=${4:-%60}
KPV2=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
LIST=$KPV2/lartpc/data_prep/uboone_official/training_data_ledger/${SAMPLE}_TRAINPOOL.txt
STAGE=/cluster/tufts/wongjiradlab/larbys/data/mcc9_scratch/tier2_staging/${SAMPLE}
mkdir -p "$STAGE"
n=0; skip=0
for i in $(seq $FIRST $LAST); do
    ZFN=$(printf "%05d" $i)
    OUT=/cluster/tufts/wongjiradlab/larbys/data/larformer/overlay_train/${SAMPLE}/merged_sp/$(printf "%03d" $((i/100)))
    if ls "$OUT"/merged_${SAMPLE}_fileno${ZFN}_*.h5 >/dev/null 2>&1; then
        skip=$((skip+1)); continue
    fi
    [ -f "$STAGE/fileno${ZFN}.root" ] && { n=$((n+1)); continue; }
    SRC=$(sed -n "${i}p" "$LIST")
    [ -n "$SRC" ] || { echo "WARN: empty line $i"; continue; }
    cp "$SRC" "$STAGE/fileno${ZFN}.root" && n=$((n+1))
done
echo "staged $n files (skipped $skip already-converted) -> $STAGE"
cd $KPV2
JID=$(sbatch --parsable --export=ALL,SAMPLE=$SAMPLE \
      --array=${FIRST}-${LAST}${THR} \
      lartpc/data_prep/uboone_official/submit_overlay_train_convert.sh)
echo "conversion array: $JID (range $FIRST-$LAST)"
