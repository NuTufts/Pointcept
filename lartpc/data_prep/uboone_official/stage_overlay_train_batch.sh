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
# Generalized 2026-09-11: LIST/OUT_ROOT/MODE/TRUTH_DIR/MARK are env-overridable
# so an ARBITRARY list (e.g. a RESERVED analysis half, or a data sample) can be
# staged+converted. Defaults reproduce the original TRAINPOOL/MC behavior.
LIST=${LIST:-$KPV2/lartpc/data_prep/uboone_official/training_data_ledger/${SAMPLE}_TRAINPOOL.txt}
STAGE=${STAGE:-/cluster/tufts/wongjiradlab/larbys/data/mcc9_scratch/tier2_staging/${SAMPLE}}
OUT_ROOT=${OUT_ROOT:-/cluster/tufts/wongjiradlab/larbys/data/larformer/overlay_train/${SAMPLE}/merged_sp}
MODE=${MODE:-mc}                 # mc | data  (data => --is-data, no label completion)
TRUTH_DIR=${TRUTH_DIR:-}         # MC only; set => sidecar extracted in the task
MARK=${MARK:-$(dirname "$OUT_ROOT")/markers}
[ -f "$LIST" ] || { echo "ERROR: list not found: $LIST"; exit 1; }
mkdir -p "$STAGE" "$MARK"
# disk guard: never start a batch that could fill the volume
AVAIL_GB=$(df -BG --output=avail /cluster/tufts/wongjiradlab | tail -1 | tr -dc '0-9')
[ "${AVAIL_GB:-0}" -lt 500 ] && { echo "ERROR: only ${AVAIL_GB}G free on wongjiradlab; aborting"; exit 1; }
echo "list=$LIST mode=$MODE out=$OUT_ROOT free=${AVAIL_GB}G"
n=0; skip=0
for i in $(seq $FIRST $LAST); do
    ZFN=$(printf "%05d" $i)
    OUT=$OUT_ROOT/$(printf "%03d" $((i/100)))
    # marker-first (records a VERIFIED conversion); h5 glob kept as the
    # fallback so pre-marker conversions still skip correctly
    if [ -s "$MARK/fileno${ZFN}.ok" ] || ls "$OUT"/merged_${SAMPLE}_fileno${ZFN}_*.h5 >/dev/null 2>&1; then
        skip=$((skip+1)); continue
    fi
    [ -f "$STAGE/fileno${ZFN}.root" ] && { n=$((n+1)); continue; }
    SRC=$(sed -n "${i}p" "$LIST")
    [ -n "$SRC" ] || { echo "WARN: empty line $i"; continue; }
    cp "$SRC" "$STAGE/fileno${ZFN}.root" && n=$((n+1))
done
echo "staged $n files (skipped $skip already-converted) -> $STAGE"
cd $KPV2
JID=$(sbatch --parsable \
      --export=ALL,SAMPLE=$SAMPLE,LIST=$LIST,STAGE=$STAGE,OUT_ROOT=$OUT_ROOT,MODE=$MODE,TRUTH_DIR=$TRUTH_DIR,MARK=$MARK,FILENO_BASE=$FIRST \
      --array=0-$((LAST-FIRST))${THR} \
      lartpc/data_prep/uboone_official/submit_overlay_train_convert.sh)
# array index = fileno - FIRST (slurm MaxArraySize is 2000, filenos go higher)
echo "conversion array: $JID (range $FIRST-$LAST)"
