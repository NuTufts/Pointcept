#!/bin/bash
# LOGIN-SIDE sequencer for a truth-sidecar-only pass (tier2 is login-only).
#   bash run_sidecar_only_tranche.sh <LIST> <TRUTH_DIR> <STAGE> <FIRST> <LAST> [BATCH=250]
# Per batch: stage the dlmerged files whose sidecar is missing, submit the
# submit_sidecar_only.sh array (it deletes the staged copies), wait, report.
# Idempotent: existing sidecars are skipped.
set -u
K=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept; cd $K
LIST=${1:?list}; TRUTH_DIR=${2:?truth dir}; STAGE=${3:?stage dir}; FIRST=${4:?first}; LAST=${5:?last}; BATCH=${6:-250}
[ -d /cluster/tier2/wongjiradlab ] || { echo "ERROR: tier2 not visible; run on a login node"; exit 2; }
mkdir -p "$STAGE" "$TRUTH_DIR" logs/overlay_train
for a in $(seq "$FIRST" "$BATCH" "$LAST"); do
  b=$((a + BATCH - 1)); [ $b -gt "$LAST" ] && b=$LAST
  echo "=== sidecars filenos $a-$b"
  AVAIL_GB=$(df -BG --output=avail /cluster/tufts/wongjiradlab | tail -1 | tr -dc '0-9')
  [ "${AVAIL_GB:-0}" -lt 500 ] && { echo "ERROR: only ${AVAIL_GB}G free; aborting"; exit 1; }
  n=0; skip=0
  for i in $(seq $a $b); do
    ZFN=$(printf %05d $i)
    if [ -s "$TRUTH_DIR/truth_fileno$ZFN.h5" ]; then skip=$((skip+1)); continue; fi
    SRC=$(sed -n "${i}p" "$LIST"); [ -n "$SRC" ] || continue
    [ -f "$STAGE/fileno$ZFN.root" ] || cp "$SRC" "$STAGE/fileno$ZFN.root" && n=$((n+1))
  done
  echo "staged $n (skipped $skip with sidecar)"
  [ $n -eq 0 ] && continue
  J=$(sbatch --parsable --export=ALL,STAGE=$STAGE,TRUTH_DIR=$TRUTH_DIR,FILENO_BASE=$a --array=0-$((b-a))%60 lartpc/data_prep/uboone_official/submit_sidecar_only.sh)
  echo "sidecar array: $J"
  while :; do sleep 60
    rows=$(sacct -j "$J" -n -X --format=State 2>/dev/null | awk 'NF' | wc -l)
    act=$(sacct -j "$J" -n -X --format=State 2>/dev/null | grep -cE "RUNNING|PENDING")
    [ "$rows" -ge 1 ] && [ "$act" -eq 0 ] && break
  done
  sacct -j "$J" -n -X --format=State | sort | uniq -c | tr -s ' ' | tr '\n' ';'; echo
  echo "  sidecars so far: $(ls "$TRUTH_DIR"/truth_fileno*.h5 2>/dev/null | wc -l)"
done
echo "ALL SIDECAR BATCHES DONE"
