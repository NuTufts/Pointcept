#!/bin/bash
# Sequenced tranche-1 conversion of generic BNB numu overlay TRAINPOOLs
# (muon rebalance for the stage-3 cache). Batches of 250 files: stage
# login-side, submit array, wait for drain (robust sacct loop), next.
set -u
K=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
cd $K
wait_drain () {
  local J=$1
  while :; do
    sleep 300
    local rows=$(sacct -j $J -n -X --format=State 2>/dev/null | awk 'NF' | wc -l)
    local act=$(sacct -j $J -n -X --format=State 2>/dev/null | grep -cE "RUNNING|PENDING")
    [ "$rows" -ge 1 ] && [ "$act" -eq 0 ] && break
  done
  sacct -j $J -n -X --format=State 2>/dev/null | sort | uniq -c
}
for SPEC in "mcc9_v29e_dl_run3b_bnb_nu 1 250" "mcc9_v29e_dl_run3b_bnb_nu 251 500" \
            "mcc9_v28_run1_bnboverlay 1 175" "mcc9_v28_run1_bnboverlay 176 350"; do
  set -- $SPEC
  echo "=== staging+converting $1 filenos $2-$3 ==="
  OUTLINE=$(bash lartpc/data_prep/uboone_official/stage_overlay_train_batch.sh $1 $2 $3 %60 | tee /dev/stderr | tail -1)
  J=$(echo "$OUTLINE" | grep -oE '[0-9]{6,}' | head -1)
  [ -n "$J" ] || { echo "ERROR: no job id for $SPEC"; continue; }
  wait_drain $J
done
echo "=== final counts ==="
for S in mcc9_v29e_dl_run3b_bnb_nu mcc9_v28_run1_bnboverlay; do
  D=/cluster/tufts/wongjiradlab/larbys/data/larformer/overlay_train/$S/merged_sp
  echo "$S: $(find $D -name '*.h5' 2>/dev/null | wc -l) h5 events"
done
df -h /cluster/tufts/wongjiradlab | tail -1
echo DONE_GENERIC_MU_TRANCHE1
