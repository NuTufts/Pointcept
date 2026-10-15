#!/bin/bash
# LOGIN-SIDE sequencer for tier2 staged conversion of an arbitrary list.
# Generalization of run_generic_mu_tranche1.sh: each SPEC is
#   "SAMPLE LIST OUT_ROOT MODE TRUTH_DIR STAGE MARK FIRST LAST [TICK]"
# (TRUTH_DIR may be "-" for data mode; TICK = backward (default: mcc9 v28/v29e
# larcv images, converter -tb) | forward (mcc9.10 images) -- check it in the smoke). Batches are run SERIALLY so the tier2
# staging area stays bounded; every step is idempotent (markers + h5 glob), so
# re-running this script resumes wherever it stopped.
#
#   bash run_tier2_tranche.sh <specfile>   # one SPEC per line, '#' comments ok
# THROTTLE=%N (default %60): concurrent array tasks per batch (tier1 sources
# need no staging copy, so a whole batch can run at once, e.g. THROTTLE=%119).
set -u
K=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
cd $K
SPECFILE=${1:?usage: run_tier2_tranche.sh <specfile>}

wait_drain () {
  local J=$1
  while :; do
    sleep 120
    local rows act
    rows=$(sacct -j "$J" -n -X --format=State 2>/dev/null | awk 'NF' | wc -l)
    act=$(sacct -j "$J" -n -X --format=State 2>/dev/null | grep -cE "RUNNING|PENDING")
    [ "$rows" -ge 1 ] && [ "$act" -eq 0 ] && break
  done
  sacct -j "$J" -n -X --format=State | sort | uniq -c | tr -s ' ' | tr '\n' ';'
  echo
}

while read -r SAMPLE LIST OUT_ROOT MODE TRUTH_DIR STAGE MARK FIRST LAST TICK; do
  case "${SAMPLE:-#}" in ''|'#'*) continue ;; esac
  [ "$TRUTH_DIR" = "-" ] && TRUTH_DIR=""
  TICK=${TICK:-backward}
  echo "=== $SAMPLE  filenos $FIRST-$LAST  mode=$MODE tick=$TICK ==="
  OUT=$(LIST=$LIST OUT_ROOT=$OUT_ROOT MODE=$MODE TRUTH_DIR=$TRUTH_DIR \
        STAGE=$STAGE MARK=$MARK TICK=$TICK \
        bash lartpc/data_prep/uboone_official/stage_overlay_train_batch.sh \
             "$SAMPLE" "$FIRST" "$LAST" "${THROTTLE:-%60}")
  echo "$OUT"
  J=$(echo "$OUT" | grep -oE 'conversion array: [0-9]+' | grep -oE '[0-9]+$')
  [ -n "$J" ] || { echo "ERROR: no job id; skipping batch"; continue; }
  wait_drain "$J"
  echo "  markers so far: $(ls "$MARK"/*.ok 2>/dev/null | wc -l) | events: $(find "$OUT_ROOT" -name '*.h5' 2>/dev/null | wc -l) | $(du -sh "$OUT_ROOT" 2>/dev/null | cut -f1)"
done < "$SPECFILE"
echo "ALL BATCHES DONE"
