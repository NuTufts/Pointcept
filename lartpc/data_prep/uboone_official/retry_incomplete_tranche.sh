#!/bin/bash
# Retry the filenos of a tranche spec that have NO marker: mid-file converter
# crashes (rc=129 "*** Break *** segmentation violation", or a hung teardown)
# leave PARTIAL merged_sp h5s + a full-file truth sidecar, and the stager's
# h5-glob fallback then skips them forever -> the sample would carry files with
# a full-file POT but only part of their events. This pass deletes the partial
# outputs (h5s, sidecar, stale staged copy), re-runs the sequencer on the spec
# lines that contain such filenos (already-converted filenos are skipped by
# their markers) and reports what is still missing. Known duds (ledger dud
# list) are never retried. LOGIN NODE ONLY (tier2 staging).
#
#   bash retry_incomplete_tranche.sh <spec> [--dry-run] [--max-rounds N]
set -u
SPEC=${1:?specfile}; shift
DRY=0; ROUNDS=2
while [ $# -gt 0 ]; do case "$1" in --dry-run) DRY=1;; --max-rounds) ROUNDS=$2; shift;; esac; shift; done
K=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
cd "$K"; U=lartpc/data_prep/uboone_official
DUDS=$U/training_data_ledger/generic_mu_tranche1_dud_filenos.txt

scan () {   # prints "fileno kind" for every unmarked fileno of the spec (kind: dud|partial|missing)
  while read -r SAMPLE LIST OUT_ROOT MODE TRUTH_DIR STAGE MARK FIRST LAST; do
    case "${SAMPLE:-#}" in ''|'#'*) continue ;; esac
    for i in $(seq "$FIRST" "$LAST"); do
      ZFN=$(printf %05d "$i")
      [ -s "$MARK/fileno$ZFN.ok" ] && continue
      if grep -q "^$SAMPLE $i\$" "$DUDS" 2>/dev/null; then echo "$i dud"; continue; fi
      OUT=$OUT_ROOT/$(printf %03d $((i/100)))
      nh5=$(ls "$OUT"/merged_${SAMPLE}_fileno${ZFN}_*.h5 2>/dev/null | wc -l)
      side=""; [ "$TRUTH_DIR" != "-" ] && [ -s "$TRUTH_DIR/truth_fileno$ZFN.h5" ] && side=1
      if [ "$nh5" -gt 0 ] || [ -n "$side" ]; then echo "$i partial nh5=$nh5 sidecar=${side:-0}"; else echo "$i missing"; fi
    done
  done < "$SPEC"
}
clean () {  # delete partial outputs of the filenos on stdin ("fileno ...")
  while read -r i _; do
    while read -r SAMPLE LIST OUT_ROOT MODE TRUTH_DIR STAGE MARK FIRST LAST; do
      case "${SAMPLE:-#}" in ''|'#'*) continue ;; esac
      [ "$i" -ge "$FIRST" ] && [ "$i" -le "$LAST" ] || continue
      ZFN=$(printf %05d "$i"); OUT=$OUT_ROOT/$(printf %03d $((i/100)))
      rm -f "$OUT"/merged_${SAMPLE}_fileno${ZFN}_*.h5 "$STAGE/fileno$ZFN.root"
      [ "$TRUTH_DIR" != "-" ] && rm -f "$TRUTH_DIR/truth_fileno$ZFN.h5"
      break
    done < "$SPEC"
  done
}

for round in $(seq 1 "$ROUNDS"); do
  mapfile -t ROWS < <(scan)
  NDUD=$(printf '%s\n' "${ROWS[@]}" | grep -c ' dud$' || true)
  mapfile -t TODO < <(printf '%s\n' "${ROWS[@]}" | grep -E ' (partial|missing)' || true)
  echo ">>> round $round: unmarked filenos: ${#ROWS[@]} (duds $NDUD, to retry ${#TODO[@]})"
  printf '    %s\n' "${TODO[@]}"
  [ ${#TODO[@]} -eq 0 ] && { echo ">>> nothing to retry"; break; }
  [ $DRY -eq 1 ] && { echo "(dry run)"; break; }
  if squeue -u "$USER" -h -o '%j' | grep -q ovl_train_conv; then echo "ERROR: conversion arrays still running; retry later" >&2; exit 2; fi
  [ -d /cluster/tier2/wongjiradlab ] || { echo "ERROR: tier2 not visible; run on a login node" >&2; exit 2; }
  printf '%s\n' "${TODO[@]}" | clean
  RETRY=${SPEC}.retry$round; : > "$RETRY"
  while read -r line; do
    set -- $line; case "${1:-#}" in ''|'#'*) continue ;; esac
    FIRST=${8}; LAST=${9}
    for t in "${TODO[@]}"; do i=${t%% *}; if [ "$i" -ge "$FIRST" ] && [ "$i" -le "$LAST" ]; then echo "$line" >> "$RETRY"; break; fi; done
  done < "$SPEC"
  echo ">>> retry spec: $RETRY ($(grep -c . "$RETRY") batches)"
  bash $U/run_tier2_tranche.sh "$RETRY"
done
mapfile -t LEFT < <(scan | grep -vE ' dud$' || true)
echo ">>> still incomplete after retries: ${#LEFT[@]}"; printf '    %s\n' "${LEFT[@]}"
if [ ${#LEFT[@]} -gt 0 ] && [ $DRY -eq 0 ]; then
  # deterministic crashers: DROP the whole file (partial h5s + sidecar) so the
  # event universe and the POT stay consistent; the list is kept for the record
  printf '%s\n' "${LEFT[@]}" | clean
  printf '%s\n' "${LEFT[@]}" > "${SPEC}.dropped"
  echo ">>> dropped ${#LEFT[@]} filenos (partial outputs + sidecars removed) -> ${SPEC}.dropped"
fi
exit 0
