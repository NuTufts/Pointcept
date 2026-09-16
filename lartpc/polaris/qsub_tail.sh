#!/bin/bash
# ---------------------------------------------------------------------------
# Build (and run) the qsub line for the CPU tail.
#
#   TAG=prod MODE=prod NODES=10 WALLTIME=03:00:00 bash lartpc/polaris/qsub_tail.sh
#   TAG=throughput MODE=debug MAX_EVENTS=200 bash lartpc/polaris/qsub_tail.sh   # test on the 200-event tree
#   DRYRUN=1 ...   STAGES=regen,nu_reco ...   NNR=.. NEXP=..
#
# TAG=prod: KP2_STREAMS=$POL_DATADIR/keypoint2_streams, TAIL=$POL_DATADIR
# other TAG: KP2_STREAMS=$POL_DATADIR/tests/$TAG,       TAIL=$POL_DATADIR/tests/$TAG/tail
# ---------------------------------------------------------------------------
set -eu
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd); cd "$REPO"
# shellcheck disable=SC1091
source lartpc/polaris/polaris_env.sh
TAG=${TAG:-prod}; MODE=${MODE:-prod}
MSP_LIST=${MSP_LIST:-$POL_DATADIR/lists/merged_sp_bnb5e19_polaris.txt}
if [ "$TAG" = prod ]; then
  KP2_STREAMS=${KP2_STREAMS:-$POL_DATADIR/keypoint2_streams}; TAIL=${TAIL:-$POL_DATADIR}
else
  KP2_STREAMS=${KP2_STREAMS:-$POL_DATADIR/tests/$TAG}; TAIL=${TAIL:-$POL_DATADIR/tests/$TAG/tail}
fi
case "$MODE" in
  debug)   QUEUE=debug;       ACCT=$POL_DEBUG_ACCOUNT; NODES=${NODES:-1};  WALLTIME=${WALLTIME:-00:45:00} ;;
  prod)    QUEUE=prod;        ACCT=$POL_ACCOUNT;       NODES=${NODES:-10}; WALLTIME=${WALLTIME:-03:00:00} ;;
  preempt) QUEUE=preemptable; ACCT=$POL_ACCOUNT;       NODES=${NODES:-4};  WALLTIME=${WALLTIME:-08:00:00} ;;
  *) echo "MODE must be debug|prod|preempt" >&2; exit 2 ;;
esac
[ -d "$KP2_STREAMS" ] || { echo "ERROR: cascade tree $KP2_STREAMS missing" >&2; exit 2; }
[ -s "$MSP_LIST" ] || { echo "ERROR: merged_sp list $MSP_LIST missing" >&2; exit 2; }
if [ "$TAG" = prod ] && [ "${FORCE:-0}" != 1 ] && ! ls "$KP2_STREAMS"/.shards/*.done >/dev/null 2>&1; then
  echo "ERROR: no finished cascade shards in $KP2_STREAMS/.shards (run check_cascade.py first; FORCE=1 to override)" >&2; exit 2
fi
if ls "$KP2_STREAMS"/.shards/*.running >/dev/null 2>&1 && [ "${FORCE:-0}" != 1 ]; then
  echo "ERROR: cascade shards still running/stale in $KP2_STREAMS/.shards (FORCE=1 to override)" >&2; exit 2
fi
mkdir -p "$POL_LOGDIR/$TAG/tail" "$TAIL"
# PBS -v separates variables with commas, so every value is single-quoted
# (STAGES=export,hadd would otherwise be split: "cannot send environment")
V="TAG='$TAG',KP2_STREAMS='$KP2_STREAMS',TAIL='$TAIL',MSP_LIST='$MSP_LIST'"
for k in STAGES NNR NEXP MAX_EVENTS FORCE_REGEN PPN_NU_RECO PPN_LARPID PPN_EXPORT; do
  v=${!k:-}; [ -n "$v" ] && V="$V,$k='$v'"
done
echo ">>> sbank check: sbank-list-allocations -r polaris -p ${ACCT%%::*} -f \"+subname users_list\"  (charging $ACCT)"
CMD=(qsub -A "$ACCT" -q "$QUEUE" -l "select=$NODES:system=polaris" -l "walltime=$WALLTIME" -l filesystems=home:eagle -l place=scatter -N "kp2tail_$TAG" -o "$POL_LOGDIR/$TAG/tail" -j oe -v "$V" lartpc/polaris/tail.pbs)
printf '>>> %q ' "${CMD[@]}"; echo
if [ "${DRYRUN:-0}" = 1 ]; then echo "(DRYRUN: not submitted)"; exit 0; fi
"${CMD[@]}"
