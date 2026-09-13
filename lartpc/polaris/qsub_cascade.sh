#!/bin/bash
# ---------------------------------------------------------------------------
# Build (and run) the qsub line for the cascade GPU pass or the test job.
#
#   MODE=debug|prod|preempt NODES=.. WALLTIME=.. TAG=.. WORKLIST=.. bash qsub_cascade.sh
#   bash qsub_cascade.sh --test          # submit test_cascade.pbs (debug queue, 1 node)
#   DRYRUN=1 ...                         # print the qsub line only
#
# Defaults: MODE=prod NODES=16 WALLTIME=03:00:00 TAG=prod
#           OUTDIR=$POL_DATADIR/keypoint2_streams   (prod)  |  $POL_DATADIR/tests/$TAG (other tags)
#           LIST=$POL_DATADIR/lists/merged_sp_bnb5e19_polaris.txt
# Guards: prod needs $POL_DATADIR/tests/throughput/.PASS (FORCE=1 overrides);
#         refuses if $OUTDIR/.shards/*.running exist (a live or killed job;
#         FORCE=1 overrides after you checked `qstat -u $USER`).
# Accounts: debug -> $POL_DEBUG_ACCOUNT, prod/preempt -> $POL_ACCOUNT.
# ---------------------------------------------------------------------------
set -eu
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd); cd "$REPO"
# shellcheck disable=SC1091
source lartpc/polaris/polaris_env.sh
TEST=0; [ "${1:-}" = "--test" ] && TEST=1
MODE=${MODE:-prod}; TAG=${TAG:-prod}
LIST=${LIST:-$POL_DATADIR/lists/merged_sp_bnb5e19_polaris.txt}
if [ $TEST -eq 1 ]; then MODE=debug; TAG=test; NODES=1; WALLTIME=${WALLTIME:-00:45:00}; fi
case "$MODE" in
  debug)   QUEUE=debug;       ACCT=$POL_DEBUG_ACCOUNT; NODES=${NODES:-1};  WALLTIME=${WALLTIME:-01:00:00} ;;
  prod)    QUEUE=prod;        ACCT=$POL_ACCOUNT;       NODES=${NODES:-16}; WALLTIME=${WALLTIME:-03:00:00} ;;
  preempt) QUEUE=preemptable; ACCT=$POL_ACCOUNT;       NODES=${NODES:-4};  WALLTIME=${WALLTIME:-12:00:00} ;;
  *) echo "MODE must be debug|prod|preempt" >&2; exit 2 ;;
esac
if [ "$TAG" = prod ]; then OUTDIR=${OUTDIR:-$POL_DATADIR/keypoint2_streams}; else OUTDIR=${OUTDIR:-$POL_DATADIR/tests/$TAG}; fi
mkdir -p "$POL_LOGDIR/$TAG"
echo ">>> sbank check: sbank-list-allocations -r polaris -p ${ACCT%%::*} -f \"+subname users_list\"  (charging $ACCT)"

if [ $TEST -eq 1 ]; then
  CMD="qsub -A $ACCT -q $QUEUE -l select=1:system=polaris -l walltime=$WALLTIME -l filesystems=home:eagle -l place=scatter -N kp2test -o $POL_LOGDIR/test -j oe lartpc/polaris/test_cascade.pbs"
else
  WORKLIST=${WORKLIST:?set WORKLIST (make_worklist.py --mode cascade ...)}
  [ -s "$WORKLIST" ] || { echo "ERROR: worklist $WORKLIST missing/empty" >&2; exit 2; }
  [ -s "$LIST" ] || { echo "ERROR: input list $LIST missing (run the test job first: it builds it)" >&2; exit 2; }
  if [ "$MODE" = prod ] && [ "${FORCE:-0}" != 1 ] && [ ! -f "$POL_DATADIR/tests/throughput/.PASS" ]; then
    echo "ERROR: $POL_DATADIR/tests/throughput/.PASS not found -- run 'bash lartpc/polaris/qsub_cascade.sh --test' first (FORCE=1 to override)" >&2; exit 2
  fi
  if [ "${FORCE:-0}" != 1 ] && ls "$OUTDIR"/.shards/*.running >/dev/null 2>&1; then
    echo "ERROR: live/stale shard markers in $OUTDIR/.shards (a job may still be writing there):" >&2
    ls "$OUTDIR"/.shards/*.running >&2; echo "check qstat -u \$USER; FORCE=1 to override" >&2; exit 2
  fi
  if [ "$MODE" = prod ] && { [ "$NODES" -lt 10 ] || [ "$NODES" -gt 24 ]; }; then
    echo "WARNING: prod routes 10-24 nodes to 'small' (3 h cap); $NODES nodes goes to medium/large" >&2
  fi
  NSH=$(grep -c . "$WORKLIST"); SLOTS=$((NODES * 4))
  echo ">>> $NSH shards on $NODES nodes x 4 GPUs = $SLOTS slots (waves: $(( (NSH + SLOTS - 1) / SLOTS )))"
  CMD="qsub -A $ACCT -q $QUEUE -l select=$NODES:system=polaris -l walltime=$WALLTIME -l filesystems=home:eagle -l place=scatter -N kp2_$TAG -o $POL_LOGDIR/$TAG -j oe -v WORKLIST=$WORKLIST,TAG=$TAG,OUTDIR=$OUTDIR,LIST=$LIST lartpc/polaris/cascade.pbs"
fi
echo ">>> $CMD"
if [ "${DRYRUN:-0}" = 1 ]; then echo "(DRYRUN: not submitted)"; exit 0; fi
$CMD
