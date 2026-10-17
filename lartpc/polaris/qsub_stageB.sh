#!/bin/bash
# ---------------------------------------------------------------------------
# Build (and run) the qsub line for the stage-B GPU pass or its test job.
#
#   POL_SAMPLE_ENV=lartpc/polaris/samples/<x>.env bash lartpc/polaris/qsub_stageB.sh --test
#   POL_SAMPLE_ENV=... MODE=prod NODES=16 PPG=2 MPS=1 bash lartpc/polaris/qsub_stageB.sh
#   DRYRUN=1 ...                         # print the qsub line only
#
# Defaults: MODE=prod NODES=16 PPG=1 MPS=0 WALLTIME=03:00:00 TAG=prod
#   THREADS:  CPU threads per process; default computed in stageB.pbs as 32/(4*PPG), clamped 2..8
#             (the same rule the test matrix uses). NOT POL_THREADS: polaris_env.sh, sourced here and
#             in the caller's shell, exports POL_THREADS=2, which used to be forwarded and override the rule.
#   WORKLIST: built here as NODES x 4 x PPG equal shards over LIST (one uniform
#             wave, so the utilization measurement is clean) unless given
#   OUTDIR:   $POL_DATADIR/kp2_images (prod) | $POL_DATADIR/tests/$TAG (other tags)
#   LIST:     $POL_DATADIR/lists/$POL_LIST_NAME (the test job builds it)
# Guards: prod needs $POL_DATADIR/tests/stageB/.PASS (FORCE=1 overrides);
#         refuses if $OUTDIR/.shards/*.running exist (FORCE=1 after `qstat -u $USER`).
# Accounts: debug -> $POL_DEBUG_ACCOUNT, prod/preempt -> $POL_ACCOUNT.
# ---------------------------------------------------------------------------
set -eu
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd); cd "$REPO"
: "${POL_SAMPLE_ENV:?set POL_SAMPLE_ENV (e.g. lartpc/polaris/samples/run4ovl_4a.env)}"
# shellcheck disable=SC1091
source lartpc/polaris/polaris_env.sh
TEST=0; [ "${1:-}" = "--test" ] && TEST=1
MODE=${MODE:-prod}; TAG=${TAG:-prod}; PPG=${PPG:-1}; MPS=${MPS:-0}
LIST=${LIST:-$POL_DATADIR/lists/$POL_LIST_NAME}
if [ $TEST -eq 1 ]; then MODE=debug; TAG=testB; NODES=1; WALLTIME=${WALLTIME:-01:00:00}; fi
case "$MODE" in
  debug)   QUEUE=debug;       ACCT=$POL_DEBUG_ACCOUNT; NODES=${NODES:-1};  WALLTIME=${WALLTIME:-01:00:00} ;;
  prod)    QUEUE=prod;        ACCT=$POL_ACCOUNT;       NODES=${NODES:-16}; WALLTIME=${WALLTIME:-03:00:00} ;;
  preempt) QUEUE=preemptable; ACCT=$POL_ACCOUNT;       NODES=${NODES:-4};  WALLTIME=${WALLTIME:-12:00:00} ;;
  *) echo "MODE must be debug|prod|preempt" >&2; exit 2 ;;
esac
mkdir -p "$POL_LOGDIR/$TAG" "$POL_DATADIR/lists"
SENV=$(readlink -f "$POL_SAMPLE_ENV")
echo ">>> sample=$POL_SAMPLE ($SENV)  images=$POL_IMGDIR  datadir=$POL_DATADIR"
echo ">>> sbank check: sbank-list-allocations -r polaris -p ${ACCT%%::*} -f \"+subname users_list\"  (charging $ACCT)"
if [ $TEST -eq 1 ]; then
  V="POL_SAMPLE_ENV='$SENV',NTHR='${NTHR:-40}',MATRIX='${MATRIX:-1:0 2:1 3:1}'${SKIP_CHECKSUM:+,SKIP_CHECKSUM=$SKIP_CHECKSUM}${POL_PACK_FMT:+,POL_PACK_FMT=$POL_PACK_FMT}${POL_SQFS_VIA:+,POL_SQFS_VIA=$POL_SQFS_VIA}${POL_PACK_COMP:+,POL_PACK_COMP=$POL_PACK_COMP}"
  CMD=(qsub -A "$ACCT" -q "$QUEUE" -l select=1:system=polaris -l "walltime=$WALLTIME" -l filesystems=home:eagle -l place=scatter -N "kp2testB_$POL_SAMPLE" -o "$POL_LOGDIR/testB" -j oe -v "$V" lartpc/polaris/test_stageB.pbs)
else
  if [ "$TAG" = prod ]; then OUTDIR=${OUTDIR:-$POL_DATADIR/kp2_images}; else OUTDIR=${OUTDIR:-$POL_DATADIR/tests/$TAG}; fi
  [ -s "$LIST" ] || { echo "ERROR: input list $LIST missing (run the test job first: it builds it)" >&2; exit 2; }
  if [ -z "${WORKLIST:-}" ]; then
    WORKLIST=$POL_DATADIR/lists/stageB_${TAG}_n${NODES}_p${PPG}.wl
    python3 lartpc/polaris/make_worklist.py --mode cascade --list "$LIST" --nshards $((NODES * POL_NGPU * PPG)) --out "$WORKLIST"
  fi
  [ -s "$WORKLIST" ] || { echo "ERROR: worklist $WORKLIST missing/empty" >&2; exit 2; }
  if [ "$MODE" = prod ] && [ "${FORCE:-0}" != 1 ] && [ ! -f "$POL_DATADIR/tests/stageB/.PASS" ]; then
    echo "ERROR: $POL_DATADIR/tests/stageB/.PASS not found -- run 'bash lartpc/polaris/qsub_stageB.sh --test' first (FORCE=1 to override)" >&2; exit 2
  fi
  if [ "${FORCE:-0}" != 1 ] && ls "$OUTDIR"/.shards/*.running >/dev/null 2>&1; then
    echo "ERROR: live/stale shard markers in $OUTDIR/.shards (a job may still be writing there):" >&2
    ls "$OUTDIR"/.shards/*.running >&2; echo "check qstat -u \$USER; FORCE=1 to override" >&2; exit 2
  fi
  if [ "$MODE" = prod ] && { [ "$NODES" -lt 10 ] || [ "$NODES" -gt 24 ]; }; then
    echo "WARNING: prod routes 10-24 nodes to 'small' (3 h cap); $NODES nodes goes to medium/large" >&2
  fi
  NSH=$(grep -c . "$WORKLIST"); SLOTS=$((NODES * POL_NGPU * PPG))
  echo ">>> $NSH shards on $NODES nodes x $POL_NGPU GPUs x $PPG proc/GPU = $SLOTS slots (waves: $(( (NSH + SLOTS - 1) / SLOTS ))), MPS=$MPS"
  V="POL_SAMPLE_ENV='$SENV',WORKLIST='$WORKLIST',TAG='$TAG',OUTDIR='$OUTDIR',LIST='$LIST',PPG='$PPG',MPS='$MPS'${THREADS:+,POL_THREADS=$THREADS}${POL_PACK_FMT:+,POL_PACK_FMT=$POL_PACK_FMT}${POL_SQFS_VIA:+,POL_SQFS_VIA=$POL_SQFS_VIA}${POL_PACK_COMP:+,POL_PACK_COMP=$POL_PACK_COMP}"
  CMD=(qsub -A "$ACCT" -q "$QUEUE" -l "select=$NODES:system=polaris" -l "walltime=$WALLTIME" -l filesystems=home:eagle -l place=scatter -N "kp2B_${POL_SAMPLE}_$TAG" -o "$POL_LOGDIR/$TAG" -j oe -v "$V" lartpc/polaris/stageB.pbs)
fi
echo ">>> ${CMD[*]}"
if [ "${DRYRUN:-0}" = 1 ]; then echo "(DRYRUN: not submitted)"; exit 0; fi
"${CMD[@]}"
