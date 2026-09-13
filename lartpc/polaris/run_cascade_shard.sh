#!/bin/bash
# ---------------------------------------------------------------------------
# One keypoint2-cascade GPU shard (called by node_launcher.sh, or by hand).
#
#   run_cascade_shard.sh <shard> <start> <n>
#
# Environment (exported by cascade.pbs / test_cascade.pbs through the jobenv
# file, or set by hand): POL_TAG, POL_OUTDIR, POL_LIST, plus everything from
# polaris_env.sh. SLOT (0-3, from node_launcher) selects the GPU.
#
# * log:     $POL_LOGDIR/$POL_TAG/cascade_shard<shard>.log (timestamped lines)
# * marker:  $POL_OUTDIR/.shards/shard<shard>.done  -> the shard is skipped on a
#            relaunch; shard<shard>.running exists while a process is live
#            (qsub_cascade.sh refuses to submit over live markers).
# * retry:   3 attempts (transient CUDA init errors); --deterministic +
#            reseed_per_event make a rerun of the same range bit-identical.
# ---------------------------------------------------------------------------
set -u
SHARD=${1:?shard}; START=${2:?start}; N=${3:?n}
# shellcheck disable=SC1091
source "${POL_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}/lartpc/polaris/polaris_env.sh"
: "${POL_TAG:?}" "${POL_OUTDIR:?}" "${POL_LIST:?}"
LOGD=$POL_LOGDIR/$POL_TAG; MARK=$POL_OUTDIR/.shards
mkdir -p "$LOGD" "$MARK" "$POL_OUTDIR"
LOG=$LOGD/cascade_shard$(printf '%03d' "$SHARD").log
DONE=$MARK/shard$(printf '%03d' "$SHARD").done
RUN=$MARK/shard$(printf '%03d' "$SHARD").running
if [ -f "$DONE" ]; then echo ">>> shard $SHARD already done ($(cat "$DONE")); skip"; exit 0; fi
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-${SLOT:-0}}
echo "$(hostname) gpu=$CUDA_VISIBLE_DEVICES start=$START n=$N $(date +%FT%T)" > "$RUN"
trap 'rm -f "$RUN"' EXIT

stamp() { while IFS= read -r l; do printf '%s %s\n' "$(date +%FT%T)" "$l"; done; }
{
  echo ">>> shard $SHARD: start-event=$START n-events=$N -> $POL_OUTDIR"
  echo ">>> list=$POL_LIST config=$POL_CONFIG"
  echo ">>> flags=$POL_INF_FLAGS"
} | stamp >> "$LOG"

rc=1
for attempt in 1 2 3; do
  echo ">>> attempt $attempt on $(hostname) gpu $CUDA_VISIBLE_DEVICES" | stamp >> "$LOG"
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | stamp >> "$LOG" || true
  set -o pipefail
  pol_exec --nv "python3 -u tools/larformer/run_larformer_keypoint2_cascade_inference.py \
      --config $POL_CONFIG --input-list $POL_LIST --output-dir $POL_OUTDIR \
      --start-event $START --n-events $N $POL_INF_FLAGS" 2>&1 | stamp >> "$LOG"
  rc=${PIPESTATUS[0]}
  set +o pipefail
  [ "$rc" -eq 0 ] && break
  echo ">>> attempt $attempt failed (rc=$rc); retrying in 20s" | stamp >> "$LOG"
  sleep 20
done
if [ "$rc" -eq 0 ]; then
  echo "shard=$SHARD start=$START n=$N host=$(hostname) gpu=$CUDA_VISIBLE_DEVICES attempts=$attempt $(date +%FT%T)" > "$DONE"
  echo "DONE shard $SHARD -> $POL_OUTDIR" | stamp >> "$LOG"
else
  echo "FAILED shard $SHARD after $attempt attempts (rc=$rc)" | stamp >> "$LOG"
fi
exit "$rc"
