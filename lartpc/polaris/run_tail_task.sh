#!/bin/bash
# ---------------------------------------------------------------------------
# One CPU-tail task inside the container (called by node_launcher.sh from
# tail_driver.sh). Mirrors lartpc/larformer_reco/slurm/submit_{nu_reco,larpid,
# export}_shard.sh and submit_export_merge.sh command for command.
#
#   run_tail_task.sh nu_reco <stream> <start> <n>
#   run_tail_task.sh larpid  <stream> <nu_reco_shard.h5>
#   run_tail_task.sh export  <shard> <start> <n>
#   run_tail_task.sh hadd
#
# Environment (from the jobenv written by tail.pbs): POL_TAG, POL_TAIL (output
# root), POL_MSP_LIST, POL_KP2_NU, POL_KP2_FM, POL_NTUPLE; plus polaris_env.sh.
# Every task writes $POL_TAIL/.done/<task>.done and is skipped when it exists.
# ---------------------------------------------------------------------------
set -u
STAGE=${1:?stage}; shift
# shellcheck disable=SC1091
source "${POL_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}/lartpc/polaris/polaris_env.sh"
: "${POL_TAG:?}" "${POL_TAIL:?}" "${POL_MSP_LIST:?}" "${POL_KP2_NU:?}" "${POL_KP2_FM:?}" "${POL_NTUPLE:?}"
LOGD=$POL_LOGDIR/$POL_TAG/tail; MARK=$POL_TAIL/.done; mkdir -p "$LOGD" "$MARK"
stamp() { while IFS= read -r l; do printf '%s %s\n' "$(date +%FT%T)" "$l"; done; }
kp2_of() { [ "$1" = fm ] && echo "$POL_KP2_FM" || echo "$POL_KP2_NU"; }

case "$STAGE" in
  nu_reco)
    STREAM=${1:?stream}; START=${2:?start}; N=${3:?n}
    ID=nu_reco_${STREAM}_$(printf '%07d' "$START"); OUTD=$POL_TAIL/nu_reco_streams_$STREAM
    OUTF=$OUTD/nu_reco_shard$(printf '%07d' "$START").h5
    CMD="python3 -u lartpc/larformer_reco/scripts/run_nu_reco.py --keypoint2-list $(kp2_of "$STREAM") \
         --merged-sp-list $POL_MSP_LIST --output-dir $OUTD/ --start $START --n $N \
         --attach-llr-tables $POL_ATTACH_LLR --attach-llr-thr $POL_ATTACH_LLR_THR ${POL_NU_RECO_EXTRA_ARGS:-}" ;;
  larpid)
    STREAM=${1:?stream}; IN=${2:?nu_reco shard}
    B=$(basename "$IN"); ID=larpid_${STREAM}_${B%.h5}; OUTD=$POL_TAIL/nu_reco_larpid_$STREAM
    OUTF=$OUTD/nu_reco_larpid_${B#nu_reco_}
    CMD="python3 -u lartpc/larformer_reco/larpid/apply_larpid.py --nu-reco-shard $IN --kp2-list $(kp2_of "$STREAM") \
         --merged-sp-list $POL_MSP_LIST --out $OUTF --sample-tag $POL_LARPID_TAG --device cpu" ;;
  export)
    SHARD=${1:?shard}; START=${2:?start}; N=${3:?n}
    ID=export_$(printf '%02d' "$SHARD"); OUTD=$(dirname "$POL_NTUPLE")
    OUTF=$(printf '%s_shard%02d.root' "${POL_NTUPLE%.root}" "$SHARD")
    CMD="python3 -u lartpc/larformer_reco/export/export_gen2ntuple.py --merged-sp-list $POL_MSP_LIST \
         --truth-dir $POL_TAIL/truth_sidecar_absent --kp2-nu-list $POL_KP2_NU --kp2-fm-list $POL_KP2_FM \
         --nu-reco-nu-dir $POL_TAIL/nu_reco_larpid_nu --nu-reco-fm-dir $POL_TAIL/nu_reco_larpid_fm \
         --weights-pkl none --start $START --n $N --out $OUTF ${POL_EXPORT_EXTRA_ARGS:-}" ;;
  hadd)
    ID=hadd; OUTD=$(dirname "$POL_NTUPLE"); OUTF=$POL_NTUPLE
    SHARDS=$(ls "${POL_NTUPLE%.root}"_shard*.root 2>/dev/null | sort | tr '\n' ' ')
    [ -n "$SHARDS" ] || { echo "ERROR: no export shards for $POL_NTUPLE" >&2; exit 2; }
    CMD="source /opt/root/bin/thisroot.sh && hadd -f $OUTF $SHARDS && python3 lartpc/polaris/hadd_check.py $OUTF $SHARDS" ;;
  *) echo "unknown stage $STAGE" >&2; exit 2 ;;
esac

DONE=$MARK/$ID.done; LOG=$LOGD/$ID.log
if [ -f "$DONE" ] && [ -e "$OUTF" ]; then echo ">>> $ID already done; skip"; exit 0; fi
mkdir -p "$OUTD"
echo ">>> $ID on $(hostname) slot ${SLOT:-?}: $CMD" | stamp >> "$LOG"
set -o pipefail
pol_exec "$CMD" 2>&1 | stamp >> "$LOG"
rc=${PIPESTATUS[0]}
if [ "$rc" -eq 0 ] && [ -e "$OUTF" ]; then
  echo "$ID host=$(hostname) $(date +%FT%T)" > "$DONE"; echo "DONE $ID" | stamp >> "$LOG"
else
  echo "FAILED $ID rc=$rc (output $OUTF $([ -e "$OUTF" ] && echo present || echo missing))" | stamp >> "$LOG"
  [ "$rc" -eq 0 ] && rc=3
fi
exit "$rc"
