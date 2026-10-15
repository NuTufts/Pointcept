#!/bin/bash
# ---------------------------------------------------------------------------
# One keypoint2-cascade GPU shard (called by node_launcher.sh, or by hand).
#
#   run_cascade_shard.sh <shard> <start> <n>
#
# Environment (exported by the PBS script through the jobenv file, or set by
# hand): POL_TAG, POL_OUTDIR, POL_LIST, plus everything from polaris_env.sh.
# SLOT (from node_launcher) selects the GPU: CUDA_VISIBLE_DEVICES = SLOT % POL_NGPU
# (several slots per GPU = several processes per GPU, with or without MPS).
#
# Two output modes:
#  * legacy (POL_LOCAL_OUT empty): per-event keypoint2 files written straight
#    into the $POL_OUTDIR index tree on Eagle.
#  * node-local (POL_LOCAL_OUT set, stage B): the shard writes its tree to
#    $POL_LOCAL_OUT/shard<NNN>/ (node NVMe), then packs it into ONE squashfs
#    image $POL_OUTDIR/kp2_<tag>_shard<NNN>.sqfs (+ .manifest with file
#    count / bytes / sha256; `tar` fallback when mksquashfs is absent), copies
#    it to Eagle, verifies the copy's sha256 and only then writes the .done
#    marker. Nothing per-event ever touches Lustre. Stage C at Tufts unpacks
#    the images (lartpc/data_prep/squashfs/unpack_kp2_images.sh).
#
# * log:     $POL_LOGDIR/$POL_TAG/cascade_shard<NNN>.log (timestamped lines)
# * marker:  $POL_OUTDIR/.shards/shard<NNN>.done  -> the shard is skipped on a
#            relaunch; shard<NNN>.running exists while a process is live
#            (the qsub wrappers refuse to submit over live markers).
# * retry:   3 attempts (transient CUDA init errors); --deterministic +
#            reseed_per_event make a rerun of the same range bit-identical.
# ---------------------------------------------------------------------------
set -u
SHARD=${1:?shard}; START=${2:?start}; N=${3:?n}
# shellcheck disable=SC1091
source "${POL_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}/lartpc/polaris/polaris_env.sh"
: "${POL_TAG:?}" "${POL_OUTDIR:?}" "${POL_LIST:?}"
S3=$(printf '%03d' "$SHARD")
LOGD=$POL_LOGDIR/$POL_TAG; MARK=$POL_OUTDIR/.shards
mkdir -p "$LOGD" "$MARK" "$POL_OUTDIR"
LOG=$LOGD/cascade_shard$S3.log
DONE=$MARK/shard$S3.done
RUN=$MARK/shard$S3.running
if [ -f "$DONE" ]; then echo ">>> shard $SHARD already done ($(cat "$DONE")); skip"; exit 0; fi
GPU=$(( ${SLOT:-0} % ${POL_NGPU:-4} ))
export CUDA_VISIBLE_DEVICES=${POL_CUDA_DEVICE_OVERRIDE:-$GPU}
echo "$(hostname) gpu=$CUDA_VISIBLE_DEVICES slot=${SLOT:-?} start=$START n=$N $(date +%FT%T)" > "$RUN"
trap 'rm -f "$RUN"' EXIT

stamp() { while IFS= read -r l; do printf '%s %s\n' "$(date +%FT%T)" "$l"; done; }
# sq <mksquashfs|unsquashfs> args...: run the squashfs tool on the host or inside the container
sq() { local bin=$1; shift; if [ "$SQ" = host ]; then "$bin" "$@"; else pol_exec "'$POL_ASSETS/bin/$bin' $*"; fi; }
if [ -n "${POL_LOCAL_OUT:-}" ]; then
  OUTD=$POL_LOCAL_OUT/shard$S3; rm -rf "$OUTD"; mkdir -p "$OUTD"
  IMG=$POL_OUTDIR/kp2_${POL_TAG}_shard$S3
  # sqfs route: host mksquashfs, or the container-built one in $POL_ASSETS/bin
  # (lartpc/polaris/build_squashfs_tools.sh) run through pol_exec; else tar.
  SQ=""
  if [ "${POL_PACK_FMT:-tar}" = sqfs ]; then
    case "${POL_SQFS_VIA:-auto}" in
      host)      command -v mksquashfs >/dev/null 2>&1 && SQ=host ;;
      container) [ -x "$POL_ASSETS/bin/mksquashfs" ] && SQ=container ;;
      *) if command -v mksquashfs >/dev/null 2>&1; then SQ=host; elif [ -x "$POL_ASSETS/bin/mksquashfs" ]; then SQ=container; fi ;;
    esac
  fi
  if [ -n "$SQ" ]; then IMG=$IMG.sqfs; else IMG=$IMG.tar; fi
else
  OUTD=$POL_OUTDIR; IMG=""
fi
T0=$(date +%s)
{
  echo ">>> shard $SHARD: start-event=$START n-events=$N -> $OUTD${IMG:+ (then packed -> $IMG)}"
  echo ">>> list=$POL_LIST config=$POL_CONFIG"
  echo ">>> flags=$POL_INF_FLAGS"
  echo ">>> slot=${SLOT:-?} gpu=$CUDA_VISIBLE_DEVICES mps=${POL_MPS:-0} threads=$POL_THREADS host=$(hostname) SHARD_T0=$T0"
} | stamp >> "$LOG"

rc=1
for attempt in 1 2 3; do
  echo ">>> attempt $attempt on $(hostname) gpu $CUDA_VISIBLE_DEVICES" | stamp >> "$LOG"
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | stamp >> "$LOG" || true
  set -o pipefail
  pol_exec --nv "python3 -u tools/larformer/run_larformer_keypoint2_cascade_inference.py \
      --config $POL_CONFIG --input-list $POL_LIST --output-dir $OUTD \
      --start-event $START --n-events $N $POL_INF_FLAGS" 2>&1 | stamp >> "$LOG"
  rc=${PIPESTATUS[0]}
  set +o pipefail
  [ "$rc" -eq 0 ] && break
  echo ">>> attempt $attempt failed (rc=$rc); retrying in 20s" | stamp >> "$LOG"
  sleep 20
done
T1=$(date +%s)
echo ">>> inference rc=$rc elapsed=$((T1 - T0))s SHARD_INF_END=$T1" | stamp >> "$LOG"

# ---- node-local mode: pack -> copy -> verify -------------------------------
if [ "$rc" -eq 0 ] && [ -n "$IMG" ]; then
  NLOC=$(find "$OUTD" -name '*.h5' | wc -l)
  TMPIMG=$POL_LOCAL_OUT/$(basename "$IMG")
  rm -f "$TMPIMG"
  case "$IMG" in
    *.sqfs) COMP=${POL_PACK_COMP:-auto}
            if [ "$COMP" = auto ]; then sq mksquashfs -help 2>&1 | grep -qE '^[[:space:]]*lz4' && COMP=lz4 || COMP=gzip; fi
            XOPT=""; [ "$COMP" = gzip ] && XOPT="-Xcompression-level 1"
            echo ">>> mksquashfs via $SQ, -comp $COMP $XOPT" | stamp >> "$LOG"
            sq mksquashfs "$OUTD" "$TMPIMG" -comp "$COMP" $XOPT -no-xattrs -noappend -processors "${POL_PACK_PROCS:-4}" -quiet -no-progress 2>&1 | stamp >> "$LOG"
            NIMG=$(sq unsquashfs -lls "$TMPIMG" 2>/dev/null | grep -c '\.h5$') ;;
    *.tar)  tar -cf "$TMPIMG" -C "$OUTD" . 2>&1 | stamp >> "$LOG"
            NIMG=$(tar -tf "$TMPIMG" | grep -c '\.h5$') ;;
  esac
  SHA=$(sha256sum "$TMPIMG" | cut -d' ' -f1); BYTES=$(stat -c %s "$TMPIMG")
  cp "$TMPIMG" "$IMG.part" && mv "$IMG.part" "$IMG"
  SHA2=$(sha256sum "$IMG" | cut -d' ' -f1)
  T2=$(date +%s)
  if [ "$NIMG" -eq "$NLOC" ] && [ "$SHA" = "$SHA2" ]; then
    { echo "shard=$SHARD start=$START n=$N tag=$POL_TAG sample=$POL_SAMPLE"
      echo "image=$IMG bytes=$BYTES format=${IMG##*.} comp=$([ "${IMG##*.}" = sqfs ] && echo "$COMP" || echo none)"
      echo "h5_in_source=$NLOC h5_in_image=$NIMG"
      echo "host=$(hostname) gpu=$CUDA_VISIBLE_DEVICES slot=${SLOT:-?} mps=${POL_MPS:-0} inference_s=$((T1 - T0)) pack_copy_s=$((T2 - T1))"
      echo "created=$(date -Is)"
      echo "sha256=$SHA"; } > "$IMG.manifest"
    echo ">>> packed $NLOC files -> $IMG ($((BYTES / 1024 / 1024)) MiB, pack+copy $((T2 - T1))s) SHARD_T1=$T2" | stamp >> "$LOG"
  else
    echo ">>> PACK/COPY FAILED: local $NLOC files, image $NIMG, sha local $SHA eagle $SHA2" | stamp >> "$LOG"
    rc=4
  fi
  rm -rf "$OUTD" "$TMPIMG"
fi

if [ "$rc" -eq 0 ]; then
  echo "shard=$SHARD start=$START n=$N host=$(hostname) gpu=$CUDA_VISIBLE_DEVICES attempts=$attempt ${IMG:+image=$IMG }$(date +%FT%T)" > "$DONE"
  echo "DONE shard $SHARD -> ${IMG:-$POL_OUTDIR}" | stamp >> "$LOG"
else
  echo "FAILED shard $SHARD after $attempt attempts (rc=$rc)" | stamp >> "$LOG"
fi
exit "$rc"
