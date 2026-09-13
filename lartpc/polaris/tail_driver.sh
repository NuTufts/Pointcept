#!/bin/bash
# ---------------------------------------------------------------------------
# CPU tail on Polaris, run INSIDE tail.pbs on the head node:
#   regen -> nu_reco (nu+fm) -> larpid (nu+fm) -> export -> hadd
# Each parallel stage = one worklist run by node_launcher.sh on every node of
# the job (mpiexec --ppn 1). Stages are idempotent (per-task .done markers), so
# a relaunch of the same job resumes. STAGES=regen,nu_reco,... limits the run.
#
# Needs (exported by tail.pbs): JOBENV, POL_TAG, POL_TAIL, POL_KP2_STREAMS,
# POL_MSP_LIST, POL_KP2_NU, POL_KP2_FM, POL_NTUPLE, NNR, NEXP, PPN_* and
# polaris_env.sh. Stage counts: nu_reco NNR shards per stream (default 120),
# export NEXP (<= 99, default 96); shards never smaller than 50 events.
# ---------------------------------------------------------------------------
set -u
: "${JOBENV:?}" "${POL_TAG:?}" "${POL_TAIL:?}" "${POL_KP2_STREAMS:?}" "${POL_MSP_LIST:?}" \
  "${POL_KP2_NU:?}" "${POL_KP2_FM:?}" "${POL_NTUPLE:?}"
STAGES=${STAGES:-regen,nu_reco,larpid,export,hadd}
NNR=${NNR:-120}; NEXP=${NEXP:-96}
PPN_NU_RECO=${PPN_NU_RECO:-24}; PPN_LARPID=${PPN_LARPID:-32}; PPN_EXPORT=${PPN_EXPORT:-24}
NN=1; HOSTFILE=${POL_HOSTFILE:-${PBS_NODEFILE:-}}
if [ -n "$HOSTFILE" ] && [ -r "$HOSTFILE" ]; then NN=$(awk '!s[$0]++' "$HOSTFILE" | wc -l); fi
WLD=$POL_TAIL/worklists; mkdir -p "$WLD" "$POL_TAIL/.done" "$POL_LOGDIR/$POL_TAG/tail" \
  "$POL_TAIL"/nu_reco_streams_{nu,fm} "$POL_TAIL"/nu_reco_larpid_{nu,fm}
PL=$POL_REPO/lartpc/polaris
want() { [[ ",$STAGES," == *",$1,"* ]]; }
banner() { echo; echo "=================== $(date +%FT%T) tail/$POL_TAG: $*"; }
launch() {  # launch <worklist> <ppn> <stage>
  local wl=$1 ppn=$2 stage=$3 rc
  [ -s "$wl" ] || { echo ">>> $stage: empty worklist, nothing to do"; return 0; }
  echo ">>> $stage: $(grep -c . "$wl") tasks on $NN node(s) x $ppn"
  if [ "$NN" -gt 1 ]; then
    mpiexec -n "$NN" --ppn 1 --hostfile "$HOSTFILE" --cpu-bind none \
      bash "$PL/node_launcher.sh" "$JOBENV" "$wl" "$ppn" "$PL/run_tail_task.sh" "$stage"
  else
    bash "$PL/node_launcher.sh" "$JOBENV" "$wl" "$ppn" "$PL/run_tail_task.sh" "$stage"
  fi
  rc=$?; echo ">>> $stage: launcher rc=$rc $(date +%FT%T)"; return $rc
}

# ---- regen: split the cascade tree into the nu / fm lists (find, never ls) --
if want regen; then
  banner "regen lists from $POL_KP2_STREAMS"
  if [ -s "$POL_KP2_NU" ] && ls "$POL_TAIL"/nu_reco_streams_nu/nu_reco_shard*.h5 >/dev/null 2>&1 && [ "${FORCE_REGEN:-0}" != 1 ]; then
    echo ">>> lists exist and nu_reco outputs exist: NOT regenerating (gidx = line number must stay fixed; FORCE_REGEN=1 to override)"
  else
    find "$POL_KP2_STREAMS" -name 'keypoint2_event*_0.h5' ! -name '*_fm_0.h5' | sort > "$POL_KP2_NU"
    find "$POL_KP2_STREAMS" -name 'keypoint2_event*_fm_0.h5' | sort > "$POL_KP2_FM"
  fi
  wc -l "$POL_KP2_NU" "$POL_KP2_FM"
  [ -s "$POL_KP2_NU" ] || { echo "ERROR: empty nu list" >&2; exit 2; }
fi

# ---- nu_reco (nu + fm in one worklist) ---------------------------------------
if want nu_reco; then
  banner "nu_reco"
  WL=$WLD/nu_reco.wl
  python3 "$PL/make_worklist.py" --mode nu_reco --list "$POL_KP2_NU" --stream nu --nshards "$NNR" --out "$WL"
  [ -s "$POL_KP2_FM" ] && python3 "$PL/make_worklist.py" --mode nu_reco --list "$POL_KP2_FM" --stream fm --nshards "$NNR" --out "$WL" --append
  launch "$WL" "$PPN_NU_RECO" nu_reco || exit 3
fi

# ---- larpid (one task per nu_reco shard file) ---------------------------------
if want larpid; then
  banner "larpid"
  WL=$WLD/larpid.wl
  python3 "$PL/make_worklist.py" --mode larpid --nu-reco-dir "$POL_TAIL/nu_reco_streams_nu" --stream nu --out "$WL"
  python3 "$PL/make_worklist.py" --mode larpid --nu-reco-dir "$POL_TAIL/nu_reco_streams_fm" --stream fm --out "$WL" --append
  launch "$WL" "$PPN_LARPID" larpid || exit 4
fi

# ---- export ---------------------------------------------------------------------
if want export; then
  banner "export (data mode, --weights-pkl none)"
  WL=$WLD/export.wl
  python3 "$PL/make_worklist.py" --mode export --list "$POL_MSP_LIST" --nshards "$NEXP" --out "$WL"
  launch "$WL" "$PPN_EXPORT" export || exit 5
fi

# ---- hadd (head node) ----------------------------------------------------------
if want hadd; then
  banner "hadd"
  ( source "$JOBENV"; SLOT=0 bash "$PL/run_tail_task.sh" hadd ) || exit 6
  tail -3 "$POL_LOGDIR/$POL_TAG/tail/hadd.log"
  echo ">>> ntuple: $POL_NTUPLE"
fi
echo ">>> tail/$POL_TAG: stages [$STAGES] complete $(date +%FT%T)"
