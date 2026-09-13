#!/bin/bash
# ---------------------------------------------------------------------------
# Generic per-node worklist runner for PBS jobs (no job arrays needed).
#
#   node_launcher.sh <jobenv|-> <worklist> <ppn> <task script> [task args...]
#
# Run once per node, e.g.
#   mpiexec -n $NN --ppn 1 --hostfile $PBS_NODEFILE --cpu-bind none \
#       bash lartpc/polaris/node_launcher.sh $JOBENV $WORKLIST 4 lartpc/polaris/run_cascade_shard.sh
# or directly on a single node (rank 0 of 1).
#
# * jobenv: a file to `source` first (exports POL_* / job variables written by
#   the PBS script) or "-" for none. Tasks inherit the exported environment.
# * The node's rank = position of $(hostname) in the unique lines of
#   $PBS_NODEFILE (fallback: $PMI_RANK, then 0); it takes worklist line i when
#   i % NNODES == rank, so a worklist longer than NNODES*ppn is queued.
# * Up to <ppn> tasks run concurrently; each gets SLOT=0..ppn-1 in its
#   environment (the GPU task maps SLOT -> CUDA_VISIBLE_DEVICES).
# * A task is `<task script> [task args...] <fields of the worklist line>`.
# * Exit status is non-zero if any task failed; per-task results are printed.
# ---------------------------------------------------------------------------
set -u
JOBENV=${1:?jobenv file or -}; WL=${2:?worklist}; PPN=${3:?ppn}; TASK=${4:?task script}
shift 4
if [ "$JOBENV" != "-" ]; then
  # shellcheck disable=SC1090
  source "$JOBENV"
fi
[ -r "$WL" ] || { echo "ERROR: worklist $WL not readable" >&2; exit 2; }
[ -r "$TASK" ] || { echo "ERROR: task script $TASK not readable" >&2; exit 2; }

# ---- node rank ------------------------------------------------------------
# $PBS_NODEFILE is readable on the head node only; the PBS script copies it
# to a shared path and exports POL_HOSTFILE in the jobenv.
NN=1; RANK=0
HF=${POL_HOSTFILE:-${PBS_NODEFILE:-}}
if [ -n "$HF" ] && [ -r "$HF" ]; then
  mapfile -t NODES < <(awk '!seen[$0]++' "$HF")
  NN=${#NODES[@]}
  H=$(hostname); HS=${H%%.*}; RANK=-1
  for i in "${!NODES[@]}"; do
    n=${NODES[$i]}
    if [ "$n" = "$H" ] || [ "${n%%.*}" = "$HS" ]; then RANK=$i; break; fi
  done
  if [ "$RANK" -lt 0 ]; then RANK=${PMI_RANK:-${PALS_RANKID:-${PALS_NODEID:--1}}}; fi
  if [ "$RANK" -lt 0 ] || [ "$RANK" -ge "$NN" ]; then
    echo "ERROR: cannot determine this node's rank ($(hostname) not in $HF, no PMI_RANK); refusing to run (would duplicate work)" >&2
    exit 2
  fi
elif [ -n "${PMI_SIZE:-}" ] && [ "${PMI_SIZE}" -gt 1 ]; then
  echo "ERROR: multi-node launch (PMI_SIZE=$PMI_SIZE) without a readable hostfile (POL_HOSTFILE)" >&2; exit 2
fi
[ "$NN" -ge 1 ] || NN=1

# ---- my share of the worklist ---------------------------------------------
mapfile -t LINES < <(grep -v '^[[:space:]]*#' "$WL" | grep -v '^[[:space:]]*$')
MY=()
for i in "${!LINES[@]}"; do
  [ $((i % NN)) -eq "$RANK" ] && MY+=("${LINES[$i]}")
done
echo ">>> node $(hostname) rank $RANK/$NN: ${#MY[@]} of ${#LINES[@]} tasks, ppn=$PPN, task=$TASK $*"
[ ${#MY[@]} -gt 0 ] || exit 0

# ---- slot pool --------------------------------------------------------------
declare -a SLOT_PID SLOT_DESC
for ((s = 0; s < PPN; s++)); do SLOT_PID[$s]=""; SLOT_DESC[$s]=""; done
ok=0; fail=0
# reap finished slots; block until at least one slot is free
reap() {
  local s pid rc freed
  while :; do
    freed=0
    for ((s = 0; s < PPN; s++)); do
      pid=${SLOT_PID[$s]}
      if [ -z "$pid" ]; then freed=1; continue; fi
      if ! kill -0 "$pid" 2>/dev/null; then
        wait "$pid"; rc=$?
        if [ $rc -eq 0 ]; then ok=$((ok + 1)); echo "    done  slot $s: ${SLOT_DESC[$s]}"
        else fail=$((fail + 1)); echo "!!! FAILED rc=$rc slot $s: ${SLOT_DESC[$s]}"; fi
        SLOT_PID[$s]=""; SLOT_DESC[$s]=""; freed=1
      fi
    done
    [ $freed -eq 1 ] && return 0
    sleep 10
  done
}
busy() { local s; for ((s = 0; s < PPN; s++)); do [ -n "${SLOT_PID[$s]}" ] && return 0; done; return 1; }

for line in "${MY[@]}"; do
  reap
  for ((s = 0; s < PPN; s++)); do [ -z "${SLOT_PID[$s]}" ] && break; done
  echo "    start slot $s: $line"
  # shellcheck disable=SC2086
  SLOT=$s bash "$TASK" "$@" $line &
  SLOT_PID[$s]=$!; SLOT_DESC[$s]="$line"
done
while busy; do reap; done
echo ">>> NODE $(hostname) rank $RANK: ok=$ok failed=$fail"
[ $fail -eq 0 ]
