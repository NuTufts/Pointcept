#!/bin/bash
# Run run_tier2_tranche.sh (serial tier2-staging conversion sequencer) in a
# detached tmux session ON THE LOGIN NODE. It cannot be a slurm job: /cluster/tier2
# is mounted on login nodes only (verified 2026-09-13: absent on batch nodes AND
# on the wongjiradlab contrib node), and the stager copies from tier2.
#
#   bash lartpc/data_prep/uboone_official/tmux_tier2_tranche.sh <specfile> <name> [<command to run after ALL BATCHES DONE>]
# Log: logs/data_prep/<name>.log      attach: tmux attach -t <name>     list: tmux ls
# Idempotent: the sequencer resumes from markers/h5 if restarted.
set -eu
SPEC=${1:?specfile}; NAME=${2:?tmux session / log name}; AFTER=${3:-}
K=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
cd "$K"; [ -s "$SPEC" ] || { echo "ERROR: $SPEC missing/empty" >&2; exit 2; }
[ -d /cluster/tier2/wongjiradlab ] || { echo "ERROR: /cluster/tier2 not visible here; run on a login node" >&2; exit 2; }
if tmux has-session -t "$NAME" 2>/dev/null; then echo "ERROR: tmux session $NAME exists (tmux attach -t $NAME)" >&2; exit 2; fi
mkdir -p logs/data_prep
LOG=logs/data_prep/$NAME.log
CMD="cd $K && echo \"=== sequencer started \$(date) on \$(hostname) spec=$SPEC ===\" >> $LOG && bash lartpc/data_prep/uboone_official/run_tier2_tranche.sh $SPEC >> $LOG 2>&1"
if [ -n "$AFTER" ]; then CMD="$CMD && if grep -q 'ALL BATCHES DONE' $LOG; then echo \"=== after: $AFTER\" >> $LOG; $AFTER >> $LOG 2>&1; fi"; fi
tmux new-session -d -s "$NAME" "bash -c '$CMD'"
echo "started tmux session $NAME on $(hostname); log $LOG"
