#!/bin/bash
# Diagnose the export slowdown under concurrency (run inside an interactive
# job: qsub -I -A neutrinoGPU::debug -q debug -l select=1:system=polaris \
#   -l walltime=00:45:00 -l filesystems=home:eagle -l place=scatter).
#
#   cd /eagle/neutrinoGPU/twongj01/larformer_pointcept
#   bash lartpc/polaris/diag_export.sh [TAG=throughput] [NEV=20] [NCONC=4]
#
# Phase 1: ONE exporter on NEV events (baseline s/event).
# Phase 2: NCONC exporters at once on disjoint ranges, while sampling `top`
#          every 10 s for squashfuse_ll / python CPU%. If squashfuse_ll sits
#          at ~100% CPU while the pythons idle, the FUSE squashfs layer is the
#          serialization point; if the pythons are at 100% each, it is CPU.
# Phase 3: cProfile of one exporter under that concurrency (top 25 by own time).
# Output: $POL_LOGDIR/diag/ (summary printed at the end).
set -u
cd "$(dirname "$0")/../.."
# shellcheck disable=SC1091
source lartpc/polaris/polaris_env.sh
TAG=${TAG:-throughput}; NEV=${NEV:-20}; NCONC=${NCONC:-4}
T=$POL_DATADIR/tests/$TAG/tail; O=$POL_LOGDIR/diag; mkdir -p "$O"
MSP=$(ls "$T"/lists/merged_sp_${TAG}_head*.txt | head -1)
ARGS="--merged-sp-list $MSP --truth-dir $T/truth_sidecar_absent --kp2-nu-list $T/lists/keypoint2_out_${TAG}_nu.txt --kp2-fm-list $T/lists/keypoint2_out_${TAG}_fm.txt --nu-reco-nu-dir $T/nu_reco_larpid_nu --nu-reco-fm-dir $T/nu_reco_larpid_fm --weights-pkl none"
run_one () {  # run_one <start> <label>
  local t0=$(date +%s)
  pol_exec "python3 -u lartpc/larformer_reco/export/export_gen2ntuple.py $ARGS --start $1 --n $NEV --out $O/diag_$2.root" > "$O/export_$2.log" 2>&1
  echo "$2: $(( $(date +%s) - t0 )) s for $NEV events = $(( ($(date +%s) - t0) / NEV )) s/event"
}
echo "=== host $(hostname); env: OMP=$OMP_NUM_THREADS OPENBLAS=$OPENBLAS_NUM_THREADS"
echo "=== squashfuse processes now:"; pgrep -a squashfuse | head -3
echo; echo "=== phase 1: single exporter, $NEV events"
run_one 0 single | tee "$O/summary.txt"

echo; echo "=== phase 2: $NCONC concurrent exporters, $NEV events each (top sampled every 10 s)"
: > "$O/top.txt"
( while :; do date +%T >> "$O/top.txt"; top -b -n1 -o %CPU | grep -E "squashfuse|python3" | head -12 >> "$O/top.txt"; sleep 10; done ) & TOPPID=$!
t0=$(date +%s)
for k in $(seq 1 "$NCONC"); do run_one $((k * NEV * 5)) conc$k & done; wait
kill $TOPPID 2>/dev/null
echo "phase 2 wall: $(( $(date +%s) - t0 )) s for $NCONC x $NEV events" | tee -a "$O/summary.txt"
echo "--- top samples (max CPU% per process name):"; awk '/squashfuse|python3/{print $12, $9}' "$O/top.txt" | sort | awk '{if($2>m[$1])m[$1]=$2} END{for(k in m)print "   ",k,"max CPU%",m[k]}' | tee -a "$O/summary.txt"
echo "--- one raw top sample:"; sed -n 1,10p "$O/top.txt"

echo; echo "=== phase 3: cProfile of one exporter under $NCONC-way concurrency"
for k in $(seq 2 "$NCONC"); do run_one $((k * NEV * 7)) bg$k > /dev/null & done
pol_exec "python3 -c \"
import cProfile, pstats, sys, io, time
sys.argv=['x'] + '$ARGS'.split() + ['--start','7','--n','$NEV','--out','$O/diag_prof.root']
sys.path.insert(0,'lartpc/larformer_reco/export'); import export_gen2ntuple as E
t0=time.time(); pr=cProfile.Profile(); pr.enable(); E.main(); pr.disable(); print('profiled wall %.1f s for $NEV events'%(time.time()-t0))
s=io.StringIO(); pstats.Stats(pr,stream=s).sort_stats('tottime').print_stats(25); print(s.getvalue()[:5000])
\"" 2>&1 | grep -v "truth sidecar" | tee "$O/profile.txt" | tail -40
wait
echo; echo "=== summary ($O/summary.txt):"; cat "$O/summary.txt"
