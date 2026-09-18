#!/bin/bash
#SBATCH --job-name=nue1_trainhalf_launch
#SBATCH --partition=batch,wongjiradlab --time=24:00:00 --mem=4G --cpus-per-task=1
#SBATCH --output=logs/export/nue_run1_trainhalf_launch.%j.log --error=logs/export/nue_run1_trainhalf_launch.%j.log
# Launch the reco chain over the run-1 INTRINSIC-NUE overlay TRAINPOOL half
# (1,986 files of mcc9_v28_run1_nueintrinsics; for TRAINING the nue-CC BDTs --
# the RESERVED half, run1_nueintrinsics_half, stays the independent analysis
# sample). The merged_sp conversion already exists
# (overlay_train/mcc9_v28_run1_nueintrinsics/merged_sp, Aug/Sep 2026, same
# content as the new conversions, labels completed); only the truth sidecars
# were missing and were produced by the sidecar-only pass
# (uboone_official/run_sidecar_only_tranche.sh) into $DATADIR/truth_sidecar.
# Excluded: the ledger's CRASH_PARTIAL filenos (partial events + full-file POT)
# and any fileno without a sidecar. Same MC knobs as launch_run1nue_half_chain.sh.
set -eu
K=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
C=/cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif
cd $K
MSP=/cluster/tufts/wongjiradlab/larbys/data/larformer/overlay_train/mcc9_v28_run1_nueintrinsics/merged_sp
DATADIR=/cluster/tufts/wongjiradlab/larbys/data/larformer/run1_nueintrinsics_trainhalf
PARTIAL=lartpc/data_prep/uboone_official/training_data_ledger/mcc9_v28_run1_nueintrinsics_CRASH_PARTIAL_filenos.txt
NTRUTH=$(ls $DATADIR/truth_sidecar/truth_fileno*.h5 2>/dev/null | wc -l)
echo ">>> truth sidecars: $NTRUTH (expect >= 1966 of 1986)"
[ "$NTRUTH" -ge 1966 ] || { echo "ERROR: sidecar pass incomplete ($NTRUTH)"; exit 2; }
if squeue -u twongj01 -h -o '%j' | grep -q "^sidecar_only"; then echo "ERROR: sidecar jobs still running"; exit 2; fi
LOCK=$DATADIR/.chain_submitted
if [ -e "$LOCK" ]; then echo "ERROR: chain already submitted ($(cat $LOCK)); remove $LOCK to force"; exit 2; fi
if squeue -u twongj01 -h -o '%j' | grep -q "^nue_run1_trainhalf_"; then echo "ERROR: nue_run1_trainhalf jobs already queued"; exit 2; fi
echo "$(date) job ${SLURM_JOB_ID:-login}" > "$LOCK"
module load apptainer 2>/dev/null || true
apptainer exec --bind /cluster:/cluster $C python3 lartpc/data_prep/uboone_official/list_merged_sp.py \
  --dir $MSP --out $DATADIR/merged_sp_all.txt
python3 - "$DATADIR/merged_sp_all.txt" "$DATADIR/merged_sp_run1nue_trainhalf.txt" "$PARTIAL" "$DATADIR/truth_sidecar" <<'PY'
import os, re, sys
src, dst, partial, tdir = sys.argv[1:5]
bad = {int(x) for x in open(partial).read().split() if x.strip().isdigit()}
keep, nbad, nnot = [], 0, 0
for l in open(src):
    m = re.search(r"fileno(\d+)_entry", l)
    if not m: continue
    f = int(m.group(1))
    if f in bad: nbad += 1; continue
    if not os.path.exists(os.path.join(tdir, "truth_fileno%05d.h5" % f)): nnot += 1; continue
    keep.append(l)
open(dst, "w").write("".join(keep))
print(f">>> {len(keep)} events kept; dropped {nbad} (CRASH_PARTIAL filenos) + {nnot} (no sidecar) -> {dst}")
PY
export WEIGHTS_PKL=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/gen2ntuple/event_weighting/weights_forCV_v48_Sep24_intrinsic_nue_run1.pkl
[ -f "$WEIGHTS_PKL" ] || { echo "ERROR: $WEIGHTS_PKL missing"; exit 2; }
TAG=nue_run1_trainhalf DATADIR=$DATADIR MSP_LIST_SRC=$DATADIR/merged_sp_run1nue_trainhalf.txt \
  TRUTH_DIR=$DATADIR/truth_sidecar LARPID_SAMPLE_TAG=nue_run1_trainhalf \
  NINF=${NINF:-16} SAMPLE_KIND=mc FLASH_WINDOW=3.6,5.2 GAMMA_SPEC=table \
  bash lartpc/larformer_reco/slurm/submit_extbnb_chain.sh
echo "LAUNCHED nue_run1_trainhalf"
