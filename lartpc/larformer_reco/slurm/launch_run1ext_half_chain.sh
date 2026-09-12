#!/bin/bash
#SBATCH --job-name=ext1_half_launch
#SBATCH --partition=batch,wongjiradlab --time=24:00:00 --mem=4G --cpus-per-task=1
#SBATCH --output=logs/export/extbnb_run1_half_launch.%j.log --error=logs/export/extbnb_run1_half_launch.%j.log
# Launch the reco chain over the run-1 EXT HALF-STRIDE-2 sample (tranches A+B,
# filenos 1-6900 of mcc9_v29e_dl_run1_C1_extbnb_stride2.txt = 25% of the full
# C1 sample, 5,772,737 spills) as ONE production under a separate DATADIR:
# the orchestrator's prep step rm -rf's the DATADIR's kp2/nu_reco dirs, so the
# tranche-A production in run1_C1_extbnb/ must not be used as DATADIR.
# Submit with --dependency=afterok:<sequencer job> or run by hand after
# conversion. Idempotent: the list is rebuilt each time.
set -eu
K=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
C=/cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif
cd $K
SRC=/cluster/tufts/wongjiradlab/larbys/data/larformer/run1_C1_extbnb
DATADIR=/cluster/tufts/wongjiradlab/larbys/data/larformer/run1_C1_extbnb_half
MAXFILENO=${MAXFILENO:-6900}
NMARK=$(ls $SRC/markers/*.ok | wc -l)
echo ">>> markers: $NMARK (expect ~$MAXFILENO minus duds)"
[ "$NMARK" -ge $((MAXFILENO - 40)) ] || { echo "ERROR: conversion incomplete ($NMARK markers)"; exit 2; }
if squeue -u twongj01 -h -o '%j' | grep -q "ovl_train_conv\|ext1_seqB"; then echo "ERROR: conversion jobs still running"; exit 2; fi
mkdir -p $DATADIR
module load apptainer 2>/dev/null || true
apptainer exec --bind /cluster:/cluster $C python3 lartpc/data_prep/uboone_official/list_merged_sp.py \
  --dir $SRC/merged_sp --out $DATADIR/merged_sp_all.txt
python3 - "$DATADIR/merged_sp_all.txt" "$DATADIR/merged_sp_run1ext_half.txt" "$MAXFILENO" <<'PY'
import re, sys
src, dst, mx = sys.argv[1], sys.argv[2], int(sys.argv[3])
keep = [l for l in open(src) if (m := re.search(r"fileno(\d+)_entry", l)) and int(m.group(1)) <= mx]
open(dst, "w").write("".join(keep)); print(f">>> {len(keep)} merged_sp events with fileno <= {mx} -> {dst}")
PY
TAG=extbnb_run1_half DATADIR=$DATADIR MSP_LIST_SRC=$DATADIR/merged_sp_run1ext_half.txt \
  NINF=${NINF:-16} SAMPLE_KIND=data FLASH_WINDOW=3.2,5.4 \
  bash lartpc/larformer_reco/slurm/submit_extbnb_chain.sh
echo "LAUNCHED extbnb_run1_half"
