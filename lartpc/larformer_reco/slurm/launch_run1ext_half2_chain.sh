#!/bin/bash
#SBATCH --job-name=ext1_half2_launch
#SBATCH --partition=batch,wongjiradlab --time=24:00:00 --mem=4G --cpus-per-task=1
#SBATCH --output=logs/export/extbnb_run1_half2_launch.%j.log --error=logs/export/extbnb_run1_half2_launch.%j.log
# Launch the reco chain over the SECOND run-1 EXT stride-2 half (tranche C,
# filenos 6901-13801 of mcc9_v29e_dl_run1_C1_extbnb_stride2.txt = another 25% of
# the full C1 sample, 6,901/27,602 x 23,090,946 = 5,773,573 spills by the
# file-fraction rule) as its OWN production, disjoint from run1_C1_extbnb_half
# (filenos <= 6900): tune on one half, predict with the other, or hadd both for
# 50% of C1. Separate DATADIR because the orchestrator's prep rm -rf's the
# DATADIR's kp2/nu_reco dirs. Data mode, FLASH_WINDOW=3.2,5.4, table gamma
# (data,1), LArPID default weights, WEIGHTS_PKL=none.
set -eu
K=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
C=/cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif
cd $K
SRC=/cluster/tufts/wongjiradlab/larbys/data/larformer/run1_C1_extbnb
DATADIR=/cluster/tufts/wongjiradlab/larbys/data/larformer/run1_C1_extbnb_half2
MINFILENO=${MINFILENO:-6901}; MAXFILENO=${MAXFILENO:-13801}
NMARK=$(ls $SRC/markers/*.ok | wc -l)
echo ">>> markers: $NMARK (expect ~$MAXFILENO minus duds/dropped)"
[ "$NMARK" -ge $((MAXFILENO - 120)) ] || { echo "ERROR: conversion incomplete ($NMARK markers)"; exit 2; }
if squeue -u twongj01 -h -o '%j' | grep -q "ovl_train_conv\|ext1_seqB"; then echo "ERROR: conversion jobs still running"; exit 2; fi
NPART=$(bash lartpc/data_prep/uboone_official/retry_incomplete_tranche.sh lartpc/data_prep/uboone_official/tranche_ext_run1_C.spec --dry-run | grep -cE "^ +[0-9]+ (partial|dudpartial) " || true)
if [ "$NPART" -gt 0 ] && [ "${ALLOW_INCOMPLETE:-0}" != 1 ]; then echo "ERROR: $NPART partially converted filenos; run retry_incomplete_tranche.sh on tranche_ext_run1_C.spec (login node)"; exit 2; fi
# one submission only: a requeued/relaunched copy of this job must not submit a
# second chain into the same DATADIR (two chains raced on 2026-09-12)
LOCK=$DATADIR/.chain_submitted
if [ -e "$LOCK" ]; then echo "ERROR: chain already submitted ($(cat $LOCK)); remove $LOCK to force"; exit 2; fi
if squeue -u twongj01 -h -o '%j' | grep -q "^extbnb_run1_half2_"; then echo "ERROR: extbnb_run1_half jobs already queued"; exit 2; fi
mkdir -p $DATADIR && echo "$(date) job ${SLURM_JOB_ID:-login}" > "$LOCK"
mkdir -p $DATADIR
module load apptainer 2>/dev/null || true
apptainer exec --bind /cluster:/cluster $C python3 lartpc/data_prep/uboone_official/list_merged_sp.py \
  --dir $SRC/merged_sp --out $DATADIR/merged_sp_all.txt
python3 - "$DATADIR/merged_sp_all.txt" "$DATADIR/merged_sp_run1ext_half2.txt" "$MINFILENO" "$MAXFILENO" <<'PY'
import re, sys
src, dst, mn, mx = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
keep = [l for l in open(src) if (m := re.search(r"fileno(\d+)_entry", l)) and mn <= int(m.group(1)) <= mx]
open(dst, "w").write("".join(keep)); print(f">>> {len(keep)} merged_sp events with {mn} <= fileno <= {mx} -> {dst}")
PY
export WEIGHTS_PKL=none
TAG=extbnb_run1_half2 DATADIR=$DATADIR MSP_LIST_SRC=$DATADIR/merged_sp_run1ext_half2.txt \
  NINF=${NINF:-24} SAMPLE_KIND=data FLASH_WINDOW=3.2,5.4 \
  bash lartpc/larformer_reco/slurm/submit_extbnb_chain.sh
echo "LAUNCHED extbnb_run1_half2"
