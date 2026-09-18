#!/bin/bash
#SBATCH --job-name=nue1_half_launch
#SBATCH --partition=batch,wongjiradlab --time=24:00:00 --mem=4G --cpus-per-task=1
#SBATCH --output=logs/export/nue_run1_half_launch.%j.log --error=logs/export/nue_run1_half_launch.%j.log
# Launch the reco chain over the run-1 INTRINSIC-NUE overlay HALF sample (MC):
# the 2,014-file RESERVED half of mcc9_v28_run1_nueintrinsics (unbiased 50% by
# md5 parity; the TRAINPOOL half is training data) converted in MC mode with
# truth sidecars into run1_nueintrinsics_half/ (tranche_nue_run1_half.spec).
# MC knobs: TRUTH_DIR, SAMPLE_KIND=mc, FLASH_WINDOW=3.6,5.2, GAMMA_SPEC=table
# -> (mc,1) = 0.8576 (measured on the BNB overlay pilot; nothing in flash_calib
# is nue-specific), the run-1 INTRINSIC-NUE xsec weight pickle, and
# LARPID_SAMPLE_TAG without 'run3' -> LArPID default weights (run-period rule).
# Marker gate: 2014 - dropped (retry_incomplete_tranche.sh) - 5. Same lock and
# partial-file guard as launch_run1ovl_half_chain.sh.
set -eu
K=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
C=/cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif
cd $K
DATADIR=/cluster/tufts/wongjiradlab/larbys/data/larformer/run1_nueintrinsics_half
NFILES=2014
# allowed missing = 26 known duds + the deterministic crashers dropped by
# retry_incomplete_tranche.sh (listed in <spec>.dropped) + a small margin
NDROP=$(cat lartpc/data_prep/uboone_official/tranche_nue_run1_half.spec.dropped 2>/dev/null | wc -l)
NDUD=${NDUD:-$((NDROP + 5))}   # no ledger dud list for nue
NMARK=$(ls $DATADIR/markers/*.ok 2>/dev/null | wc -l)
NTRUTH=$(ls $DATADIR/truth_sidecar/truth_fileno*.h5 2>/dev/null | wc -l)
echo ">>> markers: $NMARK (expect >= $((NFILES - NDUD))), truth sidecars: $NTRUTH"
[ "$NMARK" -ge $((NFILES - NDUD)) ] || { echo "ERROR: conversion incomplete ($NMARK markers)"; exit 2; }
[ "$NTRUTH" -ge $((NMARK - 5)) ] || { echo "ERROR: truth sidecars ($NTRUTH) lag markers ($NMARK)"; exit 2; }
if squeue -u twongj01 -h -o '%j' | grep -q "ovl_train_conv\|tier2_tranche_nue\|nue1_half_conv"; then echo "ERROR: conversion jobs still running"; exit 2; fi
# partial conversions (no marker, but h5s/sidecar present: mid-file converter
# crashes) must be retried or dropped first, else the sample carries files with
# a full-file POT and only part of their events
NPART=$(bash lartpc/data_prep/uboone_official/retry_incomplete_tranche.sh lartpc/data_prep/uboone_official/tranche_nue_run1_half.spec --dry-run | grep -cE "^ +[0-9]+ (partial|dudpartial) " || true)
if [ "$NPART" -gt 0 ] && [ "${ALLOW_INCOMPLETE:-0}" != 1 ]; then
  echo "ERROR: $NPART partially converted filenos; run (login node): bash lartpc/data_prep/uboone_official/retry_incomplete_tranche.sh lartpc/data_prep/uboone_official/tranche_nue_run1_half.spec  (ALLOW_INCOMPLETE=1 to override)"; exit 2
fi
LOCK=$DATADIR/.chain_submitted
if [ -e "$LOCK" ]; then echo "ERROR: chain already submitted ($(cat $LOCK)); remove $LOCK to force"; exit 2; fi
if squeue -u twongj01 -h -o '%j' | grep -q "^nue_run1_half_"; then echo "ERROR: nue_run1_half jobs already queued"; exit 2; fi
echo "$(date) job ${SLURM_JOB_ID:-login}" > "$LOCK"
module load apptainer 2>/dev/null || true
apptainer exec --bind /cluster:/cluster $C python3 lartpc/data_prep/uboone_official/list_merged_sp.py \
  --dir $DATADIR/merged_sp --out $DATADIR/merged_sp_run1nue_half.txt
echo ">>> $(wc -l < $DATADIR/merged_sp_run1nue_half.txt) merged_sp events"
export WEIGHTS_PKL=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/gen2ntuple/event_weighting/weights_forCV_v48_Sep24_intrinsic_nue_run1.pkl
[ -f "$WEIGHTS_PKL" ] || { echo "ERROR: $WEIGHTS_PKL missing"; exit 2; }
TAG=nue_run1_half DATADIR=$DATADIR MSP_LIST_SRC=$DATADIR/merged_sp_run1nue_half.txt \
  TRUTH_DIR=$DATADIR/truth_sidecar LARPID_SAMPLE_TAG=nue_run1_half \
  NINF=${NINF:-16} SAMPLE_KIND=mc FLASH_WINDOW=3.6,5.2 GAMMA_SPEC=table \
  bash lartpc/larformer_reco/slurm/submit_extbnb_chain.sh
echo "LAUNCHED nue_run1_half"
