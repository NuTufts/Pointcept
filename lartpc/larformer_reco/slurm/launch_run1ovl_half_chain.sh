#!/bin/bash
#SBATCH --job-name=ovl1_half_launch
#SBATCH --partition=batch,wongjiradlab --time=24:00:00 --mem=4G --cpus-per-task=1
#SBATCH --output=logs/export/bnbovl_run1_half_launch.%j.log --error=logs/export/bnbovl_run1_half_launch.%j.log
# Launch the reco chain over the run-1 BNB nu OVERLAY HALF sample (MC): the
# 4,740-file TRAINPOOL list of mcc9_v28_run1_bnboverlay converted in MC mode
# with truth sidecars into run1_bnboverlay_half/ (tranche_ovl_run1_half.spec).
# MC knobs: TRUTH_DIR (truth branches, potTree, xsecWeight), SAMPLE_KIND=mc,
# FLASH_WINDOW=3.6,5.2 (measured on the run-1 overlay pilot), GAMMA_SPEC=table
# -> cell (mc,1) = 0.8576 (gamma_eff 4.502), the RUN-1 xsec weight pickle, and
# LARPID_SAMPLE_TAG without 'run3' -> LArPID default weights (run-1 rule).
# Submit with --dependency=afterany:<sequencer job> or by hand after conversion.
# Idempotent: the list is rebuilt each time; one submission per DATADIR (lock).
set -eu
K=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
C=/cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif
cd $K
DATADIR=/cluster/tufts/wongjiradlab/larbys/data/larformer/run1_bnboverlay_half
NFILES=4740
# allowed missing = 26 known duds + the deterministic crashers dropped by
# retry_incomplete_tranche.sh (listed in <spec>.dropped) + a small margin
NDROP=$(cat lartpc/data_prep/uboone_official/tranche_ovl_run1_half.spec.dropped 2>/dev/null | wc -l)
NDUD=${NDUD:-$((26 + NDROP + 5))}
NMARK=$(ls $DATADIR/markers/*.ok 2>/dev/null | wc -l)
NTRUTH=$(ls $DATADIR/truth_sidecar/truth_fileno*.h5 2>/dev/null | wc -l)
echo ">>> markers: $NMARK (expect >= $((NFILES - NDUD))), truth sidecars: $NTRUTH"
[ "$NMARK" -ge $((NFILES - NDUD)) ] || { echo "ERROR: conversion incomplete ($NMARK markers)"; exit 2; }
[ "$NTRUTH" -ge $((NMARK - 5)) ] || { echo "ERROR: truth sidecars ($NTRUTH) lag markers ($NMARK)"; exit 2; }
if squeue -u twongj01 -h -o '%j' | grep -q "ovl_train_conv\|tier2_tranche_ovl\|ovl1_half_conv"; then echo "ERROR: conversion jobs still running"; exit 2; fi
# partial conversions (no marker, but h5s/sidecar present: mid-file converter
# crashes) must be retried or dropped first, else the sample carries files with
# a full-file POT and only part of their events
NPART=$(bash lartpc/data_prep/uboone_official/retry_incomplete_tranche.sh lartpc/data_prep/uboone_official/tranche_ovl_run1_half.spec --dry-run | grep -c " partial ")
if [ "$NPART" -gt 0 ] && [ "${ALLOW_INCOMPLETE:-0}" != 1 ]; then
  echo "ERROR: $NPART partially converted filenos; run (login node): bash lartpc/data_prep/uboone_official/retry_incomplete_tranche.sh lartpc/data_prep/uboone_official/tranche_ovl_run1_half.spec  (ALLOW_INCOMPLETE=1 to override)"; exit 2
fi
LOCK=$DATADIR/.chain_submitted
if [ -e "$LOCK" ]; then echo "ERROR: chain already submitted ($(cat $LOCK)); remove $LOCK to force"; exit 2; fi
if squeue -u twongj01 -h -o '%j' | grep -q "^bnbovl_run1_half_"; then echo "ERROR: bnbovl_run1_half jobs already queued"; exit 2; fi
echo "$(date) job ${SLURM_JOB_ID:-login}" > "$LOCK"
module load apptainer 2>/dev/null || true
apptainer exec --bind /cluster:/cluster $C python3 lartpc/data_prep/uboone_official/list_merged_sp.py \
  --dir $DATADIR/merged_sp --out $DATADIR/merged_sp_run1ovl_half.txt
echo ">>> $(wc -l < $DATADIR/merged_sp_run1ovl_half.txt) merged_sp events"
export WEIGHTS_PKL=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/gen2ntuple/event_weighting/weights_forCV_v48_Sep24_bnb_nu_run1.pkl
[ -f "$WEIGHTS_PKL" ] || { echo "ERROR: $WEIGHTS_PKL missing"; exit 2; }
TAG=bnbovl_run1_half DATADIR=$DATADIR MSP_LIST_SRC=$DATADIR/merged_sp_run1ovl_half.txt \
  TRUTH_DIR=$DATADIR/truth_sidecar LARPID_SAMPLE_TAG=bnbovl_run1_half \
  NINF=${NINF:-32} SAMPLE_KIND=mc FLASH_WINDOW=3.6,5.2 GAMMA_SPEC=table \
  bash lartpc/larformer_reco/slurm/submit_extbnb_chain.sh
echo "LAUNCHED bnbovl_run1_half"
