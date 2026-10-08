#!/bin/bash
#SBATCH --job-name=tranche_chain_launch
#SBATCH --partition=batch,wongjiradlab --time=24:00:00 --mem=4G --cpus-per-task=1
#SBATCH --output=logs/export/tranche_chain_launch.%j.log --error=logs/export/tranche_chain_launch.%j.log
# GENERIC launcher: reco chain over a sample converted by a tranche spec
# (run_tier2_tranche.sh / tmux_tier2_tranche.sh). Generalizes the per-sample
# launch_run1*_chain.sh scripts (same guards); new samples should use this
# instead of copying one of those. See docs/reference/Sample_Processing_Procedure.md.
#
#   SPEC=lartpc/data_prep/uboone_official/tranche_<x>.spec TAG=<tag> \
#   GAMMA_SPEC=table|<float> FLASH_WINDOW=lo,hi [WEIGHTS_PKL=<pkl>|none] \
#   [LARPID_SAMPLE_TAG=..] [NINF=16] [NDUD=..] [ALLOW_INCOMPLETE=1] \
#     sbatch --export=ALL lartpc/larformer_reco/slurm/launch_tranche_chain.sh
#
# DATADIR = parent of the spec's OUT_ROOT (merged_sp/). MODE (mc|data) and
# TRUTH_DIR come from the spec. GAMMA_SPEC and FLASH_WINDOW have NO default on
# purpose: both are baked into the GPU pass (see flash_calib.py).
# Exit 2 = a guard refused (nothing submitted).
set -eu
K=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
C=/cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif
cd $K
SPEC=${SPEC:?set SPEC (tranche spec file)}
TAG=${TAG:?set TAG (ntuple / job-name tag)}
: "${GAMMA_SPEC:?set GAMMA_SPEC (table or an explicit float)}"
: "${FLASH_WINDOW:?set FLASH_WINDOW (lo,hi us)}"
# first spec line gives the sample-wide fields; NFILES sums every line's range
read -r _S _L OUT_ROOT MODE TRUTH_DIR _ST MARK _F _LA < <(grep -vE '^\s*(#|$)' "$SPEC" | head -1)
NFILES=$(grep -vE '^\s*(#|$)' "$SPEC" | awk '{n += $9 - $8 + 1} END {print n}')
DATADIR=$(dirname "$OUT_ROOT")
NDROP=$(cat "${SPEC}.dropped" 2>/dev/null | wc -l)
NDUD=${NDUD:-$((NDROP + 5))}
NMARK=$(ls $MARK/*.ok 2>/dev/null | wc -l)
echo ">>> $TAG: DATADIR=$DATADIR mode=$MODE files=$NFILES markers=$NMARK (expect >= $((NFILES - NDUD)))"
[ "$NMARK" -ge $((NFILES - NDUD)) ] || { echo "ERROR: conversion incomplete ($NMARK markers)"; exit 2; }
if [ "$MODE" = mc ]; then
  NTRUTH=$(ls $TRUTH_DIR/truth_fileno*.h5 2>/dev/null | wc -l)
  echo ">>> truth sidecars: $NTRUTH"
  [ "$NTRUTH" -ge $((NMARK - 5)) ] || { echo "ERROR: truth sidecars ($NTRUTH) lag markers ($NMARK)"; exit 2; }
  : "${WEIGHTS_PKL:?MC sample: set WEIGHTS_PKL (xsec pickle, or none for a particle gun)}"
  export WEIGHTS_PKL
  [ "$WEIGHTS_PKL" = none ] || [ -f "$WEIGHTS_PKL" ] || { echo "ERROR: $WEIGHTS_PKL missing"; exit 2; }
  KIND=mc; TRUTH_ARG=$TRUTH_DIR
else
  export WEIGHTS_PKL=none; KIND=data; TRUTH_ARG=""
fi
if squeue -u $USER -h -o '%j' | grep -q "ovl_train_conv"; then echo "ERROR: conversion jobs still running"; exit 2; fi
# partial conversions (no marker, but h5s present) carry a full-file POT with
# only part of their events: retry or drop them first
NPART=$(bash lartpc/data_prep/uboone_official/retry_incomplete_tranche.sh "$SPEC" --dry-run | grep -cE "^ +[0-9]+ (partial|dudpartial) nh5=[1-9]" || true)
if [ "$NPART" -gt 0 ] && [ "${ALLOW_INCOMPLETE:-0}" != 1 ]; then
  echo "ERROR: $NPART partially converted filenos; run (login node): bash lartpc/data_prep/uboone_official/retry_incomplete_tranche.sh $SPEC  (ALLOW_INCOMPLETE=1 to override)"; exit 2
fi
LOCK=$DATADIR/.chain_submitted
if [ -e "$LOCK" ]; then echo "ERROR: chain already submitted ($(cat $LOCK)); remove $LOCK to force"; exit 2; fi
if squeue -u $USER -h -o '%j' | grep -q "^${TAG}_"; then echo "ERROR: ${TAG}_* jobs already queued"; exit 2; fi
echo "$(date) job ${SLURM_JOB_ID:-login}" > "$LOCK"
module load apptainer 2>/dev/null || true
apptainer exec --bind /cluster:/cluster $C python3 lartpc/data_prep/uboone_official/list_merged_sp.py \
  --dir $DATADIR/merged_sp --out $DATADIR/merged_sp_${TAG}.txt
echo ">>> $(wc -l < $DATADIR/merged_sp_${TAG}.txt) merged_sp events"
TAG=$TAG DATADIR=$DATADIR MSP_LIST_SRC=$DATADIR/merged_sp_${TAG}.txt \
  TRUTH_DIR=$TRUTH_ARG LARPID_SAMPLE_TAG=${LARPID_SAMPLE_TAG:-$TAG} \
  NINF=${NINF:-16} SAMPLE_KIND=$KIND FLASH_WINDOW=$FLASH_WINDOW GAMMA_SPEC=$GAMMA_SPEC \
  bash lartpc/larformer_reco/slurm/submit_extbnb_chain.sh
echo "LAUNCHED $TAG"
