#!/bin/bash
#SBATCH --job-name=bnb5e19_table_launch
#SBATCH --partition=batch,wongjiradlab --time=24:00:00 --mem=4G --cpus-per-task=1
#SBATCH --output=logs/export/bnb5e19_run1_table_launch.%j.log --error=logs/export/bnb5e19_run1_table_launch.%j.log
# Reprocess the FULL bnb5e19 beam-on run-1 sample (176,302 events, the
# production membership of bnb5e19_production_basenames.txt.gz) at Tufts with
# the cew6 chain + the CALIBRATED table gamma (cell (data,1) = 0.5449 ->
# gamma_eff 2.861), beam-on window 2.8-5.0 us, data mode, LArPID default
# weights. Replaces the legacy-gamma cew6 production
# (ub_on_tufts/larformer_bnb5e19_s1ep2p8cew6, gamma_eff 4.20). Fresh DATADIR:
# the orchestrator rm -rf's kp2/nu_reco dirs of the DATADIR it is given.
# 2026-09-15: chosen over the Polaris run (platform conformance unresolved).
set -eu
K=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
C=/cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif
cd $K
MSP=/cluster/tufts/wongjiradlab/larbys/data/larformer/mcc9_v28_wctagger_bnb5e19/merged_sp
DATADIR=/cluster/tufts/wongjiradlab/larbys/data/larformer/bnb5e19_run1_table
mkdir -p $DATADIR
LOCK=$DATADIR/.chain_submitted
if [ -e "$LOCK" ]; then echo "ERROR: chain already submitted ($(cat $LOCK)); remove $LOCK to force"; exit 2; fi
if squeue -u twongj01 -h -o '%j' | grep -q "^bnb5e19_run1_table_"; then echo "ERROR: bnb5e19_run1_table jobs already queued"; exit 2; fi
echo "$(date) job ${SLURM_JOB_ID:-login}" > "$LOCK"
module load apptainer 2>/dev/null || true
# production membership + order (same list the Polaris scripts use; 34 of the
# 176,336 merged_sp files are excluded)
apptainer exec --bind /cluster:/cluster $C python3 lartpc/polaris/make_polaris_list.py \
  --merged-sp $MSP --out $DATADIR/merged_sp_bnb5e19_production.txt
N=$(wc -l < $DATADIR/merged_sp_bnb5e19_production.txt); echo ">>> $N merged_sp events"
[ "$N" -eq 176302 ] || { echo "ERROR: expected 176302 events"; exit 2; }
export WEIGHTS_PKL=none      # data: no xsec weights (exporter accepts 'none')
TAG=bnb5e19_run1_table DATADIR=$DATADIR MSP_LIST_SRC=$DATADIR/merged_sp_bnb5e19_production.txt \
  LARPID_SAMPLE_TAG=bnb5e19_run1_table \
  NINF=${NINF:-32} SAMPLE_KIND=data FLASH_WINDOW=2.8,5.0 GAMMA_SPEC=table \
  bash lartpc/larformer_reco/slurm/submit_extbnb_chain.sh
echo "LAUNCHED bnb5e19_run1_table"
