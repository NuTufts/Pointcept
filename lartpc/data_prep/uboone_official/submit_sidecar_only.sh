#!/bin/bash
#SBATCH --job-name=sidecar_only
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --partition=batch,preempt,wongjiradlab
#SBATCH --output=logs/overlay_train/sidecar.%A_%a.log
#SBATCH --error=logs/overlay_train/sidecar.%A_%a.err
# Truth-sidecar-ONLY pass for an MC sample whose merged_sp conversion already
# exists without sidecars (pre-2026-09 conversions, e.g. the nueintrinsics
# TRAINPOOL half). tier2 is not mounted on compute nodes: the login-side
# sequencer run_sidecar_only_tranche.sh stages $STAGE/fileno<NNNNN>.root, this
# task extracts truth_fileno<NNNNN>.h5 and deletes the staged copy.
# fileno = FILENO_BASE + array index = line number of the sample list, the SAME
# numbering the merged_sp files carry.
set -u -o pipefail
KPV2=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
: "${STAGE:?}" "${TRUTH_DIR:?}"
FILENO=$(( ${FILENO_BASE:-0} + ${SLURM_ARRAY_TASK_ID:?} ))
ZFN=$(printf "%05d" $FILENO)
LOCAL=$STAGE/fileno${ZFN}.root; OUT=$TRUTH_DIR/truth_fileno${ZFN}.h5
mkdir -p "$TRUTH_DIR"
if [ -s "$OUT" ]; then rm -f "$LOCAL"; echo "exists, skip fileno $ZFN"; exit 0; fi
[ -f "$LOCAL" ] || { echo "ERROR: staged file missing: $LOCAL (login-side staging required)"; exit 1; }
module load apptainer 2>/dev/null || true
apptainer exec --bind /cluster:/cluster \
  /cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif bash -c "
  cd /cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/ubdl && \
  source setenv_pointcept_container.sh >/dev/null 2>&1 && cd $KPV2 && \
  python3 lartpc/larformer_reco/export/extract_truth_sidecar.py \
    --input-dlmerged $LOCAL --out $OUT.tmp && mv $OUT.tmp $OUT" | tail -2
rm -f "$LOCAL" "$OUT.tmp"
[ -s "$OUT" ] && { echo "fileno $ZFN: sidecar ok"; exit 0; }
echo "fileno $ZFN: sidecar FAILED"; exit 1
