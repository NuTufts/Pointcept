#!/bin/bash
# Backfill perevent analysis for preempted valtest tasks (CPU-only,
# --skip-inference; idempotent — already-done events are skipped).
#SBATCH --job-name=lf-valtest-backfill
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --partition=batch,preempt
#SBATCH --output=logs/lf-valtest-backfill.%j.log
#SBATCH --error=logs/lf-valtest-backfill.%j.err
set -u
CONF=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept/lartpc/larformer_analysis/slicer_eval/slurm/valtest_pi0_m2fv2ep4_ptv3deghost.conf
source "$CONF"
module load apptainer
for TID in $(seq 0 19); do
  apptainer exec --bind /cluster:/cluster "$CONTAINER" bash -c \
    "cd ${WORKDIR} && python lartpc/larformer_analysis/slicer_eval/slurm/run_valtest_per_fileno.py \
       --tag ${TAG} --rerun-lines-file ${RERUN_LINES_FILE} --manifest-csv ${MANIFEST_CSV} \
       --model-config ${MODEL_CONFIG} --model-weights ${MODEL_WEIGHTS} --model-tag ${MODEL_TAG} \
       --output-dir ${OUTPUT_DIR} --task-id ${TID} --stride ${STRIDE} \
       --gamma-beam ${GAMMA_BEAM} --gamma-cosmic ${GAMMA_COSMIC} \
       --charge-source ${CHARGE_SOURCE} --charge-share-norm ${CHARGE_SHARE_NORM} \
       --skip-inference"
done
echo "backfill done: $(find ${OUTPUT_DIR}/analysis -name 'perevent_*.h5' | wc -l) analysis files"
