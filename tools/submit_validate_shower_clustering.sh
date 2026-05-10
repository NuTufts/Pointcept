#!/bin/bash
#
# submit_validate_shower_clustering.sh
#
# SLURM array template that parallelizes shower-clustering data validation
# across N partitions. Each task runs validate_shower_clustering_data.py
# with --shard ${SLURM_ARRAY_TASK_ID}/N and writes its own pass/fail lists.
# Merge per-shard outputs at the end with concat_validate_outputs.sh.
#
# Edit DATA_DIR / OUTPUT_DIR / N_SHARDS below, then submit:
#   sbatch submit_validate_shower_clustering.sh
#
# Cost guidance (measured on real merged H5s):
#   --functional ≈ 0.75 s/file
#   ÷ N_SHARDS to estimate per-task wall time.
#
# Example: 100k files at 0.75 s/file = 75 000 s serial; with N_SHARDS=100
# you get ~750 s = ~13 min per task running concurrently.
#
#SBATCH --job-name=validate_shower_clustering
#SBATCH --output=logs/validate_%A_%a.out
#SBATCH --error=logs/validate_%A_%a.err
#SBATCH --array=0-99
#SBATCH --time=2:00:00
#SBATCH --mem-per-cpu=8000
#SBATCH --cpus-per-task=2
#SBATCH --partition=batch

set -euo pipefail

# ---- Edit these ----------------------------------------------------------
# DATA_DIR=/cluster/tufts/wongjiradlab/larbys/data/ub_on_tufts/hdf5/v3_larmatch/bnb_nu_corsika_prod2
DATA_DIR=/cluster/tufts/wongjiradlab/larbys/data/ub_on_tufts/hdf5/v3_larmatch/bnb_nu_pi0filter_corsika/merged_h5/
OUTPUT_DIR=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/lartpc_data_prep/lantern_scripts/validation_output/bnb_nu_pi0filter_corsika
N_SHARDS=100   # MUST equal the array length (count of #SBATCH --array)
FUNCTIONAL=1   # 1 = static + functional (slow but exhaustive); 0 = static only
LM_THRESHOLD=0.25
# --------------------------------------------------------------------------

if [ "${SLURM_ARRAY_TASK_COUNT:-${N_SHARDS}}" -ne "${N_SHARDS}" ]; then
    echo "WARNING: N_SHARDS=${N_SHARDS} but SLURM array has " \
         "${SLURM_ARRAY_TASK_COUNT:-?} tasks — partitions won't line up." >&2
fi

# BASH_SOURCE under slurmd points into the (read-only) job staging dir at
# /var/spool/slurm/jobNNNN/, so the script-as-staged is not a usable
# anchor for relative paths. Resolve the source repo via SLURM_SUBMIT_DIR
# (where the user ran `sbatch ...`) — that's where #SBATCH --output=logs/...
# also resolves against, so a sibling `logs/` there is what SLURM uses.
SCRIPT_DIR=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
TOOLS_DIR=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/tools
mkdir -p "${OUTPUT_DIR}"
# `logs/` next to the submission dir is the user's responsibility (standard
# SLURM convention) so we don't try to create it here.

SHARD=${SLURM_ARRAY_TASK_ID:-0}
PASS_LIST="${OUTPUT_DIR}/passes_shard$(printf '%04d' ${SHARD}).txt"
FAIL_LIST="${OUTPUT_DIR}/fails_shard$(printf '%04d' ${SHARD}).txt"

EXTRA=()
if [ "${FUNCTIONAL}" = "1" ]; then
    EXTRA+=("--functional")
fi

echo "Shard ${SHARD}/${N_SHARDS} starting at $(date)"

bash "${TOOLS_DIR}/validate_shower_clustering_data.sh" \
    "${DATA_DIR}" \
    --shard "${SHARD}/${N_SHARDS}" \
    --lm-threshold "${LM_THRESHOLD}" \
    --pass-list "${PASS_LIST}" \
    --fail-list "${FAIL_LIST}" \
    --quiet \
    "${EXTRA[@]}"

echo "Shard ${SHARD}/${N_SHARDS} finished at $(date)"
