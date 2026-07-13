#!/bin/bash
# Launch one Phase 0.5 run: snapshot the config, append a registry row, and
# sbatch submit_p05_isambard.sh with a run-derived job name (WP8-lite).
#
# Usage (from anywhere; paths resolved against the repo):
#   ./launch_p05_run.sh configs/lartpc/p05/pretrain-sonata-p05b1-mc-noghost-freerot.py
#
# Wave A order (implementation plan §3/§4 — pilots FIRST):
#   pilots : supervised-ceiling-p05a1-mc-noghost.py
#            pretrain-sonata-p05b1-mc-noghost-freerot.py
#   then   : p05a2, p05b2, p05c1, p05c3      (protected core)
#   then   : p05c4, p05c5, p05c6, p05a3
#   later  : p05b4; p05b3 needs WP7 (wire projections) first.

set -eu

WORKDIR=/projects/u6jo/work/pointcept
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

CONFIG_ARG=${1:?usage: ./launch_p05_run.sh <config.py>}
CONFIG=$(cd "$WORKDIR" && readlink -f "$CONFIG_ARG")
[ -f "$CONFIG" ] || { echo "ERROR: config not found: $CONFIG_ARG"; exit 1; }

SAVE_REL=$(grep -m1 '^save_path' "$CONFIG" | sed 's/.*=\s*"\(.*\)".*/\1/')
RUN_ID=$(basename "$SAVE_REL")
TAG=$(echo "$RUN_ID" | tr '.' '_' | tr -cd 'A-Za-z0-9_-')

REGISTRY_DIR=${WORKDIR}/exp
REGISTRY=${REGISTRY_DIR}/registry.csv
CONFIG_STORE=${REGISTRY_DIR}/configs
# Job logs go to exp/logs (absolute path in submit_p05_isambard.sh #SBATCH
# directives — the local logs/ dir is owned by another user and not writable,
# which makes SLURM kill jobs at startup with exit code 53).
mkdir -p "$CONFIG_STORE" "${REGISTRY_DIR}/logs"

# Exact config snapshot + hash recorded before submission (plan §7 rule 2)
CONFIG_HASH=$(sha256sum "$CONFIG" | cut -c1-12)
cp "$CONFIG" "${CONFIG_STORE}/${RUN_ID}.py"

cd "$SCRIPT_DIR"
JOBID=$(sbatch --parsable --job-name="$TAG" submit_p05_isambard.sh "$CONFIG")

[ -f "$REGISTRY" ] || echo "run_id,config_path,config_hash,slurm_jobid,status,submit_time,save_path,notes" > "$REGISTRY"
echo "${RUN_ID},${CONFIG},${CONFIG_HASH},${JOBID},queued,$(date -Iseconds),${SAVE_REL}," >> "$REGISTRY"

echo "submitted ${RUN_ID}: job ${JOBID} (config hash ${CONFIG_HASH})"
echo "registry: ${REGISTRY}"
