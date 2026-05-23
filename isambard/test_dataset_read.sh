#!/bin/bash
#
# Wrapper that drives test_dataset_read.py inside pointcept_cuml.sif with
# the combined SquashFS bound at $DATA_ROOT. Runs identically as:
#   bash test_dataset_read.sh    (login node or interactive worker;
#                                 #SBATCH directives are ignored)
#   sbatch test_dataset_read.sh  (queued worker; #SBATCH honored)
#
# Defaults target the smoke-test combined.sqsh so the script is safe to
# run on the login node without burning allocation. Override via env to
# hit the full combined image.
#
# Required env (no default):
#   DATA_LIST_FILE   Path to a text file with one HDF5 path per line.
#                    Paths must resolve INSIDE the container, i.e. all
#                    paths under $DATA_ROOT (defaults to /data). Build
#                    this list by stripping the Tufts source-prefix from
#                    your training list and prepending /data/.
#
# Optional env (with defaults):
#   CONTAINER        /projects/u6jo/containers/pointcept_cuml.sif
#   COMBINED_SQSH    /projects/u6jo/datasets/combined_pretrain-sonata-v7-extbnb-larmatch-test.sqsh
#   DATA_ROOT        /data
#   POINTCEPT_DIR    $HOME/ubpointcept/pointcept (on Isambard) — host path to the cloned pointcept repo
#   CONFIG_FILE      ${POINTCEPT_DIR}/configs/lartpc/pretrain-sonata-v7-extbnb-larmatch.py
#   SPLIT            train
#   NUM_BATCHES      4
#   BATCH_SIZE       2
#   NUM_WORKERS      0
#   APPLY_TRANSFORMS 0   (set to 1 to apply the config's transform pipeline)
#   GPU              auto (set GPU=0 to skip --nv; useful on login or for CPU-only smoke test)
#
# Example (smoke test on login):
#   DATA_LIST_FILE=/projects/u6jo/datasets/test_list_isambard.txt \\
#       bash test_dataset_read.sh
#
# Example (multi-worker scaling sweep via repeated runs):
#   for W in 0 1 2 4 8; do
#       DATA_LIST_FILE=... NUM_WORKERS=$W NUM_BATCHES=16 \\
#           bash test_dataset_read.sh 2>&1 | tee scan-w${W}.log
#   done
#
#SBATCH --job-name=test-dataset-read
#SBATCH --output=logs/test-dataset-read-%j.out
#SBATCH --error=logs/test-dataset-read-%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
# TODO Isambard: --partition / --account / --qos

set -euo pipefail

# === Config ===
CONTAINER="${CONTAINER:-/projects/u6jo/containers/pointcept_cuml.sif}"
COMBINED_SQSH="${COMBINED_SQSH:-/projects/u6jo/datasets/combined_pretrain-sonata-v7-extbnb-larmatch-test.sqsh}"
DATA_ROOT="${DATA_ROOT:-/data}"
DATA_LIST_FILE="${DATA_LIST_FILE:?Set DATA_LIST_FILE to an Isambard-side list; paths must start with \$DATA_ROOT (= $DATA_ROOT).}"
POINTCEPT_DIR="${POINTCEPT_DIR:-$HOME/ubpointcept/pointcept}"
CONFIG_FILE="${CONFIG_FILE:-${POINTCEPT_DIR}/configs/lartpc/pretrain-sonata-v7-extbnb-larmatch.py}"
SPLIT="${SPLIT:-train}"
NUM_BATCHES="${NUM_BATCHES:-4}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-0}"
APPLY_TRANSFORMS="${APPLY_TRANSFORMS:-0}"
GPU="${GPU:-auto}"

# Resolve script dir defensively: under sbatch on Isambard the script is
# copied to /var/spool/slurmd/job<id>/ where the user can't write, so
# ${BASH_SOURCE[0]} is unsafe. Prefer $SLURM_SUBMIT_DIR when set.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    SCRIPT_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
mkdir -p "$SCRIPT_DIR/logs" 2>/dev/null || true

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

log "Configuration:"
log "  CONTAINER        = $CONTAINER"
log "  COMBINED_SQSH    = $COMBINED_SQSH"
log "  DATA_ROOT        = $DATA_ROOT   (mount point INSIDE the container)"
log "  DATA_LIST_FILE   = $DATA_LIST_FILE"
log "  POINTCEPT_DIR    = $POINTCEPT_DIR"
log "  CONFIG_FILE      = $CONFIG_FILE"
log "  SPLIT            = $SPLIT"
log "  NUM_BATCHES      = $NUM_BATCHES"
log "  BATCH_SIZE       = $BATCH_SIZE"
log "  NUM_WORKERS      = $NUM_WORKERS"
log "  APPLY_TRANSFORMS = $APPLY_TRANSFORMS"
log "  GPU              = $GPU"

# === Preflight ===
[[ -f "$CONTAINER" ]] || { log "ERROR: container not found: $CONTAINER"; exit 2; }
[[ -f "$COMBINED_SQSH" ]] || { log "ERROR: combined sqsh not found: $COMBINED_SQSH"; exit 2; }
[[ -f "$DATA_LIST_FILE" ]] || { log "ERROR: data list not found: $DATA_LIST_FILE"; exit 2; }
[[ -d "$POINTCEPT_DIR" ]] || { log "ERROR: pointcept repo not found: $POINTCEPT_DIR"; exit 2; }
[[ -f "$CONFIG_FILE" ]] || { log "ERROR: config not found: $CONFIG_FILE"; exit 2; }
command -v apptainer >/dev/null || { log "ERROR: apptainer not on PATH"; exit 2; }

# Sanity-check the data list: first non-blank line should start with $DATA_ROOT
first_path=$(grep -vE '^\s*(#|$)' "$DATA_LIST_FILE" | head -n1 || true)
if [[ -z "$first_path" ]]; then
    log "ERROR: data list contains no usable lines: $DATA_LIST_FILE"
    exit 2
fi
case "$first_path" in
    "$DATA_ROOT"/*) ;;
    /*)
        log "WARNING: first data-list path is absolute but not under $DATA_ROOT:"
        log "  $first_path"
        log "  Reads will fail unless this path also exists inside the container."
        ;;
    *)
        log "NOTE: first data-list path is relative; it will be joined with --data-root=$DATA_ROOT inside the container."
        ;;
esac

# === Translate POINTCEPT_DIR-relative paths to their in-container form ===
# POINTCEPT_DIR is bound at /pointcept inside the container; CONFIG_FILE
# under POINTCEPT_DIR gets rewritten so apptainer exec sees the in-container path.
translate_pointcept_path() {
    local p="$1"
    if [[ "$p" == "$POINTCEPT_DIR"/* || "$p" == "$POINTCEPT_DIR" ]]; then
        echo "/pointcept${p#$POINTCEPT_DIR}"
    else
        echo "$p"
    fi
}
CONFIG_IN_CONTAINER=$(translate_pointcept_path "$CONFIG_FILE")
TEST_SCRIPT_IN_CONTAINER="/pointcept/isambard/test_dataset_read.py"

# === Assemble apptainer args ===
APPTAINER_ARGS=(exec)
if [[ "$GPU" != "0" ]]; then
    # --nv on login may fail if there are no GPUs visible; only auto-enable if nvidia-smi is around
    if [[ "$GPU" == "1" ]] || command -v nvidia-smi >/dev/null 2>&1; then
        APPTAINER_ARGS+=(--nv)
    fi
fi

# image-src=/ mounts the root of the squashfs at $DATA_ROOT.
APPTAINER_ARGS+=(--bind "$COMBINED_SQSH:$DATA_ROOT:image-src=/,ro")
# Bind the pointcept repo so the in-container `import pointcept` works.
APPTAINER_ARGS+=(--bind "$POINTCEPT_DIR:/pointcept")
# Bind the data list file (passes through unchanged so the in-container Python sees the same host path).
APPTAINER_ARGS+=(--bind "$DATA_LIST_FILE:$DATA_LIST_FILE:ro")
APPTAINER_ARGS+=(--pwd /pointcept)
APPTAINER_ARGS+=("$CONTAINER")

# === Python command ===
PY_CMD=(python "$TEST_SCRIPT_IN_CONTAINER"
        --config-file "$CONFIG_IN_CONTAINER"
        --data-list-file "$DATA_LIST_FILE"
        --data-root "$DATA_ROOT"
        --split "$SPLIT"
        --num-batches "$NUM_BATCHES"
        --batch-size "$BATCH_SIZE"
        --num-workers "$NUM_WORKERS")
if [[ "$APPLY_TRANSFORMS" == "1" ]]; then
    PY_CMD+=(--apply-transforms)
fi

log "Launching apptainer exec ..."
log "  + apptainer ${APPTAINER_ARGS[*]} ${PY_CMD[*]}"

# PYTHONPATH set via --env so /pointcept is importable as the `pointcept` package.
APPTAINERENV_PYTHONPATH=/pointcept \
    apptainer "${APPTAINER_ARGS[@]}" "${PY_CMD[@]}"

log "test_dataset_read.sh finished."
