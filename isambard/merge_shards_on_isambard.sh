#!/bin/bash
#
# Merge all SquashFS shards listed in $SHARDLIST into a single combined .sqsh
# file. Run this on Isambard as a SLURM job. Output lands next to the input
# shards in $OUTPUT_DIR.
#
# Approach: launch one apptainer container with `--mount type=squashfs`
# repeated once per shard, with image-src targeting each shard's internal
# leaf-dir so the unified view at /unified mirrors the original source tree
# without path doubling. Then mksquashfs /unified -> combined.sqsh.
#
# Usage (sbatch):
#     sbatch merge_shards_on_isambard.sh
#
# Override defaults via environment:
#     SHARDLIST=/path/to/shardlist.txt \
#     OUTPUT_NAME=combined_my_dataset \
#     sbatch merge_shards_on_isambard.sh
#
# Smoke-test interactively (no sbatch):
#     head -5 isambard_shardlist.txt > /tmp/test_shards.txt
#     SHARDLIST=/tmp/test_shards.txt OUTPUT_NAME=combined_test \
#         bash merge_shards_on_isambard.sh
#
# === SLURM directives ===
#SBATCH --job-name=merge-shards
#SBATCH --output=logs/merge-shards-%j.out
#SBATCH --error=logs/merge-shards-%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
# TODO Isambard: fill in --partition / --account / --qos as your project requires.

set -euo pipefail

# === Config (override via env) ===
STAGING_DIR="${STAGING_DIR:-/projects/u6jo/staging}"
SHARDLIST="${SHARDLIST:-$STAGING_DIR/isambard_shardlist.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-$STAGING_DIR}"
OUTPUT_NAME="${OUTPUT_NAME:-combined_pretrain-sonata-v7-extbnb-larmatch}"
CONTAINER="${CONTAINER:-/projects/u6jo/pointcept.sif}"
PROCESSORS="${PROCESSORS:-${SLURM_CPUS_PER_TASK:-8}}"

OUTPUT_SQSH="$OUTPUT_DIR/${OUTPUT_NAME}.sqsh"
OUTPUT_SHA="$OUTPUT_DIR/${OUTPUT_NAME}.sha256"

mkdir -p "$(dirname "$OUTPUT_SQSH")"
mkdir -p "$(dirname "${BASH_SOURCE[0]}")/logs"

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

# === Preflight ===
[[ -f "$SHARDLIST" ]] || { log "ERROR: shardlist not found: $SHARDLIST"; exit 2; }
[[ -f "$CONTAINER" ]] || { log "ERROR: container not found: $CONTAINER"; exit 2; }
if [[ -e "$OUTPUT_SQSH" ]]; then
    log "ERROR: $OUTPUT_SQSH already exists. Delete it or set OUTPUT_NAME=..."
    exit 2
fi

# Check there's enough free space on the output FS (~1.05x of total shard size).
total_bytes=$(awk -v dir="$STAGING_DIR" '
    {
        path = $0
        if (path == "") next
        cmd = "stat -c %s " path " 2>/dev/null"
        cmd | getline sz
        close(cmd)
        if (sz == "") { print "MISSING:" path > "/dev/stderr"; missing++; next }
        total += sz
    }
    END { print total; if (missing) exit 3 }
' "$SHARDLIST") || { log "ERROR: one or more shards missing (see stderr)"; exit 3; }

avail_bytes=$(df -B1 --output=avail "$OUTPUT_DIR" | tail -1)
needed_bytes=$(( total_bytes * 105 / 100 ))
log "Total shard bytes : $(numfmt --to=iec "$total_bytes")"
log "Required free     : $(numfmt --to=iec "$needed_bytes") (1.05x)"
log "Available in dest : $(numfmt --to=iec "$avail_bytes")"
if (( avail_bytes < needed_bytes )); then
    log "ERROR: insufficient free space in $OUTPUT_DIR"
    exit 4
fi

# === Build mount spec ===
log "Building --mount args from $SHARDLIST ..."
MOUNT_ARGS=()
shard_count=0
while IFS= read -r sqsh; do
    [[ -z "$sqsh" ]] && continue
    rel="${sqsh#$STAGING_DIR/}"
    rel="${rel%.sqsh}"
    # image-src=/$rel exposes only the leaf directory inside the shard so the
    # unified /unified/<rel>/ contains the leaf's files directly, not a
    # doubled <rel>/<rel>/ chain.
    MOUNT_ARGS+=(--mount "type=squashfs,source=$sqsh,destination=/unified/$rel,image-src=/$rel,ro")
    shard_count=$((shard_count + 1))
done < "$SHARDLIST"
log "Prepared $shard_count shard mounts"

# === Run mksquashfs inside container with all shards mounted ===
log "Launching apptainer + mksquashfs ($PROCESSORS processors)"
log "  output: $OUTPUT_SQSH"

# Apptainer may need a writable /tmp for its own bookkeeping per-container.
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${TMPDIR:-/tmp}/apptainer-merge-${SLURM_JOB_ID:-$$}}"
mkdir -p "$APPTAINER_TMPDIR"

t0=$SECONDS

apptainer exec \
    --bind "$OUTPUT_DIR:/out:rw" \
    "${MOUNT_ARGS[@]}" \
    "$CONTAINER" \
    mksquashfs /unified "/out/${OUTPUT_NAME}.sqsh" \
        -comp zstd -Xcompression-level 1 \
        -noI -noD \
        -b 128K \
        -processors "$PROCESSORS" \
        -no-recovery \
        -no-progress \
        -quiet

elapsed=$(( SECONDS - t0 ))
log "mksquashfs finished in ${elapsed}s ($(( elapsed / 60 )) min)"
ls -lh "$OUTPUT_SQSH"

# === Checksum ===
log "Computing sha256 ..."
( cd "$OUTPUT_DIR" && sha256sum "${OUTPUT_NAME}.sqsh" ) | tee "$OUTPUT_SHA"

# === Final sanity check: file count round-trip ===
log "Counting files inside combined shard (sanity check)..."
combined_files=$(apptainer exec \
    --bind "$OUTPUT_DIR:/out:ro" \
    "$CONTAINER" \
    unsquashfs -l "/out/${OUTPUT_NAME}.sqsh" 2>/dev/null \
    | grep -c '\.h5$' || true)
log "Combined .sqsh contains $combined_files .h5 files"

rm -rf "$APPTAINER_TMPDIR"
log "Done."
