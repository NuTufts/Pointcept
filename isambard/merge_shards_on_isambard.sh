#!/bin/bash
#
# Merge all SquashFS shards listed in $SHARDLIST into a single combined .sqsh
# file. Run this on Isambard as a SLURM job. Output lands next to the input
# shards in $OUTPUT_DIR.
#
# Approach (bare-host, using Isambard's documented squashfs tooling):
#   1. squashfuse-mount each shard at $MOUNT_ROOT/<rel> with `subdir=/<rel>`
#      so the unified view mirrors the original source tree without path
#      doubling (each shard's internal layout is <rel>/file.h5 because
#      build_squashfs.py builds a hardlink staging tree before mksquashfs;
#      the `subdir` option strips the prefix at mount time).
#   2. mksquashfs.static $MOUNT_ROOT -> $OUTPUT_SQSH with the same flags
#      build_squashfs.py used (-noI -noD -b 128K -comp zstd).
#   3. Aggressive cleanup discipline: pre-cleanup any stale FUSE mounts
#      left under our chosen MOUNT_ROOT by a prior crashed job, then trap
#      EXIT/TERM/INT/USR1 so we unmount on any in-script exit. SIGKILL
#      from SLURM cannot be trapped, but MOUNT_ROOT lives under
#      $SLURM_TMPDIR so the per-job tmpfs gets reclaimed at job end.
#
# Tooling: defaults are Isambard's documented mksquashfs.static + squashfuse.
# If those aren't on PATH for some reason, build the sidecar container
# (squashfs_tools.def -> squashfs_tools.sif) and prefix every binary call
# with `apptainer exec <sif>` — see the README's fallback subsection.
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
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --signal=B:USR1@120
# TODO Isambard: fill in --partition / --account / --qos as your project requires.

set -euo pipefail

# === Config (override via env) ===
STAGING_DIR="${STAGING_DIR:-/projects/u6jo/staging}"
SHARDLIST="${SHARDLIST:-$STAGING_DIR/isambard_shardlist_test.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-/projects/u6jo/datasets}"
OUTPUT_NAME="${OUTPUT_NAME:-combined_pretrain-sonata-v7-extbnb-larmatch-test}"
MKSQUASHFS="${MKSQUASHFS:-mksquashfs.static}"
UNSQUASHFS="${UNSQUASHFS:-unsquashfs}"
SQUASHFUSE="${SQUASHFUSE:-squashfuse}"
PROCESSORS="${PROCESSORS:-${SLURM_CPUS_PER_TASK:-8}}"
MOUNT_PARALLEL="${MOUNT_PARALLEL:-8}"

# Mountpoints under node-local scratch so they vanish with the job.
MOUNT_ROOT="${MOUNT_ROOT:-${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/merge_mounts.${SLURM_JOB_ID:-$$}}"

OUTPUT_SQSH="$OUTPUT_DIR/${OUTPUT_NAME}.sqsh"
OUTPUT_SHA="$OUTPUT_DIR/${OUTPUT_NAME}.sha256"

mkdir -p "$(dirname "$OUTPUT_SQSH")"
mkdir -p "$(dirname "${BASH_SOURCE[0]}")/logs"

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

# === Cleanup machinery ===
cleanup_mounts() {
    # Deepest-first so unmounts don't fail with EBUSY due to nested mounts.
    local mounts
    mounts=$(awk -v root="$MOUNT_ROOT" \
        '$2 ~ "^"root && $3 ~ /squashfs|fuse/ {print $2}' /proc/mounts \
        | sort -r)
    if [[ -n "$mounts" ]]; then
        local n
        n=$(printf '%s\n' "$mounts" | wc -l)
        log "Releasing $n FUSE mount(s) under $MOUNT_ROOT"
        printf '%s\n' "$mounts" \
            | xargs -r -n1 fusermount -uz 2>/dev/null || true
    fi
    rm -rf "$MOUNT_ROOT" 2>/dev/null || true
}

graceful_exit() {
    log "Caught signal — releasing mounts and exiting"
    cleanup_mounts
    exit 1
}

trap cleanup_mounts EXIT
trap graceful_exit USR1 TERM INT

# === Preflight ===
for tool in "$MKSQUASHFS" "$UNSQUASHFS" "$SQUASHFUSE" fusermount numfmt sha256sum; do
    command -v "$tool" >/dev/null \
        || { log "ERROR: '$tool' not on PATH. module load? or override with the matching env var (MKSQUASHFS/UNSQUASHFS/SQUASHFUSE)."; exit 2; }
done
[[ -f "$SHARDLIST" ]] || { log "ERROR: shardlist not found: $SHARDLIST"; exit 2; }
[[ -e "$OUTPUT_SQSH" ]] && { log "ERROR: $OUTPUT_SQSH already exists. Delete it or set OUTPUT_NAME=..."; exit 2; }

log "Tools resolved:"
log "  mksquashfs: $(command -v "$MKSQUASHFS")"
log "  unsquashfs: $(command -v "$UNSQUASHFS")"
log "  squashfuse: $(command -v "$SQUASHFUSE")"
log "  fusermount: $(command -v fusermount)"

# Free-space check (~1.05x of total shard size).
total_bytes=$(awk '
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

# Pre-cleanup in case a prior crashed job left mounts under our root.
cleanup_mounts
mkdir -p "$MOUNT_ROOT"

# === Mount all shards in parallel ===
shard_count=$(grep -cE '\.sqsh\s*$' "$SHARDLIST")
log "Mounting $shard_count shards into $MOUNT_ROOT (parallel=$MOUNT_PARALLEL)..."
t_mount_start=$SECONDS

mount_one() {
    local sqsh="$1"
    [[ -z "$sqsh" ]] && return 0
    local rel="${sqsh#${STAGING_DIR}/}"
    rel="${rel%.sqsh}"
    local mp="$MOUNT_ROOT/$rel"
    mkdir -p "$mp"
    # subdir=/$rel: expose only the leaf inside the shard so $mp/file.h5 works,
    # not $mp/$rel/file.h5. See the script header.
    if ! "$SQUASHFUSE" -o "subdir=/$rel" "$sqsh" "$mp" 2>&1; then
        echo "FAILED mount: $sqsh" >&2
        return 1
    fi
}
export -f mount_one
export MOUNT_ROOT STAGING_DIR SQUASHFUSE

# NUL-delimit so paths with spaces survive (shouldn't happen for these shards, but safe).
tr '\n' '\0' < "$SHARDLIST" \
    | xargs -0 -n1 -P "$MOUNT_PARALLEL" -I{} bash -c 'mount_one "$@"' _ {} \
    || { log "ERROR: at least one mount failed; see stderr"; exit 5; }

mounted=$(awk -v root="$MOUNT_ROOT" \
    '$2 ~ "^"root && $3 ~ /squashfs|fuse/' /proc/mounts | wc -l)
log "Mounted $mounted shards in $((SECONDS - t_mount_start))s"
if (( mounted != shard_count )); then
    log "ERROR: expected $shard_count mounts, got $mounted. Aborting."
    exit 5
fi

# === Merge ===
log "Running $MKSQUASHFS ($PROCESSORS processors) -> $OUTPUT_SQSH"
t_merge_start=$SECONDS

"$MKSQUASHFS" "$MOUNT_ROOT" "$OUTPUT_SQSH" \
    -comp zstd -Xcompression-level 1 \
    -noI -noD \
    -b 128K \
    -processors "$PROCESSORS" \
    -no-recovery \
    -no-progress \
    -quiet

log "mksquashfs finished in $((SECONDS - t_merge_start))s ($(( (SECONDS - t_merge_start) / 60 )) min)"
ls -lh "$OUTPUT_SQSH"

# === Checksum + sanity check ===
log "Computing sha256 ..."
( cd "$OUTPUT_DIR" && sha256sum "${OUTPUT_NAME}.sqsh" ) | tee "$OUTPUT_SHA"

log "File count check (.h5 files in combined image)..."
combined_files=$("$UNSQUASHFS" -l "$OUTPUT_SQSH" 2>/dev/null | grep -c '\.h5$' || true)
log "  combined: $combined_files .h5 files"

log "Total runtime: $((SECONDS - t_mount_start))s ($(( (SECONDS - t_mount_start) / 60 )) min)"
log "Done. Cleanup runs via EXIT trap."
