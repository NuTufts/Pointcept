#!/bin/bash
#
# Merge all SquashFS shards listed in $SHARDLIST into a single combined .sqsh
# file. Run this on Isambard as a SLURM job. Output lands in $OUTPUT_DIR.
#
# Approach (bare-host, two-phase batched merge):
#   Phase 1. For each batch of up to BATCH_SIZE shards (default 64):
#     a. squashfuse-mount the batch's shards at
#        $MOUNT_ROOT/phase1_batch_<N>/<rel> using `subdir=/<rel>` so the
#        unified view mirrors the original source tree without path doubling
#        (each shard's internal layout is <rel>/file.h5; the `subdir` option
#        strips the prefix at mount time).
#     b. mksquashfs.static against that mount tree, writing
#        $INTERMEDIATES_DIR/intermediate_<NNNN>.sqsh
#     c. Unmount the batch's shards.
#   Phase 2. If more than one intermediate was produced:
#     a. squashfuse-mount each intermediate under $MOUNT_ROOT/phase2/inter_<NNNN>
#     b. mksquashfs.static against ALL intermediate mount points as multiple
#        sources; mksquashfs merges them into one canonical tree because
#        their internal leaf paths are non-overlapping.
#     c. Unmount intermediates.
#   If only one intermediate exists, that intermediate IS the final image; we
#   just `mv` it into place.
#
# Why batched: Isambard's per-user FUSE mount limit (`mount_max` in
# /etc/fuse3.conf, default 1000) caps how many shards we can mount
# simultaneously. The single-pass merge that worked for 5 shards fails at
# 708. With BATCH_SIZE=64 and ~12 intermediates, concurrent mounts stay well
# below the cap in both phases.
#
# Resume support: if intermediate_<NNNN>.sqsh already exists at the start of
# batch N, that batch is skipped. If the final $OUTPUT_SQSH already exists,
# phase 2 is skipped entirely. So a job that times out partway through
# phase 1 picks up where it left off on resubmit. To force a rebuild, delete
# the relevant intermediate(s) and / or the final.
#
# Cleanup discipline:
#   - pre-cleanup: drop any stale FUSE mounts left under $MOUNT_ROOT by a
#     prior crashed job, then trap EXIT/TERM/INT/USR1 so we unmount on any
#     in-script exit. SIGKILL cannot be trapped; leftover mounts under /tmp
#     survive until the next job's pre-cleanup or a node reboot.
#   - intermediates: kept on success only if KEEP_INTERMEDIATES=1 (default 0
#     deletes them after phase 2 to free disk).
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
#     BATCH_SIZE=64 \
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

# Number of shards merged per intermediate. Keep below the per-user FUSE
# mount_max (default 1000 on Isambard, but other workloads count against the
# same cap, so 64 is a safe ceiling). Larger batches = fewer intermediates =
# faster phase 2 but more concurrent mounts during phase 1.
BATCH_SIZE="${BATCH_SIZE:-64}"

# Where to write per-batch intermediate .sqsh files. Default lives next to
# the final output and is cleaned after a successful phase 2 unless
# KEEP_INTERMEDIATES=1.
INTERMEDIATES_DIR="${INTERMEDIATES_DIR:-$OUTPUT_DIR/${OUTPUT_NAME}.intermediates}"
KEEP_INTERMEDIATES="${KEEP_INTERMEDIATES:-0}"

MKSQUASHFS="${MKSQUASHFS:-mksquashfs.static}"
# unsquashfs is only used for the post-merge file-count sanity check. If it
# isn't installed (Isambard ships mksquashfs.static + squashfuse but not
# necessarily unsquashfs), the check is skipped instead of aborting.
UNSQUASHFS="${UNSQUASHFS:-unsquashfs.static}"
SQUASHFUSE="${SQUASHFUSE:-squashfuse}"
# Compressor for the metadata tables in the combined .sqsh. Isambard's
# mksquashfs.static only supports gzip and lz4 (no zstd). With -noI -noD the
# compressor only affects small metadata tables, not data blocks, so output
# size barely changes. gzip is the universal default; lz4 is marginally
# faster to decompress at mount time.
COMPRESSOR="${COMPRESSOR:-gzip}"
PROCESSORS="${PROCESSORS:-${SLURM_CPUS_PER_TASK:-8}}"
MOUNT_PARALLEL="${MOUNT_PARALLEL:-8}"
# sha256sum is single-threaded and on a 7.4 TB combined image takes ~5 h
# (vs. ~1 h for the merge itself at the throughput we measured). Set
# SKIP_SHA256=1 to skip it; you can always compute it separately later
# with `sha256sum combined_*.sqsh > combined_*.sha256`.
SKIP_SHA256="${SKIP_SHA256:-0}"

# Mountpoints under /tmp. Do NOT default to $SLURM_TMPDIR — on Isambard
# (and likely other sites that configure SLURM TmpFS / PrivateTmp), that
# path has private mount propagation that makes user-space FUSE mounts
# invisible in /proc/mounts to the script's bash. Confirmed empirically:
# the same parallel mount loop succeeds under /tmp and produces 0 visible
# mounts under $SLURM_TMPDIR. /tmp on Isambard compute nodes is node-local
# and per-user effectively, and our trap EXIT plus pre-cleanup keep stale
# mounts from accumulating across jobs.
MOUNT_ROOT="${MOUNT_ROOT:-/tmp/merge_mounts.${SLURM_JOB_ID:-$$}}"

OUTPUT_SQSH="$OUTPUT_DIR/${OUTPUT_NAME}.sqsh"
OUTPUT_SHA="$OUTPUT_DIR/${OUTPUT_NAME}.sha256"

mkdir -p "$(dirname "$OUTPUT_SQSH")"
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

# === Cleanup machinery ===
# Unmount every FUSE filesystem currently mounted under the given root.
# Deepest-first so unmounts don't fail with EBUSY on nested mounts.
cleanup_under() {
    local root="$1"
    [[ -z "$root" ]] && return 0
    local mounts
    mounts=$(awk -v root="$root" \
        '$2 ~ "^"root && $3 ~ /squashfs|fuse/ {print $2}' /proc/mounts \
        | sort -r)
    if [[ -n "$mounts" ]]; then
        local n
        n=$(printf '%s\n' "$mounts" | wc -l)
        log "Releasing $n FUSE mount(s) under $root"
        printf '%s\n' "$mounts" \
            | xargs -r -n1 fusermount -uz 2>/dev/null || true
    fi
}

# Trap target: clean everything we might have mounted, then remove MOUNT_ROOT.
cleanup_mounts() {
    cleanup_under "$MOUNT_ROOT"
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
# Required tools (script aborts if missing).
for tool in "$MKSQUASHFS" "$SQUASHFUSE" fusermount numfmt sha256sum; do
    command -v "$tool" >/dev/null \
        || { log "ERROR: '$tool' not on PATH. module load? or override with the matching env var (MKSQUASHFS/SQUASHFUSE)."; exit 2; }
done
# Optional: unsquashfs is only used for the final file-count sanity check.
HAVE_UNSQUASHFS=0
if command -v "$UNSQUASHFS" >/dev/null; then
    HAVE_UNSQUASHFS=1
fi
[[ -f "$SHARDLIST" ]] || { log "ERROR: shardlist not found: $SHARDLIST"; exit 2; }
[[ -e "$OUTPUT_SQSH" ]] && { log "NOTE: $OUTPUT_SQSH already exists — phase 2 will be skipped (resume)."; }
(( BATCH_SIZE > 0 )) || { log "ERROR: BATCH_SIZE must be > 0 (got $BATCH_SIZE)"; exit 2; }

log "Tools resolved:"
log "  mksquashfs: $(command -v "$MKSQUASHFS")"
if (( HAVE_UNSQUASHFS )); then
    log "  unsquashfs: $(command -v "$UNSQUASHFS")"
else
    log "  unsquashfs: (not found - file-count sanity check will be skipped)"
fi
log "  squashfuse: $(command -v "$SQUASHFUSE")"
log "  fusermount: $(command -v fusermount)"

# Free-space check. Peak disk during phase 2 is roughly
# (intermediates) + (final being written) ≈ 2 × total_shard_bytes.
# Add 10 % margin → 2.2 ×. After phase 2 + cleanup, only the final remains
# (1 ×). If KEEP_INTERMEDIATES=1, steady-state is ~2 ×.
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
needed_bytes=$(( total_bytes * 220 / 100 ))
log "Total shard bytes : $(numfmt --to=iec "$total_bytes")"
log "Peak required     : $(numfmt --to=iec "$needed_bytes") (2.2x — intermediates + final)"
log "Available in dest : $(numfmt --to=iec "$avail_bytes")"
if (( avail_bytes < needed_bytes )); then
    log "ERROR: insufficient free space in $OUTPUT_DIR"
    exit 4
fi

# Compute batching.
shard_count=$(grep -cE '\.sqsh\s*$' "$SHARDLIST")
(( shard_count > 0 )) || { log "ERROR: shardlist contains no .sqsh entries"; exit 2; }
n_batches=$(( (shard_count + BATCH_SIZE - 1) / BATCH_SIZE ))
log "Batching: $shard_count shards / BATCH_SIZE=$BATCH_SIZE -> $n_batches intermediate(s)"
log "  INTERMEDIATES_DIR=$INTERMEDIATES_DIR"
log "  OUTPUT_SQSH=$OUTPUT_SQSH"
log "  KEEP_INTERMEDIATES=$KEEP_INTERMEDIATES"

# Pre-cleanup any stale mounts left under MOUNT_ROOT by a prior crashed job.
cleanup_mounts
mkdir -p "$MOUNT_ROOT" "$INTERMEDIATES_DIR"

# ---- Per-batch mount helper (used in phase 1 via xargs) ----
mount_shard_for_batch() {
    local sqsh="$1"
    [[ -z "$sqsh" ]] && return 0
    local rel="${sqsh#${STAGING_DIR}/}"
    rel="${rel%.sqsh}"
    local mp="$BATCH_MOUNT_ROOT/$rel"
    mkdir -p "$mp"
    # subdir=/$rel: expose only the leaf inside the shard so $mp/file.h5 works,
    # not $mp/$rel/file.h5. See the script header.
    if ! "$SQUASHFUSE" -o "subdir=/$rel" "$sqsh" "$mp" 2>&1; then
        echo "FAILED mount: $sqsh" >&2
        return 1
    fi
}
export -f mount_shard_for_batch
export STAGING_DIR SQUASHFUSE
# BATCH_MOUNT_ROOT is exported per-batch inside the loop.

intermediate_path() {
    local idx="$1"
    printf '%s/intermediate_%04d.sqsh' "$INTERMEDIATES_DIR" "$idx"
}

# === PHASE 1: build per-batch intermediates ===
log ""
log "=== PHASE 1: build $n_batches intermediate(s) ==="
t_phase1_start=$SECONDS
batches_built=0
batches_skipped=0

for (( batch_idx=0; batch_idx<n_batches; batch_idx++ )); do
    intermediate="$(intermediate_path $batch_idx)"

    if [[ -f "$intermediate" ]]; then
        log "[batch $batch_idx/$((n_batches-1))] $intermediate exists — skipping (resume)"
        batches_skipped=$((batches_skipped + 1))
        continue
    fi

    # Compute shard range for this batch (sed is 1-indexed, inclusive).
    batch_lo=$(( batch_idx * BATCH_SIZE + 1 ))
    batch_hi=$(( (batch_idx + 1) * BATCH_SIZE ))
    (( batch_hi > shard_count )) && batch_hi=$shard_count

    # Materialize this batch's shardlist as a tempfile (avoids re-piping sed
    # repeatedly and keeps xargs's input simple).
    batch_listfile=$(mktemp -p "${TMPDIR:-/tmp}" batchlist.XXXXXX)
    sed -n "${batch_lo},${batch_hi}p" "$SHARDLIST" > "$batch_listfile"
    batch_count=$(grep -cE '\.sqsh\s*$' "$batch_listfile")

    log "[batch $batch_idx/$((n_batches-1))] $batch_count shards (lines $batch_lo..$batch_hi)"

    # Set up batch mount root. Export so xargs's bash -c workers see it.
    BATCH_MOUNT_ROOT="$MOUNT_ROOT/phase1_batch_$(printf '%04d' $batch_idx)"
    export BATCH_MOUNT_ROOT
    cleanup_under "$BATCH_MOUNT_ROOT"
    rm -rf "$BATCH_MOUNT_ROOT"
    mkdir -p "$BATCH_MOUNT_ROOT"

    # Mount all shards in this batch in parallel.
    t_b_mount=$SECONDS
    tr '\n' '\0' < "$batch_listfile" \
        | xargs -0 -P "$MOUNT_PARALLEL" -I{} bash -c 'mount_shard_for_batch "$@"' _ {} \
        || { log "ERROR: batch $batch_idx: at least one mount failed; see log/stderr"; exit 5; }

    mounted=$(awk -v root="$BATCH_MOUNT_ROOT" \
        '$2 ~ "^"root && $3 ~ /squashfs|fuse/' /proc/mounts | wc -l)
    log "[batch $batch_idx/$((n_batches-1))] mounted $mounted/$batch_count in $((SECONDS - t_b_mount))s"
    if (( mounted != batch_count )); then
        log "ERROR: batch $batch_idx: expected $batch_count mounts, got $mounted. Aborting."
        rm -f "$batch_listfile"
        exit 5
    fi

    # mksquashfs the batch.
    t_b_mksqfs=$SECONDS
    "$MKSQUASHFS" "$BATCH_MOUNT_ROOT" "$intermediate" \
        -comp "$COMPRESSOR" \
        -noI -noD \
        -b 128K \
        -processors "$PROCESSORS" \
        -no-recovery \
        -no-progress \
        -quiet

    inter_size=$(numfmt --to=iec "$(stat -c %s "$intermediate")")
    log "[batch $batch_idx/$((n_batches-1))] mksquashfs done in $((SECONDS - t_b_mksqfs))s, intermediate=$inter_size"

    # Unmount this batch and clear its mount tree.
    cleanup_under "$BATCH_MOUNT_ROOT"
    rm -rf "$BATCH_MOUNT_ROOT"
    rm -f "$batch_listfile"

    batches_built=$((batches_built + 1))
done

log "Phase 1 done: built=$batches_built, skipped=$batches_skipped, total=$n_batches, in $((SECONDS - t_phase1_start))s"

# === PHASE 2: merge intermediates ===
log ""
log "=== PHASE 2: merge $n_batches intermediate(s) -> $OUTPUT_SQSH ==="
t_phase2_start=$SECONDS

if [[ -e "$OUTPUT_SQSH" ]]; then
    log "$OUTPUT_SQSH already exists — skipping phase 2 (resume)."
elif (( n_batches == 1 )); then
    only_inter="$(intermediate_path 0)"
    log "Single batch — moving $only_inter -> $OUTPUT_SQSH"
    mv "$only_inter" "$OUTPUT_SQSH"
else
    PHASE2_MOUNT_ROOT="$MOUNT_ROOT/phase2"
    cleanup_under "$PHASE2_MOUNT_ROOT"
    rm -rf "$PHASE2_MOUNT_ROOT"
    mkdir -p "$PHASE2_MOUNT_ROOT"

    sources=()
    t_p2_mount=$SECONDS
    for (( i=0; i<n_batches; i++ )); do
        intermediate="$(intermediate_path $i)"
        [[ -f "$intermediate" ]] || { log "ERROR: missing intermediate $intermediate"; exit 5; }
        mp="$PHASE2_MOUNT_ROOT/inter_$(printf '%04d' $i)"
        mkdir -p "$mp"
        if ! "$SQUASHFUSE" "$intermediate" "$mp" 2>&1; then
            log "ERROR: failed to mount intermediate $intermediate"
            exit 5
        fi
        sources+=("$mp")
    done
    log "Phase 2 mounted ${#sources[@]} intermediate(s) in $((SECONDS - t_p2_mount))s"

    # mksquashfs of all intermediate mount points as multiple sources.
    # mksquashfs merges same-named directories across sources; conflicting
    # filenames would be resolved by the right-most source winning, but for
    # us the leaf-dir paths are disjoint so no conflicts arise.
    t_p2_mksqfs=$SECONDS
    "$MKSQUASHFS" "${sources[@]}" "$OUTPUT_SQSH" \
        -comp "$COMPRESSOR" \
        -noI -noD \
        -b 128K \
        -processors "$PROCESSORS" \
        -no-recovery \
        -no-progress \
        -quiet
    log "Phase 2 mksquashfs done in $((SECONDS - t_p2_mksqfs))s"

    cleanup_under "$PHASE2_MOUNT_ROOT"
    rm -rf "$PHASE2_MOUNT_ROOT"
fi

log "Phase 2 done in $((SECONDS - t_phase2_start))s"
ls -lh "$OUTPUT_SQSH"

# === Checksum + sanity check ===
if (( SKIP_SHA256 )); then
    log "Skipping sha256 (SKIP_SHA256=1). Run later with:"
    log "  sha256sum ${OUTPUT_NAME}.sqsh > ${OUTPUT_NAME}.sha256"
else
    log "Computing sha256 (single-threaded; ~5 h on a 7.4 TB image) ..."
    t_sha_start=$SECONDS
    ( cd "$OUTPUT_DIR" && sha256sum "${OUTPUT_NAME}.sqsh" ) | tee "$OUTPUT_SHA"
    log "sha256 finished in $((SECONDS - t_sha_start))s ($(( (SECONDS - t_sha_start) / 60 )) min)"
fi

if (( HAVE_UNSQUASHFS )); then
    log "File count check (.h5 files in combined image)..."
    combined_files=$("$UNSQUASHFS" -l "$OUTPUT_SQSH" 2>/dev/null | grep -c '\.h5$' || true)
    log "  combined: $combined_files .h5 files"
else
    log "Skipping .h5 file-count check ($UNSQUASHFS not available)."
fi

# === Clean up intermediates (unless asked to keep) ===
if [[ -d "$INTERMEDIATES_DIR" ]]; then
    if [[ "$KEEP_INTERMEDIATES" == "1" ]]; then
        log "Keeping intermediates dir (KEEP_INTERMEDIATES=1): $INTERMEDIATES_DIR"
    else
        log "Removing intermediates dir: $INTERMEDIATES_DIR"
        rm -rf "$INTERMEDIATES_DIR"
    fi
fi

log "Total runtime: ${SECONDS}s ($(( SECONDS / 60 )) min)"
log "Done. Cleanup runs via EXIT trap."
