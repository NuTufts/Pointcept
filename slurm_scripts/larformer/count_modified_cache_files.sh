#!/bin/bash
#
# Count (and optionally list) cache .h5 files modified after a timestamp.
#
# Use this to watch the in-place augment job
# (tools/augment_stage12_cache_pred_masks_shard.py) make progress: it rewrites
# each file as it adds pred_instances, so "modified after launch time" == "done".
#
# Usage:
#   ./count_modified_cache_files.sh <CACHE_ROOT> <TIMESTAMP> [OUT_LIST]
#
#   CACHE_ROOT  directory tree of cache .h5 files (recursed)
#   TIMESTAMP   ISO-8601 with offset, e.g. '2026-06-19T09:00:00-04:00'
#               (-04:00 = US Eastern in summer / EDT; -05:00 in winter / EST)
#   OUT_LIST    optional path to write the sorted list of modified files
#
# Examples:
#   ./count_modified_cache_files.sh \
#       /cluster/tufts/wongjiradlab/larbys/data/ub_on_tufts/hdf5/larformer_cache_stage12_highermaxsp_predmask_ptv3crosslevelslicer_iter_75750_test \
#       '2026-06-19T09:00:00-04:00'
#
#   ./count_modified_cache_files.sh "$ROOT" '2026-06-19T09:00:00-04:00' modified.txt

set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "usage: $0 <CACHE_ROOT> <TIMESTAMP_ISO8601> [OUT_LIST]" >&2
    echo "  e.g. $0 /path/to/cache '2026-06-19T09:00:00-04:00' modified.txt" >&2
    exit 1
fi

ROOT="$1"
STAMP="$2"
OUT_LIST="${3:-}"

if [ ! -d "$ROOT" ]; then
    echo "error: not a directory: $ROOT" >&2
    exit 1
fi

# Validate the timestamp by probing find once; bfs/find rejects bad formats.
if ! find "$ROOT" -maxdepth 0 -newermt "$STAMP" >/dev/null 2>&1; then
    echo "error: bad timestamp '$STAMP' (need ISO-8601, e.g. 2026-06-19T09:00:00-04:00)" >&2
    exit 1
fi

modified=$(find "$ROOT" -type f -name '*.h5' -newermt "$STAMP" | wc -l)
total=$(find "$ROOT" -type f -name '*.h5' | wc -l)

pct="n/a"
if [ "$total" -gt 0 ]; then
    pct=$(awk -v m="$modified" -v t="$total" 'BEGIN{printf "%.1f", 100.0*m/t}')
fi

echo "cache root : $ROOT"
echo "since      : $STAMP"
echo "modified   : $modified"
echo "total .h5  : $total"
echo "progress   : ${pct}%"

if [ -n "$OUT_LIST" ]; then
    find "$ROOT" -type f -name '*.h5' -newermt "$STAMP" | sort > "$OUT_LIST"
    echo "list       : $OUT_LIST ($(wc -l < "$OUT_LIST") files)"
fi
