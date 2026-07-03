#!/bin/bash
#
# Inspect the pred_instances group written by
# tools/larformer/augment_stage12_cache_pred_masks_shard.py in a few cache .h5 files.
#
# For each file it prints the entry_0/pred_instances group attrs (provenance:
# segmenter_weight, dedup params, shard) and a summary of the first few
# instance_<k> subgroups (truth_indices shape + pred_class/primary_trackid/
# match_iou). A file with NO pred_instances group has not been augmented yet.
#
# Usage:
#   ./inspect_pred_masks.sh <LIST_FILE | CACHE_ROOT> [N]
#
#   LIST_FILE   a text file of .h5 paths (e.g. from count_modified_cache_files.sh)
#   CACHE_ROOT  a directory tree of .h5 files; inspects the N most recently
#               modified ones
#   N           how many files to inspect (default 3)
#
# Examples:
#   ./inspect_pred_masks.sh /cluster/.../larformer_cache_..._test 3
#   ./count_modified_cache_files.sh "$ROOT" '2026-06-19T09:00:00-04:00' mod.txt
#   ./inspect_pred_masks.sh mod.txt 5

set -euo pipefail

CONTAINER=/cluster/tufts/wongjiradlabnu/larbys/larbys-container/pointcept_cuml.sif

if [ "$#" -lt 1 ]; then
    echo "usage: $0 <LIST_FILE | CACHE_ROOT> [N]" >&2
    exit 1
fi

TARGET="$1"
N="${2:-3}"

# Build the list of files to inspect.
if [ -f "$TARGET" ]; then
    mapfile -t FILES < <(head -n "$N" "$TARGET")
elif [ -d "$TARGET" ]; then
    # N most recently modified .h5 under the tree.
    mapfile -t FILES < <(find "$TARGET" -type f -name '*.h5' -printf '%T@ %p\n' \
                         | sort -n | tail -n "$N" | awk '{print $2}')
else
    echo "error: '$TARGET' is neither a file (list) nor a directory" >&2
    exit 1
fi

if [ "${#FILES[@]}" -eq 0 ]; then
    echo "no .h5 files found for '$TARGET'" >&2
    exit 1
fi

echo "inspecting ${#FILES[@]} file(s):"
printf '  %s\n' "${FILES[@]}"
echo "==="

apptainer exec --bind /cluster:/cluster "$CONTAINER" python3 - "${FILES[@]}" <<'PY'
import sys, h5py

PRED_GROUP = "entry_0/pred_instances"

for path in sys.argv[1:]:
    print("\n####", path)
    try:
        with h5py.File(path, "r") as f:
            g = f.get(PRED_GROUP)
            if g is None:
                print("  NO pred_instances group (not augmented yet)")
                continue
            attrs = {k: g.attrs[k] for k in g.attrs}
            print("  group attrs:")
            for k in sorted(attrs):
                print(f"    {k} = {attrs[k]}")
            insts = sorted(k for k in g.keys() if k.startswith("instance_"))
            print(f"  n instance subgroups: {len(insts)}")
            for k in insts[:5]:
                sub = g[k]
                ti = sub["truth_indices"]
                sa = ", ".join(f"{a}={sub.attrs[a]}" for a in sub.attrs)
                print(f"    {k}: truth_indices shape={ti.shape} "
                      f"dtype={ti.dtype} attrs={{{sa}}}")
            if len(insts) > 5:
                print(f"    ... ({len(insts) - 5} more)")
    except Exception as e:
        print(f"  ERROR reading file: {type(e).__name__}: {e}")
PY
