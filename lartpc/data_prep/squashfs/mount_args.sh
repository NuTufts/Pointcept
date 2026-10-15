#!/bin/bash
# Print the apptainer --bind flags that mount every <SAMPLE>_merged_sp_<x>.sqfs
# image of a directory under one merged_sp mount point (read-only):
#   apptainer exec --nv $(bash mount_args.sh <image dir> <mount point>) pointcept.sif ...
# Two image kinds (pack_merged_sp_squashfs.sh):
#   <x> = NNN (per top-level dir, image root "NNN/"): mounted at <mnt>/NNN with
#         image-src=/NNN, so the tree looks exactly like the Tufts merged_sp/.
#   <x> = gKK (per fileno group, image root = NNN/ dirs): mounted at <mnt>/gKK
#         with image-src=/, i.e. <mnt>/gKK/NNN/merged_*.h5.
# Then build the list once: list_merged_sp.py --dir <mnt> --out <list>
# (basename-keyed, so both layouts give the same (fileno, entry) order).
IMGDIR=${1:?image dir}; MNT=${2:?mount point inside the container}
for img in "$IMGDIR"/*_merged_sp_*.sqfs; do
  d=$(basename "$img" .sqfs); d=${d##*_merged_sp_}
  case "$d" in
    [0-9][0-9][0-9]) printf -- '--bind %s:%s/%s:image-src=/%s,ro ' "$img" "$MNT" "$d" "$d" ;;
    *)               printf -- '--bind %s:%s/%s:image-src=/,ro ' "$img" "$MNT" "$d" ;;
  esac
done
echo
